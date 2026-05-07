import copy
import numpy as np
import warnings
import math
from ..device.CIMA import *
from ..helper import get_pre_layer, get_next_layer
from irtool.core import load_ir, make_op

#=================================================================
#                 CIMA WEIGHT PLACEMENT ALGORITHM
#=================================================================

def mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                         mesh_height = 6, mesh_width = 6, alpha=0, pe_bind_direction=False,
                         method='random', mapped_node_history = None, transfer_thread_num = None):

    if name in layer_xb_mapped_node.keys():
        parent_node.append(layer_xb_mapped_node[name])
    else:
        # Collect parent nodes for non-CIM ops (e.g. split/add/concat).
        parent_node_split = []
        for name_ in layer_ref[name]:
            if 'Constant' in name_:
                continue
            # parent_node_split.append(layer_xb_mapped_node[name_])
            if name_ in layer_xb_mapped_node.keys():
                parent_node_split.append(layer_xb_mapped_node[name_])
            else:
                if 'split' in name_.lower() or 'concat' in name_.lower() or 'add' in name_.lower():
                    mapping_noncim_nodes(name_, layer_ref, available_nodes_xb, parent_node_split, layer_xb_mapped_node, mapping_info,
                                        method=method, mapped_node_history = mapped_node_history, transfer_thread_num=transfer_thread_num)
                else:
                    warnings.warn(
                        f"Layer '{name}' has an unmapped predecessor '{name_}'. "
                        "This predecessor is not a supported non-CIM op, so it cannot be placed automatically."
                    )

        parent_core_id = []
        # Merge parent I/O workloads, keeping the max per edge.
        record_io_workload_parent_total = {}
        for node in parent_node_split:
            if isinstance(node, MappedLayerNode):
                if node.record_io_workload_total != None:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                core_id = '.'.join(node.addr_node.split('.')[:2])

                if core_id not in parent_core_id:
                    parent_core_id.append(core_id)

        # Build candidate XB list excluding the parent cores.
        parent_available_nodes_xb = []
        for n in available_nodes_xb:
            x = '.'.join(n.split('.')[:2])
            if x not in parent_core_id:
                parent_available_nodes_xb.append(n)

        # at some output location (conceptually an identity relay). In this case we do
        # not constrain the placement core.
        if parent_available_nodes_xb == []:
            warnings.warn(
                f"Parents of layer '{name}' already occupy all deployable distinct core IDs; "
                "a relay transfer thread will be generated at the output side."
            )
            parent_available_nodes_xb = available_nodes_xb

        len_ = len(parent_available_nodes_xb)
        if method.lower() == 'random':
            rd_index = np.random.randint(0,len_)
            device_ref = parent_available_nodes_xb[rd_index]
            core_name = '.'.join(device_ref.split('.')[:2])
            if core_name not in transfer_thread_num.keys():
                transfer_thread_num[core_name] = 0
            transfer_thread_num[core_name] += 1

        elif method.lower() == 'onebyone':
            assert len_ >= 1
            device_ref = parent_available_nodes_xb[0]

        elif method.lower() == 'a_search':

            assert mapped_node_history != None
            device_ref = None
            possible_mapping_nodes = []
            for nd in parent_available_nodes_xb:
                node_location = '.'.join(nd.split('.')[:2])
                if node_location not in mapped_node_history:
                    possible_mapping_nodes.append(nd)

            if possible_mapping_nodes != []:
                index = 0
                device_ref = possible_mapping_nodes[index]
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

            if device_ref == None:

                loc_ = [mapped_node_history.index('.'.join(x.split('.')[:2])) for x in parent_available_nodes_xb]
                oldest_index_ = np.argmin(np.array(loc_))

                device_ref = parent_available_nodes_xb[oldest_index_]
                mapped_node_history.pop(loc_[oldest_index_])
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

            assert device_ref != None

        elif method.lower() == 'workload_balance':
            assert isinstance(transfer_thread_num, dict), f'{transfer_thread_num}'
            # thread_num = []
            #     core_name = '.'.join(nd.split('.')[:2])
            #         transfer_thread_num[core_name] = 0
            #     thread_num.append(transfer_thread_num[core_name])

            # #
            # min_id = np.argmin(np.array(thread_num))
            # device_ref =  parent_available_nodes_xb[min_id]

            rd_index = np.random.randint(0,len_)
            device_ref = parent_available_nodes_xb[rd_index]

            core_name = '.'.join(device_ref.split('.')[:2])
            if core_name not in transfer_thread_num.keys():
                transfer_thread_num[core_name] = 0
            transfer_thread_num[core_name] += 1

        else:
            raise ValueError(f'Message translated to English.')

        addr = [0]

        child_node = MappedLayerNode(name, device_ref, addr, parent = parent_node_split,
                                available_nodes_xb=parent_available_nodes_xb,
                                record_io_workload_total = copy.deepcopy(record_io_workload_parent_total),
                                MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
        distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
        distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
        child_node.record_io_workload_parent = record_io_workload_parent
        child_node.record_io_workload_out = record_io_workload_out
        child_node.to_parent_cost = distance_parent
        child_node.to_out_cost = distance_out
        child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]

        for n, v in child_node.record_io_workload_parent.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

        parent_node.append(child_node)

        layer_xb_mapped_node[name] = child_node
        mapping_info[name] = device_ref

def Workload_balance_search(layer_ref, placed_nodes, available_nodes_xb, node_info, mesh_height = 6, mesh_width = 6, alpha=0, pe_bind_direction=False):

    count = 0
    record_io_workload = {}

    layer_xb_mapped_node = {}

    mapping_info = {}

    mapped_pe_cluster_history = []
    mapped_node_history = []

    # computing workload recording
    record_node_compute_workload = {}
    for n in available_nodes_xb:
        pe_cluster = '.'.join(n.split('.')[:3])
        if pe_cluster not in record_node_compute_workload.keys():
            record_node_compute_workload[pe_cluster] = 0

    # operator compute times
    record_layer_compute_workload = {}
    #
    transfer_thread_num = {}
    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            assert layer_name in node_info.keys(), f'Message translated to English.'
            layer_info = node_info[layer_name]
            record_layer_compute_workload[layer_name] = layer_info['calc_num']

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                             mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction,
                                             method='workload_balance', mapped_node_history = mapped_node_history, transfer_thread_num=transfer_thread_num)

            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1


                # len_ = len(available_nodes_xb)
                # rd_index = np.random.randint(0,len_)
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]

                #     device_ref = f'cima-0.cima-node:7.cima-pe-cluster:0.cima-xb:{i}'
                #     available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                    available_nodes_xb=available_nodes_xb,
                                    record_io_workload_total = record_io_workload,
                                    MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

                #
                record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
            else:
                parent_core_id = []
                record_io_workload_parent_total = {}

                for node in parent_node:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])

                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:
                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)

                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)

                #
                if possible_mapped_nodes != []:
                    index = 0
                    device_ref = possible_mapped_nodes[index]
                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                elif possible_mapped_pe_cluster_nodes != []:
                    # index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))
                    oldest_access_node = mapped_node_history[0]
                    possible_mapped_pe_cluster_oldest_access = []
                    for nd in possible_mapped_pe_cluster_nodes:
                        if '.'.join(nd.split('.')[:2]) == oldest_access_node:
                            possible_mapped_pe_cluster_oldest_access.append(nd)
                    if possible_mapped_pe_cluster_oldest_access != []:
                        compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in possible_mapped_pe_cluster_oldest_access]
                        workload_minimum_index = np.argmin(np.array(compute_workload_node))
                        device_ref = possible_mapped_pe_cluster_oldest_access[workload_minimum_index]
                    else:
                        compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in possible_mapped_pe_cluster_nodes]
                        workload_minimum_index = np.argmin(np.array(compute_workload_node))
                        device_ref = possible_mapped_pe_cluster_nodes[workload_minimum_index]

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                    mapped_node_history.pop(0)
                    mapped_node_history.append(oldest_access_node)

                    #
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                if device_ref == None:
                    compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in parent_available_nodes_xb]
                    workload_minimum_index = np.argmin(np.array(compute_workload_node))
                    device_ref = parent_available_nodes_xb[workload_minimum_index]
                    #
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                # device_ref = parent_available_nodes_xb[rd_index]
                assert device_ref != None

                # remove mapped device
                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             record_io_workload_total = copy.deepcopy(record_io_workload_parent_total),
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]

                for n, v in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v

                # parent_node = child_node
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    node_mapping_info_list = []
    record_io_workload = []

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    node_mapping_info_list = mapping_info


    return node_mapping_info_list, record_io_workload


def A_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height = 6, mesh_width = 6, alpha=0, pe_bind_direction=False):

    count = 0
    record_io_workload = {}
    # parent_node = []

    layer_xb_mapped_node = {}

    mapping_info = {}

    mapped_pe_cluster_history = []
    mapped_node_history = []

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                             mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha, pe_bind_direction=pe_bind_direction,
                                             method='A_search', mapped_node_history = mapped_node_history)


            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1


                # len_ = len(available_nodes_xb)
                # rd_index = np.random.randint(0,len_)
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]

                #             device_ref_ = f'cima-0.cima-node:{j}.cima-pe-cluster:{k}.cima-xb:{i}'
                #             available_nodes_xb.remove(device_ref_)

                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                    available_nodes_xb=available_nodes_xb,
                                    record_io_workload_total = record_io_workload,
                                    MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

            else:
                parent_core_id = []
                record_io_workload_parent_total = {}

                for node in parent_node:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])

                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:
                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)

                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)

                #
                if possible_mapped_nodes != []:
                    index = 0
                    device_ref = possible_mapped_nodes[index]
                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                elif possible_mapped_pe_cluster_nodes != []:
                    # index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))

                    loc_ = [mapped_node_history.index('.'.join(x.split('.')[:2])) for x in possible_mapped_pe_cluster_nodes]
                    oldest_index_ = np.argmin(np.array(loc_))
                    oldest_access_node = mapped_node_history[loc_[oldest_index_]]

                    possible_mapped_pe_cluster_oldest_access = []
                    for nd in possible_mapped_pe_cluster_nodes:
                        if '.'.join(nd.split('.')[:2]) == oldest_access_node:
                            possible_mapped_pe_cluster_oldest_access.append(nd)
                    if possible_mapped_pe_cluster_oldest_access != []:
                        index = np.random.randint(0, len(possible_mapped_pe_cluster_oldest_access))
                        # index = 0
                        device_ref = possible_mapped_pe_cluster_oldest_access[index]
                        mapped_node_history.pop(0)
                        mapped_node_history.append(oldest_access_node)

                    else:
                        index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))
                        device_ref = possible_mapped_pe_cluster_nodes[index]

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                if device_ref == None:
                    loc_ = [mapped_pe_cluster_history.index('.'.join(x.split('.')[:3])) for x in parent_available_nodes_xb]
                    oldest_index_ = np.argmin(np.array(loc_))
                    device_ref = parent_available_nodes_xb[oldest_index_]
                    mapped_pe_cluster_history.pop(loc_[oldest_index_])
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                # device_ref = parent_available_nodes_xb[rd_index]
                assert device_ref != None

                #         pe_cluster_ = '.'.join(device_ref.split('.')[:3])
                #         device_ref_remove = f'{pe_cluster_}.cima-xb:{i_}'
                #         available_nodes_xb.remove(device_ref_remove)
                # elif layer_name == 'Conv_7':
                #         pe_cluster_ = '.'.join(device_ref.split('.')[:3])
                #         device_ref_remove = f'{pe_cluster_}.cima-xb:{i_}'
                #         available_nodes_xb.remove(device_ref_remove)
                # else:
                #     # remove mapped device
                #     available_nodes_xb.remove(device_ref)

                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             record_io_workload_total = copy.deepcopy(record_io_workload_parent_total),
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]

                for n, v in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v

                # parent_node = child_node
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    node_mapping_info_list = []
    record_io_workload = []

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    # mapping_info = {}
    # c = 0
    #         cn.get_all_parent_node_mapping_info(mapping_info)
    #     else:
    #         mapping_info[cn.layer_name]  = cn.addr_node + '.' + str(cn.addr_xb)
    #     c += 1

    node_mapping_info_list = mapping_info


    return node_mapping_info_list, record_io_workload


def random_search( layer_ref, placed_nodes, available_nodes_xb, mesh_height = 6, mesh_width = 6, alpha=0, pe_bind_direction=False):

    count = 0
    record_io_workload = {}
    # parent_node = []

    layer_xb_mapped_node = {}

    mapping_info = {}

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:

                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                          mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha,
                                          pe_bind_direction=pe_bind_direction)

            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1

                len_ = len(available_nodes_xb)

                rd_index = np.random.randint(0,len_)
                device_ref = available_nodes_xb[rd_index]
                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                    available_nodes_xb=available_nodes_xb,
                                    record_io_workload_total = record_io_workload,
                                    MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            else:

                parent_core_id = []
                record_io_workload_parent_total = {}

                for node in parent_node:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    # else:
                    #     core_id = layer_xb_mapped_node[node]

                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                len_ = len(parent_available_nodes_xb)
                rd_index = np.random.randint(0,len_)
                device_ref = parent_available_nodes_xb[rd_index]
                # remove mapped device
                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             record_io_workload_total = copy.deepcopy(record_io_workload_parent_total),
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]

                for n, v in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v

                # parent_node = child_node
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    node_mapping_info_list = []
    record_io_workload = []

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    # mapping_info = {}
    # c = 0
    #         cn.get_all_parent_node_mapping_info(mapping_info)
    #     else:
    #         mapping_info[cn.layer_name]  = cn.addr_node + '.' + str(cn.addr_xb)
    #     c += 1

    node_mapping_info_list = mapping_info

    return node_mapping_info_list, record_io_workload

def onebyone_search( layer_ref, placed_nodes, available_nodes_xb, mesh_height = 6, mesh_width = 6, alpha=0, pe_bind_direction=False):

    count = 0
    record_io_workload = {}
    # parent_node = []

    layer_xb_mapped_node = {}

    mapping_info = {}

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:

                        mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                          mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha,
                                          pe_bind_direction=pe_bind_direction, method='onebyone')

            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1

                len_ = len(available_nodes_xb)

                rd_index = np.random.randint(0,len_)
                device_ref = available_nodes_xb[rd_index]
                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                    available_nodes_xb=available_nodes_xb,
                                    record_io_workload_total = record_io_workload,
                                    MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)
            else:

                parent_core_id = []
                record_io_workload_parent_total = {}

                for node in parent_node:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                    core_id = '.'.join(node.addr_node.split('.')[:2])
                    # else:
                    #     core_id = layer_xb_mapped_node[node]

                    if core_id not in parent_core_id:
                        parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)
                len_ = len(parent_available_nodes_xb)
                device_ref = parent_available_nodes_xb[0]

                # remove mapped device
                available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             record_io_workload_total = copy.deepcopy(record_io_workload_parent_total),
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
                distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
                child_node.record_io_workload_parent = record_io_workload_parent
                child_node.record_io_workload_out = record_io_workload_out
                child_node.to_parent_cost = distance_parent
                child_node.to_out_cost = distance_out
                child_node.available_nodes_xb = [x for x in parent_available_nodes_xb if x != device_ref]

                for n, v in child_node.record_io_workload_parent.items():
                    if n in child_node.record_io_workload_total.keys():
                        child_node.record_io_workload_total[n] += v
                    else:
                        child_node.record_io_workload_total[n] = v

                # parent_node = child_node
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    node_mapping_info_list = []
    record_io_workload = []

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    # mapping_info = {}
    # c = 0
    #         cn.get_all_parent_node_mapping_info(mapping_info)
    #     else:
    #         mapping_info[cn.layer_name]  = cn.addr_node + '.' + str(cn.addr_xb)
    #     c += 1

    node_mapping_info_list = mapping_info

    return node_mapping_info_list, record_io_workload

def compare_func(items):
    return list(items[1][0].values())[0][3]

def packaged_random_search( layer_ref, placed_nodes, available_nodes_xb, mesh_height = 6, mesh_width = 6, alpha=0,
                            pe_bind_direction=False, dmac_layer = None):


    # placed_nodes = dict(sorted(placed_nodes.items(), key=compare_func, reverse=True))

    count = 0
    record_io_workload = {}
    # parent_node = []

    layer_xb_mapped_node = {}
    mapping_info = {}

    # layer node mapped count
    layer_node_count = {}

    rram_available_nodes_xb = []
    dmac_available_nodes = []
    for n in available_nodes_xb:
        if 'dmac' in n:
            dmac_available_nodes.append(n)
        else:
            if mesh_height == 4 and mesh_width == 9:
                            # rram_available_nodes_xb.append(n)
                rram_available_nodes_xb.append(n)
            elif mesh_height == 6 and mesh_width == 6:
                rram_available_nodes_xb.append(n)
            else:
                raise ValueError(f'Message translated to English.')

    available_nodes_xb = rram_available_nodes_xb

    all_availabel_nodes_xb = copy.deepcopy(available_nodes_xb)

    max_pe_thread_num = 2

    transfer_thread_num = {}

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        if 'split' in name.lower() or 'concat' in name.lower() or 'add' in name.lower():
                            mapping_noncim_nodes(name, layer_ref, all_availabel_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                            mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha,
                                            pe_bind_direction=pe_bind_direction, method='random', transfer_thread_num=transfer_thread_num)
                        else:
                            warnings.warn(f'Message translated to English.')


            if dmac_layer != None:
                IsUseDMAC = False
                for dl in dmac_layer:
                    if dl in layer_name:
                        IsUseDMAC = True
                        break

                if IsUseDMAC:
                    # Use DMAC
                    parent_core_id = []
                    for node in parent_node:
                        if isinstance(node, MappedLayerNode):
                            core_id = '.'.join(node.addr_node.split('.')[:2])
                            if core_id not in parent_core_id:
                                parent_core_id.append(core_id)
                    parent_available_nodes_xb = []
                    for n in dmac_available_nodes:
                        x = '.'.join(n.split('.')[:2])
                        if x not in parent_core_id:
                            parent_available_nodes_xb.append(n)
                    rd_index = np.random.randint(0,len(parent_available_nodes_xb))
                    device_ref = parent_available_nodes_xb[rd_index]
                    dmac_available_nodes.remove(device_ref)
                    mapping_info[split_layer_name] = device_ref

                    child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                            available_nodes_xb=dmac_available_nodes,
                                            MESH_HEIGHT=mesh_height,
                                            MESH_WIDTH=mesh_width)
                    layer_xb_mapped_node[layer_name] = child_node
                    continue

            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1

                len_ = len(available_nodes_xb)
                device_num = int(math.ceil(addr[3] / 128))

                if device_num > 1:


                    # rd_index = np.random.randint(0, len_)
                    # device_ref = available_nodes_xb[rd_index]
                    # dr_list = device_ref.split(':')
                    # device_prefix = (':').join(dr_list[:-1])
                    # current_device_xb_index = int(dr_list[-1])

                    ct_ = 0
                    while True:
                        rd_index = np.random.randint(0, len_)
                        device_ref = available_nodes_xb[rd_index]
                        dr_list = device_ref.split(':')
                        device_prefix = (':').join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])
                        if current_device_xb_index % 2 == 0:
                            break
                        ct_ += 1
                        if ct_ > 100:
                            raise ValueError(f'Message translated to English.')

                    ct = 0
                    while True:
                        CanMapNode = True
                        for dn in range(device_num):
                            if  device_prefix + f':{current_device_xb_index + dn}' not in available_nodes_xb:
                                CanMapNode = False
                                break
                        if CanMapNode:
                            break

                        ct_ = 0
                        while True:
                            rd_index = np.random.randint(0, len_)
                            device_ref = available_nodes_xb[rd_index]
                            dr_list = device_ref.split(':')
                            device_prefix = (':').join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])
                            if current_device_xb_index % 2 == 0:
                                break
                            ct_ += 1
                            if ct_ > 100:
                                raise ValueError(f'Message translated to English.')

                        ct += 1
                        if ct > 100:
                            raise ValueError(f'Message translated to English.')

                    for dn in range(device_num):
                        available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')

                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'

                    if device_prefix not in layer_node_count.keys():
                        layer_node_count[device_prefix] = 1

                else:

                    rd_index = np.random.randint(0,len_)
                    device_ref = available_nodes_xb[rd_index]
                    available_nodes_xb.remove(device_ref)


                    dr_list = device_ref.split(':')
                    device_prefix = (':').join(dr_list[:-1])

                    if device_prefix not in layer_node_count.keys():
                        layer_node_count[device_prefix] = 1

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                        available_nodes_xb=available_nodes_xb,
                                        record_io_workload_total = record_io_workload,
                                        MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

                if layer_node_count[device_prefix] == max_pe_thread_num:
                    available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                    for n in available_nodes_xb_:
                        if device_prefix in n:
                            available_nodes_xb.remove(n)
            else:

                parent_core_id = []
                # record_io_workload_parent_total = {}

                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                        #             record_io_workload_parent_total[k] = v
                        #         else:
                        #             record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                        core_id = '.'.join(node.addr_node.split('.')[:2])

                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                len_ = len(parent_available_nodes_xb)
                device_num = int(math.ceil(addr[3] / 128))

                if device_num > 1:

                    # rd_index = np.random.randint(0, len_)
                    # device_ref = parent_available_nodes_xb[rd_index]
                    # dr_list = device_ref.split(':')
                    # device_prefix = (':').join(dr_list[:-1])
                    # current_device_xb_index = int(dr_list[-1])

                    ct_ = 0
                    while True:
                        rd_index = np.random.randint(0, len_)
                        device_ref = parent_available_nodes_xb[rd_index]
                        dr_list = device_ref.split(':')
                        device_prefix = (':').join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])
                        if current_device_xb_index % 2 == 0:
                            break
                        ct_ += 1
                        if ct_ > 100:
                            raise ValueError(f'Message translated to English.')

                    ct = 0
                    while True:
                        CanMapNode = True
                        for dn in range(device_num):
                            if  device_prefix + f':{current_device_xb_index + dn}' not in parent_available_nodes_xb:
                                CanMapNode = False
                                break
                        if CanMapNode:
                            break

                        # rd_index = np.random.randint(0, len_)
                        # device_ref = parent_available_nodes_xb[rd_index]
                        # dr_list = device_ref.split(':')
                        # device_prefix = (':').join(dr_list[:-1])
                        # current_device_xb_index = int(dr_list[-1])

                        ct_ = 0
                        while True:
                            rd_index = np.random.randint(0, len_)
                            device_ref = parent_available_nodes_xb[rd_index]
                            dr_list = device_ref.split(':')
                            device_prefix = (':').join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])
                            if current_device_xb_index % 2 == 0:
                                break
                            ct_ += 1
                            if ct_ > 100:
                                raise ValueError(f'Message translated to English.')

                        ct += 1
                        if ct > 100:
                            raise ValueError(f'Message translated to English.')



                    for dn in range(device_num):
                        available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')

                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'

                else:

                    if len_ > 1:
                        rd_index = np.random.randint(0, len_)
                    else:
                        rd_index = 0
                    device_ref = parent_available_nodes_xb[rd_index]
                    available_nodes_xb.remove(device_ref)
                    #
                    dr_list = device_ref.split(':')
                    device_prefix = (':').join(dr_list[:-1])
                if device_prefix not in layer_node_count.keys():
                    layer_node_count[device_prefix] = 1
                else:
                    if layer_node_count[device_prefix] < max_pe_thread_num:
                        layer_node_count[device_prefix] += 1

                if layer_node_count[device_prefix] == max_pe_thread_num:
                    available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                    for n in available_nodes_xb_:
                        if device_prefix in n:
                            available_nodes_xb.remove(n)

                if layer_node_count[device_prefix] > max_pe_thread_num:
                    raise ValueError(f'Message translated to English.')

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

    record_io_workload = []

    for layer_name in layer_ref.keys():

        if layer_name in layer_xb_mapped_node.keys():
            parent_layer_name = layer_ref[layer_name]
            parent_node = []
            for name in parent_layer_name:
                if 'graph_input' in name:
                    parent_node.append('IN')
                    continue
                assert name in layer_xb_mapped_node.keys(), f'{name} not in {layer_xb_mapped_node.keys()}'
                parent_node.append(layer_xb_mapped_node[name])

            record_io_workload_parent_total = {}
            for node in parent_node:
                if 'IN' == node:
                    continue
                if node.record_io_workload_total != None:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)

            child_node = layer_xb_mapped_node[layer_name]
            child_node.parent = parent_node
            child_node.record_io_workload_total = copy.deepcopy(record_io_workload_parent_total)

            if 'IN' in parent_node:
                continue

            distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
            distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
            child_node.record_io_workload_parent = record_io_workload_parent
            child_node.record_io_workload_out = record_io_workload_out
            child_node.to_parent_cost = distance_parent
            child_node.to_out_cost = distance_out

            for n, v in child_node.record_io_workload_parent.items():
                if n in child_node.record_io_workload_total.keys():
                    child_node.record_io_workload_total[n] += v
                else:
                    child_node.record_io_workload_total[n] = v

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    return mapping_info, record_io_workload, transfer_thread_num


def packaged_Workload_balance_search(layer_ref, placed_nodes, available_nodes_xb, node_info, mesh_height = 6, mesh_width = 6, alpha=0,
                                     pe_bind_direction=False, dmac_layer = None):

    count = 0
    record_io_workload = {}

    layer_xb_mapped_node = {}

    mapping_info = {}

    mapped_pe_cluster_history = []
    mapped_node_history = []

    # operator compute times
    record_layer_compute_workload = {}

    # layer node mapped count
    layer_node_count = {}

    rram_available_nodes_xb = []
    dmac_available_nodes = []

    for n in available_nodes_xb:
        if 'dmac' in n:
            dmac_available_nodes.append(n)
        else:
            if mesh_height == 4 and mesh_width == 9:
                rram_available_nodes_xb.append(n)
            elif mesh_height == 6 and mesh_width == 6:
                rram_available_nodes_xb.append(n)
            else:
                raise ValueError(f'Message translated to English.')

    available_nodes_xb = rram_available_nodes_xb


    all_available_nodes_xb = copy.deepcopy(available_nodes_xb)

    # computing workload recording
    record_node_compute_workload = {}
    for n in available_nodes_xb:
        core_loc = '.'.join(n.split('.')[:3])
        if core_loc not in record_node_compute_workload.keys():
            record_node_compute_workload[core_loc] = 0

    max_pe_thread_num = 2

    transfer_thread_num = {}

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            device_num = int(math.ceil(addr[3] / 128))

            assert layer_name in node_info.keys(), f'Message translated to English.'
            layer_info = node_info[layer_name]
            record_layer_compute_workload[layer_name] = layer_info['calc_num']

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        if 'split' in name.lower() or 'concat' in name.lower() or 'add' in name.lower():
                            mapping_noncim_nodes(name, layer_ref, all_available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                            mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha,
                                            pe_bind_direction=pe_bind_direction, method='workload_balance', transfer_thread_num=transfer_thread_num)
                        else:
                            warnings.warn(f'Message translated to English.')
            if dmac_layer != None:

                IsUseDMAC = False
                for dl in dmac_layer:
                    if dl in layer_name:
                        IsUseDMAC = True
                        break

                if IsUseDMAC:
                    # Use DMAC
                    parent_core_id = []
                    for node in parent_node:
                        if isinstance(node, MappedLayerNode):
                            core_id = '.'.join(node.addr_node.split('.')[:2])
                            if core_id not in parent_core_id:
                                parent_core_id.append(core_id)
                    parent_available_nodes_xb = []
                    for n in dmac_available_nodes:
                        x = '.'.join(n.split('.')[:2])
                        if x not in parent_core_id:
                            parent_available_nodes_xb.append(n)

                    rd_index = np.random.randint(0,len(parent_available_nodes_xb))
                    device_ref = parent_available_nodes_xb[rd_index]
                    dmac_available_nodes.remove(device_ref)
                    mapping_info[split_layer_name] = device_ref

                    child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                            available_nodes_xb=dmac_available_nodes,
                                            MESH_HEIGHT=mesh_height,
                                            MESH_WIDTH=mesh_width)
                    layer_xb_mapped_node[layer_name] = child_node
                    continue


            if count == 0:


                # len_ = len(available_nodes_xb)
                # rd_index = np.random.randint(0,len_)
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]

                dr_list = device_ref.split(':')
                device_prefix = (':').join(dr_list[:-1])
                current_device_xb_index = int(dr_list[-1])


                for dn in range(device_num):
                    available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')

                if device_num > 1:
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                else:
                    device_ref = device_prefix + f':{current_device_xb_index}'

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                            available_nodes_xb=available_nodes_xb,
                                            record_io_workload_total = record_io_workload,
                                            MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

                #
                record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                layer_node_count[device_prefix] = 1

                if layer_node_count[device_prefix] == max_pe_thread_num:
                    available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                    for n in available_nodes_xb_:
                        if device_prefix in n:
                            available_nodes_xb.remove(n)

            else:
                parent_core_id = []
                # record_io_workload_parent_total = {}

                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                    # # if isinstance(node, MappedLayerNode):
                    #         record_io_workload_parent_total[k] = v
                    #     else:
                    #         record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                        core_id = '.'.join(node.addr_node.split('.')[:2])

                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:

                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)

                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)

                #
                if possible_mapped_nodes != []:
                    index = 0
                    device_ref = possible_mapped_nodes[index]
                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                elif possible_mapped_pe_cluster_nodes != []:
                    #
                    compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in possible_mapped_pe_cluster_nodes]
                    workload_minimum_index = np.argmin(np.array(compute_workload_node))
                    device_ref = possible_mapped_pe_cluster_nodes[workload_minimum_index]

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))
                    # mapped_node_history.pop(0)
                    # mapped_node_history.append(oldest_access_node)

                    #
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]
                else:

                # # if device_ref == None:
                    compute_workload_node = [record_node_compute_workload['.'.join(x.split('.')[:3])] for x in parent_available_nodes_xb]
                    workload_minimum_index = np.argmin(np.array(compute_workload_node))
                    device_ref = parent_available_nodes_xb[workload_minimum_index]

                    #
                    record_node_compute_workload['.'.join(device_ref.split('.')[:3])] += record_layer_compute_workload[layer_name]

                # device_ref = parent_available_nodes_xb[rd_index]
                assert device_ref != None

                dr_list = device_ref.split(':')
                device_prefix = (':').join(dr_list[:-1])
                current_device_xb_index = int(dr_list[-1])

                for dn in range(device_num):
                    available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')

                if device_num > 1:
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                else:
                    device_ref = device_prefix + f':{current_device_xb_index}'

                # remove mapped device
                # available_nodes_xb.remove(device_ref)

                if device_prefix not in layer_node_count.keys():
                    layer_node_count[device_prefix] = 1
                else:
                    if layer_node_count[device_prefix] < max_pe_thread_num:
                        layer_node_count[device_prefix] += 1

                if layer_node_count[device_prefix] == max_pe_thread_num:
                    available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                    for n in available_nodes_xb_:
                        if device_prefix in n:
                            available_nodes_xb.remove(n)

                if layer_node_count[device_prefix] > max_pe_thread_num:
                    raise ValueError(f'Message translated to English.')

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    record_io_workload = []

    for layer_name in layer_ref.keys():

        if layer_name in layer_xb_mapped_node.keys():
            parent_layer_name = layer_ref[layer_name]
            parent_node = []
            for name in parent_layer_name:
                if 'graph_input' in name:
                    parent_node.append('IN')
                    continue
                assert name in layer_xb_mapped_node.keys(), f'{name} not in {layer_xb_mapped_node.keys()}'
                parent_node.append(layer_xb_mapped_node[name])

            record_io_workload_parent_total = {}
            for node in parent_node:
                if 'IN' == node:
                    continue
                if node.record_io_workload_total != None:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)

            child_node = layer_xb_mapped_node[layer_name]
            child_node.parent = parent_node
            child_node.record_io_workload_total = copy.deepcopy(record_io_workload_parent_total)

            if 'IN' in parent_node:
                continue

            distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
            distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
            child_node.record_io_workload_parent = record_io_workload_parent
            child_node.record_io_workload_out = record_io_workload_out
            child_node.to_parent_cost = distance_parent
            child_node.to_out_cost = distance_out

            for n, v in child_node.record_io_workload_parent.items():
                if n in child_node.record_io_workload_total.keys():
                    child_node.record_io_workload_total[n] += v
                else:
                    child_node.record_io_workload_total[n] = v

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    return mapping_info, record_io_workload, transfer_thread_num


def packaged_LRU_search(layer_ref, placed_nodes, available_nodes_xb, mesh_height = 6, mesh_width = 6, alpha=0,
                      pe_bind_direction=False, dmac_layer = None):
    '''
    Least Recently Used (LRU)
    '''
    count = 0
    record_io_workload = {}
    # parent_node = []

    layer_xb_mapped_node = {}

    mapping_info = {}

    mapped_pe_cluster_history = []
    mapped_node_history = []

    # layer node mapped count
    layer_node_count = {}

    rram_available_nodes_xb = []
    dmac_available_nodes = []
    for n in available_nodes_xb:
        if 'dmac' in n:
            dmac_available_nodes.append(n)
        else:
            if mesh_height == 4 and mesh_width == 9:
                rram_available_nodes_xb.append(n)
            elif mesh_height == 6 and mesh_width == 6:
                rram_available_nodes_xb.append(n)
            else:
                raise ValueError(f'Message translated to English.')

    available_nodes_xb = rram_available_nodes_xb

    max_pe_thread_num = 2

    for node_name, split_info in placed_nodes.items():

        for split_node in split_info:
            split_layer_name = list(split_node.keys())[0]
            addr = list(split_node.values())[0]
            layer_name = split_layer_name.split('.')[0]
            parent_node = []

            if count != 0:
                parent_layer_name = layer_ref[layer_name]

                for name in parent_layer_name:
                    if 'graph_input' in name:
                        parent_node.append('IN')
                        continue
                    # parent_node.append(layer_xb_mapped_node[name])
                    if name in layer_xb_mapped_node.keys():
                        parent_node.append(layer_xb_mapped_node[name])
                    else:
                        if 'split' in name.lower() or 'concat' in name.lower() or 'add' in name.lower():
                            mapping_noncim_nodes(name, layer_ref, available_nodes_xb, parent_node, layer_xb_mapped_node, mapping_info,
                                            mesh_height = mesh_height, mesh_width = mesh_width, alpha=alpha,
                                            pe_bind_direction=pe_bind_direction)
                        else:
                            warnings.warn(f'Message translated to English.')

            IsUseDMAC = False
            for dl in dmac_layer:
                if dl in layer_name:
                    IsUseDMAC = True
                    break

            if IsUseDMAC:
                # Use DMAC
                parent_core_id = []
                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                        core_id = '.'.join(node.addr_node.split('.')[:2])
                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)
                parent_available_nodes_xb = []
                for n in dmac_available_nodes:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                rd_index = np.random.randint(0,len(parent_available_nodes_xb))
                device_ref = parent_available_nodes_xb[rd_index]
                dmac_available_nodes.remove(device_ref)
                mapping_info[split_layer_name] = device_ref

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                        available_nodes_xb=dmac_available_nodes,
                                        MESH_HEIGHT=mesh_height,
                                        MESH_WIDTH=mesh_width)
                layer_xb_mapped_node[layer_name] = child_node
                continue

            device_num = int(math.ceil(addr[3] / 128))

            if count == 0 or 'IN' in parent_node:

                if 'IN' in parent_node:
                    assert len(parent_node) == 1


                # len_ = len(available_nodes_xb)
                # rd_index = np.random.randint(0,len_)
                rd_index = 0
                device_ref = available_nodes_xb[rd_index]

                #
                dr_list = device_ref.split(':')
                device_prefix = (':').join(dr_list[:-1])
                current_device_xb_index = int(dr_list[-1])

                for dn in range(device_num):
                    available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')

                if device_num > 1:
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                else:
                    device_ref = device_prefix + f':{current_device_xb_index}'

                #             device_ref_ = f'cima-0.cima-node:{j}.cima-pe-cluster:{k}.cima-xb:{i}'
                #             available_nodes_xb.remove(device_ref_)

                # available_nodes_xb.remove(device_ref)

                child_node = MappedLayerNode(split_layer_name, device_ref, addr,
                                    available_nodes_xb=available_nodes_xb,
                                    record_io_workload_total = record_io_workload,
                                    MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)
                # self.node_mapping_info[split_node_name] = device_ref + '.' + str(addr)
                count += 1

                # parent_node = node
                # continue

                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

                mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                layer_node_count[device_prefix] = 1
            else:
                parent_core_id = []
                # record_io_workload_parent_total = {}

                for node in parent_node:
                    if isinstance(node, MappedLayerNode):
                    #         record_io_workload_parent_total[k] = v
                    #     else:
                    #         record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)
                        core_id = '.'.join(node.addr_node.split('.')[:2])

                        if core_id not in parent_core_id:
                            parent_core_id.append(core_id)

                parent_available_nodes_xb = []
                for n in available_nodes_xb:
                    x = '.'.join(n.split('.')[:2])
                    if x not in parent_core_id:
                        parent_available_nodes_xb.append(n)

                device_ref = None
                possible_mapped_pe_cluster_nodes = []
                possible_mapped_nodes = []
                for nd in parent_available_nodes_xb:
                    pe_cluster_location = '.'.join(nd.split('.')[:3])
                    node_location = '.'.join(nd.split('.')[:2])
                    if node_location not in mapped_node_history:
                        possible_mapped_nodes.append(nd)

                    if pe_cluster_location not in mapped_pe_cluster_history:
                        possible_mapped_pe_cluster_nodes.append(nd)

                #
                if possible_mapped_nodes != []:
                    index = 0
                    while True:
                        device_ref = possible_mapped_nodes[index]
                        dr_list = device_ref.split(':')
                        device_prefix = (':').join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])
                        if current_device_xb_index + device_num <= 16:
                            CanMap = True
                            for i_ in range(device_num):
                                if device_prefix + f':{current_device_xb_index + i_}' not in available_nodes_xb:
                                    CanMap = False
                                    break
                            if CanMap:
                                break

                        index += 1
                        if index > 100:
                            raise ValueError(f'Message translated to English.')

                    mapped_node_history.append('.'.join(device_ref.split('.')[:2]))
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                elif possible_mapped_pe_cluster_nodes != []:
                    # index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))

                    loc_ = [mapped_node_history.index('.'.join(x.split('.')[:2])) for x in possible_mapped_pe_cluster_nodes]
                    oldest_index_ = np.argmin(np.array(loc_))
                    oldest_access_node = mapped_node_history[loc_[oldest_index_]]

                    possible_mapped_pe_cluster_oldest_access = []
                    for nd in possible_mapped_pe_cluster_nodes:
                        if '.'.join(nd.split('.')[:2]) == oldest_access_node:
                            possible_mapped_pe_cluster_oldest_access.append(nd)
                    if possible_mapped_pe_cluster_oldest_access != []:

                        index = 0
                        while True:
                            # index = np.random.randint(0, len(possible_mapped_pe_cluster_oldest_access))
                            # index = 0
                            device_ref = possible_mapped_pe_cluster_oldest_access[index]

                            dr_list = device_ref.split(':')
                            device_prefix = (':').join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])

                            if current_device_xb_index + device_num <= 16:
                                CanMap = True
                                for i_ in range(device_num):
                                    if device_prefix + f':{current_device_xb_index + i_}' not in available_nodes_xb:
                                        CanMap = False
                                        break
                                if CanMap:
                                    break

                            index += 1
                            if index > 100:
                                raise ValueError(f'Message translated to English.')

                        mapped_node_history.pop(0)
                        mapped_node_history.append(oldest_access_node)

                    else:
                        index = 0
                        while True:
                            # index = np.random.randint(0, len(possible_mapped_pe_cluster_nodes))
                            device_ref = possible_mapped_pe_cluster_nodes[index]
                            # get name prefix
                            dr_list = device_ref.split(':')
                            device_prefix = (':').join(dr_list[:-1])
                            current_device_xb_index = int(dr_list[-1])

                            if current_device_xb_index + device_num <= 16:
                                CanMap = True
                                for i_ in range(device_num):
                                    if device_prefix + f':{current_device_xb_index + i_}' not in available_nodes_xb:
                                        CanMap = False
                                        break
                                if CanMap:
                                    break

                            index += 1
                            if index > 100:
                                raise ValueError(f'Message translated to English.')

                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                if device_ref == None:
                    loc_ = [mapped_pe_cluster_history.index('.'.join(x.split('.')[:3])) for x in parent_available_nodes_xb]
                    loc_.sort(reverse=False)

                    index = 0
                    # oldest_index_ = np.argmin(np.array(loc_))
                    while True:
                        oldest_index_ = loc_[index]
                        device_ref = parent_available_nodes_xb[oldest_index_]
                        # get name prefix
                        dr_list = device_ref.split(':')
                        device_prefix = (':').join(dr_list[:-1])
                        current_device_xb_index = int(dr_list[-1])

                        if current_device_xb_index + device_num <= 16:
                            CanMap = True
                            for i_ in range(device_num):
                                if device_prefix + f':{current_device_xb_index + i_}' not in available_nodes_xb:
                                    CanMap = False
                                    break
                            if CanMap:
                                break

                        index += 1
                        if index > len(loc_):
                            raise ValueError(f'Message translated to English.')

                    mapped_pe_cluster_history.pop(loc_[oldest_index_])
                    mapped_pe_cluster_history.append('.'.join(device_ref.split('.')[:3]))

                # device_ref = parent_available_nodes_xb[rd_index]
                assert device_ref != None

                #         pe_cluster_ = '.'.join(device_ref.split('.')[:3])
                #         device_ref_remove = f'{pe_cluster_}.cima-xb:{i_}'
                #         available_nodes_xb.remove(device_ref_remove)
                # elif layer_name == 'Conv_7':
                #         pe_cluster_ = '.'.join(device_ref.split('.')[:3])
                #         device_ref_remove = f'{pe_cluster_}.cima-xb:{i_}'
                #         available_nodes_xb.remove(device_ref_remove)
                # else:
                #     # remove mapped device
                #     available_nodes_xb.remove(device_ref)

                dr_list = device_ref.split(':')
                device_prefix = (':').join(dr_list[:-1])
                current_device_xb_index = int(dr_list[-1])

                for dn in range(device_num):
                    try:
                        available_nodes_xb.remove(device_prefix + f':{current_device_xb_index + dn}')
                    except:
                        print(device_prefix + f':{current_device_xb_index + dn}' in available_nodes_xb)
                        print(f'Message translated to English.')
                        print(layer_name)
                        print(current_device_xb_index)
                        print(device_prefix)
                        print(device_num)
                        exit(1)

                if device_num > 1:
                    device_ref = device_prefix + f':{current_device_xb_index}-{current_device_xb_index + dn}'
                else:
                    device_ref = device_prefix + f':{current_device_xb_index}'

                # available_nodes_xb.remove(device_ref)

                if device_prefix not in layer_node_count.keys():
                    layer_node_count[device_prefix] = 1
                else:
                    if layer_node_count[device_prefix] < max_pe_thread_num:
                        layer_node_count[device_prefix] += 1

                if layer_node_count[device_prefix] == max_pe_thread_num:
                    available_nodes_xb_ = copy.deepcopy(available_nodes_xb)
                    for n in available_nodes_xb_:
                        if device_prefix in n:
                            available_nodes_xb.remove(n)

                if layer_node_count[device_prefix] > max_pe_thread_num:
                    raise ValueError(f'Message translated to English.')

                child_node = MappedLayerNode(split_layer_name, device_ref, addr, parent = parent_node,
                                             available_nodes_xb=parent_available_nodes_xb,
                                             MESH_HEIGHT=mesh_height, MESH_WIDTH=mesh_width)

                # parent_node = child_node
                mapping_info[split_layer_name] = device_ref + '.' + str(addr)

            if layer_name not in layer_xb_mapped_node.keys():
                layer_xb_mapped_node[layer_name] = child_node

            #     output_child_nodes.append(child_node)

    record_io_workload = []

    for layer_name in layer_ref.keys():

        if layer_name in layer_xb_mapped_node.keys():
            parent_layer_name = layer_ref[layer_name]
            parent_node = []
            for name in parent_layer_name:
                if 'graph_input' in name:
                    parent_node.append('IN')
                    continue
                assert name in layer_xb_mapped_node.keys(), f'{name} not in {layer_xb_mapped_node.keys()}'
                parent_node.append(layer_xb_mapped_node[name])

            record_io_workload_parent_total = {}
            for node in parent_node:
                if 'IN' == node:
                    continue
                if node.record_io_workload_total != None:
                    for k,v in node.record_io_workload_total.items():
                        if k not in record_io_workload_parent_total.keys():
                            record_io_workload_parent_total[k] = v
                        else:
                            record_io_workload_parent_total[k] = max(record_io_workload_parent_total[k], v)

            child_node = layer_xb_mapped_node[layer_name]
            child_node.parent = parent_node
            child_node.record_io_workload_total = copy.deepcopy(record_io_workload_parent_total)

            if 'IN' in parent_node:
                continue

            distance_parent, record_io_workload_parent = child_node.get_to_parent_distance(pe_bind_direction, alpha=alpha)
            distance_out, record_io_workload_out = child_node.get_to_out_distance(pe_bind_direction, alpha=alpha)
            child_node.record_io_workload_parent = record_io_workload_parent
            child_node.record_io_workload_out = record_io_workload_out
            child_node.to_parent_cost = distance_parent
            child_node.to_out_cost = distance_out

            for n, v in child_node.record_io_workload_parent.items():
                if n in child_node.record_io_workload_total.keys():
                    child_node.record_io_workload_total[n] += v
                else:
                    child_node.record_io_workload_total[n] = v

    if child_node.record_io_workload_out != None:
        for n, v in child_node.record_io_workload_out.items():
            if n in child_node.record_io_workload_total.keys():
                child_node.record_io_workload_total[n] += v
            else:
                child_node.record_io_workload_total[n] = v

    record_io_workload = child_node.record_io_workload_total

    return mapping_info, record_io_workload


class MappedLayerNode:

    def __init__(self, layer_name, addr_node, addr_xb, parent=None,
                 to_parent_cost=0, to_out_cost=0,
                 record_io_workload_parent =None,
                 record_io_workload_out = None,
                 available_nodes_xb =  None,
                 record_io_workload_total = None,
                 MESH_HEIGHT = 4,
                 MESH_WIDTH = 4
                 ):
        self.layer_name = layer_name
        self.addr_node = addr_node
        self.addr_xb = addr_xb
        self.parent = parent
        self.to_parent_cost = to_parent_cost
        self.to_out_cost = to_out_cost
        self.record_io_workload_parent = record_io_workload_parent # virtual channel
        self.record_io_workload_out = record_io_workload_out # virtual channel
        self.available_nodes_xb = available_nodes_xb
        self.record_io_workload_total = record_io_workload_total # virtual channel

        self.mesh_height = MESH_HEIGHT # mesh node config
        self.mesh_width = MESH_WIDTH # mesh node config

    def get_all_parent_layer_list(self, parent_layer_list):
        parent = self.parent
        layer_name = self.layer_name.split('.')[0]

        if layer_name not in parent_layer_list:
            parent_layer_list.append(layer_name)

        if parent != None:
            for parent_node in parent:
                layer_name = parent_node.layer_name.split('.')[0]
                if layer_name not in parent_layer_list:
                    parent_layer_list.append(layer_name)
                parent_node.get_all_parent_layer_list(parent_layer_list)

        # else:


    def get_all_parent_addr(self, node_addr_list):
        '''
        '''
        parent = self.parent
        addr_node = self.addr_node
        if addr_node not in node_addr_list:
            node_addr_list.append(addr_node)
        if parent != None:
            for parent_node in parent:
                if addr_node not in node_addr_list:
                    node_addr_list.append(parent_node.addr_node)
                parent_node.get_all_parent_addr(node_addr_list)


    def get_specify_parent_node(self, parent_name):
        '''
        '''
        parent = self.parent
        specify_parent_node = None
        if parent != None:
            for parent_node in parent:
                if parent_node.layer_name.split('.')[0] == parent_name:
                    specify_parent_node = parent_node
                    return specify_parent_node
                else:
                    specify_parent_node = parent_node.get_specify_parent_node(parent_name)
                    return specify_parent_node
        elif self.layer_name.split('.')[0] == parent_name:
            return self

        return None

    def get_all_parent_node_mapping_info(self, node_mapping_info):
        '''
        '''
        parent = self.parent
        layer_name = self.layer_name
        addr_node = self.addr_node
        addr_xb = self.addr_xb
        if layer_name not in node_mapping_info.keys():
            node_mapping_info[layer_name] = addr_node + '.' + str(addr_xb)
        if parent != None:
            for parent_node in parent:
                parent_node.get_all_parent_node_mapping_info(node_mapping_info)


    def total_cost(self):
        '''
        '''
        return self.to_parent_cost + self.to_out_cost

    def get_candidate_xb(self, radius = 2):
        ''''
        '''
        cima_node_name = self.addr_node
        device_name = cima_node_name.split('.')[0]
        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height
        node_index = int(cima_node_name.split('.')[1].split(':')[1])
        node_id = [node_index // MESH_WIDTH, node_index % MESH_WIDTH]

        assert node_id[0] <= MESH_HEIGHT
        x_direction  = []
        y_direction = []
        for i in range(1, radius+1):
            # x direciton
            x_left = node_id[0] - i
            x_right = node_id[0] + i
            if (x_left) >= 0 and x_left not in x_direction:
                x_direction.append(x_left)
            if (x_right) <= MESH_WIDTH - 1 and x_right not in x_direction:
                x_direction.append(x_right)
            # y direciton
            y_top = node_id[1] - i
            y_bottom = node_id[1] + i
            if y_top >= 0 and y_top not in y_direction:
                y_direction.append(y_top)
            if y_bottom <= MESH_WIDTH - 1 and y_bottom not in y_direction:
                y_direction.append(y_bottom)

        available_xb_id = []
        for x in x_direction:
            for y in y_direction:
                for j in range(4):
                    xb_name = f'{device_name}.cima-node:{4*x + y}.cima-xb:{j}'
                    if xb_name in self.available_nodes_xb:
                        available_xb_id.append(xb_name)

        return available_xb_id

    def get_to_parent_distance(self, pe_bind_direction=False, alpha=0.5):
        '''
        '''
        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height

        node1 = self.addr_node
        node_index1 = int(node1.split('.')[1].split(':')[1])
        node_id1 = [node_index1 // MESH_WIDTH, node_index1 % MESH_WIDTH]
        assert node_id1[0] <= MESH_HEIGHT

        distance = 0

        record_io_workload_current = {}

        for parent in self.parent:

            node2 = parent.addr_node

            node_index2 = int(node2.split('.')[1].split(':')[1])
            node_id2 = [node_index2 // MESH_WIDTH, node_index2 % MESH_WIDTH]
            assert node_id2[0] <= MESH_HEIGHT

            if pe_bind_direction:
                #
                parent_direction = int(node2.split('.')[2].split(':')[1])
                original_node_id2 = copy.deepcopy(node_id2)
                if parent_direction == 0:
                    # north
                    if node_id2[0] > 0:
                        node_id2[0] =  node_id2[0] - 1
                        distance += 1
                elif parent_direction == 1:
                    # east
                    if node_id2[1] < MESH_WIDTH - 1:
                        node_id2[1] =  node_id2[1] + 1
                        distance += 1
                elif parent_direction == 2:
                    # south
                    if node_id2[0] < MESH_HEIGHT - 1:
                        node_id2[0] =  node_id2[0] + 1
                        distance += 1
                elif parent_direction == 3:
                    # west
                    if node_id2[1] > 0:
                        node_id2[1] =  node_id2[1] - 1
                        distance += 1

                if node_id2 != node_id1 and original_node_id2 != node_id2:
                    pair1 = str(original_node_id2)+'-'+str(node_id2)
                    if pair1 not in record_io_workload_current.keys():
                        record_io_workload_current[pair1] = 1
                    else:
                        record_io_workload_current[pair1] += 1

            walked_node_list = self.get_walked_list(node_id1, node_id2)

            if walked_node_list != []:

                pair1 = str(node_id2)+'-'+str(walked_node_list[0])
                # pair1 = [str(node_id1), str(walked_node_list[0])]
                assert(node_id2 != walked_node_list[0])
                if pair1 not in record_io_workload_current.keys():
                    record_io_workload_current[pair1] = 1
                else:
                    record_io_workload_current[pair1] += 1

                for index in range(len(walked_node_list) - 1):
                    #
                    pair = str(walked_node_list[index])+'-'+str(walked_node_list[index+1])
                    # pair = [str(walked_node_list[index]), str(walked_node_list[index+1])]
                    assert(walked_node_list[index] != walked_node_list[index+1])
                    if pair not in record_io_workload_current.keys():
                        record_io_workload_current[pair] = 1
                    else:
                        record_io_workload_current[pair] += 1

            for i in range(2):
                distance += abs(node_id1[i] - node_id2[i])

            if walked_node_list != []:

                if self.record_io_workload_total != {}:
                    pair1 = str(node_id2)+'-'+str(walked_node_list[0])
                    if pair1 in self.record_io_workload_total.keys():
                        distance += self.record_io_workload_total[pair1] * alpha


                for index in range(len(walked_node_list) - 1):
                    #
                    pair = str(walked_node_list[index])+'-'+str(walked_node_list[index+1])
                    # pair = [str(walked_node_list[index]), str(walked_node_list[index+1])]
                    if self.record_io_workload_total != {}:
                        if pair in self.record_io_workload_total.keys():
                            distance += self.record_io_workload_total[pair] * alpha

                    if pair in record_io_workload_current.keys():
                        distance += record_io_workload_current[pair] * alpha


        return distance, record_io_workload_current

    def get_walked_list(self, node_id1, node_id2):

        '''
        '''

        node_list = []

        x_start = node_id2[1] # child node
        y_start = node_id2[0] # parent node

        y_dis = node_id1[0] - node_id2[0]
        x_dis = node_id1[1] - node_id2[1]

        if y_dis == 0 and x_dis == 0:
            return node_list

        if x_dis > 0:
            for x_ in range(abs(x_dis)):
                node_list.append([node_id2[0], x_start+x_+1])
        elif x_dis <0:
            for x_ in range(abs(x_dis)):
                node_list.append([node_id2[0], x_start-x_-1])

        if node_list != []:
            new_x = node_list[-1][1]
        else:
            new_x = node_id2[1]

        for y_ in range(abs(y_dis)):
            if y_dis > 0:
                node_list.append([y_start+y_+1, new_x])
            elif y_dis <0:
                node_list.append([y_start-y_-1, new_x])

        return node_list

    def get_to_out_distance(self, pe_bind_direction=False, alpha=0.5):

        MESH_WIDTH = self.mesh_width
        MESH_HEIGHT = self.mesh_height

        node1 = self.addr_node
        node_index1 = int(node1.split('.')[1].split(':')[1])
        node_id1 = [node_index1 // MESH_WIDTH, node_index1 % MESH_WIDTH]
        assert node_id1[0] <= MESH_HEIGHT
        walked_node_list = self.get_walked_list((3,0), node_id1)

        record_io_workload_current = {}

        if walked_node_list != []:

            pair1 = str(node_id1)+'-'+str(walked_node_list[0])
            # pair1 = [str(node_id1), str(walked_node_list[0])]
            assert(node_id1 != walked_node_list[0])
            if pair1 not in record_io_workload_current.keys():
                record_io_workload_current[pair1] = 1
            else:
                record_io_workload_current[pair1] += 1

            for index in range(len(walked_node_list) -1 ):

                pair = str(walked_node_list[index])+'-'+str(walked_node_list[index+1])
                # pair = [str(walked_node_list[index]), str(walked_node_list[index+1])]
                assert(walked_node_list[index] != walked_node_list[index+1])
                if pair not in record_io_workload_current.keys():
                    record_io_workload_current[pair] = 1
                else:
                    record_io_workload_current[pair] += 1

        distance = 0
        if pe_bind_direction:
            direction1 = int(node1.split('.')[2].split(':')[1])
            if direction1 == 1:
                # east
                node_id1[1] =  node_id1[1] + 1
                distance += 1
            elif direction1 in [0, 2]:
                # north, south
                distance += 1

        distance = node_id1[1] + 1

        if walked_node_list != []:

            if self.record_io_workload_total != {}:
                pair1 = str(node_id1)+'-'+str(walked_node_list[0])
                # pair1 = [str(node_id1), str(walked_node_list[0])]
                if pair1 in self.record_io_workload_total.keys():
                    distance += self.record_io_workload_total[pair1] * alpha

            for index in range(len(walked_node_list) - 1):
                #
                pair = str(walked_node_list[index])+'-'+str(walked_node_list[index+1])
                # pair = [str(walked_node_list[index]), str(walked_node_list[index+1])]

                if self.record_io_workload_total != {}:
                    if pair in self.record_io_workload_total.keys():
                        distance += self.record_io_workload_total[pair] * alpha

                if pair in record_io_workload_current.keys():
                    distance += record_io_workload_current[pair] * alpha

        return distance, record_io_workload_current


class CIMAPlacement(object):

    def __init__(self, node_weight, XB_size):
        '''
        '''
        self.node_weight = node_weight
        self.XB_size = XB_size

    def run(self):
        '''
        return:
        '''
        all_layer_addr = {}

        for split_layer_name in self.node_weight.keys():
            layer_name = split_layer_name.split('.')[0]
            if layer_name not in all_layer_addr:
                all_layer_addr[layer_name] = []
            # weight_shape = [int(self.node_weight[split_layer_name][1]), int(self.node_weight[split_layer_name][0])]
            #     pass
            node_addr = {split_layer_name:[0, 0, int(self.node_weight[split_layer_name][1]), int(self.node_weight[split_layer_name][0]) ]}
            all_layer_addr[layer_name].append(node_addr)

        return all_layer_addr

#=================================================================
#                 CIMA DMEM ALLOCATION ALGORITHM
#=================================================================

def get_assemble_layers(layer_info):
    '''
    '''
    assemble_layers = []
    for k,v in layer_info.items():
        if v.type == 'op' and v.op.op_id in ['add', 'concat', 'fused_add', 'fused_concat']:
            IsAddConstant = False
            for i in v.inputs:
                if 'Constant' in i.ref:
                    IsAddConstant = True
                    break
            if not IsAddConstant:
                assemble_layers.append(k)
            # assemble_layers.append(k)
    return assemble_layers

def get_max_length_direct_graph(layer_depth_index_dict, start_layer_name, end_layer_name):
    '''
    '''
    max_length_direct_graph = {}
    while True:
        layer_depth_index = layer_depth_index_dict[start_layer_name]
        max_depth_index = max(layer_depth_index.keys())
        max_depth_layer_name = layer_depth_index[max_depth_index]
        max_length_direct_graph[start_layer_name] = max_depth_layer_name
        if max_depth_layer_name == end_layer_name:
            break
        #
        start_layer_name = max_depth_layer_name
    return max_length_direct_graph

def get_min_length_direct_graph(layer_depth_index_dict, start_layer_name, end_layer_name):
    '''
    '''
    min_length_direct_graph = {}
    while True:
        layer_depth_index = layer_depth_index_dict[start_layer_name]
        min_depth_index = min(layer_depth_index.keys())
        min_depth_layer_name = layer_depth_index[min_depth_index]
        min_length_direct_graph[start_layer_name] = min_depth_layer_name
        if min_depth_layer_name in end_layer_name:
            break
        #
        start_layer_name = min_depth_layer_name
    return min_length_direct_graph


def get_layer_depth_index_forward(next_layers_dict, pre_layer_count, #
                                  current_layer_name, pre_layer_name, layer_depth_index, layer_count, layer_depth_index_dict):
    '''
    '''

    if current_layer_name not in layer_count.keys():
        layer_count[current_layer_name] = 1
    else:
        layer_count[current_layer_name] += 1
    if 'graph_input' not in current_layer_name and current_layer_name not in layer_depth_index_dict.keys():
        layer_depth_index_dict[current_layer_name] = {}
    if pre_layer_name != None:
        if layer_depth_index not in layer_depth_index_dict[current_layer_name].keys():
            layer_depth_index_dict[current_layer_name][layer_depth_index] = pre_layer_name
    if 'graph_output' not in current_layer_name:
        if layer_count[current_layer_name] == pre_layer_count[current_layer_name]:
            next_layers_name_list = next_layers_dict[current_layer_name]
            for nl in next_layers_name_list:
                if 'graph_input' in current_layer_name:
                    get_layer_depth_index_forward(next_layers_dict, pre_layer_count,
                                                    nl, current_layer_name, layer_depth_index+1, layer_count, layer_depth_index_dict)
                else:
                    for k in layer_depth_index_dict[current_layer_name].keys():
                        # get_layer_depth_index_forward(next_layers_dict, pre_layer_count,
                        #                                 nl, current_layer_name, layer_depth_index+1, layer_count, layer_depth_index_dict)
                        get_layer_depth_index_forward(next_layers_dict, pre_layer_count,
                                                        nl, current_layer_name, k+1, layer_count, layer_depth_index_dict)

def DMEM_backforward_inference(layer_info, layer_dmem_size, layer_direct_graph, alpha=1.001):
    '''
    '''
    output_data_size = 1
    output_dmem_size = 1
    current_layer_name = list(layer_direct_graph.keys())[0]
    # first_layer_name = current_layer_name
    while True:
        current_layer_info = layer_info[current_layer_name]
        if current_layer_info.type == 'op' and current_layer_info.op.op_id in ['conv2d', 'maxpool2d', 'averagepool2d', 'fused_conv2d']:
            stride = current_layer_info.op.stride
            kernel = current_layer_info.op.kernel
            padding = current_layer_info.op.padding
            input_data_size = max(1, (output_data_size-1)*stride + kernel-padding)
            if stride == 2:
                input_data_size += 1
        elif current_layer_info.type == 'op' and current_layer_info.op.op_id in ['resize']:
            # input_data_size = math.ceil(output_data_size / current_layer_info.op.scale[-1])
            input_data_size = output_data_size / current_layer_info.op.scale[-1]
        else:
            input_data_size = output_data_size
        input_data_size = input_data_size * alpha

        if current_layer_info.type == 'op' and current_layer_info.op.op_id in ['conv2d', 'maxpool2d', 'averagepool2d']:
            assert current_layer_name in layer_dmem_size.keys(), f'{current_layer_name}'
            input_dmem_size = output_dmem_size * stride + layer_dmem_size[current_layer_name]
            #     input_dmem_size = input_dmem_size - padding
            input_dmem_size = input_dmem_size - padding

        else:
            input_dmem_size = output_dmem_size + layer_dmem_size[current_layer_name]

        output_data_size = input_data_size
        output_dmem_size = input_dmem_size
        if current_layer_name not in layer_direct_graph.keys():
            break
        current_layer_name = layer_direct_graph[current_layer_name]

    return output_data_size, output_dmem_size

def get_minimal_circle_subgraph_end_layer_v2(pre_layers_dict, start_layer_name):
    '''
    '''
    end_layer_name = None
    pre_layers_path = {}
    for v in pre_layers_dict.get(start_layer_name):
        pre_layers_path[v] = [v]
    search_index = 0
    while True:
        all_pre_layers = {}
        for v in pre_layers_path.keys():
            if search_index >= len(pre_layers_path[v]):
                continue
            current_layer = pre_layers_path[v][search_index]
            if current_layer in pre_layers_dict.keys():
                pre_layers = pre_layers_dict.get(current_layer)
                all_pre_layers[v] = pre_layers
                for pl in pre_layers:
                    if pl not in pre_layers_path[v] and 'Constant' not in pl:
                        pre_layers_path[v].append(pl)

        find_layers = {}
        for layer_name, current_pre_layers_list in all_pre_layers.items():
            for cpl in current_pre_layers_list:
                is_found_in_all = True
                for ln, searched_path in pre_layers_path.items():
                    if ln == layer_name:
                        continue
                    if cpl not in searched_path:
                        is_found_in_all = False
                        break
                if is_found_in_all:
                    if 'graph_input' in cpl:
                        layer_suffix = -1
                    else:
                        if cpl.split('_')[-1] == 'Split':
                            layer_suffix = int(cpl.split('_')[-2])
                        else:
                            layer_suffix = int(cpl.split('_')[-1])
                    find_layers[layer_suffix] = cpl

        if find_layers != {}:
            maximal_suffix = max(find_layers.keys())
            end_layer_name = find_layers[maximal_suffix]
            break
        else:
            search_index += 1

    if 'identity' in end_layer_name and 'mcast' in end_layer_name and 'seg' in end_layer_name:
        end_layer_name = '_'.join(end_layer_name.split('_')[:2])
    return end_layer_name


def MinMaxEstimation(ir_graph, alpha=1.0002, IsSingleFrame=True):
    '''
    '''
    layer_info = ir_graph.layers

    layer_dmem_size = {}

    NON_KERNEL_DMEM_SIZE = 1
    for k, v in layer_info.items():
        if v.type == 'op' and v.op.op_id in ['conv2d', 'maxpool2d', 'averagepool2d', 'fused_conv2d']:
            stride = v.op.stride
            kernel = v.op.kernel
            # padding = v.op.padding
            # dmem_size = max(stride, kernel)
            dmem_size = kernel
            #
            layer_dmem_size[k] = dmem_size
        elif v.type == 'op' and v.op.op_id in ['constant']:
            layer_dmem_size[k] = NON_KERNEL_DMEM_SIZE
        else:
            IsScaleUp = False
            scale = 1
            if v.type == 'op':
                for i in v.inputs:
                    if ':' in i.ref:
                        ref = i.ref.split(':')[0]
                    else:
                        ref = i.ref
                    if layer_info[ref].type == 'op' and layer_info[ref].op.op_id in ['resize']:
                        IsScaleUp = True
                        scale = layer_info[ref].op.scale[-1]
                        break
            if IsScaleUp:
                layer_dmem_size[k] = scale * NON_KERNEL_DMEM_SIZE
            else:
                layer_dmem_size[k] = NON_KERNEL_DMEM_SIZE
            # layer_dmem_size[k] = NON_KERNEL_DMEM_SIZE

    assemble_layers = get_assemble_layers(layer_info)

    layer_depth_index_dict = {}
    # pre layer
    pre_layers_dict  = get_pre_layer(layer_info)
    # next layer
    next_layers_dict = get_next_layer(layer_info)

    pre_layer_count = {}
    for k,v in pre_layers_dict.items():
        sum_ = 0
        for i in v:
            if 'Constant' not in i:
                sum_ += 1
        pre_layer_count[k] = sum_
        # pre_layer_count[k] = len(v)
    #
    pre_layer_count['graph_input:0'] = 1

    #
    layer_count = {}
    get_layer_depth_index_forward(next_layers_dict, pre_layer_count, #
                                  'graph_input:0', None, 0, layer_count, layer_depth_index_dict)
    #

    max_length_layer = {}
    while assemble_layers != []:
        assemble_layer_name = assemble_layers.pop(0)
        end_layer_name = get_minimal_circle_subgraph_end_layer_v2(pre_layers_dict, assemble_layer_name)
        # # print(end_layer_name)
        max_length_direct_graph = get_max_length_direct_graph(layer_depth_index_dict, assemble_layer_name, end_layer_name)
        min_length_direct_graph = get_min_length_direct_graph(layer_depth_index_dict, assemble_layer_name, end_layer_name)
        #     input()
        max_data_size, max_dmem_size = DMEM_backforward_inference(layer_info, layer_dmem_size, max_length_direct_graph, alpha=alpha)
        min_data_size, min_dmem_size = DMEM_backforward_inference(layer_info, layer_dmem_size, min_length_direct_graph, alpha=alpha)
        while max_data_size > min_dmem_size:
            layer_dmem_size[assemble_layer_name] += 1
            max_data_size, max_dmem_size = DMEM_backforward_inference(layer_info, layer_dmem_size, max_length_direct_graph, alpha=alpha)
            min_data_size, min_dmem_size = DMEM_backforward_inference(layer_info, layer_dmem_size, min_length_direct_graph, alpha=alpha)

        if IsSingleFrame:
            fin_height = layer_info[assemble_layer_name].inputs[0].height
            layer_dmem_size[assemble_layer_name] = min(layer_dmem_size[assemble_layer_name], fin_height)

        #     layer_dmem_size[assemble_layer_name] = int(math.ceil(max_data_size - min_dmem_size)) + layer_dmem_size[assemble_layer_name]
        # input()

        max_length_layer[assemble_layer_name] = len(max_length_direct_graph.keys())

    return layer_dmem_size, max_length_layer, layer_depth_index_dict

def DMEM_place_algorithm(replace_dmem_capacity, current_layer_single_row, current_core_loc, pre_core_loc, dmem_rest,
                         layer_core_num, MAX_THREAD_NUM, Strategy = 'Uniform_Thread', IsConcat = False):
    '''
    '''

    #
    replace_core_loc = {}
    if Strategy == 'Minimal_Layer':
        dmem_rest_sorted = sorted(dmem_rest.items(), key=lambda x:x[1], reverse=True)
        HitCurrentCore = True
        while True:
            if HitCurrentCore:
                if not IsConcat:
                    provided_dmem = max(1, math.floor(min(dmem_rest[current_core_loc], replace_dmem_capacity) / current_layer_single_row)) *  current_layer_single_row
                else:
                    provided_dmem = current_layer_single_row
                if dmem_rest[current_core_loc] >= provided_dmem:
                    #
                    replace_core_loc[current_core_loc] = 0
                    #
                    replace_core_loc[current_core_loc] += provided_dmem
                    # update core num
                    layer_core_num[current_core_loc] += 1
                    # update dmem rest
                    dmem_rest[current_core_loc] -= provided_dmem
                    # update replace_dmem_capacity
                    replace_dmem_capacity = replace_dmem_capacity - provided_dmem
                    #
                    HitCurrentCore = False
                    #
                    if replace_dmem_capacity <= 0:
                        break
            for re in dmem_rest_sorted:
                if re[0] not in [current_core_loc, pre_core_loc] and layer_core_num[re[0]] < MAX_THREAD_NUM:
                    provided_dmem = math.floor(min(re[1], replace_dmem_capacity) / current_layer_single_row) *  current_layer_single_row
                    #
                    if re[0] not in replace_core_loc.keys():
                        replace_core_loc[re[0]] = 0
                    #
                    replace_core_loc[re[0]] += provided_dmem
                    # update core num
                    layer_core_num[re[0]] += 1
                    # update dmem rest
                    dmem_rest[re[0]] -= provided_dmem
                    # update replace_dmem_capacity
                    replace_dmem_capacity = replace_dmem_capacity - provided_dmem
                    #
                    if replace_dmem_capacity <= 0:
                        break
            #
            if replace_dmem_capacity <= 0:
                break

    elif Strategy == 'Uniform_Thread':
        #
        HitCurrentCore = True
        while True:
            dmem_rest_sorted = sorted(dmem_rest.items(), key=lambda x:x[1], reverse=True)
            dmem_mean = sum(dmem_rest.values()) / len(dmem_rest.values())
            if HitCurrentCore:
                diff_mean = dmem_rest[current_core_loc] - dmem_mean
                if IsConcat:
                    provided_dmem = current_layer_single_row
                else:
                    provided_dmem = max(1, math.floor(min(diff_mean, replace_dmem_capacity) / current_layer_single_row)) *  current_layer_single_row

                if dmem_rest[current_core_loc] >= provided_dmem:
                    #
                    replace_core_loc[current_core_loc] = 0
                    #
                    replace_core_loc[current_core_loc] += provided_dmem
                    # update core num
                    layer_core_num[current_core_loc] += 1
                    # update dmem rest
                    dmem_rest[current_core_loc] -= provided_dmem
                    # update replace_dmem_capacity
                    replace_dmem_capacity = replace_dmem_capacity - provided_dmem
                    #
                    HitCurrentCore = False
                    #
                    if replace_dmem_capacity <= 0:
                        break
            for re in dmem_rest_sorted:
                if re[0] not in [pre_core_loc, current_core_loc] and layer_core_num[re[0]] < MAX_THREAD_NUM:
                    #
                    if re[1] < dmem_mean:
                        continue
                    diff_mean = re[1] - dmem_mean
                    provided_dmem = max(1, math.floor(min(diff_mean, replace_dmem_capacity) / current_layer_single_row)) *  current_layer_single_row
                    if (re[1] - provided_dmem < 0) or provided_dmem == 0:
                        continue
                    #
                    if re[0] not in replace_core_loc.keys():
                        replace_core_loc[re[0]] = 0
                    #
                    replace_core_loc[re[0]] += provided_dmem
                    # update core num
                    if re[0] not in replace_core_loc.keys():
                        layer_core_num[re[0]] += 1
                    # update dmem rest
                    dmem_rest[re[0]] -= provided_dmem
                    # update replace_dmem_capacity
                    replace_dmem_capacity = replace_dmem_capacity - provided_dmem

                    #
                    if replace_dmem_capacity <= 0:
                        break
            #
            if replace_dmem_capacity <= 0:
                break

    return replace_core_loc, dmem_rest


def CIMA_DMEM_allocation(ir_graph, alpha=1.0002, IsSingleFrame=True):
    #
    DMEM_MAX_SIZE = 16384 #  unit: flit(256bit)
    MAX_THREAD_NUM = 32
    #
    layer_dmem_size, _, layer_depth_index_dict = MinMaxEstimation(ir_graph, alpha=alpha, IsSingleFrame=IsSingleFrame)
    dmem_start = {}
    # dmem_exceed_core = []
    dmem_exceed_layer = {}
    dmem_core_layer = {}
    dmem_single_row_value = {}
    layer_core_loc = {}
    layer_core_num = {}
    #
    for k,v in ir_graph.layers.items():
        if v.type == 'op' and v.CIMA_mapping_info != None:
            core_loc = v.CIMA_mapping_info.mappings[(0,0,0)].device
            core_loc = ('.').join(core_loc.split('.')[:2])
            #
            layer_core_loc[k] = core_loc
            #
            if core_loc not in layer_core_num.keys():
                layer_core_num[core_loc] = 0
            if v.op.op_id not in ['conv2d', 'fused_conv2d', 'fc', 'fused_fc',
                                    'maxpool2d', 'resize', 'avgpool2d']:
                layer_core_num[core_loc] += 1
            dmem_row_num = layer_dmem_size[k]
            in_channel_num = v.inputs[0].channel
            fin_width = v.inputs[0].width
            if v.op.op_id in ['concat', 'fused_concat']:
                in_channel_num = in_channel_num * len(v.inputs)
            dmem_size = int(math.ceil(dmem_row_num * in_channel_num * fin_width * 4 / 256)) # unit: flit/256bit
            dmem_single_row_value[k] = int(math.ceil(in_channel_num * fin_width  / 64))
            #
            if core_loc not in dmem_start.keys():
                dmem_start[core_loc] = 640
            ir_graph.layers[k].CIMA_mapping_info.in_line_buffer_addr[0][0] = dmem_start[core_loc]
            ir_graph.layers[k].CIMA_mapping_info.in_line_buffer_addr[0][1] = dmem_size
            ir_graph.layers[k].CIMA_mapping_info.credit_len[0] = fin_width
            #
            #     dmem_start[core_loc] += dmem_size
            # elif core_loc not in dmem_exceed_core:
            if dmem_size > DMEM_MAX_SIZE // 2:
                # dmem_exceed_core.append(core_loc)
                #
                dmem_exceed_layer[k] = [core_loc, dmem_size]
            elif v.op.op_id in ['concat', 'fused_concat'] and dmem_size > dmem_single_row_value[k]:
                dmem_exceed_layer[k] = [core_loc, dmem_size]
            else:
                dmem_start[core_loc] += dmem_size
            #
            if core_loc not in dmem_core_layer.keys():
                dmem_core_layer[core_loc] = []
            dmem_core_layer[core_loc].append([k, dmem_size])

    dmem_rest = {}
    for k,v in dmem_start.items():
        dmem_rest[k] = DMEM_MAX_SIZE - v
    #
    for k,v in dmem_exceed_layer.items():
        core_loc = v[0]
        # current_dmem_core_layer = sorted(dmem_core_layer[core_loc], key=lambda x:x[1], reverse=False)
        # sum_ = 0
        # exceed_layers = []
        #     sum_ += op[1]
        #         #
        #         exceed_layers.append([op[0], op[1], sum_ - DMEM_MAX_SIZE])
        # #
        #     #
        pre_layers = layer_depth_index_dict[k]
        pl_dict = {}
        max_index = 0
        for k_1, v_1 in pre_layers.items():
            if k_1 > max_index:
                max_index = k_1
            if v_1 not in pl_dict.keys():
                pl_dict[v_1] = []
            pl_dict[v_1].append(k_1)

        for k_2, v_2 in pl_dict.items():
            if max_index in v_2:
                continue
            # nearest_pre_layer_name = pre_layers[min(pre_layers.keys())]
            nearest_pre_layer_name = k_2
            current_core = core_loc
            nearest_pre_core = layer_core_loc[nearest_pre_layer_name]
            current_layer_single_row = dmem_single_row_value[k]
            exceed_capacity = v[1]
            while exceed_capacity % current_layer_single_row != 0:
                exceed_capacity += 1
            #
            if 'Concat' in k:
                IsConcat = True
            else:
                IsConcat = False
            # IsConcat = False
            replace_core_results, dmem_rest = DMEM_place_algorithm(exceed_capacity, current_layer_single_row, current_core,
                                                                    nearest_pre_core, dmem_rest, layer_core_num, MAX_THREAD_NUM,
                                                                    Strategy='Uniform_Thread',
                                                                    IsConcat=IsConcat)
            #
            rc_index = 0
            for rc_k, rc_v in replace_core_results.items():
                layer_name = k
                if rc_k == current_core:
                    ir_graph.layers[layer_name].CIMA_mapping_info.in_line_buffer_addr[0][1] = rc_v
                    continue
                identity_name = f'{layer_name}_{k_2}_dmem_identiy_{rc_index}'
                op_ = make_op('identity')
                if rc_index == 0:
                    ref_layer_name = None
                    for in_ in ir_graph.layers[layer_name].inputs:
                        if nearest_pre_layer_name in in_.ref:
                            ref_layer_name = in_.ref
                            break
                    assert ref_layer_name != None, f'Message translated to English.'
                else:
                    ref_layer_name = f'{layer_name}_{k_2}_dmem_identiy_{rc_index-1}'
                input_shape = ir_graph.layers[ref_layer_name].outputs[0]
                inputs_ = [dict(ref=ref_layer_name, channel=input_shape.channel, height=input_shape.height, width=input_shape.width)]
                outputs_ = [dict(channel=input_shape.channel,height=input_shape.height,width=input_shape.width)]
                ir_graph.add_layer(identity_name, op=op_, inputs=inputs_, outputs=outputs_)
                in_line_buffer_addr = [[dmem_start[rc_k], rc_v]]
                credit_len = [ir_graph.layers[layer_name].inputs[0].width]
                loc_info = [CIMADeviceMappingInfo(index = [0,0,0], device=rc_k, address=0)]
                ir_graph.layers[identity_name].CIMA_mapping_info = CIMAMappingInfo(col_split_num=None,row_split_num=None,
                                                            col_repeat_num=None,row_repeat_num=None,
                                                            para_diff_array=None,in_line_buffer_addr = in_line_buffer_addr,
                                                            credit_len = credit_len,
                                                            mappings=loc_info)
                ir_graph.layers[identity_name].CIMA_calc_info = CIMACalcInfo().clone()
                #
                if rc_index == len(replace_core_results.values()) - 2:
                    for in_ in ir_graph.layers[layer_name].inputs:
                        if nearest_pre_layer_name in in_.ref:
                            in_.ref = identity_name
                            break

                dmem_start[rc_k] += rc_v
                #
                rc_index += 1

    #
    ir_graph.layers = dict(ir_graph.iter_layers(deep=False, sorted=True))
    dmem_start = {}
    for k,v in ir_graph.layers.items():
        if v.type == 'op' and v.CIMA_mapping_info != None:
            core_loc = v.CIMA_mapping_info.mappings[(0,0,0)].device
            core_loc = ('.').join(core_loc.split('.')[:2])
            if core_loc not in dmem_start.keys():
                dmem_start[core_loc] = 640
            ir_graph.layers[k].CIMA_mapping_info.in_line_buffer_addr[0][0] = dmem_start[core_loc]
            dmem_start[core_loc] += v.CIMA_mapping_info.in_line_buffer_addr[0][1]

    return ir_graph



