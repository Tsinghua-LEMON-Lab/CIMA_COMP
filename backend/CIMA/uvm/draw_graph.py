import hashlib

from graphviz import Digraph

from irtool.core.ir import BaseIR, load_ir
from mapper.device.CIMA import *
from mapper.self_defined_op.fused_op import *

"""IR graphviz converter."""

_INP_A = dict(shape="box", style="rounded,filled", color="skyblue")
_INP_E = dict(penwidth="3", color="blue")
_IO_A = dict(shape="ellipse", style="filled", fillcolor="white", color="black")
_DEV_T, _DAT_T = 24, 20


def data_label(name, core_id, hardware_thread_id=None, dmem_size=None):
    lines = [str(name), "", f"Core location: {core_id}"]
    if hardware_thread_id is not None:
        lines.append(f"Hardware thread ID: {hardware_thread_id}")
    if dmem_size is not None:
        lines.append(f"Hardware DMEM size: {dmem_size}")
    return "\n".join(lines)


def node_id(name):
    # Keep node names readable in labels, but use safe IDs for DOT internals.
    return "n_" + hashlib.md5(str(name).encode("utf-8")).hexdigest()


def canonical_node_name(name):
    name = str(name)
    if name.startswith("graph_input"):
        return "graph_input"
    if name.startswith("graph_output"):
        return "graph_output"
    return name


def resolve_meta_key(meta_dict, raw_name):
    """Resolve metadata key for nodes with optional segment suffix (e.g. layer:0)."""
    if not isinstance(meta_dict, dict):
        return None
    if raw_name in meta_dict:
        return raw_name
    raw_name_str = str(raw_name)
    if ":" in raw_name_str:
        base_name = raw_name_str.split(":")[0]
        if base_name in meta_dict:
            return base_name
    canonical_name = canonical_node_name(raw_name_str)
    if canonical_name in meta_dict:
        return canonical_name
    return None


def ir_to_dot(tree_dict, edge_info, core_info, layer_thread_info=None, layer_dmem_info=None):
    graph = Digraph()
    emitted_nodes = set()

    def to_graph_id(raw_name):
        return node_id(canonical_node_name(raw_name))

    def emit_node(raw_name):
        canon_name = canonical_node_name(raw_name)
        nid = to_graph_id(raw_name)
        if nid in emitted_nodes:
            return nid
        emitted_nodes.add(nid)

        # Keep graph_input / graph_output simple and visually consistent.
        if canon_name in {"graph_input", "graph_output"}:
            graph.node(nid, label=canon_name, **_IO_A)
            return nid

        core_key = resolve_meta_key(core_info, raw_name)
        core = core_info.get(core_key, "None")
        thread_key = resolve_meta_key(layer_thread_info, raw_name)
        dmem_key = resolve_meta_key(layer_dmem_info, raw_name)

        if (
            thread_key is not None
            and dmem_key is not None
        ):
            graph.node(
                nid,
                label=data_label(
                    raw_name,
                    core,
                    hardware_thread_id=layer_thread_info[thread_key],
                    dmem_size=layer_dmem_info[dmem_key],
                ),
                **_INP_A,
            )
        elif thread_key is not None:
            graph.node(nid, label=data_label(raw_name, core, hardware_thread_id=layer_thread_info[thread_key]), **_INP_A)
        elif dmem_key is not None:
            graph.node(nid, label=data_label(raw_name, core, dmem_size=layer_dmem_info[dmem_key]), **_INP_A)
        else:
            graph.node(nid, label=data_label(raw_name, core), **_INP_A)
        return nid

    for k, v in tree_dict.items():
        k_id = emit_node(k)
        for v_ in v:
            v_id = emit_node(v_)
            graph.edge(k_id, v_id, label=edge_info[k], **_INP_E)
    return graph


def get_pre_layer(layers):
    prefix_layer = {}
    for name, layer in layers.items():
        if layer.type not in ["input"]:
            if layer.type == "op" and layer.op.op_id in ["constant"]:
                continue
            prefix_layer[name] = []
            for i in layer.inputs:
                if "graph_input" not in i.ref and "constant" not in i.ref:
                    ref = i.ref
                    if ":" in ref:
                        ref = ref.split(":")[0]
                    prefix_layer[name].append(ref)
                else:
                    prefix_layer[name].append(i.ref)
    return prefix_layer


def get_tree_edge(layers, MESH_width=6):
    next_layer = {}
    pre_layer = get_pre_layer(layers)
    edge_info = {}
    core_info = {}

    pe_direction = {0: "N", 1: "E", 2: "S", 3: "W"}

    for k, v in pre_layer.items():
        for name in v:
            if name not in next_layer.keys():
                next_layer[name] = []
            next_layer[name].append(k)
            if name not in edge_info.keys():
                if "graph_input" in name:
                    out_channel = layers["graph_input"].inputs[0].channel
                    out_height = layers["graph_input"].inputs[0].height
                    out_width = layers["graph_input"].inputs[0].width
                else:
                    out_channel = layers[name].outputs[0].channel
                    out_height = layers[name].outputs[0].height
                    out_width = layers[name].outputs[0].width
                edge_info[name] = f"(1,{out_channel},{out_height},{out_width})"
            if name not in core_info.keys():
                if ":" in name:
                    name_ = name.split(":")[0]
                else:
                    name_ = name
                if "graph_input" not in name_ and layers[name_].CIMA_mapping_info is not None:
                    mapping_info = layers[name_].CIMA_mapping_info.mappings[(0, 0, 0)]
                    devices = mapping_info.device
                    device_ = devices.split(".")
                    if ":" in device_[1]:
                        core_index = int(device_[1].split(":")[-1])
                        core_id = f"[{core_index // MESH_width}][{core_index % MESH_width}]"

                        if len(device_) == 2:
                            core_info[name] = f"Core{core_id}"
                        elif len(device_) == 3:
                            if "dmac" in device_[2]:
                                core_info[name] = f"Core{core_id}:DMAC"
                            else:
                                pe_id = pe_direction[int(device_[2].split(":")[-1])]
                                core_info[name] = f"Core{core_id}:{pe_id}"
                        elif len(device_) == 4:
                            pe_id = pe_direction[int(device_[2].split(":")[-1])]
                            xb_id = device_[3].split(":")[-1]
                            core_info[name] = f"Core{core_id}:{pe_id}:XB{xb_id}"
                        else:
                            raise ValueError(f"Unsupported device {device_}")
                    elif "dram" in device_[1]:
                        core_info[name] = "DRAM"
                else:
                    core_info[name] = "None"

    return next_layer, edge_info, core_info


def draw_ir_tree(file, layer_thread_info=None, layer_dmem_info=None, MESH_width=6, save_pdf="1.pdf"):
    if isinstance(file, BaseIR):
        ir = file
    else:
        ir = load_ir(file=file)

    next_layer, edge_info, core_info = get_tree_edge(ir.layers, MESH_width=MESH_width)
    graph = ir_to_dot(next_layer, edge_info, core_info, layer_thread_info, layer_dmem_info)

    data = graph.pipe(format="pdf")

    if isinstance(data, (bytes, bytearray)):
        mode = "wb"
    else:
        mode = "w"

    with open(save_pdf, mode) as f:
        f.write(data)
