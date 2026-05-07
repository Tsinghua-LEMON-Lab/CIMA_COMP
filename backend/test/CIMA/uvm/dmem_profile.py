import json
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math

def get_task_memory_allocation(path, save_fig=f'Dmem_allocation.svg', utilization=False):

    devices = ['HOSTI', 'DDR0', 'DDR1', 'DDR2', 'DDR3']
    # devices = []

    for i in range(4):
        for j in range(9):
           devices.append(f'Core{i}_{j}')

    with open(path, 'r') as f:
        sim_obj = json.load(f)

    # mem_task_conv = {}
    # mem_task_other = {}

    mem_task = {}
    max_dmem_size = 0
    addr_step = 128

    for k,v in sim_obj.items():
        if k in devices and k not in mem_task.keys():
            mem_task[k] = {}
            for thread, value in v.items():

                #         mem_task_conv[k] = {}
                #     mem_task_conv[k][value["Task_Name"]] = ( value["Dmem_Base"], value["Dmem_Base"] + value["Dmem_Size"])
                # else:
                #         mem_task_other[k] = {}
                #     mem_task_other[k][value["Task_Name"]] = ( value["Dmem_Base"], value["Dmem_Base"] + value["Dmem_Size"])

                # all
                if value["Task_Name"] not in ['Input', 'Output']:
                    mem_task[k][value["Task_Name"]] = ( value["Dmem_Base"], value["Dmem_Base"] + value["Dmem_Size"])

                    if value["Dmem_Base"] + value["Dmem_Size"] >= max_dmem_size:
                        max_dmem_size = value["Dmem_Base"] + value["Dmem_Size"]
    while not max_dmem_size % 256 == 0:
        max_dmem_size = max_dmem_size + 1

    max_dmem_size += max_dmem_size // addr_step


    if utilization:
        draw_mem_ultilizaiton(mem_task, save_fig)
    else:
        draw_memory_profile(mem_task, addr_limit = max_dmem_size, addr_step = addr_step, save_fig = save_fig)
    #

def draw_memory_profile(mem_task, addr_limit = 0x100000, addr_step = 64, save_fig=None):

    sorted_mem_tasks = {mem: dict(sorted(tasks.items(), key=lambda x: x[1][0])) for mem, tasks in mem_task.items()}

    fig, ax = plt.subplots(figsize = (18 * math.ceil(addr_step/64), 9))

    mem = list(sorted_mem_tasks.keys())
    tasks = list(sorted_mem_tasks.values())

    height = 0.6

    for i, (mem1, tasks_dict) in enumerate(sorted_mem_tasks.items()):
        # thread_id = 0
        for task, (start, end) in tasks_dict.items():
            random_color = generate_low_saturation_color()

            rect_width = end - start
            rect_center = (start + end) / 2
            ax.barh(i, rect_width, left=start, height=height, color = random_color, edgecolor= random_color, linewidth=0.5)
            ax.text(rect_center, i, task, fontsize=8, color='black', ha='center', va='center', weight='bold')
            # ax.text(rect_center, i, f'[{thread_id}]', fontsize=8, color='black', ha='center', va='center', weight='bold')
            # thread_id += 1
        for j in range(0, addr_limit, addr_limit//addr_step):
            ax.add_patch(plt.Rectangle((j, i - height/ 2), addr_limit//addr_step, height, edgecolor='black', linewidth=0.5, facecolor='none'))

    ax.set_ylim(-1, len(sorted_mem_tasks))
    ax.set_xlim(0, addr_limit)
    ax.set_yticks(range(len(sorted_mem_tasks)))
    ax.set_yticklabels(mem)
    ax.set_xlabel('Dmem Address / 32Byte', fontsize=14)
    ax.set_ylabel('Core', fontsize=14)
    ax.set_title('Task Distribution on Core Dmem', fontsize=14)

    plt.tight_layout()
    # plt.show()
    if save_fig != None:
        plt.savefig(save_fig)
    else:
        plt.show()

def generate_low_saturation_color():
    hue = np.random.rand()

    saturation = 0.2
    value = np.random.randint(7, 10) / 10

    rgb_color = colorsys.hsv_to_rgb(hue, saturation, value)

    return rgb_color

def draw_mem_ultilizaiton(mem_task, save_fig = f'1.svg'):

    sorted_mem_tasks = {mem: dict(sorted(tasks.items(), key=lambda x: x[1][0])) for mem, tasks in mem_task.items()}

    core_name = []
    max_utilization = []

    plt.figure(figsize=(10,5))
    max_capacity = 32768 #unit: flit(256 bit)
    for k,v in sorted_mem_tasks.items():

        if list(sorted_mem_tasks[k].keys()) != []:
            max_layer = list(sorted_mem_tasks[k].keys())[-1]
            max_mem_addr = sorted_mem_tasks[k][max_layer][1]
            core_name.append(k)

            max_utilization.append(max_mem_addr/ max_capacity * 100)

    for i, txt in enumerate(max_utilization):
        plt.annotate(round(txt,1), (core_name[i], max_utilization[i]), textcoords="offset points",
                     xytext=(0,10), ha='center', fontsize=8)

    plt.scatter(core_name, max_utilization, label=f'Max Capacity = {max_capacity * 256 / (8 * 1024 * 1024)} MB')
    plt.plot(core_name, max_utilization)
    # plt.ylim(0, 60)
    plt.ylabel(f'Dmem Utilization / %')

    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig(save_fig)


if __name__ == "__main__":

    path = f'yolov5\\'
    file = path + f'yolov5_all_layers_mapped_params_head_split_wo_Conv_0.json'

    # get_task_memory_allocation(file, save_fig=path + f'Dmem_allocation.pdf', utilization = False)
    get_task_memory_allocation(file, save_fig=path + f'Dmem_utilization.svg', utilization=True)
