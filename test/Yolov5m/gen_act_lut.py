import argparse
import json

import torch


def parse_core_pos(core_name):
    if core_name == "HOSTI":
        core_name = "Core0_4"
    y, x = core_name.replace("Core", "").split("_")
    return int(y), int(x)


def get_direction(src_core, dst_core):
    src_y, src_x = parse_core_pos(src_core)
    dst_y, dst_x = parse_core_pos(dst_core)
    if dst_x > src_x:
        return "E"
    if dst_x < src_x:
        return "W"
    if dst_y < src_y:
        return "N"
    if dst_y > src_y:
        return "S"
    return None


def is_valid_silu_task(task_name):
    return (
        "Silu" in task_name
        and "mul_add" not in task_name
        and "dmem" not in task_name
        and "identity" not in task_name
    )


def ensure_lut(core_buf, direction, task_name, act_lut):
    if direction not in core_buf:
        core_buf[direction] = {"Task_Name": []}
    if task_name not in core_buf[direction]["Task_Name"]:
        core_buf[direction]["Task_Name"].append(task_name)
    if "LUT" not in core_buf[direction]:
        core_buf[direction]["LUT"] = [int(v) for v in act_lut[task_name]]


def collect_act_info(sys_cfg, act_lut, verbose=False):
    act_info = {}
    for core, core_info in sys_cfg.items():
        if core in {"HOSTI", "Run_Time"}:
            continue

        core_buf = {}
        for _, t_info in core_info.items():
            task_name = t_info.get("Task_Name", "")
            if not is_valid_silu_task(task_name):
                continue

            if verbose:
                print(task_name)
            dst_seg = t_info.get("Dst", {}).get("seg_0", {})
            if "core" in dst_seg:
                direction = get_direction(core, dst_seg["core"])
                if direction is None:
                    raise ValueError(f"Unknown Dst_Core in Task: {task_name}")
                ensure_lut(core_buf, direction, task_name, act_lut)
            else:
                mcast_items = [(k, v) for k, v in dst_seg.items() if k.startswith("mcast_")]
                if not mcast_items:
                    continue
                if verbose:
                    print(f"Task {task_name} need multicast")
                for _, mcast_info in mcast_items:
                    direction = get_direction(core, mcast_info["core"])
                    if direction is None:
                        raise ValueError(f"Unknown Dst_Core in Task: {mcast_info['core']}")
                    ensure_lut(core_buf, direction, task_name, act_lut)

        if core_buf:
            act_info[core] = core_buf
    return act_info


def run_act_lut(
    model_name="yolov5m_wo_head",
    systemc_cfg=None,
    act_lut_path="algo\\activation_lut.pth",
    output_file=None,
    verbose=False,
):
    print("[STEP] Act LUT compile started.")
    if systemc_cfg is None:
        systemc_cfg = f"uvm\\{model_name}_systemc.json"
    if output_file is None:
        output_file = f"chip\\{model_name}_act_lut.json"

    with open(systemc_cfg, "r") as f:
        sys_cfg = json.load(f)
    print("[STEP] Act LUT: system config loaded.")
    act_lut = torch.load(act_lut_path)
    print("[STEP] Act LUT: activation LUT loaded.")

    act_info = collect_act_info(sys_cfg, act_lut, verbose=verbose)
    print("[STEP] Act LUT: routing collected.")
    with open(output_file, "w") as f:
        json.dump(act_info, f, indent=4)
    print("[STEP] Act LUT: output json generated.")
    print("[STEP] Act LUT compile finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate activation LUT config for chip deployment.")
    parser.add_argument("--model-name", default="yolov5m_wo_head")
    parser.add_argument("--systemc-cfg", default=None)
    parser.add_argument("--act-lut-path", default="algo\\activation_lut.pth")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_act_lut(
        model_name=args.model_name,
        systemc_cfg=args.systemc_cfg,
        act_lut_path=args.act_lut_path,
        output_file=args.output_file,
        verbose=args.verbose,
    )
