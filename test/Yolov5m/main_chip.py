import sys
import argparse
from pathlib import Path

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.CIMA.chip.gen_code import ChipCodeGen
# from e100_irbackend.systemC.CIMA.v2.gen_code import CodeGen
# from e100_irtool.core import load_ir
from mapper.device.CIMA import *  # noqa
from mapper.self_defined_op import *  # noqa

def run_chip(
    systemc_json="uvm\\yolov5m_wo_head_systemc.json",
    chip_json="chip\\yolov5m_wo_head_chip.json",
    module_name="yolov5",
):
    print("[STEP] Chip compile started.")
    code_chip = ChipCodeGen(systemc_json, module_name=module_name)
    print("[STEP] Chip: generator initialized.")
    code_chip.gen_layers(chip_json_file=chip_json)
    print("[STEP] Chip: chip json generated.")
    print("[STEP] Chip compile finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate chip config from SystemC JSON.")
    parser.add_argument("--model-name", default="yolov5m_wo_head")
    parser.add_argument("--systemc-json", default="uvm\\yolov5m_wo_head_systemc.json")
    parser.add_argument("--chip-json", default="chip\\yolov5m_wo_head_chip.json")
    parser.add_argument("--module-name", default="yolov5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    systemc_json = args.systemc_json
    chip_json = args.chip_json
    if "--systemc-json" not in sys.argv:
        systemc_json = f"uvm\\{args.model_name}_systemc.json"
    if "--chip-json" not in sys.argv:
        chip_json = f"chip\\{args.model_name}_chip.json"
    run_chip(
        systemc_json=systemc_json,
        chip_json=chip_json,
        module_name=args.module_name,
    )

