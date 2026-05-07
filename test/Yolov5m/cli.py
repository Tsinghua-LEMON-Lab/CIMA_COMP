import argparse
import sys
from pathlib import Path

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main_chip import run_chip
from main_mapper import run_mapper
from main_uvm import run_uvm
from main_weight_extract import run_weight_extract
from gen_act_lut import run_act_lut


def build_parser():
    parser = argparse.ArgumentParser(description="Unified CLI for Yolov5m pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mapper_parser = subparsers.add_parser("mapper", help="Run mapper and generate IR files.")
    mapper_parser.add_argument("--model-name", default="yolov5m_wo_head")

    uvm_parser = subparsers.add_parser("uvm", help="Generate SystemC/UVM files.")
    uvm_parser.add_argument("--model-name", default="yolov5m_wo_head")
    uvm_parser.add_argument("--module-name", default="yolov5")
    uvm_parser.add_argument("--output-dir", default="uvm\\")
    uvm_parser.add_argument("--run-time", type=int, default=10 * 10**6)
    uvm_parser.add_argument("--mesh-width", type=int, default=9)
    uvm_parser.add_argument("--skip-graph", action="store_true")

    weight_parser = subparsers.add_parser("weight", help="Generate chip weight JSON.")
    weight_parser.add_argument("--model-name", default="yolov5m_wo_head")
    weight_parser.add_argument("--weight-chip-file", default="algo\\weight_int_chip.pth")
    weight_parser.add_argument("--systemc-json-file", default="uvm\\yolov5m_wo_head_systemc.json")
    weight_parser.add_argument("--ir-file", default="ir\\yolov5m_wo_head_dmem_opt_mapped_ir_w_params.yaml")
    weight_parser.add_argument("--output-file", default="chip\\yolov5m_wo_head_weight.json")

    chip_parser = subparsers.add_parser("chip", help="Generate chip task JSON.")
    chip_parser.add_argument("--model-name", default="yolov5m_wo_head")
    chip_parser.add_argument("--systemc-json", default="uvm\\yolov5m_wo_head_systemc.json")
    chip_parser.add_argument("--chip-json", default="chip\\yolov5m_wo_head_chip.json")
    chip_parser.add_argument("--module-name", default="yolov5")

    act_parser = subparsers.add_parser("act", help="Generate activation LUT JSON for chip.")
    act_parser.add_argument("--model-name", default="yolov5m_wo_head")
    act_parser.add_argument("--systemc-cfg", default=None)
    act_parser.add_argument("--act-lut-path", default="algo\\activation_lut.pth")
    act_parser.add_argument("--output-file", default=None)
    act_parser.add_argument("--verbose", action="store_true")

    all_parser = subparsers.add_parser("all", help="Run mapper -> uvm -> chip -> weight -> act.")
    all_parser.add_argument("--model-name", default="yolov5m_wo_head")
    all_parser.add_argument("--module-name", default="yolov5")
    all_parser.add_argument("--output-dir", default="uvm\\")
    all_parser.add_argument("--run-time", type=int, default=10 * 10**6)
    all_parser.add_argument("--mesh-width", type=int, default=9)
    all_parser.add_argument("--skip-graph", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "mapper":
        run_mapper(model_name=args.model_name)
        return

    if args.command == "uvm":
        run_uvm(
            model_name=args.model_name,
            module_name=args.module_name,
            output_dir=args.output_dir,
            run_time=args.run_time,
            mesh_width=args.mesh_width,
            draw_graph_pdf=not args.skip_graph,
        )
        return

    if args.command == "weight":
        systemc_json_file = args.systemc_json_file
        ir_file = args.ir_file
        output_file = args.output_file
        if "--systemc-json-file" not in sys.argv:
            systemc_json_file = f"uvm\\{args.model_name}_systemc.json"
        if "--ir-file" not in sys.argv:
            ir_file = f"ir\\{args.model_name}_dmem_opt_mapped_ir_w_params.yaml"
        if "--output-file" not in sys.argv:
            output_file = f"chip\\{args.model_name}_weight.json"
        run_weight_extract(
            weight_chip_file=args.weight_chip_file,
            systemc_json_file=systemc_json_file,
            ir_file=ir_file,
            output_file=output_file,
        )
        return

    if args.command == "chip":
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
        return

    if args.command == "act":
        run_act_lut(
            model_name=args.model_name,
            systemc_cfg=args.systemc_cfg,
            act_lut_path=args.act_lut_path,
            output_file=args.output_file,
            verbose=args.verbose,
        )
        return

    if args.command == "all":
        model_name = args.model_name
        output_dir = args.output_dir
        systemc_json = output_dir + f"{model_name}_systemc.json"
        ir_file = f"ir\\{model_name}_dmem_opt_mapped_ir_w_params.yaml"
        chip_weight_json = f"chip\\{model_name}_weight.json"
        chip_act_lut_json = f"chip\\{model_name}_act_lut.json"
        chip_json = f"chip\\{model_name}_chip.json"

        run_mapper(model_name=model_name)
        run_uvm(
            model_name=model_name,
            module_name=args.module_name,
            output_dir=output_dir,
            run_time=args.run_time,
            mesh_width=args.mesh_width,
            draw_graph_pdf=not args.skip_graph,
        )
        run_chip(
            systemc_json=systemc_json,
            chip_json=chip_json,
            module_name=args.module_name,
        )
        run_weight_extract(
            weight_chip_file="algo\\weight_int_chip.pth",
            systemc_json_file=systemc_json,
            ir_file=ir_file,
            output_file=chip_weight_json,
        )
        run_act_lut(
            model_name=model_name,
            systemc_cfg=systemc_json,
            act_lut_path="algo\\activation_lut.pth",
            output_file=chip_act_lut_json,
            verbose=False,
        )
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
