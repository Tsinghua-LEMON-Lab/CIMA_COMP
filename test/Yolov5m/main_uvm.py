import argparse
import os
import sys
from pathlib import Path

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.CIMA.uvm.draw_graph import draw_ir_tree
from backend.CIMA.uvm.gen_code import UVMCodeGen
from backend.systemC.CIMA.v2.gen_code import CodeGen
from irtool.core import load_ir
from mapper.device.CIMA import *  # noqa
from mapper.self_defined_op import *  # noqa


def run_uvm(
    model_name="yolov5m_wo_head",
    module_name="yolov5",
    output_dir="uvm\\",
    run_time=10 * 10**6,
    mesh_width=9,
    draw_graph_pdf=True,
):
    print("[STEP] UVM compile started.")
    os.makedirs(output_dir, exist_ok=True)
    ir_file = f"ir\\{model_name}_dmem_opt_mapped_ir_w_params.yaml"
    ir = load_ir(file=ir_file)
    print("[STEP] UVM: IR loaded.")

    code_systemc = CodeGen(ir)
    systemc_json = output_dir + f"{model_name}_systemc.json"
    code_systemc.run(output_file=systemc_json, run_time=run_time)
    print("[STEP] UVM: SystemC config generated.")

    code_uvm = UVMCodeGen(systemc_json, module_name=module_name)
    code_uvm.to_code(
        generator=code_uvm.gen_layers(),
        file=output_dir + f"{model_name}_with_io_core_sch_tx.txt",
    )
    print("[STEP] UVM: UVM schedule generated.")
    code_uvm.gen_register_raw_data(
        raw_config_file=output_dir + f"{model_name}_raw_data_with_io_core_sch_tx.json"
    )
    print("[STEP] UVM: Register raw data generated.")

    if draw_graph_pdf:
        layer_hardware_thread = code_uvm.all_layer_hardware_index
        draw_ir_tree(
            ir_file,
            layer_thread_info=layer_hardware_thread,
            MESH_width=mesh_width,
            save_pdf=output_dir + f"{model_name}_graph.pdf",
        )
        print("[STEP] UVM: Graph PDF generated.")
    else:
        print("[STEP] UVM: Graph PDF skipped.")
    print("[STEP] UVM compile finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate SystemC/UVM outputs from mapped IR.")
    parser.add_argument("--model-name", default="yolov5m_wo_head")
    parser.add_argument("--module-name", default="yolov5")
    parser.add_argument("--output-dir", default="uvm\\")
    parser.add_argument("--run-time", type=int, default=10 * 10**6)
    parser.add_argument("--mesh-width", type=int, default=9)
    parser.add_argument("--skip-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_uvm(
        model_name=args.model_name,
        module_name=args.module_name,
        output_dir=args.output_dir,
        run_time=args.run_time,
        mesh_width=args.mesh_width,
        draw_graph_pdf=not args.skip_graph,
    )
