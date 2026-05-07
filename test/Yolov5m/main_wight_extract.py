import sys

from main_weight_extract import parse_args, run_weight_extract


if __name__ == "__main__":
    args = parse_args()
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
