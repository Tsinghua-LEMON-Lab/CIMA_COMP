from CIMA.uvm.helper import register_json2sv

if __name__ == "__main__":
    #
    path = 'fashionmnist\\'
    reg_json_file = path + f'register_raw_data.json'
    sv_file = path + f'test_sv.txt'
    #
    register_json2sv(json_file=reg_json_file, sv_file=sv_file)


