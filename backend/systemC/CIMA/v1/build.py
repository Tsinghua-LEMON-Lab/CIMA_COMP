import subprocess
import os

def info_error_check(f):
    info = f.stdout.read().decode("utf-8", "ignore")
    print(info)
    returncode = f.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, f)
    f.stdout.close()

def build_systemC(build_path,systemc_path,log_info=True):
    print('Message translated to English.')
    cmd = f"cmake -H{systemc_path} -B{build_path} -G Ninja"
    print(cmd)
    f = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_info:
        while f.poll() is None:
            line = f.stdout.readline()
            line = line.strip()
            if line:
                print(line)
    info_error_check(f)
    print('Message translated to English.')

def compile(build_path,log_info=True):
    print('Message translated to English.')
    cur_path = os.getcwd()
    os.chdir(f'{build_path}')
    cmd = f"ninja"
    f = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_info:
        while f.poll() is None:
            line = f.stdout.readline()
            line = line.strip()
            if line:
                print(line)
    info_error_check(f)
    os.chdir(f'{cur_path}')
    print('Message translated to English.')

def run(build_path,log_info=True):
    print('Message translated to English.')
    cur_path = os.getcwd()
    os.chdir(build_path)
    cmd = f"e100.exe"
    f = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log_info:
        while f.poll() is None:
            line = f.stdout.readline()
            line = line.strip()
            if line:
                print(line)
    info_error_check(f)
    os.chdir(cur_path)
    print('Message translated to English.')

def copy_file(src_file,target_path):
    cmd = fr'copy {src_file} {target_path}'
    os.system(cmd)
    print(cmd)
