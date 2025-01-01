import os
import subprocess
import sys
import time
import socket
from typing import List

def launch_py_process(script_path: str, args: List[str]) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, script_path] + args)

def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # Set a timeout to avoid long waits
        try:
            s.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False

def test_basic_reduce():
    peer_script_path = os.path.join(os.path.dirname(__file__), 'peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [])

    # wait for something to bind to master port
    while not is_port_open('127.0.0.1', 48148):
        time.sleep(0.1)

    # launch 2 peers
    process_list = []
    for _ in range(2):
        process_list.append(launch_py_process(peer_script_path, []))

    # wait for all processes to finish
    for process in process_list:
        process.wait()

    # kill master process
    master_process.kill()