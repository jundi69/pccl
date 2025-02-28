import os
import subprocess
import pccl
import sys
import time
import unittest
from typing import List, Optional, Dict

import pytest


def launch_py_process(
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    """
    Launches a Python process with optional environment variables and stdout forwarding.

    :param script_path: Path to the Python script to execute.
    :param args: List of arguments to pass to the script.
    :param env_vars: Dictionary of environment variables to set for the process.
    :return: A Popen object for the launched process.
    """
    # Merge the current environment with the new variables (if any)
    env = {**dict(os.environ), **(env_vars or {})}

    cmd = [sys.executable, script_path] + args
    print(f"Launching process: {cmd}")
    return subprocess.Popen(
        cmd,
        env=env
    )

@pytest.mark.parametrize("use_cuda", [False, True] if pccl.cuda.is_available() else [False])
def test_mnist_ddp_world_size_2(use_cuda: bool):
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 2 peers
    process_list = []
    for rank in range(2):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()


@pytest.mark.parametrize("use_cuda", [False, True] if pccl.cuda.is_available() else [False])
def test_mnist_ddp_world_size_3(use_cuda: bool):
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 3 peers
    process_list = []
    for rank in range(3):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()


@pytest.mark.parametrize("use_cuda", [False, True] if pccl.cuda.is_available() else [False])
def test_mnist_ddp_world_size_2_plus_1_late_joiner(use_cuda: bool):
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 2 peers
    process_list = []
    for rank in range(2):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'DONT_EXIT_BEFORE_REACHED_WORLD_SIZE': "3", 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    time.sleep(0.1)

    # if the other two peers are still running, add the third peer; otherwise, fail the test
    if all(p.poll() is None for p in process_list):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': '2', 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer 2; PID: {process_list[-1].pid}")
    else:
        assert False, "One of the peers exited prematurely"

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()

@unittest.skip("Enable test when distance-aware hashes are implemented")
@pytest.mark.skipif(not pccl.cuda.is_available(), reason="PCCL compiled without cuda support, skipping test")
def test_mnist_train_peers_different_devices():
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 2 peers
    process_list = []
    for rank in range(2):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'MNIST_USE_CUDA': "1" if rank == 0 else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()


@pytest.mark.parametrize("use_cuda", [False, True] if pccl.cuda.is_available() else [False])
@unittest.skip("Skipping test_mnist_ddp_world_size_4")
def test_mnist_ddp_world_size_4(use_cuda: bool):
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 4 peers
    process_list = []
    for rank in range(4):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()

@pytest.mark.parametrize("use_cuda", [False, True] if pccl.cuda.is_available() else [False])
@unittest.skip("Skipping test_mnist_ddp_world_size_16")
def test_mnist_ddp_world_size_16(use_cuda: bool):
    peer_script_path = os.path.join(os.path.dirname(__file__), 'mnist_peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'mnist_master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'INFO'})
    print(f"Launched master node; PID: {master_process.pid}")

    # launch 16 peers
    process_list = []
    for rank in range(16):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'INFO', 'RANK': str(rank), 'MNIST_USE_CUDA': "1" if use_cuda else "0"}))
        print(f"Launched peer {rank}; PID: {process_list[-1].pid}")

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()
