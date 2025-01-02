import os
import subprocess
import sys
import time
from typing import List, Optional, Dict, Union, IO


def launch_py_process(
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None,
        forward_stdout: bool = False
) -> subprocess.Popen:
    """
    Launches a Python process with optional environment variables and stdout forwarding.

    :param script_path: Path to the Python script to execute.
    :param args: List of arguments to pass to the script.
    :param env_vars: Dictionary of environment variables to set for the process.
    :param forward_stdout: If True, forwards stdout to the parent process's stdout.
    :return: A Popen object for the launched process.
    """
    # Merge the current environment with the new variables (if any)
    env = {**dict(os.environ), **(env_vars or {})}

    # Set stdout based on the forward_stdout flag
    stdout: Union[IO, None] = subprocess.PIPE if forward_stdout else None
    stderr: Union[IO, None] = subprocess.PIPE if forward_stdout else None

    return subprocess.Popen(
        [sys.executable, script_path] + args,
        env=env,
        stdout=stdout,
        stderr=stderr
    )


def debug():
    print("debug")

    if os.name == 'nt':
        print("Windows")

        p = subprocess.run(["netstat", "-ano"])
        print(p.stdout)
    elif os.name == 'posix':
        print("Linux")
        p = subprocess.run(["lsof", "-i"])
        print(p.stdout)

    # print own pid
    print("Own PID: ", os.getpid())


def test_basic_reduce():
    peer_script_path = os.path.join(os.path.dirname(__file__), 'peer.py')
    master_script_path = os.path.join(os.path.dirname(__file__), 'master.py')

    # launch master node
    master_process = launch_py_process(master_script_path, [], {'PCCL_LOG_LEVEL': 'DEBUG'}, forward_stdout=True)

    # wait for master node to start
    time.sleep(10)

    debug()

    # launch 2 peers
    process_list = []
    for rank in range(2):
        process_list.append(launch_py_process(peer_script_path, [], {'PCCL_LOG_LEVEL': 'DEBUG', 'RANK': str(rank)},
                                              forward_stdout=True))

    # wait for all processes to finish
    for process in process_list:
        exit_code = process.wait()
        assert exit_code == 0, "Peer process exited with non-zero exit code"

    # kill master process
    master_process.kill()
    master_process.wait()


if __name__ == "__main__":
    test_basic_reduce()
