import importlib


def get_module_by_name(module_name: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


import numpy as np
assert get_module_by_name("torch") is None, "torch module should not be available for numpy_only tests"

import pccl
from pccl import MasterNode, Communicator

HOST = "127.0.0.1:48148"


def test_numpy_only():
    print(f"Starting master node on {HOST}")
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()

    communicator = Communicator(HOST, 0)
    communicator.connect()

    arr = np.random.rand(27)
    result = communicator.all_reduce_async(arr, arr, op=pccl.ReduceOp.SUM)
    success, info, status = result.wait()
    assert not success, "Expected all reduce to fail, as there is no second peer."
    assert status is not None
    assert info is not None

    master.interrupt()
