import pytest
import importlib


def get_module_by_name(module_name: str):
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None

import torch
assert get_module_by_name("numpy") is None, "numpy module should not be available for pytorch_only tests"

import pccl
from pccl import MasterNode, Communicator

HOST = "127.0.0.1:28148"


def test_pytorch_only():
    print(f"Starting master node on {HOST}")
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()

    communicator = Communicator(HOST, 0)
    communicator.connect()

    arr = torch.rand(27)
    result = communicator.all_reduce_async(arr, arr, op=pccl.ReduceOp.SUM)
    success, info, status = result.wait()
    assert not success, "Expected all reduce to fail, as there is no second peer."
    assert status is not None
    assert info is not None

    master.interrupt()
