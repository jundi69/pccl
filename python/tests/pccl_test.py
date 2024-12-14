from time import sleep

import torch

def test_import_lib():
    import pccl

from pccl import *

def test_master_node_run():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()
    m.interrupt()

def test_communicator():
    m = MasterNode(listen_address='127.0.0.1:48148')
    m.run()
    c = Communicator()
    c.connect('127.0.0.1:48148')
    m.interrupt()

def test_simple_reduce():
    HOST: str = '127.0.0.1:48148'
    STEPS: int = 100
    WEIGHT_N: int = 1024
    PEERS: int = 1

    m = MasterNode(listen_address=HOST)
    m.run()

    c = Communicator()
    c.connect(HOST)

    # Create a tensor
    weights = torch.rand(WEIGHT_N, dtype=torch.float32)

    while True:
        c.update_topology()
        c.sync_shared_state()
        world_size: int = c.get_attribute(Attribute.CURRENT_WORLD_SIZE)
        if world_size < 2:
            sleep(1)
            continue



    m.interrupt()
