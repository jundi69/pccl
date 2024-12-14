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
