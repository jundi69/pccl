from time import sleep

from pccl import *

def test_reduce():
    HOST: str = '127.0.0.1:48148'
    STEPS: int = 100
    WEIGHT_N: int = 1024
    PEERS: int = 1

    # Create a weight tensor
    weights: torch.Tensor = torch.rand(WEIGHT_N, dtype=torch.float32)

    # Create a master node
    master: MasterNode = MasterNode(listen_address=HOST)
    master.run()

    # Create shared state with tensor infos
    shared_state: SharedState = SharedState([
        TensorInfo.from_torch(weights, 'weights')
    ])

    # Create a communicator and connect to the master node
    communicator: Communicator = Communicator(HOST, 0)
    communicator.connect()

    while True:
        communicator.update_topology()
        communicator.sync_shared_state(shared_state)
        world_size: int = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
        if world_size != 0 and world_size < 2:
            sleep(1)
            continue
        if shared_state.revision == STEPS:
            break

        # Create gradients tensors
        grad: torch.Tensor = torch.rand(PEERS, dtype=torch.float32)
        while True:
            handle = communicator.all_reduce_async(grad, weights, numel=PEERS, op=ReduceOp.SUM)
            success, info = handle.wait()
            if success:
                print(f"Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}")
                break

        shared_state.revision += 1

    master.interrupt()