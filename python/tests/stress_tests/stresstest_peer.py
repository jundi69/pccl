from time import sleep
import os
import logging
import torch
from pccl import SharedState, TensorInfo, Communicator, Attribute, ReduceOp, QuantizationOptions, DataType, \
    QuantizationAlgorithm, PCCLError

HOST: str = '127.0.0.1:48148'
STEPS: int = 1000
PEERS: int = 1
NUM_ELEMENTS: int = 1024 * 1024
RANK: int = int(os.getenv('RANK', "0"))

logging.basicConfig(level=logging.INFO)


def main():
    logging.info(f"(RANK={RANK}) Starting peer node connecting to {HOST}")

    # Create a weight tensor
    weights: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)

    # Create shared state with tensor infos
    shared_state: SharedState = SharedState([
        TensorInfo.from_torch(weights, 'weights')
    ])

    # Create a communicator and connect to the master node
    communicator: Communicator = Communicator(HOST, 0)
    communicator.connect(n_attempts=15)
    logging.info(f"(RANK={RANK}) Connected to the master node")

    n_performed_steps = 0
    it = 0
    world_size: int = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
    while True:
        # do step
        if it > 0:
            for retry in range(10):
                try:
                    logging.info(f"(RANK={RANK}, it={it}) update_topology()")
                    communicator.update_topology()
                    break
                except PCCLError as ex:
                    if retry == 10:
                        # the peer was likely kicked,
                        print("Exiting because peer was likely kicked. This doesn't necessary entail a failure of the stress test, as long as the shared state revision is not lost and new peers continue on. This may happen because of extremely unlucky killing such that no p2p connections can be s")
                        exit(0)
                    print(f"(RANK={RANK}, it={it}) update_topology() failed: {ex}; retrying...")
                    continue

            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)

        if world_size > 1:
            while True:
                try:
                    logging.info(f"(RANK={RANK}, it={it}) optimize_topology()")
                    # communicator.optimize_topology()
                    break
                except PCCLError:
                    sleep(0.1)
                    logging.info(f"(RANK={RANK}, it={it}) optimize_topology failed; retrying...")
                    continue
            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)


        if world_size < 2:
            logging.info(f"(RANK={RANK}, it={it}) waiting...")
            sleep(1)
            it += 1
            continue

        logging.info(f"(RANK={RANK}, it={it}) sync_shared_state()")

        info = communicator.sync_shared_state(shared_state)
        assert info is not None
        print(f"(RANK={RANK}, it={it}) tx_bytes={info.tx_bytes}, rx_bytes={info.rx_bytes}")

        # Create gradients tensors
        grad: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
        while world_size > 1:
            logging.info(f"(RANK={RANK}, it={it}) all_reduce_async()")
            handle = communicator.all_reduce_async(grad, weights,
                                                   op=ReduceOp.SUM,
                                                   quantization_options=QuantizationOptions(DataType.UINT8,
                                                                                            QuantizationAlgorithm.MIN_MAX))
            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
            if not is_success:
                logging.info(f"(RANK={RANK}, it={it}) all reduce failed; retrying...")
                continue
            logging.info(
                f"(RANK={RANK}, it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}")
            break
        if world_size == 1:
            # drop current step, as we are alone in the run and whatever we just computed would induce too much noise if we stepped here.
            # If one accepts the pattern that one waits until the world size is at least two, it would be erroneous to step here.
            print(
                "All peers have left except this peer. Dropping current step to avoid inducing too much variance with our local batch!")
            shared_state.revision += 1
            continue

        shared_state.revision += 1
        n_performed_steps += 1
        it += 1

    logging.info(f"(RANK={RANK}) Finished")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"(RANK={RANK}) failed with: {e}")
        raise e
