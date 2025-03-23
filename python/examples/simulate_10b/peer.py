from time import sleep
import time
import os
import logging
import torch
from pccl import Communicator, Attribute, ReduceOp, QuantizationOptions, DataType, \
    QuantizationAlgorithm

from python.examples.simulate_10b.profiler import Profiler

HOST: str = '10.1.2.92:48148'
STEPS: int = 1_000_000
PEERS: int = 1
NUM_ELEMENTS: int = 335544320
NODE: int = int(os.getenv('NODE', "0"))
GPU: int = int(os.getenv('GPU', "0"))

logging.basicConfig(level=logging.INFO)


def main():
    logging.info(f"(NODE={NODE}) Starting peer node connecting to {HOST}")

    # Create a weight tensor
    weights: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)

    # Create a communicator and connect to the master node
    communicator: Communicator = Communicator(HOST, peer_group=GPU)
    communicator.connect(n_attempts=15)
    logging.info(f"(NODE={NODE}) Connected to the master node")

    n_performed_steps = 0
    while n_performed_steps < STEPS:
        profiler = Profiler()
        with profiler.session("step"):
            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
            if world_size > 1:
                with profiler.session("pccl::communicator::optimize_topology"):
                    communicator.optimize_topology()
            if n_performed_steps > 0 or world_size == 1:
                logging.info(f"(NODE={NODE}, it={n_performed_steps}) update_topology()")
                with profiler.session("pccl::communicator::update_topology"):
                    communicator.update_topology()

            if world_size < 2:
                sleep(1)
                continue

            # Create gradients tensors
            grad: torch.Tensor = torch.rand(NUM_ELEMENTS, dtype=torch.float32)
            with profiler.session("pccl::communicator::all_reduce"):
                while True:
                    logging.info(f"(NODE={NODE}, it={n_performed_steps}) all_reduce_async()")
                    start = time.time()
                    handle = communicator.all_reduce_async(grad, weights,
                                                           op=ReduceOp.SUM,
                                                           quantization_options=QuantizationOptions(DataType.UINT8,
                                                                                                    QuantizationAlgorithm.MIN_MAX))

                    is_success, status, info = handle.wait()
                    end = time.time()
                    assert is_success, f"All reduce failed with stats: {status}"
                    assert info is not None

                    bandwidth_mbps = ((info.rx_bytes + info.tx_bytes) * 8 / 1e6) / (end - start)
                    logging.info(
                        f"(NODE={NODE}; GPU={GPU}, it={n_performed_steps}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes} in {end - start:.2f}s; Bandwidth: {bandwidth_mbps:.2f} mbit/s")
                    break
            n_performed_steps += 1
        profiler.print_report()

    logging.info(f"(NODE={NODE}) Finished")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"(NODE={NODE}) failed with: {e}")
        raise e
