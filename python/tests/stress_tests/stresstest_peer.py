from time import sleep
import os
import logging
import torch
from pccl import SharedState, TensorInfo, Communicator, Attribute, ReduceOp, QuantizationOptions, DataType, \
    QuantizationAlgorithm, PCCLError, ReduceOperandDescriptor, DistributionHint, AsyncReduceHandle
from typing_extensions import Optional, List

HOST: str = '127.0.0.1:48148'
STEPS: int = 1000
PEERS: int = 1
NUM_ELEMENTS: int = 1024 * 1024
RANK: int = int(os.getenv('RANK', "0"))

logging.basicConfig(level=logging.INFO)


def all_reduce_multiple_with_retry(communicator: Communicator, tensors: List[torch.Tensor], op: ReduceOp, it: int = 0):
    world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        quant_desc = QuantizationOptions(
            quantized_datatype=DataType.UINT8,
            algorithm=QuantizationAlgorithm.MIN_MAX
        )
        return communicator.all_reduce_async(x, x, operand_descriptor=op_desc,
                                             quantization_options=quant_desc,
                                             op=op, tag=tag)

    handles: List[Optional[AsyncReduceHandle]] = [None for _ in range(len(tensors))]
    done_handles = set()

    for tensor_index in range(len(tensors)):
        handles[tensor_index] = launch_all_reduce(tensors[tensor_index], tensor_index)

    while world_size > 1:
        all_done = True
        for tensor_index in range(len(tensors)):
            handle = handles[tensor_index]
            if handle is None:
                if tensor_index in done_handles:
                    continue
                else:
                    handle = handles[tensor_index] = launch_all_reduce(tensors[tensor_index], tensor_index)

            print(f"(RANK={RANK}, it={it}) all_reduce_async wait({tensor_index})")
            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.CURRENT_WORLD_SIZE)
            if not is_success:
                print(f"(RANK={RANK}, it={it}) all_reduce_async({tensor_index}) failed; New world_size: {world_size}")
                print(f"(RANK={RANK}, it={it}) waiting for all async operations to complete before retrying...")
                for j in range(len(tensors)):
                    if j == tensor_index:
                        continue
                    handle = handles[j]
                    if handle is not None:
                        is_success, status, info = handle.wait()
                        if is_success:
                            done_handles.add(j)
                        handles[tensor_index] = None
                print(f"(RANK={RANK}, it={it}) all async operations awaited; retrying...")
                handles[tensor_index] = None
                all_done = False
                break
            assert info is not None
            print(
                f"(RANK={RANK}, it={it}) Reduce completed RX: {info.rx_bytes}, TX: {info.tx_bytes}; world_size: {info.world_size}")
            handles[tensor_index] = None
            done_handles.add(tensor_index)

        if all_done:
            break

    if world_size == 1:
        # await all handles
        for handle in handles:
            if handle is not None:
                handle.wait()
        return False

    return True

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
                        # The peer was likely kicked; this can happen be collateral damage in rare cases when lots of
                        # people die at the same time, p2p connections cannot be established and the master thinks
                        # the peer cannot communicate with enough peers to build a tour. This is unresolvable because
                        # it might just be true that the peer can indeed just not talk to those peers.
                        # If they had disconnected from the master too, they would already be considered removed from the
                        # run. But if that happens just a tick late, this unlucky situation can happen.
                        print("Exiting because peer was likely kicked. This doesn't necessary entail a failure of the stress test, as long as the shared state revision is not lost and new peers continue on.")
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

        # Create fake gradients
        gradients = [torch.randn(128, 128, dtype=torch.float32) for _ in range(10)]
        all_reduce_multiple_with_retry(communicator, gradients, ReduceOp.SUM, it)
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
