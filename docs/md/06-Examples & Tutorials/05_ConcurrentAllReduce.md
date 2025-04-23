# Concurrent All-Reduce with Retry

To safely recover from failures when using multiple concurrent collective operations, it is recommended to use the PCCL-provided `pcclAllReduceMultipleWithRetry` function.
This function will correctly retry all failed in-flight operations if a peer drops gracefully or ungracefully.

```python
import torch
from pccl import Communicator, ReduceOp, Attribute, ReduceOperandDescriptor, DataType, DistributionHint, QuantizationOptions, QuantizationAlgorithm, ReduceDescriptor, ReduceOpDescriptor, PCCLError

def all_reduce_multiple_with_retry(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 16):
    """
    Launches concurrent all-reduce operations on a list of tensors,
    waits for them all, and retries if a peer fails or the world size changes.
    Will attempt to target :param max_in_flight: concurrent all-reduce operations.
    The more similar your tensors are in size, the better this in flight system will work.
    """
    descriptors = []
    tag = 0
    for tensor in tensors:
        reduce_op_descriptor = ReduceOpDescriptor.from_torch(
            send=tensor,
            recv=tensor,
            reduce_descriptor=ReduceDescriptor(
                count=tensor.numel(),
                op=op,
                tag=tag,
                operand_descriptor=ReduceOperandDescriptor(
                    datatype=DataType.FLOAT,
                    distribution_hint=DistributionHint.NORMAL
                ),
                quantization_options=QuantizationOptions(
                    quantized_datatype=DataType.FLOAT,
                    algorithm=QuantizationAlgorithm.NONE
                )
            )
        )
        descriptors.append(reduce_op_descriptor)
        tag += 1
    try:
        info = communicator.all_reduce_multiple_with_retry(descriptors, max_in_flight=max_in_flight)
        return True, info.tx_bytes, info.rx_bytes
    except PCCLError:
        return False, 0, 0
```

## Equivalent Manual Implementation (Not Recommended)

It is however also possible to implement a similar function manually, using the normal `allReduceAsync` and `pcclAwaitAsyncReduce` primitives.
This is not recommended for production code, as it is more error-prone and less efficient than using the provided `pcclAllReduceMultipleWithRetry` function.
This manual implementation is only intended for educational purposes to illustrate how the retry mechanism works.

```python
def all_reduce_multiple_with_retry__equivalent(communicator: Communicator,
                                   tensors: list[torch.Tensor],
                                   op: ReduceOp,
                                   max_in_flight: int = 16):
    """
    The following function is equivalent to the above, but uses a more manual approach. (Not recommended for production)
    """
    world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)

    total_tx = 0
    total_rx = 0

    def launch_all_reduce(x: torch.Tensor, tag: int):
        op_desc = ReduceOperandDescriptor(
            datatype=DataType.FLOAT,
            distribution_hint=DistributionHint.NORMAL
        )
        quant_desc = QuantizationOptions(
            quantized_datatype=DataType.FLOAT,
            algorithm=QuantizationAlgorithm.NONE
        )
        return communicator.all_reduce_async(
            x, x,
            operand_descriptor=op_desc,
            quantization_options=quant_desc,
            op=op,
            tag=tag
        )

    handles = [None for _ in range(len(tensors))]
    done_handles = set()

    in_flight = 0
    for tensor_index in range(len(tensors)):
        dst_tensor = tensors[tensor_index]

        if in_flight >= max_in_flight:
            break

        handles[tensor_index] = launch_all_reduce(
            dst_tensor,
            tensor_index
        )
        in_flight += 1

    while world_size > 1:
        all_done = True
        for tensor_index in range(len(tensors)):
            handle = handles[tensor_index]
            dst_tensor = tensors[tensor_index]

            if handle is None:
                if tensor_index in done_handles:
                    continue

                if in_flight >= max_in_flight:
                    continue

                handle = handles[tensor_index] = launch_all_reduce(
                    dst_tensor,
                    tensor_index
                )
                in_flight += 1

            is_success, status, info = handle.wait()
            world_size = communicator.get_attribute(Attribute.GLOBAL_WORLD_SIZE)
            if not is_success:
                print(f"Reduce failed: {status}; Starting recovery procedure")
                handles[tensor_index] = None
                # Wait for all ongoing ops to finish or fail before retry
                for j in range(len(tensors)):
                    if j == tensor_index:
                        continue
                    h_j = handles[j]
                    if h_j is not None:
                        s_j, _, _ = h_j.wait()
                        if s_j:
                            done_handles.add(j)
                        in_flight -= 1
                    handles[j] = None
                all_done = False
                break

            # success for this handle
            handles[tensor_index] = None
            done_handles.add(tensor_index)

            total_tx += info.tx_bytes
            total_rx += info.rx_bytes

            in_flight -= 1

        if all_done:
            print(
                f"Reduce completed RX: {total_rx}, TX: {total_tx}; world_size: {world_size}")
            break

    if world_size == 1:
        print(f"All peers have left except this peer. All reduce will do nothing.")
        # If we are alone, just finalize all handles and return
        for h in handles:
            if h is not None:
                h.wait()
        return False

    return True
```