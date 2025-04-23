# Hello World Example in Python

PCCL has python bindings an can interact with both Numpy arrays and PyTorch tensors.
The following is a simple example of a complete program that uses PCCL to perform an All-Reduce operation with PyTorch tensors:

```python
import time
import pccl
import numpy as np

# Hardcoded Master IP/Port
MASTER_HOST = "127.0.0.1"
MASTER_PORT = 48148
MAX_STEPS   = 5

def main():
    # 1) Create a Communicator object
    #    The second argument is 'peer_group' = 0
    comm = pccl.Communicator(f"{MASTER_HOST}:{MASTER_PORT}", 0)

    # 2) Connect to the master (blocking with limited attempts)
    print(f"[Peer] Connecting to master at {MASTER_HOST}:{MASTER_PORT} ...")
    comm.connect(n_attempts=15)
    print("[Peer] Connected!")

    # We'll keep:
    #   - a local iteration counter `local_iter` to skip update_topology on i=0
    #   - a shared_state 'revision' to keep all peers in step lock
    local_iter = 0

    # 3) Prepare some dummy data to place in "shared state"
    #    Suppose in real usage these are model/optimizer states
    dummy_weights = np.zeros(8, dtype=np.float32)

    # We'll store this dummy_weights in a single TensorInfo
    # The pccl.SharedState object can have multiple TensorInfo if you want to sync more keys
    my_weights_info = pccl.TensorInfo.from_numpy(dummy_weights, "dummy_weights", allow_content_inequality=False)
    
    # We'll wrap that in a SharedState
    shared_state = pccl.SharedState([my_weights_info])
    shared_state.revision = 0

    # 4) Enter the training loop
    # We'll do up to MAX_STEPS. Each step => ring operation + shared-state sync
    while True:
        # A) If we are not on the very local first iteration, call update_topology
        if local_iter > 0:
            # keep retrying if it fails
            while True:
                try:
                    comm.update_topology()
                    break
                except pccl.PCCLError as e:
                    # could be pccl.UpdateTopologyFailed or other
                    print(f"[Peer] UpdateTopology failed => {e}. Retrying...")
                    time.sleep(0.1)

            # B) get the updated world size
            #    after update_topology, it’s guaranteed to be fresh
            world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)

        # C) If multiple peers => optionally optimize ring
        if world_size > 1:
            while True:
                try:
                    comm.optimize_topology()  # may raise an error if it fails
                    break
                except pccl.PCCLError as e:
                    print(f"[Peer] OptimizeTopology failed => {e}. Retrying...")
                    time.sleep(0.1)
            # D) get the updated world size
            #    after optimize_topology, it’s guaranteed to be fresh
            world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
        
        
        if world_size < 2:
            # alone => no ring-based operation => wait
            print("[Peer] alone => sleeping.")
            time.sleep(1)
            local_iter += 1
            continue
            
        # E) Perform shared state synchronization
        try:
            sync_info = comm.sync_shared_state(shared_state)
            print(f"[Peer] shared_revision now {shared_state.revision}, sync => "
                  f"tx={sync_info.tx_bytes}, rx={sync_info.rx_bytes}")
            if local_iter > 1:
                # be pedantic about shared state drifts
                assert sync_info.rx_bytes == 0, "Peer shared state drifted!"
        except pccl.PCCLError as e:
            print(f"[Peer] shared-state sync fail => {e} at revision={shared_state.revision}")
            # break out => no sense continuing
            break

        # F) Example ring operation => a small All-Reduce
        local_data = np.array([local_iter * 10 + (k+1) for k in range(4)], dtype=np.float32)
        result_data = np.zeros_like(local_data)

        # We define how we want to do the reduce
        # distribution_hint can be normal if you want to help quantization
        # quantization_options can be left as defaults if no compression is needed
        # We'll do a loop to keep retrying if a rank drops
        while True:
            try:
                info = comm.all_reduce(local_data, result_data,
                                       operand_descriptor=pccl.ReduceOperandDescriptor(
                                           datatype=pccl.DataType.FLOAT,
                                           distribution_hint=pccl.DistributionHint.NONE
                                       ),
                                       quantization_options=pccl.QuantizationOptions(
                                           quantized_datatype=pccl.DataType.FLOAT,
                                           algorithm=pccl.QuantizationAlgorithm.NONE
                                       ),
                                       op=pccl.ReduceOp.SUM)
                # success => break out
                print(f"[Peer] local_iter={local_iter}, All-Reduce => result={result_data.tolist()}, "
                      f"Tx={info.tx_bytes}, Rx={info.rx_bytes}")
                break
            except pccl.PCCLError as e:
                # Could be rank disconnect or partial ring failure
                print(f"[Peer] All-Reduce fail => {e}; Retrying...")
                # check if we are now alone
                world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
                if world_size < 2:
                    print("[Peer] All-Reduce failed and now world_size < 2 => waiting until a new peer joins")
                    # We'll just break from the reduce attempt, do next iteration
                    # That can eventually call update_topology again to let new peers in
                    break
                time.sleep(0.2)

        # If we ended up alone mid-collective, skip the rest of this iteration
        world_size = comm.get_attribute(pccl.Attribute.GLOBAL_WORLD_SIZE)
        if world_size < 2:
            local_iter += 1
            continue

        # Increment the shared revision => sync
        shared_state.revision += 1

        # Stop if we've done enough steps
        if shared_state.revision >= MAX_STEPS:
            print(f"[Peer] Reached revision {shared_state.revision} => done.\n")
            break

        local_iter += 1

if __name__ == "__main__":
    main()
```