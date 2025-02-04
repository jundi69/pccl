# PCCL API Overview

## Initialization and Finalization

### `pcclInit`

```c
pcclResult_t pcclInit(void);
```
Initializes the PCCL library. Must be called once in a process before any other PCCL functions.

- Returns `pcclSuccess` on success, or an error code (e.g., pcclSystemError) if something basic (like a socket layer) can’t be initialized.

### `pcclDestroyCommunicator`

```c
pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);
```

Destroys a communicator object previously created, blocking until all internal threads and connections for that communicator have cleanly exited.

- Typically called at the end of a training process (or whenever you no longer need the communicator).
- The library remains valid, so you could create another communicator afterwards if you wish.

PCCL does not strictly require a global “finalization” call after you destroy all communicators.


## Master-Related Functions

A separate set of functions manage the master node, which is responsible for orchestrating the run. In typical usage, you either use these calls directly in your own “master” process or run the provided `ccoip_master` executable that does it internally.


### `pcclCreateMaster`

```c
pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
                              pcclMasterInstance_t **p_master_handle_out);
```
Creates a new master node handle, binding to the specified IP and port for listening to incoming client connections (by default, port `48148`).

### `pcclRunMaster`

```c
pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance);
```
Starts the master node’s event loop non-blockingly. After this call, the master is actively listening to clients and orchestrating their states.
- Returns `pcclSuccess` or `pcclInvalidArgument` if the handle was invalid or already running.

### `pcclInterruptMaster`

```c
pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance);
```
Signals the master to shut down gracefully. Typically invoked when you want to stop the entire run.

### `pcclMasterAwaitTermination`

```c
pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance);
```
Blocks the calling thread until the master node is fully terminated (i.e., the event loop ends and no more connections remain open).

### `pcclDestroyMaster`

```c
pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance);
```
Frees resources associated with the master handle. Must only be called after a successful `pcclMasterAwaitTermination()`.

### Typical Master Lifecycle

1. `pcclCreateMaster` ⇒ obtains a handle
2. `pcclRunMaster` ⇒ starts listening + the master’s event loop, returns immediately
3. Optionally handle OS signals (e.g., ^C) to call `pcclInterruptMaster`
4. `pcclMasterAwaitTermination` ⇒ blocks until master finishes
5. `pcclDestroyMaster` ⇒ finalize

## Creating & Managing Communicators

Clients that want to join a run do the following:

### `pcclCreateCommunicator`

```c
pcclResult_t pcclCreateCommunicator(const pcclCommCreateParams_t *params,
                                    pcclComm_t **comm_out);
```
Allocates and initializes a communicator object.

- Parameters:
    - `parameters->master_address`: The `ccoip_socket_address_t` specifying the master’s IP and port.
    - `parameters->peer_group`: An integer identifying the group of peers that will do collectives among themselves (0 is a common default if you only have one group).
- Returns `pcclSuccess`, or `pcclNotInitialized` if `pcclInit` wasn’t called, or `pcclInvalidArgument` if any pointer is null.

### `pcclConnect`

```c
pcclResult_t pcclConnect(pcclComm_t *communicator);
```

Establishes a connection to the master node and blocks until the master orchestrates acceptance. This can take time if the run is busy or if existing peers have not yet reached a suitable “accept new peers” phase.

- If successful, the communicator can be used for `pcclAllReduce` and `pcclSynchronizeSharedState`.
- If it fails, you might see `pcclMasterConnectionFailed` or `pcclInvalidUsage`.

### `pcclUpdateTopology`

```c
pcclResult_t pcclUpdateTopology(pcclComm_t *communicator);
```

Triggers a “vote to accept new peers” and/or remove dropped ones. If all accepted peers in this group call `pcclUpdateTopology` around the same time, the master transitions them to `CONNECTING_TO_PEERS` (or uses existing connections for the stable ones). This may also detect that certain peers have vanished, removing them from the ring.

- If new peers are waiting to join, they become accepted after a successful consensus.
- Typically called once per iteration in a stable spot of your training loop.
- If anything fails (e.g., a peer can’t connect to new peers/newly assigned neighbors), you might get `pcclUpdateTopologyFailed` status. The recommended action is to retry the optimization. The `Hello World` example from section 2 implements this best practice.

### `pcclOptimizeTopology`

```c
pcclResult_t pcclOptimizeTopology(const pcclComm_t *communicator);
```
Enters a “topology optimization” phase, where the master node reorders the ring or other topology to improve throughput. The master may request missing bandwidth measurements from peers, which may trigger point to point bandwidth tests. This will take some time and the method will block until all work requested by the master necessary for topology optimization is complete. The master will then return an improved topology, which peers will immediately adopt by re-establishing p2p connections via a similar process as `pcclUpdateTopology`, but without admitting new peers into the run.
- `Key`: This does *not* let in brand-new peers; Use `pcclUpdateTopology` for that.
- If anything fails (e.g., a peer can’t connect to newly assigned neighbors), you might get `pcclTopologyOptimizationFailed` status. The recommended action is to retry the optimization. The `Hello World` example from section 2 implements this best practice.

## Collective Operations

Currently, PCCL provides ring-based All-Reduce. Over time, more collectives (All-Gather, Scatter, etc.) may be introduced. While currently PCCL only supports CPU buffers for collective operations, future versions may support GPU buffers as well.

### `pcclAllReduce`

```c
pcclResult_t pcclAllReduce(const void *sendbuff,
                           void *recvbuff,
                           const pcclReduceDescriptor_t *descriptor,
                           const pcclComm_t *communicator,
                           pcclReduceInfo_t *reduce_info_out);
```
A blocking All-Reduce call:

- `sendbuff`, `recvbuff`: Pointers to user-allocated memory. If they’re the same pointer, in-place reduce is done. (Note, sendbuff == recvbuff will trigger internal memory allocation.)
- `descriptor->op`: Summation, product, min, max, or average.
- `descriptor->quantization_options`:  If you want to compress data in flight.
- `reduce_info_out`: (optional) stats about how many bytes were transmitted or received.

#### Error Handling:

- `pcclSuccess`:  if the ring-based reduce completed normally.
- `pcclRankConnectionLost`: if a peer vanished mid-collective. You typically re-try the same reduce.
- `pcclInvalidUsage`: if you haven’t connected or if the communicator is in an illegal state.

### `pcclAllReduceAsync(...) / pcclAwaitAsyncReduce(...)

```c
pcclResult_t pcclAllReduceAsync(const void *sendbuff,
                                void *recvbuff,
                                const pcclReduceDescriptor_t *descriptor,
                                const pcclComm_t *communicator,
                                pcclAsyncReduceOp_t *reduce_handle_out);

pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle,
                                  pcclReduceInfo_t *reduce_info_out);
```

These functions are conceptually for launching a non-blocking All-Reduce and later waiting on it. However, due to limitations of the current ring reduce implementation, you cannot truly run multiple concurrent all-reduce ops in parallel as of now. Attempting to do so can cause data collisions.
This is expected to be fixed in a future release.
For now, the recommended usage is:

1. `AllReduceAsync`: Start the operation.
2. `AwaitAsyncReduce`: Immediately wait for it in the same iteration.


### Shared-State Synchronization

`pcclSynchronizeSharedState`

```c
pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *communicator,
                                        pcclSharedState_t *shared_state,
                                        pcclSharedStateSyncInfo_t *sync_info_out);
```
Performs a group-wide check of user-defined “shared state” arrays (e.g., model or optimizer parameters).

- `shared_state->revision`: Must increment exactly by +1 each time. If you skip a number or reuse an old revision, the master rejects it (`REVISION_INCREMENT_VIOLATION`).
- `shared_state->infos`: An array of `pcclTensorInfo_t`, each describing a key string, pointer to memory, type, size, and a flag `allow_content_inequality`.
- If any mismatch is found (by hashing each tensor), that peer is told to re-request data from a designated “correct” peer.
- Once the entire group is consistent at that revision, the function returns.
- `sync_info_out->tx_bytes` / `sync_info_out->rx_bytes`: Tells you how many bytes were actually transmitted or received in that sync (often zero if you are already identical).

Common Usage:
1. You do local training steps that produce new weights.
2. Increment shared_state->revision by 1.
3. Call pcclSynchronizeSharedState(...).
4. If all peers have the same resulting data, no actual data transfer happens (zero bytes). Otherwise, some re-transmission occurs to unify them.

## Querying Attributes

### `pcclGetAttribute`

```c
pcclResult_t pcclGetAttribute(const pcclComm_t *communicator,
                              pcclAttribute_t attribute,
                              int *p_attribute_out);
```
Retrieves specific integer-valued attributes from the communicator. Currently:
- `PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE`: The total number of “accepted” peers in the same peer group as this communicator.

Common use:

```c
int world_size;
pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);
```

`Important`: This is only up-to-date if called after the last invocation of `pcclUpdateTopology()` or `pcclConnect()` if the communicator just joined the run.


## Error Handling & Logging

### Error Returns
Most functions return a `pcclResult_t`, an enum with values like:

- `pcclSuccess` (0)
- `pcclNotInitialized`, `pcclInvalidArgument`, `pcclInvalidUsage`
- `pcclMasterConnectionFailed`, `pcclRankConnectionFailed`, `pcclRankConnectionLost`
- `pcclRemoteError`, etc.

If functions do not explicitly mention a recommended action, it is typical wise to panic or otherwise handle the error in a non-recoverable way.

## Logging
PCCL uses an internal logging mechanism that writes debug/info/error messages to stdout/stderr, controlled by the environment variable `PCCL_LOG_LEVEL`. For example:

- `export PCCL_LOG_LEVEL=DEBUG` (Linux/macOS)
- `set PCCL_LOG_LEVEL=DEBUG` (Windows cmd)

Logging can be helpful for diagnosing deadlocks or mismatched states.