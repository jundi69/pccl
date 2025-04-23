# Basic Usage Workflow

Once you have built or installed PCCL, using it typically involves:

1. Launching a Master Node
2. Creating and connecting Peers
3. Performing Collective Operations
4. Shutting Down (both peers and master)

## Launching a Master Node

A *master node* is the orchestrator that tracks who has joined the run (peers) and what collective "topology" (e.g. ring
order) should be.
You can launch it in one of two ways:

### Via the PCCL API:

```cpp
#include <pccl.h>
#include <pccl.h>

#include <thread>
#include <csignal>
#include <iostream>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; exit(1); } }

static pcclMasterInstance_t* master_instance{};

void signal_handler(const int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        std::cout << "Interrupting master node..." << std::endl; // is this signal async safe?
        PCCL_CHECK(pcclInterruptMaster(master_instance));
    }
}

int main() {
    ccoip_socket_address_t listen_address {};
    listen_address.inet.ipv4 = {0, 0, 0, 0};
    listen_address.port = 48148;

    // install signal handler for interrupt & termination signals
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    PCCL_CHECK(pcclCreateMaster(listen_address, &master_instance));
    PCCL_CHECK(pcclRunMaster(master_instance));

    PCCL_CHECK(pcclMasterAwaitTermination(master_instance));
    PCCL_CHECK(pcclDestroyMaster(master_instance));

    std::cout << "Master node terminated." << std::endl;
}
```

### Via provided Executable

```bash
./ccoip_master
```

This tool handles the same calls internally as the example above.
Once the master is running, peers can connect to it by specifying the master's IP address and port.

## Creating and connecting Peers

Each worker or training process is a *peer*. In C/C++:

1. Initialize the PCCL library
2. Create a communicator using the master's address and a chosen "peer group" ID.
3. Connect to the master and wait until the master's state machine has accepted/acknowledged the peer.

```cpp
#include <pccl.h>
#include <iostream>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; exit(1); } }

int main() {
    // 1) Initialize library
    PCCL_CHECK(cclInit());
    
    // 2) Prepare communicator creation parameters
    pcclComm_t *communicator{};
    constexpr pcclCommCreateParams_t params{
            .master_address = {
                    .inet = {
                            .protocol = inetIPv4,
                            // suppose that the master is at 10.1.2.92
                            .ipv4 = {10, 1, 2, 92}
                    },
                    .port = CCOIP_PROTOCOL_PORT_MASTER
            },
            .peer_group = 0
    };
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));
    
    std::cout << "Peer successfully connected!" << std::endl;
    
    // Now the communicator is ready to do All-Reduce, Shared State Sync, etc.
   
    // 3) Clean up
    // Eventually, destroy the communicator and deinitialize the library
    PCCL_CHECK(pcclDestroyCommunicator(communicator));
    return 0;
}
```

- `pcclConnect`: May stall until the master and other peers collectively *accept* you (in PCCL, the existing peers must
  effectively vote to integrate a new join).
  If the run is in the middle of certain operations (e.g. All-Reduce), it may take time before the system transitions to
  accept the newcomer.

- `peer_group`: All peers with the same group id form a logical "communicator group" for collective operations and
  shared state synchronization.
  If you want a certain subset of peers to only all-reduce and shared state synchronize among themselves, assign them a
  unique group id.

## A Typical Iteration Flow

Once a peer is connected to the master, it can:

1. Periodically Update / Optimize the Topology
2. Check the current world size
3. Run Collectives (e.g. All-Reduce)
4. Optionally Synchronize Shared State (e.g., model parameters)

In many distributed training loops, you'll repeat a sequence of:

### (Optional) `pcclUpdateTopology(...)`

- This is where existing peers collectively decide to accept any newly connected peers or remove dropped ones.
- `Important:` If you call `pcclUpdateTopology` **too early** (e.g., at the very first iteration), you can inadvertently
  cause a deadlock because the pre-existing peers that just agreed to accept you just finished the `pcclUpdateTopology`
  step.
  Immediately calling `pcclUpdateTopology` again will not result in unanimous agreement - as the other peers will have
  moved on to subsequent operations -
  and run will stall.
  A common rule of thumb is to call `pcclUpdateTopology` only *after* at least one local iteration has passed, ensuring
  that the joined peer in the first iteration will call the same set of operations that pre-existing peers will still
  perform in the current
  iteration that they are in.

### Obtain the world size via `pcclGetAttribute`

```cpp
int world_size{};
PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
```

- *Pitfall*: When making branching decision based on world size, make sure it up-to-date *after* pcclUpdateTopology, as
  new peers may have been accepted into the run during
  this step, resulting in a dirty world size.
  Inconsistency here can lead to **deadlocks**, if the application
  logic branches based on the world size (e.g., if a peer thinks the world size is 1, but others already have the
  updated count of 2), resulting in different branches being taken and unanimous agreement not being possible anymore.

### (Optional) `pcclOptimizeTopology(...)`

- If `world_size > 1`, you may choose to optimize the ring order. This triggers bandwidth tests among peers, potentially
  re-solving the traveling salesman problem for better throughput.
  Master determines what bandwidth information out of all possible point to point bandwidth measurements it does not yet
  know and requests the peers to measure the set of missing "cost edges".
  Note: if this problem is un-feasible, it will be solved heuristically first. A full solution may be attempted
  asynchronously in the background by the master node. When better solutions are found, they will be distributed to the
  peers in the next invocation of `pcclUpdateTopology`.

- `pcclOptimizeTopology` will establish p2p connections to newly designated neighboring peers if the topology changes in
  much the same way that `pcclOptimizeTopology` will, with the difference of
  not accepting brand-new peers into the run. However, the world size can still change when peers leave!
  It is recommended that the world size is re-obtained after invoking `pcclOptimizeTopology`.

### The recommended pattern for a loop iteration is thus as follows:

```cpp
int world_size{};
PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));

for (uint64_t i = 0;;i++) {
    if (i > 0) {
        while (pcclUpdateTopology(comm) == pcclUpdateTopologyFailed) {
            std::cout << "[Peer] UpdateTopology failed => retrying...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size)); // get the new world size
    }
    
    if (world_size > 1) {
        // optimize the ring order for better throughput
        while (pcclOptimizeTopology(communicator) == pcclTopologyOptimizationFailed) {
            std::cout << "[Peer] OptimizeTopology failed => retrying...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size)); // get the new world size
    }
    
    if (world_size < 2) {
        std::this_thread::sleep_for(std::chrono::seconds(1)); // wait until we have at least 2 peers
        continue;
    }
    
    // ... perform shared state synchronization, all-reduce, etc.
}
```

This pattern follows best practices to avoid deadlocks and guard the training loop by the means that PCCL intends.

Note that having multiple call-sites to `pcclUpdateTopology` in a single iteration is not recommended, as it makes it
significantly
harder to ensure that peers are in unanimous agreement about the current state of the run.

## Shared state synchronization

If your application intents to take advantage of PCCL's shared state synchronization scheme to e.g. keep parameters in sync,  you can use the following pattern:
```cpp
float myModelWeights[4096] = {/* ... */};

pcclTensorInfo_t tinfo{
    .name                     = "model_weights",
    .data                     = myModelWeights,
    .count                    = 4096,
    .datatype                 = pcclFloat,
    .device_type              = pcclDeviceCpu,
    .allow_content_inequality = false
};
pcclSharedState_t sstate{
    .revision = (current_revision + 1), // must be exactly +1 from your previous revision
    .count    = 1,
    .infos    = &tinfo
};

pcclSharedStateSyncInfo_t sync_info{};
pcclResult_t sync_st = pcclSynchronizeSharedState(communicator, &sstate, &sync_info);
if (sync_st == pcclSuccess) {
    // If any mismatch was found, your local array is updated
    // with the "popular" version from other peers
    if (i > 0) {
        assert(sync_info.rx_bytes == 0); // assert no bit-diverge happened
    }
} else {
    // Possibly revision violation or peer dropout
}
```
Shared state synchronization unifies two main concerns:
- Checkpoint distribution on peer join
- Preventing peer divergence

The `allow_content_inequality` flag can be set to `true` if you want to allow for content inequality between peers.
This however should only be used if you are certain that the content inequality is acceptable for your use case.
Bit-parity among peers is more than feasible given that the ring reduce will guarantee that all peers receive
the same result.
If shared state is a deterministic function of the ring reduce result, you may assert that no
shared state divergence will occur, meaning that the number of received bytes is zero, as all peers will independently
advance to the same shared state. Thus, in the ideal case, shared state synchronization is relegated to a checkpoint synchronization
mechanism for joining peers while being a no-op most of the time.
As long as your optimizer is an element-wise expression - as is the case for most optimizers -
not even GPUs will behave indeterministically here, and you are free to assert bit-parity among peers.
You can be very certain that as long as you are using a well-known optimizer (such as `AdamW`) that all indeterminism
that you are observing is caused by improper synchronization of optimizer state or learning rate scheduler associated state.
Ensuring that a) your learning state scheduler behaves deterministically and b) all optimizer state is synchronized properly via shared state
should result in bit-identical independent advancement of model parameters, eliminating the need for a shared state retransmission.

## Performing Collective Operations

If you have at least 2 peers, you can do collective communications operations, such as an All-Reduce.
For example:

```cpp
float local_data[1024] = {/* ... */};
float result[1024]{};

pcclReduceDescriptor_t desc{
    .count = 1024,
    .op    = pcclSum,
    .tag   = 0,  // unique identifier for this operation
    .src_descriptor = {
        .datatype           = pcclFloat,
        .distribution_hint  = PCCL_NORMAL_DISTRIBUTION_HINT // hint distribution if know to potentially improve quantization
    },
    .quantization_options = {
        .quantized_datatype = pcclFloat,
        .algorithm          = pcclQuantNone
    }
};

pcclReduceInfo_t info{};
pcclResult_t result = pcclAllReduce(local_data, result, &desc, communicator, &info);
if (result == pcclSuccess) {
    // 'result' now has the sum across all peers
    // info.tx_bytes, info.rx_bytes contain stats
} else {
    // e.g. a peer dropped out => typically retry or handle
}
```

It is also possible to use the same buffer for both the source and destination argument, note however that PCCL will
have to internally allocate memory to fulfill this request.
You may want to take full control of memory management and prevent additional allocations by simply allocating memory
which PCCL will have to allocate anyway yourself.