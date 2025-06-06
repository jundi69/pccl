# Basic Hello World Example

The following is a simple example of a complete program that uses PCCL to perform an All-Reduce operation:

## hello_world.cpp (Peer Side)
```cpp
#include <pccl.h>
#include <iostream>
#include <thread>    // for sleep_for
#include <chrono>    // for seconds
#include <cstdlib>   // for exit
#include <cstring>   // for memset

// Helper macro for error-checking
#define PCCL_CHECK(stmt) do {                             \
    pcclResult_t _st = (stmt);                            \
    if (_st != pcclSuccess) {                             \
        std::cerr << "PCCL error: " << _st << '\n';       \
        std::exit(1);                                     \
    }                                                     \
} while(0)

// Hardcoded Master IP/Port
static constexpr uint8_t  MASTER_IP[4] = {127, 0, 0, 1};
static constexpr uint16_t MASTER_PORT  = 48148;

// We'll allow up to 5 distributed steps
static constexpr int MAX_STEPS = 5;

int main() {
    // 1) Initialize PCCL
    PCCL_CHECK(pcclInit());

    // 2) Create communicator
    pcclCommCreateParams_t params {
        .master_address = {
            .inet = {
                .protocol = inetIPv4,
                .ipv4 = { MASTER_IP[0], MASTER_IP[1], MASTER_IP[2], MASTER_IP[3] }
            },
            .port = MASTER_PORT
        },
        .peer_group = 0
    };
    pcclComm_t* comm = nullptr;
    PCCL_CHECK(pcclCreateCommunicator(&params, &comm));

    // 3) Connect to the master (blocking)
    std::cout << "[Peer] Connecting to master at "
              << int(MASTER_IP[0]) << "." << int(MASTER_IP[1]) << "."
              << int(MASTER_IP[2]) << "." << int(MASTER_IP[3])
              << ":" << MASTER_PORT << "...\n";
    PCCL_CHECK(pcclConnect(comm));
    std::cout << "[Peer] Connected!\n";

    // We'll have:
    //   - A local iteration counter "i" to skip updateTopology on i=0
    //   - A shared-state 'revision' in PCCL to keep all peers in step lock.
    int local_iter = 0; // for local logic

    // 4) Prepare some dummy data to place in shared state
    static float dummyWeights[8] = { 0.f }; // your model/optimizer state in real usage

    pcclTensorInfo_t tinfo{
        .name                     = "myWeights",
        .data                     = dummyWeights,
        .count                    = 8,
        .datatype                 = pcclFloat,
        .device_type              = pcclDeviceCpu,
        .allow_content_inequality = false
    };
    pcclSharedState_t sstate{
        .revision = shared_revision, // must match the current step
        .count    = 1,
        .infos    = &tinfo
    };

    int world_size{};
    PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));

    // 5) Enter the training loop
    // We'll do up to MAX_STEPS. Each step => we do some ring operation and a shared-state sync.
    while (true) {
        // A) If we are not on the very llocal first iteration, update topology
        if (local_iter > 0) {
            while (pcclUpdateTopology(comm) == pcclUpdateTopologyFailed) {
                std::cout << "[Peer] UpdateTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            // get up-to-date world size
            PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        }

        // C) If multiple peers are present => optionally optimize ring
        if (world_size > 1) {
            while (pcclOptimizeTopology(comm) == pcclOptimizeTopologyFailed) {
                std::cout << "[Peer] OptimizeTopology failed => retrying...\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            // get up-to-date world size
            PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
        } else {
            // alone => no ring-based operation => wait
            std::cout << "[Peer] alone => sleeping.\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
            // continue the loop to see if a new peer joined
            // next iteration => we can accept them
            local_iter++;
            continue;
        }

        // D) Example ring operation => a small All-Reduce
        float local_data[4];
        for (int k = 0; k < 4; k++) {
            local_data[k] = float(local_iter * 10 + (k + 1)); // something unique each iteration
        }
        float result_data[4] = {};

        pcclReduceDescriptor_t desc{
            .count = 4,
            .op    = pcclSum,
            .tag   = 0,
            .src_descriptor = {
                .datatype          = pcclFloat,
                .distribution_hint = pcclDistributionNone
            },
            .quantization_options = {
                .quantized_datatype = pcclFloat,
                .algorithm          = pcclQuantNone
            }
        };
        pcclReduceInfo_t reduce_info{};

        bool all_reduce_fatal_failure = false;
        for (;;) {
            pcclResult_t red_st = pcclAllReduce(local_data, result_data, &desc, comm, &reduce_info);
            if (red_st == pcclSuccess) {
                std::cout << "[Peer] local_iter=" << local_iter
                          << ", All-Reduce => result = [ ";
                for (float val : result_data) std::cout << val << " ";
                std::cout << "], Tx=" << reduce_info.tx_bytes
                          << ", Rx=" << reduce_info.rx_bytes << "\n";
                break;
            } else {
                std::cout << "[Peer] All-Reduce fail: " << red_st << "; Retrying...\n";
                // the world size may have changed after a failed all reduce if a peer drops.
                PCCL_CHECK(pcclGetAttribute(comm, PCCL_ATTRIBUTE_GLOBAL_WORLD_SIZE, &world_size));
            
                // if every peer but us dropped, we'll need to accept new peers and wait until we have at least 2 peers again
                if (world_size < 2) {
                    all_reduce_fatal_failure = true;
                    break;
                }
            }
        }
        if (all_reduce_fatal_failure) {
            std::cout << "[Peer] All-Reduce failed fatally. We will wait until we have at least 2 peers again.\n";
            local_iter++;
            continue;
        }

        // Increment the shared state revision followed by subsequent sync.
        sstate.revision++;

        // PCCL enforces that "revision" must increment by exactly 1, for each pcclSynchronizeSharedState call.
        pcclSharedStateSyncInfo_t ssi{};
        pcclResult_t sst = pcclSynchronizeSharedState(comm, &sstate, &ssi);
        if (sst == pcclSuccess) {
            std::cout << "[Peer] shared_revision now " << shared_revision
                      << ", sync => tx=" << ssi.tx_bytes
                      << ", rx=" << ssi.rx_bytes << "\n";
        } else {
            std::cerr << "[Peer] shared-state sync fail: " << sst
                      << " at revision=" << shared_revision << "\n";
            break;
        }

        // F) Stop if we've done enough steps => i.e., if shared_revision >= MAX_STEPS
        //    Each peer that sees we reached that step will break out the same iteration.
        if (shared_revision >= MAX_STEPS) {
            std::cout << "[Peer] Reached revision " << shared_revision
                      << " => done.\n";
            break;
        }

        // G) local iteration increments for next loop:
        local_iter++;
    }

    // 6) Cleanup
    PCCL_CHECK(pcclDestroyCommunicator(comm));
    std::cout << "[Peer] Exiting.\n";
    return 0;
}

```

## How to launch

First, we need a running CCoIP master instance.
Here, we will use the provided `ccoip_master` executable:

```bash
./ccoip_master
```

Next, we run two instances of the `hello_world` program:

```bash
./hello_world &
./hello_world &
await
```
