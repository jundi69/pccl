#include <ccoip.h>
#include <iostream>
#include <pccl.h>
#include <random>
#include <thread>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; exit(1); } }

void fill_uniform(float *data, const size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
}

#define MAX_STEPS 100

int main() {
    PCCL_CHECK(pcclInit());

    pcclComm_t *communicator{};
    constexpr pcclCommCreateParams_t params{
        .master_address = {
            .inet = {
                .protocol = inetIPv4,
                .ipv4 = {127, 0, 0, 1}
            },
            .port = CCOIP_PROTOCOL_PORT_MASTER
        },
        .peer_group = 0
    };
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));

    int world_size{};
    pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);

    constexpr size_t n_weights = 1024;
    const auto weights = new float[n_weights];
    fill_uniform(weights, n_weights);

    constexpr size_t count = 1;
    pcclTensorInfo_t infos[count] = {
        {"weights", weights, n_weights, pcclFloat, false}
    };
    pcclSharedState_t shared_state = {
        .revision = 1,
        .count = count,
        .infos = infos,
    };

    constexpr size_t n_peers = 1;
    const auto gradients = new float[n_peers];

    size_t i = 0;
    while (true) {
        if (i > 0 || world_size == 1) {
            PCCL_CHECK(pcclUpdateTopology(communicator));
        }
        PCCL_CHECK(pcclSynchronizeSharedState(communicator, &shared_state, nullptr));

        pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);

        if (world_size < 2) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            i++;
            continue;
        }

        if (shared_state.revision >= MAX_STEPS) {
            break;
        }

        fill_uniform(gradients, n_peers);
        pcclAsyncReduceOp_t async_op{};
        pcclReduceInfo_t reduce_info{};
        do {
            pcclAllReduceAsync(gradients, weights, n_peers, pcclFloat, pcclSum, 0, communicator, &async_op);
        } while (pcclAwaitAsyncReduce(&async_op, &reduce_info) != pcclSuccess);

        shared_state.revision++;
        i++;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
