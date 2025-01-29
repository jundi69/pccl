#include <ccoip.h>
#include <iostream>
#include <pccl.h>
#include <random>
#include <thread>

void panic(const int exit_code) {
    exit(exit_code);
}

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; panic(1); } }

void fill_uniform(float *data, const size_t count) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
}

#define MAX_STEPS 1000

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
            .revision = 0,
            .count = count,
            .infos = infos,
    };

    constexpr size_t n_elements = 1024;
    const auto gradients = new float[n_elements];

    size_t i = 0;
    while (true) {
        if (i > 0 || world_size == 1) {
            PCCL_CHECK(pcclUpdateTopology(communicator));
        }
        PCCL_CHECK(pcclOptimizeTopology(communicator));
        pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);

        if (world_size < 2) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            i++;
            continue;
        }

        PCCL_CHECK(pcclSynchronizeSharedState(communicator, &shared_state, nullptr));
        if (shared_state.revision >= MAX_STEPS) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));

        fill_uniform(gradients, n_elements);
        pcclAsyncReduceOp_t async_op{};
        pcclReduceInfo_t reduce_info{};
        do {
            constexpr pcclReduceDescriptor_t desc{
                    .count = n_elements,
                    .op = pcclSum,
                    .tag = 0,
                    .src_descriptor = {.datatype = pcclFloat, .distribution_hint = PCCL_DISTRIBUTION_HINT_NONE},
                    // .quantization_options = {.quantized_datatype = pcclFloat, .algorithm = pcclQuantNone},
                    .quantization_options = {.quantized_datatype = pcclUint8, .algorithm = pcclQuantMinMax},
            };
            pcclAllReduceAsync(gradients, weights, &desc, communicator, &async_op);
        } while (pcclAwaitAsyncReduce(&async_op, &reduce_info) != pcclSuccess);
        std::cout << "All reduce finished: " << shared_state.revision << "; Rx-Bytes:" << reduce_info.rx_bytes <<
                "; Tx-Bytes:" << reduce_info.tx_bytes << std::endl;
        shared_state.revision++;
        i++;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
