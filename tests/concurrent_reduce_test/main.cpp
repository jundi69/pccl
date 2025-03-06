#include <cassert>
#include <ccoip.h>
#include <chrono>
#include <iostream>
#include <pccl.h>
#include <random>
#include <thread>

#include "../../log/include/pccl_log.hpp"

void panic(const int exit_code) { exit(exit_code); }

#define PCCL_CHECK(status)                                                                                             \
    {                                                                                                                  \
        pcclResult_t status_val = status;                                                                              \
        if (status_val != pcclSuccess) {                                                                               \
            std::cerr << "Error: " << status_val << std::endl;                                                         \
            panic(1);                                                                                                  \
        }                                                                                                              \
    }

#if defined(_MSC_VER)
// MSVC: Turn on global optimization + favor speed
#  define FORCE_OPTIMIZE_BEGIN __pragma(optimize("gt", on))
#  define FORCE_OPTIMIZE_END   __pragma(optimize("", on))

#elif defined(__GNUC__) || defined(__clang__)
// GCC / Clang: Push current options, then force -O3
#  define FORCE_OPTIMIZE_BEGIN \
_Pragma("GCC push_options") \
_Pragma("GCC optimize(\"O3\")")

#  define FORCE_OPTIMIZE_END \
_Pragma("GCC pop_options")

#else
// Fallback: do nothing
#  define FORCE_OPTIMIZE_BEGIN
#  define FORCE_OPTIMIZE_END
#endif

FORCE_OPTIMIZE_BEGIN
void fill_uniform(float *data, const size_t count) {
    std::mt19937 gen(42);
    std::uniform_real_distribution dis(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dis(gen);
    }
}
FORCE_OPTIMIZE_END

#define MAX_STEPS 1000

int main() {
    PCCL_CHECK(pcclInit());

    pcclComm_t *communicator{};
    constexpr pcclCommCreateParams_t params{.master_address = {.inet = {.protocol = inetIPv4, .ipv4 = {127, 0, 0, 1}},
                                                               .port = CCOIP_PROTOCOL_PORT_MASTER},
                                            .peer_group = 0};
    PCCL_CHECK(pcclCreateCommunicator(&params, &communicator));
    PCCL_CHECK(pcclConnect(communicator));

    int world_size{};
    pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);

    std::vector<float *> all_weights{};
    std::vector<float *> all_gradients{};

    constexpr size_t num_weights = 12;

    constexpr size_t n_elements = 1024 * 1024 * 256;

    for (size_t i = 0; i < num_weights; i++) {
        auto weights = new float[n_elements];
        auto gradients = new float[n_elements];
        all_weights.push_back(weights);
        all_gradients.push_back(gradients);
    }


    size_t i = 0;

    while (true) {
        i++;
        if (i > 1) {
            PCCL_CHECK(pcclUpdateTopology(communicator));
            PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size));
        }
        
        if (world_size > 1) {
            // PCCL_CHECK(pcclOptimizeTopology(communicator));
            PCCL_CHECK(pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size));
        }

        if (world_size < 2) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            continue;
        }

        std::cout << "Starting all reduces...";
        // NOTE: THIS IS NOT A SAFE PATTERN FOR RECOVERY, WE JUST DO THIS FOR EASY OF TESTING
        // WHETHER ALL REDUCES CAN RUN CONCURRENTLY AT ALL
        std::vector<pcclAsyncReduceOp_t> reduce_descriptors{};
        for (size_t j = 0; j < num_weights; j++) {
            pcclReduceDescriptor_t desc{
                .count = n_elements,
                .op = pcclSum,
                .tag = j,
                .src_descriptor = {.datatype = pcclFloat, .distribution_hint = PCCL_DISTRIBUTION_HINT_NONE},
                // .quantization_options = {.quantized_datatype = pcclUint8, .algorithm = pcclQuantMinMax},
                .quantization_options = {.quantized_datatype = pcclFloat, .algorithm = pcclQuantNone},
            };
            float *weights = all_weights[j];
            float *gradients = all_gradients[j];

            pcclAsyncReduceOp_t async_op{};
            pcclAllReduceAsync(weights, gradients, &desc, communicator, &async_op);
            reduce_descriptors.push_back(async_op);
        }

        for (const auto &async_op : reduce_descriptors) {
            pcclReduceInfo_t reduce_info{};
            PCCL_CHECK(pcclAwaitAsyncReduce(&async_op, &reduce_info));
        }


        pcclGetAttribute(communicator, PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE, &world_size);

        std::cout << "All reduces finished";
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
