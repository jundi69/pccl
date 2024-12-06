#include <pccl.h>
#include <random>

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
    pcclInit();

    pcclComm_t *communicator{};
    PCCL_CHECK(pcclCreateCommunicator(&communicator));

    constexpr ccoip_socket_address_t connect_address{
        .inet.protocol = inetIPv4,
        .inet.address.ipv4 = {127, 0, 0, 1},
        .port = 48148,
    };

    PCCL_CHECK(pcclConnectMaster(communicator, connect_address));

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

    while (true) {
        PCCL_CHECK(pcclAcceptNewPeers(communicator));
        PCCL_CHECK(pcclSynchronizeSharedState(communicator, &shared_state));

        if (shared_state.revision >= MAX_STEPS) {
            break;
        }

        fill_uniform(gradients, n_peers);
        PCCL_CHECK(pcclAllReduce(gradients, weights, n_peers, pcclFloat, pcclSum, 0, communicator, nullptr));

        shared_state.revision++;
    }

    PCCL_CHECK(pcclDestroyCommunicator(communicator));
}
