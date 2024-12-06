#include <pccl.h>

#include <thread>

#define PCCL_CHECK(status) { pcclResult_t status_val = status; if (status_val != pcclSuccess) { std::cerr << "Error: " << status_val << std::endl; return status_val; } }

int main() {
    constexpr ccoip_socket_address_t listen_address{
        .address = {
            .ipv4 = {0, 0, 0, 0}
        },
        .port = 48148
    };

    pcclMasterInstance_t master_instance{};
    PCCL_CHECK(pcclCreateMaster(listen_address, &master_instance));
    PCCL_CHECK(pcclRunMaster(master_instance));
    std::this_thread::sleep_for(std::chrono::seconds(10));
    PCCL_CHECK(pcclInterruptMaster(master_instance));
    PCCL_CHECK(pcclMasterAwaitTermination(master_instance));
    PCCL_CHECK(pcclDestroyMaster(master_instance));
}
