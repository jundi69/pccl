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
