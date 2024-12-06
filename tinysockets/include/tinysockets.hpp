#pragma once

#include <ccoip_inet.h>

namespace tinysockets {
    class ServerSocket {
    private:
        ccoip_socket_address_t listen_address;
        int server_socket;

    public:
        explicit ServerSocket(const ccoip_socket_address_t &listen_address);

        [[nodiscard]] bool listen();

        ~ServerSocket();
    };
};
