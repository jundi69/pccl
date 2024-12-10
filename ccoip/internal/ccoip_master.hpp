#pragma once

#include <ccoip_inet.h>
#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPMaster {
        tinysockets::ServerSocket server_socket;

    public:
        volatile bool running = false;
        volatile bool interrupted = false;

        explicit CCoIPMaster(const ccoip_socket_address_t &listen_address);

        [[nodiscard]] bool run();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool join();

        [[nodiscard]] bool kickClient(const ccoip_socket_address_t &client_address) const;

        ~CCoIPMaster();

    private:

        void onClientRead(const ccoip_socket_address_t &client_address, std::span<uint8_t> data);
    };
}
