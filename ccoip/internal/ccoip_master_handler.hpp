#pragma once

#include <ccoip_inet.h>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPMasterHandler {
        tinysockets::ServerSocket server_socket;

    public:
        volatile bool running = false;
        volatile bool interrupted = false;

        explicit CCoIPMasterHandler(const ccoip_socket_address_t &listen_address);

        [[nodiscard]] bool run();
        [[nodiscard]] bool interrupt();
        [[nodiscard]] bool join();
        [[nodiscard]] bool kickClient(const ccoip_socket_address_t &client_address) const;

        ~CCoIPMasterHandler();

    private:
        void handleAcceptNewPeers(const ccoip_socket_address_t &client_address, const C2MPacketAcceptNewPeers &packet);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address, const C2MPacketRequestSessionJoin &packet);

        void onClientRead(const ccoip_socket_address_t &client_address, std::span<uint8_t> data);
    };
}
