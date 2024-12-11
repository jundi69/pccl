#pragma once

#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPMasterHandler {
        tinysockets::ServerSocket server_socket;

        /// Maps the socket address of the client to its assigned UUID
        /// Populated on successful session join, and cleared on client leave/disconnect
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

        /// Maps the UUID of the client to its assigned socket address
        /// Populated identically to `client_uuids` for reverse lookups
        std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

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

        void registerClient(const ccoip_socket_address_t &client_address, ccoip_uuid_t uuid);

        void unregisterClient(const ccoip_socket_address_t &client_address);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                      const C2MPacketRequestSessionJoin &packet);

        void onClientRead(const ccoip_socket_address_t &client_address, std::span<uint8_t> data);

        void onClientDisconnect(const ccoip_socket_address_t &client_address);
    };
}
