#pragma once

#include <ccoip_client_state.hpp>
#include <ccoip_packets.hpp>

#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPClientHandler {
        /// Blocking socket for connection to master node
        tinysockets::BlockingIOSocket client_socket;

        /// All state of the client is encapsulated in this object
        CCoIPClientState client_state;

        /// P2P socket for connection to peers
        tinysockets::ServerSocket p2p_socket;

        /// Thread ID of the p2p server thread
        std::thread::id p2p_server_thread_id;

        /// Open p2p connections
        std::unordered_map<ccoip_uuid_t, tinysockets::BlockingIOSocket> p2p_connections;

        bool interrupted = false;

    public:
        explicit CCoIPClientHandler(const ccoip_socket_address_t &address);

        [[nodiscard]] bool connect();

        [[nodiscard]] bool acceptNewPeers();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool updateTopology();

        [[nodiscard]] bool join();

        [[nodiscard]] bool isInterrupted() const;

        ~CCoIPClientHandler();

    private:
        [[nodiscard]] bool establishP2PConnections();

        [[nodiscard]] bool establishP2PConnection(const M2CPacketNewPeerInfo &peer);

        // p2p packet handlers
        void handleP2PHello(const ccoip_socket_address_t & client_address, const ccoip::P2PPacketHello & packet);

        // p2p server socket callbacks
        void onP2PClientRead(const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data);
    };
}
