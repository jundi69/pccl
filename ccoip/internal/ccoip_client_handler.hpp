#pragma once

#include <ccoip_client_state.hpp>
#include <ccoip_shared_state.hpp>
#include <ccoip_packets.hpp>

#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPClientHandler {
        /// Blocking socket for connection to master node
        tinysockets::BlockingIOSocket client_socket;

        /// All state of the client is encapsulated in this object
        CCoIPClientState client_state;

        /// Socket listening for p2p connections
        tinysockets::ServerSocket p2p_socket;

        /// Socket listening for shared state distribution requests
        tinysockets::ServerSocket shared_state_socket;

        /// Thread ID of the p2p server thread
        std::thread::id p2p_server_thread_id;

        /// Thread ID of the shared state server thread
        std::thread::id shared_state_server_thread_id;

        /// Open p2p connections
        std::unordered_map<ccoip_uuid_t, tinysockets::BlockingIOSocket> p2p_connections;

        bool interrupted = false;

    public:
        explicit CCoIPClientHandler(const ccoip_socket_address_t &address);

        [[nodiscard]] bool connect();

        [[nodiscard]] bool acceptNewPeers();

        [[nodiscard]] bool syncSharedState(ccoip_shared_state_t &shared_state);

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool updateTopology();

        [[nodiscard]] bool join();

        [[nodiscard]] bool isInterrupted() const;

        ~CCoIPClientHandler();

    private:
        [[nodiscard]] bool establishP2PConnections();

        [[nodiscard]] bool establishP2PConnection(const M2CPacketNewPeerInfo &peer);

        // p2p packet handlers
        void handleP2PHello(const ccoip_socket_address_t &client_address, const P2PPacketHello &packet);

        // p2p server socket callbacks
        void onP2PClientRead(const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data);

        // shared state packet handlers
        void handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                      const C2SPacketRequestSharedState &packet);

        // shared state server socket callbacks
        void onSharedStateClientRead(const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data);
    };
}
