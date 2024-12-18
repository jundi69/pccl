#pragma once

#include <ccoip_inet.h>
#include <ccoip_master_state.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPMasterHandler {
        /// Server socket
        tinysockets::ServerSocket server_socket;

        /// Thread ID of the server thread.
        /// Server socket callbacks such as @code onClientRead @endcode and @code onClientDisconnect @endcode
        /// will only ever be invoked from this thread.
        std::thread::id server_thread_id;

        /// Encapsulates the state of the master node
        /// All administrative state and transactions are encapsulated in this object
        CCoIPMasterState server_state;

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
        void sendP2PConnectionInformation();

        void checkP2PConnectionsEstablished();

        void checkAcceptNewPeersConsensus();

        /// Finds the optimal peer to distribute the shared state to the specified requester.
        std::optional<ccoip_socket_address_t> findBestSharedStateTxPeer(const ccoip_socket_address_t &peer_address);

        void checkSyncSharedStateConsensus();

        // packet handling functions
        void handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                  const C2MPacketAcceptNewPeers &packet);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                      const C2MPacketRequestSessionRegistration &packet);


        void handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                             const C2MPacketP2PConnectionsEstablished &packet);

        void handleSyncSharedState(const ccoip_socket_address_t &client_address,
                                   const ccoip::C2MPacketSyncSharedState &packet);

        // server socket callbacks
        void onClientRead(const ccoip_socket_address_t &client_address,
                          const std::span<uint8_t> &data);

        void onClientDisconnect(const ccoip_socket_address_t &client_address);
    };
}
