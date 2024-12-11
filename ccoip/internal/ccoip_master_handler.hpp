#pragma once

#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>

namespace ccoip {

    enum ConnectionState {
        /// The client is registered with the master;
        /// Initially, the client is in this state.
        /// It does not yet participate in collective communications operations
        /// or shared state distribution.
        /// In this tate, the client is not allowed to request to participate in any of the above.
        PEER_REGISTERED,

        /// Peers have accepted the client.
        /// Peers periodically accept new clients to join the running session.
        /// This is a phase unanimously agreed upon by all peers.
        /// In this phase, clients will establish p2p connections with new peers.
        PEER_ACCEPTED
    };

    struct ClientInfo {
        ConnectionState connection_state = PEER_REGISTERED;
    };

    class CCoIPMasterHandler {
        tinysockets::ServerSocket server_socket;

        /// Thread ID of the server thread.
        /// Server socket callbacks such as @code onClientRead @endcode and @code onClientDisconnect @endcode
        /// will only ever be invoked from this thread.
        std::thread::id server_thread_id;

        struct {
            /// Maps the socket address of the client to its assigned UUID
            /// Populated on successful session join, and cleared on client leave/disconnect
            std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

            /// Maps the UUID of the client to its assigned socket address
            /// Populated identically to `client_uuids` for reverse lookups
            std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

            /// Maps the uuid of the client to its client information
            std::unordered_map<ccoip_uuid_t, ClientInfo> client_info{};
        } server_state;

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
        void handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                  const C2MPacketAcceptNewPeers &packet);

        void registerClient(const ccoip_socket_address_t &client_address,
                            ccoip_uuid_t uuid);

        void unregisterClient(const ccoip_socket_address_t &client_address);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                      const C2MPacketRequestSessionRegistration &packet);

        // server socket callbacks
        void onClientRead(const ccoip_socket_address_t &client_address,
                          const std::span<uint8_t> &data);

        void onClientDisconnect(const ccoip_socket_address_t &client_address);
    };
}
