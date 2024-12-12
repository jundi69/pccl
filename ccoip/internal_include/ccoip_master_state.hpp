#pragma once

#include <ccoip_inet_utils.hpp>
#include <ccoip_types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <optional>

namespace ccoip {
    enum ConnectionPhase {
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

    enum ConnectionState {
        /// The client is idle.
        /// It does not participate in any collective communications operations
        /// or shared state distribution.
        /// When the @code ConnectionPhase @endcode is @code PEER_REGISTERED @endcode,
        /// this is the only legal state for the client to be in (with the exception of @code CONNECTING_TO_PEERS @endcode).
        IDLE,

        /// The client has voted to accept new peers.
        /// In this state, it waits for all peers to vote to accept new peers.
        VOTE_ACCEPT_NEW_PEERS,

        /// When all peers have voted to accept new peers, the client
        /// will enter the p2p establishment phase.
        CONNECTING_TO_PEERS,

        /// When the node has established p2p connections with all peers,
        /// but other peers have not yet declared that they have established p2p connections,
        /// the client will be in the @code WAITING_FOR_OTHER_PEERS @endcode state.
        /// Once all peers have declared that they have established p2p connections,
        /// the client will return to the @code IDLE @endcode state.
        WAITING_FOR_OTHER_PEERS
    };

    struct ClientInfo {
        ccoip_uuid_t client_uuid;
        ConnectionPhase connection_phase = PEER_REGISTERED;
        ConnectionState connection_state = IDLE;
        ccoip_socket_address_t socket_address;
        uint16_t p2p_listen_port;
    };

    class CCoIPMasterState {
        /// Maps the socket address of the client to its assigned UUID
        /// Populated on successful session join, and cleared on client leave/disconnect
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

        /// Maps the UUID of the client to its assigned socket address
        /// Populated identically to `client_uuids` for reverse lookups
        std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

        /// Maps the uuid of the client to its client information
        std::unordered_map<ccoip_uuid_t, ClientInfo> client_info{};

        /// set of all uuids that have voted to accept new peers.
        /// cleared once accept new peers consensus is reached
        std::unordered_set<ccoip_uuid_t> votes_accept_new_peers{};

        /// set of all uuids that have established p2p connections.
        /// cleared once all clients have established p2p connections
        std::unordered_set<ccoip_uuid_t> votes_p2p_connections_established{};

        /// Flag to indicate if the peer list has changed
        /// See @code hasPeerListChanged @endcode
        bool peer_list_changed = false;

    public:
        /// Registers a client
        [[nodiscard]] auto registerClient(const ccoip_socket_address_t &client_address,
                                          uint16_t p2p_listen_port, ccoip_uuid_t uuid) -> bool;

        /// Unregisters a client
        [[nodiscard]] bool unregisterClient(const ccoip_socket_address_t &client_address);

        /// Checks if a client is registered
        [[nodiscard]] bool isClientRegistered(const ccoip_socket_address_t &client_address) const;

        /// Called when a client votes to accept new peers.
        /// All clients must vote to accept new peers before all clients start the p2p establishment phase.
        [[nodiscard]] bool voteAcceptNewPeers(const ccoip_socket_address_t &client_address);

        /// Returns true if all clients have voted to accept new peers
        [[nodiscard]] bool acceptNewPeersConsensus() const;

        /// Transition to the p2p establishment phase
        /// Triggered after unanimous voting to accept new peers
        void transitionToP2PEstablishmentPhase();

        /// Marks that p2p connections have been established by a particular client.
        /// The particular client will be transitioned to the @code WAITING_FOR_OTHER_PEERS @endcode state
        /// until all peers have declared that they have established p2p connections.
        ///
        /// Once all peers have declared that they have established p2p connections,
        /// clients return back to the @code IDLE @endcode state.
        [[nodiscard]] bool markP2PConnectionsEstablished(const ccoip_socket_address_t &client_address);

        /// Transition to the p2p connections established phase
        /// Triggered after all clients have declared that they have established p2p connections.
        /// This means that all clients will return to the @code IDLE @endcode state.
        [[nodiscard]] bool transitionToP2PConnectionsEstablishedPhase();

        /// Returns true if all clients have declared that they have established p2p connections
        [[nodiscard]] bool p2pConnectionsEstablishConsensus() const;

        /// Returns the peers that a particular client should establish p2p connections with
        [[nodiscard]] std::vector<ClientInfo> getPeersForClient(
            const ccoip_socket_address_t &client_address
        ) const;

        /// Returns all client socket addresses
        [[nodiscard]] std::vector<ccoip_socket_address_t> getClientSocketAddresses();

        /// Returns true if the peer list has changed since the last invocation of this function
        [[nodiscard]] bool hasPeerListChanged();

    private:
        [[nodiscard]] std::optional<std::reference_wrapper<ClientInfo> > getClientInfo(
            const ccoip_socket_address_t &client_address);
    };
}
