#pragma once

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
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
        /// When the @code ConnectionPhase@endcode is @code PEER_REGISTERED@endcode,
        /// this is the only legal state for the client to be in (with the exception of @code CONNECTING_TO_PEERS@endcode).
        IDLE,

        /// The client has voted to accept new peers.
        /// In this state, it waits for all peers to vote to accept new peers.
        VOTE_ACCEPT_NEW_PEERS,

        /// When all peers have voted to accept new peers, the client
        /// will enter the p2p establishment phase.
        CONNECTING_TO_PEERS,

        /// When the node has established p2p connections with all peers,
        /// but other peers have not yet declared that they have established p2p connections,
        /// the client will be in the @code WAITING_FOR_OTHER_PEERS@endcode state.
        /// Once all peers have declared that they have established p2p connections,
        /// the client will return to the @code IDLE@endcode state.
        WAITING_FOR_OTHER_PEERS,

        /// The client has voted to synchronize shared state.
        /// In this state, it waits for all peers to vote to synchronize shared state.
        VOTE_SYNC_SHARED_STATE,

        /// When all peers have voted to synchronize shared state, depending on whether
        /// the client has an up-to-date shared state, the client will either
        /// enter the @code DISTRIBUTE_SHARED_STATE@endcode state or the @code REQUEST_SHARED_STATE@endcode state.
        /// In the @code DISTRIBUTE_SHARED_STATE@endcode state, the client will distribute its shared state to peers
        /// that request it.
        /// Shared state distribution happens over one-off, non-persistent connections, similar to HTTP/1.1 connections.
        /// This is because of the logical mismatch between the persistent nature of the p2p connections
        /// and the non-persistent nature of shared state distribution, which may be between peers that are not
        /// topological neighbors. Thus, to avoid complexity in the p2p connections, shared state distribution
        /// is done over one-off connections.
        DISTRIBUTE_SHARED_STATE,

        /// When all peers have voted to synchronize shared state, depending on whether
        /// the client has an up-to-date shared state, the client will either
        /// enter the @code DISTRIBUTE_SHARED_STATE@endcode state or the @code REQUEST_SHARED_STATE@endcode state.
        /// In the @code REQUEST_SHARED_STATE@endcode state, the client will request shared state from a peer
        /// the master node has designated. This node may or may not be a topological neighbor as determined
        /// by the reduce topology. The client will then initiate a one-off connection to the designated peer.
        REQUEST_SHARED_STATE,


        /// The client has voted to complete the shared state distribution phase.
        /// After each client has requested the set of shared state entries that were declared outdated and subsequent distribution was successful,
        /// the client will vote to complete the shared state distribution phase.
        /// Clients that do not have any outdated shared state entries will immediately vote to complete the shared state distribution phase.
        /// Consensus is required for the vote to complete, meaning all clients in the @code PEER_ACCEPTED@endcode phase must vote to complete the shared state distribution phase.
        /// This consensus ensures that no client proceeds beyond the shared state distribution phase until all clients have received the shared state entries that were outdated
        /// and further, that clients that could act as shared state distributors progress. While shared state distributors by definition
        /// have non-dirty shared state and will thus immediately vote to complete the shared state distribution phase, this however does not
        /// imply that these clients progress to the next phase! They will wait, as intended for all clients to vote to complete the shared state distribution phase
        /// and while doing so process incoming shared state distribution requests.
        /// After the shared state synchronization phase ends, clients return to the @code IDLE@endcode state.
        VOTE_COMPLETE_SHARED_STATE_SYNC
    };

    struct ClientInfo {
        ccoip_uuid_t client_uuid;
        ConnectionPhase connection_phase = PEER_REGISTERED;
        ConnectionState connection_state = IDLE;
        ccoip_socket_address_t socket_address;
        uint16_t p2p_listen_port;
    };

    class CCoIPMasterState {
    public:
        /// Represents the status of a comparison between the shared state mask and supplied shared state entries.
        enum SharedStateMismatchStatus {
            SUCCESSFUL_MATCH,
            KEY_SET_MISMATCH,
            CONTENT_HASH_MISMATCH,
            REVISION_MISMATCH
        };

    private:
        /// Maps the socket address of the client to its assigned UUID
        /// Populated on successful session join, and cleared on client leave/disconnect
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

        /// Maps the UUID of the client to its assigned socket address
        /// Populated identically to `client_uuids` for reverse lookups
        std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

        /// Maps the uuid of the client to its client information
        std::unordered_map<ccoip_uuid_t, ClientInfo> client_info{};

        /// set of all uuids that have voted to accept new peers.
        /// cleared once accept new peers consensus is reached.
        std::unordered_set<ccoip_uuid_t> votes_accept_new_peers{};

        /// set of all uuids that have established p2p connections.
        /// cleared once all clients have established p2p connections.
        std::unordered_set<ccoip_uuid_t> votes_p2p_connections_established{};

        /// set of all uuids that have voted to synchronize shared state.
        /// cleared once shared state distribution consensus is reached.
        std::unordered_set<ccoip_uuid_t> votes_sync_shared_state{};

        /// set of all uuids that have voted to complete shared state distribution.
        /// cleared once the shared state distribution phase ends.
        std::unordered_set<ccoip_uuid_t> votes_sync_shared_state_complete{};

        /// Flag to indicate if the peer list has changed
        /// See @code hasPeerListChanged@endcode
        bool peer_list_changed = false;

        /// Defines the "mask" for shared state entries.
        /// The mask is defined by the identity of the set of shared state key strings and their corresponding hashes.
        /// This mask is checked against the by clients synchronizing shared state.
        /// Mismatch of the keys results in the client being kicked.
        /// Mismatch of a hash results will result in the client being notified to re-request the shared state entry
        /// whose hash does not match.
        /// Cleared once the shared state distribution phase ends.
        std::vector<SharedStateHashEntry> shared_state_mask{};

        /// Cache of the shared state hashes for all entries in the shared state mask
        /// by their key string.
        /// Cleared when the shared state distribution phase ends.
        std::unordered_map<std::string, uint64_t> shared_state_hashes{};

        /// The revision of the current shared state.
        /// If a client requests to sync a shared state with a revision smaller than this,
        /// this means that the client has an outdated shared state that needs to be updated.
        /// If a client requests a shared state with a revision larger than this, the client
        /// has advanced the state of the revision and the master's account needs to be updated
        /// to reflect the new maximum revision; It is expected that clients will either
        /// a) independently advance their shared state and remain in sync implicitly or
        /// b) request the shared state from the master node to synchronize.
        /// A new maximum revision must only ever be one greater than the previous maximum revision; Deviation will
        /// result in a kick.
        uint64_t shared_state_revision = 0;

        /// Maps the client UUID to the shared state mismatch status populated by the last invocation of @code sharedStateMatches@endcode.
        /// Cleared once the shared state distribution phase ends.
        std::unordered_map<ccoip_uuid_t, SharedStateMismatchStatus> shared_state_responses{};

        /// Maps the client UUID to the set of shared state keys that have dirty content, meaning
        /// that the hash of the shared state entry does not match the hash in the shared state mask.
        /// Cleared once the shared state distribution phase ends.
        std::unordered_map<ccoip_uuid_t, std::vector<std::string> > shared_state_dirty_keys{};

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

        /// Called when a client votes to synchronize shared state.
        /// All clients must vote to synchronize shared state before shared state distribution can begin.
        [[nodiscard]] bool voteSyncSharedState(const ccoip_socket_address_t &client_address);

        /// Called when a client votes to complete the shared state distribution phase.
        /// This indicates that it has completed receiving the subset of shared state that was outdated.
        /// All clients must vote to complete shared state distribution before shared state distribution can end.
        /// With the shared state distribution ending, the shared state synchronization phase also ends.
        /// When consensus is reached, @code endSharedStateDistributionPhase()@endcode will be called.
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool voteDistSharedStateComplete(const ccoip_socket_address_t &client_address);

        /// Returns true if all clients have voted to synchronize the shared state
        /// All clients here shall mean the subset of clients that are in the @code PEER_ACCEPTED@endcode phase.
        [[nodiscard]] bool syncSharedStateConsensus() const;

        /// Returns true if all clients have voted to accept new peers
        [[nodiscard]] bool acceptNewPeersConsensus() const;

        /// Transition to the p2p establishment phase
        /// Triggered after unanimous voting to accept new peers
        void transitionToP2PEstablishmentPhase();

        /// Marks that p2p connections have been established by a particular client.
        /// The particular client will be transitioned to the @code WAITING_FOR_OTHER_PEERS@endcode state
        /// until all peers have declared that they have established p2p connections.
        ///
        /// Once all peers have declared that they have established p2p connections,
        /// clients return back to the @code IDLE@endcode state.
        [[nodiscard]] bool markP2PConnectionsEstablished(const ccoip_socket_address_t &client_address);

        /// Transition to the p2p connections established phase
        /// Triggered after all clients have declared that they have established p2p connections.
        /// This means that all clients will return to the @code IDLE@endcode state.
        /// Returns false if any client is not in the @code WAITING_FOR_OTHER_PEERS@endcode state
        [[nodiscard]] bool transitionToP2PConnectionsEstablishedPhase();

        /// Transition to the shared state distribution phase
        /// Triggered after all clients have voted to synchronize shared state
        /// Returns false if a client is in the PEER_ACCEPTED phase but not in the VOTE_SYNC_SHARED_STATE state or if
        /// @code sharedStateMatches@endcode has not been called for all clients.
        [[nodiscard]] bool transitionToSharedStateSyncPhase();

        /// Called to end the shared state distribution phase
        /// Will revert all clients to the @code IDLE@endcode state
        [[nodiscard]] bool endSharedStateSyncPhase();

        /// Returns true if all clients have declared that they have established p2p connections
        [[nodiscard]] bool p2pConnectionsEstablishConsensus() const;

        /// Returns true if all clients have voted to complete the shared state distribution phase
        [[nodiscard]] bool syncSharedStateCompleteConsensus() const;

        /// Returns true if the shared state entries provided match the current "mask".
        /// A "mask" is defined by the identity of the set of shared state key strings and their corresponding hashes.
        /// If @code ignore_hashes@endcode is true, only the shared state keys are compared.
        /// If this is the first client to sync shared state, then the supplied shared state will define the "mask"
        /// for subsequent checks.
        /// This status will be retained until the shared state voting phase ends and can be queried
        /// via @code getSharedStateMismatchStatus@endcode.
        [[nodiscard]] SharedStateMismatchStatus sharedStateMatches(
            const ccoip_uuid_t &peer_uuid,
            uint64_t revision,
            const std::vector<SharedStateHashEntry> &entries);

        /// Returns the peers that a particular client should establish p2p connections with
        [[nodiscard]] std::vector<ClientInfo> getPeersForClient(
            const ccoip_socket_address_t &client_address
        ) const;

        /// Returns all client socket addresses
        [[nodiscard]] std::vector<ccoip_socket_address_t> getClientSocketAddresses();

        /// Returns a list pairs of client socket addresses and their corresponding UUIDs
        [[nodiscard]] std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > getClientEntrySet();

        /// Returns true if the peer list has changed since the last invocation of this function
        [[nodiscard]] bool hasPeerListChanged();

        /// Returns the shared state mismatch status. This status is set by @code sharedStateMatches@endcode.
        /// If @code sharedStateMatches@endcode has not been called yet since the start of the current shared state voting phase,
        /// this function will return std::nullopt.
        [[nodiscard]] std::optional<SharedStateMismatchStatus> getSharedStateMismatchStatus(
            const ccoip_uuid_t &peer_uuid);

        /// Finds the client UUID from the client address; returns std::nullopt if not found
        [[nodiscard]] std::optional<ccoip_uuid_t> findClientUUID(const ccoip_socket_address_t &client_address);

        /// Returns the client info for a particular client address; returns std::nullopt if not found
        [[nodiscard]] std::optional<std::reference_wrapper<ClientInfo> > getClientInfo(
            const ccoip_socket_address_t &client_address);

        /// Returns the current shared state revision. This represents the current maximum revision of the shared state
        /// that all clients have agreed upon to be the current shared state revision.
        [[nodiscard]] uint64_t getSharedStateRevision() const;

        /// Returns the set of shared state keys of a particular peer that have dirty content, meaning that the hash of the shared state entry
        /// does not match the hash in the shared state mask.
        /// If @code sharedStateMatches@endcode has not been called yet since the start of the current shared state voting phase,
        /// this function will return an empty vector.
        [[nodiscard]] const std::vector<std::string> &getOutdatedSharedStateKeys(ccoip_uuid_t peer_uuid);

        /// Returns the shared state entry hash for a particular key; returns 0 if the key is not found
        /// For a key to be found, it must be present in the shared state mask.
        /// The shared state mask needs to be populated by @code sharedStateMatches@endcode before calling this function.
        [[nodiscard]] uint64_t getSharedStateEntryHash(const std::string &key) const;

        /// Returns the set of shared state keys
        [[nodiscard]] std::vector<std::string> getSharedStateKeys() const;
    };
}
