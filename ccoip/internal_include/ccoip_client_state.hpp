#pragma once
#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_shared_state.hpp>
#include <ccoip_types.hpp>

namespace ccoip {
    class CCoIPClientState {
        /// Maps p2p listen socket addresses of respective clients to their UUID assigned by the master.
        /// Populated by @code registerPeer()@endcode when this peer establishes a p2p connection to said p2p listen socket address.
        /// Cleared by @code unregisterPeer()@endcode when a client is no longer needed due to topology changes.
        ///
        /// If a p2p connection drops, for now it is asserted that it will also disconnect from the master, triggering
        /// a topology update. TODO: Implement a more robust mechanism to handle this.
        std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> socket_addr_to_uuid{};

        /// References the current shared state to be distributed.
        /// Is only valid when @code is_syncing_shared_state@endcode is true and is otherwise empty.
        ccoip_shared_state_t current_shared_state{};

        /// Whether the client is currently syncing shared state.
        /// If this flag is false while the shared state distribution server receives
        /// a shared state request, the server will respond with @code SHARED_STATE_NOT_DISTRIBUTED@endcode
        bool is_syncing_shared_state = false;

        /// The number of bytes sent during the shared state sync phase
        /// Accessible by @code getSharedStateSyncTxBytes()@endcode
        /// Reset by @code resetSharedStateSyncTxBytes()@endcode
        size_t shared_state_sync_tx_bytes = 0;

    public:
        /// Called to register a peer with the client state
        [[nodiscard]] bool registerPeer(const ccoip_socket_address_t &address, ccoip_uuid_t uuid);

        /// Called to unregister a peer with the client state
        [[nodiscard]] bool unregisterPeer(const ccoip_socket_address_t &address);

        /// Begins the shared state synchronization phase.
        /// On the client side, this means that we are either already in the shared state sync phase after
        /// a successful vote or that we would want to be in the shared state sync phase.
        /// The reason for this is documented in ccoip::CCoIPClientHandler::syncSharedState()
        void beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state);

        /// Ends the shared state synchronization phase
        void endSyncSharedStatePhase();

        /// Returns true if the client is currently syncing shared state
        [[nodiscard]] bool isSyncingSharedState() const;

        /// Returns the currently synced shared state
        [[nodiscard]] const ccoip_shared_state_t &getCurrentSharedState() const {
            return current_shared_state;
        }

        /// Returns the number of bytes sent during the shared state sync phase
        [[nodiscard]] size_t getSharedStateSyncTxBytes() const;

        /// Called to track additional bytes sent during the shared state sync phase
        void trackSharedStateTxBytes(size_t tx_bytes);

        /// Resets the number of bytes sent during the shared state sync phase
        void resetSharedStateSyncTxBytes();
    };
};
