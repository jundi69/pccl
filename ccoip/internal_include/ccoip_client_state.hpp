#pragma once

#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_shared_state.hpp>
#include <ccoip_types.hpp>
#include <future>
#include <optional>
#include <unordered_set>
#include <unordered_map>
#include <thread>

namespace ccoip {
    class CCoIPClientState {
        /// UUID assigned to this client by the master
        ccoip_uuid_t assigned_uuid{};

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

        /// Tags of all running collective communications operations
        std::unordered_set<uint64_t> running_collective_coms_ops_tags{};

        /// Maps tags of running collective operation tasks to their respective threads
        std::unordered_map<uint64_t, std::thread> running_reduce_tasks{};

        /// Maps tags of running collective operation tasks to their respective promises;
        /// These promises are used to signal the completion of the collective operation task
        /// and indicate whether the operation was successful or not.
        /// true=success, false=failure
        std::unordered_map<uint64_t, std::promise<bool> > running_reduce_tasks_promises{};

        // TODO: THIS IS SUBJECT TO CHANGE AND A TEMPORARY HACK!!
        //  FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        std::vector<ccoip_uuid_t> ring_order;

    public:
        /// Called to register a peer with the client state
        [[nodiscard]] bool registerPeer(const ccoip_socket_address_t &address, ccoip_uuid_t uuid);

        /// Called to unregister a peer with the client state
        [[nodiscard]] bool unregisterPeer(const ccoip_socket_address_t &address);

        /// Sets the uuid assigned to this client by the master
        void setAssignedUUID(const ccoip_uuid_t &new_assigned_uuid);

        /// Returns the uuid assigned to this client by the master
        [[nodiscard]] const ccoip_uuid_t &getAssignedUUID() const;

        /// Begins the shared state synchronization phase.
        /// On the client side, this means that we are either already in the shared state sync phase after
        /// a successful vote or that we would want to be in the shared state sync phase.
        /// The reason for this is documented in ccoip::CCoIPClientHandler::syncSharedState()
        void beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state);

        /// Ends the shared state synchronization phase
        void endSyncSharedStatePhase();

        /// Returns true if the client is currently syncing shared state
        [[nodiscard]] bool isSyncingSharedState() const;

        /// Returns true if there is a collective communications operation running with the specified tag
        [[nodiscard]] bool isCollectiveComsOpRunning(uint64_t tag) const;

        /// Returns true if there is any collective communications operation running
        [[nodiscard]] bool isAnyCollectiveComsOpRunning() const;

        /// Declares a collective communications operation with the specified tag started.
        /// Returns false if a collective communications operation with the same tag is already running.
        [[nodiscard]] bool startCollectiveComsOp(uint64_t tag);

        /// Declares a collective communications operation with the specified tag ended
        /// Returns false if no collective communications operation with the specified tag is running.
        [[nodiscard]] bool endCollectiveComsOp(uint64_t tag);

        /// Launches an asynchronous all reduce operation
        [[nodiscard]] bool launchAsyncCollectiveOp(uint64_t tag, std::function<void(std::promise<bool> &)> &&task);

        /// Join the collective communications operation with the specified tag
        /// Returns false if no collective communications operation with the specified tag is running
        [[nodiscard]] bool joinAsyncReduce(uint64_t tag);

        /// Returns true if the collective communications operation with the specified tag failed;
        /// Returns std::nullopt if the tag is not found or the operation has not completed
        [[nodiscard]] std::optional<bool> hasCollectiveComsOpFailed(uint64_t tag);

        /// Returns the currently synced shared state
        [[nodiscard]] const ccoip_shared_state_t &getCurrentSharedState() const;

        /// Returns the number of bytes sent during the shared state sync phase
        [[nodiscard]] size_t getSharedStateSyncTxBytes() const;

        /// Called to track additional bytes sent during the shared state sync phase
        void trackSharedStateTxBytes(size_t tx_bytes);

        /// Resets the number of bytes sent during the shared state sync phase
        void resetSharedStateSyncTxBytes();

        // TODO: THIS IS SUBJECT TO CHANGE, FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        void updateTopology(const std::vector<ccoip_uuid_t> &new_ring_order);

        // TODO: THIS IS SUBJECT TO CHANGE, FOR NOW ASSERT THE TOPOLOGY TO BE A SIMPLE RING REDUCE WITH A NON-PIPELINED ORDER
        [[nodiscard]] const std::vector<ccoip_uuid_t> &getRingOrder() const {
            return ring_order;
        }
    };
};
