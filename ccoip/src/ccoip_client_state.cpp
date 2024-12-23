#include "ccoip_client_state.hpp"

#include <ccoip_inet_utils.hpp>

bool ccoip::CCoIPClientState::registerPeer(const ccoip_socket_address_t &address, const ccoip_uuid_t uuid) {
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto it = socket_addr_to_uuid.find(internal_address); it != socket_addr_to_uuid.end()) {
        return false;
    }
    socket_addr_to_uuid[internal_address] = uuid;
    return true;
}

bool ccoip::CCoIPClientState::unregisterPeer(const ccoip_socket_address_t &address) {
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto n = socket_addr_to_uuid.erase(internal_address); n == 0) {
        return false;
    }
    return true;
}

void ccoip::CCoIPClientState::beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state) {
    current_shared_state = shared_state;
    is_syncing_shared_state = true;
}

void ccoip::CCoIPClientState::endSyncSharedStatePhase() {
    current_shared_state = {};
    is_syncing_shared_state = false;
}

bool ccoip::CCoIPClientState::isSyncingSharedState() const {
    return is_syncing_shared_state;
}

bool ccoip::CCoIPClientState::isCollectiveComsOpRunning(const uint64_t tag) const {
    return running_collective_coms_ops_tags.contains(tag);
}

bool ccoip::CCoIPClientState::isAnyCollectiveComsOpRunning() const {
    return !running_collective_coms_ops_tags.empty();
}

bool ccoip::CCoIPClientState::startCollectiveComsOp(const uint64_t tag) {
    if (running_collective_coms_ops_tags.contains(tag)) {
        return false;
    }
    running_collective_coms_ops_tags.insert(tag);
    return true;
}

bool ccoip::CCoIPClientState::endCollectiveComsOp(const uint64_t tag) {
    if (const auto n = running_collective_coms_ops_tags.erase(tag); n == 0) {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientState::launchAsyncCollectiveOp(const uint64_t tag,
                                                      std::function<void(std::promise<bool> &)> &&task) {
    running_reduce_tasks[tag] = std::move(std::thread([this, tag, task = std::move(task)]() {
        std::promise<bool> &promise = running_reduce_tasks_promises[tag];
        task(promise);
    }));
    return true;
}

bool ccoip::CCoIPClientState::joinAsyncReduce(const uint64_t tag) {
    if (const auto it = running_reduce_tasks.find(tag); it != running_reduce_tasks.end()) {
        it->second.join();
        running_reduce_tasks.erase(it);
        return true;
    }
    return false;
}

std::optional<bool> ccoip::CCoIPClientState::hasCollectiveComsOpFailed(const uint64_t tag) {
    if (const auto it = running_reduce_tasks_promises.find(tag); it != running_reduce_tasks_promises.end()) {
        auto &promise = it->second;
        auto future = promise.get_future();
        if (future.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
            const bool success = future.get();
            return success == false;
        }
    }
    return std::nullopt;
}


size_t ccoip::CCoIPClientState::getSharedStateSyncTxBytes() const {
    return shared_state_sync_tx_bytes;
}

void ccoip::CCoIPClientState::trackSharedStateTxBytes(const size_t tx_bytes) {
    shared_state_sync_tx_bytes += tx_bytes;
}

void ccoip::CCoIPClientState::resetSharedStateSyncTxBytes() {
    shared_state_sync_tx_bytes = 0;
}
