#include "ccoip_client_state.hpp"

#include <ccoip_inet_utils.hpp>
#include <pccl_log.hpp>

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

void ccoip::CCoIPClientState::setAssignedUUID(const ccoip_uuid_t &new_assigned_uuid) {
    assigned_uuid = new_assigned_uuid;
}

const ccoip_uuid_t &ccoip::CCoIPClientState::getAssignedUUID() const {
    return assigned_uuid;
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
    if (!startCollectiveComsOp(tag)) [[unlikely]] {
        return false;
    }
    resetCollectiveComsTxBytes(tag);
    resetCollectiveComsRxBytes(tag);
    running_collective_coms_ops_world_size[tag] = getWorldSize(); // retain applicable world size at the time of launch
    running_reduce_tasks_promises[tag] = std::promise<bool>{};
    running_reduce_tasks[tag] = std::move(std::thread([this, tag, task = std::move(task)] {
        std::promise<bool> &promise = running_reduce_tasks_promises.at(tag);
        task(promise);
        if (!endCollectiveComsOp(tag)) [[unlikely]] {
            LOG(BUG) << "Collective comms op with tag " << tag << " was not started but is being ended";
        }
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

const ccoip_shared_state_t &ccoip::CCoIPClientState::getCurrentSharedState() const {
    return current_shared_state;
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

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsRxBytes(const uint64_t tag) const {
    const auto it = collective_coms_rx_bytes.find(tag);
    if (it == collective_coms_rx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::trackCollectiveComsRxBytes(const uint64_t tag, const size_t rx_bytes) {
    collective_coms_rx_bytes[tag] += rx_bytes;
}

void ccoip::CCoIPClientState::resetCollectiveComsRxBytes(const uint64_t tag) {
    collective_coms_rx_bytes.erase(tag);
}

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsTxBytes(const uint64_t tag) const {
    const auto it = collective_coms_tx_bytes.find(tag);
    if (it == collective_coms_tx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<uint32_t> ccoip::CCoIPClientState::getCollectiveComsWorldSize(const uint64_t tag) const {
    const auto it = running_collective_coms_ops_world_size.find(tag);
    if (it == running_collective_coms_ops_world_size.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::resetCollectiveComsWorldSize(const uint64_t tag) {
    running_collective_coms_ops_world_size.erase(tag);
}

void ccoip::CCoIPClientState::trackCollectiveComsTxBytes(const uint64_t tag, const size_t tx_bytes) {
    collective_coms_tx_bytes[tag] += tx_bytes;
}

void ccoip::CCoIPClientState::resetCollectiveComsTxBytes(const uint64_t tag) {
    collective_coms_tx_bytes.erase(tag);
}

void ccoip::CCoIPClientState::updateTopology(const std::vector<ccoip_uuid_t> &new_ring_order) {
    ring_order = new_ring_order;
}
