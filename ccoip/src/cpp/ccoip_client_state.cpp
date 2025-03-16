#include "ccoip_client_state.hpp"

#include <ccoip_inet_utils.hpp>
#include <pccl_log.hpp>
#include <thread_guard.hpp>

ccoip::CCoIPClientState::CCoIPClientState() {
    main_thread_id = std::this_thread::get_id();
    collective_coms_threadpool.startup();
}

ccoip::CCoIPClientState::~CCoIPClientState() {
    for (const std::unordered_set<uint64_t> remaining_tags = running_collective_coms_ops_tags;
         const auto &tag: remaining_tags) {
        if (!joinAsyncCollectiveOp(tag)) {
            LOG(ERR) << "Failed to join collective comms op with tag " << tag << " during client state destruction";
        }
    }
    collective_coms_threadpool.shutdown();
}

void ccoip::CCoIPClientState::setMainThread(const std::thread::id main_thread_id) {
    this->main_thread_id = main_thread_id;
}

std::vector<uint64_t> ccoip::CCoIPClientState::getRunningCollectiveComsOpTags() {
    std::vector<uint64_t> running_collective_coms_op_tags; {
        std::shared_lock lock(running_collective_coms_ops_tags_mutex);
        running_collective_coms_op_tags.reserve(running_collective_coms_ops_tags.size());
        for (const auto tag: running_collective_coms_ops_tags) {
            running_collective_coms_op_tags.push_back(tag);
        }
    }
    return running_collective_coms_op_tags;
}

bool ccoip::CCoIPClientState::registerPeer(const ccoip_socket_address_t &address, const ccoip_uuid_t uuid) {
    THREAD_GUARD(main_thread_id);
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto it = socket_addr_to_uuid.find(internal_address); it != socket_addr_to_uuid.end()) {
        return false;
    }
    socket_addr_to_uuid[internal_address] = uuid;
    return true;
}

bool ccoip::CCoIPClientState::unregisterPeer(const ccoip_socket_address_t &address) {
    THREAD_GUARD(main_thread_id);
    const auto internal_address = ccoip_socket_to_internal(address);
    if (const auto n = socket_addr_to_uuid.erase(internal_address); n == 0) {
        return false;
    }
    return true;
}

void ccoip::CCoIPClientState::setAssignedUUID(const ccoip_uuid_t &new_assigned_uuid) {
    THREAD_GUARD(main_thread_id);
    if (assigned_uuid.data != ccoip_uuid{}) {
        throw std::runtime_error("Can only set assigned UUID once!");
    }
    assigned_uuid = new_assigned_uuid;
}

const ccoip_uuid_t &ccoip::CCoIPClientState::getAssignedUUID() const {
    // may be called concurrently, but is only set once during initialization
    return assigned_uuid;
}

void ccoip::CCoIPClientState::beginSyncSharedStatePhase(const ccoip_shared_state_t &shared_state) {
    THREAD_GUARD(main_thread_id);

    std::unique_lock lock(current_shared_state_mutex);
    current_shared_state = shared_state;
    is_syncing_shared_state.store(true, std::memory_order_release);
}

void ccoip::CCoIPClientState::endSyncSharedStatePhase() {
    THREAD_GUARD(main_thread_id);

    std::unique_lock lock(current_shared_state_mutex);
    current_shared_state = {};
    is_syncing_shared_state.store(false, std::memory_order_release);
}

bool ccoip::CCoIPClientState::isSyncingSharedState() const {
    return is_syncing_shared_state.load(std::memory_order_acquire);
}

bool ccoip::CCoIPClientState::isCollectiveComsOpRunning(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(running_collective_coms_ops_tags_mutex);
    return running_collective_coms_ops_tags.contains(tag);
}

bool ccoip::CCoIPClientState::isAnyCollectiveComsOpRunning() {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(running_collective_coms_ops_tags_mutex);
    return !running_collective_coms_ops_tags.empty();
}

size_t ccoip::CCoIPClientState::getWorldSize() const {
    THREAD_GUARD(main_thread_id);
    return ring_order.size();
}

bool ccoip::CCoIPClientState::startCollectiveComsOp(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    if (running_collective_coms_ops_tags.contains(tag)) {
        return false;
    }
    running_collective_coms_ops_tags.insert(tag);
    return true;
}

bool ccoip::CCoIPClientState::endCollectiveComsOp(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    if (const auto n = running_collective_coms_ops_tags.erase(tag); n == 0) {
        return false;
    }
    if (const auto n = running_collective_ops.erase(tag); n == 0) {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientState::launchAsyncCollectiveOp(const uint64_t tag,
                                                      std::function<void(std::promise<bool> &)> &&task) {
    THREAD_GUARD(main_thread_id);
    if (!startCollectiveComsOp(tag)) [[unlikely]] {
        return false;
    }
    resetCollectiveComsTxBytes(tag);
    resetCollectiveComsRxBytes(tag);

    // retain applicable world size at the time of launch
    {
        std::unique_lock lock(running_collective_coms_ops_world_size_mutex);
        running_collective_coms_ops_world_size[tag] = getWorldSize();
    }

    // set initial failure state to not completed
    {
        std::unique_lock lock(running_reduce_tasks_failure_states_mutex);
        running_reduce_tasks_failure_states[tag].store(2, std::memory_order_relaxed); // not completed
    }

    auto op_future = collective_coms_threadpool.scheduleTask([this, tag, task = std::move(task)] {
        std::promise<bool> promise{};
        task(promise);
        std::future<bool> future = promise.get_future();

        // expose the result of the task to the client state
        {
            std::unique_lock lock(running_reduce_tasks_failure_states_mutex);
            running_reduce_tasks_failure_states[tag].store(future.get() ? 1 : 0, std::memory_order_release);
        }
    });
    running_collective_ops.emplace(tag, std::move(op_future));
    return true;
}

bool ccoip::CCoIPClientState::joinAsyncCollectiveOp(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    if (const auto it = running_collective_ops.find(tag); it != running_collective_ops.end()) {
        it->second.join();
        if (!endCollectiveComsOp(tag)) [[unlikely]] {
            LOG(BUG) << "Collective comms op with tag " << tag << " was not started but is being ended";
        }
        return true;
    }
    return false;
}

std::optional<bool> ccoip::CCoIPClientState::hasCollectiveComsOpFailed(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(running_reduce_tasks_failure_states_mutex);
    if (const auto it = running_reduce_tasks_failure_states.find(tag);
        it != running_reduce_tasks_failure_states.end()) {
        const auto &state = it->second;
        return state.load() == 0;
    }
    return std::nullopt;
}

const ccoip_shared_state_t &ccoip::CCoIPClientState::getCurrentSharedState() {
    std::shared_lock lock(current_shared_state_mutex);
    return current_shared_state;
}


size_t ccoip::CCoIPClientState::getSharedStateSyncTxBytes() const {
    THREAD_GUARD(main_thread_id);
    return shared_state_sync_tx_bytes;
}

void ccoip::CCoIPClientState::trackSharedStateTxBytes(const size_t tx_bytes) {
    shared_state_sync_tx_bytes.fetch_add(tx_bytes, std::memory_order_relaxed);
}

void ccoip::CCoIPClientState::resetSharedStateSyncTxBytes() {
    THREAD_GUARD(main_thread_id);
    shared_state_sync_tx_bytes.store(0, std::memory_order_release);
}

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsRxBytes(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(collective_coms_rx_bytes_mutex);
    const auto it = collective_coms_rx_bytes.find(tag);
    if (it == collective_coms_rx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::setCollectiveConnectionRevision(const uint64_t tag, const uint64_t revision) {
    THREAD_GUARD(main_thread_id);
    std::unique_lock lock(collective_coms_connection_revisions_mutex);
    collective_coms_connection_revisions[tag] = revision;
}

std::optional<uint64_t> ccoip::CCoIPClientState::getCollectiveConnectionRevision(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(collective_coms_connection_revisions_mutex);
    const auto it = collective_coms_connection_revisions.find(tag);
    if (it == collective_coms_connection_revisions.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::trackCollectiveComsRxBytes(const uint64_t tag, const size_t rx_bytes) {
    std::unique_lock lock(collective_coms_rx_bytes_mutex);
    collective_coms_rx_bytes[tag].fetch_add(rx_bytes, std::memory_order_relaxed);
}

void ccoip::CCoIPClientState::resetCollectiveComsRxBytes(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::unique_lock lock(collective_coms_rx_bytes_mutex);
    collective_coms_rx_bytes.erase(tag);
}

std::optional<size_t> ccoip::CCoIPClientState::getCollectiveComsTxBytes(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(collective_coms_tx_bytes_mutex);
    const auto it = collective_coms_tx_bytes.find(tag);
    if (it == collective_coms_tx_bytes.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<uint32_t> ccoip::CCoIPClientState::getCollectiveComsWorldSize(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::shared_lock lock(running_collective_coms_ops_world_size_mutex);
    const auto it = running_collective_coms_ops_world_size.find(tag);
    if (it == running_collective_coms_ops_world_size.end()) {
        return std::nullopt;
    }
    return it->second;
}

void ccoip::CCoIPClientState::resetCollectiveComsWorldSize(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::unique_lock lock(running_collective_coms_ops_world_size_mutex);
    running_collective_coms_ops_world_size.erase(tag);
}

void ccoip::CCoIPClientState::trackCollectiveComsTxBytes(const uint64_t tag, const size_t tx_bytes) {
    std::unique_lock lock(collective_coms_tx_bytes_mutex);
    collective_coms_tx_bytes[tag] += tx_bytes;
}

void ccoip::CCoIPClientState::resetCollectiveComsTxBytes(const uint64_t tag) {
    THREAD_GUARD(main_thread_id);
    std::unique_lock lock(collective_coms_tx_bytes_mutex);
    collective_coms_tx_bytes.erase(tag);
}

void ccoip::CCoIPClientState::updateTopology(const std::vector<ccoip_uuid_t> &new_ring_order) {
    ring_order = new_ring_order;
}
