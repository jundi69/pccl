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

size_t ccoip::CCoIPClientState::getSharedStateSyncTxBytes() const {
    return shared_state_sync_tx_bytes;
}

void ccoip::CCoIPClientState::trackSharedStateTxBytes(const size_t tx_bytes) {
    shared_state_sync_tx_bytes += tx_bytes;
}

void ccoip::CCoIPClientState::resetSharedStateSyncTxBytes() {
    shared_state_sync_tx_bytes = 0;
}
