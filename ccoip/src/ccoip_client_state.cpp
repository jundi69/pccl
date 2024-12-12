#include "ccoip_client_state.hpp"

#include <ccoip_inet_utils.hpp>

bool ccoip::CCoIPClientState::registerPeer(const ccoip_socket_address_t &address, const ccoip_uuid_t uuid) {
    const auto internal_address = ccoip_socket_to_internal(address);
    if (!inet_addrs_to_uuids.contains(internal_address)) {
        inet_addrs_to_uuids[internal_address] = std::unordered_set<ccoip_uuid_t>();
    }
    inet_addrs_to_uuids[internal_address].insert(uuid);
    return true;
}
