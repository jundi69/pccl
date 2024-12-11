#include "ccoip_master_state.hpp"

#include <pccl_log.hpp>

void CCoIPMasterState::registerClient(const ccoip_socket_address_t &client_address, ccoip_uuid_t uuid) {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    client_uuids[internal_address] = uuid;
    uuid_clients[uuid] = internal_address;
    client_info[uuid] = ClientInfo{
        .connection_state = PEER_REGISTERED
    };
}

void CCoIPMasterState::unregisterClient(const ccoip_socket_address_t &client_address) {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        if (!uuid_clients.erase(it->second)) {
            LOG(WARN) << "Client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->sockaddr mapping. This means bi-directional mapping for client UUIDs is inconsistent";
        }
        if (!client_info.erase(it->second)) {
            LOG(WARN) << "ClientInfo of client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->ClientInfo mapping. This means client info mapping is inconsistent";
        }
        client_uuids.erase(it);
    } else {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
    }
}

bool CCoIPMasterState::isClientRegistered(const ccoip_socket_address_t &client_address) const {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    return client_uuids.contains(internal_address);
}
