#include "ccoip_master_state.hpp"

#include <ccoip_packets.hpp>
#include <pccl_log.hpp>

bool ccoip::CCoIPMasterState::registerClient(const ccoip_socket_address_t &client_address,
                                             const uint16_t p2p_listen_port,
                                             const ccoip_uuid_t uuid) {
    if (isClientRegistered(client_address)) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " already registered";
        return false;
    }

    const auto internal_address = ccoip_socket_to_internal(client_address);
    client_uuids[internal_address] = uuid;
    uuid_clients[uuid] = internal_address;
    client_info[uuid] = ClientInfo{
        .client_uuid = uuid,
        .connection_phase = PEER_REGISTERED,
        .socket_address = client_address,
        .p2p_listen_port = p2p_listen_port
    };
    peer_list_changed = true;

    // if this is the first client, consider it as voting to accept new peers
    if (client_uuids.size() == 1) {
        client_info[uuid].connection_phase = PEER_ACCEPTED; // consider it accepted
        if (!voteAcceptNewPeers(client_address)) [[unlikely]] {
            LOG(WARN) << "Failed to vote to accept new peers for first client " <<
                    ccoip_sockaddr_to_str(client_address);
            return false;
        }
        if (!acceptNewPeersConsensus()) [[unlikely]] {
            LOG(BUG) <<
                    "Inconsistent state: the first and only client voted to accept new peers, but not all clients have voted";
            return false;
        }
        transitionToP2PEstablishmentPhase();
    }
    return true;
}

bool ccoip::CCoIPMasterState::unregisterClient(const ccoip_socket_address_t &client_address) {
    if (!isClientRegistered(client_address)) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not registered";
        return false;
    }
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        if (!uuid_clients.erase(it->second)) {
            LOG(BUG) << "Client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->sockaddr mapping. This means bi-directional mapping for client UUIDs is inconsistent";
            return false;
        }
        if (!client_info.erase(it->second)) {
            LOG(BUG) << "ClientInfo of client with UUID " << uuid_to_string(it->second) <<
                    " not found in uuid->ClientInfo mapping. This means client info mapping is inconsistent";
            return false;
        }

        // remove from all voting sets
        votes_accept_new_peers.erase(it->second);
        client_uuids.erase(it);

        peer_list_changed = true;
    } else {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
    }
    return true;
}

bool ccoip::CCoIPMasterState::isClientRegistered(const ccoip_socket_address_t &client_address) const {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    return client_uuids.contains(internal_address);
}

bool ccoip::CCoIPMasterState::voteAcceptNewPeers(const ccoip_socket_address_t &client_address) {
    const auto info_opt = getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " <<
                ccoip_sockaddr_to_str(client_address);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to accept new peers
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " cannot vote to accept new peers in phase "
                << info.connection_phase;
        return false;
    }
    // in order to vote to accept new peers, the client must be idle
    if (info.connection_state != IDLE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " cannot vote to accept new peers in state "
                << info.connection_state;
        return false;
    }

    // set the client state to vote to accept new peers
    info.connection_state = VOTE_ACCEPT_NEW_PEERS;
    if (auto [_, inserted] = votes_accept_new_peers.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " found in votes_accept_new_peers set, but was in IDLE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteSyncSharedState(const ccoip_socket_address_t &client_address) {
    const auto info_opt = getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to sync shared state for unregistered client " <<
                ccoip_sockaddr_to_str(client_address);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to sync shared state
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " cannot vote to sync shared state in phase "
                << info.connection_phase;
        return false;
    }

    // in order to vote to sync shared state, the client must be idle
    if (info.connection_state != IDLE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " cannot vote to sync shared state in state "
                << info.connection_state;
        return false;
    }

    // set the client state to vote to sync shared state
    info.connection_state = VOTE_SYNC_SHARED_STATE;
    if (auto [_, inserted] = votes_sync_shared_state.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " found in votes_sync_shared_state set, but was in IDLE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::voteDistSharedStateComplete(const ccoip_socket_address_t &client_address) {
    const auto info_opt = getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to sync shared state for unregistered client " <<
                ccoip_sockaddr_to_str(client_address);
        return false;
    }
    auto &info = info_opt->get();

    // if the client is not yet accepted, it cannot vote to complete shared state distribution
    if (info.connection_phase != PEER_ACCEPTED) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " cannot vote to distribute shared state in phase " << info.connection_phase;
        return false;
    }

    // in order to vote to distribute shared state, the client must be in either
    // the DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE state
    if (info.connection_state != DISTRIBUTE_SHARED_STATE && info.connection_state != REQUEST_SHARED_STATE) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " cannot vote to distribute shared state in state " << info.connection_state;
        return false;
    }

    // set the client state to vote to distribute shared state
    info.connection_state = VOTE_COMPLETE_SHARED_STATE_SYNC;
    if (auto [_, inserted] = votes_sync_shared_state_complete.insert(info.client_uuid); !inserted) {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " found in votes_dist_shared_state_complete set, but was in DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE state before voting. This is a bug";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::markP2PConnectionsEstablished(const ccoip_socket_address_t &client_address) {
    const auto info_opt = getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Cannot vote to accept new peers for unregistered client " <<
                ccoip_sockaddr_to_str(client_address);
        return false;
    }
    auto &info = info_opt->get();
    if (info.connection_state != CONNECTING_TO_PEERS) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " cannot mark p2p connections established in state " << info.connection_state;
        return false;
    }
    info.connection_state = WAITING_FOR_OTHER_PEERS;
    if (auto [_, inserted] = votes_p2p_connections_established.insert(info.client_uuid); !inserted) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " found in votes_p2p_connections_established set, but was in CONNECTING_TO_PEERS state before voting";
        return false;
    }
    return true;
}

bool ccoip::CCoIPMasterState::transitionToP2PConnectionsEstablishedPhase() {
    for (auto &[_, info]: client_info) {
        if (info.connection_state == WAITING_FOR_OTHER_PEERS) {
            info.connection_state = IDLE;
            if (info.connection_phase == PEER_REGISTERED) {
                info.connection_phase = PEER_ACCEPTED; // update connection phase to PEER_ACCEPTED
            }
        } else {
            LOG(WARN) << "Client " << uuid_to_string(info.client_uuid) <<
                    " in state " << info.connection_state << " but expected WAITING_FOR_OTHER_PEERS";
            return false;
        }
    }
    // all clients have established p2p connections
    votes_p2p_connections_established.clear();
    return true;
}

bool ccoip::CCoIPMasterState::endSharedStateSyncPhase() {
    for (auto &[_, info]: client_info) {
        if (info.connection_phase != PEER_ACCEPTED) {
            continue;
        }
        if (info.connection_state == VOTE_COMPLETE_SHARED_STATE_SYNC) {
            info.connection_state = IDLE;
        } else {
            LOG(WARN) << "Client " << uuid_to_string(info.client_uuid) <<
                    " in state " << info.connection_state <<
                    " while terminating shared state distribution phase.";
            return false;
        }
    }
    shared_state_dirty_keys.clear();
    votes_sync_shared_state_complete.clear();
    return true;
}

bool ccoip::CCoIPMasterState::p2pConnectionsEstablishConsensus() const {
    return votes_p2p_connections_established.size() == client_uuids.size();
}

bool ccoip::CCoIPMasterState::syncSharedStateCompleteConsensus() const {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_COMPLETE_SHARED_STATE_SYNC) {
            voting_clients++;
        }
        if (info.connection_phase == PEER_ACCEPTED) {
            n_accepted_peers++;
        }
    }
    if (voting_clients != votes_sync_shared_state_complete.size()) {
        LOG(BUG) <<
                "Mismatch in number of clients voting to sync shared state between client_info and votes_sync_shared_state_complete";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

bool ccoip::CCoIPMasterState::syncSharedStateConsensus() const {
    size_t voting_clients = 0;
    size_t n_accepted_peers = 0;
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_SYNC_SHARED_STATE) {
            voting_clients++;
        }
        if (info.connection_phase == PEER_ACCEPTED) {
            n_accepted_peers++;
        }
    }
    if (voting_clients != votes_sync_shared_state.size()) {
        LOG(BUG) <<
                "Mismatch in number of clients voting to sync shared state between client_info and votes_sync_shared_state";
        return false;
    }
    return voting_clients == n_accepted_peers;
}

void ccoip::CCoIPMasterState::transitionToP2PEstablishmentPhase() {
    // set connection state of all peers to connecting to peers
    for (auto &[_, info]: client_info) {
        info.connection_state = CONNECTING_TO_PEERS;
    }

    // clear all votes to accept new peers
    votes_accept_new_peers.clear();
}

std::vector<ccoip::ClientInfo>
ccoip::CCoIPMasterState::getPeersForClient(const ccoip_socket_address_t &client_address) const {
    // get the UUID of the client
    const auto client_uuid = client_uuids.at(ccoip_socket_to_internal(client_address));

    std::vector<ClientInfo> peers{};

    // for now, simply return all clients except the client itself
    // TODO: in the future, this should be based on the topology
    for (const auto &[uuid, info]: client_info) {
        if (uuid != client_uuid) {
            peers.push_back(info);
        }
    }
    return peers;
}

std::vector<ccoip_socket_address_t> ccoip::CCoIPMasterState::getClientSocketAddresses() {
    std::vector<ccoip_socket_address_t> addresses{};
    addresses.reserve(client_uuids.size());
    for (const auto &[internal_address, _]: client_uuids) {
        addresses.push_back(internal_to_ccoip_sockaddr(internal_address));
    }
    return addresses;
}

std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > ccoip::CCoIPMasterState::getClientEntrySet() {
    std::vector<std::pair<ccoip_uuid_t, ccoip_socket_address_t> > entries{};
    entries.reserve(client_uuids.size());
    for (const auto &[internal_address, uuid]: client_uuids) {
        entries.emplace_back(uuid, internal_to_ccoip_sockaddr(internal_address));
    }
    return entries;
}

bool ccoip::CCoIPMasterState::hasPeerListChanged() {
    const bool changed = peer_list_changed;
    peer_list_changed = false;
    return changed;
}

ccoip::CCoIPMasterState::SharedStateMismatchStatus ccoip::CCoIPMasterState::sharedStateMatches(
    const ccoip_uuid_t &peer_uuid, const uint64_t revision,
    const std::vector<SharedStateHashEntry> &entries
) {
    SharedStateMismatchStatus status = SUCCESSFUL_MATCH;

    if (revision == shared_state_revision + 1) {
        shared_state_revision = revision;
    } else if (revision != shared_state_revision) {
        status = REVISION_MISMATCH;
        goto end;
    }

    if (shared_state_mask.empty()) {
        shared_state_mask = entries;
        status = SUCCESSFUL_MATCH;
        goto end;
    }
    if (shared_state_mask.size() != entries.size()) {
        status = KEY_SET_MISMATCH;
        goto end;
    }

    // compare the shared state mask with the supplied shared state entries
    {
        std::vector<std::string> dirty_content_keys{};
        for (size_t i = 0; i < shared_state_mask.size(); i++) {
            const auto &mask_entry = shared_state_mask[i];
            if (mask_entry.key != entries[i].key) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
            if (mask_entry.hash != entries[i].hash) {
                status = CONTENT_HASH_MISMATCH;
                dirty_content_keys.push_back(mask_entry.key);
            }
            if (mask_entry.allow_content_inequality != entries[i].allow_content_inequality) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
            if (mask_entry.data_type != entries[i].data_type) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
            if (mask_entry.num_elements != entries[i].num_elements) {
                status = KEY_SET_MISMATCH;
                goto end;
            }
        }
        if (!dirty_content_keys.empty()) {
            shared_state_dirty_keys[peer_uuid] = dirty_content_keys;
        }
    }

end:
    shared_state_responses[peer_uuid] = status;
    return status;
}

std::optional<ccoip::CCoIPMasterState::SharedStateMismatchStatus> ccoip::CCoIPMasterState::getSharedStateMismatchStatus(
    const ccoip_uuid_t &peer_uuid) {
    if (const auto it = shared_state_responses.find(peer_uuid); it != shared_state_responses.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ccoip::CCoIPMasterState::transitionToSharedStateSyncPhase() {
    // all clients should
    for (auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_SYNC_SHARED_STATE) {
            const auto status_opt = getSharedStateMismatchStatus(info.client_uuid);
            if (!status_opt) [[unlikely]] {
                LOG(WARN) << "Client " << uuid_to_string(info.client_uuid) <<
                        " in state VOTE_SYNC_SHARED_STATE but no shared state mismatch status found";
                return false;
            }
            const auto status = *status_opt;
            if (status == KEY_SET_MISMATCH) {
                LOG(WARN) << "Client " << uuid_to_string(info.client_uuid) <<
                        " is in state KEY_SET_MISMATCH after shared state voting phase ended!";
                return false;
            }
            info.connection_state = status == SUCCESSFUL_MATCH ? DISTRIBUTE_SHARED_STATE : REQUEST_SHARED_STATE;
        } else if (info.connection_phase == PEER_ACCEPTED) {
            LOG(WARN) << "Client " << uuid_to_string(info.client_uuid) <<
                    " in phase PEER_ACCEPTED but not in state VOTE_SYNC_SHARED_STATE after shared state voting phase ended";
            return false;
        }
    }
    shared_state_mask.clear();
    shared_state_responses.clear();
    votes_sync_shared_state.clear();
    return true;
}

std::optional<ccoip_uuid_t> ccoip::CCoIPMasterState::findClientUUID(const ccoip_socket_address_t &client_address) {
    if (const auto it = client_uuids.find(ccoip_socket_to_internal(client_address)); it != client_uuids.end()) {
        return it->second;
    }
    return std::nullopt;
}

bool ccoip::CCoIPMasterState::acceptNewPeersConsensus() const {
    std::unordered_set<ccoip_uuid_t> voting_peers{};
    for (const auto &[_, info]: client_info) {
        if (info.connection_state == VOTE_ACCEPT_NEW_PEERS || info.connection_phase == PEER_REGISTERED) {
            // if the client is in the PEER_REGISTERED state, it is not yet accepted
            // by the other peers, so it cannot vote to accept new peers.
            // However, to reach the total of all clients, we need to include it in the count.
            voting_peers.insert(info.client_uuid);
        }
    }
    return voting_peers.size() == client_uuids.size();
}

std::optional<std::reference_wrapper<ccoip::ClientInfo> > ccoip::CCoIPMasterState::getClientInfo(
    const ccoip_socket_address_t &client_address) {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        return client_info[it->second];
    }
    return std::nullopt;
}

uint64_t ccoip::CCoIPMasterState::getSharedStateRevision() const {
    return shared_state_revision;
}

const std::vector<std::string> &ccoip::CCoIPMasterState::getOutdatedSharedStateKeys(const ccoip_uuid_t peer_uuid) {
    return shared_state_dirty_keys[peer_uuid];
}
