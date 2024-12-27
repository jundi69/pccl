#include "ccoip_client_handler.hpp"
#include "ccoip_types.hpp"
#include <pccl_log.hpp>
#include <ccoip.h>
#include <future>
#include <hash_utils.hpp>

#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"
#include <thread_guard.hpp>
#include <guard_utils.hpp>
#include <reduce_kernels.hpp>

ccoip::CCoIPClientHandler::CCoIPClientHandler(const ccoip_socket_address_t &address,
                                              const uint32_t peer_group) : master_socket(address),
                                                                     // Both p2p_socket and shared_state_socket listen to the first free port above the specified port number
                                                                     // as the constructor with inet_addr and above_port is called, which will bump on failure to bind.
                                                                     // this is by design and the chosen ports will be communicated to the master, which will then distribute
                                                                     // this information to clients to then correctly establish connections. The protocol does not assert these
                                                                     // ports to be static; The only asserted static port is the master listening port.
                                                                     p2p_socket({address.inet.protocol, {}, {}},
                                                                         CCOIP_PROTOCOL_PORT_P2P),
                                                                     shared_state_socket(
                                                                         {address.inet.protocol, {}, {}},
                                                                         CCOIP_PROTOCOL_PORT_SHARED_STATE),
                                                                     peer_group(peer_group) {
}

bool ccoip::CCoIPClientHandler::connect() {
    if (connected) {
        LOG(WARN) << "CCoIPClientHandler::connect() called while already connected";
        return false;
    }

    // start listening on p2p socket
    {
        p2p_socket.setJoinCallback([this](const ccoip_socket_address_t &client_address,
                                          std::unique_ptr<tinysockets::BlockingIOSocket> &socket) {
            const auto hello_packet_opt = socket->receivePacket<P2PPacketHello>();
            if (!hello_packet_opt) {
                LOG(ERR) << "Failed to receive P2PPacketHello from " << ccoip_sockaddr_to_str(client_address);
                if (!socket->closeConnection()) {
                    LOG(ERR) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address);
                }
                return;
            }
            const auto& hello_packet = hello_packet_opt.value();

            p2p_connections_rx[hello_packet.peer_uuid] = std::move(socket);
            if (!p2p_connections_rx[hello_packet.peer_uuid]->sendPacket<P2PPacketHelloAck>({})) {
                LOG(ERR) << "Failed to send P2PPacketHelloAck to " << ccoip_sockaddr_to_str(client_address);
                if (!socket->closeConnection()) {
                    LOG(ERR) << "Failed to close connection to " << ccoip_sockaddr_to_str(client_address);
                }
                return;
            }
            LOG(INFO) << "P2P connection established with " << ccoip_sockaddr_to_str(client_address);
        });

        if (!p2p_socket.listen()) {
            LOG(ERR) << "Failed to bind P2P socket " << p2p_socket.getListenPort();
            return false;
        }
        if (!p2p_socket.runAsync()) [[unlikely]] {
            return false;
        }
        LOG(INFO) << "P2P socket listening on port " << p2p_socket.getListenPort() << "...";
    }

    // start listening with shared state distribution server
    {
        shared_state_socket.addReadCallback(
            [this](const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data) {
                onSharedStateClientRead(client_address, data);
            });
        if (!shared_state_socket.listen()) {
            LOG(ERR) << "Failed to bind shared state socket " << shared_state_socket.getListenPort();
            return false;
        }
        if (!shared_state_socket.runAsync()) [[unlikely]] {
            return false;
        }
        shared_state_server_thread_id = shared_state_socket.getServerThreadId();
        LOG(INFO) << "Shared state socket listening on port " << shared_state_socket.getListenPort() << "...";
    }

    if (!master_socket.establishConnection()) {
        return false;
    }

    // send join request packet to master
    C2MPacketRequestSessionRegistration join_request{};
    join_request.p2p_listen_port = p2p_socket.getListenPort();
    join_request.shared_state_listen_port = shared_state_socket.getListenPort();
    join_request.peer_group = peer_group;

    if (!master_socket.sendPacket<C2MPacketRequestSessionRegistration>(join_request)) {
        return false;
    }

    // receive join response packet from master
    const auto response = master_socket.receivePacket<M2CPacketSessionRegistrationResponse>();
    if (!response) {
        return false;
    }
    if (!response->accepted) {
        LOG(ERR) << "Master rejected join request";
        return false;
    }
    client_state.setAssignedUUID(response->assigned_uuid);

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    connected = true;
    return true;
}

bool ccoip::CCoIPClientHandler::acceptNewPeers() {
    if (!connected) {
        LOG(WARN) <<
                "CCoIPClientHandler::acceptNewPeers() before CCoIPClientHandler::connect() was called. Establish master connection first before performing client actions.";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) <<
                "Failed to sync shared state: Client socket has been closed; This may mean the client was kicked by the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        return false;
    }

    if (!master_socket.sendPacket<C2MPacketAcceptNewPeers>({})) {
        return false;
    }

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::syncSharedState(ccoip_shared_state_t &shared_state,
                                                ccoip_shared_state_sync_info_t &info_out) {
    if (!connected) {
        LOG(WARN) <<
                "CCoIPClientHandler::syncSharedState() before CCoIPClientHandler::connect() was called. Establish master connection first before performing client actions.";
        return false;
    }

    if (!master_socket.isOpen()) {
        LOG(ERR) <<
                "Failed to sync shared state: Client socket has been closed; This may mean the client was kicked by the master";
        return false;
    }

    if (client_state.isAnyCollectiveComsOpRunning()) {
        return false;
    }

    // prepare shared state hashes
    std::vector<SharedStateHashEntry> shared_state_hashes{};
    shared_state_hashes.reserve(shared_state.entries.size());
    for (const auto &entry: shared_state.entries) {
        auto &key = entry.key;
        auto &value = entry.value;
        shared_state_hashes.push_back(SharedStateHashEntry{
            .key = key,
            .hash = entry.allow_content_inequality ? 0 : hash_utils::CRC32(value.data(), value.size_bytes()),
            .data_type = entry.data_type,
            .allow_content_inequality = entry.allow_content_inequality
        });
    }

    // inside this block, we are guaranteed in the shared state distribution phase
    {
        // As long as we are inside this block, client_state.isSyncingSharedState() will return true

        // ATTENTION: We create this guard even before we vote to enter the shared state sync state on the server side.
        // So on the client side this phase either means that we have already successfully voted to enter the shared state sync phase
        // or that we would want to be in the shared state sync phase.
        // This is necessary because we check this state of being in the shared state sync phase during shared state distribution.
        // We turn down requests that come in while we are not in the shared state sync phase.
        // However, if we were to start the shared state sync phase AFTER sending the shared state sync vote packet,
        // not all clients receive the confirmation packet from the master after a successful vote at the same time.
        // If you are unlucky, this discrepancy can be so bad that some clients even handle this packet and send a shared
        // state request to a shared state distributor that is not yet aware that the shared state sync phase has started.
        // It will then respond with state SHARED_STATE_NOT_DISTRIBUTED.
        // Because we are basically only using this phase state for this purpose anyway, this is a good solution to
        // avoid yet another vote that would only really solve a conceptual problem, not a problem introduced by data dependencies.
        guard_utils::phase_guard guard([this, &shared_state] { client_state.beginSyncSharedStatePhase(shared_state); },
                                       [this] { client_state.endSyncSharedStatePhase(); });

        // vote for shared state sync
        C2MPacketSyncSharedState packet{};
        packet.shared_state_revision = shared_state.revision;
        packet.shared_state_hashes = shared_state_hashes;
        if (!master_socket.sendPacket<C2MPacketSyncSharedState>(packet)) {
            LOG(ERR) << "Failed to sync shared state: Failed to send C2MPacketSyncSharedState to master";
            return false;
        }
        // wait for confirmation from master that all peers have voted to sync the shared state
        const auto response = master_socket.receivePacket<M2CPacketSyncSharedState>();
        if (!response) {
            LOG(ERR) << "Failed to sync shared state: Failed to receive M2CPacketSyncSharedState response from master";
            return false;
        }

        if (response->is_outdated) {
            // if shared state is outdated, request shared state from master
            tinysockets::BlockingIOSocket req_socket(response->distributor_address);
            if (!req_socket.establishConnection()) {
                LOG(ERR) <<
                        "Failed to sync shared state: Failed to establish connection with shared state distributor: " <<
                        ccoip_sockaddr_to_str(response->distributor_address);
                return false;
            }
            C2SPacketRequestSharedState request{};
            request.requested_keys.reserve(response->outdated_keys.size());
            for (const auto &key: response->outdated_keys) {
                request.requested_keys.push_back(key);
            }
            if (!req_socket.sendPacket<C2SPacketRequestSharedState>(request)) {
                LOG(ERR) << "Failed to sync shared state: Failed to send C2SPacketRequestSharedState to distributor";
                return false;
            }
            const auto shared_state_response = req_socket.receivePacket<S2CPacketSharedStateResponse>();
            if (!shared_state_response) {
                LOG(ERR) <<
                        "Failed to sync shared state: Failed to receive S2CPacketSharedStateResponse from distributor";
                return false;
            }
            if (shared_state_response->status != SUCCESS) {
                LOG(ERR) << "Failed to sync shared state: Shared state distributor returned status " <<
                        shared_state_response->status;
                return false;
            }
            // update shared state
            shared_state.revision = shared_state_response->revision;

            std::unordered_map<std::string, const SharedStateEntry *> new_entries{};
            for (const auto &entry: shared_state_response->entries) {
                new_entries[entry.key] = &entry;
            }

            // copy content buffers of shared state entries to user controlled buffers
            for (size_t i = 0; i < shared_state.entries.size(); i++) {
                const auto &entry = shared_state.entries[i];
                if (new_entries.contains(entry.key)) {
                    const auto &new_entry = new_entries[entry.key];
                    if (new_entry->dst_size != entry.value.size_bytes()) {
                        LOG(ERR) << "Failed to sync shared state: Shared state entry size mismatch for key " << entry.
                                key;
                        return false;
                    }
                    std::memcpy(entry.value.data(), new_entry->dst_buffer.get(), new_entry->dst_size);
                    info_out.rx_bytes += new_entry->dst_size;

                    if (i < response->expected_hashes.size()) {
                        uint64_t actual_hash = hash_utils::CRC32(entry.value.data(), entry.value.size_bytes());
                        if (uint64_t expected_hash = response->expected_hashes[i]; actual_hash != expected_hash) {
                            LOG(ERR) << "Shared state distributor transmitted incorrect shared state entry for key " <<
                                    entry.key << ": Expected hash " << expected_hash << " but got " << actual_hash;
                            return false;
                        }
                    } else {
                        LOG(WARN) << "Master did not transmit expected hash for shared state entry " << entry.key <<
                                "; Skipping hash check...";
                    }
                }
            }
        }
        // indicate to master that shared state distribution is complete
        if (!master_socket.sendPacket<C2MPacketDistSharedStateComplete>({})) {
            LOG(ERR) << "Failed to sync shared state: Failed to send C2MPacketSyncSharedStateComplete to master";
            return false;
        }

        // wait for confirmation from master that all peers have synced the shared state
        if (const auto sync_response = master_socket.receivePacket<M2CPacketSyncSharedStateComplete>(); !
            sync_response) {
            LOG(ERR) <<
                    "Failed to sync shared state: Failed to receive M2CPacketSyncSharedStateComplete response from master";
            return false;
        }
        info_out.tx_bytes = client_state.getSharedStateSyncTxBytes();
        client_state.resetSharedStateSyncTxBytes();
    }
    return true;
}

bool ccoip::CCoIPClientHandler::updateTopology() {
    if (!master_socket.sendPacket<C2MPacketGetTopologyRequest>({})) {
        return false;
    }
    const auto response = master_socket.receivePacket<M2CPacketGetTopologyResponse>();
    if (!response) {
        return false;
    }
    client_state.updateTopology(response->ring_reduce_order);
    return true;
}

bool ccoip::CCoIPClientHandler::interrupt() {
    if (interrupted) {
        return false;
    }
    if (!p2p_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!shared_state_socket.interrupt()) [[unlikely]] {
        return false;
    }
    if (!master_socket.closeConnection()) [[unlikely]] {
        return false;
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPClientHandler::join() {
    p2p_socket.join();
    shared_state_socket.join();
    return true;
}

bool ccoip::CCoIPClientHandler::isInterrupted() const {
    return interrupted;
}

ccoip::CCoIPClientHandler::~CCoIPClientHandler() = default;

bool ccoip::CCoIPClientHandler::establishP2PConnections() {
    // wait for new peers packet
    const auto new_peers = master_socket.receivePacket<M2CPacketNewPeers>();
    if (!new_peers) {
        LOG(ERR) << "Failed to receive new peers packet";
        return false;
    }
    LOG(DEBUG) << "Received M2CPacketNewPeers from master";
    if (!new_peers->unchanged) {
        // establish p2p connections
        for (auto &peer: new_peers->new_peers) {
            // check if connection already exists
            if (p2p_connections_tx.contains(peer.peer_uuid)) {
                continue;
            }
            if (!establishP2PConnection(peer)) {
                LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
                return false;
            }
        }
        // close p2p connections that are no longer needed
        for (auto it = p2p_connections_tx.begin(); it != p2p_connections_tx.end();) {
            if (std::ranges::find_if(new_peers->new_peers,
                                     [&it](const M2CPacketNewPeerInfo &peer) {
                                         return peer.peer_uuid == it->first;
                                     }) == new_peers->new_peers.end()) {
                if (!closeP2PConnection(it->first, *it->second)) {
                    LOG(WARN) << "Failed to close p2p connection with peer " << uuid_to_string(it->first);
                }
                it = p2p_connections_tx.erase(it);
            } else {
                ++it;
            }
        }
    }

    // send packet to this peer has established its p2p connections
    if (!master_socket.sendPacket<C2MPacketP2PConnectionsEstablished>({})) {
        LOG(ERR) << "Failed to send P2P connections established packet";
        return false;
    }

    // wait for response from master, indicating ALL peers have established their
    // respective p2p connections
    if (const auto response = master_socket.receivePacket<M2CPacketP2PConnectionsEstablished>(); !response) {
        LOG(ERR) << "Failed to receive P2P connections established response";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::establishP2PConnection(const M2CPacketNewPeerInfo &peer) {
    auto [it, inserted] = p2p_connections_tx.emplace(peer.peer_uuid, std::make_unique<tinysockets::BlockingIOSocket>(peer.p2p_listen_addr));
    if (!inserted) {
        LOG(ERR) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
        return false;
    }
    auto &connection = it->second;
    if (!connection->establishConnection()) {
        LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    P2PPacketHello hello_packet{};
    hello_packet.peer_uuid = client_state.getAssignedUUID();
    if (!connection->sendPacket<P2PPacketHello>(hello_packet)) {
        LOG(ERR) << "Failed to send hello packet to peer " << uuid_to_string(peer.peer_uuid);
    }
    if (const auto response = connection->receivePacket<P2PPacketHelloAck>(); !response) {
        LOG(ERR) << "Failed to receive hello ack from peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    if (!client_state.registerPeer(peer.p2p_listen_addr, peer.peer_uuid)) {
        LOG(ERR) << "Failed to register peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::closeP2PConnection(const ccoip_uuid_t &uuid, tinysockets::BlockingIOSocket &socket) {
    if (!socket.closeConnection()) [[unlikely]] {
        LOG(BUG) << "Failed to close connection with peer " << uuid_to_string(uuid);
        return false;
    }
    if (!client_state.unregisterPeer(socket.getConnectSockAddr())) [[unlikely]] {
        LOG(BUG) << "Failed to unregister peer " << uuid_to_string(uuid) <<
                ". This means the client was already unregistered; This is a bug!";
        return false;
    }
    return true;
}

void ccoip::CCoIPClientHandler::handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                                         const C2SPacketRequestSharedState &packet) {
    THREAD_GUARD(shared_state_server_thread_id);

    S2CPacketSharedStateResponse response{};

    // check if we are in shared state sync phase
    if (!client_state.isSyncingSharedState()) {
        LOG(WARN) << "Received shared state request from " << ccoip_sockaddr_to_str(client_address) <<
                " while not in shared state sync phase; Responding with status=SHARED_STATE_NOT_DISTRIBUTED";
        response.status = SHARED_STATE_NOT_DISTRIBUTED;
        goto end;
    }

    // construct packet referencing memory from the current shared state
    // for the set of keys requested by the client
    {
        response.status = SUCCESS;
        const auto &shared_state = client_state.getCurrentSharedState();
        for (const auto &requested_key: packet.requested_keys) {
            const auto it = std::ranges::find_if(shared_state.entries,
                                                 [&requested_key](const ccoip_shared_state_entry_t &entry) {
                                                     return entry.key == requested_key;
                                                 });
            if (it == shared_state.entries.end()) {
                response.status = UNKNOWN_SHARED_STATE_KEY;
                goto end;
            }
            response.entries.push_back(SharedStateEntry{
                .key = requested_key,
                .src_buffer = std::span(reinterpret_cast<uint8_t *>(it->value.data()), it->value.size_bytes())
            });
            client_state.trackSharedStateTxBytes(it->value.size_bytes());
        }
    }

end:
    if (!shared_state_socket.sendPacket(client_address, response)) {
        LOG(ERR) << "Failed to send shared state response to " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPClientHandler::onSharedStateClientRead(const ccoip_socket_address_t &client_address,
                                                        const std::span<std::uint8_t> &data) {
    THREAD_GUARD(shared_state_server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == C2SPacketRequestSharedState::packet_id) {
        C2SPacketRequestSharedState packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2SPacketRequestSharedState from " <<
                    ccoip_sockaddr_to_str(client_address);
            return;
        }
        handleSharedStateRequest(client_address, packet);
    } else {
        LOG(ERR) << "Unknown packet type " << packet_type << " from " << ccoip_sockaddr_to_str(client_address);
        if (!shared_state_socket.closeClientConnection(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to close connection with " << ccoip_sockaddr_to_str(client_address);
        }
    }
}


bool ccoip::CCoIPClientHandler::allReduceAsync(const void *sendbuff, void *recvbuff, const size_t count,
                                               const ccoip_data_type_t datatype, const ccoip_reduce_op_t op,
                                               const uint64_t tag) {
    if (client_state.isCollectiveComsOpRunning(tag)) {
        // can't start a new collective coms op while one is already running
        return false;
    }

    if (!client_state.launchAsyncCollectiveOp(
        tag, [this, sendbuff, recvbuff, count, datatype, op, tag](std::promise<bool> &promise) {
            // vote commence collective comms operation and await consensus
            {
                C2MPacketCollectiveCommsInitiate initiate_packet{};
                initiate_packet.tag = tag;
                initiate_packet.count = count;
                initiate_packet.data_type = datatype;
                initiate_packet.op = op;
                if (!master_socket.sendPacket<C2MPacketCollectiveCommsInitiate>(initiate_packet)) {
                    promise.set_value(false); // failure
                    return;
                }
                const auto response = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsCommence>(
                    [tag](const M2CPacketCollectiveCommsCommence &packet) {
                        return packet.tag == tag;
                    });

                if (!response) {
                    promise.set_value(false); // failure
                    return;
                }
            }

            // TODO: THIS IS SUBJECT TO CHANGE AND FOR NOW IS A HARDCODED UNOPTIMIZED NON-PIPELINED RING REDUCE.
            //  THIS ENTIRE SECTION WILL BE DELETED IN THE NEAR FUTURE
            const auto &ring_order = client_state.getRingOrder();
            const auto &client_uuid = client_state.getAssignedUUID();

            // find position in ring order
            const auto it = std::ranges::find(ring_order, client_uuid);
            if (it == ring_order.end()) {
                promise.set_value(false); // failure
                return;
            }
            const auto position = std::distance(ring_order.begin(), it);

            const size_t byte_size = count * ccoip_data_type_size(datatype);

            std::span send_span(static_cast<const std::byte *>(sendbuff), byte_size);
            std::span recv_span(static_cast<std::byte *>(recvbuff), byte_size);
            std::memcpy(recv_span.data(), send_span.data(), send_span.size());

            if (position == 0) {
                // is first
                // initiate ring reduce

                // find next peer
                const auto next_peer = ring_order[position + 1];
                const auto next_peer_socket_opt = p2p_connections_tx.find(next_peer);
                if (next_peer_socket_opt == p2p_connections_tx.end()) {
                    LOG(ERR) << "Failed to find p2p connection for next peer " << uuid_to_string(next_peer);
                    promise.set_value(false); // failure
                    return;
                }
                auto &next_peer_socket = next_peer_socket_opt->second;

                P2PPacketReduceTerm packet{};
                packet.tag = tag;
                packet.is_reduce = true;
                packet.data = send_span;
                if (!next_peer_socket->sendPacket<P2PPacketReduceTerm>(packet)) {
                    LOG(ERR) << "Failed to send reduce term to next peer " << uuid_to_string(next_peer);
                    promise.set_value(false); // failure
                    return;
                }
            } else {
                // is not first
                // wait for data from previous peer
                const auto prev_peer = ring_order[position - 1];
                const auto prev_peer_socket_opt = p2p_connections_rx.find(prev_peer);
                if (prev_peer_socket_opt == p2p_connections_rx.end()) {
                    LOG(ERR) << "Failed to find p2p connection for previous peer " << uuid_to_string(prev_peer);
                    promise.set_value(false); // failure
                    return;
                }
                auto &prev_peer_socket = prev_peer_socket_opt->second;
                const auto rx_packet = prev_peer_socket->receivePacket<P2PPacketReduceTerm>();
                if (!rx_packet) {
                    LOG(ERR) << "Failed to receive reduce term from previous peer " << uuid_to_string(prev_peer);
                    promise.set_value(false); // failure
                    return;
                }

                // perform reduction
                performReduction(recv_span, rx_packet->data, datatype, op);

                // if is not last, send data to next peer
                if (position < ring_order.size() - 1) {
                    // find next peer
                    const auto next_peer = ring_order[position + 1];
                    const auto next_peer_socket_opt = p2p_connections_tx.find(next_peer);
                    if (next_peer_socket_opt == p2p_connections_tx.end()) {
                        LOG(ERR) << "Failed to find p2p connection for next peer " << uuid_to_string(next_peer);
                        promise.set_value(false); // failure
                        return;
                    }
                    auto &next_peer_socket = next_peer_socket_opt->second;

                    P2PPacketReduceTerm tx_packet{};
                    tx_packet.tag = tag;
                    tx_packet.is_reduce = true;
                    tx_packet.data = recv_span;
                    if (!next_peer_socket->sendPacket<P2PPacketReduceTerm>(tx_packet)) {
                        LOG(ERR) << "Failed to send reduce term to next peer " << uuid_to_string(next_peer);
                        promise.set_value(false); // failure
                        return;
                    }
                }
            }

            if (position == ring_order.size() - 1) {
                // is last
                // distribute to other peers
                for (size_t i = 0; i < ring_order.size() - 1; i++) {
                    const auto peer = ring_order[i];
                    const auto peer_socket_opt = p2p_connections_tx.find(peer);
                    if (peer_socket_opt == p2p_connections_tx.end()) {
                        LOG(ERR) << "Failed to find p2p connection for peer " << uuid_to_string(peer);
                        promise.set_value(false); // failure
                        return;
                    }
                    auto &peer_socket = peer_socket_opt->second;
                    P2PPacketReduceTerm packet{};
                    packet.tag = tag;
                    packet.is_reduce = false;
                    packet.data = recv_span;
                    if (!peer_socket->sendPacket<P2PPacketReduceTerm>(packet)) {
                        LOG(ERR) << "Failed to send reduce term to peer " << uuid_to_string(peer);
                        promise.set_value(false); // failure
                        return;
                    }
                }
            } else {
                // is not last
                // wait for final result
                const auto final_peer = ring_order[ring_order.size() - 1];
                const auto final_peer_socket_opt = p2p_connections_rx.find(final_peer);
                if (final_peer_socket_opt == p2p_connections_rx.end()) {
                    LOG(ERR) << "Failed to find p2p connection for final peer " << uuid_to_string(final_peer);
                    promise.set_value(false); // failure
                    return;
                }
                auto &final_peer_socket = final_peer_socket_opt->second;
                const auto final_rx_packet = final_peer_socket->receivePacket<P2PPacketReduceTerm>();
                if (!final_rx_packet) {
                    LOG(ERR) << "Failed to receive reduce term from final peer " << uuid_to_string(final_peer);
                    promise.set_value(false); // failure
                    return;
                }
                std::memcpy(recv_span.data(), final_rx_packet->data.data(), final_rx_packet->data.size());
                // set content to final result
            }

            // vote collective comms operation complete and await consensus
            {
                C2MPacketCollectiveCommsComplete complete_packet{};
                complete_packet.tag = tag;
                if (!master_socket.sendPacket<C2MPacketCollectiveCommsComplete>(complete_packet)) {
                    promise.set_value(false); // failure
                    return;
                }
                const auto response = master_socket.receiveMatchingPacket<M2CPacketCollectiveCommsComplete>(
                    [tag](const M2CPacketCollectiveCommsComplete &packet) {
                        return packet.tag == tag;
                    });

                if (!response) {
                    promise.set_value(false); // failure
                    return;
                }
            }
            promise.set_value(true); // success
        })) [[unlikely]] {
        return false;
    }

    return true;
}

bool ccoip::CCoIPClientHandler::joinAsyncReduce(const uint64_t tag) {
    if (!client_state.joinAsyncReduce(tag)) [[unlikely]] {
        return false;
    }
    const auto failure_opt = client_state.hasCollectiveComsOpFailed(tag);
    if (!failure_opt) [[unlikely]] {
        LOG(WARN) << "Collective coms op with tag " << tag << " was either not started or has not yet finished";
        return false;
    }
    if (*failure_opt) {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::getAsyncReduceInfo(const uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out) {
    return true;
}

bool ccoip::CCoIPClientHandler::isAnyCollectiveComsOpRunning() const {
    return client_state.isAnyCollectiveComsOpRunning();
}

size_t ccoip::CCoIPClientHandler::getWorldSize() const {
    return client_state.getWorldSize();
}
