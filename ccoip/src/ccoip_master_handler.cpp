#include "ccoip_master_handler.hpp"

#include <ccoip.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <thread_guard.hpp>
#include <tinysockets.hpp>
#include <uuid_utils.hpp>

ccoip::CCoIPMasterHandler::CCoIPMasterHandler(const ccoip_socket_address_t &listen_address) : server_socket(
    listen_address) {
    server_socket.addReadCallback([this](const ccoip_socket_address_t &client_address, const std::span<uint8_t> &data) {
        onClientRead(client_address, data);
    });
    server_socket.addCloseCallback([this](const ccoip_socket_address_t &client_address) {
        onClientDisconnect(client_address);
    });
}

bool ccoip::CCoIPMasterHandler::run() {
    if (!server_socket.listen()) {
        return false;
    }
    if (!server_socket.runAsync()) {
        return false;
    }
    server_thread_id = server_socket.getServerThreadId();
    running = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::interrupt() {
    if (interrupted) {
        return true;
    }
    if (!server_socket.interrupt()) {
        return false;
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::join() {
    if (!running) {
        return false;
    }
    server_socket.join();
    return true;
}

bool ccoip::CCoIPMasterHandler::kickClient(const ccoip_socket_address_t &client_address) const {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Kicking client " << ccoip_sockaddr_to_str(client_address);
    if (!server_socket.closeClientConnection(client_address)) [[unlikely]] {
        return false;
    }
    return true;
}

void ccoip::CCoIPMasterHandler::onClientRead(const ccoip_socket_address_t &client_address,
                                             const std::span<uint8_t> &data) {
    THREAD_GUARD(server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == C2MPacketRequestSessionRegistration::packet_id) {
        C2MPacketRequestSessionRegistration packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(
                client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
        }
        handleRequestSessionJoin(client_address, packet);
    } else if (packet_type == C2MPacketAcceptNewPeers::packet_id) {
        C2MPacketAcceptNewPeers packet{};
        packet.deserialize(buffer);
        handleAcceptNewPeers(client_address, packet);
    } else if (packet_type == C2MPacketP2PConnectionsEstablished::packet_id) {
        C2MPacketP2PConnectionsEstablished packet{};
        packet.deserialize(buffer);
        handleP2PConnectionsEstablished(client_address, packet);
    } else if (packet_type == C2MPacketSyncSharedState::packet_id) {
        C2MPacketSyncSharedState packet{};
        if (!packet.deserialize(buffer)) {
            LOG(ERR) << "Failed to deserialize C2MPacketSyncSharedState from " << ccoip_sockaddr_to_str(client_address);
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
        }
        handleSyncSharedState(client_address, packet);
    } else if (packet_type == C2MPacketDistSharedStateComplete::packet_id) {
        C2MPacketDistSharedStateComplete packet{};
        packet.deserialize(buffer);
        handleSyncSharedStateComplete(client_address, packet);
    }
}

void ccoip::CCoIPMasterHandler::handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketRequestSessionRegistration &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(client_address);

    // check if peer has already joined
    if (server_state.isClientRegistered(client_address)) {
        LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(client_address) << " has already joined";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    const uint16_t p2p_listen_port = packet.p2p_listen_port;
    const uint16_t shared_state_listen_port = packet.shared_state_listen_port;

    // generate uuid for new peer
    ccoip_uuid_t new_uuid{};
    uuid_utils::generate_uuid(new_uuid.data);

    // send response to new peer
    M2CPacketSessionRegistrationResponse response{};
    response.accepted = true;
    response.assigned_uuid = new_uuid;

    // register client uuid
    if (!server_state.registerClient(client_address,
                                     CCoIPClientVariablePorts{p2p_listen_port, shared_state_listen_port},
                                     packet.peer_group,
                                     new_uuid)) [[
        unlikely]] {
        LOG(ERR) << "Failed to register client " << ccoip_sockaddr_to_str(client_address);
        response.accepted = false;
    }

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }

    // if this is the first peer, simply send empty p2p connection information
    if (server_state.getClientSocketAddresses().size() == 1) {
        sendP2PConnectionInformation();
    } else {
        // otherwise still check if we have consensus to accept new peers
        checkAcceptNewPeersConsensus();
    }
}

void ccoip::CCoIPMasterHandler::checkP2PConnectionsEstablished() {
    // send establish new peers packets to all clients
    if (server_state.p2pConnectionsEstablishConsensus()) {
        // send confirmation packets to all clients
        for (auto &peer_address: server_state.getClientSocketAddresses()) {
            if (!server_socket.sendPacket<C2MPacketP2PConnectionsEstablished>(peer_address, {})) {
                LOG(ERR) << "Failed to send M2CPacketP2PConnectionsEstablished to " << ccoip_sockaddr_to_str(
                    peer_address
                );
            }
        }
        if (!server_state.transitionToP2PConnectionsEstablishedPhase()) [[unlikely]] {
            LOG(BUG) << "Failed to transition to P2P connections established phase; This is a bug";
        }
    }
}

void ccoip::CCoIPMasterHandler::checkAcceptNewPeersConsensus() {
    // check if all clients have voted to accept new peers
    if (server_state.acceptNewPeersConsensus()) {
        server_state.transitionToP2PEstablishmentPhase();
        sendP2PConnectionInformation();
    }
}

std::optional<ccoip_socket_address_t>
ccoip::CCoIPMasterHandler::findBestSharedStateTxPeer(const ccoip_socket_address_t &peer_address) {
    const auto info_opt = server_state.getClientInfo(peer_address);
    if (!info_opt) [[unlikely]] {
        LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
        return std::nullopt;
    }
    auto &info = info_opt->get();

    // TODO: should be topology aware
    // for now, just return the first peer that distributes the shared state
    for (const auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
        const auto peer_info_opt = server_state.getClientInfo(peer_address);
        if (!peer_info_opt) [[unlikely]] {
            LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
            continue;
        }
        const auto &peer_info = peer_info_opt->get();
        if (peer_info.peer_group != info.peer_group) {
            continue;
        }
        if (peer_info.connection_phase == PEER_ACCEPTED && peer_info.connection_state == DISTRIBUTE_SHARED_STATE) {
            return ccoip_socket_address_t{peer_address.inet, peer_info.variable_ports.shared_dist_state_listen_port};
        }
    }
    return std::nullopt;
}

void ccoip::CCoIPMasterHandler::checkSyncSharedStateConsensus(const uint32_t peer_group) {
    // check if all clients have voted to sync shared state
    if (server_state.syncSharedStateConsensus(peer_group)) {
        if (!server_state.transitionToSharedStateSyncPhase(peer_group)) [[unlikely]] {
            LOG(BUG) << "Failed to transition to shared state distribution phase; This is a bug!";
            return;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_address);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }

            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // only these are valid states for accepted clients to be in after the shared state voting phase
            if (peer_info.connection_state != DISTRIBUTE_SHARED_STATE && peer_info.connection_state !=
                REQUEST_SHARED_STATE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected DISTRIBUTE_SHARED_STATE or REQUEST_SHARED_STATE";
                continue;
            }

            M2CPacketSyncSharedState response{};

            // if the state is REQUEST_SHARED_STATE, then transitionToSharedStateDistributionPhase has
            // determined that the shared state hash does not match; notify the client to re-request the shared state
            const bool needs_update = peer_info.connection_state == REQUEST_SHARED_STATE;
            response.is_outdated = needs_update;
            if (needs_update) {
                auto best_peer_opt = findBestSharedStateTxPeer(peer_address);
                if (!best_peer_opt) [[unlikely]] {
                    LOG(BUG) << "No peer found to distribute shared state to " << ccoip_sockaddr_to_str(peer_address) <<
                            " while peers is marked to request shared state. This is a bug!";
                    continue;
                }
                response.distributor_address = *best_peer_opt;
                auto outdated_keys = server_state.getOutdatedSharedStateKeys(peer_uuid);
                if (outdated_keys.empty()) {
                    outdated_keys = server_state.getSharedStateKeys(peer_group);
                }
                response.outdated_keys = outdated_keys;

                for (const auto &key: outdated_keys) {
                    response.expected_hashes.push_back(server_state.getSharedStateEntryHash(peer_group, key));
                }
            }

            if (!server_socket.sendPacket<M2CPacketSyncSharedState>(peer_address, response)) {
                LOG(ERR) << "Failed to send M2CPacketSyncSharedState to " << ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
}

void ccoip::CCoIPMasterHandler::checkSyncSharedStateCompleteConsensus(const uint32_t peer_group) {
    // check if all clients have voted to distribute shared state complete
    if (server_state.syncSharedStateCompleteConsensus(peer_group)) {
        if (!server_state.endSharedStateSyncPhase(peer_group)) [[unlikely]] {
            LOG(BUG) << "Failed to end shared state distribution phase; This is a bug!";
            return;
        }

        // send confirmation packets to all clients
        for (auto &[peer_uuid, peer_address]: server_state.getClientEntrySet()) {
            const auto peer_info_opt = server_state.getClientInfo(peer_address);
            if (!peer_info_opt) [[unlikely]] {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) << " not found";
                continue;
            }
            const auto &peer_info = peer_info_opt->get();
            if (peer_info.connection_phase != PEER_ACCEPTED) {
                continue;
            }
            if (peer_info.peer_group != peer_group) {
                continue;
            }

            // because endSharedStateSyncPhase() was already invoked,
            // the only valid state for accepted clients to be in after a successful vote
            // on shared state distribution completion is IDLE
            if (peer_info.connection_state != IDLE) {
                LOG(BUG) << "Client " << ccoip_sockaddr_to_str(peer_address) <<
                        " in state " << peer_info.connection_state <<
                        " but expected IDLE";
                continue;
            }

            // send confirmation packet
            if (!server_socket.sendPacket<M2CPacketSyncSharedStateComplete>(peer_address, {})) {
                LOG(ERR) << "Failed to send M2CPacketSyncSharedStateComplete to " <<
                        ccoip_sockaddr_to_str(peer_address);
            }
        }
    }
}

void ccoip::CCoIPMasterHandler::handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                                                const C2MPacketP2PConnectionsEstablished &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketP2PConnectionsEstablished from " << ccoip_sockaddr_to_str(client_address);

    if (!server_state.markP2PConnectionsEstablished(client_address)) [[unlikely]] {
        LOG(WARN) << "Failed to mark P2P connections established for " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    checkP2PConnectionsEstablished();
}

void ccoip::CCoIPMasterHandler::handleSyncSharedState(const ccoip_socket_address_t &client_address,
                                                      const C2MPacketSyncSharedState &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketSyncSharedState from " << ccoip_sockaddr_to_str(client_address);

    // obtain client uuid from client address
    ccoip_uuid_t client_uuid{}; {
        const auto client_uuid_opt = server_state.findClientUUID(client_address);
        if (!client_uuid_opt) [[unlikely]] {
            LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
            if (!kickClient(client_address)) [[unlikely]] {
                LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            }
            return;
        }
        client_uuid = client_uuid_opt.value();
    }

    // check if shared state request follows the right "mask" as in:
    // 1. requests same keys as the other clients; mismatch will result in a kick
    // 2. hash of the shared state is the same as the hash of the shared state of the other clients; mismatch will
    //    result in the client being notified to re-request the shared state entry whose hash does not match.
    // If the client is the first to sync shared state, then that peer's shared state is the reference
    // "mask" for the other clients.
    const CCoIPMasterState::SharedStateMismatchStatus status = server_state
            .sharedStateMatches(client_uuid, packet.shared_state_revision, packet.shared_state_hashes);

    if (status == CCoIPMasterState::KEY_SET_MISMATCH) {
        LOG(WARN) << "Shared state key set mismatch for " << ccoip_sockaddr_to_str(client_address) <<
                ". Please make sure all clients have the same shared state keys; Client will be kicked.";
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    // vote for sync shared state
    if (!server_state.voteSyncSharedState(client_address)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to sync shared state from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    const auto info_opt = server_state.getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const auto &info = info_opt->get();
    checkSyncSharedStateConsensus(info.peer_group);
}

void ccoip::CCoIPMasterHandler::handleSyncSharedStateComplete(const ccoip_socket_address_t &client_address,
                                                              const C2MPacketDistSharedStateComplete &packet) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketSyncSharedStateComplete from " << ccoip_sockaddr_to_str(client_address);

    if (!server_state.voteDistSharedStateComplete(client_address)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to distribute shared state complete from " <<
                ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    const auto info_opt = server_state.getClientInfo(client_address);
    if (!info_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const auto &info = info_opt->get();
    checkSyncSharedStateCompleteConsensus(info.peer_group);
}

void ccoip::CCoIPMasterHandler::onClientDisconnect(const ccoip_socket_address_t &client_address) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " disconnected";
    if (!server_state.isClientRegistered(client_address)) {
        // client disconnected before ever requesting to register
        LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) <<
                " disconnected before registering. Strange, but not impossible";
        return;
    }

    const auto client_info_opt = server_state.getClientInfo(client_address);
    if (!client_info_opt) {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
        return;
    }
    const ClientInfo client_info = client_info_opt->get();
    if (!server_state.unregisterClient(client_address)) {
        LOG(WARN) << "Failed to unregister client " << ccoip_sockaddr_to_str(client_address);
        return;
    }

    // the client disconnecting might result in consensus changes for ongoing votes;
    // If the client that left was the only outstanding vote to e.g. accept new peers, then we need to re-check
    // the consensus and send response packets as necessary.
    checkAcceptNewPeersConsensus();
    checkSyncSharedStateConsensus(client_info.peer_group);
    checkSyncSharedStateCompleteConsensus(client_info.peer_group);
}

void ccoip::CCoIPMasterHandler::handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                                     const C2MPacketAcceptNewPeers &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketAcceptNewPeers from " << ccoip_sockaddr_to_str(client_address);

    if (!server_state.voteAcceptNewPeers(client_address)) [[unlikely]] {
        LOG(WARN) << "Failed to vote to accept new peers from " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
        }
        return;
    }

    checkAcceptNewPeersConsensus();
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() = default;

void ccoip::CCoIPMasterHandler::sendP2PConnectionInformation() {
    const bool unchanged = !server_state.hasPeerListChanged();

    // send establish new peers packets to all clients
    for (auto &client_address: server_state.getClientSocketAddresses()) {
        // for all connected clients
        M2CPacketNewPeers new_peers{};
        new_peers.unchanged = unchanged;

        auto peers = server_state.getPeersForClient(client_address); // get the peers for the client
        new_peers.new_peers.reserve(peers.size());
        for (const auto &client_info: peers) {
            // construct a new peers packet
            new_peers.new_peers.push_back({
                .p2p_listen_addr = ccoip_socket_address_t{
                    .inet = client_info.socket_address.inet,
                    .port = client_info.variable_ports.p2p_listen_port
                },
                .peer_uuid = client_info.client_uuid
            });
        }

        if (!server_socket.sendPacket(client_address, new_peers)) {
            LOG(ERR) << "Failed to send M2CPacketNewPeers to " << ccoip_sockaddr_to_str(client_address);
        }
    }
}
