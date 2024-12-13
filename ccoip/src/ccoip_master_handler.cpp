#include "ccoip_master_handler.hpp"

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
        packet.deserialize(buffer);
        handleRequestSessionJoin(client_address, packet);
    } else if (packet_type == C2MPacketAcceptNewPeers::packet_id) {
        C2MPacketAcceptNewPeers packet{};
        packet.deserialize(buffer);
        handleAcceptNewPeers(client_address, packet);
    } else if (packet_type == C2MPacketP2PConnectionsEstablished::packet_id) {
        C2MPacketP2PConnectionsEstablished packet{};
        packet.deserialize(buffer);
        handleP2PConnectionsEstablished(client_address, packet);
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

    const int p2p_listen_port = packet.p2p_listen_port;

    // generate uuid for new peer
    ccoip_uuid_t new_uuid{};
    uuid_utils::generate_uuid(new_uuid.data);

    // send response to new peer
    M2CPacketSessionRegistrationResponse response{};
    response.accepted = true;
    response.assigned_uuid = new_uuid;

    // register client uuid
    if (!server_state.registerClient(client_address, p2p_listen_port, new_uuid)) [[unlikely]] {
        LOG(ERR) << "Failed to register client " << ccoip_sockaddr_to_str(client_address);
        response.accepted = false;
    }

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }

    // if this is the first peer, consider it as voting to accept new peers
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
            LOG(WARN) << "Failed to transition to P2P connections established phase";
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

void ccoip::CCoIPMasterHandler::handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                                                const C2MPacketP2PConnectionsEstablished &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketP2PConnectionsEstablished from " << ccoip_sockaddr_to_str(client_address);

    if (!server_state.markP2PConnectionsEstablished(client_address)) [[unlikely]] {
        LOG(WARN) << "Failed to mark P2P connections established for " << ccoip_sockaddr_to_str(client_address);
        if (!kickClient(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to kick client " << ccoip_sockaddr_to_str(client_address);
            return;
        }
        return;
    }

    checkP2PConnectionsEstablished();
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
    if (!server_state.unregisterClient(client_address)) {
        LOG(WARN) << "Failed to unregister client " << ccoip_sockaddr_to_str(client_address);
    }
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
                    .port = client_info.p2p_listen_port
                },
                .peer_uuid = client_info.client_uuid
            });
        }

        if (!server_socket.sendPacket(client_address, new_peers)) {
            LOG(ERR) << "Failed to send M2CPacketNewPeers to " << ccoip_sockaddr_to_str(client_address);
        }
    }
}
