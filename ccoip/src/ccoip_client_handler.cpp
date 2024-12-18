#include "ccoip_client_handler.hpp"
#include "ccoip_types.hpp"
#include <pccl_log.hpp>
#include <ccoip.h>
#include <hash_utils.hpp>

#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"
#include <thread_guard.hpp>

ccoip::CCoIPClientHandler::CCoIPClientHandler(const ccoip_socket_address_t &address) : client_socket(address),
    p2p_socket({address.inet.protocol, {}, {}}, CCOIP_PROTOCOL_PORT_P2P),
    shared_state_socket({address.inet.protocol, {}, {}}, CCOIP_PROTOCOL_PORT_SHARED_STATE) {
}

bool ccoip::CCoIPClientHandler::connect() {
    // start listening on p2p socket
    {
        p2p_socket.addReadCallback(
            [this](const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data) {
                onP2PClientRead(client_address, data);
            });
        if (!p2p_socket.listen()) {
            LOG(ERR) << "Failed to bind P2P socket " << p2p_socket.getListenPort();
            return false;
        }
        if (!p2p_socket.runAsync()) [[unlikely]] {
            return false;
        }
        p2p_server_thread_id = p2p_socket.getServerThreadId();
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

    if (!client_socket.establishConnection()) {
        return false;
    }

    // send join request packet to master
    C2MPacketRequestSessionRegistration join_request{};
    join_request.p2p_listen_port = p2p_socket.getListenPort();
    if (!client_socket.sendPacket<C2MPacketRequestSessionRegistration>(join_request)) {
        return false;
    }

    // receive join response packet from master
    const auto response = client_socket.receivePacket<M2CPacketSessionRegistrationResponse>();
    if (!response) {
        return false;
    }
    if (!response->accepted) {
        LOG(ERR) << "Master rejected join request";
        return false;
    }

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    return true;
}

// establishP2PConnection:
// establish socket connection
// send hello packet
// expect hello ack
// mark connection as tx side open (tx side = our connection to the other listening party)

// on p2p listen socket, on accept, wait for hello packet and reply hello ack
// the uuid it is transmitted in the hello, we verify the ip against what the master told
// us the ip of the peer is, and then we mark the connection as rx side open (rx side = their connection to us listening)

bool ccoip::CCoIPClientHandler::acceptNewPeers() {
    if (!client_socket.sendPacket<C2MPacketAcceptNewPeers>({})) {
        return false;
    }

    if (!establishP2PConnections()) {
        LOG(ERR) << "Failed to establish P2P connections";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::syncSharedState(const ccoip_shared_state_t &shared_state) {
    // prepare shared state hashes
    std::vector<SharedStateHashEntry> shared_state_hashes{};
    shared_state_hashes.reserve(shared_state.entries.size());
    for (const auto &[key, entry]: shared_state.entries) {
        shared_state_hashes.push_back({key, FVN1a_512Hash(entry.data(), entry.size_bytes())});
    }

    // vote for shared state sync
    C2MPacketSyncSharedState packet{};
    packet.shared_state_revision = shared_state.revision;
    packet.shared_state_hashes = shared_state_hashes;
    if (!client_socket.sendPacket<C2MPacketSyncSharedState>(packet)) {
        return false;
    }
    // wait for confirmation from master that all peers have voted to sync the shared state
    const auto response = client_socket.receivePacket<M2CPacketSyncSharedState>();
    if (!response) {
        return false;
    }
    if (response->is_outdated) {
        // if shared state is outdated, request shared state from master
    }
    return true;
}

bool ccoip::CCoIPClientHandler::updateTopology() {
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
    if (!client_socket.closeConnection()) [[unlikely]] {
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
    const auto new_peers = client_socket.receivePacket<M2CPacketNewPeers>();
    if (!new_peers) {
        LOG(ERR) << "Failed to receive new peers packet";
        return false;
    }
    if (!new_peers->unchanged) {
        // establish p2p connections
        for (auto &peer: new_peers->new_peers) {
            // check if connection already exists
            if (p2p_connections.contains(peer.peer_uuid)) {
                continue;
            }

            if (!client_state.registerPeer(peer.p2p_listen_addr, peer.peer_uuid)) [[unlikely]] {
                LOG(ERR) << "Failed to register peer " << uuid_to_string(peer.peer_uuid);
                return false;
            }

            if (!establishP2PConnection(peer)) {
                LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
                return false;
            }
        }
    }

    // send packet to this peer has established its p2p connections
    if (!client_socket.sendPacket<C2MPacketP2PConnectionsEstablished>({})) {
        LOG(ERR) << "Failed to send P2P connections established packet";
        return false;
    }

    // wait for response from master, indicating ALL peers have established their
    // respective p2p connections
    if (const auto response = client_socket.receivePacket<M2CPacketP2PConnectionsEstablished>(); !response) {
        LOG(ERR) << "Failed to receive P2P connections established response";
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::establishP2PConnection(const M2CPacketNewPeerInfo &peer) {
    auto [it, inserted] = p2p_connections.emplace(peer.peer_uuid, peer.p2p_listen_addr);
    if (!inserted) {
        LOG(ERR) << "P2P connection with peer " << uuid_to_string(peer.peer_uuid) << " already exists";
        return false;
    }
    auto &connection = it->second;
    if (!connection.establishConnection()) {
        LOG(ERR) << "Failed to establish P2P connection with peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    if (!connection.sendPacket<P2PPacketHello>({})) {
        LOG(ERR) << "Failed to send hello packet to peer " << uuid_to_string(peer.peer_uuid);
    }
    if (const auto response = connection.receivePacket<P2PPacketHelloAck>(); !response) {
        LOG(ERR) << "Failed to receive hello ack from peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    if (!client_state.registerPeer(peer.p2p_listen_addr, peer.peer_uuid)) {
        LOG(ERR) << "Failed to register peer " << uuid_to_string(peer.peer_uuid);
        return false;
    }
    return true;
}

void ccoip::CCoIPClientHandler::handleP2PHello(const ccoip_socket_address_t &client_address,
                                               const P2PPacketHello &) {
    if (!p2p_socket.sendPacket(client_address, P2PPacketHelloAck{})) {
        LOG(ERR) << "Failed to send hello ack to " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPClientHandler::onP2PClientRead(const ccoip_socket_address_t &client_address,
                                                const std::span<std::uint8_t> &data) {
    THREAD_GUARD(p2p_server_thread_id);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == P2PPacketHello::packet_id) {
        P2PPacketHello packet{};
        packet.deserialize(buffer);
        handleP2PHello(client_address, packet);
    } else {
        LOG(ERR) << "Unknown packet type " << packet_type << " from " << ccoip_sockaddr_to_str(client_address);
        if (!p2p_socket.closeClientConnection(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to close connection with " << ccoip_sockaddr_to_str(client_address);
        }
    }
}

void ccoip::CCoIPClientHandler::handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                                         const C2SPacketRequestSharedState &packet) {
    // TODO: implement shared state handling
    S2CPacketSharedStateResponse response{};
    response.status = SHARED_STATE_NOT_DISTRIBUTED;
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
        packet.deserialize(buffer);
        handleSharedStateRequest(client_address, packet);
    } else {
        LOG(ERR) << "Unknown packet type " << packet_type << " from " << ccoip_sockaddr_to_str(client_address);
        if (!shared_state_socket.closeClientConnection(client_address)) [[unlikely]] {
            LOG(ERR) << "Failed to close connection with " << ccoip_sockaddr_to_str(client_address);
        }
    }
}
