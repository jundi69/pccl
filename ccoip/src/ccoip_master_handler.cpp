#include "ccoip_master_handler.hpp"

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
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
    if (!server_socket.bind()) {
        return false;
    }
    if (!server_socket.listen()) {
        return false;
    }
    if (!server_socket.runAsync()) {
        return false;
    }
    running = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::interrupt() {
    if (interrupted) {
        return false;
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
    LOG(DEBUG) << "Kicking client " << ccoip_sockaddr_to_str(client_address);
    if (!server_socket.closeClientConnection(client_address)) [[unlikely]] {
        return false;
    }
    return true;
}

void ccoip::CCoIPMasterHandler::onClientRead(const ccoip_socket_address_t &client_address,
                                             const std::span<uint8_t> data) {
    LOG(INFO) << "Received " << data.size() << " bytes from " << ccoip_sockaddr_to_str(client_address);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    if (const auto packet_type = buffer.read<uint16_t>();
        packet_type == C2MPacketRequestSessionJoin::packet_id) {
        C2MPacketRequestSessionJoin packet{};
        packet.deserialize(buffer);
        handleRequestSessionJoin(client_address, packet);
    } else if (packet_type == C2MPacketAcceptNewPeers::packet_id) {
        C2MPacketAcceptNewPeers packet{};
        packet.deserialize(buffer);
        handleAcceptNewPeers(client_address, packet);
    } else if (packet_type == M2CPacketNewPeers::packet_id) {
        M2CPacketNewPeers packet{};
        packet.deserialize(buffer);
        LOG(DEBUG) << "Received M2CPacketNewPeers from " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPMasterHandler::handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketRequestSessionJoin &) {
    LOG(DEBUG) << "Received C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(client_address);

    // check if peer has already joined
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (client_uuids.contains(internal_address)) {
        LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(client_address) << " has already joined";
        return;
    }

    // generate uuid for new peer
    ccoip_uuid new_uuid{};
    uuid_utils::generate_uuid(new_uuid);

    // send response to new peer
    M2CPacketJoinResponse response{};
    response.accepted = true;
    response.assigned_uuid.data = new_uuid;

    // register client uuid
    registerClient(client_address, ccoip_uuid_t{new_uuid});

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPMasterHandler::onClientDisconnect(const ccoip_socket_address_t &client_address) {
    LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " disconnected";

    unregisterClient(client_address);
}

void ccoip::CCoIPMasterHandler::handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                                     const C2MPacketAcceptNewPeers &) {
    LOG(DEBUG) << "Received C2MPacketAcceptNewPeers from " << ccoip_sockaddr_to_str(client_address);
}

void ccoip::CCoIPMasterHandler::registerClient(const ccoip_socket_address_t &client_address, ccoip_uuid_t uuid) {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    client_uuids[internal_address] = uuid;
    uuid_clients[uuid] = internal_address;
}

void ccoip::CCoIPMasterHandler::unregisterClient(const ccoip_socket_address_t &client_address) {
    const auto internal_address = ccoip_socket_to_internal(client_address);
    if (const auto it = client_uuids.find(internal_address); it != client_uuids.end()) {
        if (!uuid_clients.erase(it->second)) {
            LOG(WARN) << "Client with UUID " << uuid_to_string(it->second) <<
                    " not found, but its socket address was. This means bi-directional mapping for client UUIDs is inconsistent";
        }
        client_uuids.erase(it);
    } else {
        LOG(WARN) << "Client " << ccoip_sockaddr_to_str(client_address) << " not found";
    }
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() = default;
