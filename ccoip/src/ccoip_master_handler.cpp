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
    LOG(DEBUG) << "Kicking client " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
    if (!server_socket.closeClientConnection(client_address)) [[unlikely]] {
        return false;
    }
    return true;
}

void ccoip::CCoIPMasterHandler::onClientRead(const ccoip_socket_address_t &client_address,
                                             const std::span<uint8_t> data) {
    LOG(INFO) << "Received " << data.size() << " bytes from " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
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
        LOG(DEBUG) << "Received M2CPacketNewPeers from " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
    }
}

void ccoip::CCoIPMasterHandler::handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketRequestSessionJoin &) {
    LOG(DEBUG) << "Received C2MPacketRequestSessionJoin from " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);

    // generate uuid for new peer
    ccoip_uuid new_uuid{};
    uuid_utils::generate_uuid(new_uuid);

    // send response to new peer
    M2CPacketJoinResponse response{};
    response.accepted = true;
    response.assigned_uuid.data = new_uuid;

    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
    }
}

void ccoip::CCoIPMasterHandler::handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                                     const C2MPacketAcceptNewPeers &) {
    LOG(DEBUG) << "Received C2MPacketAcceptNewPeers from " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() = default;
