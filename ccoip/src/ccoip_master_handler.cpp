#include "ccoip_master_handler.hpp"

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>
#include <uuid_utils.hpp>

#ifdef _MSC_VER
#define FUNC_SIGNATURE() __FUNCSIG__
#else
#define FUNC_SIGNATURE() __PRETTY_FUNCTION__
#endif

#define THREAD_GUARD(thread_id) \
    if (std::this_thread::get_id() != thread_id) { \
        LOG(FATAL) << "Function " << FUNC_SIGNATURE() << " must be called from the server thread! This is a fatal bug!"; \
        std::terminate(); \
    }

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
    } else if (packet_type == M2CPacketNewPeers::packet_id) {
        M2CPacketNewPeers packet{};
        packet.deserialize(buffer);
        LOG(DEBUG) << "Received M2CPacketNewPeers from " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPMasterHandler::handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                                         const C2MPacketRequestSessionRegistration &) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Received C2MPacketRequestSessionJoin from " << ccoip_sockaddr_to_str(client_address);

    // check if peer has already joined
    if (server_state.isClientRegistered(client_address)) {
        LOG(WARN) << "Peer " << ccoip_sockaddr_to_str(client_address) << " has already joined";
        return;
    }

    // generate uuid for new peer
    ccoip_uuid new_uuid{};
    uuid_utils::generate_uuid(new_uuid);

    // send response to new peer
    M2CPacketSessionRegistrationResponse response{};
    response.accepted = true;
    response.assigned_uuid.data = new_uuid;

    // register client uuid
    server_state.registerClient(client_address, ccoip_uuid_t{new_uuid});

    // send response to new peer
    if (!server_socket.sendPacket(client_address, response)) [[unlikely]] {
        LOG(ERR) << "Failed to send M2CPacketJoinResponse to " << ccoip_sockaddr_to_str(client_address);
    }
}

void ccoip::CCoIPMasterHandler::onClientDisconnect(const ccoip_socket_address_t &client_address) {
    THREAD_GUARD(server_thread_id);

    LOG(DEBUG) << "Client " << ccoip_sockaddr_to_str(client_address) << " disconnected";
    server_state.unregisterClient(client_address);
}

void ccoip::CCoIPMasterHandler::handleAcceptNewPeers(const ccoip_socket_address_t &client_address,
                                                     const C2MPacketAcceptNewPeers &) {
    THREAD_GUARD(server_thread_id);
    LOG(DEBUG) << "Received C2MPacketAcceptNewPeers from " << ccoip_sockaddr_to_str(client_address);
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() = default;
