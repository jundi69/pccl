#include "ccoip_master.hpp"

#include <ccoip_inet_utils.hpp>
#include <tinysockets.hpp>


ccoip::CCoIPMaster::CCoIPMaster(const ccoip_socket_address_t &listen_address) : server_socket(listen_address) {
    server_socket.addReadCallback([this](const ccoip_socket_address_t &client_address, const std::span<uint8_t> data) {
        onClientRead(client_address, data);
    });
}

bool ccoip::CCoIPMaster::run() {
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

bool ccoip::CCoIPMaster::interrupt() {
    if (interrupted) {
        return false;
    }
    if (!server_socket.interrupt()) {
        return false;
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPMaster::join() {
    if (!running) {
        return false;
    }
    server_socket.join();
    return true;
}

void ccoip::CCoIPMaster::kickClient(const ccoip_socket_address_t &client_address) {
}

void ccoip::CCoIPMaster::onClientRead(const ccoip_socket_address_t &client_address, const std::span<uint8_t> data) {
    LOG(INFO) << "Received " << data.size() << " bytes from " << CCOIP_SOCKET_ADDR_TO_STRING(client_address);
    PacketReadBuffer buffer = PacketReadBuffer::wrap(data);
    const auto packet_type = buffer.read<uint16_t>();
    const auto packet_length = buffer.read<uint64_t>();
    if (packet_length != buffer.remaining()) {
        LOG(ERROR) << "Packet length mismatch: expected " << packet_length << " bytes, but got " << buffer.remaining();
        kickClient(client_address);
        return;
    }
}

ccoip::CCoIPMaster::~CCoIPMaster() = default;
