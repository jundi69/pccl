#include <ccoip_utils.hpp>
#include <pccl_log.hpp>
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

#include "tinysockets.hpp"

#include <sys/socket.h>

static void configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to set TCP_NODELAY option on server socket";
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    // set send and recv buf
    constexpr int buffer_size = 1 << 20; // 1 MiB
    if (setsockopt(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size, sizeof(buffer_size)) < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to set SO_SNDBUF option on server socket";
        close(socket_fd);
        exit(EXIT_FAILURE);
    }
    if (setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, sizeof(buffer_size)) < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to set SO_RCVBUF option on server socket";
        close(socket_fd);
        exit(EXIT_FAILURE);
    }

    // enable SO_REUSEADDR if available
#ifdef SO_REUSEADDR
    if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to set SO_REUSEADDR option on server socket";
        close(socket_fd);
        exit(EXIT_FAILURE);
    }
#endif

    // enable SO_BUSY_POLL if available
#ifdef SO_BUSY_POLL
    setsockopt(socket_fd, SOL_SOCKET, SO_BUSY_POLL, &opt, sizeof(opt));
#endif

    // enable TCP_QUICKACK if available
#ifdef TCP_QUICKACK
    setsockopt(socket_fd, IPPROTO_TCP, TCP_QUICKACK, &opt, sizeof(opt));
#endif
}


tinysockets::BlockingIOSocket::BlockingIOSocket(const ccoip_socket_address_t &address) : socket_fd(0),
    connect_sockaddr(address) {
}

bool tinysockets::BlockingIOSocket::establishConnection() {
    if (socket_fd != 0) [[unlikely]] {
        return false;
    }
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to create socket";
        return false;
    }
    configure_socket_fd(socket_fd);

    // connect to the server
    sockaddr_in server_address{};
    if (convert_to_sockaddr(connect_sockaddr, server_address) == -1) [[unlikely]] {
        LOG(ERROR) << "Failed to convert socket address";
        close(socket_fd);
        return false;
    }

    // connect to the server
    if (connect(socket_fd, reinterpret_cast<sockaddr *>(&server_address), sizeof(server_address)) < 0) [[unlikely]] {
        LOG(ERROR) << "Failed to connect to server";
        close(socket_fd);
        return false;
    }
    return true;
}

bool tinysockets::BlockingIOSocket::sendTlvPacket(const ccoip::packetId_t packet_id,
                                                  const PacketWriteBuffer &buffer) const {
    PacketWriteBuffer tlv_buffer{};
    tlv_buffer.reserve(buffer.size() + sizeof(packet_id) + sizeof(uint64_t));
    tlv_buffer.write(packet_id);
    tlv_buffer.write<uint64_t>(buffer.size());
    tlv_buffer.writeContents(buffer.data(), buffer.size());
    if (const ssize_t i = send(socket_fd, tlv_buffer.data(), tlv_buffer.size(), MSG_NOSIGNAL); i == -1) [[unlikely]] {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to send packet with error: " << error_message;
        return false;
    }
    return true;
}

ccoip::packetId_t tinysockets::BlockingIOSocket::receivePacketType() const {
    ccoip::packetId_t packet_id;
    if (const ssize_t i = recv(socket_fd, &packet_id, sizeof(packet_id), 0); i == -1) [[unlikely]] {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to receive packet type with error: " << error_message;
        return 0;
    }
    return packet_id;
}

size_t tinysockets::BlockingIOSocket::receivePacketLength() const {
    size_t length;
    if (const ssize_t i = recv(socket_fd, &length, sizeof(length), 0); i == -1) [[unlikely]] {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to receive packet length with error: " << error_message;
        return 0;
    }
    return length;
}

bool tinysockets::BlockingIOSocket::receivePacketData(std::span<std::uint8_t> dst) const {
    if (const ssize_t i = recv(socket_fd, dst.data(), dst.size_bytes(), 0); i == -1) [[unlikely]] {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to receive packet data with error: " << error_message;
        return false;
    }
    return true;
}
