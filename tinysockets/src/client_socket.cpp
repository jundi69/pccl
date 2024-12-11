#include <ccoip_utils.hpp>
#include <pccl_log.hpp>
#ifdef WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#endif

#include "tinysockets.hpp"
#include "win_sock_bridge.h"
#include <cstring>

#include <cstring>

static void configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockoptvp(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[
        unlikely]] {
        LOG(ERR) << "Failed to set TCP_NODELAY option on server socket";
        closesocket(socket_fd);
        exit(EXIT_FAILURE);
    }

    // set send and recvvp buf
    constexpr int buffer_size = 1 << 20; // 1 MiB
    if (setsockoptvp(socket_fd, SOL_SOCKET, SO_SNDBUF, &buffer_size,
                     sizeof(buffer_size)) < 0) [[unlikely]] {
        LOG(ERR) << "Failed to set SO_SNDBUF option on server socket";
        closesocket(socket_fd);
        exit(EXIT_FAILURE);
    }
    if (setsockoptvp(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size,
                     sizeof(buffer_size)) < 0) [[unlikely]] {
        LOG(ERR) << "Failed to set SO_RCVBUF option on server socket";
        closesocket(socket_fd);
        exit(EXIT_FAILURE);
    }

    // enable SO_REUSEADDR if available
#ifdef SO_REUSEADDR
    if (setsockoptvp(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) [[
        unlikely]] {
        LOG(ERR) << "Failed to set SO_REUSEADDR option on server socket";
        closesocket(socket_fd);
        exit(EXIT_FAILURE);
    }
#endif

    // enable SO_BUSY_POLL if available
#ifdef SO_BUSY_POLL
    setsockoptvp(socket_fd, SOL_SOCKET, SO_BUSY_POLL, &opt, sizeof(opt));
#endif

    // enable TCP_QUICKACK if available
#ifdef TCP_QUICKACK
    setsockoptvp(socket_fd, IPPROTO_TCP, TCP_QUICKACK, &opt, sizeof(opt));
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
        LOG(ERR) << "Failed to create socket";
        return false;
    }
    configure_socket_fd(socket_fd);

    // connect to the server
    sockaddr_in server_address_ipv4{};
    sockaddr_in6 server_address_ipv6{};

    if (connect_sockaddr.inet.protocol == inetIPv4) {
        if (convert_to_sockaddr_ipv4(connect_sockaddr, &server_address_ipv4) == -1) [[unlikely]] {
            LOG(ERR) << "Failed to convert socket address";
            closesocket(socket_fd);
            return false;
        }
    } else if (connect_sockaddr.inet.protocol == inetIPv6) {
        if (convert_to_sockaddr_ipv6(connect_sockaddr, &server_address_ipv6) == -1) [[unlikely]] {
            LOG(ERR) << "Failed to convert socket address";
            closesocket(socket_fd);
            return false;
        }
    } else [[unlikely]] {
        LOG(ERR) << "Unsupported protocol";
        closesocket(socket_fd);
        return false;
    }

    // connect to the server based on protocol
    if (connect_sockaddr.inet.protocol == inetIPv4) {
        if (connect(socket_fd, reinterpret_cast<sockaddr *>(&server_address_ipv4), sizeof(server_address_ipv4)) < 0) [[
            unlikely]]
        {
            LOG(ERR) << "Failed to connect to server";
            closesocket(socket_fd);
            return false;
        }
    } else if (connect_sockaddr.inet.protocol == inetIPv6) {
        if (connect(socket_fd, reinterpret_cast<const sockaddr *>(&server_address_ipv6),
                    sizeof(server_address_ipv6)) < 0) [[unlikely]] {
            LOG(ERR) << "Failed to connect to server";
            closesocket(socket_fd);
            return false;
        }
    } else [[unlikely]] {
        LOG(ERR) << "Unsupported protocol";
        closesocket(socket_fd);
        return false;
    }
    return true;
}

bool tinysockets::BlockingIOSocket::closeConnection() {
    if (socket_fd == 0) [[unlikely]] {
        return false;
    }
    closesocket(socket_fd);
    socket_fd = 0;
    return true;
}

bool tinysockets::BlockingIOSocket::isOpen() const {
    if (socket_fd == 0) [[unlikely]] {
        return false;
    }

#ifndef WIN32
    // Using MSG_PEEK with a small read: if it returns 0, the connection is closed.
    char buf;
    const ssize_t n = recv(socket_fd, &buf, 1, MSG_PEEK | MSG_DONTWAIT);
    if (n == 0) {
        return false;
    }
    if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // No data available, but socket may still be connected
            return true;
        }
        // An error occurred; some errors like ECONNRESET mean the socket is effectively closed
        return false;
    }
    // If we got here, there's data available to read, so the socket is still connected
    return true;
#else
    // set socket into non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return false;
    }

    bool is_open = false;
    char buf;
    if (int n = recv(socket_fd, &buf, 1, MSG_PEEK); n == 0) {
        // The connection was closed gracefully by the peer.
        is_open = false;
    } else if (n == SOCKET_ERROR) {
        if (const int err = WSAGetLastError(); err == WSAEWOULDBLOCK) {
            // No data available now, but the socket is still considered open.
            is_open = true;
        } else {
            // Other errors (like WSAECONNRESET) mean the socket is effectively closed.
            is_open = false;
        }
    }
    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return false;
    }
    return is_open;
#endif
}

bool tinysockets::BlockingIOSocket::sendLtvPacket(const ccoip::packetId_t packet_id,
                                                  const PacketWriteBuffer &buffer) const {
    PacketWriteBuffer tlv_buffer{};
    tlv_buffer.reserve(buffer.size() + sizeof(packet_id) + sizeof(uint64_t));
    tlv_buffer.write<uint64_t>(buffer.size() + sizeof(ccoip::packetId_t));
    tlv_buffer.write(packet_id);
    tlv_buffer.writeContents(buffer.data(), buffer.size());
    int flags = 0;
#ifndef WIN32
    flags |= MSG_NOSIGNAL;
#endif
    if (const ssize_t i = sendvp(socket_fd, tlv_buffer.data(), tlv_buffer.size(), flags); i == -1) [[unlikely]] {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to send packet with error: " << error_message;
        return false;
    }
    return true;
}

size_t tinysockets::BlockingIOSocket::receivePacketLength() const {
    uint64_t length;
    if (const ssize_t i = recvvp(socket_fd, &length, sizeof(length), 0); i == -1) {
        const std::string error_message = std::strerror(errno);
        LOG(INFO) << "Failed to receive packet length with error: " << error_message;
        return 0;
    }
    return ntohll(length);
}

bool tinysockets::BlockingIOSocket::receivePacketData(std::span<std::uint8_t> &dst) const {
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, dst.data() + n_received, dst.size_bytes() - n_received, 0);
        if (i == -1) {
            const std::string error_message = std::strerror(errno);
            LOG(INFO) << "Failed to receive packet data with error: " << error_message;
            return false;
        }
        n_received += i;
    } while (n_received < dst.size_bytes());
    return true;
}
