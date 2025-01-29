#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#include <pccl_log.hpp>

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#undef min
#else
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <sys/fcntl.h>
#include <sys/poll.h>
#endif

#ifdef WIN32

#define MSG_NOSIGNAL 0
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH

typedef long long int ssize_t;

inline uint64_t net_u64_to_host(const uint64_t net_long)
{
    return ntohll(net_long);
}

inline int setsockoptvp(const int socket_fd, const int level, const int optname, const void *optval,
                        const socklen_t optlen) {
    return setsockopt(socket_fd, level, optname, static_cast<const char *>(optval), optlen);
}

inline int getsockoptvp(const int socket_fd, const int level, const int optname, void *optval, socklen_t *optlen)
{
    return getsockopt(socket_fd, level, optname, static_cast<char *>(optval), optlen);
}

inline ssize_t recvvp(const int socket_fd, void *buffer, const size_t length, const int flags) {
    return recv(socket_fd, static_cast<char *>(buffer), static_cast<int>(length), flags);
}
inline ssize_t sendvp(const int socket_fd, const void *buffer, const size_t length, const int flags) {
    return send(socket_fd, static_cast<const char *>(buffer), static_cast<int>(length), flags);
}
#else
#include <unistd.h>

#ifndef ntohll
#include <endian.h>    // __BYTE_ORDER __LITTLE_ENDIAN
#include <byteswap.h>  // bswap_64()
#endif

inline uint64_t net_u64_to_host(uint64_t net_long) {
#ifdef ntohll
    return ntohll(net_long);
#else
#if __BYTE_ORDER == __LITTLE_ENDIAN
    return bswap_64(net_long);  // Compiler builtin GCC/Clang
#else
    return net_long;
#endif
#endif
}

inline void closesocket(const int socket_fd) {
    close(socket_fd);
}

inline int setsockoptvp(const int socket_fd, const int level, const int optname, const void *optval,
                        const socklen_t optlen) {
    return setsockopt(socket_fd, level, optname, optval, optlen);
}

inline int getsockoptvp(const int socket_fd, const int level, const int optname, void *optval, socklen_t *optlen) {
    return getsockopt(socket_fd, level, optname, optval, optlen);
}

inline ssize_t recvvp(const int socket_fd, void *buffer, const size_t length, const int flags) {
    return recv(socket_fd, buffer, length, flags);
}

inline ssize_t sendvp(const int socket_fd, const void *buffer, const size_t length, const int flags) {
    return send(socket_fd, buffer, length, flags);
}


#endif


/// Receive non-blocking data from a socket
inline ssize_t recvvp_nb(const int socket_fd, void *buffer, const size_t length, const int flags) {
    ssize_t bytes_received;
#ifdef WIN32
    // On Windows, MSG_DONTWAIT is not supported, so we have to use ioctlsocket to set the socket to non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return -1;
    }
    bytes_received = recvvp(socket_fd, buffer, length, flags);

    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return -1;
    }
#elif __APPLE__
    // MacOS does not support MSG_DONTWAIT, so we have to use ioctl to set the socket to non-blocking mode
    int socket_flags = fcntl(socket_fd, F_GETFL, 0);
    if (socket_flags == -1) {
        LOG(ERR) << "Failed to get socket flags";
        return -1;
    }
    // set to non-blocking mode
    socket_flags |= O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, socket_flags) == -1) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return -1;
    }

    // perform the non-blocking recv
    bytes_received = recvvp(socket_fd, buffer, length, flags);

    // set back to blocking mode
    socket_flags &= ~O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, socket_flags) == -1) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return -1;
    }
#else
    bytes_received = recvvp(socket_fd, buffer, length, flags | MSG_DONTWAIT);
#endif

    return bytes_received;
}

inline ssize_t sendvp_nb(const int socket_fd, const void *buffer, const size_t length, const int flags) {
    ssize_t bytes_sent;
#ifdef WIN32
    // On Windows, MSG_DONTWAIT is not supported, so we have to use ioctlsocket to set the socket to non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return std::nullopt;
    }
    bytes_sent = sendvp(socket_fd, buffer, length, flags);

    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return std::nullopt;
    }
#elif __APPLE__
    // MacOS does not support MSG_DONTWAIT, so we have to use ioctl to set the socket to non-blocking mode
    int socket_flags = fcntl(socket_fd, F_GETFL, 0);
    if (socket_flags == -1) {
        LOG(ERR) << "Failed to get socket flags";
        return -1;
    }
    // set to non-blocking mode
    socket_flags |= O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, socket_flags) == -1) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return -1;
    }

    // perform the non-blocking recv
    bytes_sent = sendvp(socket_fd, buffer, length, flags);

    // set back to blocking mode
    socket_flags &= ~O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, socket_flags) == -1) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return -1;
    }
#else
    bytes_sent = sendvp(socket_fd, buffer, length, flags | MSG_DONTWAIT);
#endif

    return bytes_sent;
}

#endif //SOCKET_UTILS_H
