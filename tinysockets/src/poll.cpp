#include <cassert>
#include <tinysockets.hpp>
#include <win_sock_bridge.h>

#include <cstring> // for std::strerror

static short ToPollEvent(const tinysockets::poll::PollEvent poll_event) {
    switch (poll_event) {
        case tinysockets::poll::POLL_INPUT:
            return POLLIN;
        case tinysockets::poll::POLL_OUTPUT:
            return POLLOUT;
        default:
            throw std::runtime_error("Invalid poll event");
    }
}

bool tinysockets::poll::PollDescriptor::hasEvent(const PollEvent event) const {
    const short target = ToPollEvent(event);
    return (event_out & target) != 0;
}

#ifndef WIN32
static int unix_poll(std::vector<tinysockets::poll::PollDescriptor> &descriptors, const int timeout) {
    std::vector<pollfd> poll_fds{};

    // convert descriptors to pollfds
    {
        poll_fds.reserve(descriptors.size());
        for (const auto &request: descriptors) {
            poll_fds.push_back(pollfd{
                    .fd = request.socket.getSocketFd(),
                    .events = ToPollEvent(request.target_event),
                    .revents = 0,
            });
        }
    }

    const int ret = poll(poll_fds.data(), poll_fds.size(), timeout);

    // copy revents back to the descriptors
    if (ret > 0) {
        assert(ret <= descriptors.size());

        for (size_t i = 0; i < descriptors.size(); i++) {
            descriptors[i].event_out = static_cast<tinysockets::poll::PollEvent>(poll_fds[i].revents);
        }
    }

    return ret;
}
#else
static int wsa_poll(std::vector<tinysockets::poll::PollDescriptor> &descriptors, const int timeout) {
    std::vector<WSAPOLLFD> poll_fds{};

    // convert descriptors to pollfds
    {
        poll_fds.reserve(descriptors.size());
        for (const auto &request: descriptors) {
            poll_fds.push_back(WSAPOLLFD{
                .fd = static_cast<SOCKET>(request.socket.getSocketFd()),
                .events = ToPollEvent(request.target_event),
                .revents = 0,
            });
        }
    }

    const int ret = WSAPoll(poll_fds.data(), poll_fds.size(), timeout);

    // copy revents back to the descriptors
    if (ret > 0) {
        assert(ret <= descriptors.size());

        for (size_t i = 0; i < descriptors.size(); i++) {
            descriptors[i].event_out = static_cast<tinysockets::poll::PollEvent>(poll_fds[i].revents);
        }
    }

    return ret;
}
#endif

int tinysockets::poll::poll(std::vector<PollDescriptor> &descriptors, const int timeout) {
#ifndef WIN32
    return unix_poll(descriptors, timeout);
#else
    return wsa_poll(descriptors, timeout);
#endif
}

std::optional<size_t> tinysockets::poll::send_nonblocking(const std::span<const std::byte> &data,
                                                          const PollDescriptor &poll_descriptor) {
    if (!poll_descriptor.hasEvent(POLL_OUTPUT)) {
        return std::nullopt;
    }

    const int socket_fd = poll_descriptor.socket.getSocketFd();

    ssize_t bytes_sent;
#ifdef WIN32
    // On Windows, MSG_DONTWAIT is not supported, so we have to use ioctlsocket to set the socket to non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return std::nullopt;
    }
    bytes_sent = sendvp(socket_fd, data.data(), data.size_bytes(), 0);

    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return std::nullopt;
    }
#elif __APPLE__
    // MacOS does not support MSG_DONTWAIT, so we have to use ioctl to set the socket to non-blocking mode
    int flags = fcntl(socket_fd, F_GETFL, 0);
    if (flags == -1) {
        LOG(ERR) << "Failed to get socket flags";
        return std::nullopt;
    }
    // set to non-blocking mode
    flags |= O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, flags) == -1) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return std::nullopt;
    }

    // perform the non-blocking recv
    bytes_sent = sendvp(socket_fd, data.data(), data.size_bytes(), 0);

    // set back to blocking mode
    flags &= ~O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, flags) == -1) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return std::nullopt;
    }
#else
    bytes_sent = sendvp(socket_fd, data.data(), data.size_bytes(), MSG_DONTWAIT);
#endif

    if (bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "send() failed: " << error_message;
        return std::nullopt;
    }
    if (bytes_sent > 0) {
        return bytes_sent;
    }

    return std::nullopt;
}

std::optional<size_t> tinysockets::poll::recv_nonblocking(const std::span<std::byte> &data,
                                                          const PollDescriptor &poll_descriptor) {
    if (!poll_descriptor.hasEvent(POLL_INPUT)) {
        return std::nullopt;
    }

    const int socket_fd = poll_descriptor.socket.getSocketFd();
    ssize_t bytes_received;
#ifdef WIN32
    // On Windows, MSG_DONTWAIT is not supported, so we have to use ioctlsocket to set the socket to non-blocking mode
    u_long mode = 1;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return std::nullopt;
    }
    bytes_received = recvvp(socket_fd, data.data(), data.size_bytes(), 0);

    // set socket back to blocking mode
    mode = 0;
    if (ioctlsocket(socket_fd, FIONBIO, &mode) != NO_ERROR) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return std::nullopt;
    }
#elif __APPLE__
    // MacOS does not support MSG_DONTWAIT, so we have to use ioctl to set the socket to non-blocking mode
    int flags = fcntl(socket_fd, F_GETFL, 0);
    if (flags == -1) {
        LOG(ERR) << "Failed to get socket flags";
        return std::nullopt;
    }
    // set to non-blocking mode
    flags |= O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, flags) == -1) {
        LOG(ERR) << "Failed to set socket into non-blocking mode";
        return std::nullopt;
    }

    // perform the non-blocking recv
    bytes_received = recvvp(socket_fd, data.data(), data.size_bytes(), 0);

    // set back to blocking mode
    flags &= ~O_NONBLOCK;
    if (fcntl(socket_fd, F_SETFL, flags) == -1) {
        LOG(ERR) << "Failed to set socket back to blocking mode";
        return std::nullopt;
    }
#else
    bytes_received = recvvp(socket_fd, data.data(), data.size_bytes(), MSG_DONTWAIT);
#endif

    if (bytes_received <= 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "recv() failed: " << error_message;
        return std::nullopt;
    } else if (bytes_received > 0) {
        return bytes_received;
    }

    return std::nullopt;
}
