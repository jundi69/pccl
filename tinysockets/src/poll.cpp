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

        for (size_t i = 0; i < ret; i++) {
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

        for (size_t i = 0; i < ret; i++) {
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
    if (const ssize_t bytes_sent = sendvp(socket_fd, data.data(), data.size_bytes(), 0);
        bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "send() failed: " << error_message;
        return std::nullopt;
    } else if (bytes_sent > 0) {
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
    if (const ssize_t bytes_received = recvvp(socket_fd, data.data(), data.size_bytes(), 0);
        bytes_received <= 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "recv() failed: " << error_message;
        return std::nullopt;
    } else if (bytes_received > 0) {
        return bytes_received;
    }

    return std::nullopt;
}
