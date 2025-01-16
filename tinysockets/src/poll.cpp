#include <tinysockets.hpp>
#include <sys/poll.h>

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
    return (event_out & target) == target;
}


void tinysockets::poll::poll(const std::initializer_list<PollDescriptor> &requests, const int timeout) {
    std::vector<pollfd> poll_fds{};

    // convert requests to pollfds
    {
        poll_fds.reserve(requests.size());
        for (const auto &request: requests) {
            poll_fds.push_back(pollfd{
                .fd = request.socket_fd,
                .events = ToPollEvent(request.target_event),
                .revents = 0,
            });
        }
    }

    ::poll(poll_fds.data(), poll_fds.size(), timeout);
}

std::optional<size_t> tinysockets::poll::send_nonblocking(const std::span<uint8_t> &data,
                                                          const PollDescriptor &poll_descriptor) {
    if (!poll_descriptor.hasEvent(POLL_OUTPUT)) {
        return std::nullopt;
    }

    const int socket_fd = poll_descriptor.socket_fd;
    if (const ssize_t bytes_sent = send(socket_fd, data.data(), data.size_bytes(), 0);
        bytes_sent < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "send() failed: " << error_message;
        return std::nullopt;
    } else if (bytes_sent > 0) {
        return bytes_sent;
    }

    return std::nullopt;
}

std::optional<size_t> tinysockets::poll::recv_nonblocking(std::span<uint8_t> &data,
                                                          const PollDescriptor &poll_descriptor) {
    if (!poll_descriptor.hasEvent(POLL_INPUT)) {
        return std::nullopt;
    }

    const int socket_fd = poll_descriptor.socket_fd;
    if (const ssize_t bytes_received = recv(socket_fd, data.data(), data.size_bytes(), 0);
        bytes_received <= 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        const std::string error_message = strerror(errno);
        LOG(WARN) << "recv() failed: " << error_message;
        return std::nullopt;
    } else if (bytes_received > 0) {
        return bytes_received;
    }

    return std::nullopt;
}
