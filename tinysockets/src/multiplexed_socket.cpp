#include "tinysockets.hpp"

#include <ccoip_utils.hpp>
#include <win_sock_bridge.h>

#include <MPSCQueue.hpp>
#include <SPSCQueue.hpp>
#include <cstring>
#include <network_order_utils.hpp>
#include <shared_mutex>
#include <threadpark.h>
#include <lockfree_map.hpp>

#ifndef _MSC_VER
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

static bool configure_socket_fd(const int socket_fd) {
    constexpr int opt = 1;

    // enable TCP_NODELAY
    if (setsockoptvp(socket_fd, IPPROTO_TCP, TCP_NODELAY, &opt, sizeof(opt)) < 0) [[unlikely]] {
        LOG(ERR) << "Failed to set TCP_NODELAY option on server socket";
        closesocket(socket_fd);
        return false;
    }

    // enable SO_BUSY_POLL if available
#ifdef SO_BUSY_POLL
    setsockoptvp(socket_fd, SOL_SOCKET, SO_BUSY_POLL, &opt, sizeof(opt));
#endif

    // enable TCP_QUICKACK if available
#ifdef TCP_QUICKACK
    setsockoptvp(socket_fd, IPPROTO_TCP, TCP_QUICKACK, &opt, sizeof(opt));
#endif
    return true;
}

struct ReceiveQueueEntry {
    uint64_t tag{};

    /// @warning may be nullptr, if the producer does not "own" the memory because
    /// it did not allocate it. If this is null, this means that the producer
    /// thread used consumer provided memory, which will be referenced in @ref data_span.
    std::unique_ptr<std::byte[]> data;

    std::span<std::byte> data_span{};
};

struct SendQueueEntry {
    uint64_t tag{};
    std::span<const uint8_t> data;
    size_t size_bytes{};
};

#define TXRX_QUEUE_DEPTH 1024

namespace tinysockets {
    struct MultiplexedIOSocketInternalState {
        MPSCQueue<SendQueueEntry> send_queue;

        LockFreeMap<std::unique_ptr<::rigtorp::SPSCQueue<ReceiveQueueEntry>>> receive_queues;

        /// @note This map differentiates three cases:
        /// - no span exists for the tag -> the RX thread should block
        /// - a span exists for the tag, but the span references a nullptr -> the RX thread should allocate the memory
        /// - a span exists for the tag and the span references a valid memory location -> the RX thread should use the provided memory
        /// After the RX thread has received the data, it will erase the entry from the map.
        LockFreeMap<std::span<uint8_t>> receive_buffers;

        LockFreeMap<std::atomic_bool> send_completed;

        tpark_handle_t *tx_park_handle = nullptr;
        tpark_handle_t *rx_park_handle = nullptr;

        explicit MultiplexedIOSocketInternalState(const int max_connections) : send_queue(TXRX_QUEUE_DEPTH),
                                                                               receive_queues(
                                                                                   static_cast<size_t>(
                                                                                       max_connections)),
                                                                               receive_buffers(
                                                                                   static_cast<size_t>(
                                                                                       max_connections)),
                                                                               send_completed(
                                                                                   static_cast<size_t>(
                                                                                       max_connections)) {
        }

        ~MultiplexedIOSocketInternalState() {
            if (tx_park_handle != nullptr) {
                tparkDestroyHandle(tx_park_handle);
            }
            if (rx_park_handle != nullptr) {
                tparkDestroyHandle(rx_park_handle);
            }
        }
    };
} // namespace tinysockets


tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const ccoip_socket_address_t &address,
                                                      const ConnectionModeFlags flags,
                                                      const int max_connections) : socket_fd(0),
    connect_sockaddr(address), flags(flags), internal_state(new MultiplexedIOSocketInternalState(max_connections)) {
}

tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const int socket_fd,
                                                      const ConnectionModeFlags flags,
                                                      const int max_connections) : socket_fd(socket_fd),
    connect_sockaddr(), flags(flags), internal_state(new MultiplexedIOSocketInternalState(max_connections)) {
}

tinysockets::MultiplexedIOSocket::MultiplexedIOSocket(const int socket_fd, const ccoip_socket_address_t &address,
                                                      const ConnectionModeFlags flags,
                                                      const int max_connections) : socket_fd(socket_fd),
    connect_sockaddr(address), flags(flags),
    internal_state(new MultiplexedIOSocketInternalState(max_connections)) {
}

bool tinysockets::MultiplexedIOSocket::establishConnection() {
    if (socket_fd != 0) {
        return false;
    }
    socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_fd < 0) [[unlikely]] {
        LOG(ERR) << "Failed to create socket";
        return false;
    }

    if (!configure_socket_fd(socket_fd)) [[unlikely]] {
        return false;
    }

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
        if (connect(socket_fd, reinterpret_cast<sockaddr *>(&server_address_ipv4), sizeof(server_address_ipv4)) < 0) {
            const std::string error_message = std::strerror(errno);
            LOG(ERR) << "Failed to connect to server; connect() failed with " << error_message;
            closesocket(socket_fd);
            return false;
        }
    } else if (connect_sockaddr.inet.protocol == inetIPv6) {
        if (connect(socket_fd, reinterpret_cast<const sockaddr *>(&server_address_ipv6), sizeof(server_address_ipv6)) <
            0) {
            const std::string error_message = std::strerror(errno);
            LOG(ERR) << "Failed to connect to server; connect() failed with " << error_message;
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

bool tinysockets::MultiplexedIOSocket::run() {
    if (socket_fd == 0) {
        return false;
    }
    running.store(true, std::memory_order_release);
    if (flags & MODE_RX) {
        internal_state->rx_park_handle = tparkCreateHandle();
        recv_thread = std::thread([this] {
            while (running.load(std::memory_order_acquire)) {
                const auto length_opt = receivePacketLength();
                if (!length_opt) {
                    if (running.load(std::memory_order_acquire)) {
                        LOG(ERR) << "Connection was closed; exiting receive loop...";
                        if (!interrupt()) [[unlikely]] {
                            LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                        }
                    } else {
                        LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting receive loop...";
                    }
                    break;
                }
                const size_t length = *length_opt;
                if (length == 0) {
                    LOG(ERR) << "Received packet with length 0; closing connection";
                    if (!interrupt()) [[unlikely]] {
                        LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                    }
                    break;
                }

                auto tag_opt = receivePacketLength();
                if (!tag_opt) {
                    if (running.load(std::memory_order_acquire)) {
                        LOG(ERR) << "Connection was closed; exiting receive loop...";
                        if (!interrupt()) [[unlikely]] {
                            LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                        }
                    } else {
                        LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting receive loop...";
                    }
                    break;
                }
                const auto tag = *tag_opt;

                std::unique_ptr<std::byte[]> data_ptr;
                std::span<uint8_t> data{};

                // lookup the receive-buffer for the tag
                {
                    tparkBeginPark(internal_state->rx_park_handle);

                    std::span<uint8_t> buffer{}; {
                        if (!internal_state->receive_buffers.contains(tag)) {
                            tparkWait(internal_state->rx_park_handle, true);

                            // writes to queue may not be visible to this thread yet
                            while (!internal_state->receive_buffers.contains(tag)) {
                                // spin for the few cycles it takes for the write-access to become visible
                                // yes, we hold on to the shared lock here preventing any write.
                                // this is strictly waiting for a write-access that occurred during
                                // the time we were not holding the lock.
                            }
                        } else {
                            tparkEndPark(internal_state->rx_park_handle);
                        }
                        buffer = **internal_state->receive_buffers.get(tag);
                    }
                    // remove the buffer from the map
                    {
                        // NOTE: this erase will never be concurrent with
                        // read accesses to receive_buffers for the same tag,
                        // as the receiveBytes() generally just never reads
                        // from receive_buffers at all, it only writes to it.
                        // And we know this write has already happened because
                        // we have waited for it above.
                        // We also know that the receiveBytes() method
                        // will wait until the receive-access is complete before
                        // starting a new receiveBytes() invocation and
                        // thus can also rule out contention for write access.
                        internal_state->receive_buffers.erase(tag);
                    }

                    // if the buffer is nullptr, we need to allocate memory
                    if (buffer.data() == nullptr) {
                        data_ptr = std::unique_ptr<std::byte[]>(new std::byte[length]);
                        data = std::span(reinterpret_cast<uint8_t *>(data_ptr.get()), length);
                    } else {
                        data_ptr = nullptr; // we don't own the memory
                        data = std::span(buffer.data(), length);
                    }
                }

                if (!receivePacketData(data)) {
                    if (running.load(std::memory_order_acquire)) {
                        LOG(ERR) << "Failed to receive packet data for packet with length " << length;
                        if (!interrupt()) [[unlikely]] {
                            LOG(ERR) << "Failed to interrupt MultiplexedIOSocket";
                        }
                    } else {
                        LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting receive loop...";
                    }
                    break;
                }

                if (!internal_state->receive_queues.contains(tag)) {
                    internal_state->receive_queues.emplace(tag,
                                                           std::make_unique<::rigtorp::SPSCQueue<ReceiveQueueEntry>>(
                                                               TXRX_QUEUE_DEPTH)
                    );
                }

                // add entry to SPMC queue of tag
                {
                    const auto &queue = **internal_state->receive_queues.get(tag);
                    queue->push(ReceiveQueueEntry{
                        .tag = tag,

                        // we just pass this for the sake of ownership
                        .data = data_ptr == nullptr ? nullptr : std::move(data_ptr),

                        // don't include the tag in the data span
                        .data_span = std::span(reinterpret_cast<std::byte *>(data.data()),
                                               data.size_bytes())
                    });
                }
            }
        });
    }

    if (flags & MODE_TX) {
        internal_state->tx_park_handle = tparkCreateHandle();
        send_thread = std::thread([this] {
            while (running.load(std::memory_order_acquire) && socket_fd != 0) {
                const SendQueueEntry *entry{}; {
                    tparkBeginPark(internal_state->tx_park_handle);
                    entry = internal_state->send_queue.dequeue(true);
                    if (entry == nullptr) {
                        tparkWait(internal_state->tx_park_handle, true);
                        do {
                            entry = internal_state->send_queue.dequeue(true);
                            // despite the fact that wakes are guaranteed not-spurious and insertion into the queue
                            // should happen before the wake, it still is possible that the queue is empty because
                            // seq cst only guarantees all writes before a seq_cst store are visible to other threads
                            // when we are seq_cst loading a value that was written with seq_cst by the producer thread
                            // from which it can be inferred that a previous store has occurred.
                            // Here, the producer calls wake, where we are at the mercy of the OS as to which
                            // memory model it uses for the atomic store. So we still can't assert the queue
                            // is always non-empty after a wake.
                        } while (entry == nullptr && running.load(std::memory_order_acquire));
                    } else {
                        tparkEndPark(internal_state->tx_park_handle);
                    }
                }

                if (entry == nullptr) {
                    LOG(DEBUG) << "no entry to send, exiting send loop...";
                    break;
                }

                const uint64_t preamble[2] = {
                    network_order_utils::host_to_network(
                        entry->size_bytes
                    ),
                    network_order_utils::host_to_network(entry->tag)
                };
                if (sendvp(socket_fd, preamble, sizeof(preamble), MSG_NOSIGNAL) == -1) {
                    LOG(ERR) << "Failed to send preamble for packet with tag " << entry->tag;
                    delete entry;
                    continue;
                }
                size_t n_sent = 0;
                do {
                    const ssize_t i =
                            sendvp(socket_fd, entry->data.data() + n_sent, entry->size_bytes - n_sent, MSG_NOSIGNAL);
                    if (i == -1) {
                        LOG(ERR) << "Failed to send packet data for packet with tag " << entry->tag;
                        break;
                    }
                    n_sent += i;
                } while (n_sent < entry->size_bytes);

                // The send_completed created by the waking thread must already be visible to us because:
                // - The insertion is a seq_cst operation, and we have already awaited whether the entry in the send queue
                // is visible to us, where the insert is also a seq_cst operation that happened after the insert of
                // the send completed flag. Since we know an event at t=2 happened, and we have sequential consistency,
                // we know that the event at t=1 must be visible to us.
                (*internal_state->send_completed.get(entry->tag))->store(true, std::memory_order_release);
                delete entry;
            }
            LOG(INFO) << "MultiplexedIOSocket::run() interrupted, exiting send loop...";
        });
    }
    return true;
}

std::pair</* success */ bool, /* aborted */ bool> tinysockets::fullDuplexSendReceive(
    const MultiplexedIOSocket &tx_socket, const MultiplexedIOSocket &rx_socket,
    const uint64_t tag,
    const std::span<const std::byte> &tx_span, const std::span<std::byte> &recv_buffer_span,
    const size_t chunk_size,
    const std::function<void(size_t n_read, size_t total_bytes_recvd_yet)> &read_callback,
    const std::function<void(size_t n_sent, size_t total_bytes_sent_yet)> &send_callback,
    const std::function<bool(bool no_event)> &no_event_callback) {
    size_t bytes_sent = 0;
    size_t bytes_recvd = 0;

    const size_t total_tx_size = tx_span.size_bytes();
    const size_t total_rx_size = recv_buffer_span.size_bytes();

    bool was_no_event = false;
    while (bytes_sent < total_tx_size || bytes_recvd < total_rx_size) {
        bool no_event = true;
        bool performed_send = false;

        // 3a) Send if ready
        if (bytes_sent < total_tx_size) {
            const size_t next_chunk_size = std::min(chunk_size, total_tx_size - bytes_sent);
            const auto send_sub = tx_span.subspan(bytes_sent, next_chunk_size);
            if (tx_socket.sendBytesAsync(tag, send_sub)) {
                no_event = false;
                performed_send = true;
                const size_t n_sent = send_sub.size_bytes();
                send_callback(send_sub.size_bytes(), bytes_sent);
                bytes_sent += n_sent;
            } else {
                return {false, false};
            }
        }

        // 3b) Receive if ready
        if (bytes_recvd < total_rx_size) {
            const auto recv_sub = recv_buffer_span.subspan(bytes_recvd);
            if (auto n_read_opt = rx_socket.receiveBytesInplace(tag, recv_sub)) {
                const auto n_read = *n_read_opt;
                if (n_read > 0) {
                    no_event = false;
                    read_callback(n_read, bytes_recvd);
                    bytes_recvd += n_read;
                } else {
                    std::this_thread::yield();
                }
            } else {
                if (performed_send) {
                    tx_socket.awaitSendOp(tag);
                }
                return {false, false};
            }
        }
        if (performed_send) {
            tx_socket.awaitSendOp(tag);
        }

        if (no_event) {
            if (!no_event_callback(true)) {
                return {
                    /*success (more meaning no error here)*/ true, /* aborted */ true
                };
            }
            was_no_event = true;
        } else {
            if (was_no_event) {
                if (!no_event_callback(true)) {
                    return {
                        /*success (more meaning no error here)*/ true, /* aborted */ true
                    };
                }
                was_no_event = false;
            }
        }
    }

    return {
        /*success*/true, /*aborted*/false
    };
}


std::optional<size_t> tinysockets::MultiplexedIOSocket::receivePacketLength() const {
    uint64_t length;
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, &length, sizeof(length), 0);
        if (const bool running = this->running.load(std::memory_order_acquire); i == -1 || i == 0 || !running) {
            const std::string error_message = std::strerror(errno);
            if (running) {
                LOG(ERR) << "[MultiplexedIOSocket] Failed to receive packet length with error: " << error_message;
            }
            return std::nullopt;
        }
        n_received += i;
    } while (n_received < sizeof(length));
    return net_u64_to_host(length);
}


bool tinysockets::MultiplexedIOSocket::receivePacketData(std::span<std::uint8_t> &dst) const {
    if (!(flags & MODE_RX)) {
        LOG(ERR) << "MultiplexedIOSocket::receivePacketData() called on a socket without RX mode";
        return false;
    }
    size_t n_received = 0;
    do {
        const ssize_t i = recvvp(socket_fd, dst.data() + n_received, dst.size_bytes() - n_received, 0);
        if (i == -1) {
            const std::string error_message = std::strerror(errno);
            return false;
        }
        n_received += i;
    } while (n_received < dst.size_bytes() && running && socket_fd != 0);
    return true;
}


bool tinysockets::MultiplexedIOSocket::sendBytesAsync(const uint64_t tag, const std::span<const std::byte> &data) const {
    if (!(flags & MODE_TX)) {
        LOG(ERR) << "MultiplexedIOSocket::sendBytes() called on a socket without TX mode";
        return false;
    }
    if (!running.load(std::memory_order_acquire)) {
        LOG(ERR) << "MultiplexedIOSocket::sendBytes() called on a socket that is not running";
        return false;
    }
    auto *entry = new SendQueueEntry();
    entry->tag = tag;
    entry->data = std::span(reinterpret_cast<const uint8_t *>(data.data()), data.size());
    entry->size_bytes = data.size_bytes();

    // enqueue the SendQueueEntry
    {
        internal_state->send_completed.getOrCreate(tag)->store(false, std::memory_order_release);
        if (!internal_state->send_queue.enqueue(entry, true)) {
            LOG(ERR) << "MultiplexedIOSocket::sendBytes() failed to enqueue data; MPSC queue is full";
            return false;
        }
        tparkWake(internal_state->tx_park_handle);
    }

    return true;
}

void tinysockets::MultiplexedIOSocket::awaitSendOp(const uint64_t tag) const {
    const auto completed = *internal_state->send_completed.get(tag);
    // TODO: OPTIMIZE THIS
    while (!completed->load(std::memory_order_acquire)) {
        if (!running.load(std::memory_order_acquire)) {
            LOG(ERR) << "MultiplexedIOSocket::sendBytes() called on a socket that is not running";
            return;
        }
        std::this_thread::yield();
    }
}

std::optional<ssize_t> tinysockets::MultiplexedIOSocket::receiveBytesInplace(const uint64_t tag,
                                                                             const std::span<std::byte> &data) const {
    if (!(flags & MODE_RX)) {
        LOG(ERR) << "MultiplexedIOSocket::receiveBytes() called on a socket without RX mode";
        return std::nullopt;
    }

    if (!tparkIsParked(internal_state->rx_park_handle)) {
        return 0;
    }

    // set the receive-buffer for the tag
    {
        internal_state->receive_buffers.emplace(
            tag, std::span(reinterpret_cast<uint8_t *>(data.data()), data.size_bytes()));
        tparkWake(internal_state->rx_park_handle);
    }

    ReceiveQueueEntry entry;
    while (true) {
        if (!running.load(std::memory_order_acquire) || socket_fd == 0) {
            return std::nullopt;
        }
        if (!internal_state->receive_queues.contains(tag)) {
            continue;
        }
        const auto &queue = **internal_state->receive_queues.get(tag);
        const auto entry_ptr = queue->front();
        if (entry_ptr == nullptr) {
            continue;
        }
        entry = std::move(*entry_ptr);
        queue->pop();
        break;
    }

    if (entry.tag != tag) {
        LOG(BUG) << "Obtained packet from SPSCQueue with unexpected tag " << entry.tag << "; expected " << tag;
        return std::nullopt;
    }

    assert(entry.data == nullptr && entry.data_span.data() == data.data());

    return static_cast<ssize_t>(entry.data_span.size_bytes());
}

std::optional<std::unique_ptr<std::byte[]>> tinysockets::MultiplexedIOSocket::receiveBytes(const uint64_t tag,
    std::span<std::byte> &span_out) const {
    if (!(flags & MODE_RX)) {
        LOG(ERR) << "MultiplexedIOSocket::receiveBytes() called on a socket without RX mode";
        return std::nullopt;
    }

    if (!tparkIsParked(internal_state->rx_park_handle)) {
        return nullptr;
    }

    // set the receive-buffer for the tag
    {
        internal_state->receive_buffers.emplace(tag, std::span(static_cast<uint8_t *>(nullptr), 0));
        tparkWake(internal_state->rx_park_handle);
    }

    while (true) {
        if (!running.load(std::memory_order_acquire) || socket_fd == 0) {
            continue;
        }
        if (!internal_state->receive_queues.contains(tag)) {
            continue;
        }
        const auto &queue = **internal_state->receive_queues.get(tag);
        const auto entry_ptr = queue->front();
        if (entry_ptr == nullptr) {
            continue;
        }
        auto entry = std::move(*entry_ptr);
        queue->pop();

        if (entry.tag != tag) {
            LOG(BUG) << "Obtained packet from SPMCQueue with unexpected tag " << entry.tag << "; expected " << tag;
            return std::nullopt;
        }
        assert(entry.data.get() != nullptr);
        span_out = std::span(entry.data_span.data(), entry.data_span.size_bytes());
        return std::move(entry.data);
    }
}

void tinysockets::MultiplexedIOSocket::discardReceivedData(const uint64_t tag) const {
    {
        const auto opt = internal_state->receive_queues.get(tag);
        if (!opt) {
            return;
        }
        const auto &queue = **opt;
        while (queue->front() != nullptr) {
            queue->pop();
        }
    }
}

bool tinysockets::MultiplexedIOSocket::closeConnection() {
    if (socket_fd == 0) {
        return false;
    }

    // exchange socket fd & set running to false before actually closing it so loop reacts as early as possible
    running.store(false, std::memory_order_release);
    const int socket_fd = this->socket_fd;
    this->socket_fd = 0;

    if (internal_state->tx_park_handle != nullptr) {
        tparkWake(internal_state->tx_park_handle); // wake up tx thread such that it can exit
    }

    // Shut everything down.
    // This is needed to ensure recv() unblock and return an error.
    // Docker is pedantic about this.
    shutdown(socket_fd, SHUT_RDWR);

    // finally, close the socket
    closesocket(socket_fd);
    return true;
}

bool tinysockets::MultiplexedIOSocket::interrupt() {
    if (!running.load(std::memory_order_acquire)) {
        // already interrupted, either through discovery by io threads or external user
        return true;
    }

    running.store(false, std::memory_order_release);
    if (internal_state->tx_park_handle != nullptr) {
        tparkWake(internal_state->tx_park_handle); // wake up tx thread such that it can exit
    }

    // Shutdown both sides of the connection.
    // This is needed to ensure recv() unblocks and return an error.
    // Docker is pedantic about this.
    shutdown(socket_fd, SHUT_RDWR);

    // finally, close the socket
    closesocket(socket_fd);
    return true;
}

void tinysockets::MultiplexedIOSocket::join() {
    if (flags & MODE_RX) {
        if (recv_thread.joinable()) {
            recv_thread.join();
        }
    }
    if (flags & MODE_TX) {
        if (send_thread.joinable()) {
            send_thread.join();
        }
    }
}

bool tinysockets::MultiplexedIOSocket::isOpen() const { return running; }

const ccoip_socket_address_t &tinysockets::MultiplexedIOSocket::getConnectSockAddr() const { return connect_sockaddr; }

tinysockets::MultiplexedIOSocket::~MultiplexedIOSocket() {
    if (!interrupt()) [[unlikely]] {
        // no way to react to failure in destructor
    }
    join();
    delete internal_state;
}
