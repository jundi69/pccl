#pragma once

#include <ccoip_inet.h>
#include <ccoip_packet.hpp>
#include <pccl_log.hpp>

#include <optional>
#include <functional>
#include <thread>
#include <span>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <pccl/common/cast_utils.hpp>

#ifdef WIN32
typedef long long int ssize_t;
struct sockaddr_in;
#else
#include <sys/types.h>
#include <netinet/in.h>
#endif

struct uv_server_stream_t;
struct uv_stream_s;
struct uv_buf_t;
struct uv_handle_s;

namespace tinysockets {
    struct ServerSocketState;

    using ServerSocketReadCallback = std::function<void(ccoip_socket_address_t, const std::span<std::uint8_t> &)>;
    using ServerSocketJoinCallback = std::function<void(const ccoip_socket_address_t &)>;
    using ServerSocketCloseCallback = std::function<void(const ccoip_socket_address_t &)>;

    class ServerSocket final {
    private:
        ccoip_socket_address_t listen_address;
        bool bump_port_on_failure;

        ServerSocketState *server_socket_state;
        std::thread server_thread;

        bool bound = false;
        bool running = false;
        bool interrupted = false;

    public:
        /// binds to the specified socket address
        explicit ServerSocket(const ccoip_socket_address_t &listen_address);

        /// bind to any free port above the given port
        explicit ServerSocket(const ccoip_inet_address_t &inet_address, uint16_t above_port);

        ServerSocket(const ServerSocket &other) = delete;

        ServerSocket(ServerSocket &&other) = delete;

        ServerSocket &operator=(const ServerSocket &other) = delete;

        ServerSocket &operator=(ServerSocket &&other) = delete;

        /// Returns false if already bound or if listen_address is invalid
        [[nodiscard]] bool listen();

        /// Returns false if already running
        [[nodiscard]] bool runAsync();

        /// Returns false if not running
        [[nodiscard]] bool interrupt();

        /// Wait for the server thread to exit
        void join();

        /// Add a callback to be called when new data is received from a client
        void addReadCallback(const ServerSocketReadCallback &callback) const;

        /// Add a callback to be called when a client connection is closed
        void addCloseCallback(const ServerSocketCloseCallback &callback) const;

        /// Add a callback to be called when a client joins
        void addJoinCallback(const ServerSocketJoinCallback &callback) const;

        /// Closes the client connection associated with the given socket address
        /// Returns false if the client connection does not exist or if the server is not running
        [[nodiscard]] bool closeClientConnection(const ccoip_socket_address_t &client_address) const;

        /// Closes all client connections
        [[nodiscard]] bool closeAllClientConnections() const;

        /// Returns the thread ID of the server thread
        [[nodiscard]] std::thread::id getServerThreadId() const;

        /// Returns the port the server is listening on; returns 0 if not listening
        [[nodiscard]] uint16_t getListenPort() const;

        ~ServerSocket();

        // Packet decoding / encoding functions

        /// Receives a packet from the client associated with the given socket address
        /// Returns std::nullopt if the server is not running or packet reception fails
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket(PacketReadBuffer &buffer) {
            const ccoip::packetId_t id = T::packet_id;
            return receiveLtvPacket<T>(id, buffer);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveLtvPacket(const ccoip::packetId_t packet_id, PacketReadBuffer &buffer) {
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }

        /// Sends a packet to the client associated with the given socket address
        /// Returns false if the server is not running or the client connection does not exist
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const ccoip_socket_address_t &client_address, const T &packet) {
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(client_address, id, buffer);
        }

        [[nodiscard]] bool sendLtvPacket(const ccoip_socket_address_t &client_address,
                                         const ccoip::packetId_t packet_id,
                                         const PacketWriteBuffer &buffer) {
            PacketWriteBuffer complete_packet{};
            complete_packet.write<uint64_t>(buffer.size() + sizeof(ccoip::packetId_t));
            complete_packet.write(packet_id);
            complete_packet.writeContents(buffer.data(), buffer.size());
            return sendRawPacket(client_address, complete_packet);
        }

    private:
        void onAsyncSignal() const;

        void onNewConnection(uv_server_stream_t *server, int status);

        void onClientRead(uv_stream_s *stream, ssize_t n_read, const uv_buf_t *buf) const;

        void onClientClose(uv_handle_s *handle) const;

        [[nodiscard]] std::optional<ccoip_socket_address_t> getUvStreamAddressCached(uv_stream_s *stream) const;

        void performLoopShutdown() const;

        [[nodiscard]] bool sendRawPacket(const ccoip_socket_address_t &client_address,
                                         const PacketWriteBuffer &buffer);
    };

    class BlockingIOSocket final {
        int socket_fd;
        ccoip_socket_address_t connect_sockaddr;

        std::mutex send_mutex;
        std::mutex recv_mutex;

        std::mutex unmatched_packets_mutex;

        struct UnmatchedPacket final {
            ccoip::packetId_t packet_id;
            std::unique_ptr<uint8_t[]> data;
            size_t byte_size;
        };

        std::vector<UnmatchedPacket> unmatched_packets{};
        std::condition_variable cv_packets_published{};

    public:
        explicit BlockingIOSocket(const ccoip_socket_address_t &address);

        explicit BlockingIOSocket(int socket_fd);

        BlockingIOSocket(const BlockingIOSocket &other) = delete;

        BlockingIOSocket(BlockingIOSocket &&other) = delete;

        BlockingIOSocket &operator=(const BlockingIOSocket &other) = delete;

        BlockingIOSocket &operator=(BlockingIOSocket &&other) = delete;

        [[nodiscard]] bool establishConnection();

        [[nodiscard]] bool closeConnection();

        [[nodiscard]] bool isOpen() const;

        [[nodiscard]] const ccoip_socket_address_t &getConnectSockAddr() const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const T &packet) {
            std::lock_guard guard{send_mutex};
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(id, buffer);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket() {
            std::lock_guard guard{recv_mutex};
            const ccoip::packetId_t id = T::packet_id;
            return receiveLtvPacket<T>(id);
        }

        /// Returns the next received packet that matches the given predicate.
        /// This function is thread-safe.
        /// If two threads are waiting for packets, one thread will read while the other thread waits.
        /// After receiving the packet, the thread that has received the packet will check the predicate.
        /// If this predicate matches, the packet is returned. Otherwise, a new packet is read.
        /// In case a packet is received that does not match the predicate and a different thread's predicate matches,
        /// the other thread will obtain and return this packet.
        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveMatchingPacket(const std::function<bool(const T &)> &predicate) {
            const ccoip::packetId_t expected_packet_id = T::packet_id;

            bool may_read = false;
            while (true) {
                std::unique_lock u_guard{unmatched_packets_mutex};

                // first check if there are any unmatched packets
                if (unmatched_packets.empty()) {
                    // if no, try to acquire the read mutex and become the reading thread
                    may_read = recv_mutex.try_lock();
                    if (may_read) {
                        break;
                    }
                    // if that fails, wait until the reading thread notifies us of a published packet
                    cv_packets_published.wait(u_guard);
                }

                // protect against spurious wake-ups and check if there are any unmatched packets
                if (!unmatched_packets.empty()) {
                    // if there are, check if any of them match the predicate
                    for (auto it = unmatched_packets.begin(); it != unmatched_packets.end();) {
                        if (it->packet_id == expected_packet_id) {
                            // if the packet is of the type we expect, attempt to parse it
                            PacketReadBuffer buffer{it->data.get(), it->byte_size};
                            if (const auto packet_id = buffer.read<ccoip::packetId_t>();
                                packet_id != expected_packet_id) [[unlikely]] {
                                LOG(ERR) <<
                                        "Packet ID logged in unmatched packet header does not match packet id encoded in attached packet buffer";
                                return std::nullopt;
                            }
                            T packet{};
                            if (!packet.deserialize(buffer)) {
                                LOG(ERR) << "Failed to deserialize packet with ID " << expected_packet_id;
                                return std::nullopt;
                            }
                            if (predicate(packet)) {
                                // if the packet matches, remove it from the unmatched packets list and return it
                                unmatched_packets.erase(it);
                                return std::move(packet);
                            }
                        }
                        // if the packet does not match, keep it in the unmatched packets list
                        ++it;
                    }
                }
            }

            // this thread has permission to exclusively read from the socket
            {
                std::unique_lock guard{recv_mutex, std::adopt_lock}; // mutex already acquired, only guard safe release
                while (true) {
                    // read the next packet
                    size_t packet_buffer_size{};
                    std::unique_ptr<uint8_t[]> packet_buffer{};

                    [this, &packet_buffer, &packet_buffer_size, expected_packet_id] {
                        const std::optional<size_t> length_opt = receivePacketLength();
                        if (!length_opt) {
                            LOG(ERR) << "Failed to receive packet length for packet ID " << expected_packet_id;
                            packet_buffer = nullptr;
                            return;
                        }
                        const size_t length = *length_opt;
                        std::unique_ptr<uint8_t[]> data_ptr{new uint8_t[length]};
                        std::span data{data_ptr.get(), length};
                        if (!receivePacketData(data)) {
                            LOG(ERR) << "Failed to receive packet data for packet ID " << expected_packet_id <<
                                    " with length "
                                    << data.
                                    size();
                            packet_buffer = nullptr;
                            return;
                        }
                        packet_buffer_size = data.size_bytes();
                        packet_buffer = std::move(data_ptr);
                    }();

                    if (!packet_buffer) {
                        // wake up all the waiting threads because we are relinquishing the read thread title
                        // even though we failed to read the packet
                        {
                            std::unique_lock u_guard{unmatched_packets_mutex};
                            cv_packets_published.notify_all();
                        }
                        return std::nullopt;
                    }

                    // attempt to parse the packet
                    ccoip::packetId_t actual_packet_id; {
                        PacketReadBuffer buffer{packet_buffer.get(), packet_buffer_size};
                        actual_packet_id = buffer.read<ccoip::packetId_t>();

                        // check if the read packet is of the type we expect. if it is not,
                        // we still must not drop the packet, but instead publish it to other threads
                        if (actual_packet_id == expected_packet_id) {
                            T packet{};
                            if (!packet.deserialize(buffer)) {
                                LOG(ERR) << "Failed to deserialize packet with ID " << expected_packet_id;
                                return std::nullopt;
                            }
                            if (predicate(packet)) {
                                // wake up all the waiting threads
                                {
                                    std::unique_lock u_guard{unmatched_packets_mutex};
                                    cv_packets_published.notify_all();
                                }
                                return std::move(packet);
                            }
                        }
                    }

                    // publish unmatched packet to other threads
                    {
                        std::unique_lock u_guard{unmatched_packets_mutex};
                        unmatched_packets.push_back({actual_packet_id, std::move(packet_buffer), packet_buffer_size});
                        cv_packets_published.notify_all();
                    }
                }
            }
        }

    private:
        // Packet decoding / encoding functions
        [[nodiscard]] bool sendLtvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        [[nodiscard]] std::optional<size_t> receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<std::uint8_t> &dst) const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveLtvPacket(const ccoip::packetId_t packet_id) {
            const std::optional<size_t> length_opt = receivePacketLength();
            if (!length_opt) {
                return std::nullopt;
            }
            const size_t length = *length_opt;
            const std::unique_ptr<uint8_t[]> data_ptr{new uint8_t[length]};
            std::span data{data_ptr.get(), length};
            if (!receivePacketData(data)) {
                LOG(ERR) << "Failed to receive packet data for packet ID " << packet_id << " with length " << data.
                        size();
                return std::nullopt;
            }
            PacketReadBuffer buffer{data.data(), data.size()};
            if (const auto actual_packet_id = buffer.read<ccoip::packetId_t>(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            if (!packet.deserialize(buffer)) {
                LOG(ERR) << "Failed to deserialize packet with ID " << packet_id;
                return std::nullopt;
            }
            return packet;
        }
    };

    using BlockingServerSocketJoinCallback = std::function<void(const ccoip_socket_address_t &, std::unique_ptr<BlockingIOSocket> &)>;

    class BlockingIOServerSocket final {
    private:
        ccoip_socket_address_t listen_address;
        bool bump_port_on_failure;
        std::thread server_thread;

        bool bound = false;
        bool interrupted = false;
        std::atomic<bool> running{false};

        int socket_fd;

        BlockingServerSocketJoinCallback join_callback = nullptr;

    public:
        /// binds to the specified socket address
        explicit BlockingIOServerSocket(const ccoip_socket_address_t &listen_address);

        /// bind to any free port above the given port
        explicit BlockingIOServerSocket(const ccoip_inet_address_t &inet_address, uint16_t above_port);

        BlockingIOServerSocket(const BlockingIOServerSocket &other) = delete;

        BlockingIOServerSocket(BlockingIOServerSocket &&other) = delete;

        BlockingIOServerSocket &operator=(const BlockingIOServerSocket &other) = delete;

        BlockingIOServerSocket &operator=(BlockingIOServerSocket &&other) = delete;

        /// Returns false if already bound or if listen_address is invalid
        [[nodiscard]] bool listen();

        /// Returns false if already running
        [[nodiscard]] bool runAsync();

        /// Returns false if not running
        [[nodiscard]] bool interrupt();

        /// Wait for the server thread to exit
        void join();

        /// Sets the callback to be called when a client joins
        void setJoinCallback(const BlockingServerSocketJoinCallback &callback);

        /// Returns the port the server is listening on; returns 0 if not listening
        [[nodiscard]] uint16_t getListenPort() const;

    private:
        /// Called when a new connection is established
        void onNewConnection(int client_socket, sockaddr_in sockaddr_in) const;
    };
};
