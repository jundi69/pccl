#pragma once

#include <ccoip_inet.h>
#include <ccoip_packet.hpp>
#include <pccl_log.hpp>

#include <optional>
#include <functional>
#include <thread>
#include <span>

#ifdef WIN32
typedef long long int ssize_t;
#else
#include <sys/types.h>
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
        ServerSocketState *server_socket_state;
        std::thread server_thread;

        bool bound = false;
        bool listening = false;
        bool running = false;
        bool interrupted = false;

    public:
        explicit ServerSocket(const ccoip_socket_address_t &listen_address);

        ServerSocket(const ServerSocket &other) = delete;

        ServerSocket(ServerSocket &&other) = delete;

        ServerSocket &operator=(const ServerSocket &other) = delete;

        ServerSocket &operator=(ServerSocket &&other) = delete;

        /// Returns false if already bound or if listen_address is invalid
        [[nodiscard]] bool bind();

        /// Returns false if already listening
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
            packet.deserialize(buffer);
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
    private:
        int socket_fd;
        ccoip_socket_address_t connect_sockaddr;

    public:
        explicit BlockingIOSocket(const ccoip_socket_address_t &address);

        BlockingIOSocket(const BlockingIOSocket &other) = delete;

        BlockingIOSocket(BlockingIOSocket &&other) = delete;

        BlockingIOSocket &operator=(const BlockingIOSocket &other) = delete;

        BlockingIOSocket &operator=(BlockingIOSocket &&other) = delete;

        [[nodiscard]] bool establishConnection();

        [[nodiscard]] bool closeConnection();

        [[nodiscard]] bool isOpen() const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(const T &packet) {
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendLtvPacket(id, buffer);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receivePacket() {
            const ccoip::packetId_t id = T::packet_id;
            return receiveLtvPacket<T>(id);
        }

    private:
        // Packet decoding / encoding functions
        [[nodiscard]] bool sendLtvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        [[nodiscard]] size_t receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<std::uint8_t> &dst) const;

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveLtvPacket(const ccoip::packetId_t packet_id) {
            const size_t length = receivePacketLength();
            const std::unique_ptr<uint8_t[]> data_ptr{new uint8_t[length]};
            std::span data{data_ptr.get(), length};
            if (!receivePacketData(data)) {
                LOG(ERR) << "Failed to receive packet data for packet ID " << packet_id << " with length " << data.
                        size();
                return std::nullopt;
            }
            PacketReadBuffer buffer{data.data(), data.size()};
            const auto actual_packet_id = buffer.read<ccoip::packetId_t>();
            if (actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            T packet{};
            packet.deserialize(buffer);
            return packet;
        }
    };
};
