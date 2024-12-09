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

namespace tinysockets {

    struct ServerSocketState;

    using ServerSocketReadCallback = std::function<void(ccoip_socket_address_t, std::span<std::uint8_t>)>;

    class ServerSocket final {
    private:
        ccoip_socket_address_t listen_address;
        ServerSocketState *server_socket_state;
        std::thread server_thread;

        bool bound = false;
        bool listening = false;
        bool running = false;

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
        [[nodiscard]] bool interrupt() const;

        /// Wait for the server thread to exit
        void join();

        /// Add a callback to be called when new data is received from a client
        void addReadCallback(const ServerSocketReadCallback &callback) const;

        ~ServerSocket();

    private:
        void onAsyncSignal() const;

        void onNewConnection(uv_server_stream_t *server, int status);

        void onClientRead(uv_stream_s *stream, ssize_t n_read, const uv_buf_t *buf) const;
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

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] bool sendPacket(T &packet) {
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendTlvPacket(id, buffer);
        }

        template<typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> recvPacket() {
            const ccoip::packetId_t id = T::packet_id;
            return receiveTlvPacket<T>(id);
        }

    private:
        [[nodiscard]] bool sendTlvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        [[nodiscard]] ccoip::packetId_t receivePacketType() const;

        [[nodiscard]] size_t receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<std::uint8_t> dst) const;

        template <typename T> requires std::is_base_of_v<ccoip::Packet, T>
        [[nodiscard]] std::optional<T> receiveTlvPacket(const ccoip::packetId_t packet_id) {
            if (const ccoip::packetId_t actual_packet_id = receivePacketType(); actual_packet_id != packet_id) {
                LOG(ERR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            std::vector<std::uint8_t> data{};
            data.resize(receivePacketLength());
            if (!receivePacketData(data)) {
                LOG(ERR) << "Failed to receive packet data for packet ID " << packet_id << " with length " << data.size();
                return std::nullopt;
            }
            PacketReadBuffer buffer{data.data(), data.size()};
            T packet{};
            packet.deserialize(buffer);
            return packet;
        }
    };
};
