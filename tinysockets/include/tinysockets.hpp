#pragma once

#include <ccoip_inet.h>
#include <ccoip_packet.hpp>
#include <pccl_log.hpp>
#include <thread>
#include <span>

struct uv_loop_s;
struct uv_tcp_s;
struct uv_async_s;
struct uv_server_stream_t;

struct uv_stream_s;
struct uv_buf_t;

namespace tinysockets {
    class ServerSocket {
        ccoip_socket_address_t listen_address;
        uv_loop_s *loop;
        uv_tcp_s *tcp_server;
        uv_async_s *async_handle;
        std::thread server_thread;

        bool bound = false;
        bool listening = false;
        bool running = false;

    public:
        explicit ServerSocket(const ccoip_socket_address_t &listen_address);

        ServerSocket(const ServerSocket &other) = delete; /// delete copy constructor
        ServerSocket(ServerSocket &&other) = delete; /// delete move constructor
        ServerSocket &operator=(const ServerSocket &other) = delete; /// delete assignment operator

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

        ~ServerSocket();

    private:
        void onAsyncSignal() const;

        void onNewConnection(uv_server_stream_t *server, int status) const;

        void onClientRead(uv_stream_s *stream, ssize_t n_read, const uv_buf_t *buf);
    };

    class BlockingIOSocket {
        int socket_fd;
        ccoip_socket_address_t connect_sockaddr;

    public:
        explicit BlockingIOSocket(const ccoip_socket_address_t &address);

        BlockingIOSocket(const BlockingIOSocket &other) = delete; /// delete copy constructor
        BlockingIOSocket(BlockingIOSocket &&other) = delete; /// delete move constructor)

        [[nodiscard]] bool establishConnection();

        template<typename T>
        [[nodiscard]] bool sendPacket(T &packet) {
            static_assert(std::is_base_of_v<ccoip::Packet, T>, "T must be a subclass of ccoip::Packet");
            const ccoip::packetId_t id = T::packet_id;
            PacketWriteBuffer buffer{};
            packet.serialize(buffer);
            return sendTlvPacket(id, buffer);
        }

        template<typename T>
        [[nodiscard]] std::optional<T> recvPacket() {
            static_assert(std::is_base_of_v<ccoip::Packet, T>, "T must be a subclass of ccoip::Packet");
            const ccoip::packetId_t id = T::packet_id;
            return receiveTlvPacket<T>(id);
        }

    private:
        [[nodiscard]] bool sendTlvPacket(ccoip::packetId_t packet_id, const PacketWriteBuffer &buffer) const;

        [[nodiscard]] ccoip::packetId_t receivePacketType() const;

        [[nodiscard]] size_t receivePacketLength() const;

        [[nodiscard]] bool receivePacketData(std::span<uint8_t> &dst) const;

        template<typename T>
        [[nodiscard]] std::optional<T> receiveTlvPacket(const ccoip::packetId_t packet_id) {
            static_assert(std::is_base_of_v<ccoip::Packet, T>, "T must be a subclass of ccoip::Packet");
            if (const ccoip::packetId_t actual_packet_id = receivePacketType(); actual_packet_id != packet_id) {
                LOG(ERROR) << "Expected packet ID " << packet_id << " but received " << actual_packet_id;
                return std::nullopt;
            }
            const uint64_t length = receivePacketLength();
            const std::unique_ptr<uint8_t[]> data_ptr{new uint8_t[length]};
            std::span data{data_ptr.get(), length};
            if (!receivePacketData(data)) {
                LOG(ERROR) << "Failed to receive packet data for packet ID " << packet_id << " with length " << length;
                return std::nullopt;
            }
            PacketReadBuffer buffer{data.data(), length};
            T packet{};
            packet.deserialize(buffer);
            return packet;
        }
    };
};
