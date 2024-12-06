#pragma once

#include "ccoip_packet_buffer.hpp"

namespace ccoip {
    typedef uint16_t packetId_t;

    class Packet {
    public:
        virtual ~Packet() = default;

        Packet(const Packet &other) = delete;

        Packet(const Packet &&other) = delete;

        virtual void serialize(PacketWriteBuffer &buffer) = 0;

        virtual void deserialize(PacketReadBuffer &buffer) = 0;
    };

    class EmptyPacket : public Packet {
    public:
        EmptyPacket();

        static size_t serialized_size;

        void serialize(PacketWriteBuffer &buffer) override;

        void deserialize(PacketReadBuffer &buffer) override;
    };
}
