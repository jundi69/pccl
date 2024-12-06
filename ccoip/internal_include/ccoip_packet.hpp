#pragma once

#include "ccoip_packet_buffer.hpp"

namespace ccoip {
    typedef uint16_t packetId_t;

    class Packet {
    public:
        Packet() = default;

        virtual ~Packet() = default;

        virtual void serialize(PacketWriteBuffer &buffer) const = 0;

        virtual void deserialize(PacketReadBuffer &buffer) = 0;
    };

    class EmptyPacket : public Packet {
    public:

        static size_t serialized_size;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };
}
