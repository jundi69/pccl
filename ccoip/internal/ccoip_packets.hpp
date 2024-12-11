#pragma once

#include <ccoip_packet.hpp>
#include <ccoip_inet.h>
#include <ccoip_packet_buffer.hpp>
#include <ccoip_types.hpp>

namespace ccoip {
    // Definitions:
    // C2M: Client to Master
    // M2C: Master to Client
    // P2P: Peer to Peer
    // P2M: Peer to Master

    // C2M packets:
#define C2M_PACKET_REQUEST_SESSION_JOIN_ID 1
#define C2M_PACKET_ACCEPT_NEW_PEERS_ID 2

    // M2C packets:
#define C2M_PACKET_JOIN_RESPONSE_ID 1
#define C2M_PACKET_NEW_PEERS_ID 2

    // C2MPacketRequestSessionJoin
    class C2MPacketRequestSessionJoin final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // C2MPacketAcceptNewPeers
    class C2MPacketAcceptNewPeers final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // M2CPacketJoinResponse
    class M2CPacketJoinResponse final : public Packet {
    public:
        static packetId_t packet_id;

        boolean accepted;
        ccoip_uuid_t assigned_uuid;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };

    // M2CPacketNewPeers
    struct M2CPacketNewPeerInfo {
        ccoip_inet_address_t inet_address;
        ccoip_uuid_t peer_uuid;
    };

    class M2CPacketNewPeers final : public Packet {
    public:
        static packetId_t packet_id;

    public:
        std::vector<M2CPacketNewPeerInfo> new_peers;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };
}
