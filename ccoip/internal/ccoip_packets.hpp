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
#define C2M_PACKET_ACCEPT_NEW_PEERS_ID 1

    // M2C packets:
#define C2M_PACKET_NEW_PEERS_ID 1

    // C2MPacketAcceptNewPeers
    class C2MPacketAcceptNewPeers final : EmptyPacket {
        static packetId_t packet_id;
    };

    packetId_t C2MPacketAcceptNewPeers::packet_id = C2M_PACKET_ACCEPT_NEW_PEERS_ID;

    struct M2CPacketNewPeerInfo {
        ccoip_inet_address_t inet_address;
        ccoip_uuid_t peer_uuid;
    };

    // M2CPacketNewPeers
    class M2CPacketNewPeers final : public Packet {
        static packetId_t packet_id;

    public:
        std::vector<M2CPacketNewPeerInfo> new_peers;

        M2CPacketNewPeers();

        void serialize(PacketWriteBuffer &buffer) override;

        void deserialize(PacketReadBuffer &buffer) override;
    };

    packetId_t M2CPacketNewPeers::packet_id = C2M_PACKET_NEW_PEERS_ID;
}
