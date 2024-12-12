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
#define C2M_PACKET_REQUEST_SESSION_REGISTRATION_ID 1
#define C2M_PACKET_ACCEPT_NEW_PEERS_ID 2
#define C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID 3

    // M2C packets:
#define M2C_PACKET_SESSION_REGISTRATION_RESPONSE_ID 1
#define M2C_PACKET_NEW_PEERS_ID 2
#define M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID 3

    // P2P packets:
#define P2P_PACKET_HELLO_ID 1
#define P2P_PACKET_HELLO_ACK_ID 2

    // C2MPacketRequestSessionRegistration
    class C2MPacketRequestSessionRegistration final : public Packet {
    public:
        static packetId_t packet_id;
        uint16_t p2p_listen_port;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };

    // C2MPacketAcceptNewPeers
    class C2MPacketAcceptNewPeers final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // C2MPacketP2PConnectionsEstablished
    class C2MPacketP2PConnectionsEstablished final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // M2CPacketSessionRegistrationResponse
    class M2CPacketSessionRegistrationResponse final : public Packet {
    public:
        static packetId_t packet_id;

        boolean accepted;
        ccoip_uuid_t assigned_uuid;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };

    // M2CPacketNewPeers
    struct M2CPacketNewPeerInfo {
        ccoip_socket_address_t p2p_listen_addr;
        ccoip_uuid_t peer_uuid;
    };

    class M2CPacketNewPeers final : public Packet {
    public:
        static packetId_t packet_id;

        bool unchanged = false;

        std::vector<M2CPacketNewPeerInfo> new_peers;

        void serialize(PacketWriteBuffer &buffer) const override;

        void deserialize(PacketReadBuffer &buffer) override;
    };

    // M2CPacketP2PConnectionsEstablished
    class M2CPacketP2PConnectionsEstablished final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // P2PPacketHello
    class P2PPacketHello final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // P2PPacketHelloAck
    class P2PPacketHelloAck final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };
}
