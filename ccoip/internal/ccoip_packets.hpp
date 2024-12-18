#pragma once

#include <ccoip_packet.hpp>
#include <ccoip_inet.h>
#include <ccoip_packet_buffer.hpp>
#include <ccoip_types.hpp>

namespace ccoip {
    // Definitions:

    // --- Main CCoIP Protocol ---
    // C2M: Client to Master
    // M2C: Master to Client
    // P2P: Peer to Peer
    // P2M: Peer to Master

    // --- CCoIP Shared State Distribution Protocol ---
    // C2S: Client to Shared State Server
    // S2C: Shared State Server to Client

    // C2M packets:
#define C2M_PACKET_REQUEST_SESSION_REGISTRATION_ID 1
#define C2M_PACKET_ACCEPT_NEW_PEERS_ID 2
#define C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID 3
#define C2M_PACKET_SYNC_SHARED_STATE_ID 4

    // M2C packets:
#define M2C_PACKET_SESSION_REGISTRATION_RESPONSE_ID 1
#define M2C_PACKET_NEW_PEERS_ID 2
#define M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID 3
#define M2C_PACKET_SYNC_SHARED_STATE_ID 4

    // P2P packets:
#define P2P_PACKET_HELLO_ID 1
#define P2P_PACKET_HELLO_ACK_ID 2

    // C2S packets:
#define C2S_PACKET_REQUEST_SHARED_STATE_ID 1

    // S2C packets:
#define S2C_PACKET_SHARED_STATE_RESPONSE_ID 1

    // C2MPacketRequestSessionRegistration
    class C2MPacketRequestSessionRegistration final : public Packet {
    public:
        static packetId_t packet_id;
        uint16_t p2p_listen_port;

        void serialize(PacketWriteBuffer &buffer) const override;

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
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

    // C2MPacketSyncSharedState
    struct SharedStateHashEntry {
        std::string key;
        uint64_t hash;
    };

    class C2MPacketSyncSharedState final : public Packet {
    public:
        static packetId_t packet_id;

        uint64_t shared_state_revision;
        boolean ignore_hashes;
        std::vector<SharedStateHashEntry> shared_state_hashes;

        void serialize(PacketWriteBuffer &buffer) const override;

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
    };

    // M2CPacketSessionRegistrationResponse
    class M2CPacketSessionRegistrationResponse final : public Packet {
    public:
        static packetId_t packet_id;

        boolean accepted;
        ccoip_uuid_t assigned_uuid;

        void serialize(PacketWriteBuffer &buffer) const override;

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
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

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
    };

    // M2CPacketP2PConnectionsEstablished
    class M2CPacketP2PConnectionsEstablished final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // M2CPacketSyncSharedState
    class M2CPacketSyncSharedState final : public Packet {
    public:
        static packetId_t packet_id;
        boolean is_outdated;
        ccoip_socket_address_t distributor_address;

        void serialize(PacketWriteBuffer &buffer) const override;

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
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

    // C2SPacketRequestSharedState
    class C2SPacketRequestSharedState final : public EmptyPacket {
    public:
        static packetId_t packet_id;
    };

    // S2CPacketSharedStateResponse
    enum SharedStateResponseStatus {
        /// The shared state was successfully distributed
        SUCCESS = 1,

        /// The shared state is not distributed by this peer
        SHARED_STATE_NOT_DISTRIBUTED = 2,

        /// The peer is currently not in shared state distribution mode and thus
        /// refuses to distribute the shared state.
        NOT_IN_SHARED_STATE_DISTRIBUTION_MODE = 3,
    };

    struct SharedStateEntry {
        /// The key of the shared state entry
        std::string key;

        /// On the client:
        /// References memory into which the shared state entry will be copied.
        /// On the server:
        /// References memory from which the shared state entry will be copied.
        std::span<uint8_t> buffer;
    };

    class S2CPacketSharedStateResponse final : public Packet {
    public:
        static packetId_t packet_id;
        SharedStateResponseStatus status;
        uint64_t revision;
        std::vector<SharedStateEntry> entries;

        void serialize(PacketWriteBuffer &buffer) const override;

        [[nodiscard]] bool deserialize(PacketReadBuffer &buffer) override;
    };
}
