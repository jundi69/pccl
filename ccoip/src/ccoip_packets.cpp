#include "ccoip_packets.hpp"

size_t ccoip::EmptyPacket::serialized_size = 0;

// C2MPacketRequestSessionRegistration
ccoip::packetId_t ccoip::C2MPacketRequestSessionRegistration::packet_id = C2M_PACKET_REQUEST_SESSION_REGISTRATION_ID;

void ccoip::C2MPacketRequestSessionRegistration::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint16_t>(p2p_listen_port);
}

void ccoip::C2MPacketRequestSessionRegistration::deserialize(PacketReadBuffer &buffer) {
    p2p_listen_port = buffer.read<uint16_t>();
}

// C2MPacketAcceptNewPeers
ccoip::packetId_t ccoip::C2MPacketAcceptNewPeers::packet_id = C2M_PACKET_ACCEPT_NEW_PEERS_ID;

// C2MPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::C2MPacketP2PConnectionsEstablished::packet_id = C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// EmptyPacket
void ccoip::EmptyPacket::serialize(PacketWriteBuffer &buffer) const {
    // do nothing
}

void ccoip::EmptyPacket::deserialize(PacketReadBuffer &buffer) {
    // do nothing
}

// M2CPacketSessionRegistrationResponse
ccoip::packetId_t ccoip::M2CPacketSessionRegistrationResponse::packet_id = M2C_PACKET_SESSION_REGISTRATION_RESPONSE_ID;

void ccoip::M2CPacketSessionRegistrationResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(accepted);
    buffer.writeFixedArray(assigned_uuid.data);
}

void ccoip::M2CPacketSessionRegistrationResponse::deserialize(PacketReadBuffer &buffer) {
    accepted = buffer.read<boolean>();
    assigned_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
}

// M2CPacketNewPeers
void ccoip::M2CPacketNewPeers::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(unchanged);
    if (unchanged) {
        return;
    }
    buffer.write<uint64_t>(new_peers.size());
    for (auto &new_peer: new_peers) {
        buffer.write<bool>(new_peer.p2p_listen_addr.inet.protocol == inetIPv4);
        if (new_peer.p2p_listen_addr.inet.protocol == inetIPv4) {
            std::array<uint8_t, 4> ipv4_data{};
            for (size_t i = 0; i < 4; i++) {
                ipv4_data[i] = new_peer.p2p_listen_addr.inet.ipv4.data[i];
            }
            buffer.writeFixedArray(ipv4_data);
        } else if (new_peer.p2p_listen_addr.inet.protocol == inetIPv6) {
            std::array<uint8_t, 16> ipv6_data{};
            for (size_t i = 0; i < 16; i++) {
                ipv6_data[i] = new_peer.p2p_listen_addr.inet.ipv6.data[i];
            }
            buffer.writeFixedArray(ipv6_data);
        }
        buffer.write<uint16_t>(new_peer.p2p_listen_addr.port);
        buffer.writeFixedArray(new_peer.peer_uuid.data);
    }
}

void ccoip::M2CPacketNewPeers::deserialize(PacketReadBuffer &buffer) {
    unchanged = buffer.read<boolean>();
    if (unchanged) {
        return;
    }
    const auto n_peers = buffer.read<uint64_t>();
    new_peers.reserve(n_peers);
    for (size_t i = 0; i < n_peers; i++) {
        M2CPacketNewPeerInfo peer_info{};
        if (buffer.read<bool>()) {
            peer_info.p2p_listen_addr.inet.protocol = inetIPv4;
            for (unsigned char &octet: peer_info.p2p_listen_addr.inet.ipv4.data) {
                octet = buffer.read<uint8_t>();
            }
        } else {
            peer_info.p2p_listen_addr.inet.protocol = inetIPv6;
            for (unsigned char &octet: peer_info.p2p_listen_addr.inet.ipv6.data) {
                octet = buffer.read<uint8_t>();
            }
        }

        const auto port = buffer.read<uint16_t>();
        peer_info.p2p_listen_addr.port = port;

        for (unsigned char &byte: peer_info.peer_uuid.data) {
            byte = buffer.read<uint8_t>();
        }
        new_peers.push_back(peer_info);
    }
}

// M2CPacketNewPeers
ccoip::packetId_t ccoip::M2CPacketNewPeers::packet_id = M2C_PACKET_NEW_PEERS_ID;

// M2CPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::M2CPacketP2PConnectionsEstablished::packet_id = M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// P2PPacketHello
ccoip::packetId_t ccoip::P2PPacketHello::packet_id = P2P_PACKET_HELLO_ID;

// P2PPacketHelloAck
ccoip::packetId_t ccoip::P2PPacketHelloAck::packet_id = P2P_PACKET_HELLO_ACK_ID;
