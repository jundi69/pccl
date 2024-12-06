#include "ccoip_packets.hpp"

size_t ccoip::EmptyPacket::serialized_size = 0;

ccoip::packetId_t ccoip::C2MPacketAcceptNewPeers::packet_id = C2M_PACKET_ACCEPT_NEW_PEERS_ID;

// EmptyPacket
void ccoip::EmptyPacket::serialize(PacketWriteBuffer &buffer) const {
    // do nothing
}

void ccoip::EmptyPacket::deserialize(PacketReadBuffer &buffer) {
    // do nothing
}

// M2CPacketNewPeers
void ccoip::M2CPacketNewPeers::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(new_peers.size());
    for (auto &new_peer: new_peers) {
        buffer.write<boolean>(new_peer.inet_address.protocol == inetIPv4);
        if (new_peer.inet_address.protocol == inetIPv4) {
            std::array<uint8_t, 4> ipv4_data{};
            for (size_t i = 0; i < 4; i++) {
                ipv4_data[i] = new_peer.inet_address.address.ipv4.data[i];
            }
            buffer.writeFixedArray(ipv4_data);
        } else if (new_peer.inet_address.protocol == inetIPv6) {
            std::array<uint8_t, 16> ipv6_data{};
            for (size_t i = 0; i < 16; i++) {
                ipv6_data[i] = new_peer.inet_address.address.ipv6.data[i];
            }
            buffer.writeFixedArray(ipv6_data);
        }
        buffer.writeFixedArray(new_peer.peer_uuid.data);
    }
}

void ccoip::M2CPacketNewPeers::deserialize(PacketReadBuffer &buffer) {
    const auto n_peers = buffer.read<uint64_t>();
    new_peers.reserve(n_peers);
    for (size_t i = 0; i < n_peers; i++) {
        M2CPacketNewPeerInfo peer_info{};
        if (buffer.read<boolean>()) {
            peer_info.inet_address.protocol = inetIPv4;
            for (unsigned char &octet: peer_info.inet_address.address.ipv4.data) {
                octet = buffer.read<uint8_t>();
            }
        } else {
            peer_info.inet_address.protocol = inetIPv6;
            for (unsigned char &octet: peer_info.inet_address.address.ipv6.data) {
                octet = buffer.read<uint8_t>();
            }
        }
        for (unsigned char &byte: peer_info.peer_uuid.data) {
            byte = buffer.read<uint8_t>();
        }
        new_peers.push_back(peer_info);
    }
}

ccoip::packetId_t ccoip::M2CPacketNewPeers::packet_id = C2M_PACKET_NEW_PEERS_ID;
