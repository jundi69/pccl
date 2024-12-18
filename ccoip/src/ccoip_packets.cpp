#include "ccoip_packets.hpp"

size_t ccoip::EmptyPacket::serialized_size = 0;

// C2MPacketRequestSessionRegistration
ccoip::packetId_t ccoip::C2MPacketRequestSessionRegistration::packet_id = C2M_PACKET_REQUEST_SESSION_REGISTRATION_ID;

void ccoip::C2MPacketRequestSessionRegistration::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint16_t>(p2p_listen_port);
}

bool ccoip::C2MPacketRequestSessionRegistration::deserialize(PacketReadBuffer &buffer) {
    p2p_listen_port = buffer.read<uint16_t>();
    return true;
}

// C2MPacketAcceptNewPeers
ccoip::packetId_t ccoip::C2MPacketAcceptNewPeers::packet_id = C2M_PACKET_ACCEPT_NEW_PEERS_ID;

// C2MPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::C2MPacketP2PConnectionsEstablished::packet_id = C2M_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// C2MPacketSyncSharedState
ccoip::packetId_t ccoip::C2MPacketSyncSharedState::packet_id = C2M_PACKET_SYNC_SHARED_STATE_ID;

void ccoip::C2MPacketSyncSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint64_t>(shared_state_revision);
    buffer.write<boolean>(ignore_hashes);
    buffer.write<uint64_t>(shared_state_hashes.size());
    for (const auto &entry: shared_state_hashes) {
        buffer.writeString(entry.key);
        if (!ignore_hashes) {
            buffer.write<uint64_t>(entry.hash);
        }
    }
}

bool ccoip::C2MPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    shared_state_revision = buffer.read<uint64_t>();
    ignore_hashes = buffer.read<boolean>();
    const auto n_entries = buffer.read<uint64_t>();
    shared_state_hashes.reserve(n_entries);
    for (size_t i = 0; i < n_entries; i++) {
        SharedStateHashEntry entry{};
        entry.key = buffer.readString();
        if (!ignore_hashes) {
            entry.hash = buffer.read<uint64_t>();
        } else {
            entry.hash = 0;
        }
        shared_state_hashes.push_back(entry);
    }
    return true;
}

// EmptyPacket
void ccoip::EmptyPacket::serialize(PacketWriteBuffer &buffer) const {
    // do nothing
}

bool ccoip::EmptyPacket::deserialize(PacketReadBuffer &buffer) {
    // do nothing
    return true;
}

// M2CPacketSessionRegistrationResponse
ccoip::packetId_t ccoip::M2CPacketSessionRegistrationResponse::packet_id = M2C_PACKET_SESSION_REGISTRATION_RESPONSE_ID;


void ccoip::M2CPacketSessionRegistrationResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(accepted);
    buffer.writeFixedArray(assigned_uuid.data);
}

bool ccoip::M2CPacketSessionRegistrationResponse::deserialize(PacketReadBuffer &buffer) {
    accepted = buffer.read<boolean>();
    assigned_uuid.data = buffer.readFixedArray<uint8_t, CCOIP_UUID_N_BYTES>();
    return true;
}

static void writeSocketAddress(PacketWriteBuffer &buffer, const ccoip_socket_address_t &socket_address) {
    buffer.write<bool>(socket_address.inet.protocol == inetIPv4);
    if (socket_address.inet.protocol == inetIPv4) {
        std::array<uint8_t, 4> ipv4_data{};
        for (size_t i = 0; i < 4; i++) {
            ipv4_data[i] = socket_address.inet.ipv4.data[i];
        }
        buffer.writeFixedArray(ipv4_data);
    } else if (socket_address.inet.protocol == inetIPv6) {
        std::array<uint8_t, 16> ipv6_data{};
        for (size_t i = 0; i < 16; i++) {
            ipv6_data[i] = socket_address.inet.ipv6.data[i];
        }
        buffer.writeFixedArray(ipv6_data);
    }
    buffer.write<uint16_t>(socket_address.port);
}

// M2CPacketNewPeers
void ccoip::M2CPacketNewPeers::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(unchanged);
    if (unchanged) {
        return;
    }
    buffer.write<uint64_t>(new_peers.size());
    for (auto &new_peer: new_peers) {
        writeSocketAddress(buffer, new_peer.p2p_listen_addr);
        buffer.writeFixedArray(new_peer.peer_uuid.data);
    }
}

static ccoip_socket_address_t readSocketAddress(PacketReadBuffer &buffer) {
    ccoip_socket_address_t address{};
    if (buffer.read<bool>()) {
        address.inet.protocol = inetIPv4;
        for (unsigned char &octet: address.inet.ipv4.data) {
            octet = buffer.read<uint8_t>();
        }
    } else {
        address.inet.protocol = inetIPv6;
        for (unsigned char &octet: address.inet.ipv6.data) {
            octet = buffer.read<uint8_t>();
        }
    }

    const auto port = buffer.read<uint16_t>();
    address.port = port;
    return address;
}

bool ccoip::M2CPacketNewPeers::deserialize(PacketReadBuffer &buffer) {
    unchanged = buffer.read<boolean>();
    if (unchanged) {
        return true;
    }
    const auto n_peers = buffer.read<uint64_t>();
    new_peers.reserve(n_peers);
    for (size_t i = 0; i < n_peers; i++) {
        M2CPacketNewPeerInfo peer_info{};
        peer_info.p2p_listen_addr = readSocketAddress(buffer);

        for (unsigned char &byte: peer_info.peer_uuid.data) {
            byte = buffer.read<uint8_t>();
        }
        new_peers.push_back(peer_info);
    }
    return true;
}

// M2CPacketSyncSharedState
ccoip::packetId_t ccoip::M2CPacketSyncSharedState::packet_id = M2C_PACKET_SYNC_SHARED_STATE_ID;

void ccoip::M2CPacketSyncSharedState::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<boolean>(is_outdated);
    writeSocketAddress(buffer, distributor_address);
}

bool ccoip::M2CPacketSyncSharedState::deserialize(PacketReadBuffer &buffer) {
    is_outdated = buffer.read<boolean>();
    distributor_address = readSocketAddress(buffer);
    return true;
}

// C2SPacketRequestSharedState
ccoip::packetId_t ccoip::C2SPacketRequestSharedState::packet_id = C2S_PACKET_REQUEST_SHARED_STATE_ID;

// S2CPacketSharedStateResponse
ccoip::packetId_t ccoip::S2CPacketSharedStateResponse::packet_id = S2C_PACKET_SHARED_STATE_RESPONSE_ID;

void ccoip::S2CPacketSharedStateResponse::serialize(PacketWriteBuffer &buffer) const {
    buffer.write<uint8_t>(static_cast<uint8_t>(status));
    buffer.write<uint64_t>(revision);

    // write entries
    buffer.write<uint64_t>(entries.size());
    for (const auto &entry: entries) {
        buffer.writeString(entry.key);
        buffer.writeContents(entry.buffer.data(), entry.buffer.size_bytes());
    }
}

bool ccoip::S2CPacketSharedStateResponse::deserialize(PacketReadBuffer &buffer) {
    status = static_cast<SharedStateResponseStatus>(buffer.read<uint8_t>());
    revision = buffer.read<uint64_t>();

    // read entries
    const auto n_entries = buffer.read<uint64_t>();
    entries.reserve(n_entries);
    for (size_t i = 0; i < n_entries; i++) {
        SharedStateEntry entry{};
        entry.key = buffer.readString();
        const auto length = buffer.read<uint64_t>();
        if (length > buffer.remaining()) {
            return false;
        }
        buffer.readContents(entry.buffer.data(), length);
        entries.push_back(entry);
    }
    return true;
}

// M2CPacketNewPeers
ccoip::packetId_t ccoip::M2CPacketNewPeers::packet_id = M2C_PACKET_NEW_PEERS_ID;

// M2CPacketP2PConnectionsEstablished
ccoip::packetId_t ccoip::M2CPacketP2PConnectionsEstablished::packet_id = M2C_PACKET_P2P_CONNECTIONS_ESTABLISHED_ID;

// P2PPacketHello
ccoip::packetId_t ccoip::P2PPacketHello::packet_id = P2P_PACKET_HELLO_ID;

// P2PPacketHelloAck
ccoip::packetId_t ccoip::P2PPacketHelloAck::packet_id = P2P_PACKET_HELLO_ACK_ID;
