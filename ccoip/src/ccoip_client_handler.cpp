#include "ccoip_client_handler.hpp"
#include "ccoip_types.hpp"
#include <pccl_log.hpp>

#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"

struct internal_inet_address_t {
    ccoip_inet_protocol_t protocol;

    union {
        ccoip_ipv4_address_t ipv4;
        ccoip_ipv6_address_t ipv6;
    } address;

    bool operator==(const internal_inet_address_t &rhs) const {
        if (protocol != rhs.protocol) {
            return false;
        }
        if (protocol == inetIPv4) {
            return memcmp(address.ipv4.data, rhs.address.ipv4.data, 4) == 0;
        }
        if (protocol == inetIPv6) {
            return memcmp(address.ipv6.data, rhs.address.ipv6.data, 16) == 0;
        }
        return false;
    }
};

template<>
struct std::hash<internal_inet_address_t> {
    std::size_t operator()(const internal_inet_address_t &inet_addr) const noexcept {
        std::size_t hash_value = 0;
        hash_value = hash_value * 31 + inet_addr.protocol;
        for (const auto &byte: inet_addr.address.ipv4.data) {
            hash_value = hash_value * 31 + byte;
        }
        for (const auto &byte: inet_addr.address.ipv6.data) {
            hash_value = hash_value * 31 + byte;
        }
        return hash_value;
    }
};

struct CCoIPClientState {
    std::unordered_map<internal_inet_address_t, std::vector<ccoip_uuid_t>> inet_addrs_to_uuids{};
};

ccoip::CCoIPClientHandler::CCoIPClientHandler(const ccoip_socket_address_t &address) : client_socket(address),
    client_state(new CCoIPClientState()) {
}

bool ccoip::CCoIPClientHandler::connect() {
    return client_socket.establishConnection();
}

// establishP2PConnection:
// establish socket connection
// send hello packet
// expect hello ack
// mark connection as tx side open (tx side = our connection to the other listening party)

// on p2p listen socket, on accept, wait for hello packet and reply hello ack
// the uuid it is transmitted in the hello, we verify the ip against what the master told
// us the ip of the peer is, and then we mark the connection as rx side open (rx side = their connection to us listening)


bool ccoip::CCoIPClientHandler::acceptNewPeers() {
    const C2MPacketAcceptNewPeers new_peers_packet{};
    if (!client_socket.sendPacket(new_peers_packet)) {
        return false;
    }

    auto response = client_socket.recvPacket<M2CPacketNewPeers>();
    if (!response) {
        return false;
    }

    /*// add all uuids received by master
    for (auto &new_peer: response.new_peers) {
        addPeer(new_peer.inet_addr, new_peer.uuid);
    }

    // create p2p connections
    for (auto &new_peer: response.new_peers) {
        p2p_connections.push_back(establishP2PConnection(new_peer));
    }

    // wait on signal until notify, check if all p2p connection
    */

    return true;
}

bool ccoip::CCoIPClientHandler::interrupt() {
    return true;
}

bool ccoip::CCoIPClientHandler::join() {
    return true;
}

ccoip::CCoIPClientHandler::~CCoIPClientHandler() {
    delete client_state;
}


void ccoip::CCoIPClientHandler::registerPeer(const ccoip_inet_address_t &address, const ccoip_uuid_t uuid) {
    internal_inet_address_t internal_address{
        .protocol = address.protocol
    };
    if (address.protocol == inetIPv4) {
        internal_address.address.ipv4 = address.address.ipv4;
    } else if (address.protocol == inetIPv6) {
        internal_address.address.ipv6 = address.address.ipv6;
    }
    auto uuids_per_ip = client_state->inet_addrs_to_uuids[internal_address];
    uuids_per_ip.push_back(uuid);
    if (uuids_per_ip.size() > 1) {
        LOG(WARN) << "Registered more than one peer per IP address " << CCOIP_INET_ADDR_TO_STRING(address);
    }
}
