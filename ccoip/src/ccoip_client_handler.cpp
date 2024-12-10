#include "ccoip_client_handler.hpp"
#include "ccoip_types.hpp"
#include <pccl_log.hpp>

#include "ccoip_inet_utils.hpp"
#include "ccoip_packets.hpp"

struct CCoIPClientState {
    std::unordered_map<internal_inet_address_t, std::vector<ccoip_uuid_t> > inet_addrs_to_uuids{};
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

    auto response = client_socket.receivePacket<M2CPacketNewPeers>();
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

    // wait on signal until notify, check if all p2p connection have been established tx side and rx side
    */

    return true;
}

bool ccoip::CCoIPClientHandler::updateTopology() {
    return true;
}

bool ccoip::CCoIPClientHandler::interrupt() {
    if (!client_socket.closeConnection()) [[unlikely]] {
        return false;
    }
    return true;
}

bool ccoip::CCoIPClientHandler::join() {
    // api for future proofing if we need to ever wait for async operations to finish
    // to clean up the client handler
    return true;
}

ccoip::CCoIPClientHandler::~CCoIPClientHandler() {
    delete client_state;
}


void ccoip::CCoIPClientHandler::registerPeer(const ccoip_inet_address_t &address, const ccoip_uuid_t uuid) const {
    internal_inet_address_t internal_address{
        .protocol = address.protocol
    };
    if (address.protocol == inetIPv4) {
        internal_address.address.ipv4 = address.address.ipv4;
    } else if (address.protocol == inetIPv6) {
        internal_address.address.ipv6 = address.address.ipv6;
    }
    auto &uuids_per_ip = client_state->inet_addrs_to_uuids[internal_address];
    uuids_per_ip.push_back(uuid);
    if (uuids_per_ip.size() > 1) {
        LOG(WARN) << "Registered more than one peer per IP address " << CCOIP_INET_ADDR_TO_STRING(address);
    }
}
