#pragma once

#include <ccoip_inet_utils.hpp>
#include <ccoip_types.hpp>
#include <unordered_map>


enum ConnectionState {
    /// The client is registered with the master;
    /// Initially, the client is in this state.
    /// It does not yet participate in collective communications operations
    /// or shared state distribution.
    /// In this tate, the client is not allowed to request to participate in any of the above.
    PEER_REGISTERED,

    /// Peers have accepted the client.
    /// Peers periodically accept new clients to join the running session.
    /// This is a phase unanimously agreed upon by all peers.
    /// In this phase, clients will establish p2p connections with new peers.
    PEER_ACCEPTED
};

struct ClientInfo {
    ConnectionState connection_state = PEER_REGISTERED;
};

class CCoIPMasterState {

    /// Maps the socket address of the client to its assigned UUID
    /// Populated on successful session join, and cleared on client leave/disconnect
    std::unordered_map<internal_inet_socket_address_t, ccoip_uuid_t> client_uuids{};

    /// Maps the UUID of the client to its assigned socket address
    /// Populated identically to `client_uuids` for reverse lookups
    std::unordered_map<ccoip_uuid_t, internal_inet_socket_address_t> uuid_clients{};

    /// Maps the uuid of the client to its client information
    std::unordered_map<ccoip_uuid_t, ClientInfo> client_info{};

public:
    /// Registers a client
    void registerClient(const ccoip_socket_address_t &client_address,
                        ccoip_uuid_t uuid);

    /// Unregisters a client
    void unregisterClient(const ccoip_socket_address_t &client_address);

    /// Checks if a client is registered
    [[nodiscard]] bool isClientRegistered(const ccoip_socket_address_t & client_address) const;
};
