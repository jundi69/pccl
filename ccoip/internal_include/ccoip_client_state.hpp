#pragma once
#include <ccoip_inet.h>
#include <ccoip_inet_utils.hpp>
#include <ccoip_types.hpp>
#include <unordered_set>

namespace ccoip {
    class CCoIPClientState {
        std::unordered_map<internal_inet_socket_address_t, std::unordered_set<ccoip_uuid_t> > inet_addrs_to_uuids{};
    public:
        [[nodiscard]] bool registerPeer(const ccoip_socket_address_t &address, ccoip_uuid_t uuid);
    };
};
