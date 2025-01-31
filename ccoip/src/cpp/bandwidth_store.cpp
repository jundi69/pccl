#include "bandwidth_store.hpp"

bool ccoip::BandwidthStore::registerPeer(const ccoip_uuid_t peer) {
    auto [_, inserted] = registered_peers.insert(peer);
    return inserted;
}

bool ccoip::BandwidthStore::storeBandwidth(const ccoip_uuid_t from, const ccoip_uuid_t to,
                                           const double send_bandwidth_mpbs) {
    if (!registered_peers.contains(from)) {
        return false;
    }
    if (!registered_peers.contains(to)) {
        return false;
    }
    bandwidth_map[from][to] = send_bandwidth_mpbs;
    return true;
}

std::optional<double> ccoip::BandwidthStore::getBandwidthMbps(const ccoip_uuid_t from, const ccoip_uuid_t to) const {
    const auto from_it = bandwidth_map.find(from);
    if (from_it == bandwidth_map.end()) {
        return std::nullopt;
    }
    const auto to_it = from_it->second.find(to);
    if (to_it == from_it->second.end()) {
        return std::nullopt;
    }
    return to_it->second;
}

std::vector<ccoip::bandwidth_entry> ccoip::BandwidthStore::getMissingBandwidthEntries(const ccoip_uuid_t peer) const {
    std::vector<bandwidth_entry> missing_entries;

    // add all entries A -> B where A is any peer and B is the specified peer
    for (const auto &distinct_peer: registered_peers) {
        if (distinct_peer == peer) {
            continue;
        }
        if (!bandwidth_map.contains(distinct_peer) || !bandwidth_map.at(distinct_peer).contains(peer)) {
            missing_entries.push_back({distinct_peer, peer});
        }
    }

    // add all entries B -> A where A is any peer and B is the specified peer
    for (const auto &distinct_peer: registered_peers) {
        if (distinct_peer == peer) {
            continue;
        }
        if (!bandwidth_map.contains(peer) || !bandwidth_map.at(peer).contains(distinct_peer)) {
            missing_entries.push_back({peer, distinct_peer});
        }
    }
    return missing_entries;
}

bool ccoip::BandwidthStore::isBandwidthStoreFullyPopulated() const {
    if (registered_peers.size() == 1) {
        return true;
    }

    bool fully_populated = true;
    for (const auto &peer: registered_peers) {
        if (!bandwidth_map.contains(peer)) {
            return false;
        }
        fully_populated &= bandwidth_map.at(peer).size() == registered_peers.size() - 1;
    }
    fully_populated &= bandwidth_map.size() == registered_peers.size();
    return fully_populated;
}

size_t ccoip::BandwidthStore::getNumberOfRegisteredPeers() const {
    return registered_peers.size();
}


bool ccoip::BandwidthStore::unregisterPeer(const ccoip_uuid_t peer) {
    const auto erased = registered_peers.erase(peer);
    if (!erased) {
        return false;
    }
    bandwidth_map.erase(peer);
    for (auto &[_, map]: bandwidth_map) {
        map.erase(peer);
    }
    return true;
}
