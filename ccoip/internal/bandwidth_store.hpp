#pragma once

#include <ccoip_types.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <optional>

namespace ccoip {

    struct bandwidth_entry {
        ccoip_uuid_t from_peer_uuid;
        ccoip_uuid_t to_peer_uuid;
    };

    class BandwidthStore {
        /**
         * Stores the bandwidth matrix.
         * Logically, we are storing entries (A, B, bandwidth) where A and B are peers and bandwidth is the bandwidth from A to B.
         * (A -> B) is stored in the entry (A, B) and (B -> A) is stored in the entry (B, A), meaning that we are storing an asymmetric edge-weighted graph.
         *
         * Physically, we are storing a map<uuid, map<uuid, double>> where the outer map maps the send peer to an inner
         * map that maps the receiving peer to the bandwidth.
         *
         * E.g. to obtain the bandwidth value for the edge (A, B), we would access bandwidth_map[A][B] and to obtain the bandwidth value for the edge (B, A), we would access bandwidth_map[B][A].
         *
         * The double value has units of mpbs or mbit/s, meaning that to get MB/s we would need to divide by 8.
         *
         * We make the following simplified assertion for sanity purposes that send bandwidth is unaffected by receive workload.
         * This is a simplification that is not always true in practice, but is reasonable for datacenter internet traffic.
         */
        std::unordered_map<ccoip_uuid_t, std::unordered_map<ccoip_uuid_t, double>> bandwidth_map;

        /**
         * Set of all distinct peers that have any entry in the bandwidth map.
         */
        std::unordered_set<ccoip_uuid_t> registered_peers{};

    public:

        /**
         * Registers the peer in the bandwidth store. Once a peer is registered, it is expected that bandwidth entries will be provided for it.
         * @code getMissingBandwidthEntries @endcode will consider all registered peers when determining missing entries.
         * @param peer the peer to register
         * @return false if the peer was already registered, true otherwise
         */
        bool registerPeer(ccoip_uuid_t peer);

        /**
         * Stores the bandwidth between two peers.
         * Such an entry is strictly unidirectional, meaning that for both directions A -> B and B -> A, two separate entries must be stored that may differ in bandwidth.
         * @param from uuid of the "from" peer
         * @param to uuid of the "to" peer
         * @param send_bandwidth_mpbs the bandwidth in mbit/s that the "from" peer can send to the "to" peer. This is different from what the "to" peer can receive from the "from" peer, which is stored in a separate entry.
         */
        [[nodiscard]] bool storeBandwidth(ccoip_uuid_t from, ccoip_uuid_t to, double send_bandwidth_mpbs);

        /**
         * @param from from peer
         * @param to to peer
         * @return the bandwidth in mbit/s that the "from" peer can send to the "to" peer. If the bandwidth is not stored, returns std::nullopt.
         */
        [[nodiscard]] std::optional<double> getBandwidthMbps(ccoip_uuid_t from, ccoip_uuid_t to) const;

        /**
         * Determines the list of missing bandwidth entries that a particular peer is part of.
         * @param peer the uuid of the peer
         * @return the list of bandwidth entries that are missing for the specified peer. This means all edges to and from the peer to others that are not populated with bandwidth data.
         */
        [[nodiscard]] std::vector<bandwidth_entry> getMissingBandwidthEntries(ccoip_uuid_t peer) const;

        /**
         * @return true if the bandwidth store is fully populated with all possible entries for all registered peers. false otherwise.
         */
        [[nodiscard]] bool isBandwidthStoreFullyPopulated() const;

        /**
         * @return the number of registered peers in the bandwidth store.
         */
        [[nodiscard]] size_t getNumberOfRegisteredPeers() const;

        /**
         * Unregisters the given peer and deletes all bandwidth data associated with a peer.
         * When said peer is part of an edge in the bandwidth graph, the edge is removed.
         * The peer will no longer be considered when determining missing bandwidth entries after this call.
         * @param peer the uuid of the peer
         * @return false if the peer was never registered. true otherwise.
         */
        [[nodiscard]] bool unregisterPeer(ccoip_uuid_t peer);
    };
}
