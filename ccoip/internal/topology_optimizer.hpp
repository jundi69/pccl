#pragma once
#include <bandwidth_store.hpp>

namespace ccoip {
    class TopologyOptimizer {
    public:
        [[nodiscard]] static bool OptimizeTopology(const BandwidthStore &bandwidth_store,
                                                   std::vector<ccoip_uuid_t> &ring_topology);
    };
};
