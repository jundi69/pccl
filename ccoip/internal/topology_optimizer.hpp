#pragma once
#include <bandwidth_store.hpp>

namespace ccoip {
    class TopologyOptimizer {
    public:
        [[nodiscard]] static bool OptimizeTopology(const BandwidthStore &bandwidth_store,
                                                   std::vector<ccoip_uuid_t> &ring_topology,
                                                   bool &topology_is_optimal);


        [[nodiscard]] static bool ImproveTopologyMoonshot(const BandwidthStore &bandwidth_store,
                                                   std::vector<ccoip_uuid_t> &ring_topology,
                                                   bool &topology_is_optimal,
                                                   bool &topology_has_improved);
    };
};
