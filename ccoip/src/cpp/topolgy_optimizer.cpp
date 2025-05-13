#include "topology_optimizer.hpp"

#include <libtsp.h>
#include <pccl_log.hpp>

bool ccoip::TopologyOptimizer::OptimizeTopology(const BandwidthStore &bandwidth_store,
                                                std::vector<ccoip_uuid_t> &ring_topology,
                                                bool &topology_is_optimal) {
    if (ring_topology.size() == 1) {
        return true;
    }

    std::vector<TspInputGraphEdge> edges{};
    const size_t n_peers = bandwidth_store.getNumberOfRegisteredPeers();
    edges.reserve(n_peers * n_peers - n_peers);

    std::unordered_map<ccoip_uuid_t, nodeid_t> peer_to_id{};
    std::unordered_map<nodeid_t, ccoip_uuid_t> id_to_peer{};

    // create input graph
    for (const auto &from: ring_topology) {
        for (const auto &to: ring_topology) {
            if (from == to) {
                continue;
            }
            const auto bandwidth = bandwidth_store.getBandwidthMbps(from, to);
            if (!bandwidth.has_value()) {
                continue; // we don't assert a fully populated bandwidth store, some peers may not have p2p connectivity
            }

            const nodeid_t from_id = peer_to_id.try_emplace(from, peer_to_id.size()).first->second;
            const nodeid_t to_id = peer_to_id.try_emplace(to, peer_to_id.size()).first->second;

            edges.push_back(TspInputGraphEdge{
                    .cost = static_cast<cost_t>(1000.0 / bandwidth.value()),
                    .from = from_id,
                    .to = to_id
            });
        }
    }
    // create bidirectional id mapping
    for (const auto &[peer, id]: peer_to_id) {
        id_to_peer[id] = peer;
    }

    const TspInputGraphDescriptor input_graph{
            .edges = edges.data(),
            .num_edges = edges.size()
    };
    constexpr TspSolverOptionsDescriptor solver_options{
            .attempt_exact = true,
            .exact_upper_bound = 8,
            .seed = 42,
            .num_iterations = 100,
            .tabu_tenure = 5,
            .num_restarts = 4,
            .time_limit_ms = 1000,
            .initial_heuristic = TSP_INIT_RANDOM_STRATEGY,
            .ant_colony_num_samples = 2048,
            .enable_3opt = true,
            .enable_4opt = false
    };

    TspSolutionDescriptor output_descriptor{};
    const TSPStatus status = tspAsymmetricSolve(&input_graph, &solver_options, &output_descriptor);
    if (status != TSP_STATUS_SUCCESS) {
        LOG(BUG) << "Failed to run ATSP solver for topology optimization! " << status << " This should never happen.";
        return false;
    }

    LOG(INFO) << "Optimized topology with cost: " << output_descriptor.solution_cost << "; Solution is " << (
        output_descriptor.solution_type == TSP_SOLUTION_TYPE_OPTIMAL
            ? "OPTIMAL"
            : output_descriptor.solution_type == TSP_SOLUTION_TYPE_APPROXIMATE
            ? "APPROXIMATE"
            : "TSP_SOLUTION_TYPE_???");

    // populate the new ring topology
    ring_topology.clear();
    for (size_t i = 0; i < output_descriptor.num_nodes; ++i) {
        ring_topology.push_back(id_to_peer[output_descriptor.tour[i]]);
    }
    topology_is_optimal = output_descriptor.solution_type == TSP_SOLUTION_TYPE_OPTIMAL;
    tspDisposeSolution(&output_descriptor);
    return true;
}

bool ccoip::TopologyOptimizer::ImproveTopologyMoonshot(const BandwidthStore &bandwidth_store,
                                               std::vector<ccoip_uuid_t> &ring_topology,
                                               bool &topology_is_optimal,
                                               bool &topology_has_improved) {
    if (ring_topology.size() == 1) {
        return true;
    }

    std::vector<TspInputGraphEdge> edges{};
    const size_t n_peers = bandwidth_store.getNumberOfRegisteredPeers();
    edges.reserve(n_peers * n_peers - n_peers);

    std::unordered_map<ccoip_uuid_t, nodeid_t> peer_to_id{};
    std::unordered_map<nodeid_t, ccoip_uuid_t> id_to_peer{};

    // create input graph
    for (const auto &from: ring_topology) {
        for (const auto &to: ring_topology) {
            if (from == to) {
                continue;
            }
            const auto bandwidth = bandwidth_store.getBandwidthMbps(from, to);
            if (!bandwidth.has_value()) {
                 continue; // don't assert a fully populated bandwidth store
            }

            const nodeid_t from_id = peer_to_id.try_emplace(from, peer_to_id.size()).first->second;
            const nodeid_t to_id = peer_to_id.try_emplace(to, peer_to_id.size()).first->second;

            edges.push_back(TspInputGraphEdge{
                    .cost = static_cast<cost_t>(1000.0 / bandwidth.value()),
                    .from = from_id,
                    .to = to_id
            });
        }
    }
    // create bidirectional id mapping
    for (const auto &[peer, id]: peer_to_id) {
        id_to_peer[id] = peer;
    }

    const TspInputGraphDescriptor input_graph{
            .edges = edges.data(),
            .num_edges = edges.size()
    };
    constexpr TspSolverOptionsDescriptor solver_options{
            .attempt_exact = true,
            .exact_upper_bound = 20,
            .seed = 42,
            .num_iterations = 128,
            .tabu_tenure = 4,
            .num_restarts = 16,
            .time_limit_ms = 30000,
            .initial_heuristic = TSP_INIT_RANDOM_STRATEGY,
            .ant_colony_num_samples = 2048,
            .enable_3opt = true,
            .enable_4opt = false
    };

    TspSolutionDescriptor initial_solution{};
    initial_solution.num_nodes = ring_topology.size();
    initial_solution.tour = new nodeid_t[ring_topology.size()];
    for (size_t i = 0; i < ring_topology.size(); ++i) {
        initial_solution.tour[i] = peer_to_id[ring_topology[i]];
    }
    initial_solution.solution_cost = 0; // will be recomputed anyway
    initial_solution.solution_type = TSP_SOLUTION_TYPE_APPROXIMATE;

    TspSolutionDescriptor output_descriptor{};
    if (tspAsymmetricImproveSolution(&input_graph, &initial_solution, &solver_options, &output_descriptor) !=
        TSP_STATUS_SUCCESS) {
        LOG(BUG) << "Failed to run ATSP solver to improve topology! This should never happen.";
        return false;
    }

    LOG(INFO) << "Improve topology yielded solution cost: " << output_descriptor.solution_cost << "; Solution is " << (
        output_descriptor.solution_type == TSP_SOLUTION_TYPE_OPTIMAL
            ? "OPTIMAL"
            : output_descriptor.solution_type == TSP_SOLUTION_TYPE_IMPROVED
            ? "IMPROVED"
            : "NO_IMPROVEMENT");

    // populate the new ring topology
    ring_topology.clear();
    for (size_t i = 0; i < output_descriptor.num_nodes; ++i) {
        ring_topology.push_back(id_to_peer[output_descriptor.tour[i]]);
    }
    topology_has_improved = output_descriptor.solution_type == TSP_SOLUTION_TYPE_IMPROVED ||
                            output_descriptor.solution_type == TSP_SOLUTION_TYPE_OPTIMAL;
    topology_is_optimal = output_descriptor.solution_type == TSP_SOLUTION_TYPE_OPTIMAL;
    tspDisposeSolution(&output_descriptor);
    delete[] initial_solution.tour;
    return true;
}
