#pragma once

#include <ccoip_inet.h>
#include <vector>
#include <atomic>
#include <mutex>
#include <thread>

#include "ccoip_types.hpp"


#define DEFAULT_NUM_BENCHMARK_CONNECTIONS 16

inline int GetNumBenchmarkConnections() {
    const char *logLevel = getenv("PCCL_NUM_BENCHMARK_CONNECTIONS");
    if (logLevel == nullptr) {
        return DEFAULT_NUM_BENCHMARK_CONNECTIONS;
    }
    return std::stoi(logLevel);
}

namespace ccoip {
    class NetworkBenchmarkRunner {
        ccoip_socket_address_t benchmark_endpoint;
        ccoip_uuid_t self_peer_uuid;
        uint32_t num_benchmark_connections = GetNumBenchmarkConnections();

        /// Stores the output bandwidth in Mbit/s for each connection
        /// Size will be equal to num_benchmark_connections
        std::vector<double> output_bandwidth_mbps{};
        std::mutex output_bandwidth_mbps_mutex{};

        /// List of all threads that are currently running benchmarks
        std::vector<std::thread> running_benchmark_threads{};

        /// State whether any benchmark has encountered a send-failure
        std::atomic_bool has_send_failure{false};

        /// State whether all benchmarks have been completed
        std::atomic_bool all_benchmarks_completed{false};

    public:
        explicit NetworkBenchmarkRunner(ccoip_uuid_t self_peer_uuid, const ccoip_socket_address_t &benchmark_endpoint);

        enum class BenchmarkResult {
            SUCCESS = 0,
            BENCHMARK_SERVER_BUSY = 1,
            CONNECTION_FAILURE = 2,
            SEND_FAILURE = 3,
            OTHER_FAILURE = 3
        };

        [[nodiscard]] BenchmarkResult runBlocking();

        /// Returns the output bandwidth in mbit/s
        [[nodiscard]] double getOutputBandwidthMbitsPerSecond();

    private:
        [[nodiscard]] BenchmarkResult launchBenchmark(int connection_number);
    };

    /// Data-consuming benchmark handler; Created for every connection created by the opposite side
    class NetworkBenchmarkHandler {
    public:
        [[nodiscard]] bool runBlocking(int socket_fd);
    };
} // namespace ccoip
