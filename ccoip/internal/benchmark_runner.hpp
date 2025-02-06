#pragma once

#include <ccoip_inet.h>

namespace ccoip {
    class NetworkBenchmarkRunner {
        ccoip_socket_address_t benchmark_endpoint;
        double output_bandwidth_mbps = 0;

    public:
        explicit NetworkBenchmarkRunner(const ccoip_socket_address_t &benchmark_endpoint);

        enum class BenchmarkResult {
            SUCCESS = 0,
            BENCHMARK_SERVER_BUSY = 1,
            CONNECTION_FAILURE = 2,
            SEND_FAILURE = 3,
            OTHER_FAILURE = 3
        };

        [[nodiscard]] BenchmarkResult runBlocking();

        /// Returns the output bandwidth in mbit/s
        [[nodiscard]] double getOutputBandwidthMbitsPerSecond() const;


    };

    class NetworkBenchmarkHandler {
    public:
        [[nodiscard]] bool runBlocking(int socket_fd);
    };
}
