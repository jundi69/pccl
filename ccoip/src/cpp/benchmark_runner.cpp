#include "benchmark_runner.hpp"

#include <ccoip_inet_utils.hpp>
#include <ccoip_packets.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <tinysockets.hpp>
#include "win_sock_bridge.h"

#define BENCHMARK_LENGTH_SECONDS 10

ccoip::NetworkBenchmarkRunner::NetworkBenchmarkRunner(const ccoip_socket_address_t &benchmark_endpoint) :
    benchmark_endpoint(benchmark_endpoint) {
}

ccoip::NetworkBenchmarkRunner::BenchmarkResult ccoip::NetworkBenchmarkRunner::runBlocking() {
    tinysockets::BlockingIOSocket socket(benchmark_endpoint);
    LOG(DEBUG) << "NetworkBenchmarkRunner connecting to endpoint " << ccoip_sockaddr_to_str(benchmark_endpoint) << "...";
    if (!socket.establishConnection()) {
        LOG(WARN) << "Failed to establish connection to benchmark endpoint "
                  << ccoip_sockaddr_to_str(benchmark_endpoint) << ".";
        return BenchmarkResult::CONNECTION_FAILURE;
    }

    const auto packet_opt = socket.receivePacket<B2CPacketBenchmarkServerIsBusy>();
    if (!packet_opt) {
        LOG(ERR) << "Failed to receive B2CPacketBenchmarkServerIsBusy from "
                 << ccoip_sockaddr_to_str(benchmark_endpoint);
        return BenchmarkResult::CONNECTION_FAILURE;
    }
    if (packet_opt->is_busy) {
        if (!socket.closeConnection()) {
            LOG(WARN) << "Failed to close connection to " << ccoip_sockaddr_to_str(benchmark_endpoint)
                      << " after benchmark server indicated it was busy. Ignoring...";
        }
        return BenchmarkResult::BENCHMARK_SERVER_BUSY;
    }

    // run benchmark for 10 seconds
    size_t total_bytes_sent = 0;

    const int socket_fd = socket.getSocketFd();

    // get socket send buffer size
    size_t send_buffer_size = 0;
    socklen_t optlen = sizeof(send_buffer_size);
    if (getsockoptvp(socket_fd, SOL_SOCKET, SO_SNDBUF, &send_buffer_size, &optlen) == -1) {
        LOG(ERR) << "Failed to get send buffer size of socket";
        return BenchmarkResult::OTHER_FAILURE;
    }

    LOG(DEBUG) << "NetworkBenchmarkRunner obtained socket send buffer size: " << send_buffer_size;

    const std::unique_ptr<uint8_t[]> buffer(new uint8_t[send_buffer_size]);

    // fill random
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint16_t> dis(0, 255); // uint16_t to make MSVC happy
        for (size_t i = 0; i < send_buffer_size; ++i) {
            buffer[i] = static_cast<uint8_t>(dis(gen));
        }
    }

    const auto start_time = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start_time < std::chrono::seconds(BENCHMARK_LENGTH_SECONDS)) {
        // don't check time every iteration
        // send data
        const auto n_sent = sendvp(socket_fd, buffer.get(), send_buffer_size, MSG_NOSIGNAL);
        if (n_sent < 0) {
            if (errno == 0) {
                break;
            }
            LOG(ERR) << "Failed to send data to benchmark server: " << std::strerror(errno);
            return BenchmarkResult::SEND_FAILURE;
        }
        total_bytes_sent += n_sent;
    }
    LOG(DEBUG) << "Finished sending data for network bandwidth benchmark";

    const auto now = std::chrono::high_resolution_clock::now();
    if (!socket.closeConnection()) {
        LOG(WARN) << "Failed to close connection to " << ccoip_sockaddr_to_str(benchmark_endpoint)
                  << " after benchmark. Ignoring...";
    }
    const auto duration = now - start_time;
    const auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
    const auto duration_seconds = static_cast<double>(duration_us) / 1e6;
    const auto bandwidth_mbits_per_second = static_cast<double>(total_bytes_sent * 8) / 1e6 / duration_seconds;
    LOG(DEBUG) << "Measured bandwidth: " << bandwidth_mbits_per_second << " Mbit/s to peer "
               << ccoip_sockaddr_to_str(benchmark_endpoint);
    output_bandwidth_mbps = bandwidth_mbits_per_second;
    return BenchmarkResult::SUCCESS;
}

double ccoip::NetworkBenchmarkRunner::getOutputBandwidthMbitsPerSecond() const { return output_bandwidth_mbps; }

bool ccoip::NetworkBenchmarkHandler::runBlocking(const int socket_fd) {
    size_t total_bytes_received = 0;

    // get socket receive buffer size
    size_t buffer_size = 0;
    socklen_t optlen = sizeof(buffer_size);
    if (getsockoptvp(socket_fd, SOL_SOCKET, SO_RCVBUF, &buffer_size, &optlen) == -1) {
        LOG(ERR) << "Failed to get receive buffer size of socket";
        return false;
    }

    LOG(DEBUG) << "NetworkBenchmarkHandler obtained socket receive buffer size: " << buffer_size;

    const std::unique_ptr<uint8_t[]> buffer(new uint8_t[buffer_size]);
    auto start = std::chrono::high_resolution_clock::now();
    while (true) {
        if (auto now = std::chrono::high_resolution_clock::now();
            now - start > std::chrono::seconds(BENCHMARK_LENGTH_SECONDS + 1)) {
            break;
        }

        std::vector descriptors = {
                tinysockets::poll::PollDescriptor{socket_fd, tinysockets::poll::PollEvent::POLL_INPUT},
        };
        auto &rx_descriptor = descriptors[0];
        poll(descriptors, 0);

        if (!rx_descriptor.hasEvent(tinysockets::poll::PollEvent::POLL_INPUT)) {
            continue;
        }

        std::span buffer_span(reinterpret_cast<std::byte *>(buffer.get()), buffer_size);
        const auto n_received = recv_nonblocking(buffer_span, rx_descriptor);
        if (!n_received) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            // check if closed
            if (errno == ECONNRESET || errno == ECONNABORTED || errno == ENOTCONN) {
                break;
            }

            LOG(ERR) << "Failed to receive data from benchmark client.";
            closesocket(socket_fd);
            return false;
        }
        if (*n_received == 0) {
            break;
        }
        total_bytes_received += *n_received;
    }
    shutdown(socket_fd, SHUT_RDWR);
    return true;
}
