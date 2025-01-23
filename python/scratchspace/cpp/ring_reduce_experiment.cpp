#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <cassert>
#include <cstring>
#include <random>
#include <algorithm>
#include <numeric>
#include <poll.h>
#include <fcntl.h>
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cerrno>

// For byte-order helpers (htole32, le32toh).
#ifdef __linux__
#include <endian.h>
#else
// Fallback for systems without <endian.h>
#include <cstdint>
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
static uint32_t htole32(const uint32_t x) { return x; }
static uint32_t le32toh(const uint32_t x) { return x; }
#else
#error "Unsupported byte order"
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Global data/variables
///////////////////////////////////////////////////////////////////////////////
static int g_world_size = 0;
static constexpr int BASE_PORT = 9000;
static std::vector<int> all_sockets;
static std::mutex io_lock;

std::atomic<size_t> bytes_sent{0};
std::atomic<size_t> bytes_received{0};

std::mutex print_lock{};

/**
 * p2p_connections_tx[rank] is a map<peer_rank, socket_fd> for outbound connections,
 * p2p_connections_rx[rank] is a map<peer_rank, socket_fd> for inbound connections.
 */
static std::vector<std::map<int, int>> p2p_connections_tx;
static std::vector<std::map<int, int>> p2p_connections_rx;

///////////////////////////////////////////////////////////////////////////////
// Close and clear all sockets
///////////////////////////////////////////////////////////////////////////////
void cleanup_sockets() {
    for (const int s: all_sockets) {
        if (s >= 0) {
            ::close(s);
        }
    }
    all_sockets.clear();
    p2p_connections_tx.clear();
    p2p_connections_rx.clear();
}

///////////////////////////////////////////////////////////////////////////////
// Read exactly n bytes
///////////////////////////////////////////////////////////////////////////////
bool read_exact(const int sock, char *buffer, const size_t n) {
    size_t total_read = 0;
    while (total_read < n) {
        const ssize_t rd = ::recv(sock, buffer + total_read, n - total_read, 0);
        if (rd <= 0) {
            // Error or connection closed
            return false;
        }
        total_read += static_cast<size_t>(rd);
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Initialize the "mailboxes" (listening sockets + connections).
///////////////////////////////////////////////////////////////////////////////
bool init_mailboxes(const int new_world_size) {
    cleanup_sockets(); {
        std::lock_guard guard(io_lock);
        bytes_sent = 0;
        bytes_received = 0;
    }

    g_world_size = new_world_size;

    p2p_connections_tx.resize(g_world_size);
    p2p_connections_rx.resize(g_world_size);

    // 1) Create server sockets for each rank
    std::vector server_sockets(g_world_size, -1);
    for (int rank = 0; rank < g_world_size; ++rank) {
        int sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sockfd < 0) {
            std::cerr << "Error creating socket for rank " << rank << ": " << strerror(errno) << "\n";
            return false;
        }

        int optval = 1;
        ::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

        sockaddr_in addr{};
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(BASE_PORT + rank);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // "127.0.0.1"

        if (::bind(sockfd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
            std::cerr << "Bind error (rank " << rank << "): " << strerror(errno) << "\n";
            ::close(sockfd);
            return false;
        }

        if (::listen(sockfd, g_world_size - 1) < 0) {
            std::cerr << "Listen error (rank " << rank << "): " << strerror(errno) << "\n";
            ::close(sockfd);
            return false;
        }

        server_sockets[rank] = sockfd;
        all_sockets.push_back(sockfd);
    }

    // 2) Each rank -> connect to next rank in the ring
    for (int rank = 0; rank < g_world_size; ++rank) {
        int peer = (rank + 1) % g_world_size;

        int client_sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (client_sock < 0) {
            std::cerr << "Error creating client socket: " << strerror(errno) << "\n";
            return false;
        }

        int optval = 1;
        ::setsockopt(client_sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

        sockaddr_in peer_addr{};
        std::memset(&peer_addr, 0, sizeof(peer_addr));
        peer_addr.sin_family = AF_INET;
        peer_addr.sin_port = htons(BASE_PORT + peer);
        peer_addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        // Try connect
        if (::connect(client_sock, reinterpret_cast<sockaddr *>(&peer_addr), sizeof(peer_addr)) < 0) {
            // If it fails with EINPROGRESS (non-blocking connect), we can poll, but for simplicity:
            if (errno != EINPROGRESS) {
                std::cerr << "[Rank " << rank << "] connect to rank "
                        << peer << " failed: " << strerror(errno) << "\n";
                ::close(client_sock);
                return false;
            }
        }

        // Send "hello" = 4 bytes, little-endian rank
        int32_t r_le = htole32(rank);
        if (::send(client_sock, &r_le, 4, 0) != 4) {
            std::cerr << "Error sending hello from rank " << rank << " to " << peer << "\n";
            ::close(client_sock);
            return false;
        }

        p2p_connections_tx[rank][peer] = client_sock;
        all_sockets.push_back(client_sock);
    }

    // 3) Accept inbound connections (from rank-1 in ring)
    for (int rank = 0; rank < g_world_size; ++rank) {
        // We only expect 1 inbound in a ring (from rank-1)
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int conn_fd = ::accept(server_sockets[rank],
                               reinterpret_cast<sockaddr *>(&client_addr),
                               &client_len);
        if (conn_fd < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // This can happen in nonblocking mode if not connected yet
                // We could poll for accept readiness.
                // For simplicity, do a blocking accept in a loop.
                // But let's do a quick fallback:
                fd_set fds;
                FD_ZERO(&fds);
                FD_SET(server_sockets[rank], &fds);
                if (::select(server_sockets[rank] + 1, &fds, nullptr, nullptr, nullptr) < 0) {
                    std::cerr << "select() error: " << strerror(errno) << "\n";
                    return false;
                }
                conn_fd = ::accept(server_sockets[rank],
                                   reinterpret_cast<sockaddr *>(&client_addr),
                                   &client_len);
                if (conn_fd < 0) {
                    std::cerr << "accept() after select still failed, rank=" << rank
                            << ", err=" << strerror(errno) << std::endl;
                    return false;
                }
            } else {
                std::cerr << "accept() error (rank=" << rank << "): "
                        << strerror(errno) << std::endl;
                return false;
            }
        }
        all_sockets.push_back(conn_fd);

        // read "hello"
        char buf[4];
        if (!read_exact(conn_fd, buf, 4)) {
            std::cerr << "Error reading remote rank in accept(), rank=" << rank << "\n";
            ::close(conn_fd);
            return false;
        }
        int32_t remote_le;
        std::memcpy(&remote_le, buf, 4);
        int remote_rank = static_cast<int>(le32toh(remote_le));

        p2p_connections_rx[rank][remote_rank] = conn_fd;
    }

    // 4) Close the server sockets
    for (int rank = 0; rank < g_world_size; ++rank) {
        ::close(server_sockets[rank]);
        if (auto it = std::find(all_sockets.begin(), all_sockets.end(), server_sockets[rank]); it != all_sockets.end()) {
            all_sockets.erase(it);
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Single-thread, full-duplex "send_and_recv" with poll
///////////////////////////////////////////////////////////////////////////////
bool send_and_recv(const int send_fd,
                   const char *send_buf, const size_t send_len,
                   const int recv_fd,
                   char *recv_buf, const size_t recv_len) {
    size_t bytes_to_send = send_len;
    size_t bytes_to_recv = recv_len;

    size_t send_offset = 0;
    size_t recv_offset = 0;

    while (bytes_to_send > 0 || bytes_to_recv > 0) {
        std::vector<pollfd> fds{};
        fds.reserve(2);
        const pollfd *rx_fd = nullptr;
        const pollfd *tx_fd = nullptr;
        if (bytes_to_send > 0) {
            fds.push_back({send_fd, POLLOUT, 0});
            tx_fd = &fds.back();
        }
        if (bytes_to_recv > 0) {
            fds.push_back({recv_fd, POLLIN, 0});
            rx_fd = &fds.back();
        }

        if (const int ret = ::poll(fds.data(), fds.size(), -1); ret < 0) {
            std::cerr << "poll() failed: " << strerror(errno) << std::endl;
            return false;
        }

        // Ready to send?
        if (tx_fd != nullptr && (tx_fd->revents & POLLOUT) && (bytes_to_send > 0)) {
            if (const ssize_t n = ::send(send_fd, send_buf + send_offset, bytes_to_send, 0); n < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                    std::cerr << "send() error: " << strerror(errno) << std::endl;
                    return false;
                }
                // else just retry
            } else if (n == 0) {
                // peer closed?
                std::cerr << "send() returned 0. Peer closed?" << std::endl;
                return false;
            } else {
                bytes_to_send -= static_cast<size_t>(n);
                send_offset += static_cast<size_t>(n);
            }
        }

        // Ready to receive?
        if (rx_fd != nullptr && (rx_fd->revents & POLLIN) && (bytes_to_recv > 0)) {
            if (const ssize_t n = ::recv(recv_fd, recv_buf + recv_offset, bytes_to_recv, 0); n < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
                    std::cerr << "recv() error: " << strerror(errno) << std::endl;
                    return false;
                }
                // else just retry
            } else if (n == 0) {
                std::cerr << "recv() returned 0. Peer closed?" << std::endl;
                return false;
            } else {
                bytes_to_recv -= static_cast<size_t>(n);
                recv_offset += static_cast<size_t>(n);
            }
        }
    }

    // If we made it here, we fully sent & received
    {
        std::lock_guard guard(io_lock);
        bytes_sent += send_len;
        bytes_received += recv_len;
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Partition 1D array into contiguous chunks
///////////////////////////////////////////////////////////////////////////////
std::vector<std::pair<int, int>> compute_chunk_boundaries(const int length, const int world_size) {
    std::vector<std::pair<int, int>> boundaries(world_size);
    const int base = length / world_size;
    const int remainder = length % world_size;
    int start = 0;
    for (int i = 0; i < world_size; ++i) {
        const int sz = base + (i < remainder ? 1 : 0);
        boundaries[i] = std::make_pair(start, start + sz);
        start += sz;
    }
    return boundaries;
}

///////////////////////////////////////////////////////////////////////////////
// Phase 1: Ring Reduce-Scatter (Full-duplex with poll)
///////////////////////////////////////////////////////////////////////////////
std::map<int, std::vector<double>>
ring_reduce_scatter(const int rank, const std::vector<double> &local_data,
                    const std::vector<std::pair<int, int>> &boundaries) {
    std::map<int, std::vector<double>> chunk_val;
    std::vector has_added(g_world_size, false);

    // Initially, rank owns chunk == rank
    {
        const int s = boundaries[rank].first;
        const int e = boundaries[rank].second;
        if (e > s) {
            chunk_val[rank] = std::vector(local_data.begin() + s, local_data.begin() + e);
            has_added[rank] = true;
        }
    }

    for (int step = 0; step < g_world_size - 1; ++step) {
        int chunk_to_send = (rank - step + g_world_size) % g_world_size;
        int next_rank = (rank + 1) % g_world_size;
        int prev_rank = (rank - 1 + g_world_size) % g_world_size;

        // Data to send
        std::vector<double> arr_to_send;
        if (auto it = chunk_val.find(chunk_to_send); it != chunk_val.end()) {
            arr_to_send = it->second;
            chunk_val.erase(it);
        }
        const size_t out_bytes = arr_to_send.size() * sizeof(double);

        // Data we expect to receive
        int recv_chunk_idx = (prev_rank - step + g_world_size) % g_world_size;
        const int s_j = boundaries[recv_chunk_idx].first;
        const int e_j = boundaries[recv_chunk_idx].second;
        const int chunk_size = e_j - s_j;
        std::vector inc_arr(chunk_size, 0.0);
        const size_t in_bytes = chunk_size * sizeof(double);

        // Single call full-duplex
        const bool ok = send_and_recv(
                /*send_fd*/ p2p_connections_tx[rank][next_rank],
                            reinterpret_cast<const char *>(arr_to_send.data()), out_bytes,
                            /*recv_fd*/ p2p_connections_rx[rank][prev_rank],
                            reinterpret_cast<char *>(inc_arr.data()), in_bytes
                );
        if (!ok) {
            std::cerr << "[Rank " << rank << "] send_and_recv failed in ring_reduce_scatter\n";
            // Return whatever we have (or handle error differently)
            return {};
        }

        // Add local_data if not added yet
        if (!inc_arr.empty() && !has_added[recv_chunk_idx]) {
            for (int i = 0; i < chunk_size; ++i) {
                inc_arr[i] += local_data[s_j + i];
            }
            has_added[recv_chunk_idx] = true;
        }

        // Save chunk
        chunk_val[recv_chunk_idx] = inc_arr;
    }

    return chunk_val;
}

///////////////////////////////////////////////////////////////////////////////
// Phase 2: Ring Allgather (Pipeline, also full-duplex poll)
///////////////////////////////////////////////////////////////////////////////
std::map<int, std::vector<double>>
ring_allgather_pipeline(const int rank,
                        std::map<int, std::vector<double>> chunk_val,
                        const std::vector<std::pair<int, int>> &boundaries) {
    if (chunk_val.size() != 1) {
        std::cerr << "[Rank " << rank << "] Error: ring_reduce_scatter produced "
                << chunk_val.size() << " chunks (expected exactly 1)\n";
    }

    // Identify the single chunk we have
    std::vector<double> current_chunk;
    if (!chunk_val.empty()) {
        current_chunk = chunk_val.begin()->second;
    }

    for (int step = 0; step < g_world_size - 1; ++step) {
        int next_rank = (rank + 1) % g_world_size;
        int prev_rank = (rank - 1 + g_world_size) % g_world_size;

        // outgoing chunk
        std::vector<double> outgoing = current_chunk;
        const size_t out_bytes = outgoing.size() * sizeof(double);

        // Which chunk to receive?
        // inc_idx = (prev_rank + 1 - step) mod g_world_size
        int inc_idx = (prev_rank + 1 - step + g_world_size) % g_world_size;
        const int s_j = boundaries[inc_idx].first;
        const int e_j = boundaries[inc_idx].second;
        const int chunk_size = e_j - s_j;
        std::vector inc_arr(chunk_size, 0.0);
        const size_t in_bytes = chunk_size * sizeof(double);

        const bool ok = send_and_recv(
                /*send_fd*/ p2p_connections_tx[rank][next_rank],
                            reinterpret_cast<const char *>(outgoing.data()), out_bytes,
                            /*recv_fd*/ p2p_connections_rx[rank][prev_rank],
                            reinterpret_cast<char *>(inc_arr.data()), in_bytes
                );
        if (!ok) {
            std::cerr << "[Rank " << rank << "] send_and_recv failed in ring_allgather_pipeline\n";
            return {};
        }

        chunk_val[inc_idx] = inc_arr;
        current_chunk = inc_arr;
    }

    return chunk_val;
}

///////////////////////////////////////////////////////////////////////////////
// Reassemble distributed chunks into a single array
///////////////////////////////////////////////////////////////////////////////
std::vector<double>
reassemble_chunks(const std::map<int, std::vector<double>> &chunk_val,
                  const std::vector<std::pair<int, int>> &boundaries,
                  const int total_length) {
    std::vector out(total_length, 0.0);
    for (const auto &[fst, snd]: chunk_val) {
        const int idx = fst;
        auto &arr = snd;
        const int s = boundaries[idx].first;
        const int e = boundaries[idx].second;
        if (!arr.empty()) {
            std::memcpy(&out[s], arr.data(), (e - s) * sizeof(double));
        }
    }
    return out;
}

///////////////////////////////////////////////////////////////////////////////
// ring_allreduce
///////////////////////////////////////////////////////////////////////////////
std::vector<double>
ring_allreduce(const int rank, const std::vector<double> &local_data) {
    const int length = (int) local_data.size();
    const auto boundaries = compute_chunk_boundaries(length, g_world_size);

    // Phase 1
    const auto partials = ring_reduce_scatter(rank, local_data, boundaries);
    // Phase 2
    const auto final_map = ring_allgather_pipeline(rank, partials, boundaries);
    // reassemble
    return reassemble_chunks(final_map, boundaries, length);
}

///////////////////////////////////////////////////////////////////////////////
// Worker thread function
///////////////////////////////////////////////////////////////////////////////
void worker(const int rank, const std::vector<double> &local_data,
            std::vector<std::vector<double>> &results) {
    const auto arr = ring_allreduce(rank, local_data);
    results[rank] = arr; {
        std::lock_guard lk(print_lock);
        std::cout << "[Rank " << rank << "] final out: ";
        for (const auto v: arr) {
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Optional example main
///////////////////////////////////////////////////////////////////////////////
#ifdef RING_REDUCE_EXPERIMENT_MAIN_IMPL
int main() {
    constexpr int test_world_size = 3;

    // 1) Initialize the mailboxes
    if (!init_mailboxes(test_world_size)) {
        std::cerr << "init_mailboxes failed.\n";
        return 1;
    }

    // 2) Generate example data for each rank
    constexpr int length = 3;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<std::vector<double>> all_data(test_world_size, std::vector<double>(length));
    for (int r = 0; r < test_world_size; ++r) {
        for (int i = 0; i < length; ++i) {
            all_data[r][i] = dist(rng);
        }
    }

    // Compute the expected sum
    std::vector<double> expected(length, 0.0);
    for (int r = 0; r < test_world_size; ++r) {
        for (int i = 0; i < length; ++i) {
            expected[i] += all_data[r][i];
        }
    }
    std::cout << "Expected sum: ";
    for (auto v : expected) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // 3) Launch each rank in a separate thread
    std::vector<std::vector<double>> results(test_world_size);
    std::vector<std::thread> threads;
    threads.reserve(test_world_size);

    for (int r = 0; r < test_world_size; ++r) {
        threads.emplace_back([r, &all_data, &results]() {
            worker(r, all_data[r], results);
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    // 4) Check correctness
    bool all_ok = true;
    for (int r = 0; r < test_world_size; ++r) {
        if (results[r].size() != (size_t)length) {
            std::cerr << "[Rank " << r << "] result size mismatch!\n";
            all_ok = false;
            continue;
        }
        for (int i = 0; i < length; ++i) {
            double diff = results[r][i] - expected[i];
            if (std::fabs(diff) > 1e-10) {
                std::cerr << "[Rank " << r << "] mismatch at index " << i
                          << " (got " << results[r][i] << ", expected " << expected[i]
                          << ", diff=" << diff << ")\n";
                all_ok = false;
            }
        }
    }
    if (all_ok) {
        std::cout << "SUCCESS: All ranks match the expected sum." << std::endl;
    }

    // Print stats
    {
        std::lock_guard<std::mutex> guard(io_lock);
        std::cout << "Bytes sent=" << bytes_sent
                  << ", Bytes received=" << bytes_received << std::endl;
    }

    cleanup_sockets();
    return all_ok ? 0 : 1;
}
#endif
