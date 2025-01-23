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

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cerrno>

// For byte-order helpers (htole32, le32toh). If not available, implement manually.
#ifdef __linux__
#include <endian.h>
#else
// Fallback for systems without <endian.h>
#include <cstdint>
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
static uint32_t htole32(const uint32_t x) {
    return x;
}

static uint32_t le32toh(const uint32_t x) {
    return x;
}
#else
#error "Unsupported byte order"
#endif
#endif

///////////////////////////////////////////////////////////////////////////////
// Global data and constants
///////////////////////////////////////////////////////////////////////////////
static int g_world_size = 0;
static constexpr int BASE_PORT = 9000;
static std::vector<int> all_sockets; // Keep track of every socket so we can close them
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
            // Error or connection closed prematurely
            return false;
        }
        total_read += static_cast<size_t>(rd);
    }
    return true;
}

///////////////////////////////////////////////////////////////////////////////
// Socket-based “mailbox” initialization
///////////////////////////////////////////////////////////////////////////////
bool init_mailboxes(const int new_world_size) {
    cleanup_sockets(); // ensure old sockets are closed

    {
        std::lock_guard guard(io_lock);
        bytes_sent = 0;
        bytes_received = 0;
    }

    g_world_size = new_world_size;

    // Prepare space for TX/RX (each rank has a map)
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
        // Enable reuse
        int optval = 1;
        ::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

        sockaddr_in addr{};
        std::memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_port = htons(BASE_PORT + rank);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK); // "localhost"

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

    // 2) Connect from each rank to every other rank (TX side)
    for (int rank = 0; rank < g_world_size; ++rank) {
        int peer = (rank + 1) % g_world_size;

        // establish needed p2p connection
        {
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

            if (::connect(client_sock, reinterpret_cast<sockaddr *>(&peer_addr), sizeof(peer_addr)) < 0) {
                std::cerr << "[Rank " << rank << "] connect to rank "
                        << peer << " failed: " << strerror(errno) << "\n";
                ::close(client_sock);
                return false;
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
    }

    // 3) Accept incoming connections on each rank's listening socket (RX side).
    // Each rank expects exactly (g_world_size - 1) inbound connections.
    for (int rank = 0; rank < g_world_size; ++rank) {
        {
            sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            int conn_fd = ::accept(server_sockets[rank], reinterpret_cast<sockaddr *>(&client_addr), &client_len);
            if (conn_fd < 0) {
                std::cerr << "Error in accept() for rank " << rank << ": " << strerror(errno) << "\n";
                return false;
            }
            all_sockets.push_back(conn_fd);

            // Read remote rank "hello"
            char buf[4];
            if (!read_exact(conn_fd, buf, 4)) {
                std::cerr << "Error reading remote rank in accept()\n";
                ::close(conn_fd);
                return false;
            }
            int32_t remote_le;
            std::memcpy(&remote_le, buf, 4);
            int remote_rank = static_cast<int>(le32toh(remote_le));

            p2p_connections_rx[rank][remote_rank] = conn_fd;
        }
    }

    // 4) Close the server sockets now that all inbound connections are accepted
    for (int rank = 0; rank < g_world_size; ++rank) {
        ::close(server_sockets[rank]);
        // remove from the global list
        auto it = std::find(all_sockets.begin(), all_sockets.end(), server_sockets[rank]);
        if (it != all_sockets.end()) {
            all_sockets.erase(it);
        }
    }

    return true;
}

///////////////////////////////////////////////////////////////////////////////
// send_bytes / recv_bytes
///////////////////////////////////////////////////////////////////////////////

void send_bytes(const int src_rank, const int dst_rank, const char *data, const size_t n) {
    print_lock.lock();
    std::cout << "send_bytes: src=" << src_rank << ", dst=" << dst_rank << ", n=" << n << "\n";
    print_lock.unlock();

    const int sockfd = p2p_connections_tx[src_rank][dst_rank];
    size_t total_sent = 0;
    while (total_sent < n) {
        const ssize_t sent = ::send(sockfd, data + total_sent, n - total_sent, 0);
        if (sent <= 0) {
            std::cerr << "send() failed from rank " << src_rank << " to rank " << dst_rank
                    << " with errno=" << strerror(errno) << "\n";
            return;
        }
        total_sent += static_cast<size_t>(sent);
    } {
        std::lock_guard guard(io_lock);
        bytes_sent += n;
    }
}

std::vector<double> recv_bytes(const int dst_rank, const int src_rank, const size_t n_elems) {
    // n_elems is number of double elements
    const int sockfd = p2p_connections_rx[dst_rank][src_rank];
    const size_t n_bytes = n_elems * sizeof(double);
    std::vector<char> buffer(n_bytes);

    size_t total_read = 0;
    while (total_read < n_bytes) {
        const ssize_t rd = ::recv(sockfd, buffer.data() + total_read, n_bytes - total_read, 0);
        if (rd <= 0) {
            std::cerr << "recv() failed (rank " << dst_rank << " from " << src_rank
                    << "), errno=" << strerror(errno) << "\n";
            // Return an empty vector on error
            return {};
        }
        total_read += static_cast<size_t>(rd);
    } {
        std::lock_guard guard(io_lock);
        bytes_received += n_bytes;
    }

    // Convert raw bytes to double vector
    std::vector<double> result(n_elems);
    std::memcpy(result.data(), buffer.data(), n_bytes);
    return result;
}

///////////////////////////////////////////////////////////////////////////////
// Partition a 1D array of 'length' into 'world_size' contiguous chunks
///////////////////////////////////////////////////////////////////////////////
std::vector<std::pair<int, int>> compute_chunk_boundaries(const int length, const int world_size) {
    // boundaries[i] = (start, end) of chunk i
    std::vector<std::pair<int, int>> boundaries(world_size);
    const int base = length / world_size;
    const int remainder = length % world_size;
    int start = 0;
    for (int i = 0; i < world_size; ++i) {
        const int sz = base + ((i < remainder) ? 1 : 0);
        int end = start + sz;
        boundaries[i] = std::make_pair(start, end);
        start = end;
    }
    return boundaries;
}

///////////////////////////////////////////////////////////////////////////////
// PHASE 1: Ring Reduce-Scatter
///////////////////////////////////////////////////////////////////////////////
std::map<int, std::vector<double>> ring_reduce_scatter(const int rank,
                                                       const std::vector<double> &local_data,
                                                       const std::vector<std::pair<int, int>> &boundaries) {
    /**
     * chunk_val[chunk_idx] = data for that chunk
     * has_added[chunk_idx] = whether we have added local_data into that chunk yet
     */
    std::map<int, std::vector<double>> chunk_val;
    std::vector has_added(g_world_size, false);

    // This rank initially "owns" chunk == rank
    const int s_i = boundaries[rank].first;
    if (const int e_i = boundaries[rank].second; e_i > s_i) {
        chunk_val[rank] = std::vector(local_data.begin() + s_i, local_data.begin() + e_i);
        has_added[rank] = true;
    }

    // Pipeline steps = P - 1
    for (int step = 0; step < g_world_size - 1; ++step) {
        // chunk_to_send is the chunk this rank currently holds that must move on
        int chunk_to_send = (rank - step + g_world_size) % g_world_size;
        const int next_rank = (rank + 1) % g_world_size;
        const int prev_rank = (rank - 1 + g_world_size) % g_world_size;

        // Send the chunk if we have it, else send empty
        std::vector<double> arr_to_send;
        if (auto it = chunk_val.find(chunk_to_send); it != chunk_val.end()) {
            arr_to_send = it->second;
            // remove it since we are passing it on
            chunk_val.erase(it);
        }

        // Send out arr_to_send
        if (!arr_to_send.empty()) {
            send_bytes(rank, next_rank,
                       reinterpret_cast<const char *>(arr_to_send.data()),
                       arr_to_send.size() * sizeof(double));
        } else {
            // send zero bytes if empty
            send_bytes(rank, next_rank, nullptr, 0);
        }

        // Figure out which chunk we will receive from the prev_rank
        // This is the chunk with index (prev_rank - step) mod P
        int prev_chunk_idx = (prev_rank - step + g_world_size) % g_world_size;
        const int s_j = boundaries[prev_chunk_idx].first;
        const int e_j = boundaries[prev_chunk_idx].second;
        const int chunk_size = e_j - s_j;

        // Now receive the chunk from prev_rank
        std::vector<double> inc_arr;
        if (chunk_size > 0) {
            inc_arr = recv_bytes(rank, prev_rank, chunk_size);
        } else {
            // if chunk_size=0, we expect an empty message
            inc_arr = recv_bytes(rank, prev_rank, 0);
        }

        // If we haven't added local_data to this chunk yet, add it
        if (!inc_arr.empty() && !has_added[prev_chunk_idx] && chunk_size > 0) {
            for (int i = 0; i < chunk_size; ++i) {
                inc_arr[i] += local_data[s_j + i];
            }
            has_added[prev_chunk_idx] = true;
        }

        chunk_val[prev_chunk_idx] = inc_arr;
    }

    return chunk_val;
}

///////////////////////////////////////////////////////////////////////////////
// PHASE 2: Ring Allgather (Pipeline)
///////////////////////////////////////////////////////////////////////////////
std::map<int, std::vector<double>> ring_allgather_pipeline(const int rank,
                                                           std::map<int, std::vector<double>> chunk_val,
                                                           const std::vector<std::pair<int, int>> &boundaries) {
    // We expect chunk_val to have exactly 1 entry at the start
    // We'll pick it up, pass it around, and gather new ones
    if (chunk_val.size() != 1) {
        std::cerr << "[Rank " << rank << "] Error: ring_reduce_scatter produced "
                << chunk_val.size() << " chunks, expected exactly 1.\n";
    }
    // Identify the single chunk we have
    std::vector<double> current_chunk_arr; {
        if (const auto it = chunk_val.begin(); it != chunk_val.end()) {
            current_chunk_arr = it->second;
        }
    }

    for (int step = 0; step < g_world_size - 1; ++step) {
        const int next_rank = (rank + 1) % g_world_size;
        const int prev_rank = (rank - 1 + g_world_size) % g_world_size;

        // Send the chunk we currently have
        if (!current_chunk_arr.empty()) {
            send_bytes(rank, next_rank,
                       reinterpret_cast<const char *>(current_chunk_arr.data()),
                       current_chunk_arr.size() * sizeof(double));
        } else {
            send_bytes(rank, next_rank, nullptr, 0);
        }

        // Which chunk are we receiving from prev_rank now?
        // inc_idx = (prev_rank + 1 - step) mod g_world_size
        int inc_idx = (prev_rank + 1 - step + g_world_size) % g_world_size;
        const int s_j = boundaries[inc_idx].first;
        const int e_j = boundaries[inc_idx].second;
        const int chunk_size = e_j - s_j;

        std::vector<double> inc_arr;
        if (chunk_size > 0) {
            inc_arr = recv_bytes(rank, prev_rank, chunk_size);
        } else {
            inc_arr = recv_bytes(rank, prev_rank, 0);
        }

        // Save it
        chunk_val[inc_idx] = inc_arr;
        current_chunk_arr = inc_arr;
    }

    return chunk_val;
}

///////////////////////////////////////////////////////////////////////////////
// Reassemble distributed chunks into a single vector
///////////////////////////////////////////////////////////////////////////////
std::vector<double> reassemble_chunks(const std::map<int, std::vector<double>> &chunk_val,
                                      const std::vector<std::pair<int, int>> &boundaries,
                                      const int total_length) {
    std::vector out(total_length, 0.0);
    for (auto &kv: chunk_val) {
        const int i = kv.first;
        auto &arr = kv.second;
        const int s_i = boundaries[i].first;
        const int e_i = boundaries[i].second;
        if (!arr.empty()) {
            std::memcpy(&out[s_i], arr.data(), (e_i - s_i) * sizeof(double));
        }
    }
    return out;
}

///////////////////////////////////////////////////////////////////////////////
// Complete ring_allreduce
///////////////////////////////////////////////////////////////////////////////
std::vector<double> ring_allreduce(const int rank, const std::vector<double> &local_data) {
    const int length = static_cast<int>(local_data.size());
    const auto boundaries = compute_chunk_boundaries(length, g_world_size);
    const auto partials = ring_reduce_scatter(rank, local_data, boundaries);
    const auto final_map = ring_allgather_pipeline(rank, partials, boundaries);
    return reassemble_chunks(final_map, boundaries, length);
}

///////////////////////////////////////////////////////////////////////////////
// Worker thread function
///////////////////////////////////////////////////////////////////////////////
void worker(const int rank, const std::vector<double> &local_data, std::vector<std::vector<double>> &results) {
    const auto arr = ring_allreduce(rank, local_data);
    results[rank] = arr;

    print_lock.lock();
    std::cout << "[Rank " << rank << "] final out: ";
    for (const auto v: arr) {
        std::cout << v << " ";
    }
    std::cout << std::endl;
    print_lock.unlock();
}

#ifdef RING_REDUCE_EXPERIMENT_MAIN_IMPL
int main() {
    constexpr int test_world_size = 3;

    // 1) Initialize the "mailboxes" (socket layer)
    if (!init_mailboxes(test_world_size)) {
        std::cerr << "init_mailboxes failed.\n";
        return 1;
    }

    // 2) Generate example data for each rank
    constexpr int length = 3;
    std::mt19937 rng(42);
    std::normal_distribution dist(0.0, 1.0);

    // Each rank has a length-sized array
    std::vector all_data(test_world_size, std::vector<double>(length));
    for (int r = 0; r < test_world_size; ++r) {
        for (int i = 0; i < length; ++i) {
            all_data[r][i] = dist(rng);
        }
    }

    // Compute the expected sum across ranks
    std::vector expected(length, 0.0);
    for (int r = 0; r < test_world_size; ++r) {
        for (int i = 0; i < length; ++i) {
            expected[i] += all_data[r][i];
        }
    }

    std::cout << "Expected sum: ";
    for (const auto v: expected) {
        std::cout << v << " ";
    }
    std::cout << std::endl;

    // 3) Launch each rank in a separate thread
    std::vector results(test_world_size, std::vector<double>(0));
    std::vector<std::thread> threads;
    threads.reserve(test_world_size);
    for (int r = 0; r < test_world_size; ++r) {
        threads.emplace_back([r, &all_data, &results]() {
            worker(r, all_data[r], results);
        });
    }
    for (auto &t: threads) {
        t.join();
    }

    // 4) Check correctness
    bool all_ok = true;
    for (int r = 0; r < test_world_size; ++r) {
        if (results[r].size() != static_cast<size_t>(length)) {
            std::cerr << "[Rank " << r << "] result size mismatch!\n";
            all_ok = false;
            continue;
        }
        for (int i = 0; i < length; ++i) {
            if (const double diff = results[r][i] - expected[i]; std::fabs(diff) > 1e-10) {
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

    // Print bytes sent/received
    {
        std::lock_guard guard(io_lock);
        std::cout << "Bytes sent=" << bytes_sent
                << ", Bytes received=" << bytes_received << std::endl;
    }

    // Cleanup
    cleanup_sockets();
    return (all_ok ? 0 : 1);
}
#endif
