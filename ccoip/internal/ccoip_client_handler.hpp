#pragma once

#include <ccoip_client.hpp>
#include <ccoip_client_state.hpp>
#include <ccoip_packets.hpp>
#include <ccoip_shared_state.hpp>

#include <tinysockets.hpp>

namespace ccoip {
    class CCoIPClientHandler {
        /// Blocking socket for connection to master node
        tinysockets::QueuedSocket master_socket;

        /// All state of the client is encapsulated in this object
        CCoIPClientState client_state;

        /// Socket listening for p2p connections
        tinysockets::BlockingIOServerSocket p2p_socket;

        /// Socket listening for shared state distribution requests
        tinysockets::ServerSocket shared_state_socket;

        /// Socket listening for bandwidth benchmark requests
        tinysockets::BlockingIOServerSocket benchmark_socket;

        /// Thread ID of the shared state server thread
        std::thread::id shared_state_server_thread_id;

        /// Open p2p connections; Tx connections (we have established this connection to the peer)
        /// Maps the destination UUID to the list of connections of the connection pool.
        /// Any of the connections can be used to send data to the peer.
        std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>> p2p_connections_tx;

        /// Open p2p connections; Rx connections (peer has established this connection to us)
        /// Maps the source UUID to the list of connections of the connection pool.
        /// Any of the connections can be used to receive data from the peer.
        std::unordered_map<ccoip_uuid_t, std::vector<std::shared_ptr<tinysockets::MultiplexedIOSocket>>> p2p_connections_rx;
        std::shared_mutex p2p_connections_rx_mutex;

        /// Peer group of the client
        uint32_t peer_group;

        /// The id of the designated main thread
        std::thread::id main_thread_id;

        /// Number of p2p connections to create for each peer
        uint32_t p2p_connection_pool_size;

        bool interrupted = false;

        bool accepted = false;

        /// Connection revision number;
        /// Sequentially incremented each time p2p connections are re-established
        uint64_t connection_revision{};

        /// Threads for currently ongoing bandwidth benchmarks (if any).
        /// These threads run the receiving side of a particular benchmark connection respectively.
        /// Only one peer can be benchmarked at a time;
        /// however, it may open multiple connections for benchmarking total throughput across multiple connections.
        std::vector<std::thread> current_benchmark_threads{};
        std::mutex current_benchmark_threads_mutex{};

        /// Atomic integer for the number of currently running benchmark threads.
        std::atomic<uint64_t> num_running_benchmark_threads{};

        /// The uuid of the peer that is currently being benchmarked.
        /// Only additional connections from the peer that is being benchmarked are accepted.
        /// @note: protected by current_benchmark_threads_mutex
        ccoip_uuid_t benchmark_peer{};

    public:
        explicit CCoIPClientHandler(const ccoip_socket_address_t &address, uint32_t peer_group, uint32_t p2p_connection_pool_size);

        [[nodiscard]] bool connect();

        [[nodiscard]] bool requestAndEstablishP2PConnections(bool accept_new_peers);

        [[nodiscard]] bool arePeersPending(bool &pending_out);

        [[nodiscard]] bool syncSharedState(ccoip_shared_state_t &shared_state,
                                           ccoip_shared_state_sync_info_t &info_out);

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool optimizeTopology();

        [[nodiscard]] bool join();

        [[nodiscard]] bool isInterrupted() const;

        ~CCoIPClientHandler();

        [[nodiscard]] bool allReduceAsync(const void *sendbuff, void *recvbuff, size_t count,
                                          ccoip_data_type_t datatype, ccoip_data_type_t quantized_data_type,
                                          ccoip_quantization_algorithm_t quantization_algorithm, ccoip_reduce_op_t op,
                                          uint64_t tag);

        [[nodiscard]] bool joinAsyncReduce(uint64_t tag);

        [[nodiscard]] bool getAsyncReduceInfo(uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out);

        [[nodiscard]] bool isAnyCollectiveComsOpRunning();

        [[nodiscard]] size_t getGlobalWorldSize() const;

        [[nodiscard]] size_t getLocalWorldSize() const;

        [[nodiscard]] size_t getLargestPeerGroupWorldSize() const;

        [[nodiscard]] size_t getNumDistinctPeerGroups() const;

        void setMainThread(const std::thread::id main_thread_id);

    private:
        enum EstablishP2PConnectionResult { SUCCESS = 0, FAILED = 1, RETRY_NEEDED = 2 };
        [[nodiscard]] EstablishP2PConnectionResult establishP2PConnections();

        [[nodiscard]] bool establishP2PConnections(const PeerInfo &peer);

        [[nodiscard]] bool closeP2PConnection(const ccoip_uuid_t &uuid, tinysockets::MultiplexedIOSocket &socket);

        // shared state packet handlers
        void handleSharedStateRequest(const ccoip_socket_address_t &client_address,
                                      const C2SPacketRequestSharedState &packet);

        // shared state server socket callbacks
        void onSharedStateClientRead(const ccoip_socket_address_t &client_address, const std::span<std::uint8_t> &data);
    };
} // namespace ccoip
