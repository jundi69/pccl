#pragma once

#include <ccoip_inet.h>
#include <ccoip_master_state.hpp>
#include <ccoip_packets.hpp>
#include <tinysockets.hpp>
#include <vector>
#include <thread>

namespace ccoip {
    class CCoIPMasterHandler {
        /// Server socket
        tinysockets::ServerSocket server_socket;

        /// Thread ID of the server thread.
        /// Server socket callbacks such as @code onClientRead @endcode and @code onClientDisconnect @endcode
        /// will only ever be invoked from this thread.
        std::thread::id server_thread_id;

        /// Encapsulates the state of the master node
        /// All administrative state and transactions are encapsulated in this object
        CCoIPMasterState server_state;

        /// Topology optimization moonshot thread.
        /// This thread is launched to asynchronously optimize the topology as well as possible.
        std::optional<std::thread> topology_optimization_moonshot_thread = std::nullopt;

        /// Flag to indicate if the topology optimization moonshot thread is running.
        std::atomic_bool topology_optimization_moonshot_thread_running = false;

        /// Temporary exchange variable used by the moonshot topology optimization thread to deposit the new topology.
        /// Will be used to update the real topology in the master state afterward.
        std::vector<ccoip_uuid_t> next_ring_topology{};

        /// Temporary exchange variable used by the moonshot topology optimization thread to deposit the new topology.
        /// Will be used to update the real topology in the master state afterward.
        /// Indicates whether @code next_ring_topology@endcode is optimal.
        bool next_ring_is_optimal = false;

    public:
        volatile bool running = false;
        volatile bool interrupted = false;

        explicit CCoIPMasterHandler(const ccoip_socket_address_t &listen_address);

        [[nodiscard]] bool run();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool join();

        [[nodiscard]] bool kickClient(const ccoip_socket_address_t &client_address) const;

        ~CCoIPMasterHandler();

    private:
        void sendP2PConnectionInformation(bool changed, const ClientInfo &client_info);

        void sendP2PConnectionInformation();

        [[nodiscard]] bool checkEstablishP2PConnectionConsensus();

        [[nodiscard]] bool checkTopologyOptimizationConsensus();

        [[nodiscard]] bool checkTopologyOptimizationCompletionConsensus();

        [[nodiscard]] bool checkP2PConnectionsEstablished();

        [[nodiscard]] bool checkSyncSharedStateConsensus(uint32_t peer_group);

        [[nodiscard]] bool checkSyncSharedStateCompleteConsensus(uint32_t peer_group);

        [[nodiscard]] bool checkCollectiveCommsInitiateConsensus(uint32_t peer_group, uint64_t tag, bool topology_changed);

        [[nodiscard]] bool checkCollectiveCommsCompleteConsensus(uint32_t peer_group, uint64_t tag);


        /// Finds the optimal peer to distribute the shared state to the specified requester.
        std::optional<ccoip_socket_address_t> findBestSharedStateTxPeer(const ccoip_uuid_t &peer_uuid);

        // packet handling functions
        void handleEstablishP2PConnections(const ccoip_socket_address_t &client_address,
                                  const C2MPacketRequestEstablishP2PConnections &packet);

        void handleRequestSessionJoin(const ccoip_socket_address_t &client_address,
                                      const C2MPacketRequestSessionRegistration &packet);


        void handleP2PConnectionsEstablished(const ccoip_socket_address_t &client_address,
                                             const C2MPacketP2PConnectionsEstablished &packet);

        void handleGetTopologyRequest(const ccoip_socket_address_t &client_address,
                                      const C2MPacketGetTopologyRequest &packet);

        void handleOptimizeTopology(const ccoip_socket_address_t &client_address,
                                   const C2MPacketOptimizeTopology &packet);

        void handleReportPeerBandwidth(const ccoip_socket_address_t &client_address,
                                      const C2MPacketReportPeerBandwidth &packet);

        void handleOptimizeTopologyWorkComplete(const ccoip_socket_address_t &client_address,
                                               const C2MPacketOptimizeTopologyWorkComplete &packet);

        void handleSyncSharedState(const ccoip_socket_address_t &client_address,
                                   const C2MPacketSyncSharedState &packet);

        void handleSyncSharedStateComplete(const ccoip_socket_address_t &client_address,
                                           const C2MPacketDistSharedStateComplete &packet);

        void handleCollectiveCommsInitiate(const ccoip_socket_address_t &client_address,
                                           const C2MPacketCollectiveCommsInitiate &packet);

        void sendCollectiveCommsAbortPackets(uint32_t peer_group, uint64_t tag, bool aborted);

        void handleCollectiveCommsComplete(const ccoip_socket_address_t &client_address,
                                           const C2MPacketCollectiveCommsComplete &packet);

        // server socket callbacks
        void onClientRead(const ccoip_socket_address_t &client_address,
                          const std::span<uint8_t> &data);

        void onClientDisconnect(const ccoip_socket_address_t &client_address);
    };
}
