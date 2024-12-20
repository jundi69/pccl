#pragma once

#include <ccoip_inet.h>
#include <ccoip_shared_state.hpp>

namespace ccoip {
    class CCoIPClientHandler;

    class CCoIPClient {
        CCoIPClientHandler *client;

    public:
        explicit CCoIPClient(const ccoip_socket_address_t &master_socket_address);

        CCoIPClient(const CCoIPClient &other) = delete;

        CCoIPClient(CCoIPClient &&other) = delete;

        CCoIPClient &operator=(const CCoIPClient &other) = delete;

        CCoIPClient &operator=(CCoIPClient &&other) = delete;

        /// Connect to the master
        [[nodiscard]] bool connect() const;

        /// Accept new peers if necessary and establish p2p connections
        [[nodiscard]] bool acceptNewPeers() const;

        /// Synchronize the shared state
        /// @param shared_state the shared state object referencing user-owned memory to be synchronized
        /// @param info_out is populated with information about this sync operation such as the number of bytes sent/received to facilitate it
        [[nodiscard]] bool syncSharedState(ccoip_shared_state_t &shared_state,
                                           ccoip_shared_state_sync_info_t &info_out) const;

        /// Interrupt the client
        [[nodiscard]] bool interrupt() const;

        /// Performs a topology update if required
        [[nodiscard]] bool updateTopology() const;

        /// Wait for the client to gracefully terminate after interruption
        [[nodiscard]] bool join() const;

        /// Returns true if the client has been interrupted
        [[nodiscard]] bool isInterrupted() const;

        ~CCoIPClient();
    };
};
