#pragma once

#include <ccoip_inet.h>

namespace ccoip {
    class CCoIPClientHandler;

    class CCoIPClient {
        CCoIPClientHandler *client;

    public:
        explicit CCoIPClient(const ccoip_socket_address_t &listen_address);

        CCoIPClient(const CCoIPClient &other) = delete;
        CCoIPClient(CCoIPClient &&other) = delete;
        CCoIPClient &operator=(const CCoIPClient &other) = delete;
        CCoIPClient &operator=(CCoIPClient &&other) = delete;

        /// Connect to the master
        [[nodiscard]] bool connect() const;

        /// Accept new peers if necessary and establish p2p connections
        [[nodiscard]] bool acceptNewPeers() const;

        /// Interrupt the client
        [[nodiscard]] bool interrupt() const;

        /// Performs a topology update if required
        [[nodiscard]] bool updateTopology() const;

        /// Wait for the client to gracefully terminate after interruption
        [[nodiscard]] bool join() const;

        ~CCoIPClient();
    };
};
