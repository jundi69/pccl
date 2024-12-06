#pragma once

#include "ccoip_types.hpp"
#include <tinysockets.hpp>

struct CCoIPClientState;

namespace ccoip {
    class CCoIPClientHandler {
    private:
        tinysockets::BlockingIOSocket client_socket;

        CCoIPClientState *client_state;
    public:
        explicit CCoIPClientHandler(const ccoip_socket_address_t &address);

        CCoIPClientHandler(const CCoIPClientHandler &other) = delete;
        CCoIPClientHandler(CCoIPClientHandler &&other) = delete;
        CCoIPClientHandler &operator=(const CCoIPClientHandler &other) = delete;
        CCoIPClientHandler &operator=(CCoIPClientHandler &&other) = delete;

        [[nodiscard]] bool connect();

        [[nodiscard]] bool acceptNewPeers();

        [[nodiscard]] bool interrupt();

        [[nodiscard]] bool join();

        ~CCoIPClientHandler();

    private:
        void registerPeer(const ccoip_inet_address_t &address, ccoip_uuid_t uuid);
    };
}
