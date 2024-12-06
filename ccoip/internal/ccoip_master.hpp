#pragma once

#include <ccoip_inet.h>

namespace ccoip {
    class CCoIPMaster {
    private:
        ccoip_socket_address_t listen_address;

    public:
        volatile bool running = false;
        volatile bool interrupted = false;

        explicit CCoIPMaster(const ccoip_socket_address_t &listen_address);

        void run();
    };
}
