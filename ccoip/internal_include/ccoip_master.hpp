#pragma once

#include <ccoip_inet.h>

namespace ccoip {
    class CCoIPMasterHandler;

    class CCoIPMaster {
    private:
        ccoip_socket_address_t listen_address;
        CCoIPMasterHandler *master;

    public:
        explicit CCoIPMaster(const ccoip_socket_address_t &listen_address);
        CCoIPMaster(const CCoIPMaster &other) = delete;
        CCoIPMaster(CCoIPMaster &&other) = delete;
        CCoIPMaster &operator=(const CCoIPMaster &other) = delete;
        CCoIPMaster &operator=(CCoIPMaster &&other) = delete;

        /// returns false if the handler is already running or has been interrupted
        [[nodiscard]] bool launch();

        /// returns false if the handler has already been interrupted or was never launched
        [[nodiscard]] bool interrupt() const;

        /// returns false if the handler is not running
        /// blocks until the handler has terminated
        [[nodiscard]] bool join() const;

        ~CCoIPMaster();
    };
};
