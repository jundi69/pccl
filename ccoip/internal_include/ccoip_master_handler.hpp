#pragma once

#include <ccoip_inet.h>

namespace ccoip {
    struct CCoIPMaster;

    class CCoIPMasterHandler {
    private:
        ccoip_socket_address_t listen_address;
        CCoIPMaster *master;

    public:
        explicit CCoIPMasterHandler(const ccoip_socket_address_t &listen_address);
        CCoIPMasterHandler(const CCoIPMasterHandler &other) = delete;
        CCoIPMasterHandler(CCoIPMasterHandler &&other) = delete;
        CCoIPMasterHandler &operator=(const CCoIPMasterHandler &other) = delete;
        CCoIPMasterHandler &operator=(CCoIPMasterHandler &&other) = delete;

        /// returns false if the handler is already running or has been interrupted
        [[nodiscard]] bool launch();

        /// returns false if the handler has already been interrupted or was never launched
        [[nodiscard]] bool interrupt() const;

        /// returns false if the handler is not running
        /// blocks until the handler has terminated
        [[nodiscard]] bool join() const;

        ~CCoIPMasterHandler();
    };
};
