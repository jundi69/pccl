#include "ccoip_master_handler.hpp"

#include "ccoip_master.hpp"

ccoip::CCoIPMasterHandler::CCoIPMasterHandler(const ccoip_socket_address_t &listen_address): listen_address(listen_address), master(nullptr) {
}

bool ccoip::CCoIPMasterHandler::launch() {
    if (master != nullptr) {
        return false;
    }
    master = new CCoIPMaster(listen_address);
    return master->run();
}

bool ccoip::CCoIPMasterHandler::interrupt() const {
    if (master == nullptr) {
        return false;
    }
    return master->interrupt();
}

bool ccoip::CCoIPMasterHandler::join() const {
    if (master == nullptr) {
        return false;
    }
    return master->join();
}

ccoip::CCoIPMasterHandler::~CCoIPMasterHandler() {
    delete master;
}
