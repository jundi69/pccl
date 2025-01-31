#include "ccoip_master_handler.hpp"

#include "ccoip_master.hpp"

ccoip::CCoIPMaster::CCoIPMaster(const ccoip_socket_address_t &listen_address): listen_address(listen_address), master(nullptr) {
}

bool ccoip::CCoIPMaster::launch() {
    if (master != nullptr) {
        return false;
    }
    master = new CCoIPMasterHandler(listen_address);
    return master->run();
}

bool ccoip::CCoIPMaster::interrupt() const {
    if (master == nullptr) {
        return false;
    }
    return master->interrupt();
}

bool ccoip::CCoIPMaster::join() const {
    if (master == nullptr) {
        return false;
    }
    return master->join();
}

ccoip::CCoIPMaster::~CCoIPMaster() {
    delete master;
}
