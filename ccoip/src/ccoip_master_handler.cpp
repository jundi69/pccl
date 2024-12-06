#include "ccoip_master_handler.h"

#include "ccoip_master.h"

#include <thread>

ccoip::CCoIPMasterHandler::CCoIPMasterHandler(const ccoip_socket_address_t &listen_address): listen_address(listen_address) {
}

bool ccoip::CCoIPMasterHandler::launch() {
    if (master != nullptr || main_thread != nullptr) {
        return false;
    }
    master = std::make_unique<ccoip::CCoIPMaster>(listen_address);
    main_thread = std::make_unique<std::thread>(&CCoIPMaster::run, master.get());
    return true;
}

bool ccoip::CCoIPMasterHandler::interrupt() const {
    if (master == nullptr || main_thread == nullptr) {
        return false;
    }
    master->interrupted = true;
    return true;
}

bool ccoip::CCoIPMasterHandler::join() const {
    if (master == nullptr || main_thread == nullptr) {
        return false;
    }
    main_thread->join();
    return true;
}
