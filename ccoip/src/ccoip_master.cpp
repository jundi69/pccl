#include "ccoip_master.hpp"

#include <tinysockets.hpp>


ccoip::CCoIPMaster::CCoIPMaster(const ccoip_socket_address_t &listen_address) : server_socket(listen_address) {
}

bool ccoip::CCoIPMaster::run() {
    if (!server_socket.bind()) {
        return false;
    }
    if (!server_socket.listen()) {
        return false;
    }
    if (!server_socket.runAsync()) {
        return false;
    }
    running = true;
    return true;
}

bool ccoip::CCoIPMaster::interrupt() {
    if (interrupted) {
        return false;
    }
    if (!server_socket.interrupt()) {
        return false;
    }
    interrupted = true;
    return true;
}

bool ccoip::CCoIPMaster::join() {
    if (!running) {
        return false;
    }
    server_socket.join();
    return true;
}

ccoip::CCoIPMaster::~CCoIPMaster() = default;
