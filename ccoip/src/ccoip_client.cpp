#include "ccoip_client.hpp"

#include <ccoip_client_handler.hpp>

ccoip::CCoIPClient::CCoIPClient(const ccoip_socket_address_t &master_socket_address) : client(
    new CCoIPClientHandler(master_socket_address)) {
}

bool ccoip::CCoIPClient::connect() const {
    return client->connect();
}

bool ccoip::CCoIPClient::acceptNewPeers() const {
    return client->acceptNewPeers();
}

bool ccoip::CCoIPClient::syncSharedState(ccoip_shared_state_t &shared_state,
                                         ccoip_shared_state_sync_info_t &info_out) const {
    return client->syncSharedState(shared_state, info_out);
}

bool ccoip::CCoIPClient::interrupt() const {
    return client->interrupt();
}

bool ccoip::CCoIPClient::updateTopology() const {
    return client->updateTopology();
}

bool ccoip::CCoIPClient::join() const {
    return client->join();
}

bool ccoip::CCoIPClient::isInterrupted() const {
    return client->isInterrupted();
}

ccoip::CCoIPClient::~CCoIPClient() {
    delete client;
}
