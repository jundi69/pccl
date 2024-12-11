#include "ccoip_client.hpp"

#include <ccoip_client_handler.hpp>

ccoip::CCoIPClient::CCoIPClient(const ccoip_socket_address_t &listen_address) :
client(new CCoIPClientHandler(listen_address)) {
}

bool ccoip::CCoIPClient::connect() const {
    return client->connect();
}

bool ccoip::CCoIPClient::acceptNewPeers() const {
    return client->acceptNewPeers();
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

ccoip::CCoIPClient::~CCoIPClient() {
    delete client;
}
