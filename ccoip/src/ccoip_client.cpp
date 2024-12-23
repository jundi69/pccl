#include "ccoip_client.hpp"

#include <ccoip_client_handler.hpp>

ccoip::CCoIPClient::CCoIPClient(const ccoip_socket_address_t &master_socket_address,
                                const uint32_t peer_group) : client(
    new CCoIPClientHandler(master_socket_address, peer_group)) {
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

bool ccoip::CCoIPClient::allReduceAsync(const void *sendbuff, void *recvbuff, const size_t count, const ccoip_data_type_t datatype,
                                        const ccoip_reduce_op_t op, const uint64_t tag) const {
    return client->allReduceAsync(sendbuff, recvbuff, count, datatype, op, tag);
}

bool ccoip::CCoIPClient::joinAsyncReduce(const uint64_t tag) const {
    return client->joinAsyncReduce(tag);
}

bool ccoip::CCoIPClient::getAsyncReduceInfo(const uint64_t tag, std::optional<ccoip_reduce_info_t> &info_out) const {
    return client->getAsyncReduceInfo(tag, info_out);
}

bool ccoip::CCoIPClient::join() const {
    return client->join();
}

bool ccoip::CCoIPClient::isInterrupted() const {
    return client->isInterrupted();
}

bool ccoip::CCoIPClient::isAnyCollectiveComsOpRunning() const {
    return client->isAnyCollectiveComsOpRunning();
}

ccoip::CCoIPClient::~CCoIPClient() {
    delete client;
}
