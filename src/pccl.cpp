#include "pccl.h"

#include <ccoip_master_handler.h>

pcclResult_t pcclInit() {
    PCCL_SUCCEED();
}

pcclResult_t pcclCreateCommunicator(struct pcclComm_t **comm_out) {
    PCCL_SUCCEED();
}

pcclResult_t pcclGetAttribute(const struct pcclComm_t *communicator, enum pcclAttribute_t attribute,
                              int *p_attribute_out) {
    PCCL_SUCCEED();
}

pcclResult_t pcclTopologySaveGraph(const struct pcclComm_t *communicator, const char *filename) {
    PCCL_SUCCEED();
}

pcclResult_t pcclSaveReducePlan(const struct pcclComm_t *communicator, const char *filename) {
    PCCL_SUCCEED();
}

pcclResult_t pcclDestroyCommunicator(struct pcclComm_t *communicator) {
    PCCL_SUCCEED();
}

pcclResult_t pcclConnectMaster(struct pcclComm_t *comm, struct ccoip_socket_address_t socket_address) {
    PCCL_SUCCEED();
}

pcclResult_t pcclAcceptNewPeers(struct pcclComm_t *comm) {
    PCCL_SUCCEED();
}

pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, enum pcclDataType_t datatype,
                           enum pcclRedOp_t op, uint64_t tag, const struct pcclComm_t *comm,
                           struct pcclReduceInfo_t *reduce_info_out) {
    PCCL_SUCCEED();
}

pcclResult_t pcclCommunicateSharedState(const struct pcclComm_t *comm, struct pcclSharedState_t *shared_state) {
    PCCL_SUCCEED();
}

pcclResult_t pcclPollSharedState(const struct pcclComm_t *comm, struct pcclSharedState_t *shared_state) {
    PCCL_SUCCEED();
}

struct pcclMasterInstanceState_t {
    std::unique_ptr<ccoip::CCoIPMasterHandler> master_handler;
};

pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address, struct pcclMasterInstance_t *p_master_handle_out) {
    PCCL_VALIDATE(p_master_handle_out != nullptr, pcclInvalidArgument);
    *p_master_handle_out = {
        .state = new pcclMasterInstanceState_t{
            .master_handler = std::make_unique<ccoip::CCoIPMasterHandler>(listen_address),
        }
    };
    PCCL_SUCCEED();
}

pcclResult_t pcclRunMaster(const pcclMasterInstance_t master_instance) {
    PCCL_VALIDATE(master_instance.state != nullptr, pcclInvalidArgument);
    if (!master_instance.state->master_handler->launch()) {
        PCCL_FAIL(pcclInvalidUsage);
    }
    PCCL_SUCCEED();
}

pcclResult_t pcclInterruptMaster(const pcclMasterInstance_t master_instance) {
    PCCL_VALIDATE(master_instance.state != nullptr, pcclInvalidArgument);
    if (!master_instance.state->master_handler->interrupt()) {
        PCCL_FAIL(pcclInvalidUsage);
    }
    PCCL_SUCCEED();
}

pcclResult_t pcclMasterAwaitTermination(struct pcclMasterInstance_t master_handle) {
    PCCL_VALIDATE(master_handle.state != nullptr, pcclInvalidArgument);

    PCCL_SUCCEED();
}

pcclResult_t pcclDestroyMaster(pcclMasterInstance_t master_instance) {
    PCCL_VALIDATE(master_instance.state != nullptr, pcclInvalidArgument);
    delete master_instance.state;
    master_instance.state = nullptr;
    PCCL_SUCCEED();
}
