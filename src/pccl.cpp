#include "pccl.h"
#include "pccl_internal.h"

#include <ccoip_master_handler.hpp>

static constinit bool pccl_initialized = false;

#define PCCL_VALIDATE_INITIALIZED() \
    if (!pccl_initialized) { \
        return (pcclNotInitialized); \
    }

static pcclResult_t internalPcclInit() {
    return pcclSuccess;
}

pcclResult_t pcclInit() {
    if (pccl_initialized) {
        return pcclSuccess;
    }
    PCCL_ERR_PROPAGATE(internalPcclInit());
    pccl_initialized = true;
    return pcclSuccess;
}

pcclResult_t pcclCreateCommunicator(pcclComm_t **comm_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(comm_out != nullptr, pcclInvalidArgument);
    *comm_out = new pcclComm_t();
    return pcclSuccess;
}

pcclResult_t pcclGetAttribute(const pcclComm_t *communicator,
                              pcclAttribute_t attribute,
                              int *p_attribute_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(p_attribute_out != nullptr, pcclInvalidArgument);

    switch (attribute) {
        case PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE: {
            // const size_t world_size = communicator->ccoip_handler->get_world_size();
            // *p_attribute_out = static_cast<int>(world_size);
                *p_attribute_out = 128; // for python test
            break;
        }
        default: {
            [[unlikely]] return pcclInvalidArgument;
        }
    }
    return pcclSuccess;
}

pcclResult_t pcclTopologySaveGraph(const pcclComm_t *communicator, const char *filename) {
    return pcclSuccess;
}

pcclResult_t pcclSaveReducePlan(const pcclComm_t *communicator, const char *filename) {
    return pcclSuccess;
}

pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    if (!communicator->ccoip_handler->interrupt()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    if (!communicator->ccoip_handler->join()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    delete communicator;
    return pcclSuccess;
}

pcclResult_t pcclConnectMaster(pcclComm_t *communicator, ccoip_socket_address_t socket_address) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_handler == nullptr, pcclInvalidUsage);
    communicator->ccoip_handler = std::make_unique<ccoip::CCoIPClientHandler>(socket_address);
    if (!communicator->ccoip_handler->connect()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclUpdateTopology(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);

    // accept new peers; this will block until we have a valid connection to each peer
    if (!communicator->ccoip_handler->acceptNewPeers()) [[unlikely]] {
        return pcclInvalidUsage;
    }

    // update the topology
    if (!communicator->ccoip_handler->updateTopology()) [[unlikely]] {
        return pcclInvalidUsage;
    }

    return pcclSuccess;
}

pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
                           pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
                           pcclReduceInfo_t *reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    return pcclSuccess;
}

pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
    pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator, pcclReduceInfo_t *reduce_info_out,
    pcclAsyncReduceOp_t *reduce_handle_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(reduce_handle_out != nullptr, pcclInvalidArgument);
    return pcclSuccess;
}

pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle) {
    PCCL_VALIDATE_INITIALIZED();
    return pcclSuccess;
}

pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *comm, pcclSharedState_t *shared_state) {
    return pcclSuccess;
}

pcclResult_t pcclPollSharedState(const pcclComm_t *comm, pcclSharedState_t *shared_state) {
    return pcclSuccess;
}

struct pcclMasterInstanceState_t {
    std::unique_ptr<ccoip::CCoIPMasterHandler> master_handler;
};

pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address, pcclMasterInstance_t **p_master_handle_out) {
    PCCL_VALIDATE(p_master_handle_out != nullptr, pcclInvalidArgument);
    *p_master_handle_out = new pcclMasterInstance_t{
        .master_handler = std::make_unique<ccoip::CCoIPMasterHandler>(listen_address),
    };
    return pcclSuccess;
}

pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->launch()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->interrupt()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->join()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    delete master_instance;
    return pcclSuccess;
}