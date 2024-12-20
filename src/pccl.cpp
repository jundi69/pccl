#include "pccl.h"
#include "pccl_internal.hpp"
#include <optional>
#include <ccoip_master.hpp>

static constinit bool pccl_initialized = false;

#define PCCL_VALIDATE_INITIALIZED() \
    if (!pccl_initialized) { \
        return (pcclNotInitialized); \
    }

static pcclResult_t internalPcclInit() {
    return pcclSuccess;
}

size_t pcclDataTypeSize(const pcclDataType_t datatype) {
    switch (datatype) {
    case pcclUint8:
    case pcclInt8:
        return 1;
    case pcclUint16:
        return 2;
    case pcclUint32:
    case pcclInt32:
    case pcclFloat:
        return 4;
    case pcclUint64:
    case pcclInt64:
    case pcclDouble:
        return 8;
    default:
        return 0;
    }
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
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(p_attribute_out != nullptr, pcclInvalidArgument);

    switch (attribute) {
        case PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE: {
            // const size_t world_size = communicator->ccoip_handler->get_world_size();
            // *p_attribute_out = static_cast<int>(world_size);
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

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    if (communicator->ccoip_client != nullptr) {
        if (!communicator->ccoip_client->interrupt()) [[unlikely]] {
            return pcclInvalidUsage;
        }
        if (!communicator->ccoip_client->join()) [[unlikely]] {
            return pcclInvalidUsage;
        }
    }
    delete communicator;
    return pcclSuccess;
}

pcclResult_t pcclConnect(pcclComm_t *communicator, ccoip_socket_address_t socket_address) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client == nullptr, pcclInvalidUsage);
    communicator->ccoip_client = std::make_unique<ccoip::CCoIPClient>(socket_address);
    if (!communicator->ccoip_client->connect()) [[unlikely]] {
        if (!communicator->ccoip_client->interrupt()) [[unlikely]] {
            return pcclInternalError;
        }
        if (!communicator->ccoip_client->join()) [[unlikely]] {
            return pcclInternalError;
        }
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclUpdateTopology(pcclComm_t *communicator) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    // accept new peers; this will block until we have a valid connection to each peer
    if (!communicator->ccoip_client->acceptNewPeers()) [[unlikely]] {
        return pcclInvalidUsage;
    }

    // update the topology
    if (!communicator->ccoip_client->updateTopology()) [[unlikely]] {
        return pcclInvalidUsage;
    }

    return pcclSuccess;
}

pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
                                pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
                                pcclReduceInfo_t *reduce_info_out,
                                pcclAsyncReduceOp_t *reduce_handle_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    PCCL_VALIDATE(reduce_handle_out != nullptr, pcclInvalidArgument);
    return pcclSuccess;
}

pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
                           pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
                           pcclReduceInfo_t *reduce_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);
    pcclAsyncReduceOp_t reduce_handle{};
    pcclAllReduceAsync(sendbuff, recvbuff, count, datatype, op, tag, communicator, reduce_info_out, &reduce_handle);
    pcclAwaitAsyncReduce(&reduce_handle);
    return pcclSuccess;
}

pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle) {
    PCCL_VALIDATE_INITIALIZED();
    return pcclSuccess;
}

static std::optional<ccoip_data_type_t> getCCoIPDataType(const pcclDataType_t datatype) {
    switch (datatype) {
        case pcclUint8: return ccoipUint8;
        case pcclUint16: return ccoipUint16;
        case pcclUint32: return ccoipUint32;
        case pcclUint64: return ccoipUint64;
        case pcclInt8: return ccoipInt8;
        case pcclInt16: return ccoipInt16;
        case pcclInt32: return ccoipInt32;
        case pcclInt64: return ccoipInt64;
        case pcclFloat: return ccoipFloat;
        case pcclDouble: return ccoipDouble;
    }
    return std::nullopt;
}

pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *communicator, pcclSharedState_t *shared_state,
                                        pcclSharedStateSyncInfo_t *sync_info_out) {
    PCCL_VALIDATE_INITIALIZED();
    PCCL_VALIDATE(communicator != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(communicator->ccoip_client != nullptr, pcclInvalidUsage);

    // sync shared state
    ccoip_shared_state_t shared_state_internal{};
    shared_state_internal.revision = shared_state->revision;
    for (size_t i = 0; i < shared_state->count; ++i) {
        const pcclTensorInfo_t &entry = shared_state->infos[i];
        const size_t entry_bytes = entry.count * pcclDataTypeSize(entry.datatype);
        auto ccoip_data_type = getCCoIPDataType(entry.datatype);
        if (!ccoip_data_type) {
            return pcclInvalidArgument;
        }
        shared_state_internal.entries.push_back(ccoip_shared_state_entry_t{
            .key = entry.name,
            .data_type = *ccoip_data_type,
            .value = std::span(static_cast<std::byte *>(entry.data), entry_bytes),
            .allow_content_inequality = entry.allow_content_inequality
        });
    }
    ccoip_shared_state_sync_info_t info{};
    if (!communicator->ccoip_client->syncSharedState(shared_state_internal, info)) [[unlikely]] {
        return pcclInvalidUsage;
    }
    if (sync_info_out != nullptr) {
        *sync_info_out = pcclSharedStateSyncInfo_t{
            .tx_bytes = info.tx_bytes,
            .rx_bytes = info.rx_bytes,
        };
    }
    return pcclSuccess;
}

pcclResult_t pcclPollSharedState(const pcclComm_t *comm, pcclSharedState_t *shared_state) {
    return pcclSuccess;
}

struct pcclMasterInstanceState_t {
    std::unique_ptr<ccoip::CCoIPMaster> master_handler;
};

pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address, pcclMasterInstance_t **p_master_handle_out) {
    PCCL_VALIDATE(p_master_handle_out != nullptr, pcclInvalidArgument);
    *p_master_handle_out = new pcclMasterInstance_t{
        .master_handler = std::make_unique<ccoip::CCoIPMaster>(listen_address),
    };
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->launch()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->interrupt()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

// ReSharper disable once CppParameterMayBeConstPtrOrRef
pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    if (!master_instance->master_handler->join()) [[unlikely]] {
        return pcclInvalidUsage;
    }
    return pcclSuccess;
}

pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance) {
    PCCL_VALIDATE(master_instance != nullptr, pcclInvalidArgument);
    PCCL_VALIDATE(master_instance->master_handler != nullptr, pcclInvalidArgument);
    master_instance->master_handler = nullptr;
    delete master_instance;
    return pcclSuccess;
}
