#pragma once

#include <ccoip_inet.h>

#ifdef __cplusplus
#include <cstddef>
#else
#include <stddef.h>
#include <stdbool.h>
#endif

#include "pccl_status.h"

#ifdef _MSC_VER
#define PCCL_EXPORT __declspec(dllexport)
#else
#define PCCL_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum pcclDataType_t {
    pcclUint8 = 0,
    pcclInt8 = 1,
    pcclUint16 = 2,
    pcclUint32 = 3,
    pcclInt32 = 4,
    pcclUint64 = 5,
    pcclInt64 = 6,
    pcclFloat = 7,
    pcclDouble = 8
} pcclDataType_t;

typedef enum pcclRedOp_t {
    pcclSum,
    pcclAvg,
    pcclProd,
    pcclMax,
    pcclMin
} pcclRedOp_t;

typedef enum pcclAttribute_t {
    PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE = 1
} pcclAttribute_t;

typedef struct pcclComm_t pcclComm_t;

typedef struct pcclRankInfo_t pcclRankInfo_t;

typedef struct pcclReduceInfo_t {
    uint32_t world_size;
    uint64_t tx_bytes;
    uint64_t rx_bytes;
} pcclReduceInfo_t;

typedef struct {
    pcclComm_t *comm;
    uint64_t tag;
} pcclAsyncReduceOp_t;

/**
 * Uniquely identifies a rank in a communicator.
 * Even if a rank ordinal is reused after a rank is removed and a new rank added in its place, the UUID will be unique.
 */
typedef struct pcclRankUuid_t {
    uint8_t data[16];
} pcclRankUuid_t;

typedef struct pcclTensorInfo_t {
    const char *name;
    void *data;
    size_t count;
    pcclDataType_t datatype;
    bool allow_content_inequality;
} pcclTensorInfo_t;

typedef struct pcclSharedState_t {
    uint64_t revision;
    size_t count;
    pcclTensorInfo_t *infos;
} pcclSharedState_t;

/**
 * Initializes the pccl library.
 * Must be called before pccl library functions are used.
 */
PCCL_EXPORT pcclResult_t pcclInit();

/**
 * Creates a new communicator.
 * @param comm_out Pointer to the communicator to be created.
 * @return @code pcclSuccess@endcode if the communicator was created successfully.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 */
PCCL_EXPORT pcclResult_t pcclCreateCommunicator(pcclComm_t **comm_out);

/**
 * Gets an attribute of a communicator.
 * @param communicator The communicator to get the attribute from.
 * @param attribute The attribute type to retrieve the value of.
 * @param p_attribute_out Pointer to the attribute value to be set.
 *
 * @return @code pcclSuccess@endcode if the attribute was successfully retrieved.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 * @return @code pcclInvalidArgument@endcode if the communicator or pAttributeOut is null or if the specified attribute type is unknown.
 */
PCCL_EXPORT pcclResult_t pcclGetAttribute(const pcclComm_t *communicator, pcclAttribute_t attribute,
                                          int *p_attribute_out);

/**
 * Saves the topology of a communicator to a dot file for visual inspection.
 *
 * @param communicator The communicator to save the topology of.
 * @param filename The name of the file to save the topology to.
 *
 * @return pcclSuccess if the topology was saved successfully.
 */
PCCL_EXPORT pcclResult_t pcclTopologySaveGraph(const pcclComm_t *communicator, const char *filename);

/**
 * Saves the reduce plan instructions of a communicator to a file for inspection.
 * @param communicator The communicator to save the reduce plan of.
 * @param filename The name of the file to save the reduce plan to.
 *
 * @return pcclSuccess if the reduce plan was saved successfully.
 */
PCCL_EXPORT pcclResult_t pcclSaveReducePlan(const pcclComm_t *communicator, const char *filename);

/**
 * Destroys a communicator.
 * Will block until the threads associated with the communicator have exited.
 *
 * @param communicator The communicator to destroy.
 *
 * @return pcclSuccess if the communicator was destroyed successfully.
 */
PCCL_EXPORT pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);

/**
 * Establishes a connection to a master node.
 * This function must be called on a communicator for the communicator to be usable.
 *
 * @param communicator The communicator to connect to the master node.
 * @param socket_address The address of the master node to connect to.
 *
 * @return @code pcclSuccess@endcode if the connection was established successfully.
 * @return @code pcclInvalidArgument@endcode if the communicator is null.
 * @return @code pcclInvalidUsage@endcode if the communicator is already connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclConnectMaster(pcclComm_t *communicator, ccoip_socket_address_t socket_address);


/**
 * Will block if new peers are joining the session and will handle connection establishment with them.
 *
 * @param communicator the communicator to accept new peers on.
 *
 * @return @code pcclSuccess@endcode if the new peers were accepted successfully.
 */
PCCL_EXPORT pcclResult_t pcclAcceptNewPeers(pcclComm_t *communicator);

/**
 * Performs an all reduce operation on a communicator. Blocks untill the all reduce is complete.
 *
 * @param sendbuff The buffer to send data from.
 * @param recvbuff The buffer to receive data into.
 * @param count The number of elements in the buffer.
 * @param datatype The data type of the elements in the buffer.
 * @param op The reduction operation to perform.
 * @param tag The tag to identify the operation.
 * @param communicator The communicator to perform the operation on.
 * @param reduce_info_out The reduce info to be filled with information about the operation.
 *
 * @return @code pcclSuccess@endcode if the all reduce operation was successful.
 * @return @code pcclInvalidArgument@endcode if the communicator, sendbuff, recvbuff, count is less or equal to zero, or tag is less than zero.
 * @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
 * @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
 */
PCCL_EXPORT pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
                                       pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
                                       pcclReduceInfo_t *reduce_info_out);

/**
* Performs an all reduce operation on a communicator. Async version of @code pcclAllReduce@endcode.
*
* @param sendbuff The buffer to send data from.
* @param recvbuff The buffer to receive data into.
* @param count The number of elements in the buffer.
* @param datatype The data type of the elements in the buffer.
* @param op The reduction operation to perform.
* @param tag The tag to identify the operation.
* @param communicator The communicator to perform the operation on.
* @param reduce_info_out The reduce info to be filled with information about the operation.
* @param reduce_handle_out The reduce op handle to be filled with an async handle to the operation.
*
* @return @code pcclSuccess@endcode if the all reduce operation was successful.
* @return @code pcclInvalidArgument@endcode if the communicator, sendbuff, recvbuff, count is less or equal to zero, or tag is less than zero.
* @return @code pcclNotInitialized@endcode if @code pcclInit@endcode has not been called yet.
* @return @code pcclInvalidUsage@endcode if the communicator is not connected to a master node.
* @return @code pcclInvalidArgument@endcode if the reduce handle output is null.
*/
PCCL_EXPORT pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
                                            pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
                                            pcclReduceInfo_t *reduce_info_out,
                                            pcclAsyncReduceOp_t *reduce_handle_out);

/**
 * Awaits the completion of an async reduce operation. Blocks until the operation is complete.
 *
 * @param reduce_handle The handle to the async reduce operation.
 *
 * @return @code pcclSuccess@endcode if the async reduce operation was successful.
 * @return @code pcclInvalidArgument@endcode if the reduce handle is null or invalid.
 */
PCCL_EXPORT pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle);

/**
 * Synchronizes the shared state between all peers that are currently accepted.
 * If the shared state revision of this peer is outdated, the shared state will be updated.
 * The function will not unblock until it is confirmed all peers have the same shared state revision.
 */
PCCL_EXPORT pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *comm,
                                                    pcclSharedState_t *shared_state);

typedef struct pcclMasterInstanceState_t pcclMasterInstanceState_t;

typedef struct pcclMasterInstance_t {
    pcclMasterInstanceState_t *state;
} pcclMasterInstance_t;

/**
 * Creates a master node handle.
 * @param listen_address The address to listen for incoming connections on.
 * @param p_master_handle_out Pointer to the master node handle to be created.
 * @return @code pcclSuccess@endcode if the master node handle was created successfully.
 */
PCCL_EXPORT pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
                                          pcclMasterInstance_t *p_master_handle_out);

/**
 * Runs a master node. This function is non-blocking.
 * @param master_instance The master node handle to run.
 * @return @code pcclSuccess@endcode if the master node was run successfully.
 * @return @code pcclInvalidArgument@endcode if the master handle is already running.
 */
PCCL_EXPORT pcclResult_t pcclRunMaster(pcclMasterInstance_t master_instance);

/**
 * Interrupts a master node.
 * @param master_instance The master node handle to interrupt.
 * @return @code pcclSuccess@endcode if the master node was interrupted successfully.
 */
PCCL_EXPORT pcclResult_t pcclInterruptMaster(pcclMasterInstance_t master_instance);

/**
 * Awaits termination of a master node. This function is blocking.
 * @param master_instance The master node handle to await termination of.
 * @return @code pcclSuccess@endcode if the master node was terminated successfully.
 * @return @code pcclInvalidArgument@endcode if the master handle is not running / was never interrupted.
 */
PCCL_EXPORT pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t master_instance);

/**
 * Destroys a master node. Must only be called after pcclMasterAwaitTermination has been called and returned.
 * @param master_instance The master node handle to destroy.
 * @return @code pcclSuccess@endcode if the master node was destroyed successfully.
 */
PCCL_EXPORT pcclResult_t pcclDestroyMaster(pcclMasterInstance_t master_instance);

#ifdef __cplusplus
}
#endif
