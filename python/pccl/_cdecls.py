# Autogenered by /home/mario/Documents/projects/pccl-refactor/python/gen_bindings.py 2024-12-11 01:10:38.829175, do NOT edit!

__PCCL_CDECLS: str = '''

typedef enum ccoip_inet_protocol_t {
inetIPv4,
inetIPv6
} ccoip_inet_protocol_t;
typedef struct ccoip_ipv4_address_t {
uint8_t data[4];
} ccoip_ipv4_address_t;
typedef struct ccoip_ipv6_address_t {
uint8_t data[16];
} ccoip_ipv6_address_t;
typedef struct ccoip_inet_address_t {
ccoip_inet_protocol_t protocol;
ccoip_ipv4_address_t ipv4;
ccoip_ipv6_address_t ipv6;
} ccoip_inet_address_t;
typedef struct ccoip_socket_address_t {
ccoip_inet_address_t inet;
uint16_t port;
} ccoip_socket_address_t;
typedef enum pcclResult_t {
pcclSuccess = 0,
pcclNotInitialized = 1,
pcclSystemError = 2,
pcclInternalError = 3,
pcclInvalidArgument = 4,
pcclInvalidUsage = 5,
pcclRemoteError = 6,
pcclInProgress = 7,
pcclNumResults = 8,
pcclMasterConnectionFailed = 9,
pcclRankConnectionFailed = 10,
pcclRankConnectionLost = 11,
pcclNoSharedStateAvailable = 12,
} pcclResult_t;
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
typedef struct pcclAsyncReduceOp_t{
pcclComm_t *comm;
uint64_t tag;
} pcclAsyncReduceOp_t;
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
 pcclResult_t pcclInit();
 pcclResult_t pcclCreateCommunicator(pcclComm_t **comm_out);
 pcclResult_t pcclGetAttribute(const pcclComm_t *communicator, pcclAttribute_t attribute,
int *p_attribute_out);
 pcclResult_t pcclTopologySaveGraph(const pcclComm_t *communicator, const char *filename);
 pcclResult_t pcclSaveReducePlan(const pcclComm_t *communicator, const char *filename);
 pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);
 pcclResult_t pcclConnectMaster(pcclComm_t *communicator, ccoip_socket_address_t socket_address);
 pcclResult_t pcclUpdateTopology(pcclComm_t *communicator);
 pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
pcclReduceInfo_t *reduce_info_out);
 pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
pcclRedOp_t op, uint64_t tag, const pcclComm_t *communicator,
pcclReduceInfo_t *reduce_info_out,
pcclAsyncReduceOp_t *reduce_handle_out);
 pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle);
 pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *comm,
pcclSharedState_t *shared_state);
typedef struct pcclMasterInstanceState_t pcclMasterInstance_t;
 pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
pcclMasterInstance_t **p_master_handle_out);
 pcclResult_t pcclRunMaster(pcclMasterInstance_t *master_instance);
 pcclResult_t pcclInterruptMaster(pcclMasterInstance_t *master_instance);
 pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t *master_instance);
 pcclResult_t pcclDestroyMaster(pcclMasterInstance_t *master_instance);
'''

