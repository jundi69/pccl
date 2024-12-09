# Autogenered by /home/mario/Documents/projects/pccl-refactor/python/gen_bindings.py 2024-12-09 02:46:09.604925, do NOT edit!

__PCCL_CDECLS: str = '''

typedef struct pcclComm_t pcclComm_t;
typedef struct pcclRankInfo_t pcclRankInfo_t;
typedef struct pcclReduceInfo_t pcclReduceInfo_t;
typedef struct { {;
typedef struct pcclRankUuid_t pcclRankUuid_t;
typedef struct pcclTensorInfo_t pcclTensorInfo_t;
typedef struct pcclSharedState_t pcclSharedState_t;
typedef struct pcclMasterInstanceState_t pcclMasterInstanceState_t;
typedef struct pcclMasterInstance_t pcclMasterInstance_t;
typedef struct ccoip_ipv4_address_t ccoip_ipv4_address_t;
typedef struct ccoip_ipv6_address_t ccoip_ipv6_address_t;
typedef struct ccoip_inet_address_t ccoip_inet_address_t;
typedef struct ccoip_socket_address_t ccoip_socket_address_t;

typedef int pcclDataType_t;
typedef int pcclRedOp_t;
typedef int pcclAttribute_t;
typedef int pcclResult_t;
typedef int ccoip_inet_protocol_t;

PCCL_EXPORT pcclResult_t pcclInit();
PCCL_EXPORT pcclResult_t pcclCreateCommunicator(pcclComm_t **comm_out);
PCCL_EXPORT pcclResult_t pcclGetAttribute(const pcclComm_t *communicator, pcclAttribute_t attribute,
PCCL_EXPORT pcclResult_t pcclTopologySaveGraph(const pcclComm_t *communicator, const char *filename);
PCCL_EXPORT pcclResult_t pcclSaveReducePlan(const pcclComm_t *communicator, const char *filename);
PCCL_EXPORT pcclResult_t pcclDestroyCommunicator(pcclComm_t *communicator);
PCCL_EXPORT pcclResult_t pcclConnectMaster(pcclComm_t *communicator, ccoip_socket_address_t socket_address);
PCCL_EXPORT pcclResult_t pcclAcceptNewPeers(pcclComm_t *communicator);
PCCL_EXPORT pcclResult_t pcclAllReduce(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
PCCL_EXPORT pcclResult_t pcclAllReduceAsync(const void *sendbuff, void *recvbuff, size_t count, pcclDataType_t datatype,
PCCL_EXPORT pcclResult_t pcclAwaitAsyncReduce(const pcclAsyncReduceOp_t *reduce_handle);
PCCL_EXPORT pcclResult_t pcclSynchronizeSharedState(const pcclComm_t *comm,
PCCL_EXPORT pcclResult_t pcclCreateMaster(ccoip_socket_address_t listen_address,
PCCL_EXPORT pcclResult_t pcclRunMaster(pcclMasterInstance_t master_instance);
PCCL_EXPORT pcclResult_t pcclInterruptMaster(pcclMasterInstance_t master_instance);
PCCL_EXPORT pcclResult_t pcclMasterAwaitTermination(pcclMasterInstance_t master_instance);
PCCL_EXPORT pcclResult_t pcclDestroyMaster(pcclMasterInstance_t master_instance);
'''

