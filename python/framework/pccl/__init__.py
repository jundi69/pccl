import faulthandler
import torch

from ipaddress import ip_address, IPv4Address, IPv6Address
from enum import Enum
from torch import Tensor
from pccl._loader import load_native_module

# To debug Python to C FFI calls:
# $ cp examples/any.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

# Enable faulthandler for debugging
faulthandler.enable()
ffi, C = load_native_module()  # Load native module

class Result(Enum): # Keep in sync with pccl_status.h.
    """PCCL result codes."""
    SUCCESS = C.pcclSuccess
    NOT_INITIALIZED = C.pcclNotInitialized
    SYSTEM_ERROR = C.pcclSystemError
    INTERNAL_ERROR = C.pcclInternalError
    INVALID_ARGUMENT = C.pcclInvalidArgument
    INVALID_USAGE = C.pcclInvalidUsage
    REMOTE_ERROR = C.pcclRemoteError
    IN_PROGRESS = C.pcclInProgress
    NUM_RESULTS = C.pcclNumResults
    MASTER_CONNECTION_FAILED = C.pcclMasterConnectionFailed
    RANK_CONNECTION_FAILED = C.pcclRankConnectionFailed
    RANK_CONNECTION_LOST = C.pcclRankConnectionLost
    NO_SHARED_STATE_AVAILABLE = C.pcclNoSharedStateAvailable

class DataType(Enum):
    """PCCL primitive data types."""
    UINT8 = C.pcclUint8
    INT8 = C.pcclInt8
    UINT16 = C.pcclUint16
    UINT32 = C.pcclUint32
    INT32 = C.pcclInt32
    UINT64 = C.pcclUint64
    INT64 = C.pcclInt64
    FLOAT = C.pcclFloat
    DOUBLE = C.pcclDouble

    def to_torch_dtype(self):
        """Converts a DataType to the corresponding PyTorch dtype."""
        mapping = {
            DataType.UINT8: torch.uint8,
            DataType.INT8: torch.int8,
            DataType.UINT16: torch.uint16,
            DataType.UINT32: torch.uint32,
            DataType.INT32: torch.int32,
            DataType.UINT64: torch.uint64,
            DataType.INT64: torch.int64,
            DataType.FLOAT: torch.float32,
            DataType.DOUBLE: torch.float64,
        }
        return mapping[self]

class ReduceOp(Enum):
    """PCCL reduction operations."""
    SUM = C.pcclSum
    AVG = C.pcclAvg
    PROD = C.pcclProd
    MAX = C.pcclMax
    MIN = C.pcclMin

class Attribute(Enum):
    """PCCL attributes."""
    CURRENT_WORLD_SIZE = C.PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE

class ReduceInfo:
    def __init__(self, world_size: int, tx_bytes: int, rx_bytes: int):
        self.world_size = world_size
        self.tx_bytes = tx_bytes
        self.rx_bytes = rx_bytes

class AsyncReduceHandle:
    def __init__(self, handle: ffi.CData):
        self._handle = handle

    def await_reduce(self):
        """Awaits the completion of an async reduce operation. Blocks until the operation is complete."""
        PCCLError.check(C.pcclAwaitAsyncReduce(self._handle))

class PCCLError(Exception):
    """PCCL specific exception."""
    def __init__(self, result: Result):
        super().__init__(f'PCCL error: {result}')
        self.result = result

    def __str__(self):
        return f'{super().__str__()}: {self.result.name}'

    @staticmethod
    def check(result: int):
        """Check the result and raise an exception if necessary."""
        if result != Result.SUCCESS.value:
            raise PCCLError(Result(result))

# Init PCCL
PCCLError.check(C.pcclInit())

def _create_ccoip_socket_address(address: IPv4Address | IPv6Address, port: int) -> ffi.CData:
    """Create a ccoip_socket_address_t."""
    socket_addr = ffi.new("ccoip_socket_address_t*")
    if isinstance(address, IPv4Address):
        socket_addr.inet.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv4)
        packed_ipv4 = address.packed
        for i, byte in enumerate(packed_ipv4):
            socket_addr.inet.ipv4.data[i] = byte & 255
    elif isinstance(address, IPv6Address):
        socket_addr.inet6.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv6)
        packed_ipv6 = address.packed
        for i, byte in enumerate(packed_ipv6):
            socket_addr.inet6.ipv6.data[i] = byte & 255
    else:
        raise ValueError(f'Unsupported IP address: {address}')
    socket_addr.port = port & 0xffff
    return socket_addr

class Communicator:
    """PCCL communicator."""

    def __init__(self):
        self._comm = ffi.new('pcclComm_t**')
        PCCLError.check(C.pcclCreateCommunicator(self._comm))

    def __del__(self):
        C.pcclDestroyCommunicator(self._comm[0])

    def get_attribute(self, attribute: Attribute) -> int:
        """Get a communicator attribute."""
        value = ffi.new('int*')
        PCCLError.check(C.pcclGetAttribute(self._comm[0], attribute.value, value))
        return value[0]

    def save_topology_graph(self, filename: str):
        """Save the topology graph into a graphviz dot file for visualization."""
        assert filename and filename.endswith('.dot')
        PCCLError.check(C.pcclTopologySaveGraph(self._comm[0], bytes(filename, 'utf-8')))

    def save_reduce_plan(self, filename: str):
        """Save the reduce plan into a graphviz dot file for visualization."""
        assert filename and filename.endswith('.dot')
        PCCLError.check(C.pcclSaveReducePlan(self._comm[0], bytes(filename, 'utf-8')))

    def connect_master(self, address: str):
        """
        Establishes a connection to a master node.
        This function must be called on a communicator for the communicator to be usable.
        """
        assert ":" in address, f'Invalid address: {address}, expected format: ip:port'
        ip, port = address.split(":")
        ip = ip_address(ip)
        ccoip_address = _create_ccoip_socket_address(ip, int(port))
        PCCLError.check(C.pcclConnectMaster(self._comm[0], ccoip_address[0]))

    def all_reduce(self, send: Tensor, recv: Tensor, dtype: DataType, op: ReduceOp, tag: int) -> ReduceInfo:
        """Performs an all reduce operation on a communicator. Blocks until the all reduce is complete."""
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.dtype == dtype.to_torch_dtype(), 'Input and output tensors must have the same dtype'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        numel: int = send.numel()
        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        PCCLError.check(C.pcclAllReduce(sendbuff, recvbuff, numel, dtype.value, op.value, tag, self._comm[0], info))
        return ReduceInfo(info.world_size, info.tx_bytes, info.rx_bytes)


    def all_reduce_async(self, send: Tensor, recv: Tensor, dtype: DataType, op: ReduceOp, tag: int) -> (ReduceInfo, AsyncReduceHandle):
        """Performs an all reduce operation on a communicator. Async version of all_reduce."""
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.dtype == dtype.to_torch_dtype(), 'Input and output tensors must have the same dtype'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        numel: int = send.numel()
        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        handle: ffi.CData = ffi.new('pcclAsyncReduceOp_t*')
        PCCLError.check(C.pcclAllReduceAsync(sendbuff, recvbuff, numel, dtype.value, op.value, tag, self._comm[0], info, handle))
        return ReduceInfo(info.world_size, info.tx_bytes, info.rx_bytes), AsyncReduceHandle(handle)

    def update_topology(self):
        """
        Update the topology of a communicator if required.
        Topology updates are required when new peers join, in which case @code pcclUpdateTopology@endcode will
        automatically handle connection establishment with the new peer(s).
        Topology updates can also be triggered by the master node in response to bandwidth changes or other events.
        This function will block until the topology update is complete.
        """
        PCCLError.check(C.pcclUpdateTopology(self._comm[0]))

    def sync_shared_state(self):
        """Awaits the completion of an async reduce operation. Blocks until the operation is complete."""
        PCCLError.check(C.pcclAwaitAsyncReduce(self._comm[0]))

class MasterNode:
    def __init__(self, listen_address: str):
        assert ":" in listen_address, f'Invalid listen address: {listen_address}, expected format: ip:port'
        ip, port = listen_address.split(":")
        ip = ip_address(ip)
        ccoip_address = _create_ccoip_socket_address(ip, int(port))
        self._socket_address = ccoip_address
        self._master = ffi.new('pcclMasterInstance_t**')
        PCCLError.check(C.pcclCreateMaster(ccoip_address[0], self._master))

    def run(self):
        """Runs a master node. This function is non-blocking."""
        PCCLError.check(C.pcclRunMaster(self._master[0]))

    def interrupt(self):
        """Interrupts a master node."""
        PCCLError.check(C.pcclInterruptMaster(self._master[0]))

    def __del__(self):
        PCCLError.check(C.pcclMasterAwaitTermination(self._master[0]))
        PCCLError.check(C.pcclDestroyMaster(self._master[0]))
