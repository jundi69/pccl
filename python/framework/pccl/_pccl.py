from typing import Union, Optional, Tuple
import time
from ipaddress import ip_address, IPv4Address, IPv6Address
from enum import Enum
from pccl._loader import load_native_module
import logging
import importlib


class ModuleDummy:
    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, name: str):
        raise RuntimeError(
            f"Module {self.name} is not available. Please install the module via pip if you want pccl to interoperate with it.")


# check if torch is available
if importlib.util.find_spec('torch') is not None:
    import torch
else:
    # dummy that fails on first access
    torch = ModuleDummy('torch')

# check if numpy is available
if importlib.util.find_spec('numpy') is not None:
    import numpy as np
else:
    # dummy that fails on first access
    np = ModuleDummy('numpy')

# To debug Python to C FFI calls:
# $ cp examples/any.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

# Enable faulthandler for debugging
PY_PCCL_DEBUG = False

if PY_PCCL_DEBUG:
    import faulthandler

    faulthandler.enable()

ffi, C = load_native_module()  # Load native module


class Result(Enum):  # Keep in sync with pccl_status.h.
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


class PCCLError(Exception):
    """PCCL specific exception."""

    def __init__(self, result: Result, func_name: str):
        super().__init__(f'{func_name} failed with error: {result.name}')
        self.result = result

    def __str__(self):
        return f'{super().__str__()}: {self.result.name}'

    @staticmethod
    def check(result: int, func_name: str):
        """Check the result and raise an exception if necessary."""
        if result != Result.SUCCESS.value:
            raise PCCLError(Result(result), func_name)


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
        dtype_map = {
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
        assert self in dtype_map, f'Unsupported DataType: {self}'
        return dtype_map[self]

    def to_numpy_dtype(self):
        """Converts a DataType to the corresponding Numpy dtype."""
        dtype_map = {
            np.dtypes.UInt8DType: DataType.UINT8,
            np.dtypes.Int8DType: DataType.INT8,
            np.dtypes.UInt16DType: DataType.UINT16,
            np.dtypes.UInt32DType: DataType.UINT32,
            np.dtypes.Int32DType: DataType.INT32,
            np.dtypes.UInt64DType: DataType.UINT64,
            np.dtypes.Int64DType: DataType.INT64,
            np.dtypes.Float32DType: DataType.FLOAT,
            np.dtypes.Float64DType: DataType.DOUBLE,
        }
        dtype_clazz = type(self)
        assert dtype_clazz in dtype_map, f'Unsupported dtype: {dtype_clazz}'
        # noinspection PyTypeChecker
        return dtype_map[dtype_clazz]

    @classmethod
    def from_torch_dtype(cls, dtype: 'torch.dtype'):
        """Converts a PyTorch dtype to the corresponding DataType."""
        dtype_map = {
            torch.uint8: DataType.UINT8,
            torch.int8: DataType.INT8,
            torch.uint16: DataType.UINT16,
            torch.uint32: DataType.UINT32,
            torch.int32: DataType.INT32,
            torch.uint64: DataType.UINT64,
            torch.int64: DataType.INT64,
            torch.float32: DataType.FLOAT,
            torch.float64: DataType.DOUBLE,
        }
        assert dtype in dtype_map, f'Unsupported dtype: {dtype}'
        return dtype_map[dtype]

    @classmethod
    def from_numpy_dtype(cls, dtype: 'np.dtype'):
        """Converts a Numpy dtype to the corresponding DataType."""
        dtype_map = {
            np.dtypes.UInt8DType: DataType.UINT8,
            np.dtypes.Int8DType: DataType.INT8,
            np.dtypes.UInt16DType: DataType.UINT16,
            np.dtypes.UInt32DType: DataType.UINT32,
            np.dtypes.Int32DType: DataType.INT32,
            np.dtypes.UInt64DType: DataType.UINT64,
            np.dtypes.Int64DType: DataType.INT64,
            np.dtypes.Float32DType: DataType.FLOAT,
            np.dtypes.Float64DType: DataType.DOUBLE,
        }
        dtype_clazz = type(dtype)
        assert dtype_clazz in dtype_map, f'Unsupported dtype: {dtype}'
        # noinspection PyTypeChecker
        return dtype_map[dtype_clazz]


class TensorInfo:
    def __init__(self, name: str, data_ptr: int, *, numel: int, dtype: DataType, allow_content_inequality: bool):
        self.name = name
        self.data_ptr = data_ptr
        self.numel = numel
        self.dtype = dtype
        self.allow_content_inequality = allow_content_inequality

    @classmethod
    def from_torch(cls, tensor: 'torch.Tensor', name: str, *, allow_content_inequality: bool = False):
        """Creates a TensorInfo from a PyTorch tensor."""
        assert tensor.is_contiguous(), 'Input tensor must be contiguous'
        assert tensor.device.type == 'cpu', 'Only CPU tensors are supported'
        numel: int = tensor.numel()
        data_ptr: int = tensor.data_ptr()
        dtype: DataType = DataType.from_torch_dtype(tensor.dtype)
        return cls(name, data_ptr, numel=numel, dtype=dtype, allow_content_inequality=allow_content_inequality)

    @classmethod
    def from_numpy(cls, tensor: 'np.ndarray', name: str, *, allow_content_inequality: bool = False):
        """Creates a TensorInfo from a Numpy tensor."""
        assert tensor.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        numel: int = tensor.size
        data_ptr: int = tensor.ctypes.data
        dtype: DataType = DataType.from_numpy_dtype(tensor.dtype)
        return cls(name, data_ptr, numel=numel, dtype=dtype, allow_content_inequality=allow_content_inequality)


class SharedState:
    def __init__(self, tensor_infos: list[TensorInfo]):
        assert tensor_infos, 'At least one tensor info must be provided'
        self._infos = ffi.new('pcclTensorInfo_t[]', len(tensor_infos))
        self._name_strings = []
        for i, info in enumerate(tensor_infos):
            name_bytes = info.name.encode('utf-8')
            name_cstr = ffi.new('char[]', name_bytes)
            self._infos[i].name = name_cstr

            # Keep a reference to prevent the string from being freed; tie it to lifetime of SharedState python object
            self._name_strings.append(name_cstr)

            self._infos[i].data = ffi.cast('void*', info.data_ptr)
            self._infos[i].count = ffi.cast('size_t', info.numel)
            self._infos[i].datatype = ffi.cast('pcclDataType_t', info.dtype.value)
            self._infos[i].allow_content_inequality = info.allow_content_inequality
        self._state = ffi.new('pcclSharedState_t*', {
            'revision': 0,
            'count': ffi.cast('size_t', len(tensor_infos)),
            'infos': self._infos,
        })

    @property
    def revision(self):
        return self._state[0].revision

    @revision.setter
    def revision(self, value: int):
        self._state[0].revision = value

    def push_revision(self):
        self._state[0].revision += 1


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


class SharedStateSyncInfo:
    def __init__(self, tx_bytes: int, rx_bytes: int):
        self.tx_bytes = tx_bytes
        self.rx_bytes = rx_bytes


class AsyncReduceHandle:
    def __init__(self, handle: ffi.CData):
        self._handle = handle
        self._info: Optional[Tuple[bool, int, ReduceInfo]] = None

    def wait(self) -> Tuple[bool, int, ReduceInfo]:
        """Awaits the completion of an async reduce operation. Blocks until the operation is complete."""

        if self._info is not None:  # guard native await function because it will fail on the second invocation
            return self._info

        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        status = C.pcclAwaitAsyncReduce(self._handle, info)
        is_success: bool = status == Result.SUCCESS.value
        self._info = (is_success, status, ReduceInfo(info.world_size, info.tx_bytes, info.rx_bytes))
        return self._info


# Init PCCL
PCCLError.check(C.pcclInit(), "pcclInit")

def _create_ccoip_socket_address(address: Union[IPv4Address, IPv6Address], port: int) -> ffi.CData:
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

    def __init__(self, address: str, peer_group: int = 0):
        assert ":" in address, f'Invalid address: {address}, expected format: ip:port'
        params: ffi.CData = ffi.new('pcclCommCreateParams_t*')
        ip, port = address.split(":")
        ip = ip_address(ip)
        params.master_address = _create_ccoip_socket_address(ip, int(port))[0]
        params.peer_group = ffi.cast('uint32_t', peer_group)
        self._comm = ffi.new('pcclComm_t**')
        PCCLError.check(C.pcclCreateCommunicator(params, self._comm), "pcclCreateCommunicator")

    def __del__(self):
        C.pcclDestroyCommunicator(self._comm[0])

    def get_attribute(self, attribute: Attribute) -> int:
        """Get a communicator attribute."""
        value = ffi.new('int*')
        PCCLError.check(C.pcclGetAttribute(self._comm[0], attribute.value, value), "pcclGetAttribute")
        return value[0]

    def save_topology_graph(self, filename: str):
        """Save the topology graph into a graphviz dot file for visualization."""
        assert filename and filename.endswith('.dot')
        PCCLError.check(C.pcclTopologySaveGraph(self._comm[0], bytes(filename, 'utf-8')), "pcclTopologySaveGraph")

    def save_reduce_plan(self, filename: str):
        """Save the reduce plan into a graphviz dot file for visualization."""
        assert filename and filename.endswith('.dot')
        PCCLError.check(C.pcclSaveReducePlan(self._comm[0], bytes(filename, 'utf-8')), "pcclSaveReducePlan")

    def connect(self, n_attempts: int = 5):
        """
        Establishes a connection to a master node.
        Performs the specified number of attempts with a one second sleep interval.
        This function must be called on a communicator for the communicator to be usable.
        """
        for attempt in range(1, n_attempts + 1):
            try:
                PCCLError.check(C.pcclConnect(self._comm[0]), "pcclConnect")
                logging.info(f"Connected to the master node")
                break
            except PCCLError as e:
                logging.error(f"Failed to connect to the master node (Attempt {attempt}/{n_attempts}): {e}")
                time.sleep(1)
        else:
            raise Exception("Failed to connect to the master node")

    def _all_reduce_n(self, sendbuff: ffi.CData, recvbuff: ffi.CData, *,
                      num_elements: int, dtype: DataType, op: ReduceOp,
                      tag: int = 0) -> ReduceInfo:
        info: ffi.CData = ffi.new('pcclReduceInfo_t*')
        PCCLError.check(
            C.pcclAllReduce(sendbuff, recvbuff, num_elements, dtype.value, op.value, tag, self._comm[0], info),
            "pcclAllReduce"
        )
        return ReduceInfo(info.world_size, info.tx_bytes, info.rx_bytes)

    def all_reduce(self, send: Union['torch.Tensor', 'np.ndarray'], recv: Union['torch.Tensor', 'np.ndarray'], *,
                   op: ReduceOp, tag: int = 0) -> ReduceInfo:
        """Performs an all reduce operation on a communicator. Blocks until the all reduce is complete."""
        if not isinstance(torch, ModuleDummy) and isinstance(send, torch.Tensor) and isinstance(recv, torch.Tensor):
            return self._all_reduce_pt(send, recv, op=op, tag=tag)
        elif not isinstance(np, ModuleDummy) and isinstance(send, np.ndarray) and isinstance(recv, np.ndarray):
            return self._all_reduce_np(send, recv, op=op, tag=tag)
        else:
            raise ValueError(
                f'Unsupported input types: {type(send)}, {type(recv)}; send and recv must either be both torch.Tensor or both np.ndarray')

    def _all_reduce_pt(self, send: 'torch.Tensor', recv: 'torch.Tensor', *, op: ReduceOp, tag: int = 0) -> ReduceInfo:
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'
        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        num_elements: int = recv.numel()
        dtype: DataType = DataType.from_torch_dtype(send.dtype)
        return self._all_reduce_n(sendbuff, recvbuff, num_elements=num_elements, dtype=dtype, op=op, tag=tag)

    def _all_reduce_np(self, send: 'np.ndarray', recv: 'np.ndarray', *, op: ReduceOp, tag: int = 0) -> ReduceInfo:
        assert send.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        assert recv.flags['C_CONTIGUOUS'], 'Output tensor must be contiguous'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.size == recv.size, 'Input and output tensors must have the same number of elements'
        sendbuff: ffi.CData = ffi.cast('void*', send.ctypes.data)
        recvbuff: ffi.CData = ffi.cast('void*', recv.ctypes.data)
        dtype: DataType = DataType.from_numpy_dtype(send.dtype)
        num_elements: int = recv.size
        return self._all_reduce_n(sendbuff, recvbuff, num_elements=num_elements, dtype=dtype, op=op, tag=tag)

    def all_reduce_async(self, send: Union['torch.Tensor', 'np.ndarray'], recv: Union['torch.Tensor', 'np.ndarray'], *,
                         op: ReduceOp, tag: int = 0) -> AsyncReduceHandle:
        """Performs an all reduce operation on a communicator. Async version of all_reduce."""
        if not isinstance(torch, ModuleDummy) and isinstance(send, torch.Tensor) and isinstance(recv, torch.Tensor):
            return self._all_reduce_async_pt(send, recv, op=op, tag=tag)
        elif not isinstance(np, ModuleDummy) and isinstance(send, np.ndarray) and isinstance(recv, np.ndarray):
            return self._all_reduce_async_np(send, recv, op=op, tag=tag)
        else:
            raise ValueError(
                f'Unsupported input types: {type(send)}, {type(recv)}; send and recv must either be both torch.Tensor or both np.ndarray')
        pass

    def _all_reduce_async_pt(self, send: 'torch.Tensor', recv: 'torch.Tensor', *, op: ReduceOp,
                             tag: int = 0) -> AsyncReduceHandle:
        assert send.is_contiguous(), 'Input tensor must be contiguous'
        assert recv.is_contiguous(), 'Output tensor must be contiguous'
        assert send.device == recv.device, 'Input and output tensors must be on the same device'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.device.type == 'cpu', 'Only CPU tensors are supported'
        assert send.numel() == recv.numel(), 'Input and output tensors must have the same number of elements'
        sendbuff: ffi.CData = ffi.cast('void*', send.data_ptr())
        recvbuff: ffi.CData = ffi.cast('void*', recv.data_ptr())
        dtype: DataType = DataType.from_torch_dtype(send.dtype)
        num_elements: int = recv.numel()
        return self._all_reduce_async_n(sendbuff, recvbuff, num_elements=num_elements, dtype=dtype, op=op, tag=tag)

    def _all_reduce_async_np(self, send: 'np.ndarray', recv: 'np.ndarray', *, op: ReduceOp,
                             tag: int = 0) -> AsyncReduceHandle:
        assert send.flags['C_CONTIGUOUS'], 'Input tensor must be contiguous'
        assert recv.flags['C_CONTIGUOUS'], 'Output tensor must be contiguous'
        assert send.dtype == recv.dtype, 'Input and output tensors must have the same dtype'
        assert send.size == recv.size, 'Input and output tensors must have the same number of elements'
        sendbuff: ffi.CData = ffi.cast('void*', send.ctypes.data)
        recvbuff: ffi.CData = ffi.cast('void*', recv.ctypes.data)
        dtype: DataType = DataType.from_numpy_dtype(send.dtype)
        num_elements: int = recv.size
        return self._all_reduce_async_n(sendbuff, recvbuff, num_elements=num_elements, dtype=dtype, op=op, tag=tag)

    def _all_reduce_async_n(self, sendbuff: ffi.CData, recvbuff: ffi.CData, *,
                            num_elements: int, dtype: DataType, op: ReduceOp,
                            tag: int = 0) -> AsyncReduceHandle:
        handle: ffi.CData = ffi.new('pcclAsyncReduceOp_t*')
        PCCLError.check(
            C.pcclAllReduceAsync(sendbuff, recvbuff, num_elements, dtype.value, op.value, tag, self._comm[0], handle),
            "pcclAllReduceAsync"
        )
        return AsyncReduceHandle(handle)

    def update_topology(self):
        """
        Update the topology of a communicator if required.
        Topology updates are required when new peers join, in which case @code pcclUpdateTopology@endcode will
        automatically handle connection establishment with the new peer(s).
        Topology updates can also be triggered by the master node in response to bandwidth changes or other events.
        This function will block until the topology update is complete.
        """
        PCCLError.check(C.pcclUpdateTopology(self._comm[0]), "pcclUpdateTopology")

    def sync_shared_state(self, shared_state: SharedState) -> SharedStateSyncInfo:
        """
        Synchronizes the shared state between all peers that are currently accepted.
        If the shared state revision of this peer is outdated, the shared state will be updated.
        The function will not unblock until it is confirmed all peers have the same shared state revision.
        """
        sync_info: ffi.CData = ffi.new('pcclSharedStateSyncInfo_t*')
        PCCLError.check(C.pcclSynchronizeSharedState(self._comm[0], shared_state._state, sync_info), "pcclSynchronizeSharedState")
        return SharedStateSyncInfo(sync_info.tx_bytes, sync_info.rx_bytes)


class MasterNode:
    def __init__(self, listen_address: str):
        assert ":" in listen_address, f'Invalid listen address: {listen_address}, expected format: ip:port'
        ip, port = listen_address.split(":")
        ip = ip_address(ip)
        ccoip_address = _create_ccoip_socket_address(ip, int(port))
        self._socket_address = ccoip_address
        self._master = ffi.new('pcclMasterInstance_t**')
        PCCLError.check(C.pcclCreateMaster(ccoip_address[0], self._master), "pcclCreateMaster")

    def run(self):
        """Runs a master node. This function is non-blocking."""
        PCCLError.check(C.pcclRunMaster(self._master[0]), "pcclRunMaster")

    def interrupt(self):
        """Interrupts a master node."""
        PCCLError.check(C.pcclInterruptMaster(self._master[0]), "pcclInterruptMaster")

    def __del__(self):
        PCCLError.check(C.pcclMasterAwaitTermination(self._master[0]), "pcclMasterAwaitTermination")
        PCCLError.check(C.pcclDestroyMaster(self._master[0]), "pcclDestroyMaster")
