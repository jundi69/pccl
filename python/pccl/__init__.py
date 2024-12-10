import faulthandler
from ipaddress import ip_address, IPv4Address, IPv6Address
from enum import Enum
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

def _create_ccoip_socket_address(address: IPv4Address | IPv6Address, port: int) -> ffi.CData:
    """Create a ccoip_socket_address_t."""
    socket_addr = ffi.new("ccoip_socket_address_t*")
    if isinstance(address, IPv4Address):
        socket_addr.inet.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv4)
        packed_ipv4 = address.packed
        for i, byte in enumerate(packed_ipv4):
            socket_addr.inet.address.ipv4.data[i] = byte & 255
    elif isinstance(address, IPv6Address):
        socket_addr.inet6.protocol = ffi.cast("ccoip_inet_protocol_t", C.inetIPv6)
        packed_ipv6 = address.packed
        for i, byte in enumerate(packed_ipv6):
            socket_addr.inet6.address.ipv6.data[i] = byte & 255
    else:
        raise ValueError(f'Unsupported IP address: {address}')
    socket_addr.port = port & 0xffff
    return socket_addr

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

    def await_termination(self):
        """Awaits termination of a master node. This function is blocking."""
        PCCLError.check(C.pcclMasterAwaitTermination(self._master[0]))

    def __del__(self):
        self.await_termination()
        PCCLError.check(C.pcclDestroyMaster(self._master[0]))

if __name__ == '__main__':
    c = Communicator()
    print(c.get_attribute(Attribute.CURRENT_WORLD_SIZE))
    m = MasterNode(listen_address='127.0.0.1:8080')
    m.run()
    m.interrupt()
    m.await_termination()
