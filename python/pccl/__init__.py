import faulthandler
from enum import Enum

from pccl._loader import load_native_module

# To debug Python to C FFI calls:
# $ cp examples/any.py tmp.py && gdb -ex r --args python3 tmp.py
# See also https://wiki.python.org/moin/DebuggingWithGdb

# Enable faulthandler for debugging
faulthandler.enable()
ffi, C = load_native_module()  # Load native module

class PCCLResult(Enum): # Keep in sync with pccl_status.h. TODO: auto-generate this enum
    """PCCL result codes."""
    SUCCESS = 0,
    NOT_INITIALIZED = 1,
    SYSTEM_ERROR = 2,
    INTERNAL_ERROR = 3,
    INVALID_ARGUMENT = 4,
    INVALID_USAGE = 5,
    REMOTE_ERROR = 6,
    IN_PROGRESS = 7,
    NUM_RESULTS = 8,
    MASTER_CONNECTION_FAILED = 9,
    RANK_CONNECTION_FAILED = 10,
    RANK_CONNECTION_LOST = 11,
    NO_SHARED_STATE_AVAILABLE = 12,

class PCCLError(Exception):
    """PCCL specific exception."""
    def __init__(self, result: PCCLResult):
        super().__init__(f'PCCL error: {result}')
        self.result = result

# Init PCCL
assert C.pcclInit() == PCCLResult.SUCCESS.value, 'Failed to initialize PCCL'
