from cffi import FFI
from enum import Enum

from __capi import __C_DECLS

ffi = FFI()
ffi.cdef(__C_DECLS)  # Define the C declarations
lib_path: str = '../../build/release/libpccl.dylib' # Replaced by python wheel and some platform check code
C = ffi.dlopen(str(lib_path))  # Load the shared library

def pccl_check(r):
    if r != C.pcclSuccess:
        raise RuntimeError(f'Error: {r}')


pccl_check(C.pcclInit())

class Attribute(Enum):
    WORLD_SIZE = C.PCCL_ATTRIBUTE_CURRENT_WORLD_SIZE

# create python oop wrapper
class Communicator:
    def __init__(self):
        self.comm = ffi.new('pcclComm_t**')
        pccl_check(C.pcclCreateCommunicator(self.comm))

    def get_attribute(self, attribute: Attribute) -> int:
        attr = ffi.new('int*')
        pccl_check(C.pcclGetAttribute(self.comm[0], attribute.value, attr))
        return attr[0]

    def __del__(self):
        pccl_check(C.pcclDestroyCommunicator(self.comm[0]))



## user uses api
comm = Communicator()
print(comm.get_attribute(Attribute.WORLD_SIZE))
assert  128 == comm.get_attribute(Attribute.WORLD_SIZE)
