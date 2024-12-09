import sys
from pathlib import Path
from _cdecls import __PCCL_CDECLS

PCCL_LIBS = [
    ('win32', 'pccl.dll'),
    ('linux', 'libpccl.so'),
    ('darwin', 'libpccl.dylib'),
]

def load_native_module():
    platform = sys.platform
    lib_name = next((lib for os, lib in PCCL_LIBS if platform.startswith(os)), None)
    assert lib_name, f'Unsupported platform: {platform}'

    # Locate the library in the package directory
    #pkg_path = Path(__file__).parent
    pkg_path = Path('../../bin/debug')
    lib_path = pkg_path / lib_name
    assert lib_path.exists(), f'PCCL shared library not found: {lib_path}'

    # Load the library using cffi
    from cffi import FFI
    ffi = FFI()
    ffi.dlopen('m')  # Math library
    ffi.cdef(__PCCL_CDECLS)  # Define the C declarations
    lib = ffi.dlopen(str(lib_path))  # Load the shared library
    return ffi, lib