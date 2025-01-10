from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pccl")
except PackageNotFoundError:
    __version__ = "dev"

from pccl._pccl import *
