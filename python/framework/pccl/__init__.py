from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pccl")
except PackageNotFoundError:
    __version__ = "dev"

from pccl._pccl import *
import pccl._cuda as cuda

__all__ = ["cuda",
           "Communicator", "MasterNode", "ReduceOp", "Attribute", "DataType", "DistributionHint",
           "QuantizationAlgorithm", "ReduceOperandDescriptor", "QuantizationOptions", "ReduceDescriptor",
           "TensorInfo", "SharedState", "SharedStateSyncInfo", "ReduceInfo", "AsyncReduceHandle", "PCCLError"]
