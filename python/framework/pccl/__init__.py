from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pccl")
except PackageNotFoundError:
    __version__ = "dev"

from pccl._pccl import *

__all__ = ["Communicator", "MasterNode", "ReduceOp", "Attribute", "DataType", "DistributionHint",
           "QuantizationAlgorithm", "ReduceOperandDescriptor", "QuantizationOptions", "ReduceDescriptor",
           "TensorInfo", "SharedState", "SharedStateSyncInfo", "ReduceInfo", "AsyncReduceHandle", "PCCLError"]