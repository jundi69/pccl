# Introduction to DiLoCo

DiLoCo is an optimization scheme which requires drastically less frequent communication
than naive DDP.
While PCCL is a generalized collective communications library and can thus implement many other optimization schemes,
it was designed with the unique requirements of DiLoCo in mind.

PCCL can implement both synchronous and "streaming" (asynchronous) DiLoCo utilizing PCCL's
async collective primitives paired with its shared state system.