# Introduction
## What is PCCL?

The Prime Collective Communications Library (PCCL) is a fault-tolerant collective-communications framework designed to run over standard TCP/IP networks—including wide-area and internet-scale environments. It facilitates distributed training of larger models by letting multiple peers dynamically join or leave a training run at any point, with minimal risk of bringing the entire process to a halt.

Unlike classic HPC-focused MPI libraries / vendor-specific solutions like *NCCL*, PCCL does not require all processes to start together in a tightly coupled, homogeneous cluster. Instead, it employs a *central master* node—a lightweight orchestrator—that tracks which peers are currently part of the run and coordinates the collective operations (e.g. All-Reduce, model state synchronization). This architecture allows: