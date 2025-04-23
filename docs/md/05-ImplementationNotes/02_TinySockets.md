# TinySockets
PCCL relies on standard TCP sockets for:
- *Master* connections (long-lived to the orchestrator)
- *Peer-to-Peer* ring connections (one or two per ring neighbor)
- *Shared-State Distribution* “one-off” ephemeral connections (similar to HTTP GET/POST style transfers)
- *Bandwidth Benchmark* connections (short-lived, used to measure throughput between pairs)

## Queued Socket Mechanism
For the master connection—and potentially any socket that might carry messages for multiple “logical” operations—PCCL uses a queued socket approach. That is, we maintain an internal queue of incoming packets, and let each part of the library read only the packets intended for it by consuming only packets that match a particular predicate. This helps avoid concurrency issues where multiple threads might accidentally consume each other’s data.

## Dedicated RX/TX Threads
PCCL uses dedicated RX/TX threads for sending and receiving concurrently.
Threads add read or write requests to a queue for a particular tag, and data will be sent or received on that threads behalf.
The RX thread will read from the socket and dispatch the data to the correct queue given the received tag.
The TX thread will send data from the queue to the socket while prepending the tag for distinction.
Waking up the TX thread is done via [threadpark](https://github.com/PrimeIntellect-ai/threadpark/tree/main), a custom-built lightweight thread parking library that utilizing futex-like apis on all major operating systems to facilitate efficient wakes.