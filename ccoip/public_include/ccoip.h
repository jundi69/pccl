#pragma once

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

/// Port that the master node is listening on for incoming connections.
/// Master node connections are persistent and are not closed until the client disconnects.
#define CCOIP_PROTOCOL_PORT_MASTER 48148

/// Port that each peer listens on for incoming connections shared state distribution requests
/// The connections for the shared state distribution are non-persistent and are closed after the shared state is distributed.
#define CCOIP_PROTOCOL_PORT_SHARED_STATE 48149

/// Port that each peer listens on for incoming connections from other peers
/// P2P connections are persistent and are not closed until the client disconnects or the master node
/// changes the topology such that the peer is no longer a neighbor.
/// Each peer connects to the listening port of the other peer, establishing its TX connection.
/// That peer also connects to the listening port of the first peer, establishing this peer's TX connection.
/// One peer's RX connection is the other peer's TX connection and vice versa.
/// This design is chosen for simplicity and to avoid the need for a central server to manage connections.
#define CCOIP_PROTOCOL_PORT_P2P 48150

/// Port that each peer listens on for incoming connections from other peers for bandwidth benchmarking.
/// The connections here are non-persistent and are closed after the benchmarking is complete.
/// Only one connection is accepted at each point in time to preserve the integrity of the benchmark.
#define CCOIP_PROTOCOL_PORT_BANDWIDTH_BENCHMARK 48151
