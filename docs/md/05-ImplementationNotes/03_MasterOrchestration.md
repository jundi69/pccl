# Master Orchestration

The master runs a single-threaded event loop (currently using `libuv`) that listens for new clients, processes control packets (e.g. “vote to accept new peers,” “optimize topology,” etc.), and updates the central state machine in `CCoIPMasterState`.

## Single-Threaded, Deterministic State
Since the master is conceptually a big “authoritative state machine”. Each inbound request from a client (join, sync, etc.) triggers a state update or transition. Particular state transitions trigger confirmation packets to be sent to clients to proceed.

## Master Crash = End of Run
If the master node process crashes or is forcibly killed, the peer side eventually sees “lost connection” errors. There is no built-in “master re-election” or replication. The recommended approach is to:

1. Simply restart the master process
2. Have peers automatically reconnect to the new master (e.g. fisrt panic the application & relaunch via a script on unsucessful exit code)
3. Load from checkpoint to restore the shared state.

It should be noted that CCoIP itself is not designed to "retain" shared state, simply to distribute it while a run is ongoing. If a run does indeed crawl to a halt, the shared state is lost. Therefore, it is recommended that peers save their own shared state to disk periodically, and reload it on restart.
As PCCL guarantees bit-identical shared state among all peers at all times, it is expected that after peers load their saved shared state from disk and begin synchronizing the shared state again, that all previously connected peers will unanimously agree on the shared state hashes and continue from there.

However, it should be noted that a crash of the master process is very unlikely.

## Topology Optimization (Bandwidth Tests & TSP)
One of PCCL’s features is "bandwidth-aware ring ordering". Since ring-based reduce can be bottlenecked by the slowest link, it helps to measure peer-to-peer throughput and reorder accordingly.

1. **Bandwidth Store**: The master keeps an asymmetric cost matrix (`BandwidthStore`) of measured bandwidth from peer A to peer B.
2. **Benchmark Requests**: When a peer calls `pcclOptimizeTopology`, the master identifies missing edges (i.e., pairs not yet measured) and instructs the relevant peer(s) to do a quick TCP test.
3. **TSP Heuristic:** The master uses a traveling-salesman “shortest path” (or “highest bandwidth”) approach to find a ring ordering that tries to maximize total link speed. For small problems `world_size <= 16` an exact solution will be attempted in a set timeout limit, for larger problems it might attempt a simpler heuristic (path of immediate “closest” peer, random tour or ant colony optimization with 2 Opt & 3 Opt local search, etc.). If an optimal solution cannot be found, the master may start a “moonshot” approach in the background to either target an optimal solution for higher `world_size` or to continue improving the current solution heuristically.
   Once a better ring is found, p2p connections will be re-established in that order (without letting brand-new peers in) the next time the client calls `pcclUpdateTopology`. The clients adopt the new ring as soon as they collectively vote and connect to each new neighbor.
   Solutions that are found immediately as part of the topology optimization phase without a background “moonshot” are adopted immediately by the peers as part of `pcclOptimizeTopology` in a fashion similar to `pcclUpdateTopology`, but without admitting newcomers into the run while still “going through the same motions” of voting to establish p2p connections by peers, followed by distribution of the p2p connection information by the master to said peers, along with subsequent connection establishment performed by the peers followed by subsequent confirmation of the connection establishment to the master.

## Shared-State Hashing & Distribution

When you call `pcclSynchronizeSharedState`, each peer does:

1. **Hash Each Tensor**:
- On CPU: SimpleHash (with OpenMP optimizations, if supported by compiler) or a CRC32-based approach (with SSE/AVX/ARM Neon optimizations if available).
- On GPU: SimpleHash, a custom deterministic kernel if compiled with CUDA support.

2. **Report Revision & Hash**: The peer sends these to the master for that group’s “mask election.”
3. **Master Chooses a Mask**: By popularity, it decides which set of (keys, hashes) is canonical. Peers that deviate are assigned to fetch the updated data from a designated “correct” peer via ephemeral connections. Peers can withdraw from shared state content hash popularity election if they e.g. declare a shared state synchronization strategy such as `PCCL_SHARED_STATE_SYNC_STRATEGY_RECEIVE_ONLY`. If no peer puts forth its content for election, all peers of the peer group will be kicked upon final shared state synchronziation consensus.
4. **One-Increment Rule**:
   The master checks that each new `shared_state->revision` is exactly 1 higher than before. If not, you see a `REVISION_INCREMENT_VIOLATION`.
   If it were to ever happen that the last peer that distributed the shared state leaves the run, the shared state is effectively lost.
   Because new peers will have freshly initialized revision counters, which thus triggers a mismatch, no peer will ever be able to synchronize the shared state successfully again until the master is restarted.
   It is the responsibility of the application developer to ensure to periodically checkpoint the shared state.
   In the best case, newly joining peers load the most recent shared state from disk, and because the shared
   state revision matches the expected value, said peer will become shared state distributor, restoring the normal flow of operations.
   If the peer only periodically checkpoints shared state, the load may not result in restoring the exact last seen shared state revision.
   The master will of course not accept this revision, even though it might be the best available recovery choice.
   It is intended that in such a scenario the master must be restarted for purposes of safety and consistency of behavior.

5. Dirty Keys: If a peer’s local hash for “weight_1” mismatches the mask, the peer sets up a direct ephemeral TCP connection to the distributing peer to its shared state distribution socket. After transmission, the content is hashed again and compared to the expected value. If the hash matches, the peer proceeds. On hash mismatch, the call to `pcclSynchronizeSharedState` will return an error code.

In practice, if your model steps are *bitwise deterministic across* peers, the “dirty keys” scenario rarely happens. But it remains crucial for newly joining peers who need a full checkpoint or for accidental drift scenarios.
In the ideal case, the training code may even assert that no data is ever *received* during shared state synchronization after the first local iteration, where it may obtain the current popular shared state from the set of pre-existing peers post joining the run.

## Ring-Reduce Implementation
PCCL’s All-Reduce uses a pipeline ring approach:

1. `Reduce-Scatter`: `In world_size-1` steps, each peer’s chunk is successively passed around the ring and accumulated (e.g., sum).
2. `Reduce-Gather`: Another `world_size-1` steps to distribute final chunks so everyone ends up with the fully reduced array.

### Chunking & Quantization
- The library divides the buffer among peers, so each rank “owns” a slice.
- It optionally quantizes data if `pcclQuantMinMax` or others are selected. This can help reduce link usage on slower WAN connections. As quantization is also performed using optimized SIMD instructions, the overhead should be negligible for most WAN and even Ethernet connections.

## Firewalls & Ports
- By default, the master listens on port `48148`
- Each peer tries to bind to a small range ([`48149`..`48151`]) for p2p, shared-state distribution, and bandwidth test sockets.
  However, these ports are not defacto static as is the case with most network protocols. Rather, these ports are “bump allocated“ where initially the implementation tries to bind to the target port (e.g. `48151` for the benchmark socket, or `48149` for the shared state server), but if this fails, the next higher port is tried until a free port is found. This ensures multiple peers can run on the same machine without port conflicts. Peers will "find" each other by reporting their ports to the master, which will inturn share this information with other peers.

- `Important`: For wide-area or internet usage, you must open these ports in your firewall & forward them to your computer when behind NAT. When only hosting one peer per IP address, only opening port `48149`, `48150`, `48151` is required.
  When hosting more peers per IP address (e.g behind NAT), the recommended approach is to open a port range above `48148` proportional to the amount of peers using this IP address.