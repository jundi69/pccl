## Concurrency and Threading Model

### Master
- **Single Thread**: The master’s event loop is not multithreaded. This ensures consistent updates to `CCoIPMasterState`.

### Client

- **Blocking API**: By default, calls like `pcclAllReduce` or `pcclSynchronizeSharedState` block the caller until the operation finishes (or fails).
- **Async All-Reduce**: `pcclAllReduceAsync` spawns an internal thread or creates a logical execution plan and returns immediately. The user can poll for completion or wait on a condition variable. This concurrency model is useful for overlapping computation with communication, or for running multiple operations in parallel.
  This will be intended API design in future versions of PCCL.
- **Queued-Sockets:** If two internal threads might read from the same socket, PCCL enforces a queue mechanism to route matching packets to the correct consumer via predicate matching.

### Overall Rule: One Operation at a Time (Per Group)
Because the master enforces that the entire group do the same operation in lockstep, you rarely need your own concurrency around these calls.
The library expects you not to overlap, say, an All-Reduce with a shared-state sync in the same communicator - neither through means
of the native concurrency features implemented by PCCL (`pcclAllReduceAsync`, `pcclAwaitAsyncReduce`), nor through the use of concurrent threads.
The only exception to this rule is that it is allowed to launch multiple async collective communications without awaiting the respective previous handle - in other words, multiple async collective communications operations may be in flight at the same time.
Note however that even in this case the overarching "Operation" is "performing collective communications operations" and *only* performing collective communications operations.

### Threadsafe
PCCL is generally not threadsafe and should only ever be used from the main thread (except in case of stated exceptions).
PCCL will generally enforce that public facing apis are called on the main thread registered for the communicator.

#### There are some exceptions to this rule:
PCCL allows the user to launch and await multiple async collective operations from threads other than the main thread, as long as the main thread
does not call other user facing function on the same communicator in the meantime.
The async work ongoing on the other thread must be awaited before the main thread can call user facing api functions on the same communicator again.
The main thread may however call the `pcclAreNewPeersPending` function while collective communications operations are being scheduled and awaited by a different thread.
If `pcclAreNewPeersPending` returns true, the main thread should call `pcclUpdateTopology` to accept the new peers into the run.
Before doing so, the main thread should await the completion of the ongoing collective communication operations on the other thread.
However, the main thread should not call `pcclAwaitAsyncReduce` on handles created by the other thread. No guarantees are made about the behavior of awaiting handles created by different threads.
Instead, the main thread should await the completion of the work performed by the other thread which will itself call `pcclAwaitAsyncReduce` on the handles it created.