import numpy as np
import threading
import time

###############################################################################
# Standalone experiment for implementation a ring reduce.
# Socket layer is emulated with process-local mailboxes.
###############################################################################
mailboxes = {}
mailbox_lock = threading.Lock()

bytes_sent = 0
bytes_received = 0

world_size: int = 0

def init_mailboxes(new_world_size: int):
    """
    Initialize a byte-stream mailbox for each (src, dst) pair.
    Also reset global counters for bytes sent/received.
    """
    global world_size, mailboxes, bytes_sent, bytes_received

    world_size = new_world_size

    mailboxes = {}
    bytes_sent = 0
    bytes_received = 0

    for src in range(world_size):
        for dst in range(world_size):
            if src != dst:
                # Each mailbox is now a single bytearray
                mailboxes[(src, dst)] = bytearray()




###############################################################################
# RAW BYTE SEND/RECV (Socket-like)
###############################################################################
def send_bytes(src, dst, data: bytes):
    """
    Emulate writing raw bytes onto a TCP stream from src to dst.
    We simply append to the mailbox bytearray.
    """
    global bytes_sent
    with mailbox_lock:
        mailboxes[(src, dst)].extend(data)
        bytes_sent += len(data)


def recv_bytes(dst, src, n: int) -> bytes:
    """
    Emulate reading exactly `n` raw bytes from a TCP stream.
    We block until enough bytes are available in mailboxes[(src, dst)].
    """
    global bytes_received
    while True:
        with mailbox_lock:
            stream = mailboxes[(src, dst)]
            if len(stream) >= n:
                chunk = stream[:n]  # slice out n bytes
                del stream[:n]  # remove them from the front
                bytes_received += n
                return bytes(chunk)  # convert to an immutable bytes object
        time.sleep(0.0001)  # avoid busy‐waiting


###############################################################################
# PARTITION 1D ARRAY INTO world_size CHUNKS
###############################################################################
def compute_chunk_boundaries(length, world_size):
    """
    Partition a 1D array of 'length' into 'world_size' contiguous chunks.
    Returns list of (start, end) for each chunk i.
    """
    base = length // world_size
    remainder = length % world_size
    boundaries = []
    start = 0
    for i in range(world_size):
        size = base + (1 if i < remainder else 0)
        end = start + size
        boundaries.append((start, end))
        start = end
    return boundaries


# Data type for serialization
DTYPE = np.float64


###############################################################################
# PHASE 1: Ring Reduce-Scatter
###############################################################################
def ring_reduce_scatter(rank, local_data, boundaries):
    """
    Perform the ring reduce-scatter in (P-1) pipeline steps, writing raw bytes
    to the next rank and reading raw bytes from the previous rank.
    """
    chunk_val = {}
    has_added = {i: False for i in range(world_size)}

    # This rank "owns" chunk = rank initially
    s_i, e_i = boundaries[rank]
    local_slice = local_data[s_i:e_i]
    chunk_val[rank] = local_slice.copy()
    has_added[rank] = True

    for step in range(world_size - 1):
        # Determine which chunk to send
        chunk_to_send = (rank - step) % world_size
        next_rank = (rank + 1) % world_size

        # Send the chunk if we have it, else send an empty array
        if chunk_to_send in chunk_val:
            arr_to_send = chunk_val.pop(chunk_to_send)
        else:
            arr_to_send = np.array([], dtype=DTYPE)

        # Convert to DTYPE (no metadata) and send as raw bytes
        send_bytes(rank, next_rank, arr_to_send.astype(DTYPE, copy=False).tobytes())

        # Receive the next chunk from prev_rank
        prev_rank = (rank - 1) % world_size
        prev_chunk_idx = (prev_rank - step) % world_size

        # We know how many elements chunk `prev_chunk_idx` SHOULD have:
        #   chunk_size = boundaries[prev_chunk_idx][1] - boundaries[prev_chunk_idx][0]
        # If boundaries say 0, we read 0 bytes. Otherwise, we read exactly that many.
        if 0 <= prev_chunk_idx < world_size:
            s_j, e_j = boundaries[prev_chunk_idx]
            chunk_size = e_j - s_j
            nbytes = chunk_size * np.dtype(DTYPE).itemsize

            # Read exactly nbytes from the stream
            inc_bytes = recv_bytes(rank, prev_rank, nbytes)
            inc_arr = np.frombuffer(inc_bytes, dtype=DTYPE).copy()

            # Accumulate local data if we haven't yet
            if not has_added[prev_chunk_idx] and chunk_size > 0:
                inc_arr += local_data[s_j:e_j]
                has_added[prev_chunk_idx] = True

            chunk_val[prev_chunk_idx] = inc_arr

    return chunk_val


###############################################################################
# PHASE 2: Ring Allgather (Pipeline)
###############################################################################
def ring_allgather_pipeline(rank, chunk_val, boundaries):
    """
    Perform the ring allgather in (P-1) pipeline steps.
    We again send/receive only raw bytes, but know how many to expect
    by chunk boundaries.
    """
    # We assume chunk_val has exactly 1 entry initially.
    owned_items = list(chunk_val.items())
    assert len(owned_items) == 1, "Should have exactly 1 chunk after reduce-scatter"
    current_chunk_idx, current_chunk_arr = owned_items[0]

    for step in range(world_size - 1):
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        # Send the chunk we currently have
        send_bytes(rank,
                   next_rank,
                   current_chunk_arr.astype(DTYPE, copy=False).tobytes())

        # Receive from prev_rank
        # We figure out which chunk we are *now* getting:
        #   inc_idx = (prev_rank + 1 - step) % world_size
        inc_idx = (prev_rank + 1 - step) % world_size
        if 0 <= inc_idx < world_size:
            s_j, e_j = boundaries[inc_idx]
            chunk_size = e_j - s_j
            nbytes = chunk_size * np.dtype(DTYPE).itemsize
            inc_bytes = recv_bytes(rank, prev_rank, nbytes)
            inc_arr = np.frombuffer(inc_bytes, dtype=DTYPE).copy()

            # This becomes our new "current chunk"
            chunk_val[inc_idx] = inc_arr
            current_chunk_idx, current_chunk_arr = inc_idx, inc_arr

    return chunk_val


###############################################################################
# REASSEMBLE CHUNKS INTO FULL ARRAY
###############################################################################
def reassemble_chunks(chunk_val, boundaries, total_length):
    """
    Combine the distributed chunks back into a single array.
    """
    out = np.zeros(total_length, dtype=DTYPE)
    for i, arr in chunk_val.items():
        if 0 <= i < world_size:
            s_i, e_i = boundaries[i]
            out[s_i:e_i] = arr
    return out


###############################################################################
# COMPLETE RING ALLREDUCE
###############################################################################
def ring_allreduce(rank, local_data):
    length = len(local_data)
    boundaries = compute_chunk_boundaries(length, world_size)
    partials = ring_reduce_scatter(rank, local_data, boundaries)
    final_chunks = ring_allgather_pipeline(rank, partials, boundaries)
    return reassemble_chunks(final_chunks, boundaries, length)


###############################################################################
# THREAD WORKER
###############################################################################
def worker(rank, local_data, results):
    arr = ring_allreduce(rank, local_data)
    results[rank] = arr
    print(f"[Rank {rank}] final out:", arr)


def main():
    test_world_size = 3

    init_mailboxes(test_world_size)

    length = 2
    np.random.seed(42)

    p0_data = np.random.randn(length)
    p1_data = np.random.randn(length)
    p2_data = np.random.randn(length)

    all_data = np.stack([p0_data, p1_data, p2_data], axis=0)
    expected = np.sum(all_data, axis=0)
    print("Expected sum:", expected)

    results = [np.array([])] * test_world_size
    threads = []

    def launch_th(rank: int):
        worker(rank, all_data[rank], results)

    for r in range(test_world_size):
        t = threading.Thread(target=launch_th, args=(r,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Check correctness
    for r in range(test_world_size):
        diff = results[r] - expected
        print(f"Rank {r}: out={results[r]}, diff={diff}")
        assert np.allclose(results[r], expected), f"Rank {r} mismatch!"

    print("SUCCESS: All ranks match the expected sum.")
    print(f"Bytes sent={bytes_sent}, Bytes received={bytes_received}")


if __name__ == "__main__":
    main()
