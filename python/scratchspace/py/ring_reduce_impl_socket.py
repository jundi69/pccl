import numpy as np
import threading
import time
import socket

###############################################################################
# Globals
###############################################################################
world_size: int = 0

# Each rank will have a dict of TX sockets to other peers,
# and an RX dict of sockets accepted from other peers.
p2p_connections_tx = []
p2p_connections_rx = []

# We'll keep a global list of all sockets so we can clean them up easily.
all_sockets = []

# A global base port so each rank listens on base_port + rank.
BASE_PORT = 9000

# Global counters
bytes_sent = 0
bytes_received = 0

# A global lock to protect increments of bytes_sent / bytes_received
io_lock = threading.Lock()

###############################################################################
# Helper function: Close and clear all previously opened sockets
###############################################################################
def cleanup_sockets():
    """
    Closes all sockets from the previous run if any exist,
    and clears out the p2p connections tables. This helps
    avoid 'Address already in use' errors in repeated tests.
    """
    global p2p_connections_tx, p2p_connections_rx, all_sockets
    # Close everything
    for s in all_sockets:
        try:
            s.close()
        except:
            pass
    all_sockets.clear()

    p2p_connections_tx.clear()
    p2p_connections_rx.clear()

###############################################################################
# UTILITY: read exactly N bytes from a socket
###############################################################################
def _read_exact(sock: socket.socket, n: int) -> bytes:
    """
    Repeatedly call sock.recv until exactly n bytes have been received.
    """
    buf = bytearray(n)
    pos = 0
    while pos < n:
        chunk = sock.recv(n - pos)
        if not chunk:
            raise RuntimeError("Socket closed prematurely while trying to read data.")
        buf[pos:pos+len(chunk)] = chunk
        pos += len(chunk)
    return bytes(buf)

###############################################################################
# PART 0: Initialization / Socket Management
###############################################################################
def init_mailboxes(new_world_size: int):
    """
    Initialize a socket-based P2P connection grid for the given `new_world_size`.
    We preserve this API name so existing tests continue to call it.

    Steps:
      1) Clean up sockets from any prior run.
      2) Create a listening socket on (localhost, BASE_PORT + rank) for each rank.
      3) Connect from each rank to every other rank’s listening socket (TX side).
      4) Accept incoming connections on each rank’s listening socket (RX side).
    We also re-init the global counters, etc.
    """
    global world_size
    global bytes_sent, bytes_received
    global p2p_connections_tx, p2p_connections_rx, all_sockets

    # 1) Cleanup anything from a prior init, to avoid TIME_WAIT and binding issues
    cleanup_sockets()

    world_size = new_world_size

    # Reset global counters
    with io_lock:
        bytes_sent = 0
        bytes_received = 0

    # Prepare the TX/RX connection tables. Each entry is a dict keyed by peer rank -> socket
    p2p_connections_tx = [dict() for _ in range(world_size)]
    p2p_connections_rx = [dict() for _ in range(world_size)]

    # 2) Create server sockets for each rank
    server_sockets = []
    for rank in range(world_size):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("localhost", BASE_PORT + rank))
        s.listen(world_size - 1)
        server_sockets.append(s)
        all_sockets.append(s)  # keep track of it globally

    # 3) Connect from each rank to every other rank (TX side).
    # For each rank, we create (world_size - 1) client sockets.
    for rank in range(world_size):
        for peer in range(world_size):
            if peer == rank:
                continue
            client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # also set REUSEADDR on the client side to be safe
            client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            client_sock.connect(("localhost", BASE_PORT + peer))
            all_sockets.append(client_sock)

            # Send a small "hello" so peer knows who we are
            client_sock.sendall(rank.to_bytes(4, "little"))

            # Store in TX table
            p2p_connections_tx[rank][peer] = client_sock

    # 4) Accept incoming connections on each rank’s listening socket (RX side).
    # Each rank expects exactly (world_size - 1) inbound connections.
    for rank in range(world_size):
        for _ in range(world_size - 1):
            conn, _ = server_sockets[rank].accept()
            all_sockets.append(conn)
            # Read the remote rank from the "hello" message
            remote_rank_bytes = _read_exact(conn, 4)
            remote_rank = int.from_bytes(remote_rank_bytes, "little")

            # Store in RX table
            p2p_connections_rx[rank][remote_rank] = conn

    # Close the server sockets now that we've accepted everything
    for s in server_sockets:
        s.close()
        # remove them from the global list so we don't try to re-close
        all_sockets.remove(s)

###############################################################################
# RAW BYTE SEND/RECV (Socket-like) using p2p_connections_{tx,rx}
###############################################################################
def send_bytes(src, dst, data: bytes):
    """
    Send raw bytes over the TX socket from `src` to `dst`.
    """
    global bytes_sent
    sock = p2p_connections_tx[src][dst]

    sock.sendall(data)

    with io_lock:
        bytes_sent += len(data)


def recv_bytes(dst, src, n: int) -> bytes:
    """
    Receive exactly `n` bytes from the RX socket for `dst` from peer `src`.
    Blocks until n bytes are read.
    """
    global bytes_received
    sock = p2p_connections_rx[dst][src]

    chunk = _read_exact(sock, n)

    with io_lock:
        bytes_received += len(chunk)
    return chunk

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

DTYPE = np.float64

###############################################################################
# PHASE 1: Ring Reduce-Scatter
###############################################################################
def ring_reduce_scatter(rank, local_data, boundaries):
    """
    Perform the ring reduce-scatter in (P-1) pipeline steps.
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

        # Convert to DTYPE and send as raw bytes
        send_bytes(rank, next_rank, arr_to_send.astype(DTYPE, copy=False).tobytes())

        # Receive from prev_rank
        prev_rank = (rank - 1) % world_size
        prev_chunk_idx = (prev_rank - step) % world_size

        s_j, e_j = boundaries[prev_chunk_idx]
        chunk_size = e_j - s_j
        nbytes = chunk_size * np.dtype(DTYPE).itemsize
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
    """
    # We assume chunk_val has exactly 1 entry initially
    owned_items = list(chunk_val.items())
    assert len(owned_items) == 1, "Should have exactly 1 chunk after reduce-scatter"
    current_chunk_idx, current_chunk_arr = owned_items[0]

    for step in range(world_size - 1):
        next_rank = (rank + 1) % world_size
        prev_rank = (rank - 1) % world_size

        # Send the chunk we currently have
        send_bytes(rank, next_rank, current_chunk_arr.astype(DTYPE, copy=False).tobytes())

        # Which chunk are we receiving from prev_rank now?
        inc_idx = (prev_rank + 1 - step) % world_size
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

    # Initialize "mailboxes", which is now actually our socket layer:
    init_mailboxes(test_world_size)

    length = 3
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