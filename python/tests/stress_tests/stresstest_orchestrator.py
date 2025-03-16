import os
import sys
import time
import random
import threading
import subprocess
from typing import Dict, List, Optional

# If 5 minutes pass without a "Reduce completed" line, we fail.
MAX_IDLE_SECONDS = 5 * 60  # 5 minutes in seconds

def stream_output_to_stdout(proc: subprocess.Popen,
                            alive_dict: Dict[int, bool],
                            last_reduce_timestamp: List[float]):
    """
    Continuously read lines from `proc.stdout` and write them to sys.stdout.
    Once the first line is read, mark the process as alive in `alive_dict`.
    If a line contains "Reduce completed RX:", update `last_reduce_timestamp[0]`.
    """
    pid = proc.pid
    first_line_seen = False

    for line in proc.stdout:
        # Echo the line to our own stdout (or could store it, parse further, etc.)
        sys.stdout.write(f"[PEER {pid}] {line}")
        sys.stdout.flush()

        # If it contains our special marker, update the timestamp
        if "Reduce completed RX:" in line:
            last_reduce_timestamp[0] = time.time()

            if not first_line_seen:
                alive_dict[pid] = True
                first_line_seen = True


    # When the peer exits or the pipe closes:
    sys.stdout.write(f"[PEER {pid}] -- output stream closed.\n")
    sys.stdout.flush()

def launch_py_process(script_path: str,
                      args: List[str],
                      env_vars: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    """
    Launches a Python process with environment variables and captured stdout/stderr.
    """
    env = {**os.environ, **(env_vars or {})}
    cmd = [sys.executable, script_path] + args

    print(f"[LAUNCH] {cmd} with env={env_vars}")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,     # Return text (str) instead of bytes
        bufsize=1      # Line-buffered
    )

def display_large_text_in_textedit(message, font_size=84):
    """
    Displays the given message in macOS TextEdit in large text.

    :param message: The string to display
    :param font_size: The RTF 'font size' in half-points (e.g., 72 = 36pt font).
    """
    # Create basic RTF content with a specified font size
    # \fsNN sets the font size in half-points, so \fs72 = 36pt
    # \f0 picks the first font in the font table, which we define as Arial here
    rtf_content = rf"""{{
\rtf1\ansi\deff0
{{\fonttbl{{\f0 Arial;}}}}
\f0\fs{font_size} {message}
}}"""

    # Write the RTF file to a temporary location
    rtf_path = "/tmp/unit_test_fail.rtf"
    with open(rtf_path, "w") as f:
        f.write(rtf_content)

    # Launch TextEdit with the RTF file
    subprocess.run(["open", "-a", "TextEdit", rtf_path])

def run_stress_test(duration_hours: float = 8.0,
                    max_peers: int = 10,
                    spawn_script_name: str = "stresstest_peer.py",
                    master_script_name: str = "stresstest_master.py"):
    """
    Spawns a master process and repeatedly spawns/kills peers. If we go 5+ minutes
    without seeing a "Reduce completed RX: ..." line from any peer, we consider
    the test stuck and clean up early.
    """
    base_dir = os.path.dirname(__file__)
    peer_script_path = os.path.join(base_dir, spawn_script_name)
    master_script_path = os.path.join(base_dir, master_script_name)

    # Dictionary to indicate if a PID has printed anything yet
    alive_flags: Dict[int, bool] = {}

    # We keep track of the last time a "Reduce completed" line was seen:
    # Use a list of one float to allow modification from both main & threads.
    last_reduce_timestamp = [time.time()]

    # A list of active peer Popen objects
    peers: List[subprocess.Popen] = []

    # 1) Launch master node
    master_process = launch_py_process(
        master_script_path,
        args=[],
        env_vars={"PCCL_LOG_LEVEL": "DEBUG"}
    )
    print(f"[INFO] Master launched, PID={master_process.pid}")
    alive_flags[master_process.pid] = False

    # Start a thread to read & forward the master’s stdout
    master_thread = threading.Thread(
        target=stream_output_to_stdout,
        args=(master_process, alive_flags, last_reduce_timestamp),
        daemon=True
    )
    master_thread.start()

    # Helper to spawn a peer & start reading its stdout
    def spawn_peer():
        p = launch_py_process(
            peer_script_path,
            args=[],
            env_vars={"PCCL_LOG_LEVEL": "DEBUG"}
        )
        peers.append(p)
        alive_flags[p.pid] = False  # Not alive until first line
        print(f"[INFO] Spawned peer PID={p.pid}; waiting for first line...")

        th = threading.Thread(
            target=stream_output_to_stdout,
            args=(p, alive_flags, last_reduce_timestamp),
            daemon=True
        )
        th.start()
        return p

    # 2) Launch initial 2 peers
    for _ in range(2):
        spawn_peer()

    # 3) Main stress loop
    end_time = time.time() + duration_hours * 3600
    last_spawn_from_singleton: Optional[float] = None

    def get_num_alive_peers() -> int:
        """
        Return count of peers that are:
         1) still running (p.poll() is None),
         2) marked as alive in alive_flags.
        """
        alive_count = 0
        for p in peers:
            if p.poll() is None and alive_flags.get(p.pid, False):
                alive_count += 1
        return alive_count

    failure = False
    while True:
        now = time.time()
        if now >= end_time:
            print("[INFO] Time limit reached; stopping stress test.")
            break

        # Check if 5+ minutes have passed since last reduce
        if (now - last_reduce_timestamp[0]) > MAX_IDLE_SECONDS:
            display_large_text_in_textedit("[ERROR] No 'Reduce completed' lines for 5+ minutes. Considering test stuck!")
            failure = True
            break

        # Clean out peers that have exited
        still_running = []
        for p in peers:
            if p.poll() is None:
                still_running.append(p)
            else:
                print(f"[INFO] Peer PID={p.pid} ended (code={p.returncode}).")
        peers[:] = still_running

        num_alive = get_num_alive_peers()
        total_peers = len(peers)  # includes not-yet-alive ones

        # If we have 0 or 1 alive peers, spawn
        if num_alive <= 2:
            if total_peers < max_peers:
                new_peer = spawn_peer()
                if num_alive <= 2:
                    # We just went from 2 -> 3 (once the new peer prints). We'll check time
                    last_spawn_from_singleton = time.time()
        else:
            # We have >=3 alive peers => random action: spawn or kill
            action_candidates = []
            if total_peers < max_peers:
                action_candidates.append("spawn")
            if num_alive > 2:
                action_candidates.append("kill")

            if action_candidates:
                weights = [0.6 if a == "spawn" else 0.4 for a in action_candidates]
                chosen = random.choices(action_candidates, weights=weights, k=1)[0]

                if chosen == "spawn":
                    spawn_peer()
                    if num_alive <= 2:
                        last_spawn_from_singleton = time.time()
                else:
                    # chosen == "kill"
                    # Only kill if enough time after going from 1->2
                    if last_spawn_from_singleton is not None:
                        elapsed = time.time() - last_spawn_from_singleton
                    else:
                        elapsed = 999999.0

                    if elapsed > 1.0:
                        # Kill a random alive peer
                        alive_peers = [p for p in peers
                                       if p.poll() is None and alive_flags.get(p.pid, False)]
                        if len(alive_peers) > 1:
                            victim = random.choice(alive_peers)
                            print(f"[INFO] KILLING peer PID={victim.pid}")
                            victim.kill()
                            victim.wait()
                            print(f"[INFO] Confirmed peer PID={victim.pid} is terminated")
                        else:
                            # Not enough alive peers to kill
                            pass
                    else:
                        # Not safe to kill yet
                        pass

        # Sleep random time
        time.sleep(random.uniform(0.5, 2.0))

    # 4) Done or stuck => cleanup
    print("[INFO] Cleaning up peers...")
    for p in peers:
        if p.poll() is None:
            print(f"[INFO] Killing peer PID={p.pid}")
            p.kill()
            p.wait()

    print("[INFO] All peers terminated. Killing master process...")
    if master_process.poll() is None:
        master_process.kill()
        master_process.wait()

    print("[INFO] Master terminated. Stress test orchestrator finished.")

    if not failure:
        display_large_text_in_textedit("STRESS TEST COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_stress_test(duration_hours=8.0, max_peers=10)
