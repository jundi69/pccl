import os
import time
import random
import subprocess
import sys
from typing import List, Optional, Dict

###############################################################################
# Helper function to launch a single Python process.
###############################################################################
def launch_py_process(
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    """
    Launches a Python process with optional environment variables and stdout forwarding.

    :param script_path: Path to the Python script to execute.
    :param args: List of arguments to pass to the script.
    :param env_vars: Dictionary of environment variables to set for the process.
    :return: A Popen object for the launched process.
    """
    env = {**dict(os.environ), **(env_vars or {})}
    cmd = [sys.executable, script_path] + args
    print(f"[LAUNCH] {cmd} with env={env_vars}")
    return subprocess.Popen(cmd, env=env)

###############################################################################
# The main long-running stress test orchestrator.
###############################################################################
def run_stress_test(
        duration_hours: float = 8.0,
        max_peers: int = 10,
        spawn_script_name: str = "stresstest_peer.py",
        master_script_name: str = "stresstest_master.py"
):
    """
    Spawns a master process and repeatedly spawns/kills peers to provoke
    potential timing issues for a dynamic membership collective comms library.

    :param duration_hours: How many hours to keep running the stress test.
    :param max_peers: Maximum number of peers to allow at once.
    :param spawn_script_name: Name/path of the peer script to launch.
    :param master_script_name: Name/path of the master script.
    """
    # Resolve script paths relative to this file's location.
    base_dir = os.path.dirname(__file__)
    peer_script_path = os.path.join(base_dir, spawn_script_name)
    master_script_path = os.path.join(base_dir, master_script_name)

    # 1) Launch a single master node.
    master_process = launch_py_process(
        master_script_path,
        args=[],
        env_vars={"PCCL_LOG_LEVEL": "DEBUG"}
    )
    print(f"[INFO] Master launched with PID={master_process.pid}")

    # 2) Launch an initial set of peers (start with 2).
    peers = []
    for rank in range(2):
        p = launch_py_process(
            peer_script_path,
            args=[],
            env_vars={
                "PCCL_LOG_LEVEL": "DEBUG",
                "RANK": str(rank)
            }
        )
        peers.append(p)
        print(f"[INFO] Peer {rank} launched with PID={p.pid}")

    print("[INFO] Initial peers launched.")

    # Tracking time
    end_time = time.time() + duration_hours * 3600
    last_spawn_from_singleton: Optional[float] = None

    # 3) Enter main loop for random spawn/kill operations.
    while True:
        now = time.time()
        if now >= end_time:
            break  # Stop after the allotted duration.

        # Decide what to do: spawn a new peer or kill an existing peer?
        # But honor constraints:
        #   - If we have only 1 peer, we must spawn a new one (cannot kill).
        #   - If we just spawned a peer from 1→2, wait at least 1 sec before killing.

        num_peers = len(peers)

        # If we have exactly 1 peer, we either do nothing or spawn a new peer
        # to avoid killing the last peer.
        if num_peers == 1:
            if num_peers < max_peers:
                # Spawn a new peer
                p = launch_py_process(
                    peer_script_path,
                    args=[],
                    env_vars={"PCCL_LOG_LEVEL": "DEBUG"}
                )
                peers.append(p)
                print(f"[INFO] Spawned peer (from 1→2); PID={p.pid}")
                # Mark this time so we remember not to kill again too soon.
                last_spawn_from_singleton = time.time()
            else:
                # If we're at max_peers (very unlikely if max_peers>1),
                # just sleep and continue. No kill is allowed.
                pass

        else:
            # We have 2 or more peers.
            # Randomly choose spawn or kill, with a preference for "spawn" if small.
            action_candidates = []
            if num_peers < max_peers:
                action_candidates.append("spawn")
            if num_peers > 1:
                action_candidates.append("kill")

            action_weights = []
            for action in action_candidates:
                if action == "spawn":
                    action_weights.append(0.6)
                else:  # "kill"
                    action_weights.append(0.4)

            # Weighted choice: if we have very few peers, spawn is more likely.
            # Otherwise, kill is also an option.
            chosen_action = random.choices(
                action_candidates,
                weights=action_weights,
                k=1
            )[0]

            if chosen_action == "spawn":
                # Launch a new peer
                p = launch_py_process(
                    peer_script_path,
                    args=[],
                    env_vars={"PCCL_LOG_LEVEL": "DEBUG"}
                )
                peers.append(p)
                print(f"[INFO] Spawned new peer; PID={p.pid}")

                # If we just went from 1→2 peers, track time for the safety interval
                if num_peers == 1:
                    last_spawn_from_singleton = time.time()

            else:  # chosen_action == "kill"
                # Only kill if enough time has passed since last spawning from 1→2
                if last_spawn_from_singleton is not None:
                    time_since_singleton_spawn = time.time() - last_spawn_from_singleton
                else:
                    time_since_singleton_spawn = 999999.0  # effectively "forever"

                # We require at least 1 second after going from 1→2 before we kill again.
                if time_since_singleton_spawn > 1.0:
                    # Kill a random peer from the current list.
                    # We want to avoid killing all peers, but "action_candidates" check
                    # already ensures num_peers > 1.
                    victim = random.choice(peers)
                    victim_pid = victim.pid
                    print(f"[INFO] KILLING peer PID={victim_pid}")
                    victim.kill()
                    # We can either wait() or do it asynchronously
                    victim.wait()
                    peers.remove(victim)
                    print(f"[INFO] Confirmed peer PID={victim_pid} is terminated")
                else:
                    # If it's not safe to kill yet, just skip or spawn instead.
                    pass

        # Sleep a random time between 0.5 and 2 seconds to mix up the timing.
        sleep_time = random.uniform(0.5, 2.0)
        time.sleep(sleep_time)

    # 4) Done with the main stress test; now shut everything down gracefully.
    print("[INFO] Stress test duration complete. Terminating all peers...")

    for peer in peers:
        if peer.poll() is None:
            print(f"[INFO] Killing peer PID={peer.pid}")
            peer.kill()
            peer.wait()

    print("[INFO] All peers terminated. Killing master process...")
    if master_process.poll() is None:
        master_process.kill()
        master_process.wait()

    print("[INFO] Master terminated. Stress test orchestrator finished.")

###############################################################################
# If you want to run this module by itself, do so here:
###############################################################################
if __name__ == "__main__":
    # Example: run for ~8 hours with a max of 10 peers.
    run_stress_test(duration_hours=8.0, max_peers=10)
