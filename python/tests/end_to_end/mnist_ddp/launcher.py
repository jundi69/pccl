import os
import sys
import time
import uuid
import random
import threading
from ctypes import windll
from typing import List, Optional, Dict
import subprocess

import win32event
import win32gui
import win32process
from win32.lib import win32con


def launch_py_process_in_cmd(
        script_path: str,
        args: List[str],
        env_vars: Optional[Dict[str, str]] = None
) -> subprocess.Popen:
    env = {**dict(os.environ), **(env_vars or {})}
    cmd = [sys.executable, script_path] + args

    # Set up the STARTUPINFO to start the window hidden
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = win32con.SW_HIDE

    return subprocess.Popen(
        cmd,
        env=env,
        # Use CREATE_NEW_CONSOLE flag to open a new console window
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startupinfo,
    )

def get_screen_size():
    """Retrieve the width and height of the primary monitor."""
    user32 = windll.user32
    user32.SetProcessDPIAware()
    width = user32.GetSystemMetrics(0)
    height = user32.GetSystemMetrics(1)
    return width, height


def get_window_size(hwnd):
    """Retrieve the width and height of a window."""
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    return width, height


def find_inv_console_window_by_pid(pid, timeout=10):
    hwnd = None
    start_time = time.time()
    while time.time() - start_time < timeout:
        def callback(handle, extra):
            nonlocal hwnd
            if hwnd:
                return  # Already found
            if not win32gui.IsWindowVisible(handle):
                _, found_pid = win32process.GetWindowThreadProcessId(handle)
                if found_pid == pid:
                    # Ensure it's a console window
                    class_name = win32gui.GetClassName(handle)
                    if class_name == 'ConsoleWindowClass':
                        hwnd = handle
        win32gui.EnumWindows(callback, None)
        if hwnd:
            return hwnd
        time.sleep(0.5)
    return None


def set_window_position(hwnd, x, y):
    """Set the window position to (x, y) without changing its size."""
    # Get current window size
    rect = win32gui.GetWindowRect(hwnd)
    width = rect[2] - rect[0]
    height = rect[3] - rect[1]
    win32gui.SetWindowPos(
        hwnd,
        win32con.HWND_TOP,
        x,
        y,
        width,
        height,
        win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
    )

def reposition_window(proc, event_name):
    """Reposition the window of a process to a random position on the screen."""
    pid = proc.pid
    hwnd = find_inv_console_window_by_pid(pid)
    if not hwnd:
        print(f"Window for process {pid} not found")
        return
    screen_width, screen_height = get_screen_size()
    window_width, window_height = get_window_size(hwnd)

    x_pos = random.randint(0, screen_width - window_width)
    y_pos = random.randint(0, screen_height - window_height)

    # Wait for first message event to only show window after first message is print
    event_handle = win32event.OpenEvent(win32event.EVENT_ALL_ACCESS, False, event_name)
    if event_handle is None:
        print(f"Could not open event {event_name}")
        return

    # Wait for maximum 60 seconds
    result = win32event.WaitForSingleObject(event_handle, 60000)
    if result != win32event.WAIT_OBJECT_0:
        print(f"Timed out waiting for event {event_name}")
        return

    set_window_position(hwnd, x_pos, y_pos)

    # Show window after repositioning to avoid flickering
    win32gui.ShowWindow(hwnd, win32con.SW_SHOW)

window_procs = []

def launch_rank(rank: int):
    event_name = f"DemoImportsCompleteEvent_{uuid.uuid4()}"
    event_handle = win32event.CreateEvent(None, True, False, event_name)
    if event_handle is None:
        print(f"Failed to create event {event_name}")
        return

    proc = launch_py_process_in_cmd('mnist_peer.py', [], {'PCCL_LOG_LEVEL': 'DEBUG', 'RANK': str(rank), 'FIRST_MSG_EVENT_NAME': event_name, 'USE_TORCH_NUM_THREADS': '1'})
    def defer_show():
        nonlocal event_handle # Ensure event handle is not garbage collected
        reposition_window(proc, event_name)
        start_time = time.time()
        window_procs.append((proc, start_time))


    thread = threading.Thread(target=defer_show)
    thread.start()

    return proc

MIN_RANKS = 5
MAX_RANKS = 8
MAX_PROCESSES = 20
MIN_RUN_TIME = 10
TIME_BETWEEN_KILLS = 3
NO_MORE_PROCESSES = False

def main():
    global NO_MORE_PROCESSES
    procs = []
    def wait_for_success():
        nonlocal procs
        for p in procs:
            ret_code = p.wait()
            if ret_code == 0:
                global NO_MORE_PROCESSES
                NO_MORE_PROCESSES = True

    thread = threading.Thread(target=wait_for_success)
    thread.start()

    i = 0
    last_kill = 0
    while not NO_MORE_PROCESSES:
        while len(procs) < MAX_PROCESSES:
            proc = launch_rank(0)
            procs.append(proc)

        if time.time() - last_kill > TIME_BETWEEN_KILLS and len(window_procs) > 0 and len(window_procs) > MIN_RANKS:
            # kill random process
            tup = random.choice(window_procs)
            random_proc, start_time = tup
            if time.time() - start_time > MIN_RUN_TIME:
                random_proc.kill()
                window_procs.remove(tup)
                procs.remove(random_proc)
                last_kill = time.time()

        i += 1
        time.sleep(0.1)

    thread.join()
    for p in procs:
        p.wait()


if __name__ == '__main__':
    main()