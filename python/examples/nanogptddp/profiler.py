import time


class SessionNode:
    """Tree node holding data about an individual session."""

    def __init__(self, name):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.children = []
        self.parent = None

    @property
    def duration(self):
        """Returns the duration of the session (in seconds)."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return 0


class Profiler:
    """Profiler that tracks nested sessions and prints their durations in a tree structure."""

    def __init__(self):
        # List of top-level sessions (roots)
        self.root_sessions = []
        # Stack of currently open sessions
        self.session_stack = []

    def session(self, name: str):
        """
        Returns a context manager for timing a named session.
        Usage:
            with profiler.session("my_task"):
                # do something
        """
        return _SessionContextManager(self, name)

    def start_session(self, name: str):
        """
        Manually start a session (used internally by the context manager).
        You can also call this directly if you'd prefer to manage start/end calls yourself.
        """
        new_node = SessionNode(name)
        new_node.start_time = time.perf_counter()

        # If there's a session open, set the new one as a child of the topmost session
        if self.session_stack:
            parent_node = self.session_stack[-1]
            parent_node.children.append(new_node)
            new_node.parent = parent_node
        else:
            # Otherwise, it's a root session
            self.root_sessions.append(new_node)

        # Push this session on the stack
        self.session_stack.append(new_node)

    def end_session(self):
        """
        Manually end a session (used internally by the context manager).
        """
        if not self.session_stack:
            raise RuntimeError("No session is currently open to end.")
        node = self.session_stack.pop()
        node.end_time = time.perf_counter()

    def print_report(self):
        """Prints a tree-structured timing report of all recorded sessions."""
        for session in self.root_sessions:
            self._print_session(session, level=0)

    def _print_session(self, session_node: SessionNode, level: int):
        indent = '  ' * level
        print(f"{indent}- {session_node.name}: {session_node.duration:.6f} seconds")
        for child in session_node.children:
            self._print_session(child, level + 1)


class _SessionContextManager:
    """Context manager used by Profiler to start/end sessions."""

    def __init__(self, profiler, name):
        self.profiler = profiler
        self.name = name

    def __enter__(self):
        self.profiler.start_session(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_session()
        # Returning False so that any exception is still raised
        return False
