import time

from PIL import Image, ImageDraw, ImageFont
import random


class Profiler:
    """Profiler that tracks nested sessions and prints their durations in a tree structure."""

    def __init__(self):
        # List of top-level sessions (roots)
        self.root_sessions = []
        # Stack of currently open sessions
        self.session_stack = []
        self.prev_time = 0

    def session(self, name: str):
        """Returns a context manager for timing a named session."""
        return _SessionContextManager(self, name)

    def start_session(self, name: str):
        new_node = SessionNode(name)
        new_node.start_time = max(time.perf_counter(), self.prev_time)
        self.prev_time = new_node.start_time

        if self.session_stack:
            parent_node = self.session_stack[-1]
            parent_node.children.append(new_node)
            new_node.parent = parent_node
        else:
            self.root_sessions.append(new_node)

        self.session_stack.append(new_node)

    def end_session(self):
        if not self.session_stack:
            raise RuntimeError("No session is currently open to end.")
        node = self.session_stack.pop()
        node.end_time = time.perf_counter()

    def print_report(self):
        """Prints a tree-structured timing report of all recorded sessions."""
        for session in self.root_sessions:
            self._print_session(session, level=0)

    def _print_session(self, session_node: "SessionNode", level: int):
        indent = '  ' * level
        print(f"{indent}- {session_node.name}: {session_node.duration:.6f} seconds")
        for child in session_node.children:
            self._print_session(child, level + 1)

    def export_timeline(
            self,
            filename=None,
            width=1200,
            row_height=40,
            return_image=False
    ):
        """
        Export (or return) a timeline image of this Profiler’s data.
        If `return_image=True`, we return a PIL Image object instead of saving to a file.
        If `filename` is given and `return_image=False`, we save to disk.
        """

        # 1) Gather all nodes (and their depths)
        all_nodes = []
        def collect_nodes(node, depth=0):
            all_nodes.append((node, depth))
            for c in node.children:
                collect_nodes(c, depth+1)

        for root in self.root_sessions:
            collect_nodes(root)

        if not all_nodes:
            print("No recorded sessions to draw.")
            return None

        # 2) Time range
        min_start = min(n.start_time for (n, _) in all_nodes)
        max_end   = max(n.end_time   for (n, _) in all_nodes)
        total_duration = max_end - min_start
        if total_duration <= 0:
            print("Profiler data has no measurable duration; cannot draw timeline.")
            return None

        # 3) Layout
        max_depth = max(depth for (_, depth) in all_nodes)
        PADDING_X = 50
        PADDING_Y = 30
        TIMELINE_AXIS_HEIGHT = 40
        BARS_OFFSET = 15

        effective_width = width - 2 * PADDING_X
        height = (
                TIMELINE_AXIS_HEIGHT
                + BARS_OFFSET
                + (max_depth + 1) * row_height
                + PADDING_Y
        )

        # 4) Create image
        img = Image.new("RGB", (width, height), color=(250, 250, 250))
        draw = ImageDraw.Draw(img)

        # 5) Font
        try:
            font = ImageFont.truetype("Arial.ttf", int(row_height / 3))
        except OSError:
            font = ImageFont.load_default()

        # 6) Deterministic color seeding (example)
        random.seed(1234)

        # Pastel color generator
        def pastel_color():
            r = 150 + random.randint(0, 105)
            g = 150 + random.randint(0, 105)
            b = 150 + random.randint(0, 105)
            return (r, g, b)

        # 7) Timeline axis
        axis_y = TIMELINE_AXIS_HEIGHT // 2
        draw.line(
            [(PADDING_X, axis_y), (PADDING_X + effective_width, axis_y)],
            fill=(0,0,0), width=1
        )

        # Ticks
        num_ticks = 10
        step = total_duration / num_ticks
        for i in range(num_ticks + 1):
            t = i * step
            x_tick = PADDING_X + int((t / total_duration) * effective_width)

            # Full gray line down
            draw.line([(x_tick, axis_y), (x_tick, height)], fill=(180,180,180), width=1)
            # Short black tick
            tick_size = 5
            draw.line([(x_tick, axis_y - tick_size), (x_tick, axis_y + tick_size)], fill=(0,0,0), width=1)

            # Label
            label_text = f"{t:.2f}s"
            bbox = draw.textbbox((0,0), label_text, font=font)
            label_w = bbox[2] - bbox[0]
            label_h = bbox[3] - bbox[1]
            label_x = x_tick - label_w // 2
            label_y = axis_y + tick_size + 2
            draw.text((label_x, label_y), label_text, fill=(0,0,0), font=font)

        # Helper to truncate text if bar is too short
        def truncate_text_to_fit(original_text, max_width):
            text = original_text
            bbox = draw.textbbox((0,0), text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_width:
                return text

            base_text = text
            while len(base_text) > 0:
                trial_text = base_text + "..."
                w = draw.textbbox((0,0), trial_text, font=font)
                if (w[2] - w[0]) <= max_width:
                    return trial_text
                base_text = base_text[:-1]
            return "..."

        # 8) Draw session bars
        for (node, depth) in all_nodes:
            node_start_offset = node.start_time - min_start
            node_end_offset   = node.end_time   - min_start

            x1 = PADDING_X + int((node_start_offset / total_duration)*effective_width)
            x2 = PADDING_X + int((node_end_offset   / total_duration)*effective_width)
            y1 = TIMELINE_AXIS_HEIGHT + BARS_OFFSET + depth * row_height
            y2 = y1 + (row_height - 10)

            rect_color = pastel_color()
            outline_color = (120,120,120)

            draw.rectangle([x1, y1, x2, y2], fill=rect_color, outline=outline_color)

            # Label
            raw_label = f"{node.name} ({node.duration:.4f}s)"
            bar_width = (x2 - x1) - 10
            label_text = truncate_text_to_fit(raw_label, bar_width)
            draw.text((x1+5, y1+5), label_text, fill=(0,0,0), font=font)

        # 9) Output
        if return_image:
            # Return the PIL image object
            return img
        else:
            # Save to a file if filename is given
            if filename:
                img.save(filename)
                print(f"Timeline exported to {filename}.")
            return None


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


import imageio
import numpy as np

class ProfilerCollection:
    """
    A collection of Profiler snapshots, one per 'frame' or iteration.
    You can push multiple fully-populated profilers into it, then
    export them as a video showing how the profiling changed over time.
    """
    def __init__(self):
        self.frames = []  # will hold (profiler, label) pairs

    def add_profiler(self, profiler, label=None):
        """Store a profiler and optionally a textual label (e.g. 'Frame #12')."""
        if label is None:
            label = f"Frame {len(self.frames)}"
        self.frames.append((profiler, label))

    def render_as_video(self, out_filename="profiler_video.mp4", fps=2):
        """
        Render all stored profilers as frames in a video (mp4) file.
        Requires 'imageio' and 'imageio-ffmpeg' installed.
        """
        if not self.frames:
            print("ProfilerCollection is empty; nothing to render.")
            return

        # We'll create a writer with the given fps.
        with imageio.get_writer(out_filename, fps=fps) as writer:
            for idx, (profiler, label) in enumerate(self.frames):
                # Grab an in-memory image from the profiler
                img = profiler.export_timeline(return_image=True)  # returns a PIL Image
                if img is None:
                    # Possibly means no data or an error
                    continue

                # If you want to overlay the label on the image, do it here:
                # e.g. using PIL to draw text on `img` if desired
                #  draw = ImageDraw.Draw(img)
                #  draw.text((10,10), label, fill=(0,0,0))  # etc.

                # Convert the PIL image to a NumPy array for imageio
                frame = np.array(img)
                writer.append_data(frame)

        print(f"Video exported to {out_filename}, with {len(self.frames)} frames at {fps} fps.")