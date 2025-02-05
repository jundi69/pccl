import os
import time

import imageio
import numpy as np
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
            width=2400,
            row_height=80,
            return_image=False
    ):
        """
        Draw a timeline image of this Profiler's data.
        If return_image=True, return a PIL Image object instead of saving to a file.
        If filename is provided and return_image=False, save the image to disk as filename.
        """

        # Gather nodes
        all_nodes = []
        def collect_nodes(node, depth=0):
            all_nodes.append((node, depth))
            for c in node.children:
                collect_nodes(c, depth+1)
        for root in self.root_sessions:
            collect_nodes(root)

        if not all_nodes:
            print("No recorded sessions.")
            return None

        min_start = min(n.start_time for n,_ in all_nodes)
        max_end   = max(n.end_time for n,_ in all_nodes)
        total_duration = max_end - min_start
        if total_duration <= 0:
            print("Profiler data has no measurable duration.")
            return None

        # Layout
        max_depth = max(d for _,d in all_nodes)
        PADDING_X = 50
        PADDING_Y = 30
        TIMELINE_AXIS_HEIGHT = 40
        BARS_OFFSET = 15
        effective_width = width - 2*PADDING_X
        height = TIMELINE_AXIS_HEIGHT + BARS_OFFSET + (max_depth+1)*row_height + PADDING_Y

        # Create image
        img = Image.new("RGB", (width, height), (250,250,250))
        draw = ImageDraw.Draw(img)

        # Font
        try:
            font = ImageFont.truetype("Arial.ttf", int(row_height/3))
        except OSError:
            font = ImageFont.load_default()

        # Make color generation deterministic
        random.seed(1234)

        def pastel_color():
            r = 150 + random.randint(0,105)
            g = 150 + random.randint(0,105)
            b = 150 + random.randint(0,105)
            return (r,g,b)

        # Draw top timeline axis
        axis_y = TIMELINE_AXIS_HEIGHT // 2
        draw.line([(PADDING_X, axis_y), (PADDING_X+effective_width, axis_y)], fill=(0,0,0), width=1)

        # Ticks
        num_ticks = 10
        step = total_duration / num_ticks
        for i in range(num_ticks+1):
            t = i * step
            x_tick = PADDING_X + int((t/total_duration)*effective_width)

            # Long gray line downward
            draw.line([(x_tick, axis_y), (x_tick, height)], fill=(180,180,180), width=1)

            # Short black tick
            tick_size = 5
            draw.line([(x_tick, axis_y - tick_size), (x_tick, axis_y + tick_size)], fill=(0,0,0), width=1)

            # Label
            label_text = f"{t:.2f}s"
            bbox = draw.textbbox((0, 0), label_text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            draw.text((x_tick - w//2, axis_y + tick_size + 2), label_text, fill=(0,0,0), font=font)

        # Helper to truncate text if bar is too short
        def truncate_text_to_fit(original_text, max_width):
            bbox = draw.textbbox((0,0), original_text, font=font)
            text_w = bbox[2] - bbox[0]
            if text_w <= max_width:
                return original_text

            base = original_text
            while base:
                trial = base + "..."
                trial_w = draw.textbbox((0,0), trial, font=font)
                if (trial_w[2] - trial_w[0]) <= max_width:
                    return trial
                base = base[:-1]
            return "..."

        # Draw session bars
        outline_color = (120,120,120)
        for node, depth in all_nodes:
            start_off = node.start_time - min_start
            end_off   = node.end_time - min_start
            x1 = PADDING_X + int((start_off/total_duration)*effective_width)
            x2 = PADDING_X + int((end_off  /total_duration)*effective_width)
            y1 = TIMELINE_AXIS_HEIGHT + BARS_OFFSET + depth*row_height
            y2 = y1 + row_height - 10

            rect_color = pastel_color()
            draw.rectangle([(x1,y1),(x2,y2)], fill=rect_color, outline=outline_color)

            # Label
            raw_text = f"{node.name} ({node.duration:.4f}s)"
            bar_width = (x2 - x1) - 10
            label_text = truncate_text_to_fit(raw_text, bar_width)
            draw.text((x1+5, y1+5), label_text, fill=(0,0,0), font=font)

        # Return or save
        if return_image:
            return img
        else:
            if filename:
                img.save(filename)
                print(f"Timeline exported to {filename}")
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

class ProfilerCollection:
    """
    Accumulates (profiler, label) frames. We can then export them
    all to a video. If you call render_as_video multiple times with
    'append=True', only new frames since the last invocation are
    encoded, and we 'concat' them to the existing video.
    """

    def __init__(self):
        self.frames = []  # list of (profiler, label)
        self._rendered_up_to = 0  # how many frames we've already included in the output

    def add_profiler(self, profiler, label=None):
        """Append a new frame (profiler snapshot) to the collection."""
        if label is None:
            label = f"Frame {len(self.frames)}"
        self.frames.append((profiler, label))

    def render_as_video(self, out_filename="profiler_video.mp4", fps=2, append=True):
        """
        Render frames as a video. If append=True and the output file already
        exists, we do an incremental update:
          - Encode only the new frames into a temporary mp4
          - Use ffmpeg concat to merge it with the existing mp4
        Otherwise, we just encode all frames from scratch.
        """
        if not self.frames:
            print("No frames in ProfilerCollection; nothing to render.")
            return

        start_index = 0
        do_concat = False

        if append and os.path.exists(out_filename):
            # We'll only render new frames from self._rendered_up_to..end
            if self._rendered_up_to >= len(self.frames):
                print("No new frames to append.")
                return
            start_index = self._rendered_up_to
            do_concat = True
            print(f"Appending {len(self.frames) - start_index} frames to {out_filename}.")
        else:
            # (Re)write from scratch
            print(f"Writing {len(self.frames)} frames from scratch to {out_filename}.")

        # Encode partial video to a temp file
        partial_filename = out_filename + ".partial.mp4"
        self._encode_frames_to_video(partial_filename, fps, start_index, len(self.frames))

        if do_concat:
            # Concat existing out_filename + partial_filename => final
            merged_filename = out_filename + ".merged.mp4"
            self._concat_videos_ffmpeg(out_filename, partial_filename, merged_filename)

            # Move merged => out_filename
            os.remove(out_filename)
            os.rename(merged_filename, out_filename)
            os.remove(partial_filename)

            self._rendered_up_to = len(self.frames)
            print(f"Appended new frames; final video: {out_filename}")
        else:
            # We just wrote the entire video from scratch
            # move partial => out_filename
            if os.path.exists(out_filename):
                os.remove(out_filename)
            os.rename(partial_filename, out_filename)

            self._rendered_up_to = len(self.frames)
            print(f"Video saved to {out_filename}")


    def _encode_frames_to_video(self, tmp_filename, fps, start_idx, end_idx):
        """
        Writes frames [start_idx..end_idx) to tmp_filename using imageio.
        """
        with imageio.get_writer(tmp_filename, fps=fps) as writer:
            for i in range(start_idx, end_idx):
                profiler, label = self.frames[i]
                # 1) Generate the timeline image
                img = profiler.export_timeline(return_image=True)
                if img is None:
                    continue

                # 2) Optional: overlay the label with PIL
                draw = ImageDraw.Draw(img)
                draw.text((10,10), label, fill=(0,0,0))

                # 3) Convert to numpy array for imageio
                frame = np.array(img)
                writer.append_data(frame)

    def _concat_videos_ffmpeg(self, video1, video2, output):
        """
        Use ffmpeg concat demuxer to merge video1 + video2 into 'output'.
        Both must have identical encoding parameters.
        """
        list_file = "concat_list.txt"
        with open(list_file, "w") as f:
            f.write(f"file '{os.path.abspath(video1)}'\n")
            f.write(f"file '{os.path.abspath(video2)}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file, "-c", "copy", output
        ]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
        os.remove(list_file)