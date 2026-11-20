"""Background mp4 writing so video encoding never blocks the control loop."""

import queue
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from point_prompt import draw_points


class VideoWriter:
    def __init__(self, path: Path, fps: float):
        self.path = path
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self._writer = imageio.get_writer(
            path, fps=fps, codec="libx264", macro_block_size=2, ffmpeg_params=["-preset", "veryfast"]
        )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.n_frames = 0

    def _run(self):
        while (frame := self._queue.get()) is not None:
            self._writer.append_data(frame)
        self._writer.close()

    def add(self, frame: np.ndarray):
        self._queue.put(frame)
        self.n_frames += 1

    def close(self):
        self._queue.put(None)
        self._thread.join()


def agent_view(
    policy_obs: dict, camera_names: list[str], points: dict | None = None, point_camera: str | None = None
) -> np.ndarray:
    """The exact images sent to the policy this step, side by side, with any conditioning points drawn
    on the point camera's panel (on a copy; the policy's images are untouched)."""
    frames = [
        draw_points(policy_obs[name], points, labels=False) if points and name == point_camera else policy_obs[name]
        for name in camera_names
    ]
    height = min(f.shape[0] for f in frames)
    return np.concatenate([f[:height] for f in frames], axis=1)
