"""Point conditioning: click points on a camera frame and format them as MolmoBot was trained.

Prompt format matches SynthmanipDataset._format_point_prompt (MolmoBot/olmo/data/synthmanip_dataset.py):
    goal + ' <points coords="1 1 XXX YYY">label</points>'
with x, y as 0-1000 integers normalized to the point camera's image.
"""

import cv2
import numpy as np

Points = dict[str, tuple[float, float]]  # label -> (x, y), normalized 0-1 in the image

_COLORS = [(255, 0, 255), (0, 255, 255), (255, 128, 0), (0, 255, 0)]  # RGB, one per label


def format_point_prompt(task: str, points: Points) -> str:
    tags = []
    for label, (x, y) in points.items():
        coords = f"1 {int(np.clip(x, 0.0, 1.0) * 1000):03d} {int(np.clip(y, 0.0, 1.0) * 1000):03d}"
        tags.append(f'<points coords="1 {coords}">{label}</points>')
    return f"{task} {' '.join(tags)}" if tags else task


def draw_points(image: np.ndarray, points: Points, labels: bool = True) -> np.ndarray:
    """Copy of an RGB image with the points drawn on it."""
    out = image.copy()
    h, w = out.shape[:2]
    radius = max(3, round(min(h, w) / 70))
    for i, (label, (x, y)) in enumerate(points.items()):
        center = (round(x * w), round(y * h))
        color = _COLORS[i % len(_COLORS)]
        cv2.circle(out, center, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(out, center, radius, color, -1, cv2.LINE_AA)
        if labels:
            org = (center[0] + radius + 4, center[1] - radius)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def select_points(frame: np.ndarray, labels: list[str], window: str = "Select points") -> Points:
    """Show an RGB frame and collect one click per label.

    Enter/space confirms once every label has a point, r restarts, Esc/q (or closing the window) aborts
    with KeyboardInterrupt.
    """
    h, w = frame.shape[:2]
    points: Points = {}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(labels):
            points[labels[len(points)]] = (x / w, y / h)

    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, w, h)
    cv2.setMouseCallback(window, on_mouse)
    try:
        while True:
            if len(points) < len(labels):
                msg = f"Click: {labels[len(points)]}   (r: restart, Esc: abort)"
            else:
                msg = "Enter: confirm   r: restart   Esc: abort"
            canvas = draw_points(frame, points)
            cv2.rectangle(canvas, (0, 0), (w, 34), (0, 0, 0), -1)
            cv2.putText(canvas, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            key = cv2.waitKey(30) & 0xFF
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
                raise KeyboardInterrupt("Point selection aborted")
            if key == ord("r"):
                points.clear()
            elif key in (13, 10, 32) and len(points) == len(labels):
                return dict(points)
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)
