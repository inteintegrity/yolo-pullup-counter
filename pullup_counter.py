"""YOLO pose based pull-up counter.

The program detects one person in each video frame, draws the person's box and
COCO pose skeleton, then counts a repetition when the nose moves above the
pull-up bar and subsequently returns to the hanging position.

Example:
    python pullup_counter.py --source video/input/test.mp4
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# Ultralytics tries to write its settings outside the workspace by default.
# Keeping this inside the project makes the script work in restricted Windows
# environments as well as in a normal local Python installation.
PROJECT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))

from ultralytics import YOLO  # noqa: E402


NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10

# COCO keypoint connections. The hand/foot links make the overlay easy to
# read, while the main body links are kept brighter below.
SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass
class PoseFrame:
    """The few pose values used by the counter and visualizer."""

    box: tuple[int, int, int, int]
    box_conf: float
    keypoints: np.ndarray
    keypoint_conf: np.ndarray
    nose_y: Optional[float]
    shoulder_y: Optional[float]
    wrist_y: Optional[float]


class PullUpCounter:
    """Hysteresis state machine for pull-up repetitions.

    A top position is recognized when the nose is close to or above the bar.
    A bottom position is recognized when the nose has returned clearly below
    the bar. Separate thresholds prevent keypoint jitter from double-counting.
    """

    def __init__(self, min_keypoint_conf: float = 0.35) -> None:
        self.min_keypoint_conf = min_keypoint_conf
        self.count = 0
        self.state = "WAITING"
        self.bar_y: Optional[float] = None
        self.smoothed_nose_y: Optional[float] = None
        self.smoothed_shoulder_y: Optional[float] = None
        self.stable_frames = 0
        self.last_event = ""

    @staticmethod
    def _mean_visible(values: list[Optional[float]]) -> Optional[float]:
        visible = [v for v in values if v is not None]
        return float(np.mean(visible)) if visible else None

    def update(self, pose: PoseFrame) -> None:
        """Update bar calibration and state from one detected pose."""

        if pose.wrist_y is not None:
            # Both hands stay on the bar during this exercise. The slow EMA
            # ignores one-frame wrist jitter without lagging the bar estimate.
            if self.bar_y is None:
                self.bar_y = pose.wrist_y
            else:
                self.bar_y = 0.04 * pose.wrist_y + 0.96 * self.bar_y

        if (pose.nose_y is None and pose.shoulder_y is None) or self.bar_y is None:
            self.state = "DETECTING"
            self.stable_frames = 0
            return

        if pose.nose_y is not None:
            if self.smoothed_nose_y is None:
                self.smoothed_nose_y = pose.nose_y
            else:
                self.smoothed_nose_y = 0.35 * pose.nose_y + 0.65 * self.smoothed_nose_y

        if pose.shoulder_y is not None:
            if self.smoothed_shoulder_y is None:
                self.smoothed_shoulder_y = pose.shoulder_y
            else:
                self.smoothed_shoulder_y = (
                    0.35 * pose.shoulder_y + 0.65 * self.smoothed_shoulder_y
                )

        height = max(pose.box[3] - pose.box[1], 1)
        # Only use the nose signal when the nose is visible in the current
        # frame. Do not reuse an old nose location while the bar occludes it.
        relative_nose = (
            (self.smoothed_nose_y - self.bar_y) / height
            if pose.nose_y is not None and self.smoothed_nose_y is not None
            else None
        )
        relative_shoulder = (
            (self.smoothed_shoulder_y - self.bar_y) / height
            if self.smoothed_shoulder_y is not None
            else None
        )

        # Nose near the bar means the chin/head has reached the top. The
        # lower threshold is deliberately much farther away for hysteresis.
        # The person fills most of this portrait frame, so a dead hang is
        # about 0.12 body-heights below the bar. At the top, the nose may be
        # hidden by the bar; the shoulder signal then provides a stable proxy.
        top_threshold = 0.08
        bottom_threshold = 0.12
        # Shoulder is a fallback only while the current frame's nose is
        # unavailable. Using both signals at once would make a single noisy
        # keypoint able to flip the state machine twice in one repetition.
        active_signal = relative_nose if relative_nose is not None else relative_shoulder
        reached_top = active_signal is not None and active_signal <= top_threshold
        # For the return-to-hang transition, only the nose signal is used.
        # The shoulder-to-bar ratio changes with arm bend and can otherwise
        # look like a false bottom while the athlete is still at the top.
        reached_bottom = relative_nose is not None and relative_nose >= bottom_threshold

        if self.state in ("WAITING", "DETECTING", "NO PERSON"):
            if reached_bottom:
                self.stable_frames += 1
                if self.stable_frames >= 3:
                    self.state = "HANG"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
        elif self.state == "HANG":
            self.state = "PULLING"
            if reached_top:
                self.stable_frames += 1
                if self.stable_frames >= 2:
                    self.count += 1
                    self.last_event = "REP COMPLETE"
                    self.state = "TOP"
                    self.stable_frames = 0
            else:
                self.stable_frames = 0
        elif self.state == "PULLING":
            if reached_top:
                self.stable_frames += 1
                if self.stable_frames >= 2:
                    self.count += 1
                    self.last_event = "REP COMPLETE"
                    self.state = "TOP"
                    self.stable_frames = 0
            elif reached_bottom:
                self.state = "HANG"
                self.stable_frames = 0
        elif self.state == "TOP":
            if reached_bottom:
                self.state = "HANG"
                self.stable_frames = 0

    def overlay_values(self, pose: Optional[PoseFrame]) -> tuple[Optional[float], Optional[float]]:
        """Return values used for the optional debug lines in the overlay."""

        if pose is None or self.bar_y is None or pose.nose_y is None:
            return self.bar_y, None
        height = max(pose.box[3] - pose.box[1], 1)
        relative_nose = (pose.nose_y - self.bar_y) / height
        return self.bar_y, relative_nose


def _point(values: np.ndarray, confidences: np.ndarray, index: int, threshold: float) -> Optional[tuple[float, float]]:
    if index >= len(values) or index >= len(confidences) or float(confidences[index]) < threshold:
        return None
    x, y = values[index]
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return float(x), float(y)


def choose_pose(result, keypoint_conf_threshold: float) -> Optional[PoseFrame]:
    """Select the highest-confidence detected person and normalize its pose."""

    if result.boxes is None or len(result.boxes) == 0 or result.keypoints is None:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    box_conf = result.boxes.conf.cpu().numpy()
    keypoints = result.keypoints.xy.cpu().numpy()
    keypoint_conf = result.keypoints.conf
    if keypoint_conf is None:
        keypoint_conf_np = np.ones(keypoints.shape[:2], dtype=np.float32)
    else:
        keypoint_conf_np = keypoint_conf.cpu().numpy()

    # The demo contains one athlete. The confidence-first selection avoids
    # accidentally switching to a distant bystander when someone walks past.
    person_index = int(np.argmax(box_conf))
    xy = keypoints[person_index]
    kp_conf = keypoint_conf_np[person_index]
    x1, y1, x2, y2 = boxes[person_index]

    nose = _point(xy, kp_conf, NOSE, keypoint_conf_threshold)
    left_shoulder = _point(xy, kp_conf, LEFT_SHOULDER, keypoint_conf_threshold)
    right_shoulder = _point(xy, kp_conf, RIGHT_SHOULDER, keypoint_conf_threshold)
    left_wrist = _point(xy, kp_conf, LEFT_WRIST, keypoint_conf_threshold)
    right_wrist = _point(xy, kp_conf, RIGHT_WRIST, keypoint_conf_threshold)

    shoulder_y = PullUpCounter._mean_visible([
        left_shoulder[1] if left_shoulder else None,
        right_shoulder[1] if right_shoulder else None,
    ])
    wrist_y = PullUpCounter._mean_visible([
        left_wrist[1] if left_wrist else None,
        right_wrist[1] if right_wrist else None,
    ])

    return PoseFrame(
        box=(int(x1), int(y1), int(x2), int(y2)),
        box_conf=float(box_conf[person_index]),
        keypoints=xy,
        keypoint_conf=kp_conf,
        nose_y=nose[1] if nose else None,
        shoulder_y=shoulder_y,
        wrist_y=wrist_y,
    )


def draw_pose(frame: np.ndarray, pose: PoseFrame, threshold: float) -> None:
    """Draw a clean bounding box and pose skeleton without a data table."""

    x1, y1, x2, y2 = pose.box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 80), 4)
    label = f"Person {pose.box_conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    label_y = max(y1 - 10, th + 8)
    cv2.rectangle(frame, (x1, label_y - th - 10), (x1 + tw + 12, label_y + 4), (40, 220, 80), -1)
    cv2.putText(frame, label, (x1 + 6, label_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (10, 30, 10), 2, cv2.LINE_AA)

    for a, b in SKELETON:
        pa = _point(pose.keypoints, pose.keypoint_conf, a, threshold)
        pb = _point(pose.keypoints, pose.keypoint_conf, b, threshold)
        if pa is None or pb is None:
            continue
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (255, 210, 50), 4, cv2.LINE_AA)

    for index, (x, y) in enumerate(pose.keypoints):
        if index >= len(pose.keypoint_conf) or float(pose.keypoint_conf[index]) < threshold:
            continue
        color = (0, 80, 255) if index == NOSE else (255, 255, 255)
        cv2.circle(frame, (int(x), int(y)), 6 if index == NOSE else 5, color, -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    counter: PullUpCounter,
    pose: Optional[PoseFrame],
    show_debug: bool,
) -> None:
    """Draw the count and short status labels in the top-left corner."""

    height, width = frame.shape[:2]
    panel_w = min(440, max(330, width // 3))
    panel_h = 165 if show_debug else 135
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (20 + panel_w, 20 + panel_h), (8, 12, 22), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
    cv2.rectangle(frame, (20, 20), (20 + panel_w, 20 + panel_h), (80, 210, 255), 2)

    cv2.putText(frame, "PULL-UPS", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"COUNT  {counter.count}", (40, 118), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0, 80, 255), 4, cv2.LINE_AA)
    cv2.putText(frame, counter.state, (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 240, 160), 2, cv2.LINE_AA)

    if show_debug:
        bar_y, relative_nose = counter.overlay_values(pose)
        debug = "bar --  nose --" if bar_y is None or relative_nose is None else f"bar {bar_y:.0f}  nose/bar {relative_nose:.2f}"
        cv2.putText(frame, debug, (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (205, 215, 225), 1, cv2.LINE_AA)


def process_video(
    source: Path,
    output: Path,
    csv_path: Path,
    model_path: str,
    conf: float,
    keypoint_conf: float,
    imgsz: int,
    show_debug: bool,
) -> int:
    """Process one video and return the final repetition count."""

    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output}")

    model = YOLO(model_path)
    counter = PullUpCounter(keypoint_conf)
    rows: list[dict[str, object]] = []
    frame_index = 0
    lost_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=conf,
                iou=0.5,
                imgsz=imgsz,
                device="cpu",
                classes=[0],
                verbose=False,
            )
            pose = choose_pose(results[0], keypoint_conf)
            if pose is not None:
                lost_frames = 0
                counter.update(pose)
                draw_pose(frame, pose, keypoint_conf)
            else:
                lost_frames += 1
                if lost_frames > 15:
                    counter.state = "NO PERSON"
                    counter.stable_frames = 0

            draw_hud(frame, counter, pose, show_debug)
            writer.write(frame)

            bar_y, relative_nose = counter.overlay_values(pose)
            rows.append({
                "frame": frame_index,
                "time_sec": round(frame_index / fps, 3),
                "count": counter.count,
                "state": counter.state,
                "person_conf": round(pose.box_conf, 4) if pose else "",
                "bar_y": round(bar_y, 2) if bar_y is not None else "",
                "nose_y": round(pose.nose_y, 2) if pose and pose.nose_y is not None else "",
                "nose_bar_ratio": round(relative_nose, 4) if relative_nose is not None else "",
            })

            frame_index += 1
            if frame_index % 30 == 0 or frame_index == total:
                print(f"Progress: {frame_index}/{total} frames, count: {counter.count}")
    finally:
        cap.release()
        writer.release()

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(rows[0].keys()) if rows else ["frame", "time_sec", "count", "state"]
        csv_writer = csv.DictWriter(handle, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(rows)

    return counter.count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Count pull-ups with a YOLO pose model")
    parser.add_argument("--source", type=Path, default=Path("video/input/test.mp4"), help="Input video path")
    parser.add_argument("--output", type=Path, default=None, help="Output video path; defaults to video/output")
    parser.add_argument("--csv", type=Path, default=None, help="Per-frame CSV output path")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="YOLO pose model, e.g. yolo11s-pose.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Person detection confidence threshold")
    parser.add_argument("--keypoint-conf", type=float, default=0.35, help="Keypoint confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size; 960 may improve small-person accuracy but is slower")
    parser.add_argument("--debug", action="store_true", help="Show bar/nose debug values in the video")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.output is None:
        args.output = args.source.parent.parent / "output" / f"{args.source.stem}_counted.mp4"
    if args.csv is None:
        args.csv = args.output.with_suffix(".csv")

    print(f"Input video: {args.source}")
    print(f"Output video: {args.output}")
    print(f"Model: {args.model}")
    count = process_video(
        source=args.source,
        output=args.output,
        csv_path=args.csv,
        model_path=args.model,
        conf=args.conf,
        keypoint_conf=args.keypoint_conf,
        imgsz=args.imgsz,
        show_debug=args.debug,
    )
    print(f"Done. Final pull-up count: {count}")
    print(f"Counted video: {args.output}")
    print(f"Per-frame data: {args.csv}")


if __name__ == "__main__":
    main()
