"""Video multi-object tracking with line-crossing counting (ByteTrack + YOLOv8)."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


DEFAULT_TRACKER = Path(__file__).resolve().parent / "configs" / "bytetrack_occlusion.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to trained best.pt.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs.")
    parser.add_argument(
        "--line-y", type=float, default=None,
        help="Horizontal line as fraction of frame height (0.0-1.0).",
    )
    parser.add_argument(
        "--line-x", type=float, default=None,
        help="Vertical line as fraction of frame width (0.0-1.0).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.1,
        help="Detection confidence passed into tracker. ByteTrack benefits from low-confidence boxes.",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument(
        "--tracker",
        default=str(DEFAULT_TRACKER),
        help="Tracker yaml. Defaults to an occlusion-tolerant ByteTrack config in configs/.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--codec", default="mp4v", help="Output video codec.")
    parser.add_argument(
        "--reconnect-window",
        type=int,
        default=90,
        help="Frames to search for a recently lost display ID when ByteTrack creates a new raw ID.",
    )
    parser.add_argument(
        "--reconnect-distance",
        type=float,
        default=0.12,
        help="Max reconnect distance as a fraction of frame diagonal.",
    )
    parser.add_argument(
        "--min-size-ratio",
        type=float,
        default=0.35,
        help="Minimum area ratio for reconnecting a new raw ID to a previous display ID.",
    )
    parser.add_argument(
        "--max-size-ratio",
        type=float,
        default=3.0,
        help="Maximum area ratio for reconnecting a new raw ID to a previous display ID.",
    )
    parser.add_argument(
        "--disable-missing-prediction",
        action="store_true",
        help="Disable short-term motion prediction for tracks hidden while crossing the virtual line.",
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=10,
        help="Save a snapshot PNG every N frames. Default 10.",
    )
    parser.add_argument(
        "--snapshot-start", type=int, default=0,
        help="Start saving snapshots from this frame index. Default 0.",
    )
    return parser.parse_args()


def crosses_line(prev_y: float, curr_y: float, line_y: float) -> bool:
    return (prev_y < line_y <= curr_y) or (curr_y <= line_y < prev_y)


def class_family(class_name: str) -> str:
    name = class_name.strip().lower()
    car_like = {
        "ambulance",
        "car",
        "minivan",
        "pickup",
        "policecar",
        "suv",
        "taxi",
        "van",
    }
    large_vehicle = {"army vehicle", "bus", "garbagevan", "human hauler", "minibus", "truck"}
    two_wheeler = {"bicycle", "motorbike", "scooter"}
    three_wheeler = {"auto rickshaw", "rickshaw", "three wheelers -cng-"}
    if name in car_like:
        return "car_like"
    if name in large_vehicle:
        return "large_vehicle"
    if name in two_wheeler:
        return "two_wheeler"
    if name in three_wheeler:
        return "three_wheeler"
    return name


@dataclass
class DisplayTrackState:
    display_id: int
    raw_track_id: int
    class_name: str
    family: str
    last_frame: int
    last_cx: float
    last_cy: float
    width: float
    height: float
    prev_frame: int | None = None
    prev_cx: float | None = None
    prev_cy: float | None = None
    counted: bool = False

    @property
    def area(self) -> float:
        return max(self.width * self.height, 1.0)

    def predicted_center(self, frame_idx: int) -> tuple[float, float]:
        if self.prev_frame is None or self.prev_cx is None or self.prev_cy is None:
            return self.last_cx, self.last_cy
        frame_gap = max(self.last_frame - self.prev_frame, 1)
        vx = (self.last_cx - self.prev_cx) / frame_gap
        vy = (self.last_cy - self.prev_cy) / frame_gap
        missing = max(frame_idx - self.last_frame, 0)
        return self.last_cx + vx * missing, self.last_cy + vy * missing

    def update(
        self,
        raw_track_id: int,
        class_name: str,
        cx: float,
        cy: float,
        width: float,
        height: float,
        frame_idx: int,
    ) -> None:
        self.prev_frame = self.last_frame
        self.prev_cx = self.last_cx
        self.prev_cy = self.last_cy
        self.raw_track_id = raw_track_id
        self.class_name = class_name
        self.family = class_family(class_name)
        self.last_frame = frame_idx
        self.last_cx = cx
        self.last_cy = cy
        self.width = width
        self.height = height


@dataclass
class TrackUpdate:
    display_id: int
    crossed: bool
    reconnected: bool


class TrackPostProcessor:
    def __init__(
        self,
        *,
        line_px: float,
        use_vertical: bool,
        reconnect_window: int,
        reconnect_distance_px: float,
        min_size_ratio: float = 0.35,
        max_size_ratio: float = 3.0,
        predict_missing_crossings: bool = True,
    ) -> None:
        self.line_px = line_px
        self.use_vertical = use_vertical
        self.reconnect_window = reconnect_window
        self.reconnect_distance_px = reconnect_distance_px
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.predict_missing_crossings = predict_missing_crossings
        self.raw_to_display: dict[int, int] = {}
        self.display_states: dict[int, DisplayTrackState] = {}
        self.active_display_ids: set[int] = set()
        self.current_raw_ids: set[int] = set()
        self.crossed_display_ids: set[int] = set()
        self.next_display_id = 1
        self.count = 0
        self.predicted_crossings = 0

    def begin_frame(self, raw_track_ids: list[int] | None = None) -> None:
        self.active_display_ids = set()
        self.current_raw_ids = set(raw_track_ids or [])

    def update(
        self,
        *,
        raw_track_id: int,
        class_name: str,
        cx: float,
        cy: float,
        width: float,
        height: float,
        frame_idx: int,
    ) -> TrackUpdate:
        raw_track_id = int(raw_track_id)
        reconnected = False
        if raw_track_id in self.raw_to_display:
            display_id = self.raw_to_display[raw_track_id]
            if display_id in self.active_display_ids:
                display_id = self._new_display_id()
                self.raw_to_display[raw_track_id] = display_id
        else:
            display_id = self._find_reconnect(class_name, cx, cy, width, height, frame_idx)
            if display_id is None:
                display_id = self._new_display_id()
            else:
                reconnected = True
            self.raw_to_display[raw_track_id] = display_id

        state = self.display_states.get(display_id)
        crossed = False
        if state is not None:
            prev_coord = state.last_cx if self.use_vertical else state.last_cy
            curr_coord = cx if self.use_vertical else cy
            if not state.counted and crosses_line(prev_coord, curr_coord, self.line_px):
                self._record_crossing(display_id, state)
                crossed = True
            state.update(raw_track_id, class_name, cx, cy, width, height, frame_idx)
        else:
            self.display_states[display_id] = DisplayTrackState(
                display_id=display_id,
                raw_track_id=raw_track_id,
                class_name=class_name,
                family=class_family(class_name),
                last_frame=frame_idx,
                last_cx=cx,
                last_cy=cy,
                width=width,
                height=height,
            )

        self.active_display_ids.add(display_id)
        return TrackUpdate(display_id=display_id, crossed=crossed, reconnected=reconnected)

    def finish_frame(self, frame_idx: int) -> None:
        if not self.predict_missing_crossings:
            return
        for display_id, state in self.display_states.items():
            if display_id in self.active_display_ids or state.counted:
                continue
            if state.prev_frame is None or state.prev_cx is None or state.prev_cy is None:
                continue
            gap = frame_idx - state.last_frame
            if gap <= 0 or gap > self.reconnect_window:
                continue

            pred_x, pred_y = state.predicted_center(frame_idx)
            prev_coord = state.last_cx if self.use_vertical else state.last_cy
            pred_coord = pred_x if self.use_vertical else pred_y
            if crosses_line(prev_coord, pred_coord, self.line_px):
                self._record_crossing(display_id, state)
                self.predicted_crossings += 1

    def _record_crossing(self, display_id: int, state: DisplayTrackState) -> None:
        state.counted = True
        self.crossed_display_ids.add(display_id)
        self.count += 1

    def _new_display_id(self) -> int:
        display_id = self.next_display_id
        self.next_display_id += 1
        return display_id

    def _find_reconnect(
        self,
        class_name: str,
        cx: float,
        cy: float,
        width: float,
        height: float,
        frame_idx: int,
    ) -> int | None:
        family = class_family(class_name)
        area = max(width * height, 1.0)
        best_id: int | None = None
        best_score = float("inf")

        for display_id, state in self.display_states.items():
            if display_id in self.active_display_ids:
                continue
            if state.raw_track_id in self.current_raw_ids:
                continue
            gap = frame_idx - state.last_frame
            if gap <= 0 or gap > self.reconnect_window:
                continue
            if state.family != family:
                continue

            ratio = area / state.area
            if ratio < self.min_size_ratio or ratio > self.max_size_ratio:
                continue

            pred_x, pred_y = state.predicted_center(frame_idx)
            dist = math.hypot(cx - pred_x, cy - pred_y)
            allowed = self.reconnect_distance_px + 0.5 * max(width, height, state.width, state.height)
            if dist > allowed:
                continue

            score = dist / max(allowed, 1.0) + 0.01 * gap
            if score < best_score:
                best_id = display_id
                best_score = score
        return best_id


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    (out_dir / "snapshots").mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    if args.line_x is not None:
        use_vertical = True
        line_px = int(args.line_x * frame_width)
    else:
        use_vertical = False
        line_frac = args.line_y if args.line_y is not None else 0.5
        line_px = int(line_frac * frame_height)
    reconnect_distance_px = args.reconnect_distance * math.hypot(frame_width, frame_height)

    import torch
    device_str = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    results_gen = model.track(
        source=args.video,
        tracker=str(args.tracker),
        persist=True,
        stream=True,
        conf=args.conf,
        iou=args.iou,
        device=device_str,
        verbose=False,
    )

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out_video = cv2.VideoWriter(
        str(out_dir / "tracked.mp4"), fourcc, fps, (frame_width, frame_height)
    )

    post = TrackPostProcessor(
        line_px=line_px,
        use_vertical=use_vertical,
        reconnect_window=args.reconnect_window,
        reconnect_distance_px=reconnect_distance_px,
        min_size_ratio=args.min_size_ratio,
        max_size_ratio=args.max_size_ratio,
        predict_missing_crossings=not args.disable_missing_prediction,
    )
    all_frame_records: list[dict[str, Any]] = []

    for frame_idx, result in enumerate(results_gen):
        frame_bgr = result.orig_img.copy()

        if use_vertical:
            cv2.line(frame_bgr, (line_px, 0), (line_px, frame_height), (0, 255, 255), 2)
        else:
            cv2.line(frame_bgr, (0, line_px), (frame_width, line_px), (0, 255, 255), 2)

        frame_detections: list[dict[str, Any]] = []
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            post.begin_frame(ids.tolist())
            xyxy = result.boxes.xyxy.cpu().numpy()
            xywh = result.boxes.xywh.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for track_id, box_xyxy, box_xywh, cls_id, conf in zip(ids, xyxy, xywh, cls_ids, confs):
                cx, cy = float(box_xywh[0]), float(box_xywh[1])
                cls_name = model.names.get(int(cls_id), str(cls_id))
                update = post.update(
                    raw_track_id=int(track_id),
                    class_name=cls_name,
                    cx=cx,
                    cy=cy,
                    width=float(box_xywh[2]),
                    height=float(box_xywh[3]),
                    frame_idx=frame_idx,
                )
                display_id = update.display_id

                x1, y1, x2, y2 = map(int, box_xyxy)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"{cls_name} #{display_id} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame_bgr, (x1, y1 - th - 4), (x1 + tw, y1), (0, 200, 0), -1)
                cv2.putText(frame_bgr, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                frame_detections.append({
                    "track_id": display_id,
                    "class": cls_name,
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                    "conf": round(float(conf), 3),
                    "reconnected": update.reconnected,
                })
        else:
            post.begin_frame()

        post.finish_frame(frame_idx)

        cv2.putText(
            frame_bgr, f"Count: {post.count}",
            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
        )

        out_video.write(frame_bgr)

        frame_detections.sort(key=lambda det: det["track_id"])
        all_frame_records.append({"frame": frame_idx, "detections": frame_detections, "count": post.count})

        if frame_idx >= args.snapshot_start and (frame_idx - args.snapshot_start) % args.snapshot_interval == 0:
            cv2.imwrite(str(out_dir / "snapshots" / f"frame_{frame_idx:05d}.png"), frame_bgr)

        if frame_idx % 30 == 0:
            print(f"frame={frame_idx} count={post.count} active_tracks={len(frame_detections)}", flush=True)

    out_video.release()

    with (out_dir / "frames.json").open("w", encoding="utf-8") as f:
        json.dump(all_frame_records, f, ensure_ascii=False, indent=2)
        f.write("\n")

    summary = {
        "total_crossed": post.count,
        "display_ids_crossed": sorted(post.crossed_display_ids),
        "line_orientation": "vertical" if use_vertical else "horizontal",
        "line_fraction": args.line_x if use_vertical else (args.line_y or 0.5),
        "line_px": line_px,
        "frame_count": len(all_frame_records),
        "tracker": str(args.tracker),
        "conf": args.conf,
        "reconnect_window": args.reconnect_window,
        "reconnect_distance_fraction": args.reconnect_distance,
        "reconnect_distance_px": round(reconnect_distance_px, 1),
        "predict_missing_crossings": not args.disable_missing_prediction,
        "predicted_missing_crossings": post.predicted_crossings,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Done. total_crossed={post.count} video -> {out_dir}/tracked.mp4", flush=True)


if __name__ == "__main__":
    main()
