"""
Extract evenly-spaced frames from a video for Label Studio annotation.

Usage:
    python scripts/extract_frames.py                        # uses default video
    python scripts/extract_frames.py --video path/to/video  # custom video
    python scripts/extract_frames.py --count 40             # extract 40 frames (default 30)

Output:
    data/annotations/images/<video_stem>_frame_XXXXX.jpg
"""

import argparse
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO = ROOT / "data" / "Object Detection Test.mp4"
OUT_DIR = ROOT / "data" / "annotation_frames"


def extract(video_path: Path, count: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {video_path.name}  |  {total} frames  |  {fps:.1f} fps")

    # Pick `count` evenly-spaced frame indices
    step = max(1, total // count)
    indices = list(range(0, total, step))[:count]

    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        name = f"{video_path.stem}_frame_{idx:05d}.jpg"
        out  = OUT_DIR / name
        cv2.imwrite(str(out), frame)
        saved += 1
        print(f"  saved {out.name}")

    cap.release()
    print(f"\nDone — {saved} frames saved to {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--count", type=int,  default=30)
    args = parser.parse_args()
    extract(args.video, args.count)
