#!/usr/bin/env python
"""Render Molmo-7B-D-0924 ego-hand pointing onto video (model comparison #1).

Molmo-7B-D has no native video/tracking mode: it points at one still image at
a time. This re-runs it every --sample_every frames and holds the last point(s)
in between -- the same scheme lib/pipeline/tools.detect_track_ego uses for the
production --ego_filter -- and draws each returned point as a dot, so you can
see exactly what the production ego-filter is keying its decision on.

Compare the output against render_molmopoint8b.py (MolmoPoint-8B, which
natively tracks points across the whole clip with a stable id per hand).

Env: same as run_camspace.py -- transformers==4.45.2 (requirements_blackwell.txt).
Molmo-7B-D auto-downloads to the HF cache on first run (~15GB in bf16).

Example:
  python scripts/molmo_compare/render_molmo7b.py \
      --video_path /path/to/ego1/color.mp4 --out /tmp/molmo7b_render.mp4 \
      --sample_every 5 --max_frames 300
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HOME", "/hdd/hf_home")

import cv2
from PIL import Image

HAWOR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, HAWOR)

from lib.pipeline.molmo_ego import load_molmo, point_ego_hands, EGO_PROMPT

POINT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--prompt", default=EGO_PROMPT)
    ap.add_argument("--sample_every", type=int, default=5,
                     help="run Molmo every N frames; hold the last point(s) in between")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--max_crops", type=int, default=None,
                     help="cap Molmo's high-res crop count (speed/quality tradeoff)")
    ap.add_argument("--radius", type=int, default=10)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    print(f"Loading Molmo-7B-D-0924 ...")
    model, processor = load_molmo()

    held_points, held_text = [], ""
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break

        if i % args.sample_every == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            held_points, held_text = point_ego_hands(
                model, processor, pil, prompt=args.prompt, max_crops=args.max_crops)
            print(f"frame {i}: {len(held_points)} point(s) -> {held_text!r}")

        for j, (x, y) in enumerate(held_points):
            color = POINT_COLORS[j % len(POINT_COLORS)]
            cv2.circle(frame, (int(x), int(y)), args.radius, color, -1)
            cv2.circle(frame, (int(x), int(y)), args.radius + 2, (255, 255, 255), 2)
        cv2.putText(frame, f"Molmo-7B-D  frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    print(f"Wrote {args.out} ({i} frames)")


if __name__ == "__main__":
    main()
