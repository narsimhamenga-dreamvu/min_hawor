#!/usr/bin/env python
"""Render allenai/Molmo2-4B (general, NOT the VideoPoint-specialized checkpoint)
video *tracking* onto video (comparison #3, vs render_molmo7b.py and
render_molmo2_videopoint4b.py).

Molmo2-VideoPoint-4B (render_molmo2_videopoint4b.py) turned out to only ever
emit a single-frame <points> block, regardless of prompt (verified: 5 prompt
variants, all one timestamp). The general Molmo2-4B checkpoint, using the
"Tracking Video QA" recipe from its own model card, DOES emit multi-timestamp
<tracks .../> with a persistent id per point -- confirmed on this repo's
SEC001 test clip: 2 hand ids followed across 11 timestamps of a 5s clip.

Two things were required to get this working (neither obvious from the docs):
  1. Use the general Molmo2-4B checkpoint, not Molmo2-VideoPoint-4B.
  2. Pass max_fps=8 (or similar) on the video input -- the model's default
     frame sampling is too sparse for tracking. NB: max_fps=8 on a 1920x1080
     clip OOM'd a 24GB L4 when the model loaded in fp32 (dtype="auto"
     resolved to fp32 here); forcing dtype=torch.bfloat16 explicitly fixed
     it, and this script also defaults max_fps down to 4 for headroom.

See https://huggingface.co/allenai/Molmo2-4B (Tracking Video QA section).

Env: same as render_molmo2_videopoint4b.py -- transformers==4.57.1 + molmo_utils,
torch>=2.6 (needed by Molmo2's masking code; torch==2.5.1 is the newest on the
cu121 wheel index, so this env uses the cu124 index instead -- see
scripts/molmo_compare/README.md).

Example:
  python scripts/molmo_compare/render_molmo2_4b_track.py \
      --video_path /path/to/clip.mp4 --out /tmp/molmo2_4b_track_render.mp4
"""
import argparse
import os
import re

os.environ.setdefault("HF_HOME", "/hdd/hf_home")

import cv2
import torch
from molmo_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "allenai/Molmo2-4B"

EGO_HANDS_PROMPT = "Track the back of the palm of each hand of the person wearing the camera."

TRACK_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (128, 0, 255)]

# --- verbatim from the model card's "Tracking Video QA" quick-start ---
COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def _points_from_num_str(text, image_w, image_h):
    for points in POINTS_REGEX.finditer(text):
        ix, x, y = points.group(1), points.group(2), points.group(3)
        x, y = float(x) / 1000 * image_w, float(y) / 1000 * image_h
        if 0 <= x <= image_w and 0 <= y <= image_h:
            yield ix, x, y


def extract_video_points(text, image_w, image_h, extract_ids=True):
    """Flattened list of (frame_id_seconds, id, x, y) from model output text."""
    all_points = []
    for coord in COORD_REGEX.finditer(text):
        for point_grp in FRAME_REGEX.finditer(coord.group(1)):
            frame_id = float(point_grp.group(1))
            for idx, x, y in _points_from_num_str(point_grp.group(2), image_w, image_h):
                if extract_ids:
                    all_points.append((frame_id, idx, x, y))
                else:
                    all_points.append((frame_id, x, y))
    return all_points
# --- end verbatim ---


def color_for(object_id):
    return TRACK_COLORS[hash(str(object_id)) % len(TRACK_COLORS)]


def load_frames(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default=EGO_HANDS_PROMPT)
    ap.add_argument("--max_fps", type=float, default=4,
                     help="video sampling rate fed to the model; higher = denser "
                          "tracks but more GPU memory (8 OOM'd a 24GB L4 at 1920x1080)")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_frames", type=int, default=None,
                     help="caps the rendered output only; the model still sees the whole clip")
    ap.add_argument("--radius", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto")

    messages = [{
        "role": "user",
        "content": [
            dict(type="text", text=args.prompt),
            dict(type="video", video=args.video_path, max_fps=args.max_fps),
        ],
    }]
    _, videos, video_kwargs = process_vision_info(messages)
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(videos=videos, video_metadata=video_metadatas, text=text,
                        padding=True, return_tensors="pt", **video_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Raw model output:", generated_text)

    W, H = video_metadatas[0]["width"], video_metadatas[0]["height"]
    video_points = extract_video_points(generated_text, image_w=W, image_h=H, extract_ids=True)
    print(f"Decoded {len(video_points)} point(s), timestamps: {sorted(set(p[0] for p in video_points))}")

    frames, fps = load_frames(args.video_path, args.max_frames)

    # timestamps from the model are in seconds -> nearest source frame index, then
    # held forward (like the Molmo-7B-D render) until the next timestamp.
    by_frame = {}
    for t_sec, object_id, x, y in video_points:
        frame_idx = int(round(t_sec * fps))
        by_frame.setdefault(frame_idx, []).append((object_id, x, y))
    sample_idxs = sorted(by_frame)

    Hf, Wf = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (Wf, Hf))

    held = []
    for i, frame in enumerate(frames):
        while sample_idxs and sample_idxs[0] <= i:
            held = by_frame[sample_idxs.pop(0)]
        for object_id, x, y in held:
            color = color_for(object_id)
            cv2.circle(frame, (int(x), int(y)), args.radius, color, -1)
            cv2.putText(frame, str(object_id), (int(x) + 12, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Molmo2-4B track  frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"Wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
