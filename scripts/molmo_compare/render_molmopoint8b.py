#!/usr/bin/env python
"""Render allenai/MolmoPoint-8B video pointing+tracking onto video (comparison #2).

Unlike Molmo-7B-D (render_molmo7b.py), MolmoPoint-8B ingests the whole video
clip natively and returns a *stable* object_id per tracked point across
frames -- real tracking via one forward pass over the clip, not per-frame
re-detection. Point decoding uses grounding tokens through
model.extract_video_points(); see
https://huggingface.co/allenai/MolmoPoint-8B and
https://github.com/allenai/molmo2/blob/main/MOLMO_POINT_README.md

IMPORTANT -- separate env required: this needs transformers>=4.57.1 plus the
`molmo_utils` package from github.com/allenai/molmo2. That is INCOMPATIBLE
with transformers==4.45.2, which run_camspace.py/--ego_filter requires for
Molmo-7B-D's custom modeling code (newer transformers breaks it, per
requirements_blackwell.txt). Do NOT install this into the minhawor env --
create a fresh conda env for this script.

NOTE: this script was written from the model card / README above (not
executed against real weights in this environment -- no GPU download was
done here). model.extract_video_points()'s return-tuple order is documented
inconsistently across AI2's own pages (object_id-first on the MolmoPoint-8B
card, frame-first on the Molmo2-VideoPoint-4B card) -- the debug print below
shows the raw tuples on first run so you can confirm/fix the unpacking order
in `by_frame` if points land in the wrong place.

Example:
  python scripts/molmo_compare/render_molmopoint8b.py \
      --video_path /path/to/ego1/color.mp4 --out /tmp/molmopoint8b_render.mp4 \
      --max_frames 300
"""
import argparse
import os

os.environ.setdefault("HF_HOME", "/hdd/hf_home")

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL_ID = "allenai/MolmoPoint-8B"

EGO_HANDS_PROMPT = (
    "Track the hands of the person wearing the camera throughout this video "
    "(the hands attached to arms entering the frame from the bottom/sides, "
    "close to the camera). Do not track other people's hands. Give each hand "
    "a consistent id for the whole video."
)

TRACK_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
                (255, 128, 0), (128, 0, 255)]


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
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--prompt", default=EGO_HANDS_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_frames", type=int, default=None,
                     help="also caps the rendered output, not just decoding")
    ap.add_argument("--radius", type=int, default=10)
    args = ap.parse_args()

    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype="auto", device_map="auto")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, padding_side="left")

    messages = [{
        "role": "user",
        "content": [
            dict(type="text", text=args.prompt),
            dict(type="video", video=args.video_path),
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, videos=[args.video_path], padding=True,
                        return_tensors="pt", return_pointing_metadata=True)
    metadata = inputs.pop("metadata")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            **inputs,
            logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=args.max_new_tokens,
        )
    gen = out[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(gen, skip_special_tokens=True)
    print("Raw model output:", generated_text)

    video_points = model.extract_video_points(
        generated_text,
        metadata["token_pooling"],
        metadata["subpatch_mapping"],
        metadata["timestamps"],
        metadata["video_size"],
    )
    print(f"Decoded {len(video_points)} raw point tuples, first 5: {video_points[:5]}")

    # Documented as [object_id, frame_num, x, y] on the MolmoPoint-8B card.
    # If the debug print above shows a different order, fix the unpacking here.
    by_frame = {}
    for object_id, frame_num, x, y in video_points:
        by_frame.setdefault(int(frame_num), []).append((object_id, x, y))

    frames, fps = load_frames(args.video_path, args.max_frames)
    H, W = frames[0].shape[:2]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for i, frame in enumerate(frames):
        for object_id, x, y in by_frame.get(i, []):
            color = color_for(object_id)
            cv2.circle(frame, (int(x), int(y)), args.radius, color, -1)
            cv2.putText(frame, str(object_id), (int(x) + 12, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"MolmoPoint-8B  frame={i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"Wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
