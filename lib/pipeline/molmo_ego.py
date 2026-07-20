"""Molmo-based egocentric-hand pointing, used to assist YOLO hand detection.

In egocentric video, YOLO detects every hand in view (including bystanders').
Molmo-7B-D can *point* at things given a text prompt; we prompt it to point at
the camera-wearer's own hands, then keep only the YOLO detections that a Molmo
point falls inside. This disambiguates the wearer's hands from other people's.

Coordinates: Molmo emits points in XML with x,y normalized to 0-100 (percent of
image width/height). We convert to pixels with the actual image size.
"""
import re
import numpy as np
from PIL import Image

MOLMO_ID = "allenai/Molmo-7B-D-0924"

# default prompt: the wearer's hands are the ones attached to arms entering the
# frame from the bottom / close to the camera.
EGO_PROMPT = ("Point at the hands of the person wearing the camera "
              "(the hands attached to the arms coming from the bottom of the image). "
              "Do not point at other people's hands.")

# <point x="35.5" y="60.2" alt="...">...</point>
_PT_RE = re.compile(r'<point\s+x="([0-9.]+)"\s+y="([0-9.]+)"', re.I)
# <points x1=".." y1=".." x2=".." y2=".." ...>  (multiple)
_PTS_X_RE = re.compile(r'x(\d+)="([0-9.]+)"', re.I)
_PTS_Y_RE = re.compile(r'y(\d+)="([0-9.]+)"', re.I)


def load_molmo(device_map="cuda", dtype=None):
    """Load Molmo-7B-D. Defaults to bf16 on a single CUDA device: fp32 (~28GB)
    overflows a 24GB card and device_map='auto' silently offloads layers to CPU
    (very slow). bf16 (~15GB) fits fully and is much faster."""
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    if dtype is None:
        dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        MOLMO_ID, trust_remote_code=True, torch_dtype=dtype, device_map=device_map)
    model = AutoModelForCausalLM.from_pretrained(
        MOLMO_ID, trust_remote_code=True, torch_dtype=dtype, device_map=device_map)
    return model, processor


def parse_points(text, img_w, img_h):
    """Parse Molmo point XML into a list of (x_px, y_px). Handles both the single
    <point .../> form and the multi <points x1.. y1.. x2.. y2../> form."""
    pts = []
    # multi form
    for m in re.finditer(r'<points\b([^>]*)>', text, re.I):
        attrs = m.group(1)
        xs = {int(i): float(v) for i, v in _PTS_X_RE.findall(attrs)}
        ys = {int(i): float(v) for i, v in _PTS_Y_RE.findall(attrs)}
        for i in sorted(xs):
            if i in ys:
                pts.append((xs[i] / 100.0 * img_w, ys[i] / 100.0 * img_h))
    # single form
    for x, y in _PT_RE.findall(text):
        pts.append((float(x) / 100.0 * img_w, float(y) / 100.0 * img_h))
    return pts


def _process(processor, images, prompt, max_crops):
    """processor.process(), optionally capping the high-res crop count.

    Molmo tiles each image into up to `max_crops` (default 12) 336x336 crops; the
    ViT then encodes every crop, which dominates runtime and scales linearly with
    batch. Lowering max_crops (e.g. 2-4) is the main speed lever — trades spatial
    resolution for speed. The default 12 is always injected by process(), so it
    must be overridden via images_kwargs, not the image_processor attribute."""
    if max_crops is None:
        return processor.process(images=images, text=prompt)
    return processor.process(images=images, text=prompt,
                             images_kwargs={"max_crops": int(max_crops)})


def point_ego_hands(model, processor, pil_image, prompt=EGO_PROMPT, max_new_tokens=200,
                    max_crops=None):
    """Return (points_px, raw_text) where points_px is a list of (x,y) in pixels."""
    import torch
    from transformers import GenerationConfig
    inputs = _process(processor, [pil_image], prompt, max_crops)
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
    # match model dtype (e.g. bf16) for float tensors (images/masks); keep int ids
    inputs = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v)
              for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    gen = out[0, inputs['input_ids'].size(1):]
    text = processor.tokenizer.decode(gen, skip_special_tokens=True)
    W, H = pil_image.size
    return parse_points(text, W, H), text


def point_ego_hands_batch(model, processor, pil_images, prompt=EGO_PROMPT, max_new_tokens=200,
                          max_crops=None):
    """Batched pointing over several frames in one forward pass.

    For a fixed prompt + image resolution Molmo produces identical tensor shapes
    per frame (same crop grid, same token count), so they stack along the batch
    dim with no padding. Falls back to per-image if shapes ever differ.
    Returns a list (one per input image) of point lists [(x_px, y_px), ...].

    NB: batching does NOT speed up the ViT (it saturates on crops even at batch 1),
    so per-batch time scales ~linearly with batch size. The real lever is max_crops.
    """
    import torch
    from transformers import GenerationConfig
    if len(pil_images) == 0:
        return []
    per = [_process(processor, [im], prompt, max_crops) for im in pil_images]
    ref = {k: v.shape for k, v in per[0].items()}
    stackable = all(set(p) == set(ref) and all(p[k].shape == ref[k] for k in p) for p in per)
    if not stackable:  # safety fallback
        return [point_ego_hands(model, processor, im, prompt, max_new_tokens, max_crops)[0]
                for im in pil_images]

    batch = {k: torch.stack([p[k] for p in per], 0).to(model.device) for k in per[0]}
    batch = {k: (v.to(model.dtype) if torch.is_floating_point(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = model.generate_from_batch(
            batch,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer,
        )
    L = batch['input_ids'].size(1)
    results = []
    for i, im in enumerate(pil_images):
        text = processor.tokenizer.decode(out[i, L:], skip_special_tokens=True)
        W, H = im.size
        results.append(parse_points(text, W, H))
    return results


def point_in_box(pt, box):
    """box = [x1,y1,x2,y2,(conf)]; pt = (x,y)."""
    x, y = pt
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]
