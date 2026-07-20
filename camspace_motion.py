#!/usr/bin/env python
"""Mask-free, multi-GPU camera-space motion estimation.

Two speedups over scripts/scripts_test_video/hawor_video.py:hawor_motion_estimation:

 1. NO hand-mask rendering. hawor_video renders a per-frame MANO mask (pytorch3d)
    for every frame; those masks feed ONLY DROID-SLAM and the neural infiller,
    which the cam-space pipeline never runs. Skipping them removes thousands of
    1080p rasterizations per video for zero change to the cam_space output.

 2. Multi-GPU sharding. The HaWoR motion model runs INDEPENDENT 16-frame temporal
    windows (see HAWOR.inference: `(b t) -> b t`, t=16, no cross-window state), so
    a track can be split on 16-frame boundaries and processed on different GPUs
    with numerically identical results. We flatten (left-hand, right-hand, and long
    tracks) into window-aligned pieces, balance them across GPUs (LPT), and run one
    subprocess per GPU (each sees a single GPU via CUDA_VISIBLE_DEVICES). Left and
    right hands therefore run concurrently instead of one-after-the-other.

Output is drop-in identical to hawor_motion_estimation: cam_space/<idx>/<s>_<e>.json
per piece + tracks_*/frame_chunks_all.npy. Consumed unchanged by
render_cam_noinfiller.cam2world_no_infiller / run_camspace.

Run modes:
  (orchestrator)  imported: run_parallel(args, start, end, seq_folder, gpus)
  (worker)        python camspace_motion.py --worker --shard <pkl> ...   [internal]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
from glob import glob
from collections import defaultdict

import joblib
from natsort import natsorted

HAWOR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HAWOR)

# torch>=2.6 defaults torch.load(weights_only=True) -> rejects the omegaconf config
# pickled in the (trusted) HaWoR checkpoint. Force full loads (same as run_camspace).
import torch as _torch
_ORIG_TORCH_LOAD = _torch.load
def _torch_load_full(*a, **k):
    k["weights_only"] = False
    return _ORIG_TORCH_LOAD(*a, **k)
_torch.load = _torch_load_full

T_WINDOW = 16  # HaWoR temporal window; all splits align to this


# --------------------------------------------------------------------------- #
# CPU: group tracks into left/right hand chunks (mirrors hawor_video.py 66-159)
# --------------------------------------------------------------------------- #
def build_hand_chunks(seq_folder, start_idx, end_idx, imgfiles):
    """Return {0: [(frame_ck, boxes_ck, do_flip), ...], 1: [...]} for left/right."""
    from lib.pipeline.tools import parse_chunks
    from lib.eval_utils.custom_utils import interpolate_bboxes

    tracks = np.load(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy',
                     allow_pickle=True).item()
    tid = np.array([tr for tr in tracks])

    left_trk, right_trk = [], []
    for idx in tid:
        trk = tracks[idx]
        valid = np.array([t['det'] for t in trk])
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        (right_trk if is_right.sum() / len(is_right) >= 0.5 else left_trk).extend(trk)
    left_trk = sorted(left_trk, key=lambda x: x['frame'])
    right_trk = sorted(right_trk, key=lambda x: x['frame'])
    final_tracks = {0: left_trk, 1: right_trk}

    out = defaultdict(list)
    for idx in (0, 1):
        trk = final_tracks[idx]
        valid = np.array([t['det'] for t in trk])
        if valid.sum() < 2:
            continue
        boxes = np.concatenate([t['det_box'] for t in trk])
        nz = np.where(np.any(boxes != 0, axis=1))[0]
        boxes[nz[0]:nz[-1] + 1] = interpolate_bboxes(boxes[nz[0]:nz[-1] + 1])
        valid[nz[0]:nz[-1] + 1] = True
        boxes = boxes[nz[0]:nz[-1] + 1]
        is_right = np.concatenate([t['det_handedness'] for t in trk])[valid]
        frame = np.array([t['frame'] for t in trk])[valid]
        do_flip = (is_right.sum() / len(is_right) < 0.5)  # left hand -> flip

        frame_chunks, boxes_chunks = parse_chunks(frame, boxes, min_len=1)
        for fck, bck in zip(frame_chunks, boxes_chunks):
            out[idx].append((np.asarray(fck), np.asarray(bck), bool(do_flip)))
    return out


def split_on_windows(frame_ck, boxes_ck, target_len):
    """Split one chunk into <=target_len pieces aligned to T_WINDOW boundaries."""
    n = len(frame_ck)
    if n <= target_len:
        return [(frame_ck, boxes_ck)]
    step = max(T_WINDOW, (target_len // T_WINDOW) * T_WINDOW)
    pieces = []
    for s in range(0, n, step):
        e = min(s + step, n)
        pieces.append((frame_ck[s:e], boxes_ck[s:e]))
    return pieces


def plan_pieces(chunks, n_gpus):
    """Flatten hand chunks into window-aligned pieces and LPT-balance across GPUs.

    Returns shards: list (len n_gpus) of lists of work items
    {idx, frame_ck(list[int]), boxes_ck(np), do_flip}.
    """
    total = sum(len(f) for lst in chunks.values() for (f, _, _) in lst)
    target = max(T_WINDOW * 4, ((total // max(1, n_gpus)) // T_WINDOW + 1) * T_WINDOW)

    items = []
    for idx, lst in chunks.items():
        for (fck, bck, do_flip) in lst:
            for (fpiece, bpiece) in split_on_windows(fck, bck, target):
                items.append({"idx": int(idx), "frame_ck": [int(x) for x in fpiece],
                              "boxes_ck": np.asarray(bpiece), "do_flip": bool(do_flip)})
    # longest-processing-time-first onto least-loaded shard
    items.sort(key=lambda w: -len(w["frame_ck"]))
    shards = [[] for _ in range(n_gpus)]
    load = [0] * n_gpus
    for w in items:
        g = int(np.argmin(load))
        shards[g].append(w)
        load[g] += len(w["frame_ck"])
    return shards, load


# --------------------------------------------------------------------------- #
# GPU worker: run one shard's pieces on the single visible GPU (no masks)
# --------------------------------------------------------------------------- #
def load_model(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    from lib.models.hawor import HAWOR
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        model_cfg.defrost()
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
        model_cfg.freeze()
    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
    return model


def run_worker(a):
    import torch
    from hawor.utils.rotation import (angle_axis_to_rotation_matrix,
                                       rotation_matrix_to_angle_axis)

    shard = joblib.load(a.shard)                       # list of work items
    imgfiles = np.array(joblib.load(a.imgfiles))
    img_center = [a.img_cx, a.img_cy]
    amp_dtype = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[a.motion_precision]

    device = torch.device('cuda')
    model = load_model(a.checkpoint).to(device).eval()

    for w in shard:
        idx, do_flip = w["idx"], w["do_flip"]
        frame_ck = np.asarray(w["frame_ck"]); boxes_ck = np.asarray(w["boxes_ck"])
        print(f"[gpu {a.gpu}] idx={idx} frames {frame_ck[0]}..{frame_ck[-1]} "
              f"({len(frame_ck)})", flush=True)
        results = model.inference(imgfiles[frame_ck], boxes_ck, img_focal=a.img_focal,
                                  img_center=img_center, do_flip=do_flip,
                                  batch_size=a.motion_batch, num_workers=a.motion_workers,
                                  amp_dtype=amp_dtype)
        data_out = {
            "init_root_orient": results["pred_rotmat"][None, :, 0],
            "init_hand_pose":   results["pred_rotmat"][None, :, 1:],
            "init_trans":       results["pred_trans"][None, :, 0],
            "init_betas":       results["pred_shape"][None, :],
        }
        init_root = rotation_matrix_to_angle_axis(data_out["init_root_orient"])
        init_hand = rotation_matrix_to_angle_axis(data_out["init_hand_pose"])
        if do_flip:
            init_root[..., 1] *= -1; init_root[..., 2] *= -1
            init_hand[..., 1] *= -1; init_hand[..., 2] *= -1
        data_out["init_root_orient"] = angle_axis_to_rotation_matrix(init_root)
        data_out["init_hand_pose"] = angle_axis_to_rotation_matrix(init_hand)

        out_dir = os.path.join(a.seq_folder, 'cam_space', str(idx))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{frame_ck[0]}_{frame_ck[-1]}.json"), "w") as f:
            json.dump({k: v.tolist() for k, v in data_out.items()}, f, indent=1)


# --------------------------------------------------------------------------- #
# Orchestrator: build pieces, launch one subprocess per GPU, gather
# --------------------------------------------------------------------------- #
def resolve_focal(args, seq_folder):
    img_focal = args.img_focal
    if img_focal is None:
        try:
            img_focal = float(open(os.path.join(seq_folder, 'est_focal.txt')).read())
        except Exception:
            img_focal = 600
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as f:
                f.write(str(img_focal))
            print(f'No focal length provided, use default {img_focal}')
    return float(img_focal)


def run_parallel(args, start_idx, end_idx, seq_folder, gpus):
    """Drop-in for hawor_motion_estimation, sharded across `gpus` (list of ids)."""
    tracks_dir = f'{seq_folder}/tracks_{start_idx}_{end_idx}'
    fca_path = f'{tracks_dir}/frame_chunks_all.npy'
    img_folder = f"{seq_folder}/extracted_images"
    imgfiles = np.array(natsorted(glob(f'{img_folder}/*.jpg')))
    img_focal = resolve_focal(args, seq_folder)

    if os.path.exists(fca_path):
        print("skip hawor motion estimation (cached)")
        return joblib.load(fca_path), img_focal

    chunks = build_hand_chunks(seq_folder, start_idx, end_idx, imgfiles)
    shards, load = plan_pieces(chunks, len(gpus))
    n_pieces = sum(len(s) for s in shards)
    print(f"[camspace_motion] {n_pieces} pieces over {len(gpus)} GPU(s); "
          f"per-GPU frames: {load}")

    # principal point (once): explicit override, else image center
    cx = args.img_cx if getattr(args, "img_cx", None) is not None else None
    cy = args.img_cy if getattr(args, "img_cy", None) is not None else None
    if cx is None or cy is None:
        c = imgfiles_center(imgfiles)
        cx = c[0] if cx is None else cx
        cy = c[1] if cy is None else cy

    # persist imgfiles + shard payloads for the workers
    work_dir = os.path.join(tracks_dir, "_shards")
    os.makedirs(work_dir, exist_ok=True)
    img_pkl = os.path.join(work_dir, "imgfiles.pkl")
    joblib.dump([str(x) for x in imgfiles], img_pkl)

    procs = []
    for g, shard in zip(gpus, shards):
        if not shard:
            continue
        shard_pkl = os.path.join(work_dir, f"shard_gpu{g}.pkl")
        joblib.dump(shard, shard_pkl)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, os.path.join(HAWOR, "camspace_motion.py"), "--worker",
               "--gpu", str(g), "--shard", shard_pkl, "--imgfiles", img_pkl,
               "--seq_folder", seq_folder, "--checkpoint", args.checkpoint,
               "--img_focal", str(img_focal),
               "--img_cx", str(cx), "--img_cy", str(cy),
               "--motion_precision", getattr(args, "motion_precision", "bf16"),
               "--motion_workers", str(getattr(args, "motion_workers", 4)),
               "--motion_batch", str(getattr(args, "motion_batch", 1))]
        procs.append((g, subprocess.Popen(cmd, env=env)))

    failed = [g for (g, p) in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"motion-estimation worker(s) failed on GPU(s) {failed}")

    # frame_chunks_all[idx] = list of piece frame-ranges (order irrelevant; disjoint)
    frame_chunks_all = defaultdict(list)
    for shard in shards:
        for w in shard:
            frame_chunks_all[w["idx"]].append(np.asarray(w["frame_ck"]))
    joblib.dump(frame_chunks_all, fca_path)
    return frame_chunks_all, img_focal


def imgfiles_center(imgfiles):
    import cv2
    img = cv2.imread(imgfiles[0])
    return [img.shape[1] / 2.0, img.shape[0] / 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard"); ap.add_argument("--imgfiles")
    ap.add_argument("--seq_folder"); ap.add_argument("--checkpoint")
    ap.add_argument("--img_focal", type=float)
    ap.add_argument("--img_cx", type=float); ap.add_argument("--img_cy", type=float)
    ap.add_argument("--motion_precision", default="bf16")
    ap.add_argument("--motion_workers", type=int, default=4)
    ap.add_argument("--motion_batch", type=int, default=1)
    a = ap.parse_args()
    if not a.worker:
        ap.error("camspace_motion.py is a worker entry; call run_parallel() to orchestrate")
    os.chdir(HAWOR)
    run_worker(a)


if __name__ == "__main__":
    main()
