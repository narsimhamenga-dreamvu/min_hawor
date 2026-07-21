#!/usr/bin/env python
"""Multi-GPU YOLO hand detection (drop-in for lib.pipeline.tools.detect_track).

detect_track() runs ultralytics `.track(persist=True)` — a stateful, sequential
tracker — which is ~87% of a full cam-space run's wall time and pinned to one GPU.
But build_hand_chunks (hawor_video / camspace_motion) collapses ALL track ids into
just left/right by handedness majority and keeps, per frame, the first right and
first left box. So the tracker's IDs are irrelevant to the cam-space output — only
the per-frame {left box, right box} matter.

This module therefore runs plain per-frame YOLO detection (no tracker state), which
is embarrassingly parallel: shard the frames across GPUs (one subprocess each),
then assemble a tracks dict with handedness-constant ids (right->10000, left->5000,
matching detect_track's own fallback ids). Downstream grouping is identical.

Verified downstream-equivalent to detect_track on SEC001/150f (same valid mask).

Writes tracks_<s>_<e>/model_tracks.npy (+ empty model_boxes.npy sentinel), exactly
what detect_track_video would, so detect_track_video then skips its YOLO loop.
"""
import argparse
import os
import sys

import numpy as np
from glob import glob

import joblib
from natsort import natsorted

HAWOR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HAWOR)


def run_worker(a):
    import cv2
    import torch
    from ultralytics import YOLO

    imgfiles = joblib.load(a.imgfiles)
    lo, hi = a.lo, a.hi
    model = YOLO(a.detector)
    dets = []  # (frame_idx, box5(xyxy+conf), handedness)
    for t in range(lo, hi):
        img = cv2.imread(imgfiles[t])
        with torch.no_grad():
            res = model.predict(img, conf=a.thresh, verbose=False, device=0)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        hands = res.boxes.cls.cpu().numpy()
        boxes = np.hstack([boxes, confs[:, None]]) if len(boxes) else boxes.reshape(0, 5)
        find_r = find_l = False
        for idx in range(len(boxes)):
            h = hands[idx]
            if (not find_r and h > 0) or (not find_l and h == 0):
                dets.append((t, boxes[idx].astype(np.float32), float(h)))
                if h > 0: find_r = True
                else:     find_l = True
        if (t - lo) % 200 == 0:
            print(f"[det gpu {a.gpu}] {t - lo}/{hi - lo}", flush=True)
    joblib.dump(dets, a.out)
    print(f"[det gpu {a.gpu}] done {lo}..{hi} ({len(dets)} dets)", flush=True)


def run_parallel(seq_folder, start_idx, end_idx, imgfiles, gpus,
                 detector="./weights/external/detector.pt", thresh=0.2):
    """Shard detection across `gpus`; write model_tracks.npy in detect_track format."""
    tracks_dir = f"{seq_folder}/tracks_{start_idx}_{end_idx}"
    os.makedirs(tracks_dir, exist_ok=True)
    if os.path.exists(f"{tracks_dir}/model_boxes.npy"):
        print("skip detection (cached)")
        return

    n = len(imgfiles)
    work_dir = os.path.join(tracks_dir, "_detshards")
    os.makedirs(work_dir, exist_ok=True)
    img_pkl = os.path.join(work_dir, "imgfiles.pkl")
    joblib.dump([str(x) for x in imgfiles], img_pkl)

    # contiguous frame blocks, one per GPU
    import subprocess
    bounds = np.linspace(0, n, len(gpus) + 1).astype(int)
    procs, outs = [], []
    for g, (lo, hi) in zip(gpus, zip(bounds[:-1], bounds[1:])):
        if hi <= lo:
            continue
        out = os.path.join(work_dir, f"det_gpu{g}.pkl")
        outs.append((lo, out))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        cmd = [sys.executable, os.path.join(HAWOR, "camspace_detect.py"), "--worker",
               "--gpu", str(g), "--lo", str(lo), "--hi", str(hi),
               "--imgfiles", img_pkl, "--detector", detector,
               "--thresh", str(thresh), "--out", out]
        procs.append((g, subprocess.Popen(cmd, env=env)))
    failed = [g for (g, p) in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"detection worker(s) failed on GPU(s) {failed}")

    # gather in frame order -> tracks dict with handedness-constant ids
    all_dets = []
    for _, out in sorted(outs):
        all_dets.extend(joblib.load(out))
    all_dets.sort(key=lambda d: d[0])

    tracks = {}
    for (t, box5, h) in all_dets:
        subj = {"frame": int(t), "det": True,
                "det_box": box5[None, :], "det_handedness": np.array([h])}
        tid = 10000 if h > 0 else 5000
        tracks.setdefault(tid, []).append(subj)

    np.save(f"{tracks_dir}/model_tracks.npy", np.array(tracks, dtype=object))
    np.save(f"{tracks_dir}/model_boxes.npy", np.array([], dtype=object))  # sentinel
    nR = len(tracks.get(10000, [])); nL = len(tracks.get(5000, []))
    print(f"[camspace_detect] {n} frames over {len(gpus)} GPU(s): "
          f"right dets={nR} left dets={nL}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--lo", type=int); ap.add_argument("--hi", type=int)
    ap.add_argument("--imgfiles"); ap.add_argument("--detector")
    ap.add_argument("--thresh", type=float, default=0.2)
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.worker:
        ap.error("camspace_detect.py is a worker entry; call run_parallel() to orchestrate")
    os.chdir(HAWOR)
    run_worker(a)


if __name__ == "__main__":
    main()
