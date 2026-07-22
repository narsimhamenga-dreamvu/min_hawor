#!/usr/bin/env python
"""Camera-space hand trajectories — minimal single-video runner.

Runs ONLY the stages needed for per-view cam-space MANO hands, nothing else:

    build frames  ->  detect+track (YOLO [+ optional Molmo ego filter])
                  ->  motion estimation (camera-space MANO)
                  ->  save fusion-ready npz   [-> optional cam-view vis]

Intentionally SKIPPED: DROID-SLAM, the neural infiller, and the world-space vis
(none are needed for camera-space poses; see run_mv1_noinfiller.py). So this
runner has no dependency on thirdparty/DROID-SLAM or thirdparty/Metric3D.

Input — either:
  --src <session_dir>   RealSense-style capture: uses color_frames/*.{jpg,png} in
                        frames_index.csv order if present (else color.mp4), and
                        auto-reads intrinsics.json -> focal / cx / cy.
  --video_path <mp4>    any video (GoPro GX*.MP4, DJI *.MP4, ...); pass --img_focal.

Output (in <out>/camspace_hands.npz), per hand  (index 0 = left, 1 = right):
  trans        (N,3)   root translation in THIS camera's frame
  root_orient  (N,3)   root/wrist global orientation (axis-angle), camera frame
  hand_pose    (N,45)  finger articulation (wrist-relative -> view-independent)
  betas        (N,10)  MANO shape
  valid        (N,)    bool, True where the hand was detected this frame
plus intrinsics {fx,fy,cx,cy,width,height,focal} and frame_index (N,) mapping
each output frame back to its source frame id (for cross-camera time alignment).

This npz is the unit the multi-view fusion consumes: take trans/root_orient from
the ego view, and hand_pose/betas from whichever view is best per frame/hand.

Example:
  # ego (RealSense, auto intrinsics), with Molmo ego-hand filtering
  python run_camspace.py --src /mnt/.../ptron/2026_07_06/ego1/P0042 \
      --out /hdd/ptron_camspace/P0042/ego1 --ego_filter
  # exo (GoPro, explicit focal), no Molmo
  python run_camspace.py --video_path /mnt/.../exo1/GX010181.MP4 \
      --out /hdd/ptron_camspace/P0042/exo1 --img_focal 900
"""
import argparse
import json
import os
import sys

import numpy as np
from glob import glob
from natsort import natsorted

HAWOR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HAWOR)

# torch >= 2.6 flipped torch.load's default to weights_only=True, which rejects
# the omegaconf config objects pickled into the (trusted) HaWoR checkpoint
# (UnpicklingError: Unsupported global omegaconf.dictconfig.DictConfig). Force a
# full load process-wide. Safe here: all checkpoints loaded are our own weights.
import torch as _torch
_ORIG_TORCH_LOAD = _torch.load
def _torch_load_full(*a, **k):
    k["weights_only"] = False
    return _ORIG_TORCH_LOAD(*a, **k)
_torch.load = _torch_load_full


def read_intrinsics(session_dir):
    """Return (focal, fx, fy, cx, cy, W, H) from a capture's intrinsics.json, or None."""
    p = os.path.join(session_dir, "intrinsics.json")
    if not os.path.isfile(p):
        return None
    d = json.load(open(p))
    ci = d.get("color_intrinsics", d)
    fx, fy = float(ci["fx"]), float(ci["fy"])
    return (0.5 * (fx + fy), fx, fy,
            float(ci["ppx"]), float(ci["ppy"]),
            int(ci["width"]), int(ci["height"]))


def build_frames(src_or_video, img_folder, max_frames=0):
    """Populate img_folder with sequential 000000.jpg... Returns list of source frame ids.

    Priority: color_frames/*.{jpg,png} (in frames_index.csv order) > a video file (ffmpeg).
    max_frames > 0 caps the number of frames built (for quick test/clip runs).
    """
    os.makedirs(img_folder, exist_ok=True)
    existing = natsorted(glob(os.path.join(img_folder, "*.jpg")))
    if existing:
        print(f"[frames] {len(existing)} already present, skip")
        # best-effort frame ids: 0..N-1 (real ids only known at build time)
        return list(range(len(existing)))

    color_frames = os.path.join(src_or_video, "color_frames") if os.path.isdir(src_or_video) else None
    idx_csv = os.path.join(src_or_video, "frames_index.csv") if os.path.isdir(src_or_video) else None

    if color_frames and os.path.isdir(color_frames) and idx_csv and os.path.isfile(idx_csv):
        import csv
        rows = sorted(csv.DictReader(open(idx_csv)), key=lambda r: int(r["frame_index"]))
        frame_ids, seq, missing = [], 0, 0
        for r in rows:
            if max_frames and seq >= max_frames:
                break
            fi = int(r["frame_index"])
            src_frame = None
            for cand_ext in (".jpg", ".jpeg", ".png"):
                cand = os.path.join(color_frames, f"{fi:06d}{cand_ext}")
                if os.path.exists(cand):
                    src_frame = cand
                    break
            if src_frame is None:
                missing += 1
                continue
            os.symlink(src_frame, os.path.join(img_folder, f"{seq:06d}.jpg"))
            frame_ids.append(fi)
            seq += 1
        print(f"[frames] linked {seq} color_frames ({missing} missing) in index order"
              + (f" [capped at {max_frames}]" if max_frames else ""))
        if seq == 0:
            raise RuntimeError("no color_frames matched frames_index.csv")
        return frame_ids

    # fall back to a video file (either <src>/color.mp4 or an explicit --video_path)
    video = src_or_video
    if os.path.isdir(src_or_video):
        cands = ([os.path.join(src_or_video, "color.mp4")]
                 + sorted(glob(os.path.join(src_or_video, "*.MP4")))
                 + sorted(glob(os.path.join(src_or_video, "*.mp4"))))
        video = next((c for c in cands if os.path.isfile(c)), None)
        if video is None:
            raise FileNotFoundError(f"no color_frames/ and no video in {src_or_video}")
    print(f"[frames] extracting {video} @ fps=30 ...")
    import subprocess
    cmd = ["ffmpeg", "-i", video, "-vf", "fps=30", "-start_number", "0"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += [os.path.join(img_folder, "%06d.jpg")]
    subprocess.run(cmd, check=True)
    n = len(natsorted(glob(os.path.join(img_folder, "*.jpg"))))
    print(f"[frames] extracted {n} frames")
    return list(range(n))


def main():
    ap = argparse.ArgumentParser(description="cam-space hand trajectories (single video)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--src", help="RealSense-style session dir (color_frames/ or color.mp4 + intrinsics.json)")
    src.add_argument("--video_path", help="a single video file (GoPro/DJI/...)")
    ap.add_argument("--out", required=True, help="output session dir")
    ap.add_argument("--img_focal", type=float, default=None, help="override; else intrinsics.json, else est (600)")
    ap.add_argument("--img_cx", type=float, default=None)
    ap.add_argument("--img_cy", type=float, default=None)
    ap.add_argument("--ego_filter", action="store_true", help="Molmo ego-hand filtering (use for the EGO view)")
    ap.add_argument("--molmo_sample_every", type=int, default=30)
    ap.add_argument("--molmo_min_votes", type=int, default=2)
    ap.add_argument("--molmo_batch", type=int, default=1)
    ap.add_argument("--motion_workers", type=int, default=4)
    ap.add_argument("--motion_batch", type=int, default=1)
    ap.add_argument("--motion_precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids for parallel, mask-free motion estimation "
                         "(e.g. '0,1,2,3'). Splits left/right hands + long tracks on 16-frame "
                         "window boundaries across GPUs. Default: single-GPU legacy path.")
    ap.add_argument("--max_frames", type=int, default=0, help="cap frames built (0=all); for quick test/clip runs")
    ap.add_argument("--checkpoint", default="./weights/hawor/checkpoints/hawor.ckpt")
    ap.add_argument("--vis", action="store_true", help="also render the cam-view mp4 (subprocess)")
    ap.add_argument("--median_betas", action="store_true", help="vis with per-hand median betas")
    a = ap.parse_args()

    os.chdir(HAWOR)  # relative weight paths (./weights/...) resolve here
    seq_folder = os.path.abspath(a.out).rstrip("/")
    os.makedirs(seq_folder, exist_ok=True)
    img_folder = os.path.join(seq_folder, "extracted_images")

    # ---- frames ----
    source = a.src if a.src else a.video_path
    frame_ids = build_frames(source, img_folder, max_frames=a.max_frames)

    # ---- intrinsics ----
    focal, fx, fy, cx, cy, W, H = a.img_focal, a.img_focal, a.img_focal, a.img_cx, a.img_cy, None, None
    if a.src:
        intr = read_intrinsics(a.src)
        if intr:
            f0, fx0, fy0, cx0, cy0, W, H = intr
            focal = a.img_focal if a.img_focal is not None else f0
            fx = fx0 if a.img_focal is None else a.img_focal
            fy = fy0 if a.img_focal is None else a.img_focal
            cx = a.img_cx if a.img_cx is not None else cx0
            cy = a.img_cy if a.img_cy is not None else cy0
            print(f"[intrinsics] focal={focal:.2f} fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} {W}x{H}")
    if focal is None:
        print("[intrinsics] no focal given and no intrinsics.json -> motion-est default (600)")

    from types import SimpleNamespace
    args = SimpleNamespace(
        video_path=seq_folder + ".mp4",   # basename -> seq_folder (frames already built)
        img_focal=focal, img_cx=cx, img_cy=cy,
        input_type="file", checkpoint=a.checkpoint,
        ego_filter=a.ego_filter,
        molmo_sample_every=a.molmo_sample_every,
        molmo_min_votes=a.molmo_min_votes,
        molmo_batch=a.molmo_batch,
        motion_workers=a.motion_workers,
        motion_batch=a.motion_batch,
        motion_precision=a.motion_precision,
    )
    # detect_track_video derives seq_folder from video_path basename; make it match ours
    assert os.path.join(os.path.dirname(args.video_path),
                        os.path.basename(args.video_path).split(".")[0]) == seq_folder

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from render_cam_noinfiller import cam2world_no_infiller, get_cameras

    # ---- detect + track ----
    print(f"[detect+track] ego_filter={a.ego_filter}")
    # parallel YOLO detection (non-ego only): pre-write tracks so detect_track_video
    # skips its single-GPU tracker loop. Molmo ego-filter stays sequential.
    if a.gpus and not a.ego_filter:
        gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
        _imgfiles = natsorted(glob(os.path.join(img_folder, "*.jpg")))
        import camspace_detect
        print(f"[detect] parallel YOLO on GPUs {gpus}")
        camspace_detect.run_parallel(seq_folder, 0, len(_imgfiles), _imgfiles, gpus,
                                     detector="./weights/external/detector.pt")
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    n_frames = len(imgfiles)

    # ---- motion estimation (camera-space MANO) ----
    if a.gpus:
        gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
        print(f"[motion estimation] parallel + mask-free on GPUs {gpus}")
        import camspace_motion
        frame_chunks_all, img_focal = camspace_motion.run_parallel(args, start_idx, end_idx, seq_folder, gpus)
    else:
        print("[motion estimation] single-GPU (legacy path w/ mask render) ...")
        from scripts.scripts_test_video.hawor_video import hawor_motion_estimation
        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
    if focal is None:
        focal = fx = fy = float(img_focal)

    # ---- gather camera-space params (identity cameras == raw camera frame) ----
    R_c2w, t_c2w = get_cameras(seq_folder, start_idx, end_idx, n_frames, no_slam=True)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        cam2world_no_infiller(frame_chunks_all, R_c2w, t_c2w, seq_folder, n_frames)

    if cx is None:
        cx = (W / 2.0) if W else 0.0
    if cy is None:
        cy = (H / 2.0) if H else 0.0
    frame_index = np.asarray(frame_ids[:n_frames] if len(frame_ids) >= n_frames
                             else list(range(n_frames)), dtype=np.int64)

    out_npz = os.path.join(seq_folder, "camspace_hands.npz")
    np.savez(
        out_npz,
        trans=pred_trans.cpu().numpy(),            # (2, N, 3)
        root_orient=pred_rot.cpu().numpy(),        # (2, N, 3)
        hand_pose=pred_hand_pose.cpu().numpy(),    # (2, N, 45)
        betas=pred_betas.cpu().numpy(),            # (2, N, 10)
        valid=pred_valid.cpu().numpy(),            # (2, N) bool
        frame_index=frame_index,                   # (N,)
        focal=np.float32(focal), fx=np.float32(fx), fy=np.float32(fy),
        cx=np.float32(cx), cy=np.float32(cy),
        width=np.int64(W or 0), height=np.int64(H or 0),
        hand_order=np.array(["left", "right"]),
    )
    nl = int(pred_valid[0].sum())
    nr = int(pred_valid[1].sum())
    print(f"[done] {out_npz}  frames={n_frames} valid: left={nl} right={nr}")

    # ---- optional cam-view mp4 (subprocess to isolate the GL context) ----
    if a.vis:
        import subprocess
        cmd = [sys.executable, "render_cam_noinfiller.py", "--seq_folder", seq_folder,
               "--no_slam", "--suffix", "camspace"]
        if focal is not None:
            cmd += ["--img_focal", str(focal), "--fx", str(fx), "--fy", str(fy),
                    "--cx", str(cx), "--cy", str(cy)]
        if a.median_betas:
            cmd += ["--median_betas"]
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen", PYOPENGL_PLATFORM="egl")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
