"""Render the CAM-view HaWoR visualization WITHOUT the neural infiller.

Reuses already-computed per-session artifacts (tracks_*/frame_chunks_all.npy,
cam_space/*.json, SLAM/hawor_slam_w_scale_*.npz, extracted_images/) so nothing
heavy (YOLO / DROID-SLAM / Metric3D / HaWoR) re-runs.

Difference vs demo.py: we do the camera->world lift for DETECTED frames only and
SKIP the TransformerModel gap-filling. Frames where a hand was not detected are
left invalid and the hand mesh is hidden (vertices set to NaN) instead of being
hallucinated. Output goes to a distinct dir  vis_cam_noinf_<start>_<end>/  so it
never collides with the infiller result in  vis_cam_<start>_<end>/.

Usage:
  python render_cam_noinfiller.py --seq_folder /hdd/gdm/ptron/2026_07_06/P0013 \
                                  --img_focal 708.52
"""
import argparse
import json
import os
from glob import glob

import joblib
import numpy as np
import torch
from natsort import natsorted

from lib.eval_utils.custom_utils import cam2world_convert, load_slam_cam
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.vis.run_vis2 import run_vis2_on_video_cam

# extra faces closing the MANO wrist hole (same as demo.py)
FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                      [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                      [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                      [120, 108, 78], [78, 108, 79]])


def find_track_dir(seq_folder):
    """Return (tracks_dir, start_idx, end_idx) for the tracks_<s>_<e> folder."""
    cands = sorted(glob(os.path.join(seq_folder, "tracks_*_*")))
    for d in cands:
        if os.path.exists(os.path.join(d, "frame_chunks_all.npy")):
            s, e = os.path.basename(d).replace("tracks_", "").split("_")
            return d, int(s), int(e)
    raise FileNotFoundError(f"no tracks_*/frame_chunks_all.npy under {seq_folder}")


def get_cameras(seq_folder, start_idx, end_idx, n_frames, no_slam):
    """Per-frame camera-to-world (R_c2w, t_c2w).

    For a CAM visualization SLAM is NOT needed: motion estimation already outputs
    poses in the camera frame, and the cam->world->cam round-trip cancels (SLAM
    scale & rotation drop out), leaving K @ (camera-space vertices). So with
    no_slam we use identity R_c2w / zero t_c2w, which makes world == camera frame
    and renders exactly the raw per-frame prediction. SLAM only matters for the
    WORLD view. Loading SLAM here would just reintroduce a transform that cancels.
    """
    if no_slam:
        R = torch.eye(3).unsqueeze(0).repeat(n_frames, 1, 1)
        t = torch.zeros(n_frames, 3)
        return R, t
    fpath = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    _, _, R_c2w_all, t_c2w_all = load_slam_cam(fpath)
    return R_c2w_all, t_c2w_all


def cam2world_no_infiller(frame_chunks_all, R_c2w_all, t_c2w_all, seq_folder, n_frames):
    """Replicate hawor_infiller's cam->world lift, but WITHOUT the filling net.
    With identity cameras (no_slam) this is a pass-through (world == camera)."""
    pred_trans = torch.zeros(2, n_frames, 3)
    pred_rot = torch.zeros(2, n_frames, 3)
    pred_hand_pose = torch.zeros(2, n_frames, 45)
    pred_betas = torch.zeros(2, n_frames, 10)
    pred_valid = torch.zeros(2, n_frames)

    for idx in [0, 1]:  # 0=left, 1=right
        for frame_ck in frame_chunks_all[idx]:
            if len(frame_ck) == 0:
                continue
            pred_path = os.path.join(seq_folder, "cam_space", str(idx),
                                     f"{frame_ck[0]}_{frame_ck[-1]}.json")
            with open(pred_path, "r") as f:
                data_out = {k: torch.tensor(v) for k, v in json.load(f).items()}

            R_c2w = R_c2w_all[frame_ck]
            t_c2w = t_c2w_all[frame_ck]
            data_world = cam2world_convert(R_c2w, t_c2w, data_out,
                                           "right" if idx > 0 else "left")
            pred_trans[[idx], frame_ck] = data_world["init_trans"]
            pred_rot[[idx], frame_ck] = data_world["init_root_orient"]
            pred_hand_pose[[idx], frame_ck] = data_world["init_hand_pose"].flatten(-2)
            pred_betas[[idx], frame_ck] = data_world["init_betas"]
            pred_valid[[idx], frame_ck] = 1

    return pred_trans, pred_rot, pred_hand_pose, pred_betas, (pred_valid > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_folder", required=True, help="session dir with tracks_/cam_space[/SLAM]")
    ap.add_argument("--img_focal", type=float, default=None)
    ap.add_argument("--fx", type=float, default=None, help="focal-x override for projection K (default img_focal)")
    ap.add_argument("--fy", type=float, default=None, help="focal-y override for projection K (default img_focal)")
    ap.add_argument("--cx", type=float, default=None, help="principal-point x for projection K (default W/2)")
    ap.add_argument("--cy", type=float, default=None, help="principal-point y for projection K (default H/2)")
    ap.add_argument("--suffix", default="noinf", help="output dir tag: vis_cam_<suffix>_<s>_<e>")
    ap.add_argument("--no_slam", action="store_true",
                    help="skip SLAM entirely (identity camera); correct for the CAM view")
    ap.add_argument("--median_betas", action="store_true",
                    help="use each hand's median betas (over valid frames) for ALL frames")
    ap.add_argument("--betas_l", default=None, help="explicit 10 comma-sep betas for LEFT hand (overrides)")
    ap.add_argument("--betas_r", default=None, help="explicit 10 comma-sep betas for RIGHT hand (overrides)")
    args = ap.parse_args()

    seq_folder = args.seq_folder.rstrip("/")
    img_folder = os.path.join(seq_folder, "extracted_images")
    imgfiles = natsorted(glob(os.path.join(img_folder, "*.jpg")))
    if not imgfiles:
        raise FileNotFoundError(f"no frames in {img_folder}")
    n_frames = len(imgfiles)

    tracks_dir, start_idx, end_idx = find_track_dir(seq_folder)

    img_focal = args.img_focal
    if img_focal is None:
        # prefer the focal actually used at SLAM time (stored in the npz), then est_focal.txt
        slam_npz = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
        try:
            img_focal = float(np.load(slam_npz, allow_pickle=True)["img_focal"])
        except Exception:
            with open(os.path.join(seq_folder, "est_focal.txt")) as f:
                img_focal = float(f.read())
    frame_chunks_all = joblib.load(os.path.join(tracks_dir, "frame_chunks_all.npy"))

    R_c2w_all, t_c2w_all = get_cameras(seq_folder, start_idx, end_idx, n_frames, args.no_slam)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = \
        cam2world_no_infiller(frame_chunks_all, R_c2w_all, t_c2w_all, seq_folder, n_frames)

    # optional constant-betas override (per hand): explicit values, else median over valid frames
    for idx, ov in [(0, args.betas_l), (1, args.betas_r)]:
        if ov is not None:
            b = torch.tensor([float(x) for x in ov.split(",")][:10]).float()
            pred_betas[idx, :] = b
            print(f"  {'left' if idx==0 else 'right'} betas overridden -> {b.tolist()}")
        elif args.median_betas and pred_valid[idx].any():
            med = pred_betas[idx][pred_valid[idx]].median(0).values
            pred_betas[idx, :] = med
            print(f"  {'left' if idx==0 else 'right'} median betas -> {[round(x,3) for x in med.tolist()]}")

    vis_start, vis_end = 0, pred_trans.shape[1] - 1
    n_valid_l = int(pred_valid[0, vis_start:vis_end].sum())
    n_valid_r = int(pred_valid[1, vis_start:vis_end].sum())
    print(f"[{os.path.basename(seq_folder)}] frames={n_frames} focal={img_focal} "
          f"detected: left={n_valid_l} right={n_valid_r} (rest hidden, no infiller)")

    # faces
    faces = get_mano_faces()
    faces_right = np.concatenate([faces, FACES_NEW], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    # right hand
    pred_glob_r = run_mano(pred_trans[1:2, vis_start:vis_end], pred_rot[1:2, vis_start:vis_end],
                           pred_hand_pose[1:2, vis_start:vis_end], betas=pred_betas[1:2, vis_start:vis_end])
    right_verts = pred_glob_r["vertices"][0].cpu()          # (T, 778, 3)
    # left hand
    pred_glob_l = run_mano_left(pred_trans[0:1, vis_start:vis_end], pred_rot[0:1, vis_start:vis_end],
                                pred_hand_pose[0:1, vis_start:vis_end], betas=pred_betas[0:1, vis_start:vis_end])
    left_verts = pred_glob_l["vertices"][0].cpu()           # (T, 778, 3)

    # HIDE undetected frames (this is the "without infiller" behaviour): NaN verts
    valid_r = pred_valid[1, vis_start:vis_end].bool()
    valid_l = pred_valid[0, vis_start:vis_end].bool()
    right_verts[~valid_r] = float("nan")
    left_verts[~valid_l] = float("nan")

    right_dict = {"vertices": right_verts.unsqueeze(0), "faces": faces_right}
    left_dict = {"vertices": left_verts.unsqueeze(0), "faces": faces_left}

    # world->viewer axis flip + recompute w2c (identical to demo.py). Reuse the
    # cameras loaded above (identity when --no_slam, in which case this reduces
    # to projecting the raw camera-space vertices with K).
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_all = torch.einsum("ij,njk->nik", R_x, R_c2w_all)
    t_c2w_all = torch.einsum("ij,nj->ni", R_x, t_c2w_all)
    R_w2c_all = R_c2w_all.transpose(-1, -2)
    t_w2c_all = -torch.einsum("bij,bj->bi", R_w2c_all, t_c2w_all)
    left_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"])
    right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"])

    fx = args.fx if args.fx is not None else img_focal
    fy = args.fy if args.fy is not None else img_focal
    image_names = imgfiles[vis_start:vis_end]
    output_pth = os.path.join(seq_folder, f"vis_cam_{args.suffix}_{vis_start}_{vis_end}")
    os.makedirs(output_pth, exist_ok=True)
    print(f"  -> rendering cam vis (no infiller) fx={fx} fy={fy} cx={args.cx} cy={args.cy} to {output_pth}")
    run_vis2_on_video_cam(left_dict, right_dict, output_pth, img_focal, image_names,
                          R_w2c=R_w2c_all[vis_start:vis_end], t_w2c=t_w2c_all[vis_start:vis_end],
                          interactive=False, fx=fx, fy=fy, cx=args.cx, cy=args.cy)
    print("  finish")


if __name__ == "__main__":
    main()
