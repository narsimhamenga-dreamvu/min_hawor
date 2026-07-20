# min_hawor — camera-space hand trajectories (minimal)

A stripped-down HaWoR that runs **only** the per-view camera-space hand stages:

```
build frames -> detect+track (YOLO [+ Molmo ego filter]) -> motion estimation
             -> camspace_hands.npz  [-> optional cam-view mp4]
```

Everything not on that path is removed: **DROID-SLAM, the neural infiller, the
world-space visualization, all of `thirdparty/`, and every experimental script.**
Built for an **8× RTX PRO 6000 Blackwell (sm_120)** server (CUDA 12.8).

## Layout
```
run_camspace.py            single-video runner  -> camspace_hands.npz
run_ptron_session.sh       fan out one ptron session's 7 views across GPUs 0..6
render_cam_noinfiller.py   optional cam-view mp4 (used by --vis)
build_frames_from_index.py RealSense color_frames -> extracted_images
requirements_blackwell.txt cu128 / torch 2.8 deps (see below)
hawor/ lib/ infiller/ scripts/   HaWoR internals (SLAM/eval files dropped)
weights/  _DATA/            model weights + MANO (not committed; see Setup)
```

## Setup (on the Blackwell box)
Prereqs: NVIDIA driver ≥570, **CUDA 12.8 toolkit** (`nvcc`, for the pytorch3d
compile), **Python 3.10**.

```bash
conda create -y -n minhawor python=3.10 && conda activate minhawor
pip install --upgrade pip
pip install -r requirements_blackwell.txt
# chumpy is separate: its setup.py imports pip, which is absent under PEP517
# build isolation, so build it against the env instead:
pip install --no-build-isolation "chumpy@git+https://github.com/mattloper/chumpy"
# pytorch3d must be COMPILED for sm_120 (no reliable prebuilt for pt2.8+cu128):
TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME=/usr/local/cuda-12.8 \
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

**Weights / MANO** (not in git). Place, or symlink, next to `run_camspace.py`:
```
weights/hawor/checkpoints/hawor.ckpt     (3.1G, motion model)
weights/hawor/model_config.yaml
weights/external/detector.pt             (52M, YOLO hand detector)
_DATA/data/mano/MANO_{LEFT,RIGHT}.pkl
_DATA/data_left/mano_left/MANO_LEFT.pkl
```
Molmo (`allenai/Molmo-7B-D-0924`) auto-downloads to the HF cache the first time
`--ego_filter` is used. `droid.pth` and `infiller.pt` are **not** needed.

## Run one video
```bash
# ego (RealSense: auto focal/cx/cy from intrinsics.json) + Molmo ego filter
python run_camspace.py --src /mnt/.../ptron/2026_07_06/ego1/P0042 \
    --out /hdd/ptron_camspace/P0042/ego1 --ego_filter

# exo/wrist (GoPro/DJI: a bare mp4; give a focal or let motion-est estimate one)
python run_camspace.py --video_path /mnt/.../exo1/GX010181.MP4 \
    --out /hdd/ptron_camspace/P0042/exo1 --img_focal 900
```
Add `--vis` to also render the cam-view mp4.

### Faster motion estimation (`--gpus`)
```bash
python run_camspace.py --src <dir> --out <out> --gpus 0,1,2,3,4,5,6,7
```
Runs motion estimation **mask-free** (the per-frame MANO mask render only feeds
SLAM/the infiller, which this pipeline skips) and **sharded across GPUs**: left
and right hands — and long tracks, split on 16-frame window boundaries — run
concurrently, one model replica per GPU. Output is bit-identical to the default
single-GPU path (verified: `max|Δ|=0`). Best for long videos; for very short
clips the per-GPU model-load overhead dominates, so the default path is fine.
For the 7-view ptron batch, prefer one video per GPU (`run_ptron_session.sh`).

## Output — `camspace_hands.npz`
Per hand (axis 0: `0=left, 1=right`), in **that camera's frame**:

| key | shape | meaning |
|---|---|---|
| `trans` | (2, N, 3) | root translation |
| `root_orient` | (2, N, 3) | root/wrist orientation (axis-angle) |
| `hand_pose` | (2, N, 45) | finger articulation (wrist-relative → view-independent) |
| `betas` | (2, N, 10) | MANO shape |
| `valid` | (2, N) bool | hand detected this frame |
| `frame_index` | (N,) | source frame id (for cross-camera time alignment) |
| `focal,fx,fy,cx,cy,width,height` | scalars | intrinsics used |

## 8-GPU fan-out (one ptron session, all 7 views)
```bash
BASE=/mnt/bucket-gdm-mount/ptron/2026_07_06 OUT=/hdd/ptron_camspace \
  bash run_ptron_session.sh P0042
```
ego1→GPU0, exo1..4→GPU1..4, wrist1,2→GPU5,6. Each view writes its own
`camspace_hands.npz`.

## Multi-view fusion (separate step)
Fusing these per-view npz files — **root position + orientation from the ego
view, best `hand_pose`/`betas` across all views** — is a downstream step and is
not part of this runner.
