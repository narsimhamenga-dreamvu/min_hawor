#!/bin/bash
# =============================================================================
# Run the cam-space hand stage on ALL views of one ptron session, one view per
# GPU (built for the 8x RTX PRO 6000 Blackwell box). Views:
#   ego1 (RealSense, auto-intrinsics, Molmo ego filter) + exo1..4 (GoPro) +
#   wrist1,wrist2 (DJI)  = 7 views -> GPUs 0..6.
#
# Each view writes <OUT>/<session>/<view>/camspace_hands.npz. Multi-view FUSION
# (root+orient from ego, best hand_pose/betas across views) is a SEPARATE step.
#
# Usage:
#   BASE=/mnt/bucket-gdm-mount/ptron/2026_07_06 OUT=/hdd/ptron_camspace \
#     bash run_ptron_session.sh P0042
# Optional env: EXO_FOCAL / WRIST_FOCAL (GoPro/DJI focal; omit -> motion-est
#   estimates one), PY (python bin), EGO_FILTER=0 to disable Molmo on ego.
# =============================================================================
set -uo pipefail

SESSION="${1:?usage: run_ptron_session.sh <SESSION e.g. P0042>}"
BASE="${BASE:?set BASE=/mnt/.../ptron/<date>}"
OUT="${OUT:?set OUT=/hdd/ptron_camspace}"
PY="${PY:-python}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QT_QPA_PLATFORM=offscreen PYOPENGL_PLATFORM=egl

launch() {  # <gpu> <view> <extra run_camspace.py args...>
  local gpu="$1" view="$2"; shift 2
  local out="$OUT/$SESSION/$view"
  mkdir -p "$out"
  echo ">> [GPU $gpu] $view -> $out"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$HERE/run_camspace.py" --out "$out" "$@" \
    >"$out/run.log" 2>&1 &
}

first() { ls "$1" 2>/dev/null | head -1; }   # first matching file (glob-safe)

# --- ego1: RealSense session dir; auto intrinsics; Molmo ego filter ----------
EGO_DIR="$BASE/ego1/$SESSION"
[ "${EGO_FILTER:-1}" = "1" ] && EGO_ARGS="--ego_filter" || EGO_ARGS=""
[ -d "$EGO_DIR" ] && launch 0 ego1 --src "$EGO_DIR" $EGO_ARGS

# --- exo1..4: GoPro GX*.MP4 --------------------------------------------------
for i in 1 2 3 4; do
  d="$BASE/exo$i/$SESSION"; [ -d "$d" ] || d="$BASE/exo$i"
  mp4="$(first "$d"/GX*.MP4)"; [ -z "$mp4" ] && mp4="$(first "$d"/*.MP4)"
  [ -n "$mp4" ] || { echo "!! no exo$i video under $d"; continue; }
  EF=""; [ -n "${EXO_FOCAL:-}" ] && EF="--img_focal $EXO_FOCAL"
  launch "$i" "exo$i" --video_path "$d/$(basename "$mp4")" $EF
done

# --- wrist1,2: DJI *.MP4 -> GPUs 5,6 -----------------------------------------
g=5
for i in 1 2; do
  d="$BASE/wrist$i/$SESSION"; [ -d "$d" ] || d="$BASE/wrist$i"
  mp4="$(first "$d"/DJI*.MP4)"; [ -z "$mp4" ] && mp4="$(first "$d"/*.MP4)"
  [ -n "$mp4" ] || { echo "!! no wrist$i video under $d"; g=$((g+1)); continue; }
  WF=""; [ -n "${WRIST_FOCAL:-}" ] && WF="--img_focal $WRIST_FOCAL"
  launch "$g" "wrist$i" --video_path "$d/$(basename "$mp4")" $WF
  g=$((g+1))
done

wait
echo ">> all views done for $SESSION -> $OUT/$SESSION/*/camspace_hands.npz"
