# Molmo model tests — findings

Log of what was actually run, on this box, for the "which Molmo variant should
front the ego-hand-pointing step" question. See `README.md` for setup/usage;
this file is results only.

Test videos:
- `example/video_0.mp4` from the [HaWoR repo](https://github.com/ThunderVVV/HaWoR) (public demo clip, 1920x1080, 121 frames / 4s)
- `/hdd/gdm_data/sec/2026_07_06/SEC001/color.mp4` — real production RealSense ego capture (1920x1080, 4401 frames / ~2.4 min). A trimmed 5s/150-frame copy lives at `/hdd/molmo_compare/sec001/color_clip5s.mp4`.

All rendered comparison videos are under `/hdd/molmo_compare/` (not in git — ephemeral output, regenerate via the scripts in this folder).

## 1. Can `transformers` just be upgraded in one env?

**No — confirmed by reproduction, not just the repo's pin comment.**

Built `Molmo-7B-D-0924` from its config (random-init weights, skips the ~28GB
download) and ran it through the real processor + `generate_from_batch()`:

- `transformers==4.45.2` (current pin): works.
- `transformers==4.57.1`: crashes — `AttributeError: 'NoneType' object has no
  attribute 'size'` at `modeling_molmo.py:1836`
  (`past_key_values[0][0].size(-2)`). Molmo-7B-D's vendored `forward()`
  assumes the legacy tuple-of-tuples KV-cache; transformers now passes a
  `Cache` object during generation by default, and indexing it that way
  breaks.
- Side finding: newer transformers' stricter static import-checker for
  `trust_remote_code` files also trips on an unused conditional
  `import tensorflow` inside Molmo's own preprocessing code — needed a
  `tensorflow-cpu` stub installed just to get past that, unrelated to the
  real Cache incompatibility above.

**Conclusion:** two separate conda envs are required, not optional:
- `molmo7b` — `torch==2.5.1+cu121`, `transformers==4.45.2` (Molmo-7B-D)
- `molmo2` — `torch==2.6.0+cu124`, `transformers==4.57.1` + `molmo_utils` (Molmo2 family)

(`cu121` tops out at torch 2.5.1; Molmo2's masking code needs `torch>=2.6`, so
`molmo2` uses the `cu124` wheel index instead. Driver 535 / CUDA 12.2 runs
`cu124` fine via CUDA minor-version compatibility — confirmed with a live
matmul on the L4.)

HF weights cache is redirected to `/hdd/hf_home` (both scripts set
`HF_HOME` via `os.environ.setdefault` before importing transformers) —
the default `~/.cache/huggingface` is on the small root disk and a Molmo
download nearly filled it once already.

## 2. Molmo-7B-D vs Molmo2-VideoPoint-4B (single-frame pointing)

Ran both on `video_0.mp4` (HaWoR demo clip):

- **Molmo-7B-D** (`lib/pipeline/molmo_ego.py`, per-frame resampling every 5
  frames, hold last point in between — same scheme as production
  `--ego_filter`): correctly points at both hands almost every sampled
  frame across the full clip, correctly says "there are none" on a few
  frames where hands leave view.
- **Molmo2-VideoPoint-4B** (native whole-video input, prompted to "track...
  throughout this video... give each hand a consistent id"): emitted
  exactly **one** `<points>` block anchored at frame 0 and nothing else —
  no coverage for the other 120 frames, despite explicitly being asked to
  track over time.

Repeated on the real SEC001 5s clip — same pattern. Both models correctly
localized both hands at frame 0 (pixel-for-pixel comparable), but 4B never
produced a second timestamp.

**Prompt-tuning did not fix it.** Tried 5 different phrasings on
Molmo2-VideoPoint-4B (imperative "track", "point to... every frame",
explicit per-hand ids, a counting-style prompt, "for each frame..."). All
five produced the same result: one `<points>` block, one timestamp, never
`<tracks>`. This matches a caveat in the model's own card: it's fine-tuned
specifically for the single-shot pointing + counting eval, not sustained
tracking — the `<tracks>` grammar exists in its vocabulary but this
checkpoint's weights just never reach for it.

## 3. Molmo2-4B (general checkpoint) — real tracking, short clips

Switching to the **general** `allenai/Molmo2-4B` (not the VideoPoint
specialization) and following its own "Tracking Video QA" recipe (distinct
from its "Pointing Video QA" recipe) got genuine multi-timestamp tracking
working. Two non-obvious requirements:

1. Must be the general Molmo2-4B checkpoint — VideoPoint-4B never emits `<tracks>` (see above).
2. Must pass `max_fps=4` (or similar) on the video input — default sampling is too sparse for tracking to produce more than one or two timestamps.

First attempt at `max_fps=8` OOM'd the 24GB L4 (`dtype="auto"` resolved to
fp32 for this checkpoint); forcing `dtype=torch.bfloat16` explicitly dropped
the weight footprint to 9.7GB and fixed it. Settled on `max_fps=4` for
headroom.

Result on the SEC001 5s clip, prompt `"Track the hands of the person wearing
the camera."`:
```
<tracks coords="0.0 1 380 822 2 873 871;0.5 1 376 822 2 872 874;...
  4.0 1 307 894 2 848 881;4.5 1 509 924;5.0 1 398 999">
```
11 timestamps (0.0s–5.0s, every 0.5s), two persistent hand ids. Visually
confirmed matching the 7B render at frame 75 (both models land on the same
spots). One real limitation even here: track `2` (right hand) stopped
updating after t=4.0s while track `1` continued to 5.0s — coverage isn't
perfectly complete even in this short-clip success case.

Script: `render_molmo2_4b_track.py`.

### Prompt refinement: "back of the palm"

Changed the default prompt to `"Track the back of the palm of each hand of
the person wearing the camera."` — reran on the same SEC001 5s clip. Output
stayed well-formed (same 11 timestamps, same id-2 dropout at t=4.0s), but
points visually land more precisely on the dorsal/knuckle surface of the
hand rather than a generic point somewhere on it. This is now the script's
default prompt.

## 4. Full-length video (4401 frames / ~2.4 min, SEC001 untrimmed)

- **Molmo-7B-D**: completed cleanly across all 4401 frames (880 sampled
  calls at `sample_every=5`), ~2 hours wall-clock on this box. Detected both
  hands with smoothly-varying, plausible coordinates the entire way through.
  `molmo7b_render_full.mp4`.

- **Molmo2-4B native tracking**: attempted with `max_fps=1` (~147 sampled
  frames, to stay within GPU memory) and `max_new_tokens=6000` (to leave
  room for ~150 timestamps of track text). Did **not** OOM this time, but
  **degenerated into repetitive garbage past ~9 seconds of clip time**:
  - track `2` (right hand) disappears entirely after t=4.0s and never returns.
  - track `1` (left hand) goes through a few more real-looking updates through
    ~t=18s, then locks onto a fixed coordinate `(756, 996)` repeated
    identically for **80+ consecutive timestamps** (t≈65.5s through the end
    at t=146.5s — more than a minute of the clip is the same frozen number).

  This is a generation-degeneracy failure (repetition collapse over a long
  structured output), not a token-budget or memory problem — the budget was
  sufficient (298 points decoded) and the process ran to completion.
  `molmo2_4b_track_full.mp4`.

## Bottom line

- **Molmo2-VideoPoint-4B**: not usable for this task at all — never tracks
  regardless of prompt, only ever a single-frame point.
- **Molmo2-4B (general) native tracking**: works well on short clips
  (~5–10s) with the right prompt/checkpoint/`max_fps`, but reliably
  degenerates into repeated garbage coordinates on a full production-length
  clip (minutes). Would need chunking into short windows + stitching to be
  viable at real clip lengths — not tested here.
- **Molmo-7B-D per-frame resampling** (the existing production
  `--ego_filter` approach): the only one of the three that held up across a
  full 2.4-minute real capture end to end. Remains the reliable choice
  unless/until native tracking is made to work in chunks.

## Scripts in this folder

| script | model | notes |
|---|---|---|
| `render_molmo7b.py` | `allenai/Molmo-7B-D-0924` | production-equivalent per-frame resampling |
| `render_molmo2_videopoint4b.py` | `allenai/Molmo2-VideoPoint-4B` | kept for reference; confirmed single-frame-only, see §2 |
| `render_molmo2_4b_track.py` | `allenai/Molmo2-4B` | real tracking, short clips only, see §3–4 |
| `render_molmopoint8b.py` | `allenai/MolmoPoint-8B` | written from docs, **never actually run** — superseded by the Molmo2-4B path above, kept only in case MolmoPoint-8B is revisited |
