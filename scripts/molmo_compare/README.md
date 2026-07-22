# Molmo model comparison (pointing/tracking for ego-hand disambiguation)

Two standalone renderers, for eyeballing which Molmo variant better isolates
the camera-wearer's hands before it's wired into `run_camspace.py --ego_filter`
(currently uses Molmo-7B-D, see `lib/pipeline/molmo_ego.py`).

| script | model | mechanism |
|---|---|---|
| `render_molmo7b.py` | `allenai/Molmo-7B-D-0924` | image-only; re-run every N frames, hold last point in between |
| `render_molmopoint8b.py` | `allenai/MolmoPoint-8B` | native video pointing + tracking, stable `object_id` per hand across the whole clip |

Both draw the model's output points onto a copy of the source video and write
an mp4, so the two can be played side by side.

## Env split (important)

`render_molmo7b.py` runs in the existing `minhawor` env
(`requirements_blackwell.txt`, `transformers==4.45.2` -- pinned because newer
transformers breaks Molmo-7B-D's custom modeling code).

`render_molmopoint8b.py` needs `transformers>=4.57.1` + the `molmo_utils`
package from `github.com/allenai/molmo2` -- **do not** install these into
`minhawor`, they conflict with the 4.45.2 pin. Use a separate conda env:

```bash
conda create -y -n molmo2 python=3.11
conda activate molmo2
pip install torch torchvision pillow einops accelerate decord2
pip install "transformers==4.57.1" molmo_utils   # or: pip install "git+https://github.com/allenai/molmo2"
```

## Status

Written from AI2's model card / README docs, **not executed against real
weights yet** (no model download was done while writing these) -- run them
and iterate. In particular `render_molmopoint8b.py` prints the raw decoded
point tuples before assuming their field order (AI2's own pages disagree on
whether it's `object_id`-first or `frame`-first); check that printout on
first run and fix the unpacking in `by_frame` if it's off.

`allenai/Molmo2-VideoPoint-4B` is a lighter alternative to MolmoPoint-8B if
8B is too heavy for the 24GB L4 in this box -- same video-pointing idea, no
tracking `object_id` (see the Molmo2 collection on HF).
