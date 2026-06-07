# CARLA CSE × feed-forward 3DGS (MVSplat / DepthSplat)

Drop these standalone scripts into your GPU box to evaluate **MVSplat** and
**DepthSplat** on the CARLA **CSE** benchmark, scored with the same
`eval_cse.py` (PSNR / SSIM / LPIPS-vgg) as GS-Net and 3DGS — apples-to-apples.

> MVSplat and DepthSplat share the **same pixelSplat `.torch` data format**
> (DepthSplat is built on the MVSplat/pixelSplat codebase). So **one converter
> feeds both methods.**

## Pipeline overview

```
runs/cse_scenes/<id>/            colmap_to_pixelsplat.py        datasets/carla_cse/test/
  sparse/0/{cameras,images,test} ───────────────────────────►   <chunk>.torch
  images/                                                        index.json
                                                                 cse_meta.json
                                                                       │
                                  cse_render_export.py (per method)    ▼
   pretrained re10k ckpt ──────► feed-forward: context=source views ──► <method>/<id>/renders/
                                  render at each TARGET pose            e02_00.png ... (named per test.txt)
                                                                       │
                                  gsnet/eval_cse.py                    ▼
                                                                 PSNR / SSIM / LPIPS
```

## Step 1 — convert CARLA CSE scenes to `.torch`  (DONE, validated)

`colmap_to_pixelsplat.py` reads a `runs/cse_scenes/<id>/` COLMAP text model and
emits the pixelSplat chunk format used by both methods.

```bash
python colmap_to_pixelsplat.py \
    --scenes runs/cse_scenes/110 runs/cse_scenes/210 runs/cse_scenes/310 \
             runs/cse_scenes/410 runs/cse_scenes/510 \
    --out    datasets/carla_cse/test
```

Outputs in `datasets/carla_cse/test/`:
- `00000N.torch` — list of scene dicts (`key`, `cameras [N,18]`, `images` jpeg bytes, `timestamps`)
- `index.json` — `{scene_id: chunk_file}`
- `cse_meta.json` — per scene: frame `names`, `is_target` (True = in `test.txt`),
  `cam_centers` (world space) → used by the render-export to pick context views
  and to name outputs exactly as the GT target filenames.

Conventions (already handled):
- intrinsics normalized by image size: `[fx/W, fy/H, cx/W, cy/H, 0, 0]`
- extrinsics = world→camera, OpenCV (COLMAP) convention, 3×4 row-major — **no conversion**

## Step 2 — render & export target views  (TODO: `cse_render_export.py`, per method)

For each test scene, for every **target** pose (even cam, listed in `test.txt`):
1. pick the K nearest **source** views as context (geometric nearest by `cam_centers`;
   K=2 for MVSplat, K up to 6 for DepthSplat),
2. run the model feed-forward → Gaussians,
3. render at the target camera (output resolution = GT target resolution so
   `eval_cse.py`'s shape check passes),
4. save as `<method>/<id>/renders/<target_name>.png`.

## Step 3 — score (unchanged, your script)

```bash
python gsnet/eval_cse.py --multi \
  mvsplat/110/renders:runs/cse_scenes/110  mvsplat/210/renders:runs/cse_scenes/210 \
  mvsplat/310/renders:runs/cse_scenes/310  mvsplat/410/renders:runs/cse_scenes/410 \
  mvsplat/510/renders:runs/cse_scenes/510  --out mvsplat/cse_scores.json
```

## Checkpoints (zero-shot)

- MVSplat: `re10k.ckpt` (Google Drive, indoor RealEstate10K) — also `acid.ckpt` (outdoor aerial)
- DepthSplat: `depthsplat-gs-{small,base,large}-re10k-256x256-view2-*.pth` (HuggingFace `haofeixu/depthsplat`)

## Expectation for zero-shot (set this in the rebuttal framing)

RealEstate10K models are trained on **small-baseline indoor** view interpolation.
The CARLA ring has a **0.75 m radius with 30° between cameras** — a much wider
baseline and a different domain. Zero-shot numbers are expected to be **well below
GS-Net**; that gap is exactly the indoor→outdoor transfer question the reviewers
raised. The *fair* number comes from the CARLA-fine-tuned run (Step 4, pending
training data).

## Step 4 — CARLA fine-tune (pending training scenes 101–109 … 501–509)

Convert training scenes the same way into `datasets/carla_cse/train/` (with their
own `index.json`), then fine-tune from the re10k checkpoint. MVSplat needs only
posed multi-view RGB; DepthSplat's depth branch can additionally use CARLA's GT
depth if desired. Configs to be added once the training data format is confirmed.
