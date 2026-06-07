# Handoff: run **MVSplat** zero-shot on the CARLA **CSE** benchmark

You're working **inside the cloned `mvsplat` repo** (https://github.com/donydchen/mvsplat).
Goal: evaluate MVSplat **zero-shot** (official RealEstate10K weights) on our CARLA
**CSE (Cross-Sensor)** benchmark and score it with our `eval_cse.py`, so the
numbers are directly comparable to GS-Net and 3DGS.

> Sister task in another window does the same for DepthSplat. MVSplat and
> DepthSplat share the **same pixelSplat `.torch` data format**, so the converter
> below is identical for both.

---

## TL;DR — what to build

```
runs/cse_scenes/<id>/  ──(1) colmap_to_pixelsplat.py──►  datasets/carla_cse/test/{*.torch, index.json, cse_meta.json}
                       ──(2) DatasetCarlaCSE + ViewSamplerCSE──►  feed-forward render at each TARGET pose
                       ──(2) export──►  mvsplat/<id>/renders/<target_name>.png   (named exactly per test.txt)
                       ──(3) eval_cse.py──►  PSNR / SSIM / LPIPS
```

You implement: **(1)** the converter (code provided, just save & run), **(2)** a
CARLA-CSE dataset + view-sampler + a render-export script that reuses MVSplat's own
encoder/decoder, **(3)** run the provided `eval_cse.py`.

---

## Background — the CARLA CSE benchmark (read this, it defines correctness)

- **5 test sequences**: `110 / 210 / 310 / 410 / 510`. For each:
  - **Source = 60 views** (reconstruction input): 6 **odd** cameras (cam01/03/05/07/09/11) × 10 frames.
  - **Target = 60 views** (held out, evaluation only): 6 **even** cameras (cam02/04/06/08/10/12, each yaw-offset 30°) × 10 frames. **Never seen during reconstruction.**
- A 12-camera ring, radius **0.75 m**, 30° between adjacent cameras. Each even
  (target) camera sits angularly **between** two odd (source) cameras → this is
  close to **interpolation**, which is favorable for MVSplat's 2-view regime.
- Each scene is a standard COLMAP text model:
  ```
  runs/cse_scenes/<id>/
    sparse/0/cameras.txt   # 1 PINHOLE intrinsic shared by all cams: fx fy cx cy W H
    sparse/0/images.txt    # 120 images, qvec/tvec = world->camera
    sparse/0/test.txt      # the 60 TARGET image names (eval set); the rest are source
    sparse/0/points3D.*    # source (odd) sparse SfM points (optional init)
    images/<name>          # all 120 images; target GT = images/<name> for name in test.txt
  ```
- **Pose convention (COLMAP, OpenCV)**: `R = qvec2rotmat(q)` is world→camera,
  `t` is world→camera; camera center `C = -R^T t`. This is the **same convention
  MVSplat/pixelSplat use** (they store world→camera and invert internally) → **no
  conversion needed**.
- **Deliverable per scene**: a `renders/` folder with the 60 target views,
  **named exactly as the names in `test.txt`** (e.g. `e02_03.png`), at the **same
  resolution** as the GT (the eval script asserts equal shape).

You will be given (ask the user / find in the workspace):
- `runs/cse_scenes/110 … 510` (the 5 test scenes),
- `eval_cse.py` (the scorer; do not modify its metric definitions).

---

## Step 1 — convert CARLA CSE scenes to the `.torch` format

Save this as `tools/colmap_to_pixelsplat.py` and run it. It's already been validated.

```python
#!/usr/bin/env python3
"""
Convert a CARLA CSE COLMAP scene (runs/cse_scenes/<id>/) into the pixelSplat
`.torch` chunk format used by BOTH MVSplat and DepthSplat.

pixelSplat / MVSplat chunk schema (per scene dict):
  {
    "key":        str,                      # scene id (e.g. "110")
    "cameras":    float tensor [N, 18],     # [fx, fy, cx, cy, 0, 0,  w2c(3x4 row-major)]
                                            #   intrinsics NORMALIZED by image size
                                            #   extrinsics = world->camera, OpenCV convention
    "images":     list[uint8 tensor],       # each = raw JPEG bytes of one frame
    "url":        str,
    "timestamps": long tensor [N],
  }
Also emits cse_meta.json: per scene the frame names, is_target (name in test.txt),
and camera centers (for choosing context views and naming the rendered outputs).
"""
import argparse, io, json, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def read_cameras_txt(path):
    cameras = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            cameras[int(tok[0])] = dict(
                model=tok[1], width=int(tok[2]), height=int(tok[3]),
                params=list(map(float, tok[4:])))
    return cameras


def read_images_txt(path):
    images = []
    with open(path) as f:
        lines = list(f)
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        tok = line.split()
        images.append(dict(
            name=tok[9],
            qvec=np.array(list(map(float, tok[1:5])), dtype=np.float64),
            tvec=np.array(list(map(float, tok[5:8])), dtype=np.float64),
            camera_id=int(tok[8])))
        i += 2  # skip the 2D-points line
    return images


def intrinsics_from_camera(cam):
    p, model = cam["params"], cam["model"]
    if model == "PINHOLE":
        return p[0], p[1], p[2], p[3]
    if model == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    return p[0], (p[1] if len(p) > 1 else p[0]), \
           (p[2] if len(p) > 2 else cam["width"]/2.0), \
           (p[3] if len(p) > 3 else cam["height"]/2.0)


def jpeg_bytes_tensor(image_path, jpeg_quality=95):
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(data), img.size


def convert_scene(scene_dir, jpeg_quality=95):
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    sparse = scene_dir / "sparse" / "0"
    cameras = read_cameras_txt(sparse / "cameras.txt")
    images = read_images_txt(sparse / "images.txt")
    test_txt = sparse / "test.txt"
    target_names = {ln.strip() for ln in open(test_txt) if ln.strip()} if test_txt.exists() else set()
    img_dir = scene_dir / "images"

    cam_rows, image_tensors, names, is_target, cam_centers, timestamps = [], [], [], [], [], []
    for idx, im in enumerate(images):
        cam = cameras[im["camera_id"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = intrinsics_from_camera(cam)
        R = qvec2rotmat(im["qvec"]); t = im["tvec"]
        w2c_3x4 = np.concatenate([R, t.reshape(3, 1)], axis=1)
        cam_rows.append(np.concatenate([
            np.array([fx/W, fy/H, cx/W, cy/H, 0.0, 0.0]), w2c_3x4.reshape(-1)]))
        cam_centers.append((-R.T @ t).tolist())
        tens, (iw, ih) = jpeg_bytes_tensor(img_dir / im["name"], jpeg_quality)
        if (iw, ih) != (W, H):
            print(f"[warn] {im['name']}: image {iw}x{ih} != cameras.txt {W}x{H}")
        image_tensors.append(tens)
        names.append(im["name"]); is_target.append(im["name"] in target_names); timestamps.append(idx)

    scene_dict = {
        "key": scene_id,
        "cameras": torch.tensor(np.stack(cam_rows, 0), dtype=torch.float32),
        "images": image_tensors, "url": "",
        "timestamps": torch.tensor(timestamps, dtype=torch.long)}
    meta = {
        "scene_id": scene_id, "width": cameras[images[0]["camera_id"]]["width"],
        "height": cameras[images[0]["camera_id"]]["height"],
        "names": names, "is_target": is_target, "cam_centers": cam_centers,
        "num_source": int(sum(1 for x in is_target if not x)),
        "num_target": int(sum(is_target))}
    return scene_dict, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene"); ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_size", type=int, default=1)
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()
    scene_dirs = ([args.scene] if args.scene else []) + (args.scenes or [])
    assert scene_dirs
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index, meta_all, chunk, cidx = {}, {}, [], 0

    def flush():
        nonlocal chunk, cidx
        if not chunk:
            return
        fname = f"{cidx:06d}.torch"; torch.save(chunk, out / fname)
        for s in chunk:
            index[s["key"]] = fname
        print(f"[ok] {fname}: {len(chunk)} scene(s)"); chunk = []; cidx += 1

    for sd in scene_dirs:
        scene_dict, meta = convert_scene(sd, args.jpeg_quality)
        print(f"[scene {scene_dict['key']}] {len(scene_dict['images'])} frames "
              f"({meta['num_source']} src / {meta['num_target']} tgt)")
        chunk.append(scene_dict); meta_all[scene_dict["key"]] = meta
        if len(chunk) >= args.chunk_size:
            flush()
    flush()
    json.dump(index, open(out / "index.json", "w"), indent=2)
    json.dump(meta_all, open(out / "cse_meta.json", "w"), indent=2)
    print(f"[done] -> {out}")


if __name__ == "__main__":
    main()
```

Run:
```bash
python tools/colmap_to_pixelsplat.py \
    --scenes runs/cse_scenes/110 runs/cse_scenes/210 runs/cse_scenes/310 \
             runs/cse_scenes/410 runs/cse_scenes/510 \
    --out    datasets/carla_cse/test
```

Sanity: each scene should print `60 src / 60 tgt`.

---

## Step 2 — render & export target views (the real work)

**Reuse MVSplat's own encoder + decoder.** Do NOT reimplement rendering. Read these
repo files first to match the exact API:
- `src/dataset/dataset_re10k.py` — how a batch is built: keys `context`/`target`,
  each with `image [B,V,3,H,W]`, `intrinsics [B,V,3,3]` (normalized), `extrinsics
  [B,V,4,4]` (cam-to-world), `near`, `far`. **Copy its pose-normalization and
  image-resize logic verbatim** so context+target stay in a consistent frame.
- `src/dataset/view_sampler/` — the `ViewSampler` base class and
  `view_sampler_evaluation.py`.
- `src/model/model_wrapper.py` / `src/model/encoder/encoder_costvolume.py` /
  `src/model/decoder/` — `encoder(context) -> gaussians`; `decoder.forward(gaussians,
  extrinsics, intrinsics, near, far, (H,W)) -> color`.

Implement two small things, modeled on the repo's own classes:

1. **`DatasetCarlaCSE`** (sibling of `DatasetRE10k`): loads our `.torch` chunks via
   `index.json`, plus `cse_meta.json`. Decodes JPEG bytes, builds intrinsics
   `K=[[fx,0,cx],[0,fy,cy],[0,0,1]]` from the normalized row (fx,fy,cx,cy already
   normalized → keep as the repo does), and `extrinsics = inverse(w2c)`.

2. **`ViewSamplerCSE`**: enumerates **one item per target view**. For a target
   index `t`, context = the **2 nearest source views** by Euclidean distance
   between `cam_centers[t]` and the source `cam_centers` (source = `is_target==False`).
   Return `context_indices=[a,b]`, `target_indices=[t]`.

**near/far**: CARLA is unbounded outdoor — do not use re10k's defaults blindly.
Per scene, compute robust depth bounds from the source SfM points
(`points3D`): project them into the source cameras and take ~5th/95th percentile
depth (clamp near ≥ small positive). Pass these as `near`/`far`. If points are
unavailable, start with `near=0.5, far=150` and adjust.

**resolution**: feed context images to the encoder at the model's trained input
size (reuse the repo's resize to `cfg.dataset.image_shape`, typically 256×256).
Render the target at the **GT target resolution** (so `eval_cse.py` shapes match) —
i.e. pass the target's full (H,W) to the decoder. Intrinsics are normalized so
they're resolution-independent.

**export**: for each target `t`, save the rendered RGB to
`outputs/mvsplat/<scene_id>/renders/<names[t]>` (PNG, the exact target filename
from `cse_meta.json`/`test.txt`). 60 files per scene.

A clean way to drive it: add a small `src/scripts/render_cse.py` that builds the
config, instantiates `DatasetCarlaCSE` + `ViewSamplerCSE`, loads the checkpoint via
the repo's `ModelWrapper`, loops batches, renders, and writes PNGs.

---

## Step 3 — score

```bash
python eval_cse.py --multi \
  outputs/mvsplat/110/renders:runs/cse_scenes/110 \
  outputs/mvsplat/210/renders:runs/cse_scenes/210 \
  outputs/mvsplat/310/renders:runs/cse_scenes/310 \
  outputs/mvsplat/410/renders:runs/cse_scenes/410 \
  outputs/mvsplat/510/renders:runs/cse_scenes/510 \
  --out outputs/mvsplat/cse_scores.json
```
Report the per-seq + 5-seq-average PSNR/SSIM/LPIPS table.

---

## Environment & checkpoints

- Env: follow the repo (`environment.yaml`; PyTorch 2.1.2). Needs a CUDA GPU.
- Checkpoints (Google Drive, linked in the MVSplat README): save to `checkpoints/`.
  - **`re10k.ckpt`** — RealEstate10K (**indoor**) → the main zero-shot point.
  - *(optional)* `acid.ckpt` — ACID (outdoor aerial); a second zero-shot reference.
- Use **2 context views** (re10k weights are 2-view).

---

## Expectations & framing

RealEstate10K weights are trained on **small-baseline indoor** interpolation. The
CARLA ring is wider-baseline outdoor, so **zero-shot numbers are expected to be
clearly below GS-Net** — that gap *is* the indoor→outdoor transfer evidence the
reviewers asked for. The fair, in-domain number will come later from a
**CARLA fine-tuned** run (training scenes 101–109 … 501–509, RGB+pose only; not part
of this task yet).

## Acceptance checklist
- [ ] Converter prints `60 src / 60 tgt` for all 5 scenes.
- [ ] `renders/` has exactly 60 PNGs per scene, names == `test.txt`, shape == GT.
- [ ] `eval_cse.py` runs without "no overlap"/shape errors and prints the table.
- [ ] Sanity-check 2–3 rendered images visually (geometry roughly aligned, not noise).
- [ ] Save `cse_scores.json` and the printed table.
