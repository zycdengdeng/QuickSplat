#!/usr/bin/env python3
"""
Convert a CARLA CSE COLMAP scene (runs/cse_scenes/<id>/) into the pixelSplat
`.torch` chunk format used by BOTH MVSplat and DepthSplat.

Why one script for both methods:
  DepthSplat is built on the pixelSplat / MVSplat codebase, so all three share
  the exact same on-disk data format. Convert once, evaluate with either model.

pixelSplat / MVSplat / DepthSplat chunk schema (per scene dict):
  {
    "key":        str,                      # scene id (e.g. "110")
    "cameras":    float tensor [N, 18],     # [fx, fy, cx, cy, 0, 0,  w2c(3x4 row-major)]
                                            #   intrinsics NORMALIZED by image size
                                            #   extrinsics = world->camera, OpenCV convention
    "images":     list[uint8 tensor],       # each = raw JPEG bytes of one frame
    "url":        str,                       # unused, kept for compatibility
    "timestamps": long tensor [N],          # frame index, kept for compatibility
  }
A directory ("test/" for zero-shot eval) holds one or more `000000.torch`
chunk files (each a `list[dict]`) plus an `index.json` mapping scene_key -> chunk_filename.

COLMAP <-> pixelSplat convention note:
  COLMAP stores world->camera (qvec, tvec) in OpenCV convention (+X right, +Y down,
  +Z forward). pixelSplat's loader also expects world->camera OpenCV 3x4 and inverts
  it internally to camera-to-world. So the extrinsics map across with ZERO conversion.

We additionally emit `cse_meta.json` next to the chunks, recording for each scene
the per-frame filename, the source/target split (target = names listed in test.txt),
and the camera center. The CSE view-sampler / render-export step uses this to pick
context (source) views for each target pose and to name the rendered outputs exactly
as the GT target filenames (so they match `eval_cse.py`).

Usage:
  # one scene
  python colmap_to_pixelsplat.py \
      --scene runs/cse_scenes/110 \
      --out   datasets/carla_cse/test

  # all 5 test scenes into one dataset dir
  python colmap_to_pixelsplat.py \
      --scenes runs/cse_scenes/110 runs/cse_scenes/210 runs/cse_scenes/310 \
               runs/cse_scenes/410 runs/cse_scenes/510 \
      --out    datasets/carla_cse/test
"""
import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# COLMAP text-model parsing
# --------------------------------------------------------------------------- #
def qvec2rotmat(qvec):
    """COLMAP quaternion (w, x, y, z) -> 3x3 rotation matrix (world->camera)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,     2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,     1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_cameras_txt(path):
    """Return {camera_id: dict(model, width, height, params)}."""
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            cam_id = int(tok[0])
            model = tok[1]
            width, height = int(tok[2]), int(tok[3])
            params = list(map(float, tok[4:]))
            cameras[cam_id] = dict(model=model, width=width, height=height, params=params)
    return cameras


def read_images_txt(path):
    """Return list of dicts {name, qvec, tvec, camera_id} in file order.

    images.txt stores two lines per image; the second (2D points) is skipped.
    """
    images = []
    with open(path, "r") as f:
        lines = [ln for ln in f]
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        tok = line.split()
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        qvec = np.array(list(map(float, tok[1:5])), dtype=np.float64)
        tvec = np.array(list(map(float, tok[5:8])), dtype=np.float64)
        camera_id = int(tok[8])
        name = tok[9]
        images.append(dict(name=name, qvec=qvec, tvec=tvec, camera_id=camera_id))
        i += 2  # skip the 2D-points line
    return images


def intrinsics_from_camera(cam):
    """Return (fx, fy, cx, cy) in PIXELS for a PINHOLE/SIMPLE_PINHOLE camera."""
    p = cam["params"]
    model = cam["model"]
    if model == "PINHOLE":
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    elif model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        # Distorted models: drop distortion, use the focal/principal terms.
        # For CARLA CSE the model is PINHOLE so this branch is a safety net only.
        fx = p[0]
        fy = p[1] if len(p) > 1 else p[0]
        cx = p[2] if len(p) > 2 else cam["width"] / 2.0
        cy = p[3] if len(p) > 3 else cam["height"] / 2.0
    return fx, fy, cx, cy


# --------------------------------------------------------------------------- #
# Scene conversion
# --------------------------------------------------------------------------- #
def jpeg_bytes_tensor(image_path, jpeg_quality=95):
    """Load an image and return a uint8 tensor of its JPEG-encoded bytes."""
    img = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    data = np.frombuffer(buf.getvalue(), dtype=np.uint8).copy()
    return torch.from_numpy(data), img.size  # (tensor, (W, H))


def convert_scene(scene_dir, jpeg_quality=95):
    """Convert one cse_scenes/<id> directory into (scene_dict, meta_dict)."""
    scene_dir = Path(scene_dir)
    scene_id = scene_dir.name
    sparse = scene_dir / "sparse" / "0"
    cameras = read_cameras_txt(sparse / "cameras.txt")
    images = read_images_txt(sparse / "images.txt")

    # target (test) view names
    test_txt = sparse / "test.txt"
    target_names = set()
    if test_txt.exists():
        target_names = {ln.strip() for ln in open(test_txt) if ln.strip()}
    else:
        print(f"[warn] {test_txt} missing; no source/target split recorded for {scene_id}")

    img_dir = scene_dir / "images"

    cam_rows = []
    image_tensors = []
    names = []
    is_target = []
    cam_centers = []
    timestamps = []

    for idx, im in enumerate(images):
        cam = cameras[im["camera_id"]]
        W, H = cam["width"], cam["height"]
        fx, fy, cx, cy = intrinsics_from_camera(cam)

        # Normalize intrinsics by image size (pixelSplat convention).
        fx_n, fy_n = fx / W, fy / H
        cx_n, cy_n = cx / W, cy / H

        # world->camera 3x4 (OpenCV), same convention pixelSplat expects.
        R = qvec2rotmat(im["qvec"])      # world->camera rotation
        t = im["tvec"]                   # world->camera translation
        w2c_3x4 = np.concatenate([R, t.reshape(3, 1)], axis=1)  # (3,4)

        row = np.concatenate([
            np.array([fx_n, fy_n, cx_n, cy_n, 0.0, 0.0], dtype=np.float64),
            w2c_3x4.reshape(-1),  # row-major 12 values
        ])
        cam_rows.append(row)

        # camera center C = -R^T t  (for geometric nearest-context selection)
        C = -R.T @ t
        cam_centers.append(C.tolist())

        img_path = img_dir / im["name"]
        tens, (iw, ih) = jpeg_bytes_tensor(img_path, jpeg_quality=jpeg_quality)
        if (iw, ih) != (W, H):
            print(f"[warn] {im['name']}: image {iw}x{ih} != cameras.txt {W}x{H}; "
                  f"intrinsics normalized by cameras.txt size.")
        image_tensors.append(tens)

        names.append(im["name"])
        is_target.append(im["name"] in target_names)
        timestamps.append(idx)

    scene_dict = {
        "key": scene_id,
        "cameras": torch.tensor(np.stack(cam_rows, axis=0), dtype=torch.float32),
        "images": image_tensors,
        "url": "",
        "timestamps": torch.tensor(timestamps, dtype=torch.long),
    }
    meta = {
        "scene_id": scene_id,
        "width": cameras[images[0]["camera_id"]]["width"],
        "height": cameras[images[0]["camera_id"]]["height"],
        "names": names,
        "is_target": is_target,           # True => held-out target (in test.txt)
        "cam_centers": cam_centers,        # world-space camera centers, frame order
        "num_source": int(sum(1 for x in is_target if not x)),
        "num_target": int(sum(is_target)),
    }
    return scene_dict, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", help="a single cse_scenes/<id> directory")
    ap.add_argument("--scenes", nargs="+", help="multiple cse_scenes/<id> directories")
    ap.add_argument("--out", required=True, help="output dataset dir (e.g. datasets/carla_cse/test)")
    ap.add_argument("--chunk_size", type=int, default=1, help="scenes per .torch chunk")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    args = ap.parse_args()

    scene_dirs = []
    if args.scene:
        scene_dirs.append(args.scene)
    if args.scenes:
        scene_dirs.extend(args.scenes)
    assert scene_dirs, "Provide --scene or --scenes"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    index = {}
    meta_all = {}
    chunk_scenes = []
    chunk_idx = 0

    def flush():
        nonlocal chunk_scenes, chunk_idx
        if not chunk_scenes:
            return
        fname = f"{chunk_idx:06d}.torch"
        torch.save(chunk_scenes, out / fname)
        for s in chunk_scenes:
            index[s["key"]] = fname
        print(f"[ok] wrote {fname} with {len(chunk_scenes)} scene(s)")
        chunk_scenes = []
        chunk_idx += 1

    for sd in scene_dirs:
        scene_dict, meta = convert_scene(sd, jpeg_quality=args.jpeg_quality)
        print(f"[scene {scene_dict['key']}] {len(scene_dict['images'])} frames "
              f"({meta['num_source']} source / {meta['num_target']} target)")
        chunk_scenes.append(scene_dict)
        meta_all[scene_dict["key"]] = meta
        if len(chunk_scenes) >= args.chunk_size:
            flush()
    flush()

    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    with open(out / "cse_meta.json", "w") as f:
        json.dump(meta_all, f, indent=2)
    print(f"[done] index.json + cse_meta.json written to {out}")


if __name__ == "__main__":
    main()
