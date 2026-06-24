#!/usr/bin/env python3
"""
Batch-convert lidar/*.laz under each Bench2Drive-base-extracted scene into BEV png.
Reuses lidar_to_bev.load_points + bev_projection.
Output: <scene>/lidar_bev/<frame>.png  (alongside lidar/)

Usage:
  # All scenes (multiprocessing)
  python tools/batch_lidar_to_bev.py --workers 16
  # Run a specific scene as a trial
  python tools/batch_lidar_to_bev.py --scenes Accident_Town03_Route101_Weather23
  # Convert only the first N (quick validation)
  python tools/batch_lidar_to_bev.py --limit 20
"""
import os
import sys
import glob
import argparse
from multiprocessing import Pool
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lidar_to_bev import load_points, bev_projection


def convert_one(args):
    laz, out_png = args
    if os.path.exists(out_png):
        return ("skip", out_png)
    try:
        pts = load_points(laz)
        bev = bev_projection(pts)
        Image.fromarray(bev, mode="RGB").save(out_png)
        return ("ok", out_png)
    except Exception as e:
        return ("err", f"{laz}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted")
    ap.add_argument("--scenes", nargs="*", default=None, help="scene dir names; default = all scenes")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="limit total #files (for a quick test run)")
    a = ap.parse_args()

    scene_dirs = (
        [os.path.join(a.root, s) for s in a.scenes]
        if a.scenes else sorted(glob.glob(os.path.join(a.root, "*/")))
    )

    tasks = []
    for sd in scene_dirs:
        sd = sd.rstrip("/")
        lidar_dir = os.path.join(sd, "lidar")
        if not os.path.isdir(lidar_dir):
            continue
        out_dir = os.path.join(sd, "lidar_bev")
        os.makedirs(out_dir, exist_ok=True)
        for laz in sorted(glob.glob(os.path.join(lidar_dir, "*.laz"))):
            stem = os.path.splitext(os.path.basename(laz))[0]
            tasks.append((laz, os.path.join(out_dir, stem + ".png")))

    if a.limit:
        tasks = tasks[: a.limit]
    print(f"{len(tasks)} laz files over {len(scene_dirs)} scene(s), {a.workers} workers", flush=True)

    ok = skip = err = 0
    with Pool(a.workers) as p:
        for i, (st, info) in enumerate(p.imap_unordered(convert_one, tasks, chunksize=8)):
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                err += 1
                print("ERR", info, flush=True)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tasks)}  ok={ok} skip={skip} err={err}", flush=True)
    print(f"DONE ok={ok} skip={skip} err={err}", flush=True)


if __name__ == "__main__":
    main()
