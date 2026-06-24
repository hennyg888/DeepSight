#!/usr/bin/env python3
"""Side-by-side base vs lidar demo MP4 for the SAME scenario.

Left = base run CAM_FRONT, right = lidar run CAM_FRONT, each with a speed/brake HUD
and a title bar (scenario + final DS). Shorter clip holds its last frame so both
play to the same length. Front-cam only (official style).

Usage:
  python tools/make_compare_video.py <base_scenario_dir> <lidar_scenario_dir> \
      --title "HardBreakRoute" --left-ds 100 --right-ds 1 [out.mp4] [--fps 10] [--max 400]
"""
import sys, os, glob, json, argparse, subprocess, shutil
import cv2
import numpy as np

TILE_W = 800

# Bundled ffmpeg; re-encode cv2's mp4v output to H.264 for inline VSCode/browser preview.
_FFMPEG = "/home/s56cai/DeepSight/.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"


def to_h264(path):
    ff = _FFMPEG if os.path.exists(_FFMPEG) else shutil.which("ffmpeg")
    if not ff:
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-preset", "fast", "-movflags", "+faststart", tmp])
    if r.returncode == 0:
        os.replace(tmp, path)


def frames_of(sd):
    return sorted(glob.glob(os.path.join(sd, "camera", "CAM_FRONT", "*.jpg")))


def render_side(fp, sd, tag, ds):
    img = cv2.imread(fp)
    idx = int(os.path.splitext(os.path.basename(fp))[0])
    mp = os.path.join(sd, "meta", f"{idx:04d}.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        hud = f"speed {float(m['speed']):.1f}  thr {float(m['throttle']):.2f}  brk {float(m['brake']):.2f}"
        cv2.putText(img, hud, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    h = int(img.shape[0] * TILE_W / img.shape[1])
    img = cv2.resize(img, (TILE_W, h))
    bar = np.zeros((46, TILE_W, 3), np.uint8)
    is_base = tag.startswith("base")
    color = (80, 220, 80) if is_base else (80, 160, 255)
    label = "camera-only" if is_base else "camera+lidar"
    cv2.putText(bar, f"{label}   DS={ds}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def render_bev(lidar_fp, lidar_dir, H):
    """Square LiDAR BEV tile (HxH) for the lidar run's current frame; black if missing."""
    idx = os.path.splitext(os.path.basename(lidar_fp))[0]
    bp = os.path.join(lidar_dir, "lidar_bev", idx + ".png")
    img = cv2.imread(bp) if os.path.exists(bp) else None
    tile = np.zeros((H, H, 3), np.uint8) if img is None else cv2.resize(img, (H, H))
    cv2.putText(tile, "LiDAR BEV", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 160, 255), 2, cv2.LINE_AA)
    return tile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir"); ap.add_argument("lidar_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--title", default="")
    ap.add_argument("--left-ds", default="?"); ap.add_argument("--right-ds", default="?")
    ap.add_argument("--fps", type=int, default=10); ap.add_argument("--max", type=int, default=400)
    ap.add_argument("--lidar-bev", action="store_true",
                    help="append the LiDAR BEV (from the lidar run) as a 3rd right-most column")
    a = ap.parse_args()

    bf, lf = frames_of(a.base_dir), frames_of(a.lidar_dir)
    if not bf or not lf:
        print("missing frames", len(bf), len(lf)); sys.exit(1)
    N = min(max(len(bf), len(lf)), a.max)
    out = a.out or f"/home/s56cai/DeepSight/logs/cmp_{a.title or 'route'}.mp4"
    writer = None
    for i in range(N):
        L = render_side(bf[min(i, len(bf) - 1)], a.base_dir, "base", a.left_ds)
        R = render_side(lf[min(i, len(lf) - 1)], a.lidar_dir, "lidar", a.right_ds)
        H = max(L.shape[0], R.shape[0])
        L = cv2.copyMakeBorder(L, 0, H - L.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        R = cv2.copyMakeBorder(R, 0, H - R.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        sep = np.full((H, 4, 3), 60, np.uint8)
        parts = [L, sep, R]
        if a.lidar_bev:
            parts += [sep, render_bev(lf[min(i, len(lf) - 1)], a.lidar_dir, H)]
        canvas = np.hstack(parts)
        if a.title:
            cv2.putText(canvas, a.title, (canvas.shape[1] // 2 - 120, H - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        if writer is None:
            writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                     (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    writer.release()
    to_h264(out)
    print(f"wrote {N} frames -> {out} (H.264)")


if __name__ == "__main__":
    main()
