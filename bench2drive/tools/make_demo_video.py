#!/usr/bin/env python3
"""Official-style demo MP4 from a saved eval scenario (front camera + HUD text).

Mirrors tools/generate_video.py (front view + speed/steer/throttle/brake overlay),
but reads the layout the 220 runs actually saved: camera/CAM_FRONT/*.jpg + meta/*.json.

Usage:
  python tools/make_demo_video.py <scenario_dir> [out.mp4] [--fps 10]
"""
import sys, os, glob, json, argparse, subprocess, shutil
import cv2

# Bundled ffmpeg (from imageio-ffmpeg in .venv). cv2's mp4v isn't playable in
# VSCode/browser HTML5 preview; re-encode to H.264 so it previews inline.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario_dir")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    sd = a.scenario_dir.rstrip("/")
    out = a.out or os.path.join("/home/s56cai/DeepSight/logs", os.path.basename(sd) + ".mp4")

    frames = sorted(glob.glob(os.path.join(sd, "camera", "CAM_FRONT", "*.jpg")))
    if not frames:
        print("no CAM_FRONT frames in", sd); sys.exit(1)

    h, w = cv2.imread(frames[0]).shape[:2]
    video = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
    n = 0
    for fp in frames:
        idx = int(os.path.splitext(os.path.basename(fp))[0])   # 5-digit frame index
        img = cv2.imread(fp)
        mp = os.path.join(sd, "meta", f"{idx:04d}.json")
        if os.path.exists(mp):
            m = json.load(open(mp))
            txt = (f"speed: {float(m['speed']):.2f}  steer: {float(m['steer']):.2f}  "
                   f"throttle: {float(m['throttle']):.2f}  brake: {float(m['brake']):.2f}")
            cv2.putText(img, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        video.write(img)
        n += 1
    video.release()
    to_h264(out)
    print(f"wrote {n} frames @ {a.fps}fps -> {out} (H.264)")


if __name__ == "__main__":
    main()
