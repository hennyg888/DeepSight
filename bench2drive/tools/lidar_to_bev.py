#!/usr/bin/env python3
"""
Convert a bench2drive LiDAR .laz file to a BEV pseudo-image (PNG).

Channels (RGB):
  R - height    (max z per pixel cell, clipped and normalized)
  G - intensity (mean LiDAR intensity per cell, normalized)
  B - density   (log-scaled point count per cell, normalized)

bench2drive camera images: 1600 × 900 (W × H).
Deepsight resizes to 644 x 364

Usage:
  python lidar_to_bev.py path/to/scene/lidar/00000.laz [output.png]
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

try:
    import laspy
except ImportError:
    sys.exit("laspy is required: pip install 'laspy[lazrs]'")

# ── BEV parameters matching bench2drive conventions ───────────────────────────
IMG_W, IMG_H = 644, 644      # should be multiple of 28 for patch size 14 with 2x2 merge in qwen 2.5 vl
# BEV_RANGE_M_ENV: override per-checkpoint lineage (35=base_lidar/sq/afz, 30=lidar30)
BEV_RANGE_M  = float(os.environ.get('BEV_RANGE_M', '30'))           # ±m in each direction
Z_MIN        = -3.0           # clip height below this (m)
Z_MAX        =  5.0           # clip height above this (m)

EGO_DOT_RADIUS =  IMG_W // (BEV_RANGE_M * 2)       # yellow ego dot radius in pixels, ~= 1m in real world
EGO_DOT_COLOR  = (255, 255, 0) # yellow

LOG_DENSITY_MAX = np.log1p(200)  # static ceiling for density channel (~N pts/cell → 255)
# ─────────────────────────────────────────────────────────────────────────────

#TODO:
#ensure no spatial distortions or scaling
#increase dot size to be larger, more than 1px
#ensure all info needed to generate are available at inference time
#make pixel color channel scaling absolute not relative to current frame, adjust fixed min / max so visually colors are distinct
#generate more frame samples

def load_points(laz_path: str) -> np.ndarray:
    """
    Read a .laz file and return (N, 4) float32: [ego_x, ego_y, z, intensity].

    Coordinate transform (from bench2drive visualize.py):
      lidar frame (x, y, z)  →  ego-car frame (y, -x, z)
    Intensity is read from the LAS uint16 field and normalised to [0, 1].
    """
    las = laspy.read(laz_path)

    raw_x = np.asarray(las.x, dtype=np.float32)
    raw_y = np.asarray(las.y, dtype=np.float32)
    z     = np.asarray(las.z, dtype=np.float32)

    # lidar → ego coordinate transform
    ego_x = raw_y
    ego_y = -raw_x

    # intensity is stored as uint16 (0–65535) in LAS point format 0
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float32) / 65535.0
    else:
        intensity = np.zeros(len(raw_x), dtype=np.float32)

    return np.stack([ego_x, ego_y, z, intensity], axis=1)


def bev_projection(points: np.ndarray) -> np.ndarray:
    """
    Project (N, 4) points to a (H, W, 3) uint8 BEV image.

    Uses a grid accumulation approach (PointPillars-style) rather than
    per-point drawing, so each pixel cell aggregates all points that fall
    into it before normalisation.
    """
    ego_x, ego_y, z, intensity = (points[:, i] for i in range(4))

    # ── 1. spatial filter ────────────────────────────────────────────────────
    mask = (
        (ego_x >= -BEV_RANGE_M) & (ego_x <= BEV_RANGE_M) &
        (ego_y >= -BEV_RANGE_M) & (ego_y <= BEV_RANGE_M)
    )
    ego_x, ego_y, z, intensity = ego_x[mask], ego_y[mask], z[mask], intensity[mask]

    # ── 2. map to pixel indices ───────────────────────────────────────────────
    # ego_x → column (width axis), ego_y → row (height axis)
    col = np.clip(
        ((ego_x + BEV_RANGE_M) / (2 * BEV_RANGE_M) * (IMG_W - 1)).astype(np.int32),
        0, IMG_W - 1,
    )
    row = np.clip(
        ((ego_y + BEV_RANGE_M) / (2 * BEV_RANGE_M) * (IMG_H - 1)).astype(np.int32),
        0, IMG_H - 1,
    )
    flat = row * IMG_W + col   # linear cell index
    N    = IMG_H * IMG_W

    # ── 3. per-cell aggregation ───────────────────────────────────────────────
    # density: point count
    density = np.bincount(flat, minlength=N).astype(np.float32)

    # height: max z per cell (scatter-max)
    z_clipped = np.clip(z, Z_MIN, Z_MAX)
    height = np.full(N, Z_MIN, dtype=np.float32)
    np.maximum.at(height, flat, z_clipped)

    # intensity: mean per cell
    int_sum = np.zeros(N, dtype=np.float32)
    np.add.at(int_sum, flat, intensity)
    count_nz = np.where(density > 0, density, 1.0)   # avoid div-by-zero
    mean_intensity = int_sum / count_nz

    # ── 4. normalise to uint8 ─────────────────────────────────────────────────
    def to_uint8(arr: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        arr = np.clip(arr, vmin, vmax)
        arr = (arr - vmin) / max(vmax - vmin, 1e-6)
        return (arr * 255).astype(np.uint8)

    ch_height    = to_uint8(height, Z_MIN, Z_MAX)
    ch_intensity = to_uint8(mean_intensity, 0.0, 1.0)

    log_density  = np.log1p(density)
    #print(f"Density: max {density.max():.1f} pts/cell  |  log1p max {log_density.max():.2f}")
    ch_density   = to_uint8(log_density, 0.0, LOG_DENSITY_MAX)

    # zero out empty cells in height and intensity channels so the
    # background stays black rather than carrying a Z_MIN colour
    empty = density == 0
    ch_height[empty]    = 0
    ch_intensity[empty] = 0

    # ── 5. assemble RGB image ─────────────────────────────────────────────────
    img = np.stack(
        [
            ch_height.reshape(IMG_H, IMG_W),     # R
            ch_intensity.reshape(IMG_H, IMG_W),  # G
            ch_density.reshape(IMG_H, IMG_W),    # B
        ],
        axis=2,
    )

    # ── 5b. expand each coloured (non-empty) pixel into its 4 neighbours ──────
    # overlapping contributions (a pixel reached by more than one coloured
    # source) are averaged rather than overwritten.
    src_mask = ~empty.reshape(IMG_H, IMG_W)
    img_f    = img.astype(np.float32)
    sum_acc   = np.zeros_like(img_f)
    count_acc = np.zeros((IMG_H, IMG_W), dtype=np.int32)

    for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
        src_r0, src_r1 = max(0, -dr), IMG_H - max(0, dr)
        dst_r0, dst_r1 = max(0, dr), IMG_H - max(0, -dr)
        src_c0, src_c1 = max(0, -dc), IMG_W - max(0, dc)
        dst_c0, dst_c1 = max(0, dc), IMG_W - max(0, -dc)

        shifted_mask = src_mask[src_r0:src_r1, src_c0:src_c1]
        shifted_img  = img_f[src_r0:src_r1, src_c0:src_c1, :]

        sum_acc[dst_r0:dst_r1, dst_c0:dst_c1, :] += np.where(shifted_mask[..., None], shifted_img, 0.0)
        count_acc[dst_r0:dst_r1, dst_c0:dst_c1]  += shifted_mask

    has_color = count_acc > 0
    img = np.zeros_like(img_f)
    img[has_color] = sum_acc[has_color] / count_acc[has_color, None]
    img = np.round(img).astype(np.uint8)

    # ── 6. draw yellow ego dot at the ego-vehicle position ────────
    pil = Image.fromarray(img, mode="RGB")
    draw = ImageDraw.Draw(pil)
    cx = int((0 + BEV_RANGE_M) / (2 * BEV_RANGE_M) * (IMG_W - 1))
    cy = int((0 + BEV_RANGE_M) / (2 * BEV_RANGE_M) * (IMG_H - 1))
    draw.ellipse(
        [cx - EGO_DOT_RADIUS, cy - EGO_DOT_RADIUS,
         cx + EGO_DOT_RADIUS, cy + EGO_DOT_RADIUS],
        fill=EGO_DOT_COLOR,
    )

    return np.array(pil)


DEFAULT_LAZ = (
    "/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted/SignalizedJunctionLeftTurnEnterFlow_Town12_Route887_Weather3/lidar/00003.laz"
    #"/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted/ConstructionObstacle_Town06_Route73_Weather21/lidar/00026.laz"
    #"/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted/BlockedIntersection_Town12_Route936_Weather0/lidar/00014.laz"
    #"/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted/CrossingBicycleFlow_Town12_Route1063_Weather23/lidar/00045.laz"
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a bench2drive LiDAR .laz file to a BEV PNG "
                    "(R=height, G=intensity, B=density).",
    )
    parser.add_argument("laz_file", nargs="?", default=DEFAULT_LAZ,
                        help="Input .laz file (default: hard-coded sample)")
    parser.add_argument(
        "output_png",
        nargs="?",
        default=None,
        help="Output PNG path (default: <input_stem>_bev.png)",
    )
    args = parser.parse_args()

    if args.output_png:
        out_path = args.output_png
    else:
        stem = os.path.splitext(os.path.basename(args.laz_file))[0]
        out_path = os.path.join(os.path.dirname(__file__), f"{stem}_bev.png")

    print(f"Loading:    {args.laz_file}")
    points = load_points(args.laz_file)
    print(f"Points:     {len(points):,}")

    print(f"Projecting: BEV {IMG_W}×{IMG_H} px  |  ±{BEV_RANGE_M} m  |  z [{Z_MIN}, {Z_MAX}] m")
    bev = bev_projection(points)

    Image.fromarray(bev, mode="RGB").save(out_path)
    print(f"Saved:      {out_path}")


if __name__ == "__main__":
    main()
