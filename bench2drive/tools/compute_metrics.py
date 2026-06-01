#!/usr/bin/env python3
"""
Compute bench2drive's four core metrics from a multi-GPU sharded result dir
(task*.json / route_*.json + each route's metric_info.json):
  DS (Driving Score) / SR (Success Rate) / Efficiency / Comfortness
Reuses the comfort computation from tools/efficiency_smoothness_benchmark.py.
For the Ability(%) 5-dimension metric use tools/ability_benchmark.py (needs CARLA + full 220).

Usage:
  python tools/compute_metrics.py --run_dir results_multi/<RUN_NAME>
  python tools/compute_metrics.py --run_dir <dir> --total 220   # use 220 as denominator for the official benchmark
"""
import os
import sys
import json
import glob
import re
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from efficiency_smoothness_benchmark import seg_compute_comfort_metric


def load_records(run_dir):
    # Works for both static shards (task*.json) and dynamic scheduling (route_*.json);
    # skips merged.json and any non-result json
    recs = []
    for jf in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        if os.path.basename(jf) == "merged.json":
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if "_checkpoint" not in d:
            continue
        recs.extend(d.get("_checkpoint", {}).get("records", []))
    return recs


def is_success(r):
    # Official definition (merge_route_json.py): status in {Completed, Perfect}
    # and no infraction other than min_speed
    if r.get("status") not in ("Completed", "Perfect"):
        return False
    for k, v in r.get("infractions", {}).items():
        if k != "min_speed_infractions" and len(v) > 0:
            return False
    return True


def efficiency_of(r):
    # Official definition: mean of "X% of surrounding traffic" in min_speed_infractions (drop >1000%)
    vals = []
    for s in r.get("infractions", {}).get("min_speed_infractions", []):
        m = re.search(r"\b\d+\.?\d*%", s)
        if not m:
            continue
        p = float(m.group().rstrip("%"))
        if p > 1000:
            continue
        vals.append(p)
    return sum(vals) / len(vals) if vals else None


def find_metric_info(run_dir, route_id):
    # record save_name carries a prefix (drivetransformer_..._only_traj_) and the dir lives
    # under Scenarios/, so joining the path directly fails -> fuzzy glob by route_id
    c = glob.glob(os.path.join(run_dir, "*", "Scenarios", f"*{route_id}*", "metric_info.json"))
    return c[0] if c else None


def comfort_of(mi_path):
    d = json.load(open(mi_path))
    keys = ["acceleration", "angular_velocity", "forward_vector", "right_vector", "location", "rotation"]
    data = {k: [] for k in keys}
    for fr in d.values():
        for k in keys:
            data[k].append(fr[k])
    data = {k: np.array(v) for k, v in data.items()}
    return seg_compute_comfort_metric(**data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--total", type=int, default=0, help="denominator (0 = actual #routes evaluated; use 220 for the official benchmark)")
    a = ap.parse_args()

    recs = load_records(a.run_dir)
    n = len(recs)
    if n == 0:
        print(f"[error] no task*.json/route_*.json records found under {a.run_dir}")
        return
    denom = a.total if a.total > 0 else n
    succ = sum(is_success(r) for r in recs)

    ds = sum(r["scores"]["score_composed"] for r in recs) / denom
    sr = succ / denom

    effs = [e for e in (efficiency_of(r) for r in recs) if e is not None]
    eff = sum(effs) / len(effs) if effs else float("nan")

    comforts, miss = [], 0
    for r in recs:
        mi = find_metric_info(a.run_dir, r["route_id"])
        if mi:
            comforts.append(comfort_of(mi))
        else:
            miss += 1
    comf = sum(comforts) / len(comforts) if comforts else float("nan")

    print("=" * 60)
    print(f" bench2drive metrics  (run_dir={a.run_dir})")
    print("=" * 60)
    print(f"  routes evaluated : {n}   (denominator = {denom})")
    print(f"  DS  Driving Score: {ds:.2f}")
    print(f"  SR  Success Rate : {sr * 100:.2f}%   ({succ}/{denom})")
    print(f"  Efficiency       : {eff:.2f}   (avg % of surrounding-traffic speed; {len(effs)} routes with data)")
    print(f"  Comfortness      : {comf:.4f}  (0~1 comfort pass-rate; {len(comforts)} routes{', ' + str(miss) + ' missing metric_info' if miss else ''})")
    print("=" * 60)
    if a.total == 0:
        print("  Note: denominator = actual #routes. For the official bench2drive220 comparison add --total 220.")
    elif n != a.total:
        print(f"  WARNING: only {n} routes but denominator={a.total}; DS/SR will be underestimated (missing counted as 0).")


if __name__ == "__main__":
    main()
