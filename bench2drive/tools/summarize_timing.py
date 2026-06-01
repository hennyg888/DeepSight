#!/usr/bin/env python3
"""
Read per-route wall-clock (meta.duration_system) from all task*.json under a run dir,
summarize, and extrapolate to a full 220-route multi-GPU run.

usage:
    python tools/summarize_timing.py --run_dir results_multi/<RUN_NAME> --total_routes 220
"""
import os
import json
import glob
import math
import argparse


def load_records(run_dir):
    records = []
    for jf in sorted(glob.glob(os.path.join(run_dir, "task*.json"))):
        try:
            data = json.load(open(jf))
        except Exception as e:
            print(f"  [warn] cannot read {jf}: {e}")
            continue
        task = os.path.basename(jf).replace(".json", "")
        for r in data.get("_checkpoint", {}).get("records", []):
            meta = r.get("meta", {}) or {}
            records.append({
                "task": task,
                "route_id": r.get("route_id"),
                "scenario": r.get("scenario_name"),
                "status": r.get("status"),
                "dur_sys": meta.get("duration_system"),   # real wall-clock seconds
                "dur_game": meta.get("duration_game"),     # simulation seconds
                "score": r.get("scores", {}).get("score_composed"),
            })
    return records


def fmt(sec):
    if sec is None:
        return "   n/a   "
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results_multi/<RUN_NAME>")
    ap.add_argument("--total_routes", type=int, default=220)
    args = ap.parse_args()

    recs = load_records(args.run_dir)
    if not recs:
        print(f"[error] no task*.json records found under {args.run_dir}")
        return

    print("\n" + "=" * 78)
    print(f" Per-route wall-clock (run_dir = {args.run_dir})")
    print("=" * 78)
    print(f" {'task':<7}{'route_id':<24}{'status':<26}{'wallclock':>10}{'  score':>8}")
    print("-" * 78)
    for r in sorted(recs, key=lambda x: (x["task"], x["route_id"] or "")):
        sc = f"{r['score']:.1f}" if isinstance(r["score"], (int, float)) else "n/a"
        print(f" {r['task']:<7}{str(r['route_id']):<24}{str(r['status']):<26}{fmt(r['dur_sys']):>10}{sc:>8}")

    durs = [r["dur_sys"] for r in recs if isinstance(r["dur_sys"], (int, float))]
    n = len(durs)
    if n == 0:
        print("\n[error] no duration_system field found, cannot estimate")
        return
    durs.sort()
    mean_t = sum(durs) / n
    median_t = durs[n // 2]
    max_t = durs[-1]
    min_t = durs[0]

    # actual parallel wall-clock recorded by the launcher
    wall = None
    try:
        ws = int(open(os.path.join(args.run_dir, "wall_start.txt")).read().strip())
        we = int(open(os.path.join(args.run_dir, "wall_end.txt")).read().strip())
        wall = we - ws
    except Exception:
        pass

    n_task = len({r["task"] for r in recs})

    print("\n" + "=" * 78)
    print(" Summary")
    print("=" * 78)
    print(f"  Routes evaluated:          {n}")
    print(f"  Tasks (GPUs) used:         {n_task}")
    print(f"  Per-route wall-clock  mean:   {fmt(mean_t)}  ({mean_t:.0f}s)")
    print(f"                        median: {fmt(median_t)}  ({median_t:.0f}s)")
    print(f"                        min:    {fmt(min_t)}  /  max: {fmt(max_t)}")
    if wall is not None:
        serial = sum(durs)
        speedup = serial / wall if wall else 0
        print(f"  Actual parallel wall-clock:   {fmt(wall)}  (serial would take {fmt(serial)}, measured speedup {speedup:.1f}x)")

    print("\n" + "=" * 78)
    print(f" Estimated wall-clock for a full {args.total_routes}-route multi-GPU run")
    print("=" * 78)
    print(f"  (based on mean {mean_t:.0f}s/route; each GPU runs its batch serially, wall-clock = busiest GPU)")
    print(f"  {'GPUs':>6}{'routes/GPU':>14}{'est. wall (mean)':>20}{'est. wall (max)':>20}")
    print("-" * 78)
    for ngpu in sorted({4, 6, 8, 9, n_task}):
        per_card = math.ceil(args.total_routes / ngpu)
        est_mean = per_card * mean_t
        est_max = per_card * max_t
        print(f"  {ngpu:>6}{per_card:>14}{fmt(est_mean):>20}{fmt(est_max):>20}")
    print("-" * 78)
    print(f"  Note: duration_system includes per-route model reload (~a few s) + CARLA sim + per-frame inference.")
    print(f"        Failed (timed out / crashed) routes can take abnormally long and inflate the mean;")
    print(f"        cross-check the table above and exclude them for a cleaner estimate.\n")


if __name__ == "__main__":
    main()
