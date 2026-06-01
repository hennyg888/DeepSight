# Bench2Drive Evaluation Guide

How we evaluate the Qwen-based driving agent on Bench2Drive (CARLA closed-loop),
which scripts to use, what each metric means, and where the current results live.

> **Paths**: the scripts hard-code absolute paths for *our* machine
> (`/home/s56cai/DeepSight` project root, `/home/s56cai/DeepSight/.venv-b2d` Python env,
> `/home/s56cai/ckpt/DeepSight` and `.../saves/.../base/checkpoint-3148` model checkpoints,
> `/home/s56cai/Carla` CARLA install). Each script has a `NOTE for collaborators` header —
> adjust these to your environment before running.

---

## 1. Pipeline at a glance

1. **Run** the agent on routes → each route produces a result `json` (+ a per-frame
   `metric_info.json` used for the comfort metric).
2. **Compute metrics** from those json files: DS / SR / Efficiency / Comfortness / Ability.

CARLA is started/managed automatically by `leaderboard_evaluator.py` as a subprocess —
you do **not** start CARLA by hand.

---

## 2. Scripts

### 2.1 Running evaluation

| Script | What it does |
|---|---|
| `leaderboard/scripts/run_evaluation.sh` | Low-level launcher (official). Runs `leaderboard_evaluator.py` for one route xml on one GPU/port. All the scripts below call it. `RESUME=True` lets it skip already-finished routes. |
| `leaderboard/scripts/run_evaluation_qwen.sh` | **Single-GPU** run of one route file (default: dev10). Edit `TEAM_CONFIG` (checkpoint) and `BASE_ROUTES` here. Good for a first smoke test. |
| `leaderboard/scripts/run_evaluation_qwen_multi.sh` | **Multi-GPU static sharding** for dev10: splits the 10 routes into N shards (one per GPU), each GPU = one CARLA + one Qwen. Auto-runs timing stats at the end. |
| `leaderboard/scripts/run_220_dynamic.sh` | **Multi-GPU dynamic scheduling** for the full 220 (recommended for 220). A `flock` queue + 9 workers do work-stealing: whichever GPU is free grabs the next route, so slow routes don't bottleneck. Supports **resume** (`bash run_220_dynamic.sh <RUN_DIR>` continues unfinished routes) and **auto-retries CARLA crashes** (2x). Auto-computes DS/SR/Efficiency/Comfortness at the end. **Run it inside `tmux`/`nohup`** — a 220 run takes ~14h and an SSH drop will kill it. |
| `leaderboard/scripts/rerun_tasks.sh` | Re-run specific shards/routes into an existing run dir (e.g. to recover CARLA-crashed routes). `bash rerun_tasks.sh <RUN_DIR> <task_ids...>`. |

### 2.2 Computing metrics

| Script | What it does |
|---|---|
| `tools/compute_metrics.py` | One-shot **DS / SR / Efficiency / Comfortness** from a run dir. Handles both static shards (`task*.json`) and dynamic (`route_*.json`), and the `save_name` prefix quirk for finding `metric_info.json`. Usage: `python tools/compute_metrics.py --run_dir <dir> --total 220`. |
| `tools/ability_benchmark.py` | **Ability** 5 dimensions + Mean. Maps each route's scenario type into 5 buckets and computes per-bucket success rate. Needs `merged.json` and starts CARLA (to compute junction completion for the Traffic_Signs bucket). Output: `<RUN_DIR>/ability.json`. |
| `tools/merge_route_json.py` | Merges all per-route json in a dir into `merged.json` (also prints DS/SR, **hard-coded ÷220** so only correct for a full 220 run). Needed as input to `ability_benchmark.py`. |
| `tools/efficiency_smoothness_benchmark.py` | Official script `compute_metrics.py` reuses for the comfort computation; can also be run directly for Efficiency + Comfortness. |
| `tools/summarize_timing.py` | Per-route wall-clock table + extrapolated time for a full 220 run, read from `meta.duration_system`. |
| `leaderboard/scripts/eval_all.sh` | **One command for ALL 5 metric classes** after a run: merge → compute_metrics (DS/SR/Efficiency/Comfortness) → ability_benchmark (Ability). Usage: `bash leaderboard/scripts/eval_all.sh <RUN_DIR>`. |
| `tools/split_xml.py` | Splits a route xml into N shards (used by the multi-GPU scripts). |
| `tools/clean_carla.sh` | Kills leftover CARLA / evaluator processes. Run before each launch. |

---

## 3. Metrics explained

| Metric | Meaning | How it's computed | Source in json |
|---|---|---|---|
| **DS** (Driving Score ↑) | Overall driving quality (route completion × infraction penalty) | mean of `score_composed` over all routes (÷220 for the official benchmark) | `records[].scores.score_composed` |
| **SR** (Success Rate % ↑) | Fraction of routes finished cleanly | count of `status ∈ {Completed, Perfect}` **and** no infraction (other than min-speed), ÷ N | `records[].status` + `.infractions` |
| **Efficiency ↑** | How fast the ego drives vs surrounding traffic | mean of "X% of surrounding traffic" in `min_speed_infractions` (drop >1000%), averaged over routes | `records[].infractions.min_speed_infractions` |
| **Comfortness ↑** | Smoothness of motion | per-frame check that 6 quantities (lon/lat accel, jerk, yaw rate/accel) stay within human-comfort thresholds; reported as a 0–1 pass-rate | each route's `metric_info.json` |
| **Ability (%) ↑** | Per-capability success (Merging / Overtaking / Emergency Brake / Give Way / Traffic Sign + Mean) | route scenario types are bucketed into the 5 abilities; per-bucket success rate | `merged.json` records + `bench2drive220.xml` |

---

## 4. Where the results are

> Results live under the run machine's `/home/s56cai/DeepSight/bench2drive/`.

### 4.1 Full benchmark (220 routes) — `results_220/`

**`results_220/deepsight_cot_220_0601_0027/`** — the DeepSight **CoT** checkpoint
(`/home/s56cai/ckpt/DeepSight`) evaluated on all 220 routes.

Current scores:

```
routes evaluated : 220
DS  Driving Score : 79.66
SR  Success Rate  : 55.45%  (122/220)
Efficiency        : 196.82  (avg % of surrounding-traffic speed; 219 routes with data)
Comfortness       : 0.1883  (0~1 comfort pass-rate; 220 routes)

Ability (%):
  Overtaking       : 0.667
  Merging          : 0.450
  Emergency_Brake  : 0.633
  Give_Way         : 0.200
  Traffic_Signs    : 0.616
  Mean             : 0.513
```

(Ability details saved in `results_220/deepsight_cot_220_0601_0027/ability.json`.)

### 4.2 Quick validation (dev10, 10 routes) — `results_multi/`

dev10 is a 10-route subset for fast pipeline checks — too small for statistically
meaningful Ability, but good for sanity-checking DS/SR/Efficiency/Comfortness.

| Directory | Model | Notes |
|---|---|---|
| `results_multi/deepsight_cot_dev10_multi_0531_1746/` | DeepSight **CoT** checkpoint | dev10 result. Metrics: DS 90.63 / SR 80% / Efficiency 129.35 / Comfortness 0.1289 |
| `results_multi/deepsight_base_dev10_multi_0531_0235/` | **Base** checkpoint trained from scratch on the base dataset (1 epoch) | dev10 result. Run `compute_metrics.py --run_dir <dir>` for its numbers. |

### 4.3 Layout inside one run dir

```
<RUN_DIR>/
  route_<idx>.json     # (220 dynamic) per-route result   |  task<i>.json (dev10 static shards)
  route_<idx>/Scenarios/.../metric_info.json   # per-frame motion data (for Comfortness)
  route_<idx>.log      # per-route stdout/stderr
  merged.json          # all routes merged (created by eval_all.sh; input to Ability)
  ability.json         # Ability 5-dim result (created by ability_benchmark.py)
  wall_start.txt / wall_end.txt   # timing
```

---

## 5. Typical commands

```bash
# activate env (has carla); paths are for our machine — adjust to yours
source /home/s56cai/DeepSight/.venv-b2d/bin/activate
cd /home/s56cai/DeepSight/bench2drive

# --- quick dev10 sanity check (multi-GPU) ---
bash tools/clean_carla.sh
bash leaderboard/scripts/run_evaluation_qwen_multi.sh

# --- full 220 (run inside tmux! ~14h) ---
bash tools/clean_carla.sh
bash leaderboard/scripts/run_220_dynamic.sh

# --- compute all 5 metric classes for a finished run ---
bash leaderboard/scripts/eval_all.sh results_220/deepsight_cot_220_0601_0027

# --- just DS/SR/Efficiency/Comfortness, any time ---
python tools/compute_metrics.py --run_dir <RUN_DIR> --total 220
```
