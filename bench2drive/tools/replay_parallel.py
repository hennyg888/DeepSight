"""Parallelize replay_scenario.py across GPUs: one CARLA server + one replay_worker.py per
GPU, each churning through its own static shard of scenario dirs (round-robin split), one
scenario at a time from frame 0 to completion. See replay_worker.py for per-GPU crash
recovery (CARLA server restart on death, per-scenario timeout).

Usage:
    python replay_parallel.py --scenarios_root ../Bench2Drive-base-partial
    python replay_parallel.py --scenarios_root /home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted \
        --scenario_list ../bad_weather_dirs.txt
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay_worker.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scenarios_root', default='/home/s56cai/DeepSight/bench2drive/Bench2Drive-base-extracted',
                    help='dir containing scenario subfolders, each with an anno/ subdir')
    p.add_argument('--scenario_list', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bad_weather_dirs.txt'),
                    help='text file, one scenario folder name per line (relative to --scenarios_root); '
                         'pass --scenario_list "" to instead replay every subfolder of --scenarios_root with an anno/ dir')
    p.add_argument('--gpu_rank_list', type=int, nargs='+', default=list(range(9)))
    p.add_argument('--base_port', type=int, default=30000, help='+150 per task to avoid CARLA port clashes')
    p.add_argument('--host', default='localhost')
    p.add_argument('--carla_sh', default='/home/hhguo/Carla/CarlaUE4.sh')
    p.add_argument('--startup_timeout', type=float, default=120.0)
    p.add_argument('--scenario_timeout', type=float, default=1200.0)
    p.add_argument('--simple_replay', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stagger', type=float, default=8.0, help='seconds between launching each task, so GPUs don\'t all boot CARLA at once')
    p.add_argument('--output_dir', default=None, help='defaults to Bench2Drive/Replayed_runs/{timestamp}')
    return p.parse_args()


def discover_scenarios(root):
    return sorted(p for p in glob.glob(os.path.join(root, '*')) if os.path.isdir(os.path.join(p, 'anno')))


def load_scenario_list(root, list_file):
    scenarios = []
    with open(list_file) as f:
        names = [line.strip() for line in f if line.strip()]
    for name in names:
        path = os.path.join(root, name)
        if not os.path.isdir(os.path.join(path, 'anno')):
            print(f'[skip] {name!r} from {list_file} has no anno/ dir under {root}, skipping')
            continue
        scenarios.append(path)
    return scenarios


def shard(scenarios, n):
    shards = [[] for _ in range(n)]
    for i, s in enumerate(scenarios):
        shards[i % n].append(s)
    return shards


def main():
    args = parse_args()
    if args.scenario_list:
        scenarios = load_scenario_list(args.scenarios_root, args.scenario_list)
    else:
        scenarios = discover_scenarios(args.scenarios_root)
    if not scenarios:
        raise FileNotFoundError(f'no scenario dirs (with an anno/ subdir) found under {args.scenarios_root}')
    n = len(args.gpu_rank_list)
    shards = shard(scenarios, n)
    source = args.scenario_list or args.scenarios_root
    print(f'found {len(scenarios)} scenarios from {source}, split across {n} GPUs: {args.gpu_rank_list}')

    run_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / 'Replayed_runs' / datetime.now().strftime('%m%d_%H%M')
    run_dir.mkdir(parents=True, exist_ok=True)

    procs = []
    for i, gpu in enumerate(args.gpu_rank_list):
        shard_file = run_dir / f'shard{i}.json'
        with open(shard_file, 'w') as f:
            json.dump({'scenarios': shards[i]}, f, indent=2)
        log_dir = run_dir / f'task{i}_logs'
        log_dir.mkdir(exist_ok=True)
        port = args.base_port + i * 150

        cmd = [
            sys.executable, WORKER_SCRIPT,
            '--shard_file', str(shard_file), '--gpu_rank', str(gpu), '--port', str(port), '--host', args.host,
            '--carla_sh', args.carla_sh, '--startup_timeout', str(args.startup_timeout),
            '--scenario_timeout', str(args.scenario_timeout),
            '--simple_replay' if args.simple_replay else '--no-simple_replay',
            '--results_file', str(run_dir / f'results{i}.json'), '--log_dir', str(log_dir),
            '--carla_log', str(run_dir / f'carla{i}.log'),
        ]
        print(f'[launch] task{i} gpu={gpu} port={port} scenarios={len(shards[i])}')
        worker_log_f = open(run_dir / f'task{i}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=worker_log_f, stderr=subprocess.STDOUT)
        procs.append((proc, worker_log_f))
        time.sleep(args.stagger)

    print(f'[wait] all {n} tasks launched, waiting... (live: tail -f {run_dir}/task0.log)')
    for proc, log_f in procs:
        proc.wait()
        log_f.close()

    total_ok, total_failed = 0, 0
    for i in range(n):
        with open(run_dir / f'results{i}.json') as f:
            results = json.load(f)
        ok = sum(1 for v in results.values() if v == 'ok')
        failed = len(results) - ok
        total_ok += ok
        total_failed += failed
        if failed:
            bad = [name for name, v in results.items() if v != 'ok']
            print(f'task{i}: {ok} ok, {failed} failed -> {bad}')

    print(f'[done] {total_ok} ok, {total_failed} failed, out of {len(scenarios)} total. Results in {run_dir}')


if __name__ == '__main__':
    main()
