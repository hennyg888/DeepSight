"""Single-GPU worker for parallel replay: owns one CARLA server, works through one shard of
scenario dirs against it sequentially (each scenario from frame 0 to completion), restarting
the server if it dies (see replay_scenario.py's known Large-Map segfault risk) rather than
taking down the whole shard. Invoked by replay_parallel.py, one process per GPU.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

import carla

REPLAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'replay_scenario.py')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--shard_file', required=True, help='JSON file with {"scenarios": [dir, ...]}')
    p.add_argument('--gpu_rank', type=int, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--host', default='localhost')
    p.add_argument('--carla_sh', default='/home/hhguo/Carla/CarlaUE4.sh')
    p.add_argument('--startup_timeout', type=float, default=120.0, help='seconds to wait for CARLA to come up')
    p.add_argument('--scenario_timeout', type=float, default=1200.0, help='seconds before killing a stuck replay_scenario.py call')
    p.add_argument('--simple_replay', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--results_file', required=True)
    p.add_argument('--log_dir', required=True)
    p.add_argument('--carla_log', required=True)
    return p.parse_args()


def launch_carla(carla_sh, port, gpu_rank, log_path):
    log_f = open(log_path, 'a')
    proc = subprocess.Popen(
        [carla_sh, '-RenderOffScreen', '-nosound', '-fps=10',
         f'-carla-rpc-port={port}', f'-graphicsadapter={gpu_rank}'],
        stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True,
    )
    return proc, log_f


def kill_carla(proc):
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def server_alive(host, port):
    try:
        client = carla.Client(host, port)
        client.set_timeout(3.0)
        client.get_server_version()
        return True
    except RuntimeError:
        return False


def wait_for_server(host, port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_alive(host, port):
            return True
        time.sleep(2.0)
    return False


def replay_one(scenario_dir, host, port, simple_replay, timeout, log_path):
    cmd = [sys.executable, REPLAY_SCRIPT, '--scenario_dir', scenario_dir, '--host', host, '--port', str(port),
           '--simple_replay' if simple_replay else '--no-simple_replay']
    with open(log_path, 'w') as log_f:
        try:
            result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            log_f.write('\n[worker] timed out, killed\n')
            return False


def main():
    args = parse_args()
    with open(args.shard_file) as f:
        shard = json.load(f)['scenarios']
    os.makedirs(args.log_dir, exist_ok=True)

    carla_proc, carla_log_f = launch_carla(args.carla_sh, args.port, args.gpu_rank, args.carla_log)
    if not wait_for_server(args.host, args.port, args.startup_timeout):
        raise RuntimeError(f'[gpu{args.gpu_rank}] CARLA server on port {args.port} did not come up within {args.startup_timeout}s')

    results = {}
    for scenario_dir in shard:
        name = os.path.basename(os.path.normpath(scenario_dir))

        if not server_alive(args.host, args.port):
            print(f'[gpu{args.gpu_rank}] server down, restarting before {name}')
            kill_carla(carla_proc)
            carla_proc, carla_log_f = launch_carla(args.carla_sh, args.port, args.gpu_rank, args.carla_log)
            if not wait_for_server(args.host, args.port, args.startup_timeout):
                results[name] = 'server_restart_failed'
                print(f'[gpu{args.gpu_rank}] {name}: server_restart_failed')
                continue

        log_path = os.path.join(args.log_dir, f'{name}.log')
        ok = replay_one(scenario_dir, args.host, args.port, args.simple_replay, args.scenario_timeout, log_path)
        results[name] = 'ok' if ok else 'failed'
        print(f'[gpu{args.gpu_rank}] {name}: {results[name]}')

        with open(args.results_file, 'w') as f:
            json.dump(results, f, indent=2)

    kill_carla(carla_proc)
    carla_log_f.close()
    with open(args.results_file, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
