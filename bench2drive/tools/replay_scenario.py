"""Kinematic replay of a Bench2Drive-mini scenario from its anno/*.json.gz ground truth.

Every actor (ego, NPC vehicles/pedestrians, scenario props) is spawned once and then
teleported to its recorded world-space pose each frame, with physics disabled. Nothing is
re-simulated or driven by scenario_runner/leaderboard -- this only talks to the bare CARLA
client API. See ../replay_plan.md for the full design.

With --record (default on), the same camera suite as tools/data_collect.py is attached to
the ego and saved to Replayed/{scenario_name}/, using the same file layout/formats as the
camera/ subtree documented for the original benchmark (rgb/depth/semantic/instance jpg+png).
Unlike tools/data_collect.py, lidar and radar are intentionally not attached/recorded here
(camera-only replay) -- no laspy/h5py dependency needed. Requires the same extra dep as
tools/data_collect.py's camera path: opencv-python.

--simple_replay (default on) further limits recording to just TOP_DOWN rgb, FRONT rgb, and
FRONT depth -- enough for a quick sanity check without spawning/rendering the full 6-view x
4-modality suite. Pass --no-simple_replay for the full camera suite.

Targets CARLA 0.9.16 (see ../replay_plan.md for how this differs from upstream Bench2Drive's
0.9.15 target).

Usage:
    cd tools
    python replay_scenario.py \
        --scenario_dir ../Bench2Drive-mini/ControlLoss_Town11_Route401_Weather11 \
        --num_frames 10 --weather ClearNoon

DONE:
find all "bad" weather patterns where bev is degraded:
8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 25
8, 14, - rain ~11%
9, 11, 12, 13 - fog ~18%
10, 19, 20, 21, 22, 23, 25 - night ~31%
parallelize script and carla server runner + error handling and imperfect replay record scenario name
more testing: pedestrians, bikes, kid runs out at night
"""
import argparse
import glob
import gzip
import json
import math
import os
import queue
import re
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import convert_depth

FRAME_RATE = 10.0
SENSOR_WARMUP_TICKS = 10
TILE_STREAM_TICKS = 20  # let Large Map tiles (Town11-13/15) stream in around the hero actor

# Recorded z sits fractionally inside/below this CARLA build's ground mesh (observed: a
# location.z of 0.1543 collides on spawn in Town03 under 0.9.16, but +0.1 clears it) --
# try_spawn_actor collision-checks against world geometry, set_transform doesn't, so spawn
# elevated then teleport down to the exact recorded pose once physics is disabled.
SPAWN_Z_OFFSETS = (0.0, 0.2, 0.5, 1.0)

CAMERA_VIEWS = [
    # name, x, y, z, roll, pitch, yaw, fov
    ('FRONT', 0.80, 0.0, 1.60, 0.0, 0.0, 0.0, 70),
    ('FRONT_LEFT', 0.27, -0.55, 1.60, 0.0, 0.0, -55.0, 70),
    ('FRONT_RIGHT', 0.27, 0.55, 1.60, 0.0, 0.0, 55.0, 70),
    ('BACK', -2.0, 0.0, 1.60, 0.0, 0.0, 180.0, 110),
    ('BACK_LEFT', -0.32, -0.55, 1.60, 0.0, 0.0, -110.0, 70),
    ('BACK_RIGHT', -0.32, 0.55, 1.60, 0.0, 0.0, 110.0, 70),
]
CAMERA_WIDTH, CAMERA_HEIGHT = 1600, 900


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scenario_dir', required=True, help='path to an extracted scenario folder, e.g. Bench2Drive-mini/ControlLoss_Town11_Route401_Weather11')
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=2000)
    p.add_argument('--start_frame', type=int, default=0, help='skip to this frame index in the recorded anno sequence before starting the replay (default: 0)')
    p.add_argument('--num_frames', type=int, default=None, help='limit to N recorded frames starting at --start_frame (default: all remaining)')
    p.add_argument('--weather', default='ClearNoon', help='name of a carla.WeatherParameters preset, e.g. ClearNoon, ClearNight, WetNoon')
    p.add_argument('--no_record', action='store_true', help='skip camera capture, only replay actor poses')
    p.add_argument('--simple_replay', action=argparse.BooleanOptionalAction, default=True,
                    help='only record TOP_DOWN rgb + FRONT rgb + FRONT depth (default). Pass --no-simple_replay for the full 6-view/4-modality camera suite.')
    p.add_argument('--output_dir', default=None, help='defaults to Bench2Drive/Replayed/{scenario_name}')
    return p.parse_args()


def town_from_scenario_dir(scenario_dir):
    name = os.path.basename(os.path.normpath(scenario_dir))
    match = re.search(r'_(Town\w+)_Route', name)
    if not match:
        raise ValueError(f"could not parse town from scenario dir name '{name}'")
    return match.group(1)


def load_frames(scenario_dir, start_frame, num_frames):
    anno_dir = os.path.join(scenario_dir, 'anno')
    end_frame = start_frame + num_frames if num_frames is not None else None
    frame_files = sorted(glob.glob(os.path.join(anno_dir, '*.json.gz')))[start_frame:end_frame]
    if not frame_files:
        raise FileNotFoundError(f'no anno frames found in {anno_dir} for start_frame={start_frame}, num_frames={num_frames}')
    frames = []
    for path in frame_files:
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            frames.append(json.load(f))
    return frames


def matrix_to_location_rotation(matrix):
    # Inverse of tools/utils.py:get_matrix's pitch/roll/yaw -> matrix convention.
    x, y, z = matrix[0, 3], matrix[1, 3], matrix[2, 3]
    pitch = math.degrees(math.asin(np.clip(matrix[2, 0], -1.0, 1.0)))
    yaw = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
    roll = math.degrees(math.atan2(-matrix[2, 1], matrix[2, 2]))
    return (x, y, z), (pitch, roll, yaw)


def resolve_transform(entry):
    # docs/anno.md (upstream Bench2Drive): static vehicles' location/rotation are wrong due
    # to a CARLA API bug baked into the recording; the world2vehicle matrix is not, so recover
    # the true pose from its inverse instead. This is an artifact of how the anno data was
    # originally recorded, independent of which CARLA version replays it.
    if entry.get('state') == 'static' and 'world2vehicle' in entry:
        vehicle2world = np.linalg.inv(np.array(entry['world2vehicle']))
        (x, y, z), (pitch, roll, yaw) = matrix_to_location_rotation(vehicle2world)
    else:
        x, y, z = entry['location']
        pitch, roll, yaw = entry['rotation']
    return carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


def spawn_at(world, bp, transform):
    """try_spawn_actor collision-checks against world geometry; recorded z can sit just
    inside the ground mesh, so retry a few millimeters-to-meters higher before giving up."""
    for dz in SPAWN_Z_OFFSETS:
        loc = transform.location
        raised = carla.Transform(carla.Location(x=loc.x, y=loc.y, z=loc.z + dz), transform.rotation)
        actor = world.try_spawn_actor(bp, raised)
        if actor is not None:
            return actor
    return None


def get_blueprint(world, type_id, color, role_name='replay'):
    library = world.get_blueprint_library()
    if type_id.startswith('/Game/'):
        # Parked-vehicle decorations (leaderboard/leaderboard/utils/parked_vehicles.py) are
        # static meshes, not spawnable vehicle blueprints -- the anno data records their mesh
        # path as type_id (tools/data_collect.py:1336) because that's what CARLA itself uses:
        # they're spawned via the generic static.prop.mesh blueprint with mesh_path set.
        bp = library.filter('static.prop.mesh')[0]
        bp.set_attribute('mesh_path', type_id)
        return bp
    try:
        bp = library.find(type_id)
    except IndexError:
        return None
    if color and bp.has_attribute('color'):
        bp.set_attribute('color', color)
    if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', role_name)
    return bp


TRAFFIC_LIGHT_STATES = {
    0: carla.TrafficLightState.Red,
    1: carla.TrafficLightState.Yellow,
    2: carla.TrafficLightState.Green,
}


def apply_traffic_light(world, entry):
    state = TRAFFIC_LIGHT_STATES.get(entry.get('state'))
    if state is None:
        return
    loc = carla.Location(*entry['location'])
    lights = list(world.get_actors().filter('traffic.traffic_light*'))
    if not lights:
        return
    nearest = min(lights, key=lambda tl: tl.get_location().distance(loc))
    nearest.set_state(state)


def apply_frame(world, frame, actors):
    """Spawn/teleport/destroy actors so `actors` matches this frame's bounding_boxes."""
    seen_ids = set()
    # Ego first: on CARLA's Large Maps (Town11-13/15), tile streaming follows whichever
    # actor is tagged 'hero'/'ego_vehicle' -- spawning it before anything else means the
    # correct tile is already loading by the time NPCs/sensors show up at the same location.
    entries = sorted(frame['bounding_boxes'], key=lambda e: e['class'] != 'ego_vehicle')
    for entry in entries:
        cls = entry['class']
        type_id = entry.get('type_id')

        if cls == 'traffic_light':
            apply_traffic_light(world, entry)
            continue
        if cls == 'traffic_sign' and not type_id.startswith('static.'):
            continue  # permanent map sign/light, not scenario-placed content

        actor_id = entry['id']
        seen_ids.add(actor_id)
        transform = resolve_transform(entry)

        actor = actors.get(actor_id)
        if actor is None:
            role_name = 'hero' if cls == 'ego_vehicle' else 'replay'
            bp = get_blueprint(world, type_id, entry.get('color'), role_name=role_name)
            if bp is None:
                print(f'  [skip] unknown blueprint {type_id!r} (id {actor_id})')
                continue
            actor = spawn_at(world, bp, transform)
            if actor is None:
                print(f'  [skip] could not spawn {type_id!r} (id {actor_id})')
                continue
            if actor.type_id.startswith(('vehicle.', 'walker.')):
                try:
                    actor.set_simulate_physics(False)
                except RuntimeError:
                    pass
            actor.set_transform(transform)  # snap down to the exact recorded pose (set_transform isn't collision-checked)
            actors[actor_id] = actor
        else:
            actor.set_transform(transform)

    for actor_id in list(actors):
        if actor_id not in seen_ids:
            actors.pop(actor_id).destroy()


# --- Sensor recording -------------------------------------------------------
# Mirrors tools/data_collect.py's Env_Manager.sensors()/tick()/save() camera path so the
# replayed output matches the original benchmark's file layout/formats. Lidar and radar are
# deliberately not attached -- this replay only records cameras.

def build_sensor_specs(simple_replay=True):
    # simple_replay: just enough for a quick sanity check (TOP_DOWN rgb + FRONT rgb/depth)
    # instead of spawning/rendering the full 6-view x 4-modality suite.
    views = [v for v in CAMERA_VIEWS if v[0] == 'FRONT'] if simple_replay else CAMERA_VIEWS
    kinds = (('sensor.camera.rgb', ''), ('sensor.camera.depth', '_DEPTH'))
    if not simple_replay:
        kinds += (('sensor.camera.semantic_segmentation', '_SEM_SEG'), ('sensor.camera.instance_segmentation', '_INS_SEG'))

    specs = []
    for name, x, y, z, roll, pitch, yaw, fov in views:
        for kind, suffix in kinds:
            tag = f'CAM_{name}{suffix}'
            specs.append({
                'id': tag, 'type': kind, 'x': x, 'y': y, 'z': z,
                'roll': roll, 'pitch': pitch, 'yaw': yaw,
                'attributes': {'image_size_x': CAMERA_WIDTH, 'image_size_y': CAMERA_HEIGHT, 'fov': fov, 'role_name': tag},
            })
    specs.append({
        'id': 'TOP_DOWN', 'type': 'sensor.camera.rgb', 'x': 0.0, 'y': 0.0, 'z': 50.0,
        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
        'attributes': {'image_size_x': CAMERA_WIDTH, 'image_size_y': CAMERA_HEIGHT, 'fov': 110, 'role_name': 'TOP_DOWN'},
    })
    return specs


def parse_sensor_data(data):
    if isinstance(data, carla.Image):
        array = np.frombuffer(data.raw_data, dtype=np.uint8)
        return np.reshape(array, (data.height, data.width, 4)).copy()
    raise TypeError(f'unsupported sensor data type: {type(data)}')


class SensorSync:
    """Minimal standalone stand-in for leaderboard's SensorInterface (avoids depending
    on the leaderboard/scenario_runner packages for a plain client-side replay)."""

    def __init__(self, tags):
        self._tags = set(tags)
        self._queue = queue.Queue()

    def make_callback(self, tag):
        def callback(data):
            self._queue.put((tag, data.frame, parse_sensor_data(data)))
        return callback

    def get_data(self, frame, timeout=30.0):
        collected = {}
        while len(collected) < len(self._tags):
            tag, data_frame, value = self._queue.get(timeout=timeout)
            if data_frame != frame:
                continue
            collected[tag] = value
        return collected


def spawn_sensors(world, ego, specs, sync):
    library = world.get_blueprint_library()
    sensors = []
    for spec in specs:
        bp = library.find(spec['type'])
        for key, value in spec['attributes'].items():
            bp.set_attribute(key, str(value))
        transform = carla.Transform(
            carla.Location(x=spec['x'], y=spec['y'], z=spec['z']),
            carla.Rotation(pitch=spec['pitch'], roll=spec['roll'], yaw=spec['yaw']),
        )
        sensor = world.spawn_actor(bp, transform, attach_to=ego)
        sensor.listen(sync.make_callback(spec['id']))
        sensors.append(sensor)
    return sensors


def prepare_output_dirs(out_dir, simple_replay=True):
    views = [v for v in CAMERA_VIEWS if v[0] == 'FRONT'] if simple_replay else CAMERA_VIEWS
    camera_kinds = ['rgb', 'depth'] if simple_replay else ['rgb', 'semantic', 'instance', 'depth']
    for kind in camera_kinds:
        for name, *_ in views:
            (out_dir / 'camera' / f'{kind}_{name.lower()}').mkdir(parents=True, exist_ok=True)
    (out_dir / 'camera' / 'rgb_top_down').mkdir(parents=True, exist_ok=True)


def save_sensor_frame(data, frame_idx, out_dir, simple_replay=True):
    frame_name = f'{frame_idx:05d}'

    cv2.imwrite(str(out_dir / 'camera' / 'rgb_top_down' / f'{frame_name}.jpg'), data['TOP_DOWN'][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 20])
    views = [v for v in CAMERA_VIEWS if v[0] == 'FRONT'] if simple_replay else CAMERA_VIEWS
    for name, *_ in views:
        view = name.lower()
        cv2.imwrite(str(out_dir / 'camera' / f'rgb_{view}' / f'{frame_name}.jpg'), data[f'CAM_{name}'][:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 20])
        cv2.imwrite(str(out_dir / 'camera' / f'depth_{view}' / f'{frame_name}.png'), convert_depth(data[f'CAM_{name}_DEPTH'][:, :, :3]))
        if not simple_replay:
            cv2.imwrite(str(out_dir / 'camera' / f'semantic_{view}' / f'{frame_name}.png'), data[f'CAM_{name}_SEM_SEG'][:, :, 2])
            cv2.imwrite(str(out_dir / 'camera' / f'instance_{view}' / f'{frame_name}.png'), data[f'CAM_{name}_INS_SEG'])


def main():
    args = parse_args()
    town = town_from_scenario_dir(args.scenario_dir)
    frames = load_frames(args.scenario_dir, args.start_frame, args.num_frames)
    record = not args.no_record
    scenario_name = os.path.basename(os.path.normpath(args.scenario_dir))
    out_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent.parent / 'Replayed' / scenario_name
    print(f'loaded {len(frames)} frame(s) for town {town}' + (f', recording to {out_dir}' if record else ''))
    if record:
        prepare_output_dirs(out_dir, args.simple_replay)

    ego_id = next(e['id'] for e in frames[0]['bounding_boxes'] if e['class'] == 'ego_vehicle')

    client = carla.Client(args.host, args.port)
    # Large Maps (Town12/13/15) stream a lot more on load_world and can take well over 60s,
    # especially with many CARLA servers loading maps concurrently and contending for disk/GPU I/O.
    client.set_timeout(300.0)
    world = client.load_world(town)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FRAME_RATE
    world.apply_settings(settings)

    weather = getattr(carla.WeatherParameters, args.weather)
    weather.sun_altitude_angle = 90.0  # sun directly overhead, regardless of preset (presets top out at 45)
    world.set_weather(weather)

    actors = {}
    sensors = []
    try:
        apply_frame(world, frames[0], actors)  # bootstrap so the ego (tagged 'hero') exists first

        # On Large Maps (Town11-13/15) the tile around the hero actor streams in
        # asynchronously; spawning 24 camera sensors before it's loaded has been observed to
        # segfault the server. Move the spectator there too (belt-and-suspenders alongside the
        # 'hero' role_name) and tick a few times to let streaming catch up before going further.
        world.get_spectator().set_transform(actors[ego_id].get_transform())
        for _ in range(TILE_STREAM_TICKS):
            world.tick()

        sync = None
        if record:
            specs = build_sensor_specs(args.simple_replay)
            sync = SensorSync(spec['id'] for spec in specs)
            sensors = spawn_sensors(world, actors[ego_id], specs, sync)
            print(f'spawned {len(sensors)} camera sensors, warming up...')
            for _ in range(SENSOR_WARMUP_TICKS):
                world.tick()

        for frame_idx, frame in enumerate(frames):
            apply_frame(world, frame, actors)
            frame_number = world.tick()
            print(f'frame {frame_idx + 1}/{len(frames)} applied ({len(actors)} actors)')
            if record:
                data = sync.get_data(frame_number)
                saved_idx = args.start_frame + frame_idx
                save_sensor_frame(data, saved_idx, out_dir, args.simple_replay)
                print(f'  saved cameras for frame {saved_idx}')
    finally:
        for sensor in sensors:
            sensor.stop()
            sensor.destroy()
        for actor in actors.values():
            actor.destroy()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)


if __name__ == '__main__':
    main()
