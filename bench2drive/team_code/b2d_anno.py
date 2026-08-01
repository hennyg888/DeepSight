"""Bench2Drive-format `anno/` and `lidar/` writers for eval-time agents.

The DeepSight training-sample builders (DeepSight/src/tools/create_date_set.py and
crop_bev_for_bench2drive.py) read `<scene>/anno/%05d.json[.gz]` in the schema that
tools/data_collect.py produces during Bench2Drive data collection.  The eval agents
(team_code/qwen_b2d_agent*.py) create the `anno/` directory but never write into it,
so their runs cannot be turned into training samples.  This module fills that gap.

It is a trimmed re-implementation of Env_Manager.get_sensors_anno /
get_bounding_boxes / save from tools/data_collect.py -- that file cannot simply be
imported because it pulls in h5py and easydict at module scope, neither of which is
installed in the eval venv.

Fidelity vs. data_collect.py:
  * exact          -- the scalar fields, `weather`, `sensors` (incl. TOP_DOWN
                      intrinsic/world2cam/cam2ego) and the ego box at index 0.
                      These are the only things create_date_set.py and
                      crop_bev_for_bench2drive.py actually read.
  * close          -- `vehicle` and `walker` boxes carry the same field names, but
                      `num_points` is always -1 (no LiDAR hit counting) and
                      `state` is always 'dynamic'.
  * not emitted    -- static (level-bbs matched) vehicles, traffic lights and
                      traffic signs.  Add them here if a downstream labeller
                      needs them.
  * null           -- `command_far`/`x_command_far`/`y_command_far`,
                      `should_brake`, `only_ap_brake`.  An eval agent runs a single
                      near-range RoutePlanner and has no autopilot brake flags;
                      these are left null rather than filled with a plausible-looking
                      wrong value.

`write_lidar` covers the other half of the layout: the raw `lidar/%05d.laz` point
clouds, in the same LAS format 0 / 1 mm grid encoding data_collect.py uses, alongside
the rendered `lidar_bev/%05d.png` the agents already produce.

Note the eval agent's top-down camera is registered as `CAM_BEV`; it is written into
the anno under the key `TOP_DOWN` so downstream code that does
`anno['sensors']['TOP_DOWN']` works unchanged.  The two sensor specs are identical
(x=y=0, z=50, pitch=-90, 1600x900, fov=110 -> fx=fy=560.166).
"""

import gzip
import json
import os
import sys

import carla
import laspy
import numpy as np

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# tools/utils.py holds the projection/transform helpers data_collect.py builds anno
# with; reuse them so our matrices are bit-identical to the training data's.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from utils import build_projection_matrix, get_matrix, compute_2d_distance

# Actors further than this from the ego (2D, metres) are left out of the boxes list.
# Matches DIS_CAR_SAVE / DIS_WALKER_SAVE in tools/utils.py.
DIS_ACTOR_SAVE = 100

# Eval-agent sensor id -> key used in the Bench2Drive anno schema.
SENSOR_ID_TO_ANNO_KEY = {'CAM_BEV': 'TOP_DOWN'}

# LIDAR_TOP mounting point, from the sensor spec both agents declare.  Raw CARLA
# LiDAR points come back in the sensor frame; the training .laz files store them
# in the ego frame, which for this rig is a pure translation (yaw/roll/pitch = 0).
LIDAR_POS = np.array([-0.39, 0.0, 1.84])

# laspy quantises coordinates onto an integer grid of this size (metres), same as
# tools/data_collect.py.
LIDAR_POINT_PRECISION = 0.001


def _cube_vertices(bbx_loc, extent):
    """Eight local-frame corners of a bounding box, ordered as in tools/utils.py."""
    x, y, z = bbx_loc.x, bbx_loc.y, bbx_loc.z
    dx, dy, dz = extent.x, extent.y, extent.z
    return [
        [x - dx, y - dy, z - dz], [x + dx, y - dy, z - dz],
        [x - dx, y + dy, z - dz], [x + dx, y + dy, z - dz],
        [x - dx, y - dy, z + dz], [x + dx, y - dy, z + dz],
        [x - dx, y + dy, z + dz], [x + dx, y + dy, z + dz],
    ]


def _world_cord(actor):
    transform = actor.get_transform()
    verts = []
    for lx, ly, lz in _cube_vertices(actor.bounding_box.location, actor.bounding_box.extent):
        g = transform.transform(carla.Location(lx, ly, lz))
        verts.append([g.x, g.y, g.z])
    return verts


def _forward_speed(actor):
    transform = actor.get_transform()
    velocity = actor.get_velocity()
    vel = np.array([velocity.x, velocity.y, velocity.z])
    pitch = np.deg2rad(transform.rotation.pitch)
    yaw = np.deg2rad(transform.rotation.yaw)
    orientation = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    return float(np.dot(vel, orientation))


def _lane_ids(location):
    """road/lane/section id of the waypoint under `location`, or -1 when off-map."""
    try:
        wp = CarlaDataProvider.get_map().get_waypoint(location)
        return wp.road_id, wp.lane_id, wp.section_id
    except Exception:
        return -1, -1, -1


def get_sensors_anno(sensor_specs):
    """Per-sensor extrinsics/intrinsics, keyed as in the Bench2Drive anno schema.

    `sensor_specs` is the list returned by the agent's own sensors() method; it
    supplies the rig-relative x/y/z/roll/pitch/yaw needed for cam2ego (the actor
    transform alone is world-frame).
    """
    spec_by_id = {s['id']: s for s in sensor_specs}
    world = CarlaDataProvider.get_world()

    actors = {}
    for pattern in ('*sensor.camera.rgb*', '*sensor.lidar.ray_cast*', '*sensor.other.radar*'):
        for actor in world.get_actors().filter(pattern):
            role = actor.attributes.get('role_name')
            if role in spec_by_id:
                actors[role] = actor

    results = {}
    for sensor_id, actor in actors.items():
        spec = spec_by_id[sensor_id]
        transform = actor.get_transform()
        location, rotation = transform.location, transform.rotation
        world2sensor = transform.get_inverse_matrix()
        sensor2ego = get_matrix(
            location=[spec['x'], spec['y'], spec['z']],
            rotation=[spec['pitch'], spec['roll'], spec['yaw']],
        ).tolist()

        result = {
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
        }
        if 'camera' in actor.type_id:
            width = int(actor.attributes['image_size_x'])
            height = int(actor.attributes['image_size_y'])
            fov = float(actor.attributes['fov'])
            result.update({
                'intrinsic': build_projection_matrix(width, height, fov=fov).tolist(),
                'world2cam': world2sensor,
                'cam2ego': sensor2ego,
                'fov': fov,
                'image_size_x': width,
                'image_size_y': height,
            })
        elif 'lidar' in actor.type_id:
            result.update({'world2lidar': world2sensor, 'lidar2ego': sensor2ego})
        else:
            result.update({'world2radar': world2sensor, 'radar2ego': sensor2ego})

        results[SENSOR_ID_TO_ANNO_KEY.get(sensor_id, sensor_id)] = result

    return results


def get_bounding_boxes(ego):
    """Ego box first (create_date_set.py indexes [0]), then nearby vehicles/walkers."""
    world = CarlaDataProvider.get_world()

    def _light_state(actor):
        try:
            return str(actor.get_light_state()).split('.')[-1]
        except Exception:
            return 'None'

    def actor_entry(actor, cls, ego_location=None):
        transform = actor.get_transform()
        location, rotation = transform.location, transform.rotation
        extent = actor.bounding_box.extent
        center = transform.transform(actor.bounding_box.location)
        road_id, lane_id, section_id = _lane_ids(location)
        try:
            brake = actor.get_control().brake
        except Exception:
            brake = 0.0
        entry = {
            'class': cls,
            'id': str(actor.id),
            'type_id': actor.type_id,
            'base_type': actor.attributes.get('base_type', 'None'),
            'location': [location.x, location.y, location.z],
            'rotation': [rotation.pitch, rotation.roll, rotation.yaw],
            'bbx_loc': [actor.bounding_box.location.x,
                        actor.bounding_box.location.y,
                        actor.bounding_box.location.z],
            'center': [center.x, center.y, center.z],
            'extent': [extent.x, extent.y, extent.z],
            'world_cord': _world_cord(actor),
            'semantic_tags': list(actor.semantic_tags),
            'color': actor.attributes.get('color', 'None'),
            'speed': _forward_speed(actor),
            'brake': brake,
            'road_id': road_id,
            'lane_id': lane_id,
            'section_id': section_id,
        }
        if cls == 'ego_vehicle':
            entry['world2ego'] = transform.get_inverse_matrix()
        else:
            entry['state'] = 'dynamic'
            entry['num_points'] = -1  # no LiDAR hit counting at eval time
            entry['light_state'] = _light_state(actor)
            entry['distance'] = compute_2d_distance(location, ego_location)
            entry['world2vehicle'] = transform.get_inverse_matrix()
        return entry

    results = [actor_entry(ego, 'ego_vehicle')]
    ego_location = ego.get_transform().location

    for pattern, cls in (('*vehicle*', 'vehicle'), ('*walker.pedestrian*', 'walker')):
        for actor in world.get_actors().filter(pattern):
            if not actor.is_alive or actor.id == ego.id:
                continue
            if compute_2d_distance(actor.get_transform().location, ego_location) > DIS_ACTOR_SAVE:
                continue
            results.append(actor_entry(actor, cls, ego_location))

    return results


def get_weather():
    w = CarlaDataProvider.get_world().get_weather()
    return {
        'cloudiness': w.cloudiness,
        'precipitation': w.precipitation,
        'precipitation_deposits': w.precipitation_deposits,
        'wind_intensity': w.wind_intensity,
        'sun_azimuth_angle': w.sun_azimuth_angle,
        'sun_altitude_angle': w.sun_altitude_angle,
        'fog_density': w.fog_density,
        'fog_distance': w.fog_distance,
        'wetness': w.wetness,
        'fog_falloff': w.fog_falloff,
    }


def write_lidar(save_path, frame, lidar_raw):
    """Write `<save_path>/lidar/<frame:05>.laz` the way tools/data_collect.py does.

    `lidar_raw` is the (N, 4) [x, y, z, intensity] array CARLA hands the agent in
    `input_data['LIDAR_TOP'][1]`.  Only xyz is stored, translated into the ego frame
    -- LAS point format 0 has no intensity field, which is why the BEV renderer
    (make_lidar_bev) zeroes intensity to stay consistent with the training data.
    """
    points = np.asarray(lidar_raw)[:, :3] + LIDAR_POS

    header = laspy.LasHeader(point_format=0)
    # An empty sweep would make np.min raise; keep the frame numbering contiguous by
    # writing a valid but empty file instead.
    header.offsets = np.min(points, axis=0) if len(points) else np.zeros(3)
    header.scales = np.full(3, LIDAR_POINT_PRECISION)

    out_file = os.path.join(str(save_path), 'lidar', f'{frame:05}.laz')
    with laspy.open(out_file, mode='w', header=header) as writer:
        record = laspy.ScaleAwarePointRecord.zeros(len(points), header=header)
        record.x = points[:, 0]
        record.y = points[:, 1]
        record.z = points[:, 2]
        writer.write_points(record)
    return out_file


def write_anno(save_path, frame, ego, tick_data, control, sensor_specs):
    """Write `<save_path>/anno/<frame:05>.json.gz` in the Bench2Drive anno schema.

    Frame numbering matches the agent's camera files (`camera/CAM_*/<frame:05>.jpg`),
    so a downstream builder can index annos and images with the same integer.
    """
    near_command = tick_data['command_near']
    near_command = getattr(near_command, 'value', near_command)
    near_xy = tick_data['command_near_xy']

    anno_data = {
        'x': float(tick_data['pos'][0]),
        'y': float(tick_data['pos'][1]),
        'throttle': control.throttle,
        'steer': control.steer,
        'brake': control.brake,
        'reverse': control.reverse,
        'theta': float(tick_data['compass']),
        'speed': float(tick_data['speed']),
        # An eval agent runs one near-range RoutePlanner; there is no far node.
        'x_command_far': None,
        'y_command_far': None,
        'command_far': None,
        'x_command_near': float(near_xy[0]),
        'y_command_near': float(near_xy[1]),
        'command_near': near_command,
        'should_brake': None,
        'x_target': float(near_xy[0]),
        'y_target': float(near_xy[1]),
        'next_command': near_command,
        'weather': get_weather(),
        'acceleration': np.asarray(tick_data['acceleration']).tolist(),
        'angular_velocity': np.asarray(tick_data['angular_velocity']).tolist(),
        'bounding_boxes': get_bounding_boxes(ego),
        'sensors': get_sensors_anno(sensor_specs),
        'only_ap_brake': None,
    }

    out_file = os.path.join(str(save_path), 'anno', f'{frame:05}.json.gz')
    with gzip.open(out_file, 'wt', encoding='utf-8') as f:
        json.dump(anno_data, f, indent=4)
    return out_file
