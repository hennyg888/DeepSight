"""PDM-lite expert driving a bench2drive route while recording Bench2Drive-format data.

Wraps fail2drive's privileged AutoPilot (`/home/hhguo/fail2drive/team_code/autopilot.py`,
the carla_garage PDM-lite expert) in the sensor rig and on-disk layout that
`QwenAgent` produces, so an expert run is directly usable as DeepSight training data:

    <SAVE_PATH>/Scenarios/<route>_<scenario>_<timestamp>/
        camera/CAM_{FRONT,FRONT_LEFT,FRONT_RIGHT,BACK,BACK_LEFT,BACK_RIGHT,BEV}/%05d.jpg
        lidar_bev/%05d.png
        anno/%05d.json.gz          (only with SAVE_ANNO=1)
        lidar/%05d.laz             (only with SAVE_ANNO=1)
        metric_info.json

The point of running the expert rather than the model is the trajectory labels: the
future waypoints in a training sample are the ego's own driven path, so a model rollout
teaches the model to imitate itself. PDM-lite drives from privileged simulator state.

Run it with the `expert` profile:

    SAVE_ANNO=1 bash leaderboard/scripts/run_custom_route.sh \
        leaderboard/data/route121_car_glare.xml expert

Scope note -- obstacle avoidance is OFF. PDM-lite's overtaking/route-shifting logic
(`_manage_route_obstacle_scenarios`) is driven entirely by
`CarlaDataProvider.active_scenarios`, which fail2drive's scenario classes populate and
this fork's scenario_runner has no notion of. The list is therefore always empty here
and the expert falls back to plain IDM behaviour: it drives the route and comes to a
stop behind whatever is blocking its lane, which is the intended behaviour for the
obstacle scenarios this is collected on. If you ever want the expert to overtake, that
requires adding `active_scenarios` registration to this fork's scenario classes -- see
fail2drive's `scenario_runner/srunner/scenarios/route_obstacles.py:405` for the shape of
the tuple each scenario appends.
"""

import datetime
import json
import os
import pathlib
import sys

import carla
import cv2
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from leaderboard.utils.route_manipulation import downsample_route
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

# PDM-lite and its support modules (config, nav_planner, privileged_route_planner, ...)
# import each other by bare module name, so their directory has to be importable.
# Appended rather than prepended so it cannot shadow this repo's own modules.
FAIL2DRIVE_ROOT = os.environ.get('FAIL2DRIVE_ROOT', '/home/hhguo/fail2drive')
FAIL2DRIVE_TEAM_CODE = os.path.join(FAIL2DRIVE_ROOT, 'team_code')
if FAIL2DRIVE_TEAM_CODE not in sys.path:
    sys.path.append(FAIL2DRIVE_TEAM_CODE)

from autopilot import AutoPilot  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'))
from lidar_to_bev import bev_projection  # noqa: E402
from team_code.b2d_anno import write_anno, write_lidar  # noqa: E402

SAVE_PATH = os.environ.get('SAVE_PATH', None)
SAVE_ANNO = os.environ.get('SAVE_ANNO', '0') not in ('0', '', 'false', 'False')

CAM_KEYS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BEV']


def get_entry_point():
    return 'PdmLiteB2DAgent'


class PdmLiteB2DAgent(AutoPilot):
    """AutoPilot that drives on privileged state and records the Bench2Drive sensor set."""

    def setup(self, path_to_conf_file, keypoints=None, manager=None):
        """Signature matches this fork's evaluator: setup(agent_config, xyz_list, manager).

        AutoPilot's own setup takes (path_to_conf_file, route_index, traffic_manager);
        `keypoints` and `manager` are what leaderboard_evaluator.py actually passes, and
        neither maps onto those, so they are handled here instead of forwarded.
        """
        # AutoPilot.setup builds its own carla_garage-style output directory when
        # SAVE_PATH is set, keyed off os.environ["TOWN"] / ["REPETITION"] -- neither of
        # which this fork's run_evaluation.sh defines, so it would KeyError. Hide the
        # variable from it; leaving its self.save_path as None also disables its
        # ScenarioLogger, measurements dump and destroy()-time writes, none of which we
        # want. Our own recording is entirely separate, under self.record_path.
        save_path_env = os.environ.pop('SAVE_PATH', None)
        try:
            super().setup(path_to_conf_file, route_index=0, traffic_manager=None)
        finally:
            if save_path_env is not None:
                os.environ['SAVE_PATH'] = save_path_env

        self.manager = manager

        # PDM-lite's _manage_route_obstacle_scenarios reads CarlaDataProvider
        # .active_scenarios every frame -- fail2drive's scenario classes append
        # themselves to it, and this fork's CarlaDataProvider does not define the
        # attribute at all (AttributeError on the first step without this). Seeding it
        # empty, per route, is what keeps the expert on plain IDM: it drives the route
        # and stops behind an obstacle instead of shifting its route around it. See this
        # module's docstring if you ever want the overtaking behaviour back.
        CarlaDataProvider.active_scenarios = []
        # AutoPilot declares Track.MAP because carla_garage requests an opendrive_map
        # sensor. This agent's sensors() drops it (run_step reads the map straight off
        # the client instead), so we can stay on the SENSORS track that
        # run_evaluation.sh selects.
        self.track = autonomous_agent.Track.SENSORS

        self.metric_info = {}
        self.record_path = None
        if SAVE_PATH is not None:
            # Same directory naming as QwenAgent, so both agents' runs are interchangeable
            # inputs to the sample builders. The evaluator appends '+<save_name>' to the
            # agent config for exactly this purpose.
            save_name = path_to_conf_file.split('+')[-1]
            string = pathlib.Path(os.environ['ROUTES']).stem + '_' + save_name
            self.record_path = pathlib.Path(SAVE_PATH) / 'Scenarios' / string
            self.record_path.mkdir(parents=True, exist_ok=False)
            (self.record_path / 'anno').mkdir()
            (self.record_path / 'lidar').mkdir()
            (self.record_path / 'lidar_bev').mkdir()
            for cam_key in CAM_KEYS:
                (self.record_path / 'camera' / cam_key).mkdir(parents=True, exist_ok=True)

    def _init(self, hd_map):
        """Route/controller setup, with SCENARIO_RUNNER_ROOT pointed at fail2drive.

        PrivilegedRoutePlanner.compute_speed_limits loads
        `{SCENARIO_RUNNER_ROOT}/speed_limits/<Town>_speed_limits.npy` -- a 75 MB
        precomputed data set that ships with fail2drive's scenario_runner and has no
        counterpart in this fork, so the variable run_evaluation.sh exports points
        somewhere that file does not exist. That single lookup (the only
        SCENARIO_RUNNER_ROOT read in all of PDM-lite) is the reason for the override;
        it is restored immediately so scenario_runner itself is unaffected.
        """
        previous = os.environ.get('SCENARIO_RUNNER_ROOT')
        os.environ['SCENARIO_RUNNER_ROOT'] = os.path.join(FAIL2DRIVE_ROOT, 'scenario_runner')
        try:
            super()._init(hd_map)
        finally:
            if previous is None:
                os.environ.pop('SCENARIO_RUNNER_ROOT', None)
            else:
                os.environ['SCENARIO_RUNNER_ROOT'] = previous

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """Keep the dense route around; PrivilegedRoutePlanner plans on it.

        This fork's base class only stores a 50 m-downsampled plan and discards the dense
        one, but AutoPilot._init reads self.org_dense_route_world_coord. Mirrors
        fail2drive's autonomous_agent_local.set_global_plan, including its 200 m sparse
        downsampling (the base class uses 50 m, which would change how the expert's
        command planner fires).
        """
        self.org_dense_route_gps = global_plan_gps
        self.org_dense_route_world_coord = global_plan_world_coord

        ds_ids = downsample_route(global_plan_world_coord, 200)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1])
                                         for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._plan_gps_HACK = global_plan_gps

    def sensors(self):
        """The Bench2Drive rig, plus the two sensors AutoPilot reads.

        Camera extrinsics/intrinsics are copied verbatim from
        qwen_b2d_agent_with_lidar.py -- the training data depends on them matching.
        AutoPilot's own sensors() is deliberately not called: its opendrive_map would
        force the MAP track, and only one sensor.other.imu / speedometer is permitted, so
        its lowercase 'imu' and 'speed' ids are used for the shared instances.
        """
        return [
            {'type': 'sensor.camera.rgb',
             'x': 0.80, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': 1600, 'height': 900, 'fov': 70, 'id': 'CAM_FRONT'},
            {'type': 'sensor.camera.rgb',
             'x': 0.27, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
             'width': 1600, 'height': 900, 'fov': 70, 'id': 'CAM_FRONT_LEFT'},
            {'type': 'sensor.camera.rgb',
             'x': 0.27, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
             'width': 1600, 'height': 900, 'fov': 70, 'id': 'CAM_FRONT_RIGHT'},
            {'type': 'sensor.camera.rgb',
             'x': -2.0, 'y': 0.0, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
             'width': 1600, 'height': 900, 'fov': 110, 'id': 'CAM_BACK'},
            {'type': 'sensor.camera.rgb',
             'x': -0.32, 'y': -0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
             'width': 1600, 'height': 900, 'fov': 70, 'id': 'CAM_BACK_LEFT'},
            {'type': 'sensor.camera.rgb',
             'x': -0.32, 'y': 0.55, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
             'width': 1600, 'height': 900, 'fov': 70, 'id': 'CAM_BACK_RIGHT'},
            {'type': 'sensor.camera.rgb',
             'x': 0.0, 'y': 0.0, 'z': 50.0, 'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
             'width': 1600, 'height': 900, 'fov': 110, 'id': 'CAM_BEV'},
            # Lowercase ids: AutoPilot.tick_autopilot reads input_data["imu"], and the
            # longitudinal controller reads input_data["speed"].
            {'type': 'sensor.other.imu',
             'x': -1.4, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'sensor_tick': 0.05, 'id': 'imu'},
            {'type': 'sensor.speedometer', 'reading_frequency': 20, 'id': 'speed'},
            {'type': 'sensor.lidar.ray_cast',
             'x': -0.39, 'y': 0.0, 'z': 1.84, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'range': 85, 'rotation_frequency': 10, 'channels': 64,
             'upper_fov': 10, 'lower_fov': -30, 'points_per_second': 600000,
             'atmosphere_attenuation_rate': 0.004,
             'dropoff_general_rate': 0.0, 'dropoff_intensity_limit': 0.0,
             'dropoff_zero_intensity': 0.0,
             'id': 'LIDAR_TOP'},
        ]

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        control = super().run_step(input_data, timestamp, sensors=sensors, plant=plant)

        if self.record_path is not None:
            self.metric_info[self.step] = self.get_metric_info()
            self.record_frame(input_data, control)

        return control

    def record_frame(self, input_data, control):
        """Not named save() -- AutoPilot has its own save(), called from _get_control."""
        frame = self.step

        for cam_key in CAM_KEYS:
            img = input_data[cam_key][1][:, :, :3]
            # Quality 20 matches both tools/data_collect.py and QwenAgent; the training
            # set was built at this compression level.
            cv2.imwrite(str(self.record_path / 'camera' / cam_key / f'{frame:05}.jpg'),
                        img, [cv2.IMWRITE_JPEG_QUALITY, 20])

        Image.fromarray(self.make_lidar_bev(input_data['LIDAR_TOP'][1])).save(
            self.record_path / 'lidar_bev' / f'{frame:05}.png')

        with open(self.record_path / 'metric_info.json', 'w') as f:
            json.dump(self.metric_info, f, indent=4)

        if SAVE_ANNO:
            # Never let a bad frame kill the run -- a hole in anno/ just costs that sample.
            # Separate try blocks so a failure in one does not drop the other.
            try:
                write_anno(self.record_path, frame, self._vehicle,
                           self.anno_tick_data(input_data, control), control, self.sensors())
            except Exception as e:
                print(f'[pdm_lite_b2d] anno frame {frame}: {e}')
            try:
                write_lidar(self.record_path, frame, input_data['LIDAR_TOP'][1])
            except Exception as e:
                print(f'[pdm_lite_b2d] lidar frame {frame}: {e}')

    def anno_tick_data(self, input_data, control):
        """The subset of QwenAgent's tick_data dict that b2d_anno.write_anno consumes."""
        location = self._vehicle.get_transform().location
        imu = input_data['imu'][1]
        compass = imu[-1]
        acceleration, angular_velocity = imu[:3], imu[3:6]
        if np.isnan(compass):  # CARLA emits nan compass for a few frames after spawn
            compass = 0.0
            acceleration, angular_velocity = np.zeros(3), np.zeros(3)

        # Read the navigational command off the expert's own state rather than calling
        # self._command_planner.run_step() again: that method pops consumed waypoints off
        # its route deque, so a second call per frame would advance the plan twice.
        # AutoPilot._get_control has already stashed this frame's values -- self.commands
        # holds far_command.value (a RoadOption, the same 1..6 encoding the anno schema
        # uses) and target_point_prev the world-frame target it came from.
        near_command = self.commands[-1]
        target_point = self.target_point_prev

        return {
            'pos': [location.x, location.y],
            'speed': input_data['speed'][1]['speed'],
            'compass': compass,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'command_near': near_command,
            'command_near_xy': [float(target_point[0]), float(target_point[1])],
        }

    def make_lidar_bev(self, lidar_raw):
        """Identical projection to QwenAgent.make_lidar_bev -- see the comment there.

        Kept in sync deliberately: the BEV pngs recorded here are meant to be
        indistinguishable from the ones the model is trained and evaluated on.
        """
        ego_x = lidar_raw[:, 1]
        ego_y = -(lidar_raw[:, 0] - 0.39)
        z = lidar_raw[:, 2] + 1.84
        intensity = np.zeros(len(lidar_raw), dtype=np.float32)
        pts = np.stack([ego_x, ego_y, z, intensity], axis=1).astype(np.float32)
        return bev_projection(pts)

    def destroy(self, results=None):
        if self.record_path is not None:
            with open(self.record_path / 'metric_info.json', 'w') as f:
                json.dump(self.metric_info, f, indent=4)
        super().destroy(results)
