# Plan: Recreate Bench2Drive-mini scenarios in CARLA 0.9.16 with new weather/time of day

## Goal

Use the per-frame ground-truth annotation data in `Bench2Drive-mini/*/anno/*.json.gz` to
recreate a scenario in CARLA, keeping the ego trajectory, NPC vehicles/pedestrians, and
route exactly as recorded — but with weather and time of day changed. Camera-only: unlike
upstream Bench2Drive's replay tool, this DeepSight version does not attach or record
lidar/radar, only the RGB/depth/semantic/instance camera rig.

This is DeepSight's adaptation of `Bench2Drive/tools/replay_scenario.py` and
`Bench2Drive/replay_plan.md` (the standalone upstream repo, CARLA 0.9.15) for this repo's
`bench2drive/` submodule-like copy, which targets **CARLA 0.9.16** per `README.md` §4 /
`SETUP.md`.

## What's actually in Bench2Drive-mini here

`bench2drive/Bench2Drive-mini/` already contains both the original `.tar.gz` archives and
extracted scenario folders side by side (e.g. `Accident_Town03_Route156_Weather0/anno/00000.json.gz`,
`00001.json.gz`, ...) — confirmed identical in format to upstream. Each frame, at 10Hz
(`frame_rate = 10.0`, matching `tools/data_collect.py`), contains full **world-coordinate
ground truth** for every actor via `bounding_boxes[]` entries:

- `class` (`ego_vehicle`, `vehicle`, `walker`, `traffic_light`, `traffic_sign`)
- `id` — unique actor id
- `type_id` — CARLA blueprint id (e.g. `vehicle.lincoln.mkz_2020`)
- `color`
- `location`, `rotation` — world coordinates
- `state` — `dynamic`/`static` for vehicles, `0`/`1`/`2` (red/yellow/green) for traffic lights
  (verified against a sample frame in this repo's copy: ego lacks `state`, `vehicle` entries
  carry `state`/`world2vehicle`, `traffic_light` entries carry integer `state`)
- `center`, `extent`, `world2vehicle` (or `world2ego`/`world2sign`), `world_cord` — bounding
  box + transform matrix

`tools/utils.py` and `tools/data_collect.py` in this repo are byte-identical to upstream
Bench2Drive's copies (`diff` confirms), so the anno format, the `get_matrix`/`convert_depth`
helpers, and the sensor-spec conventions this plan reuses are all unchanged.

### No original route/scenario XML is available locally

Same situation as upstream: `leaderboard/data/bench2drive220.xml` uses a different route-id
scheme than the mini scenario folder names (e.g. `Route156` doesn't appear in it), so
re-running through `scenario_runner`'s route/scenario system isn't possible with what's
present. The anno data itself — a kinematic replay of recorded poses — is the only faithful
source, which matches the ask anyway.

## Key difference from upstream: CARLA 0.9.16, not 0.9.15

Upstream Bench2Drive/DeepSight's own `bench2drive/README.md` (mirrored 1:1 from upstream)
still says CARLA 0.9.15, but this DeepSight repo's actual closed-loop-eval instructions
(`README.md` §4, `SETUP.md` line 330) target **CARLA 0.9.16**:

- Install: `CARLA_0.9.16.tar.gz` + `AdditionalMaps_0.9.16.tar.gz` (via `ImportAssets.sh`),
  Python 3.10 wheel `carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`.
  - Download and unpack `CARLA_0.9.16.tar.gz` in its own `Carla` dir
  - Download `AdditionalMaps_0.9.16.tar.gz` into `Carla/Import` then run `ImportAssets.sh`
  - Pip install `Carla/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl`
    in the same venv where you want to run the replay_scenario script, venv only needs opence-python
- Confirmed present on this machine: CARLA 0.9.16 is already extracted at `/home/hhguo/Carla`
  (capital C), with `AdditionalMaps_0.9.16` already imported — `Town11`/`Town12`/`Town13`/
  `Town15` map assets exist under `/home/hhguo/Carla/CarlaUE4/Content/Carla/Maps/` and
  `.../Config/`, and cp310/cp311/cp312 wheels exist under
  `/home/hhguo/Carla/PythonAPI/carla/dist/`.
<!-- - **Stale env var warning**: `$CARLA_ROOT` is currently set to `/home/hhguo/carla`
  (lowercase `c`), which does not exist. The real install is `/home/hhguo/Carla`
  (capital `C`). Re-export `CARLA_ROOT` before running anything, or the replay script's
  `import carla` will fail to resolve. -->
- No dedicated Python 3.10 `b2d` venv with the 0.9.16 wheel installed exists yet on this
  machine (`SETUP.md` line 330 explicitly says this isn't set up). You need to create one
  before running the replay script — see Start sequence below. 
  <!-- (There is an old `.venv` in
  the sibling standalone `/home/hhguo/Bench2Drive` checkout with a `carla.pth` pointing at
  the *0.9.15* egg — do not reuse that one; it's a different CARLA build with different map
  geometry/blueprint ids.) -->
- Relevant CARLA 0.9.16 changelog entries (from `/home/hhguo/Carla/CHANGELOG`, 0.9.15→0.9.16)
  checked for anything that could break this replay approach — nothing does, but worth
  knowing about: "Synchronized actor BoundingBox between server and client", "Add actor_id to
  bounding boxes", and "Fixed `frame`/`timestamp`/`transform` of `SensorData` not matching to
  the actually sent image for camera sensors" (this last one only tightens the frame-number
  matching the `SensorSync` class below already relies on — a good sign, not a risk).
- Blueprint/map-geometry compatibility is still a concern in principle (recorded
  `location`/`rotation`/`road_id`/`lane_id` are baked into whatever CARLA build originally
  produced the Bench2Drive-mini dataset, not necessarily 0.9.16), but empirically the same
  `type_id`s (e.g. `vehicle.lincoln.mkz_2020`, `vehicle.nissan.patrol_2021`) and map names are
  still expected to resolve under 0.9.16, since the upstream Bench2Drive dataset generation and
  eval pipeline itself has already moved to using newer CARLA releases with `AdditionalMaps`.
  `get_blueprint()` degrades gracefully (skips + logs) if a `type_id` is missing in 0.9.16, so
  a bad match fails loud rather than silently misplacing an actor.

## Plan: kinematic log-replay with weather override (camera-only)

1. **CARLA server**
   - CARLA **0.9.16**, with `AdditionalMaps_0.9.16` imported (already done on this machine at
     `/home/hhguo/Carla`).
   - Load the correct town for the chosen scenario (parsed from the folder name, e.g. `Town03`)
     via `client.load_world(town)`.

2. **New script**: `tools/replay_scenario.py`, reusing `convert_depth` from `tools/utils.py`:
   - Set synchronous mode with `fixed_delta_seconds = 0.1` to match the 10Hz save rate.
   - Set weather **once** via `carla.WeatherParameters(...)` preset by name (e.g. `ClearNoon`,
     `ClearNight`, `WetNoon` — `sun_altitude_angle` controls time of day, negative = night).
   - Iterate `anno/*.json.gz` in numeric order; maintain an `id -> carla.Actor` registry:
     - **First appearance of an id**: spawn the matching blueprint (`type_id`) with the
       recorded `color`, then call `actor.set_simulate_physics(False)` so it won't drift off
       the commanded pose between ticks.
     - **Every frame**: `actor.set_transform(carla.Transform(location, rotation))` using the
       recorded values.
     - **Static vehicles**: `location`/`rotation` are wrong when `state == "static"` (a CARLA
       API bug baked into how the anno data was originally recorded, independent of replay
       CARLA version). Recover the true pose from `center`/`extent` plus the inverse of
       `world2vehicle` instead.
     - **Traffic lights**: match recorded entries to the loaded map's traffic-light actors by
       nearest position (not by `id` — ids are not guaranteed stable across world reloads), then
       call `set_state()` to match the recorded state.
     - **Cleanup**: destroy any actor id no longer present in the current frame's id set.
   - Call `world.tick()` after applying each frame's state.

3. **Ego vehicle**: apply the same teleport-replay treatment as NPCs (not physics-driven
   control), so the trajectory is pixel-identical to the recording regardless of
   weather/friction changes.

4. **Sensor recording (camera-only, matches `tools/data_collect.py`'s camera path)**
   - Attaches the same camera rig `tools/data_collect.py`'s `Env_Manager` uses: 6 RGB + 6
     depth + 6 semantic + 6 instance cameras (1600×900) at the nuScenes-like mount points,
     plus the debug top-down RGB camera. Specs/attributes come from
     `build_sensor_specs()`/`_preprocess_sensor_spec` conventions in `data_collect.py`.
   - **Deliberately not attached**: `LIDAR_TOP`, all 5 radars, `LIDAR_TOP_SEG`, and the
     `IMU`/`GNSS`/`SPEED` pseudosensors. This is a narrower scope than upstream's
     `replay_scenario.py` (which recorded lidar+radar too) — this DeepSight version only
     needs camera output. Dropping lidar/radar also removes the `laspy`/`h5py` dependency
     upstream's version needed; only `opencv-python` is required here.
   - Save formats: RGB/top-down as JPEG quality 20, semantic segmentation as the raw image's
     red channel, instance segmentation as the full BGRA buffer, depth via
     `tools/utils.py:convert_depth` (imported directly, not reimplemented).
   - Sensor↔frame synchronization uses a small standalone `SensorSync` class (same
     queue-and-match-by-frame pattern as `leaderboard`'s `SensorInterface`/`CallBack`, copied
     inline so the script still only depends on the bare `carla` module).
   - Ego is bootstrapped one frame early so cameras have an actor to attach to, then given a
     10-tick warm-up before frame-by-frame capture begins.
   - Not regenerated: `anno/*.json.gz`. The original ground truth anno is still valid input.
   - Output layout: `Bench2Drive/Replayed/{scenario_name}/camera/{rgb,semantic,instance,depth}_{view}/*.{jpg,png}`
     plus `camera/rgb_top_down/*.jpg` — same as upstream, minus the `lidar/`/`radar/` dirs.

5. **Validate**: run one scenario folder with `--num_frames 10`, confirm the saved camera
   frames match the original recording's geometry with only lighting/weather differing.

## Start sequence

1. **Fix `CARLA_ROOT`** (the currently-exported value is stale):
   ```bash
   export CARLA_ROOT=/home/hhguo/Carla
   ```

2. **Create the Python 3.10 `b2d` venv with the 0.9.16 wheel** (per `README.md` §4/§Step 2,
   not yet done on this machine):
   ```bash
   conda create -n b2d python=3.10 -y && conda activate b2d
   pip install $CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
   pip install opencv-python  # only extra dep the camera-only replay script needs
   ```

3. **Terminal 1 — start the CARLA server** and leave it running (startup is blocking, per
   `bench2drive/tools/check_carla.md`):
   ```bash
   cd $CARLA_ROOT
   ./CarlaUE4.sh -RenderOffScreen -nosound -fps=10 -carla-rpc-port=2000
   ```
   - Drop `-RenderOffScreen` to watch it live instead of only via the spectator/client.
   - `-fps=10` matches the anno `frame_rate` but isn't load-bearing — the replay script drives
     timing itself via synchronous mode + `fixed_delta_seconds=0.1`, so the server ticks only
     when the client tells it to.

4. **Wait for the server to be ready** before launching the client (same check as
   `bench2drive/tools/check_carla.md`):
   ```bash
   python -c "import carla; c = carla.Client('localhost', 2000); c.set_timeout(60.0); print(c.get_server_version())"
   ```
   Retry until this succeeds — should print `0.9.16`.

5. **Terminal 2 — run the replay script** against the already-running server:
   ```bash
   cd bench2drive/tools
   python replay_scenario.py \
     --scenario_dir ../Bench2Drive-mini/Accident_Town03_Route156_Weather0 \
     --num_frames 10 --weather ClearNoon
   ```
   - No `PYTHONPATH`/`SCENARIO_RUNNER_ROOT`/`LEADERBOARD_ROOT` exports needed here (unlike
     `bench2drive/leaderboard/scripts/run_evaluation.sh`) — the replay script only talks to
     the bare CARLA client API, not `scenario_runner`/`leaderboard`.
   - The script internally calls `client.load_world(town)` for the town parsed from
     `--scenario_dir`'s name, so it will reload the map even if the server booted into a
     different default one.
   - Pass `--no_record` to skip camera capture and only replay actor poses.
   - Output (unless `--no_record`) lands in `bench2drive/Replayed/{scenario_name}/`.

6. **[OPTIONAL] Between runs / when done**, kill the server before starting a fresh one:
   ```bash
   bash bench2drive/tools/clean_carla.sh
   ```

## Status

`bench2drive/tools/replay_scenario.py` is implemented per the above: bare CARLA client API,
no `scenario_runner`/`leaderboard` involved, camera-only recording (no lidar/radar), targeting
CARLA 0.9.16. Adapted from upstream `Bench2Drive/tools/replay_scenario.py` by removing the
`LIDAR_TOP`/`RADAR_VIEWS` sensor specs, the `laspy`/`h5py` saving code paths, and the
`lidar_to_ego_coordinate` helper; the pose-replay logic (`apply_frame`, `resolve_transform`,
traffic light matching) is otherwise unchanged since it doesn't depend on the CARLA server
version or on which sensors are attached.

Example run (first 10 frames, clear/noon):
```bash
cd bench2drive/tools
python replay_scenario.py \
  --scenario_dir ../Bench2Drive-mini/ControlLoss_Town11_Route401_Weather11 \
  --num_frames 10 --weather ClearNoon
```

Performance note: this spawns 24 camera sensors at 1600×900 per replay — matches the camera
portion of the original benchmark's cost, so expect it to be GPU/render-heavy, same as running
`bench2drive/tools/data_collect.py` itself (minus the lidar/radar overhead).
