# Sensor overlays

Post-processing that sits between the simulator's rendered sensor data and whatever
consumes it — the model at inference, or the recorder during expert data collection — so
a route can be perturbed with effects CARLA itself does not render.

The first effect built on it is **glare**: a light source washing out the rgb cameras.
The layer is deliberately general, because more perturbation types and more ways of
anchoring them are coming.

Code lives in `leaderboard/leaderboard/utils/sensor_overlays/`.

---

## Quick start

Add an `<overlays>` block to a route xml, as a sibling of `<scenarios>` and `<weathers>`:

```xml
<route id="24252" town="Town11">
  <waypoints>...</waypoints>
  <scenarios>...</scenarios>

  <overlays>
    <overlay type="glare" anchor="sun" angular_size="45" intensity="1.0" />
  </overlays>

  <weathers>...</weathers>
</route>
```

Then run the route the usual way — no flag, no env var, no agent change:

```bash
bash leaderboard/scripts/run_custom_route.sh leaderboard/data/route149_overlay_glare_car_test.xml
```

It works on every profile (`base`, `lidar30`, `afz`, `expert`), and routes without an
`<overlays>` block are completely unaffected.

Two worked examples ship in `leaderboard/data/`:

| Route file | What it does |
|---|---|
| `route149_sun_glare_car.xml` | CARLA renders a real low sun (swept `<weathers>` azimuth); overlay adds glare on top, `anchor="sun"` with no angles so it follows that sun |
| `route149_overlay_glare_car_test.xml` | Plain ClearNoon sky, glare comes **only** from the overlay, anchored to the parked car and growing as the ego closes in |

Running the pair side by side separates "the model can't see past a bright source" from
everything else low-sun lighting changes about a scene.

---

## The three pieces

Every `<overlay>` element names three independent things. They are separate so a new
effect doesn't have to reinvent anchoring, and a new anchoring mode works with every
existing effect.

| Piece | Question it answers | File | Registry |
|---|---|---|---|
| **overlay type** | *what* gets drawn | `overlays.py` | `@register_overlay('name')` |
| **anchor** | *where it is in the world* | `anchors.py` | `@register_anchor('name')` |
| **stream** | *which images it touches* | `streams.py` | — (built from the agent's sensors) |

The anchor's own attributes sit on the same element as the overlay's, flat:

```xml
<overlay type="glare"  anchor="actor" role_name="scenario no lights"  angular_size="45" />
<!--      ^ overlay     ^ anchor      ^ anchor attribute              ^ overlay attribute -->
```

### Why this is world-anchored, not screen-anchored

The anchor resolves to a **world** bearing or location; the stream projects it through
the ego's **live pose** each frame. So the glare stays put in the world: turn the ego and
it slides across the cameras, crosses the seam into `CAM_FRONT_LEFT`, and eventually
leaves the view — the opposite of a lens-dirt texture painted at a fixed screen position.

---

## Overlay types

### `glare`

A glare halo from a light source, composited onto the rgb cameras after CARLA renders
them. It is a **lens/sensor artefact**, not a light in the scene: it does not illuminate
anything, cast shadows, or affect the LiDAR.

| Attribute | Default | Meaning |
|---|---|---|
| `image` | `assets/glare.png` | Sprite path, absolute or relative to the repo root. Alpha honoured. |
| `angular_size` | `30` | Apparent diameter in **degrees** — see [Sizing](#sizing-and-intensity). |
| `size` | — | Diameter in **metres** instead, which makes the glare **grow as the ego approaches**. Only valid for finite anchors (`actor`, `location`); raises for `sun` / `distant`. |
| `intensity` | `1.0` | Multiplier on the sprite before blending. |
| `blend` | `screen` | `screen`: light adds and never darkens what is already bright. `alpha`: plain compositing, which occludes. |
| `min_altitude` | `0` for distant sources, ungated for anchored ones | Skip while the source is below this elevation, i.e. no glare after it sets. Give it explicitly to gate either kind. |
| `streams` / `cameras` | all rgb cameras except `CAM_BEV` | Comma-separated stream ids to draw into. |
| `enabled` | `true` | `false` keeps the element in the xml without drawing anything. |

Applies to `camera` streams only. `CAM_BEV` (the 50 m top-down camera) is excluded by
default — put it in `streams=` explicitly if you want it.

---

## Anchors

### `sun` — an infinitely distant light, aimed in CARLA's sun convention

`azimuth` is the same quantity as `<weather sun_azimuth_angle=...>`, so a value can be
copied straight out of the weather block. It points *from* the light, so the bearing the
ego must look along to see it is `azimuth - 180`.

| Attribute | Default | Meaning |
|---|---|---|
| `azimuth` | live weather | Degrees, sun convention. |
| `altitude` | live weather | Degrees above the horizon. |
| `follow_weather` | `true` when either angle is omitted | Read the simulator's live sun each frame, so the overlay follows a route's `<weathers>` keyframes by itself. |

```xml
<overlay type="glare" anchor="sun" />                          <!-- tracks the route's sun -->
<overlay type="glare" anchor="sun" azimuth="268.6" altitude="2.0" />   <!-- pinned -->
```

### `location` — a fixed point on the map

Finite, so it parallaxes and grows as the ego approaches.

| Attribute | Default | Meaning |
|---|---|---|
| `x`, `y` | required | World coordinates. |
| `z` | `0` | World height. |

### `actor` — rides a spawned actor

| Attribute | Default | Meaning |
|---|---|---|
| `role_name` | — | Match on `role_name` (scenario actors use e.g. `scenario no lights`). |
| `type_id` | — | Match on a blueprint id prefix, e.g. `vehicle.tesla`. |
| `actor_id` | — | Match on a numeric actor id. |
| `ego` | `false` | `true` anchors to the ego itself. |
| `z_offset` | `0` | Metres above the actor's origin, e.g. headlight height. |

The first of `role_name` / `type_id` / `actor_id` that is given wins, and the match is
cached once found. Before the scenario spawns its actor the overlay simply doesn't draw —
no crash, no warning spam.

### `distant` — infinitely far away, in the direction of another anchor

For a source *beyond* something in the scene: a sun sitting behind the obstacle the ego
is driving towards. Takes the bearing from the wrapped anchor, but keeps the size and
zero-parallax behaviour of a distant source.

| Attribute | Default | Meaning |
|---|---|---|
| `toward` | `actor` | Which anchor supplies the direction; its attributes sit on the same element. |
| `altitude` | the target's own elevation | Degrees above the horizon. |

```xml
<overlay type="glare" anchor="distant" toward="actor" role_name="scenario no lights"
         altitude="2.0" angular_size="45" />
```

Reach for it when a distant source must stay collinear with something in the scene.
`route149_sun_glare_car.xml` needed four `sun_azimuth` keyframes to keep its sun lined up
with a car 100 m ahead across an 11° bend; `distant` recomputes that bearing every frame
instead, exactly rather than by interpolation — while keeping the constant angular size
and zero parallax that `actor` would give up.

---

## Streams

The images an overlay can be composited into. An overlay declares which stream *kinds* it
understands and is only handed those; the rest pass through untouched.

| Stream | Kind | Where it comes from | Geometry |
|---|---|---|---|
| `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`, `CAM_BEV` | `camera` | Built automatically from every `sensor.camera.rgb` in the agent's `sensors()` | Pinhole: mount xyz + yaw/pitch, fov, resolution |
| `LIDAR_BEV` | `bev` | The rendered raster from `tools/lidar_to_bev.bev_projection`, passed in explicitly by the agents | Orthographic, ego-centred, forward-up, ±`BEV_RANGE_M` |

Camera images are BGR (written with cv2), the BEV raster is RGB (written with PIL); each
stream carries its colour order and sprites are swapped to match.

**The LiDAR BEV is passed through the layer even though `glare` ignores it.** That is
deliberate: a future BEV-aware overlay (occluding a sector, dropping a region, marking a
phantom return) needs no agent edits at all.

Raw `lidar/*.laz` point clouds are never routed through the layer — only the *rendered*
BEV image is a stream.

---

## Sizing and intensity

`angular_size` is the apparent diameter in **degrees of field of view**, not pixels. The
layer converts it per stream:

```
box_px = 2 · focal · tan(angular_size / 2)        focal = (width/2) / tan(fov/2)
```

Degrees rather than pixels is what makes it physically consistent: a real light source
subtends a fixed angle, so the same value lands at the same apparent size on the 70° front
camera and the 110° rear one, and does not change if you re-render at another resolution.

On `CAM_FRONT` (1600 px wide, 70° fov, focal ≈ 1142 px):

| `angular_size` | sprite box | visible core | % of frame width |
|---|---|---|---|
| 10° | 200 px | 169 px | 10.6 % |
| 20° | 403 px | 345 px | 21.6 % |
| 30° (default) | 612 px | 524 px | 32.8 % |
| 45° | 946 px | 820 px | 51.2 % |
| 60° | 1319 px | 1143 px | 71.4 % |
| 70° | 1600 px | 1386 px | 86.6 % |

"Visible core" is smaller than the box because `glare.png` fades to transparent before its
edges — roughly 86 % of the box rises above a dark background. An `angular_size` equal to
the camera's fov fills the frame by definition.

### Constant size vs growing with proximity

`angular_size` holds a **constant** apparent size: the source is treated as infinitely far
away, so closing distance changes nothing. That is right for a sun.

`size` (metres) makes the glare **grow as the ego approaches**, because a finite anchor has
a distance and the sprite is projected like a real object:

```
box_px = focal · size_m / distance_m
```

It requires a finite anchor (`actor`, `location`) — combining it with `sun` or `distant`
raises on the first frame. With `size="12"` on `CAM_FRONT`, measured along
`route149_overlay_glare_car_test.xml`:

| Distance to the parked car | Lit width | % of frame |
|---|---|---|
| 100 m | 116 px | 7.2 % |
| 80 m | 146 px | 9.1 % |
| 60 m | 196 px | 12.2 % |
| 40 m | 298 px | 18.6 % |
| 20 m | 608 px | 38.0 % |
| 8 m | 1583 px | 98.9 % |

Pick `size_m` from the range that matters: `size_m = box_px · distance / focal`. To fill
about half the front camera at the 15 m stopping distance, `12 · 1142 / 15 ≈ 914 px`.

**To resize, edit the one attribute:**

```xml
<overlay type="glare" anchor="sun" angular_size="60" intensity="1.0" />
```

Related levers:

- **`intensity`** scales sprite brightness before the screen blend — the right knob when
  the glare is the right *extent* but washes out too much or too little. `>1` saturates a
  larger fraction of the disc, `<1` gives a fainter halo at the same size.
- **`size`** (metres) for finite anchors only; it then shrinks with distance like a real
  object. Combining it with `sun` / `distant` raises on the first frame rather than
  silently drawing nothing.
- Sizes are per-overlay, so for a different apparent size on one camera, add a second
  `<overlay>` with its own `angular_size` and `streams="CAM_FRONT_LEFT"`.

---

## What ends up in the results directory

Overlays are applied **before** anything reads the images, so the model and the recording
see byte-identical pixels. There is no un-overlaid copy of a camera frame kept anywhere.

| Artifact under `Scenarios/<run>/` | Overlay applied? |
|---|---|
| `camera/CAM_FRONT`, `CAM_FRONT_LEFT/RIGHT`, `CAM_BACK*` `/*.jpg` | **yes** |
| `camera/CAM_BEV/*.jpg`, `rgb_bev/*.png` | no — `CAM_BEV` is excluded by default |
| `lidar_bev/*.png` | passes through the layer; unchanged by `glare` (camera-only) |
| `lidar/*.laz` | no — raw point cloud, never a stream |
| `anno/*.json.gz`, `meta/*.json`, `metric_info.json` | no — derived from simulator state |

The expert (`pdm_lite_b2d_agent.py`) overlays everything it records, so a dataset
collected on an overlay route carries exactly what the model faces at inference. The
expert drives from privileged simulator state, so this only ever changes what is
recorded, never how it drives.

To keep a clean reference copy alongside the overlaid one, re-run the route with the
`<overlays>` block removed — nothing currently writes pre-overlay frames.

---

## How it hooks into the run

```
route xml <overlays>
   └─ RouteParser.parse_overlays              leaderboard/utils/route_parser.py
        └─ config.overlays  (list of Overlay)
             └─ set_active_overlays(...)      leaderboard_evaluator.py, per route
                  └─ apply_sensor_overlays(...)   called by each agent, per frame
```

The evaluator publishes the parsed overlays to a module-level slot so agents — which
never see the route config — can pick them up without a new agent API. Each active
overlay is printed at route start:

```
> Sensor overlay: glare(anchor=distant(toward=actor(scenario no lights), altitude=2.0), size=45.0 deg, intensity=1.0, image=glare.png)
```

Agent call sites:

| Agent | Where | Streams passed |
|---|---|---|
| `team_code/qwen_b2d_agent.py` | `tick()`, before anything reads the image dict | cameras |
| `team_code/qwen_b2d_agent_with_lidar.py` | `tick()`; and `make_lidar_bev()` after rendering | cameras; `LIDAR_BEV` |
| `team_code/pdm_lite_b2d_agent.py` | `record_frame()`, before the jpg/png writes | cameras + `LIDAR_BEV` |

The single entry point:

```python
from leaderboard.utils.sensor_overlays import apply_sensor_overlays, BevStream

apply_sensor_overlays(images, ego_vehicle, sensors=self.sensors(),
                      streams={'LIDAR_BEV': BevStream('LIDAR_BEV', BEV_RANGE_M, 644, 644)})
```

`images` maps stream id → `HxWx3` uint8 array and is modified in place (a read-only
sensor buffer is copied first, and the copy re-bound into the dict — so use the returned
dict if you hold a reference to a single array). It is a cheap no-op when the route
declared no overlays, which is why agents call it unconditionally.

**Pass every image the frame produced**, including ones the current effects ignore. That
is what lets a new overlay type reach them without another round of agent edits.

---

## Extending it

### A new effect

Subclass `SpriteOverlay` (which already handles image loading, angular/metric sizing,
screen and alpha blending, edge clipping and sprite caching) or `Overlay` for something
that draws itself:

```python
@register_overlay('mud')
class MudOverlay(SpriteOverlay):
    STREAM_KINDS = ('camera',)
    DEFAULT_EXCLUDED_STREAMS = ('CAM_BEV',)
    DEFAULT_IMAGE = os.path.join('assets', 'mud.png')
```

It is usable as `<overlay type="mud" .../>` immediately. Add `'bev'` to `STREAM_KINDS`
for something that should also mark up the LiDAR raster.

### A new anchoring mode

```python
@register_anchor('lane_end')
class LaneEndAnchor(Anchor):
    def __init__(self, attrib):
        super().__init__(attrib)
        ...
    def resolve(self, ctx):
        return Target(location=...)          # finite
        # or Target(world_yaw=..., altitude=...)   # infinitely distant
```

`ctx` is an `OverlayContext`: `ego_vehicle`, `ego_transform`, `world`, `weather`,
`timestamp`. The world and weather lookups are lazy and shared by every overlay in the
frame. Return `None` to skip this frame — that's how `actor` handles an actor that hasn't
spawned yet.

---

## Gotchas

- **`role_name="scenario no lights"` is not unique.** `ParkedObstacleNoLightsWithGlare`
  gives it to its second vehicle too. With more than one match the anchor takes whichever
  actor the world lists first.
- **`ActorAnchor` caches its match** until the actor stops being alive. An actor destroyed
  and respawned under the same role is re-found; one swapped out silently is not.
- **`size` in metres with `sun` / `distant`** raises immediately — use `angular_size`.
- **`min_altitude` behaves differently per anchor kind.** A source anchored to an actor
  sits at roughly camera height, so its elevation hovers around 0 and goes negative as
  soon as the ego looks slightly down at it — the sun's default gate of 0 would blank it
  out. The gate therefore applies only to infinitely distant sources unless you write
  `min_altitude` explicitly.
- **`follow_weather` and a fixed weather don't mix meaningfully.** If the route pins
  ClearNoon and the overlay follows the weather, the glare sits wherever the noon sun is
  (high, azimuth 0), which is probably not what you meant — pin the angles or use
  `distant`.
- **The overlay is not a light.** It does not brighten the road, change exposure, or cast
  anything. If you need the scene itself relit, that is a `<weathers>` change.
- **`CAM_BEV` is excluded by default** from `glare`, and `rgb_bev/*.png` is derived from
  `CAM_BEV`, so debug BEV renders stay clean unless you opt in.

---

## Verifying a change

The geometry is exercised offline, without a CARLA server, using a stand-in `carla`
module and the agents' real sensor list. The checks that matter:

- a source dead ahead lands at the image centre, at the elevation its altitude implies;
- yawing the ego ±30° slides it symmetrically and hands it to `CAM_FRONT_LEFT` /
  `CAM_FRONT_RIGHT`, continuous across the seam;
- nothing draws behind the camera or below `min_altitude`;
- angular size holds across a change of fov and resolution;
- with `anchor="distant"` toward the parked car, the glare tracks its bearing to ≤1 px at
  four points along the route, at constant size and elevation — and the bearing at the
  start line (88.57° → azimuth 268.57°) matches `route149_sun_glare_car.xml`'s
  hand-computed `268.6`, which independently confirms the azimuth convention;
- with `anchor="actor"` and a metric `size`, the sprite grows monotonically on approach
  and matches `focal · size_m / distance` (116 px at 100 m → 1583 px at 8 m for
  `size="12"`).

A live run is still worth doing once for anything that touches projection, since only the
offline geometry is covered above.
