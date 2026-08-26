#!/usr/bin/env python

"""
Sensor overlay layer: post-processing that sits between the simulator's rendered sensor
data and whatever consumes it (a model at inference, or the recorder when collecting
expert data), so a run can be perturbed with effects CARLA itself doesn't render.

The layer is agent-agnostic: every agent hands the same call its per-frame images plus
its sensors() list, and gets them back modified in place. Nothing is applied unless the
route xml asks for it, so agents can call it unconditionally.

A route declares overlays like this:

    <route id="..." town="...">
      <waypoints>...</waypoints>
      <scenarios>...</scenarios>
      <overlays>
        <!-- a low sun washing out the forward cameras, tracking the <weathers> sun -->
        <overlay type="glare" anchor="sun" angular_size="45" intensity="1.0"/>
      </overlays>
    </route>

Each <overlay> names three independent pieces:
  * type    what to draw          -- overlays.py   (glare, ...)
  * anchor  where it sits         -- anchors.py    (sun, location, actor, ...)
  * streams which images to touch -- streams.py    (rgb cameras, the LiDAR BEV raster)
The anchor's own attributes sit on the same element. New types/anchors register themselves
with @register_overlay / @register_anchor and become usable in the xml immediately.

Because the anchor lives in world coordinates and the streams project through the ego's
live pose, an overlay is static in the *world*: turn the ego and it slides across the
cameras and out of view, instead of being painted at a fixed screen position.
"""

import numpy as np

from .anchors import (ANCHOR_TYPES, Anchor, OverlayContext, Target, build_anchor,
                      register_anchor)
from .overlays import OVERLAY_TYPES, Overlay, SpriteOverlay, build_overlay, register_overlay
from .streams import BevStream, CameraStream, SensorStream, build_camera_streams

__all__ = ['ANCHOR_TYPES', 'OVERLAY_TYPES', 'Anchor', 'BevStream', 'CameraStream',
           'Overlay', 'OverlayContext', 'SensorStream', 'SpriteOverlay', 'Target',
           'apply_sensor_overlays', 'build_anchor', 'build_camera_streams',
           'build_overlay', 'get_active_overlays', 'parse_overlays', 'register_anchor',
           'register_overlay', 'set_active_overlays']


# The route being run publishes its overlays here (see leaderboard_evaluator), so agents
# -- which never see the route config -- can pick them up without a new agent API.
_ACTIVE = []


def set_active_overlays(overlays):
    """Called once per route by the leaderboard, with the parsed overlays (possibly [])"""
    global _ACTIVE
    _ACTIVE = list(overlays or [])


def get_active_overlays():
    return [o for o in _ACTIVE if o.enabled]


def parse_overlays(route_element):
    """
    Parses a route's <overlays> block (and a bare <glare> element, accepted as shorthand
    for a single glare overlay) into a list of Overlay instances. Returns [] if neither
    is present.
    """
    overlays = []
    elements = []

    block = route_element.find('overlays')
    if block is not None:
        elements += list(block.iter('overlay'))
    elements += list(route_element.findall('glare'))

    for element in elements:
        attrib = dict(element.attrib)
        overlay_type = attrib.pop('type', 'glare' if element.tag == 'glare' else None)
        if overlay_type is None:
            raise ValueError("<overlay> needs a 'type' attribute")
        anchor = build_anchor(attrib.pop('anchor', 'sun'), attrib)
        overlays.append(build_overlay(overlay_type, attrib, anchor))

    return overlays


def apply_sensor_overlays(images, ego_vehicle, sensors=None, streams=None, timestamp=None):
    """
    Applies the active route's overlays to this frame's images, in place, and returns the
    (possibly re-bound) dict.

      images     stream id -> HxWx3 uint8 array. Pass *every* image the frame produced,
                 including the LiDAR BEV raster: an overlay only touches the streams its
                 type supports, and passing the rest through is what lets a future
                 overlay draw on them without another round of agent edits.
      ego_vehicle  the ego actor, for its live pose (and the world, for actor anchors)
      sensors    the agent's sensors() list; every rgb camera in it becomes a stream
      streams    extra streams that aren't CARLA sensors, e.g.
                 {'LIDAR_BEV': BevStream('LIDAR_BEV', BEV_RANGE_M, 644, 644)}

    A no-op -- and cheap -- when the route declared no overlays.
    """
    overlays = get_active_overlays()
    if not overlays:
        return images

    stream_map = build_camera_streams(sensors)
    if streams:
        stream_map.update(streams)

    ctx = OverlayContext(ego_vehicle, timestamp)
    for overlay in overlays:
        target = overlay.anchor.resolve(ctx)
        if target is None:
            continue  # e.g. an actor anchor whose actor isn't spawned yet
        for stream_id in overlay.stream_ids(stream_map):
            image = images.get(stream_id)
            if image is None:
                continue
            if not image.flags['WRITEABLE'] or not image.flags['C_CONTIGUOUS']:
                # Sensor buffers arrive as read-only slices; own a copy before drawing
                image = np.ascontiguousarray(image)
                images[stream_id] = image
            overlay.apply(image, stream_map[stream_id], target, ctx)

    return images
