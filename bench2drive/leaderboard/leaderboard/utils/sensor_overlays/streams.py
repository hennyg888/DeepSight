#!/usr/bin/env python

"""
Streams: the image feeds an overlay can be composited into, and the geometry needed to
place a world-anchored target in their pixels.

Every image an agent produces per frame is a stream: each rgb camera, and the rendered
LiDAR BEV raster. They are all handed to the overlay layer together, so a future overlay
can mark up the BEV (or only the BEV) without touching the agents again; an overlay
declares which stream *kinds* it understands and the rest are passed through untouched.

  CameraStream  perspective, built straight from an entry of the agent's sensors() list,
                so mount position, yaw/pitch, fov and resolution all come from the one
                place the simulator also reads them from.
  BevStream     the top-down LiDAR raster from tools/lidar_to_bev.py: orthographic,
                ego-centred, forward-up, +right to the right, covering +-range_m.
"""

import math

from .anchors import wrap180, INFINITY


class Placement(object):
    """Where a target landed in a stream's pixels, plus what it takes to size it there"""

    def __init__(self, u, v, distance, px_per_rad=None, px_per_metre=None):
        self.u = u
        self.v = v
        self.distance = distance
        self._px_per_rad = px_per_rad
        self._px_per_metre = px_per_metre

    def angular_size_px(self, degrees):
        """Pixels spanned by an angular size, for streams that have a focal length"""
        if self._px_per_rad is None:
            # Orthographic stream: fall back on the metric size at the target's distance
            if self.distance == INFINITY or self._px_per_metre is None:
                return None
            return self.metric_size_px(2.0 * self.distance * math.tan(math.radians(degrees) / 2.0))
        return 2.0 * self._px_per_rad * math.tan(math.radians(degrees) / 2.0)

    def metric_size_px(self, metres):
        """Pixels spanned by a size in metres at the target's distance"""
        if self._px_per_metre is not None:
            return self._px_per_metre * metres
        if self.distance == INFINITY:
            return None
        return self._px_per_rad * metres / self.distance


class SensorStream(object):
    """Base class. 'kind' is what overlays match on to decide whether they apply."""

    kind = None

    def __init__(self, stream_id, width, height, color_order='bgr'):
        self.id = stream_id
        self.width = int(width)
        self.height = int(height)
        # Camera images travel as BGR (they are written with cv2), the BEV raster as RGB
        # (it is written with PIL), so sprites have to be swapped for one of them.
        self.color_order = color_order

    def project(self, target, ctx):
        """Returns a Placement, or None when the target isn't in front of this stream"""
        raise NotImplementedError


class CameraStream(SensorStream):
    """A perspective rgb camera, described by its entry in the agent's sensors() list"""

    kind = 'camera'

    def __init__(self, spec):
        super(CameraStream, self).__init__(spec['id'], spec['width'], spec['height'])
        self.spec = spec
        self.mount_yaw = float(spec.get('yaw', 0.0))
        self.mount_pitch = float(spec.get('pitch', 0.0))
        self.mount_x = float(spec.get('x', 0.0))
        self.mount_y = float(spec.get('y', 0.0))
        self.mount_z = float(spec.get('z', 0.0))
        self.fov = float(spec['fov'])
        self.focal = (self.width / 2.0) / math.tan(math.radians(self.fov) / 2.0)

    def _location(self, ctx):
        """World location of the camera itself, i.e. the ego pose times the mount offset"""
        import carla
        return ctx.ego_transform.transform(
            carla.Location(x=self.mount_x, y=self.mount_y, z=self.mount_z))

    def project(self, target, ctx):
        world_yaw, altitude, distance = target.direction_from(self._location(ctx))

        # Bearing relative to the camera axis: the ego's heading and the mount yaw are
        # what make the overlay slide across the view as the vehicle turns.
        rel_yaw = math.radians(wrap180(world_yaw - ctx.ego_transform.rotation.yaw - self.mount_yaw))
        # CARLA pitch is positive nose-up while altitude is measured above the horizon,
        # so a camera pitched down (negative) raises the target in the image.
        rel_pitch = math.radians(altitude + self.mount_pitch)

        # Camera frame: x forward, y right, z up
        d_x = math.cos(rel_pitch) * math.cos(rel_yaw)
        if d_x <= 1e-3:
            return None  # behind the image plane
        d_y = math.cos(rel_pitch) * math.sin(rel_yaw)
        d_z = math.sin(rel_pitch)

        return Placement(u=self.width / 2.0 + self.focal * d_y / d_x,
                         v=self.height / 2.0 - self.focal * d_z / d_x,
                         distance=distance,
                         px_per_rad=self.focal)


class BevStream(SensorStream):
    """
    The rendered LiDAR BEV raster: ego-centred and axis-aligned to the ego, covering
    +-range_m in both directions, with the ego's forward axis pointing up the image.
    Mirrors the mapping in tools/lidar_to_bev.bev_projection.
    """

    kind = 'bev'

    def __init__(self, stream_id, range_m, width, height, color_order='rgb'):
        super(BevStream, self).__init__(stream_id, width, height, color_order=color_order)
        self.range_m = float(range_m)
        self.px_per_metre = (self.width - 1) / (2.0 * self.range_m)

    def project(self, target, ctx):
        if target.is_infinite:
            return None  # a bearing alone has no place on a metric top-down raster

        location = target.location
        ego_location = ctx.ego_transform.location
        yaw = math.radians(ctx.ego_transform.rotation.yaw)
        d_x, d_y = location.x - ego_location.x, location.y - ego_location.y

        # Rotate the world offset into the ego frame: forward and right, in metres
        forward = d_x * math.cos(yaw) + d_y * math.sin(yaw)
        right = -d_x * math.sin(yaw) + d_y * math.cos(yaw)
        if abs(forward) > self.range_m or abs(right) > self.range_m:
            return None

        return Placement(u=(right + self.range_m) / (2.0 * self.range_m) * (self.width - 1),
                         v=(-forward + self.range_m) / (2.0 * self.range_m) * (self.height - 1),
                         distance=math.hypot(forward, right),
                         px_per_metre=self.px_per_metre)


def build_camera_streams(sensors):
    """CameraStream for every rgb camera in an agent's sensors() list, keyed by sensor id"""
    return {spec['id']: CameraStream(spec)
            for spec in (sensors or []) if spec.get('type') == 'sensor.camera.rgb'}
