#!/usr/bin/env python

"""
Overlays: what actually gets drawn, once an anchor has said where and a stream has said
in which pixels.

An overlay declares the stream kinds it understands (STREAM_KINDS) and is only ever
handed those, so e.g. the glare -- a camera artefact -- never touches the LiDAR BEV even
though the BEV is passed through the same layer.

Add a new one by subclassing Overlay (or SpriteOverlay, which already does image loading,
scaling, blending and clipping) and decorating it with @register_overlay('name'); it is
then usable as <overlay type="name" .../> in a route xml.
"""

import os

import cv2
import numpy as np

OVERLAY_TYPES = {}


def register_overlay(name):
    def decorator(cls):
        cls.type_name = name
        OVERLAY_TYPES[name] = cls
        return cls
    return decorator


def build_overlay(name, attrib, anchor, base_dir=None):
    if name not in OVERLAY_TYPES:
        raise ValueError("Unknown overlay type '{}', expected one of {}".format(
            name, ', '.join(sorted(OVERLAY_TYPES))))
    return OVERLAY_TYPES[name](attrib, anchor, base_dir)


def as_bool(value, default):
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


class Overlay(object):
    """
    Base class. Attributes common to every overlay:

      type      which Overlay subclass to build
      anchor    which Anchor subclass to aim it with (default 'sun'); the anchor's own
                attributes live on the same element
      streams   comma separated stream ids to draw into. Defaults to every stream of a
                kind this overlay supports, minus DEFAULT_EXCLUDED_STREAMS.
      enabled   false keeps the element in the xml without drawing anything
    """

    type_name = None
    STREAM_KINDS = ()
    DEFAULT_EXCLUDED_STREAMS = ()

    def __init__(self, attrib, anchor, base_dir=None):
        self.attrib = attrib
        self.anchor = anchor
        self.base_dir = base_dir or os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
        self.enabled = as_bool(attrib.get('enabled'), True)
        # 'cameras' is accepted as a synonym so a camera-only overlay reads naturally
        selected = attrib.get('streams', attrib.get('cameras'))
        self.selected_streams = tuple(s.strip() for s in selected.split(',') if s.strip()) \
            if selected else None

    def resolve_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.base_dir, path)

    def stream_ids(self, streams):
        """The ids of 'streams' (id -> SensorStream) this overlay should be applied to"""
        if self.selected_streams is not None:
            return [s for s in self.selected_streams
                    if s in streams and streams[s].kind in self.STREAM_KINDS]
        return [s for s, stream in streams.items()
                if stream.kind in self.STREAM_KINDS and s not in self.DEFAULT_EXCLUDED_STREAMS]

    def apply(self, image, stream, target, ctx):
        """Draws into 'image' (HxWx3 uint8) in place and returns it"""
        raise NotImplementedError


class SpriteOverlay(Overlay):
    """
    Composites an image file onto a stream, centred on the anchor's projected position.

      image         sprite path, absolute or relative to the repo root. An alpha channel
                    is honoured.
      angular_size  apparent diameter in degrees (perspective streams). Being an angle
                    rather than a pixel count, it survives a change of fov or resolution.
      size          diameter in metres instead, for a finite anchor; on a perspective
                    stream this shrinks with distance the way a real object would.
      intensity     multiplier applied to the sprite before blending (default 1.0).
      blend         'screen' (default): light adds and never darkens what is already
                    bright -- the right model for a light source. 'alpha': plain
                    alpha compositing, which occludes instead.
    """

    DEFAULT_ANGULAR_SIZE = 30.0
    DEFAULT_IMAGE = None  # subclasses set the sprite they ship with

    def __init__(self, attrib, anchor, base_dir=None):
        super(SpriteOverlay, self).__init__(attrib, anchor, base_dir)

        image = attrib.get('image', self.DEFAULT_IMAGE)
        if image is None:
            raise ValueError("overlay type '{}' needs an 'image'".format(self.type_name))
        self.image_path = self.resolve_path(image)
        self.intensity = float(attrib.get('intensity', 1.0))
        self.blend = attrib.get('blend', 'screen')
        if self.blend not in ('screen', 'alpha'):
            raise ValueError("overlay blend must be 'screen' or 'alpha', got '{}'".format(self.blend))
        self.size_m = float(attrib['size']) if 'size' in attrib else None
        self.angular_size = float(attrib.get('angular_size', self.DEFAULT_ANGULAR_SIZE))

        sprite = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if sprite is None:
            raise ValueError("Couldn't read the overlay image '{}'".format(self.image_path))
        if sprite.ndim == 2:
            sprite = cv2.cvtColor(sprite, cv2.COLOR_GRAY2BGRA)
        elif sprite.shape[2] == 3:
            sprite = cv2.cvtColor(sprite, cv2.COLOR_BGR2BGRA)
        self._bgr = sprite[:, :, :3].astype(np.float32)
        self._alpha = sprite[:, :, 3].astype(np.float32) / 255.0
        # A route only ever uses a handful of (size, colour order) pairs -- one per
        # stream, constant unless the anchor's distance changes -- so this cache is tiny.
        self._scaled = {}

    def _scaled_sprite(self, size, color_order):
        key = (size, color_order)
        cached = self._scaled.get(key)
        if cached is None:
            bgr = self._bgr if color_order == 'bgr' else self._bgr[:, :, ::-1]
            colour = np.clip(bgr * self._alpha[..., None] * self.intensity, 0, 255).astype(np.uint8)
            alpha = np.clip(self._alpha * self.intensity, 0, 1).astype(np.float32)
            interp = cv2.INTER_AREA if size < colour.shape[0] else cv2.INTER_LINEAR
            cached = (cv2.resize(colour, (size, size), interpolation=interp),
                      cv2.resize(alpha, (size, size), interpolation=interp))
            if len(self._scaled) > 32:
                self._scaled.clear()
            self._scaled[key] = cached
        return cached

    def sprite_size_px(self, placement):
        """Diameter in pixels for this frame: metric if the xml gave one, angular otherwise"""
        if self.size_m is not None:
            size_px = placement.metric_size_px(self.size_m)
            if size_px is None:
                # A size in metres is meaningless at infinite distance. Raise rather than
                # draw nothing, so the mistake surfaces on the first frame.
                raise ValueError(
                    "overlay '{}' gives size=\"{}\" in metres, but anchor '{}' is an "
                    "infinitely distant source -- use angular_size (degrees) instead"
                    .format(self.type_name, self.size_m, self.anchor))
            return size_px
        return placement.angular_size_px(self.angular_size)

    def apply(self, image, stream, target, ctx):
        placement = stream.project(target, ctx)
        if placement is None:
            return image

        size_px = self.sprite_size_px(placement)
        if size_px is None or size_px < 2:
            return image
        size = int(round(size_px))
        colour, alpha = self._scaled_sprite(size, stream.color_order)

        # Clip the sprite rectangle against the image, so a source at the edge of the
        # frustum draws the part of its halo that is actually in view
        x_0, y_0 = int(round(placement.u - size / 2.0)), int(round(placement.v - size / 2.0))
        height, width = image.shape[:2]
        src_x, src_y = max(0, -x_0), max(0, -y_0)
        dst_x0, dst_y0 = max(0, x_0), max(0, y_0)
        dst_x1, dst_y1 = min(width, x_0 + size), min(height, y_0 + size)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return image

        colour = colour[src_y:src_y + (dst_y1 - dst_y0), src_x:src_x + (dst_x1 - dst_x0)]
        region = image[dst_y0:dst_y1, dst_x0:dst_x1].astype(np.float32)

        if self.blend == 'screen':
            blended = 255.0 - (255.0 - region) * (255.0 - colour.astype(np.float32)) / 255.0
        else:
            a = alpha[src_y:src_y + (dst_y1 - dst_y0), src_x:src_x + (dst_x1 - dst_x0)][..., None]
            blended = colour.astype(np.float32) + region * (1.0 - a)

        image[dst_y0:dst_y1, dst_x0:dst_x1] = np.clip(blended, 0, 255).astype(np.uint8)
        return image


@register_overlay('glare')
class GlareOverlay(SpriteOverlay):
    """
    A glare halo from a light source, composited onto the rgb cameras after CARLA has
    rendered them and before the model sees them -- a lens/sensor artefact the simulator
    doesn't reproduce, not a light in the scene. It therefore leaves the LiDAR (and the
    BEV raster rendered from it) alone, and never touches the top-down CAM_BEV.

    Extra attribute on top of SpriteOverlay's:
      min_altitude  skip while the source sits below this elevation. Defaults to 0 for an
                    infinitely distant source -- no glare once the sun has set -- and to
                    no gate at all for a source anchored to something in the scene, which
                    sits at roughly camera height and would otherwise be cut off by a
                    horizon it is nowhere near. Give it explicitly to gate either kind.
    """

    STREAM_KINDS = ('camera',)
    DEFAULT_EXCLUDED_STREAMS = ('CAM_BEV',)
    DEFAULT_IMAGE = os.path.join('assets', 'glare.png')

    def __init__(self, attrib, anchor, base_dir=None):
        super(GlareOverlay, self).__init__(attrib, anchor, base_dir)
        self.min_altitude = float(attrib.get('min_altitude', 0.0))
        self._gate_altitude = 'min_altitude' in attrib

    def apply(self, image, stream, target, ctx):
        if self._gate_altitude or target.is_infinite:
            _, altitude, _ = target.direction_from(ctx.ego_transform.location)
            if altitude < self.min_altitude:
                return image
        return super(GlareOverlay, self).apply(image, stream, target, ctx)

    def __str__(self):
        size = '{} m'.format(self.size_m) if self.size_m is not None \
            else '{} deg'.format(self.angular_size)
        return 'glare(anchor={}, size={}, intensity={}, image={})'.format(
            self.anchor, size, self.intensity, os.path.basename(self.image_path))
