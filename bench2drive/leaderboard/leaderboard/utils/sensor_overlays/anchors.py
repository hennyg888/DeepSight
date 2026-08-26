#!/usr/bin/env python

"""
Anchors: where an overlay's subject sits in the *simulation world* on a given frame.

An anchor answers one question per frame -- "where is the thing I'm drawing?" -- and
hands back a Target. Streams then work out where that target lands in their own pixels.
Splitting it this way is what keeps an overlay world-anchored rather than screen-anchored:
the anchor never knows about cameras, and the projection never knows what is being drawn.

Targets come in two flavours:
  * infinitely distant (the sun, a far-off floodlight) -- only a bearing is meaningful,
    so the overlay's size is angular and its position doesn't parallax with ego motion;
  * finite (an actor, a fixed world point) -- has a location, so its bearing, its distance
    and hence its apparent size all change as the ego drives.

Add a new anchor by subclassing Anchor and decorating it with @register_anchor('name');
it is then usable as anchor="name" in a route xml's <overlay> element.
"""

import math

INFINITY = float('inf')

ANCHOR_TYPES = {}


def register_anchor(name):
    def decorator(cls):
        cls.type_name = name
        ANCHOR_TYPES[name] = cls
        return cls
    return decorator


def build_anchor(name, attrib):
    if name not in ANCHOR_TYPES:
        raise ValueError("Unknown overlay anchor '{}', expected one of {}".format(
            name, ', '.join(sorted(ANCHOR_TYPES))))
    return ANCHOR_TYPES[name](attrib)


def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


class OverlayContext(object):
    """
    Per-frame simulation state that anchors and overlays read. Built once per frame by
    apply_sensor_overlays and shared by every overlay, so the (comparatively expensive)
    transform / weather lookups happen at most once.
    """

    def __init__(self, ego_vehicle, timestamp=None):
        self.ego_vehicle = ego_vehicle
        self.ego_transform = ego_vehicle.get_transform()
        self.timestamp = timestamp
        self._world = None
        self._weather = None

    @property
    def world(self):
        if self._world is None:
            self._world = self.ego_vehicle.get_world()
        return self._world

    @property
    def weather(self):
        if self._weather is None:
            self._weather = self.world.get_weather()
        return self._weather


class Target(object):
    """
    Resolved subject of an overlay for one frame: either a world location, or a pure
    bearing for a source treated as infinitely far away.
    """

    def __init__(self, location=None, world_yaw=None, altitude=0.0):
        if location is None and world_yaw is None:
            raise ValueError("A Target needs either a location or a world_yaw")
        self.location = location
        self.world_yaw = world_yaw
        self.altitude = altitude

    @property
    def is_infinite(self):
        return self.location is None

    def direction_from(self, location):
        """
        Returns (world_yaw, altitude, distance) in degrees / metres, as seen from
        'location'. An infinite target ignores the viewpoint and reports distance inf.
        """
        if self.is_infinite:
            return self.world_yaw, self.altitude, INFINITY

        d_x = self.location.x - location.x
        d_y = self.location.y - location.y
        d_z = self.location.z - location.z
        planar = math.hypot(d_x, d_y)
        return (math.degrees(math.atan2(d_y, d_x)),
                math.degrees(math.atan2(d_z, planar)) if planar > 1e-6 else 90.0,
                math.sqrt(planar * planar + d_z * d_z))


class Anchor(object):
    """Base class: resolve() returns this frame's Target, or None to skip the overlay"""

    type_name = None

    def __init__(self, attrib):
        self.attrib = attrib

    def resolve(self, ctx):
        raise NotImplementedError

    def __str__(self):
        return self.type_name


@register_anchor('sun')
class SunAnchor(Anchor):
    """
    A light source infinitely far away, aimed with CARLA's *sun* convention: 'azimuth'
    is the same quantity as <weather sun_azimuth_angle=...> (it points from the light,
    so the bearing the ego must look along to see it is azimuth - 180), and 'altitude'
    matches sun_altitude_angle.

    Attributes:
      azimuth, altitude   degrees; omit either to take the simulator's live sun value,
                          which makes the overlay follow a route's <weathers> keyframes.
      follow_weather      defaults to true when azimuth or altitude is omitted. Set it
                          false only together with both angles, to pin the source.
    """

    def __init__(self, attrib):
        super(SunAnchor, self).__init__(attrib)
        self.azimuth = float(attrib['azimuth']) if 'azimuth' in attrib else None
        self.altitude = float(attrib['altitude']) if 'altitude' in attrib else None
        follow = attrib.get('follow_weather')
        self.follow_weather = (self.azimuth is None or self.altitude is None) if follow is None \
            else str(follow).strip().lower() in ('1', 'true', 'yes', 'on')
        if not self.follow_weather and (self.azimuth is None or self.altitude is None):
            raise ValueError("anchor 'sun' needs both azimuth and altitude, "
                             "or follow_weather=\"true\" to read them off the weather")

    def resolve(self, ctx):
        azimuth, altitude = self.azimuth, self.altitude
        if self.follow_weather:
            weather = ctx.weather
            if azimuth is None:
                azimuth = float(weather.sun_azimuth_angle)
            if altitude is None:
                altitude = float(weather.sun_altitude_angle)
        # CARLA's azimuth points from the source, the bearing to it is the opposite one
        return Target(world_yaw=wrap180(azimuth - 180.0), altitude=altitude)

    def __str__(self):
        return 'sun(azimuth={}, altitude={}{})'.format(
            self.azimuth if self.azimuth is not None else 'weather',
            self.altitude if self.altitude is not None else 'weather',
            ', follow_weather' if self.follow_weather else '')


@register_anchor('location')
class LocationAnchor(Anchor):
    """
    A fixed point in the map, at x/y/z. Finite, so it parallaxes and grows as the ego
    approaches it.
    """

    def __init__(self, attrib):
        super(LocationAnchor, self).__init__(attrib)
        import carla
        self._location = carla.Location(float(attrib['x']), float(attrib['y']),
                                        float(attrib.get('z', 0.0)))

    def resolve(self, ctx):
        return Target(location=self._location)

    def __str__(self):
        return 'location({:.1f}, {:.1f}, {:.1f})'.format(
            self._location.x, self._location.y, self._location.z)


@register_anchor('distant')
class DistantAnchor(Anchor):
    """
    A light infinitely far away, in the direction of whatever another anchor resolves to.
    Use it for a source that is *beyond* something in the scene -- a sun sitting behind
    the obstacle the ego is driving towards -- where the bearing has to track that thing
    but the size and parallax must stay those of a distant source.

    Attributes:
      toward     which anchor supplies the direction (default 'actor'); that anchor's
                 own attributes sit on the same element
      altitude   elevation in degrees; omit to use the aimed-at target's own elevation
    """

    def __init__(self, attrib):
        super(DistantAnchor, self).__init__(attrib)
        toward = attrib.get('toward', 'actor')
        if toward == 'distant':
            raise ValueError("anchor 'distant' cannot aim at another 'distant' anchor")
        self._toward = build_anchor(toward, attrib)
        self.altitude = float(attrib['altitude']) if 'altitude' in attrib else None

    def resolve(self, ctx):
        target = self._toward.resolve(ctx)
        if target is None:
            return None
        # Taken from the ego rather than from each camera: for a source treated as
        # infinitely far, the mount offsets are exactly the parallax being ignored.
        world_yaw, altitude, _ = target.direction_from(ctx.ego_transform.location)
        return Target(world_yaw=world_yaw,
                      altitude=self.altitude if self.altitude is not None else altitude)

    def __str__(self):
        return 'distant(toward={}, altitude={})'.format(
            self._toward, self.altitude if self.altitude is not None else 'target')


@register_anchor('actor')
class ActorAnchor(Anchor):
    """
    An actor in the simulation, so the overlay rides whatever it is attached to (a
    scenario vehicle's headlights, a prop, ...). The actor is matched on the first of
    'role_name' / 'type_id' / 'actor_id' that is given, and the match is cached.

    Attributes:
      role_name / type_id / actor_id   how to find it ('type_id' matches a prefix, e.g.
                                       vehicle.tesla)
      z_offset                         metres added to the actor's origin (default 0),
                                       e.g. to sit at headlight height
      ego                              true anchors to the ego itself
    """

    def __init__(self, attrib):
        super(ActorAnchor, self).__init__(attrib)
        self.role_name = attrib.get('role_name')
        self.type_id = attrib.get('type_id')
        self.actor_id = int(attrib['actor_id']) if 'actor_id' in attrib else None
        self.ego = str(attrib.get('ego', '')).strip().lower() in ('1', 'true', 'yes', 'on')
        self.z_offset = float(attrib.get('z_offset', 0.0))
        if not (self.role_name or self.type_id or self.actor_id is not None or self.ego):
            raise ValueError("anchor 'actor' needs one of role_name, type_id, actor_id or ego")
        self._actor = None

    def _find_actor(self, ctx):
        if self.ego:
            return ctx.ego_vehicle
        for actor in ctx.world.get_actors():
            if self.actor_id is not None:
                if actor.id == self.actor_id:
                    return actor
            elif self.role_name is not None:
                if actor.attributes.get('role_name') == self.role_name:
                    return actor
            elif actor.type_id.startswith(self.type_id):
                return actor
        return None

    def resolve(self, ctx):
        if self._actor is None or not self._actor.is_alive:
            self._actor = self._find_actor(ctx)
        if self._actor is None:
            # The actor may simply not be spawned yet -- retry next frame rather than raise
            return None
        location = self._actor.get_transform().location
        import carla
        return Target(location=location + carla.Location(z=self.z_offset))

    def __str__(self):
        which = 'ego' if self.ego else (self.role_name or self.type_id or self.actor_id)
        return 'actor({})'.format(which)
