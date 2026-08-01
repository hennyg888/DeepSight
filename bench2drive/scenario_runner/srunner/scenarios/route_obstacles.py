#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Scenarios in which another (opposite) vehicle 'illegally' takes
priority, e.g. by running a red traffic light.
"""

from __future__ import print_function

import math

import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorDestroy,
                                                                      SwitchWrongDirectionTest,
                                                                      BasicAgentBehavior,
                                                                      ScenarioTimeout,
                                                                      Idle, WaitForever,
                                                                      HandBrakeVehicle,
                                                                      OppositeActorFlow)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, ScenarioTimeoutTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (ROUTE_END_REQUESTED_VAR,
                                                                               DriveDistance,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle,
                                                                               StandStill,
                                                                               WaitForCollision,
                                                                               WaitUntilInFront,
                                                                               WaitUntilInFrontPosition)
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.background_manager import LeaveSpaceInFront, SetMaxSpeed, ChangeOppositeBehavior, ChangeRoadBehavior


def get_value_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return p_type(config.other_parameters[name]['value'])
    else:
        return default

def get_bool_parameter(config, name, default):
    """Route XML has no native bool, so accept the usual spellings of one"""
    if name not in config.other_parameters:
        return default
    return str(config.other_parameters[name]['value']).strip().lower() in ('1', 'true', 'yes', 'on')


def get_interval_parameter(config, name, p_type, default):
    if name in config.other_parameters:
        return [
            p_type(config.other_parameters[name]['from']),
            p_type(config.other_parameters[name]['to'])
        ]
    else:
        return default


class Accident(BasicScenario):
    """
    This class holds everything required for a scenario in which there is an accident
    in front of the ego, forcing it to lane change. A police vehicle is located before
    two other cars that have been in an accident.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout
        
        self._first_distance = 10
        self._second_distance = 6

        self._trigger_distance = 50
        self._end_distance = 50
        self._wait_duration = 5
        self._offset = 0.6

        self._lights = carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2 | carla.VehicleLightState.Position

        self._distance = get_value_parameter(config, 'distance', float, 120)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        super().__init__(
            "Accident", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_side_prop(self, wp):
        # Spawn the accident indication signal
        prop_wp = wp
        while True:
            if self._direction == "right":
                wp = prop_wp.get_right_lane()
            else:
                wp = prop_wp.get_left_lane()
            if wp is None or wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                break
            prop_wp = wp

        displacement = 0.3 * prop_wp.lane_width
        r_vec = prop_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=0.2)
        spawn_transform.rotation.yaw += 90
        signal_prop = CarlaDataProvider.request_new_actor('static.prop.warningaccident', spawn_transform)
        if not signal_prop:
            raise ValueError("Couldn't spawn the indication prop asset")
        signal_prop.set_simulate_physics(False)
        self.other_actors.append(signal_prop)

    def _spawn_obstacle(self, wp, blueprint, accident_actor=False):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        if accident_actor:
            actor = CarlaDataProvider.request_new_actor(
                blueprint, spawn_transform, rolename='scenario no lights', attribute_filter={'base_type': 'car', 'generation': 2})
        else:
            actor = CarlaDataProvider.request_new_actor(
                blueprint, spawn_transform, rolename='scenario')
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Spawn the accident indication signal
        self._spawn_side_prop(starting_wp)

        # Spawn the police vehicle
        self._accident_wp = self._move_waypoint_forward(starting_wp, self._distance)
        police_car = self._spawn_obstacle(self._accident_wp, 'vehicle.dodge.charger_police_2020')

        # Set its initial conditions
        lights = police_car.get_light_state()
        lights |= self._lights
        police_car.set_light_state(carla.VehicleLightState(lights))
        police_car.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(police_car)

        # Create the first vehicle that has been in the accident
        self._first_vehicle_wp = self._move_waypoint_forward(self._accident_wp, self._first_distance)
        first_actor = self._spawn_obstacle(self._first_vehicle_wp, 'vehicle.*', True)

        # Set its initial conditions
        first_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(first_actor)

        # Create the second vehicle that has been in the accident
        second_vehicle_wp = self._move_waypoint_forward(self._first_vehicle_wp, self._second_distance)
        second_actor = self._spawn_obstacle(second_vehicle_wp, 'vehicle.*', True)

        self._accident_wp = second_vehicle_wp
        self._end_wp = self._move_waypoint_forward(second_vehicle_wp, self._end_distance)

        # Set its initial conditions
        second_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(second_actor)

    def _create_behavior(self):
        """
        The vehicle has to drive the reach a specific point but an accident is in the middle of the road,
        blocking its route and forcing it to lane change.
        """
        root = py_trees.composites.Sequence(name="Accident")
        if self.route_mode:
            total_dist = self._distance + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._first_vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class AccidentTwoWays(Accident):
    """
    Variation of the Accident scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [20, 100])
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        reference_wp = self._accident_wp.get_left_lane()
        if not reference_wp:
            raise ValueError("Couldnt find a left lane to spawn the opposite traffic")

        root = py_trees.composites.Sequence(name="AccidentTwoWays")
        if self.route_mode:
            total_dist = self._distance + self._first_distance + self._second_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._first_vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(active=False))
        behavior.add_child(OppositeActorFlow(reference_wp, self.ego_vehicles[0], self._opposite_interval))

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

class ParkedObstacle(BasicScenario):
    """
    Scenarios in which a parked vehicle is incorrectly parked,
    forcing the ego to lane change out of the route's lane
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._trigger_distance = 50
        self._end_distance = 50
        self._wait_duration = 5
        self._offset = 0.7

        self._lights = carla.VehicleLightState.RightBlinker | carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.Position

        self._distance = get_value_parameter(config, 'distance', float, 120)
        self._direction = get_value_parameter(config, 'direction', str, 'right')
        if self._direction not in ('left', 'right'):
            raise ValueError(f"'direction' must be either 'right' or 'left' but {self._direction} was given")

        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._scenario_timeout = 240

        # Opt-in: end the scenario once the ego has *resolved* the obstacle rather than
        # once it has driven past one. The stock end condition (WaitUntilInFrontPosition,
        # below) only fires after the ego has overtaken, so a route whose whole point is
        # that the ego should stop -- and never merge out -- otherwise runs until the
        # 240 s scenario timeout no matter what the ego does. Off by default so the 220
        # standard routes, where overtaking *is* the expected behaviour, are unaffected.
        self._end_on_encounter = get_bool_parameter(config, 'end_on_encounter', False)
        self._stop_distance = get_value_parameter(config, 'stop_distance', float, 15.0)
        self._stop_duration = get_value_parameter(config, 'stop_duration', float, 2.0)

        super().__init__(
            "ParkedObstacle", ego_vehicles, config, world, randomize, debug_mode, criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_side_prop(self, wp):
        # Spawn the accident indication signal
        prop_wp = wp
        while True:
            if self._direction == "right":
                wp = prop_wp.get_right_lane()
            else:
                wp = prop_wp.get_left_lane()
            if wp is None or wp.lane_type not in (carla.LaneType.Driving, carla.LaneType.Parking):
                break
            prop_wp = wp

        displacement = 0.3 * prop_wp.lane_width
        r_vec = prop_wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=0.2)
        spawn_transform.rotation.yaw += 90
        signal_prop = CarlaDataProvider.request_new_actor('static.prop.warningaccident', spawn_transform)
        if not signal_prop:
            raise ValueError("Couldn't spawn the indication prop asset")
        signal_prop.set_simulate_physics(False)
        self.other_actors.append(signal_prop)

    def _spawn_obstacle(self, wp, blueprint):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        actor = CarlaDataProvider.request_new_actor(
            blueprint, spawn_transform, rolename='scenario no lights', attribute_filter={'base_type': 'car', 'generation': 2})
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Create the side prop
        self._spawn_side_prop(self._starting_wp)

        # Create the first vehicle that has been in the accident
        self._vehicle_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        parked_actor = self._spawn_obstacle(self._vehicle_wp, 'vehicle.*')

        lights = parked_actor.get_light_state()
        lights |= self._lights
        parked_actor.set_light_state(carla.VehicleLightState(lights))
        parked_actor.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(parked_actor)
        # Kept by name: other_actors[0] is the side prop, and subclasses append further
        # actors (the glare vehicle), so positional indexing is not stable.
        self._parked_actor = parked_actor

        self._end_wp = self._move_waypoint_forward(self._vehicle_wp, self._end_distance)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance.
        """
        root = py_trees.composites.Sequence(name="ParkedObstacle")
        if self.route_mode:
            total_dist = self._distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))
        if self._end_on_encounter:
            end_condition.add_child(self._create_encounter_end_condition())

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self._end_on_encounter:
            # Raised before the ActorDestroy children below, so the last recorded frames
            # still show the parked car rather than the empty road it leaves behind.
            root.add_child(py_trees.blackboard.SetBlackboardVariable(
                "Request route end: {}".format(self.config.name), ROUTE_END_REQUESTED_VAR, True))

        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

    def _create_encounter_end_condition(self):
        """
        SUCCESS once the ego has resolved the obstacle, either way it can go: it came to a
        stop in front of the parked car, or it drove into it.

        Both outcomes end the route -- which of the two happened is already recorded by
        the route's CollisionTest, so this only has to decide *when* to stop, not judge.

        The stop branch is a Sequence rather than a Parallel: 'closed to within
        stop_distance, and then stood still for stop_duration'. It deliberately does not
        re-check the distance once satisfied, so an ego that crawls the last few metres
        still counts as having stopped. That is safe here only because these routes assume
        the ego never merges out and leaves; a scenario where it might should gate the
        StandStill on distance as well.
        """
        obstacle_actors = self._encounter_actors()

        resolved = py_trees.composites.Parallel(
            name="ObstacleEncounterResolved", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        # An obstacle can be made of several actors (a pedestrian crowd); reaching or
        # hitting any one of them counts as having met it.
        reached = py_trees.composites.Parallel(
            name="EgoReachedObstacle", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        for actor in obstacle_actors:
            # freespace: bumper-to-bumper rather than centre-to-centre, so stop_distance
            # means the gap you would actually see in the camera frame.
            reached.add_child(InTriggerDistanceToVehicle(
                actor, self.ego_vehicles[0], self._stop_distance, freespace=True,
                name="EgoReachedObstacle"))

        stopped = py_trees.composites.Sequence(name="EgoStoppedInFrontOfObstacle")
        stopped.add_child(reached)
        stopped.add_child(StandStill(
            self.ego_vehicles[0], duration=self._stop_duration, name="EgoStandStill"))
        resolved.add_child(stopped)

        resolved.add_child(WaitForCollision(
            self.ego_vehicles[0], obstacle_actors, name="EgoHitObstacle"))

        return resolved

    def _encounter_actors(self):
        """
        The actors that count as 'the obstacle' for the end condition above. Subclasses
        that put more than one thing in the ego's way override this.
        """
        return [self._parked_actor]

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class ParkedObstacleTwoWays(ParkedObstacle):
    """
    Variation of the ParkedObstacle scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_interval = get_interval_parameter(config, 'frequency', float, [20, 100])
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        The vehicle has to drive the whole predetermined distance. Adapt the opposite flow to
        let the ego invade the opposite lane.
        """
        reference_wp = self._vehicle_wp.get_left_lane()
        if not reference_wp:
            raise ValueError("Couldnt find a left lane to spawn the opposite traffic")

        root = py_trees.composites.Sequence(name="ParkedObstacleTwoWays")
        if self.route_mode:
            total_dist = self._distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))

        end_condition = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))
        end_condition.add_child(WaitUntilInFrontPosition(self.ego_vehicles[0], self._end_wp.transform, False))

        behavior = py_trees.composites.Sequence()
        behavior.add_child(InTriggerDistanceToLocation(
            self.ego_vehicles[0], self._vehicle_wp.transform.location, self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(active=False))
        behavior.add_child(OppositeActorFlow(reference_wp, self.ego_vehicles[0], self._opposite_interval))

        end_condition.add_child(behavior)
        root.add_child(end_condition)

        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(True))
            root.add_child(ChangeOppositeBehavior(active=True))
        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root


class ParkedObstacleNoLights(ParkedObstacle):
    """
    Variation of the ParkedObstacle scenario where the parked vehicle sits in the middle of
    the ego's lane with every light off (no hazards, no position/head lights, no brake lights)
    and without the roadside warning prop, i.e. an unlit, unsignalled obstacle.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        # Read before the parent's __init__, which triggers _initialize_actors
        self._lane_offset = get_value_parameter(config, 'offset', float, 0.0)
        self._color = get_value_parameter(config, 'color', str, '0,0,0')
        self._model = get_value_parameter(config, 'model', str, 'vehicle.tesla.model3')
        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _spawn_side_prop(self, wp):
        """No roadside warning sign for this variation"""
        pass

    def _spawn_obstacle(self, wp, blueprint):
        """
        Same as the parent, but pins the blueprint to a single model and forces its color,
        instead of drawing a random model/color pair out of the blueprint library. The
        'blueprint' argument the parent passes in ('vehicle.*') is deliberately ignored.
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()
        if self._direction == 'left':
            r_vec *= -1

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        actor = CarlaDataProvider.request_new_actor(
            self._model, spawn_transform, rolename='scenario no lights', color=self._color)
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        # Center the obstacle in the lane instead of displacing it towards the lane edge
        self._offset = self._lane_offset
        self._lights = carla.VehicleLightState.NONE

        super()._initialize_actors(config)

        # The parent ORs its mask onto whatever the blueprint spawned with, so clear the light
        # state outright, and make sure the parked car isn't holding the brake pedal (which
        # would light up the brake lights) while still being immobile.
        parked_actor = self.other_actors[-1]
        parked_actor.set_light_state(carla.VehicleLightState.NONE)
        parked_actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=True))

    def _create_behavior(self):
        behavior = super()._create_behavior()
        behavior.name = "ParkedObstacleNoLights"
        return behavior


class ParkedObstacleNoLightsWithGlare(ParkedObstacleNoLights):
    """
    ParkedObstacleNoLights plus a second stationary vehicle in the lane next to the ego's,
    turned around to face the ego with its high beams on. The glare is meant to wash out the
    camera so the unlit obstacle in the ego's own lane is harder to pick out.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        # Read before the parent's __init__, which triggers _initialize_actors
        self._glare_side = get_value_parameter(config, 'glare_side', str, 'left')
        if self._glare_side not in ('left', 'right'):
            raise ValueError(f"'glare_side' must be either 'left' or 'right' but {self._glare_side} was given")

        # Defaults to None so they can be resolved from 'distance' once the parent has parsed it
        self._glare_distance = get_value_parameter(config, 'glare_distance', float, None)
        self._glare_aim_distance = get_value_parameter(config, 'glare_aim_distance', float, None)
        self._glare_yaw_offset = get_value_parameter(config, 'glare_yaw_offset', float, 0.0)
        self._glare_model = get_value_parameter(config, 'glare_model', str, 'vehicle.tesla.model3')
        self._glare_lights = carla.VehicleLightState.HighBeam | carla.VehicleLightState.LowBeam \
            | carla.VehicleLightState.Position

        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _get_glare_waypoint(self, wp):
        """
        Returns the waypoint of the neighbouring lane the glare vehicle is parked on,
        preferring the requested side and falling back to the other one
        """
        sides = [self._glare_side, 'right' if self._glare_side == 'left' else 'left']
        for side in sides:
            side_wp = wp.get_left_lane() if side == 'left' else wp.get_right_lane()
            if side_wp and side_wp.lane_type == carla.LaneType.Driving:
                return side_wp

        raise ValueError("Couldn't find a neighbouring driving lane to park the glare vehicle on")

    def _spawn_glare_vehicle(self, wp, aim_location):
        """
        Spawns a stationary vehicle on the given waypoint, aimed at 'aim_location',
        with its high beams on
        """
        spawn_transform = carla.Transform(wp.transform.location + carla.Location(z=1), wp.transform.rotation)

        # Toe the vehicle in towards the ego's lane instead of leaving it square with its own one,
        # so the headlight cones point into the camera rather than sweeping past it. Aiming also
        # covers the yaw flip, as a neighbouring lane may run either way.
        aim_vector = aim_location - spawn_transform.location
        spawn_transform.rotation.yaw = math.degrees(math.atan2(aim_vector.y, aim_vector.x)) \
            + self._glare_yaw_offset

        actor = CarlaDataProvider.request_new_actor(
            self._glare_model, spawn_transform, rolename='scenario no lights')
        if not actor:
            raise ValueError("Couldn't spawn the glare vehicle")

        # 'scenario no lights' keeps RouteLightsBehavior from managing this actor, so the high
        # beams stay on for the whole route instead of being stripped once the ego is far away
        actor.set_light_state(carla.VehicleLightState(self._glare_lights))
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=True))

        return actor

    def _initialize_actors(self, config):
        super()._initialize_actors(config)

        # Sit alongside the obstacle by default, so its halo overlaps the black car in the image
        # instead of the ego seeing the two of them side by side
        glare_distance = self._glare_distance
        if glare_distance is None:
            glare_distance = self._distance

        # Point at the ego's lane some way back down the route. Aiming all the way at the trigger
        # point would toe the vehicle in by only a few degrees; a shorter distance aims the beams
        # at where the ego actually is while it closes on the obstacle.
        aim_distance = self._glare_aim_distance
        if aim_distance is None:
            aim_distance = 15.0
        aim_distance = min(aim_distance, glare_distance)

        glare_wp = self._get_glare_waypoint(self._move_waypoint_forward(self._starting_wp, glare_distance))
        aim_wp = self._move_waypoint_forward(self._starting_wp, glare_distance - aim_distance)
        self.other_actors.append(self._spawn_glare_vehicle(glare_wp, aim_wp.transform.location))

    def _create_behavior(self):
        behavior = super()._create_behavior()
        behavior.name = "ParkedObstacleNoLightsWithGlare"
        return behavior


class PedestrianCrowd(ParkedObstacle):
    """
    Variation of ParkedObstacle where the thing blocking the ego's lane is a small group of
    pedestrians standing in the road rather than a parked vehicle: same geometry, same
    'drive up to it' end condition, a very different object to recognise.

    Like ParkedObstacleNoLights there is no roadside warning prop -- the crowd is
    unsignalled -- and the pedestrians are stationary: they are an obstacle to be seen and
    stopped for, not a crossing hazard (srunner.scenarios.pedestrian_crossing covers that).
    """

    # Turned to face the oncoming ego
    MEMBER_YAW_OFFSET = 180.0
    DEFAULT_MODEL = 'walker.pedestrian.*'
    # Roughly a shoulder-and-a-half apart: reads as a group in the camera frame while
    # still fitting inside a ~3.5 m lane at the default size of 3.
    DEFAULT_LATERAL_SPACING = 0.9
    DEFAULT_ROW_SPACING = 0.7

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        # Read before the parent's __init__, which triggers _initialize_actors
        self._crowd_size = int(get_value_parameter(config, 'crowd_size', float, 3))
        self._model = get_value_parameter(config, 'model', str, self.DEFAULT_MODEL)
        self._lateral_spacing = get_value_parameter(config, 'lateral_spacing', float, self.DEFAULT_LATERAL_SPACING)
        self._row_spacing = get_value_parameter(config, 'row_spacing', float, self.DEFAULT_ROW_SPACING)
        self._yaw_offset = get_value_parameter(config, 'yaw_offset', float, self.MEMBER_YAW_OFFSET)
        self._lane_offset = get_value_parameter(config, 'offset', float, 0.0)

        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _spawn_side_prop(self, wp):
        """No roadside warning sign: the crowd is unsignalled"""
        pass

    def _member_longitudinal(self, i):
        """
        Distance along the road of member `i` from the group's anchor point. Alternating
        so the row zig-zags rather than drifting to one end, which keeps the members
        behind from being fully occluded by the ones in front.
        """
        return self._row_spacing * (i % 2)

    def _crowd_transforms(self, wp):
        """
        Places the group across the ego's lane, centred on `offset`, all of them turned by
        the same `yaw_offset` from the road direction.
        """
        r_vec = wp.transform.get_right_vector()
        f_vec = wp.transform.get_forward_vector()

        transforms = []
        for i in range(self._crowd_size):
            # Centre the row on the lane centre: e.g. -1, 0, +1 for three members
            lateral = (i - (self._crowd_size - 1) / 2) * self._lateral_spacing
            lateral += self._lane_offset * wp.lane_width / 2
            longitudinal = self._member_longitudinal(i)

            location = wp.transform.location + carla.Location(
                x=lateral * r_vec.x + longitudinal * f_vec.x,
                y=lateral * r_vec.y + longitudinal * f_vec.y,
                z=1)
            rotation = carla.Rotation(yaw=wp.transform.rotation.yaw + self._yaw_offset)
            transforms.append(carla.Transform(location, rotation))

        return transforms

    def _spawn_member(self, spawn_transform):
        """Spawns one member of the group. Overridden for groups that are not walkers."""
        walker = CarlaDataProvider.request_new_actor(
            self._model, spawn_transform, rolename='scenario', actor_category='pedestrian')
        if not walker:
            raise ValueError("Couldn't spawn a pedestrian of the crowd")
        return walker

    def _initialize_actors(self, config):
        """
        Same waypoint layout as the parent -- the group stands where the parked car would --
        but spawning several actors instead of one, so the parent's light-state and
        hand-brake handling (which only exists on carla.Vehicle) does not apply here.
        """
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)
        self._spawn_side_prop(self._starting_wp)

        self._vehicle_wp = self._move_waypoint_forward(self._starting_wp, self._distance)

        self._members = []
        for spawn_transform in self._crowd_transforms(self._vehicle_wp):
            actor = self._spawn_member(spawn_transform)
            self._members.append(actor)
            self.other_actors.append(actor)

        self._end_wp = self._move_waypoint_forward(self._vehicle_wp, self._end_distance)

    def _encounter_actors(self):
        """Reaching or hitting any member of the group resolves the encounter"""
        return self._members

    def _create_behavior(self):
        behavior = super()._create_behavior()
        behavior.name = self.__class__.__name__
        return behavior


class ParkedCyclists(PedestrianCrowd):
    """
    Variation of PedestrianCrowd where the group blocking the ego's lane is a set of parked
    bicycles turned side-on, across the road rather than along it -- the silhouette an
    approaching car sees is the full length of the bike, not its narrow rear end.

    They are vehicles rather than walkers, so unlike the pedestrian version they get the
    hand brake and an explicitly cleared light state (bicycles carry no lights of their own,
    but the 'no lights' role name also keeps RouteLightsBehavior from adding any at night).
    """

    # Square across the road, so the ego sees the bike broadside
    MEMBER_YAW_OFFSET = 90.0
    DEFAULT_MODEL = 'vehicle.*'
    # Side-on a bicycle is ~1.6 m long, so three of them cannot sit abreast in a ~3.5 m
    # lane; they are lined up along the road instead (see _member_longitudinal) and the
    # lateral term only jitters them off the centre line.
    DEFAULT_LATERAL_SPACING = 0.5
    DEFAULT_ROW_SPACING = 2.0

    def _member_longitudinal(self, i):
        """A queue along the road rather than the parent's two-deep zig-zag"""
        return self._row_spacing * i

    def _spawn_member(self, spawn_transform):
        actor = CarlaDataProvider.request_new_actor(
            self._model, spawn_transform, rolename='scenario no lights',
            attribute_filter={'base_type': 'bicycle'})
        if not actor:
            raise ValueError("Couldn't spawn a bicycle of the group")

        actor.set_light_state(carla.VehicleLightState.NONE)
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=True))
        return actor


class HazardAtSideLane(BasicScenario):
    """
    Added the dangerous scene of ego vehicles driving on roads without sidewalks,
    with three bicycles encroaching on some roads in front.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """
        self._world = world
        self._map = CarlaDataProvider.get_map()
        self.timeout = timeout

        self._obstacle_distance = 9
        self._trigger_distance = 50
        self._end_distance = 50
        self._extra_space = 30

        self._offset = 0.55
        self._wait_duration = 5

        self._target_locs = []

        self._bicycle_bps = ["vehicle.bh.crossbike", "vehicle.diamondback.century", "vehicle.gazelle.omafiets"]

        self._distance = get_value_parameter(config, 'distance', float, 100)
        self._max_speed = get_value_parameter(config, 'speed', float, 60)
        self._bicycle_speed = get_value_parameter(config, 'bicycle_speed', float, 10)
        self._bicycle_drive_distance = get_value_parameter(config, 'bicycle_drive_distance', float, 50)
        self._scenario_timeout = 240

        super().__init__("HazardAtSideLane",
                         ego_vehicles,
                         config,
                         world,
                         randomize,
                         debug_mode,
                         criteria_enable=criteria_enable)

    def _move_waypoint_forward(self, wp, distance):
        dist = 0
        next_wp = wp
        while dist < distance:
            next_wps = next_wp.next(1)
            if not next_wps or next_wps[0].is_junction:
                break
            next_wp = next_wps[0]
            dist += 1
        return next_wp

    def _spawn_obstacle(self, wp, blueprint):
        """
        Spawns the obstacle actor by displacing its position to the right
        """
        displacement = self._offset * wp.lane_width / 2
        r_vec = wp.transform.get_right_vector()

        spawn_transform = wp.transform
        spawn_transform.location += carla.Location(x=displacement * r_vec.x, y=displacement * r_vec.y, z=1)
        actor = CarlaDataProvider.request_new_actor(blueprint, spawn_transform)
        if not actor:
            raise ValueError("Couldn't spawn an obstacle actor")

        return actor

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        rng = CarlaDataProvider.get_random_seed()
        self._starting_wp = self._map.get_waypoint(config.trigger_points[0].location)

        # Spawn the first bicycle
        first_wp = self._move_waypoint_forward(self._starting_wp, self._distance)
        bicycle_1 = self._spawn_obstacle(first_wp, rng.choice(self._bicycle_bps))

        wps = first_wp.next(self._bicycle_drive_distance)
        if not wps:
            raise ValueError("Couldn't find an end location for the bicycles")
        self._target_locs.append(wps[0].transform.location)

        # Set its initial conditions
        bicycle_1.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(bicycle_1)

        # Spawn the second bicycle
        second_wp = self._move_waypoint_forward(first_wp, self._obstacle_distance)
        bicycle_2 = self._spawn_obstacle(second_wp, rng.choice(self._bicycle_bps))

        wps = second_wp.next(self._bicycle_drive_distance)
        if not wps:
            raise ValueError("Couldn't find an end location for the bicycles")
        self._target_locs.append(wps[0].transform.location)

        # Set its initial conditions
        bicycle_2.apply_control(carla.VehicleControl(hand_brake=True))
        self.other_actors.append(bicycle_2)

    def _create_behavior(self):
        """
        Activate the bicycles and wait for the ego to be close-by before changing the side traffic.
        End condition is based on the ego behind in front of the bicycles, or timeout based.
        """
        root = py_trees.composites.Sequence(name="HazardAtSideLane")
        if self.route_mode:
            total_dist = self._distance + self._obstacle_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=self._extra_space))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        # End condition
        end_condition = py_trees.composites.Sequence(name="End Condition")
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], check_distance=False))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        main_behavior.add_child(end_condition)

        # Bicycle movement. Move them for a set distance, then stop
        offset = self._offset * self._starting_wp.lane_width / 2
        opt_dict = {'offset': offset}
        for actor, target_loc in zip(self.other_actors, self._target_locs):
            bicycle = py_trees.composites.Sequence(name="Bicycle behavior")
            bicycle.add_child(BasicAgentBehavior(actor, target_loc, target_speed=self._bicycle_speed, opt_dict=opt_dict))
            bicycle.add_child(HandBrakeVehicle(actor, 1))  # In case of collisions
            bicycle.add_child(WaitForever())  # Don't make the bicycle stop the parallel behavior
            main_behavior.add_child(bicycle)

        behavior = py_trees.composites.Sequence(name="Side lane behavior")
        behavior.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self.other_actors[0], self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SetMaxSpeed(self._max_speed))
        behavior.add_child(WaitForever())

        main_behavior.add_child(behavior)

        root.add_child(main_behavior)
        if self.route_mode:
            root.add_child(SetMaxSpeed(0))
            root.add_child(ChangeRoadBehavior(extra_space=0))

        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = [ScenarioTimeoutTest(self.ego_vehicles[0], self.config.name)]
        if not self.route_mode:
            criteria.append(CollisionTest(self.ego_vehicles[0]))
        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self.remove_all_actors()


class HazardAtSideLaneTwoWays(HazardAtSideLane):
    """
    Variation of the HazardAtSideLane scenario but the ego now has to invade the opposite lane
    """
    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=180):

        self._opposite_frequency = get_value_parameter(config, 'frequency', float, 100)

        super().__init__(world, ego_vehicles, config, randomize, debug_mode, criteria_enable, timeout)

    def _create_behavior(self):
        """
        Activate the bicycles and wait for the ego to be close-by before changing the opposite traffic.
        End condition is based on the ego behind in front of the bicycles, or timeout based.
        """

        root = py_trees.composites.Sequence(name="HazardAtSideLaneTwoWays")
        if self.route_mode:
            total_dist = self._distance + self._obstacle_distance + 20
            root.add_child(LeaveSpaceInFront(total_dist))
            root.add_child(ChangeRoadBehavior(extra_space=self._extra_space))

        main_behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        main_behavior.add_child(ScenarioTimeout(self._scenario_timeout, self.config.name))

        # End condition
        end_condition = py_trees.composites.Sequence(name="End Condition")
        end_condition.add_child(WaitUntilInFront(self.ego_vehicles[0], self.other_actors[-1], check_distance=False))
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self._end_distance))
        main_behavior.add_child(end_condition)

        # Bicycle movement. Move them for a set distance, then stop
        offset = self._offset * self._starting_wp.lane_width / 2
        opt_dict = {'offset': offset}
        for actor, target_loc in zip(self.other_actors, self._target_locs):
            bicycle = py_trees.composites.Sequence(name="Bicycle behavior")
            bicycle.add_child(BasicAgentBehavior(actor, target_loc, target_speed=self._bicycle_speed, opt_dict=opt_dict))
            bicycle.add_child(HandBrakeVehicle(actor, 1))  # In case of collisions
            bicycle.add_child(WaitForever())  # Don't make the bicycle stop the parallel behavior
            main_behavior.add_child(bicycle)

        behavior = py_trees.composites.Sequence(name="Side lane behavior")
        behavior.add_child(InTriggerDistanceToVehicle(
            self.ego_vehicles[0], self.other_actors[0], self._trigger_distance))
        behavior.add_child(Idle(self._wait_duration))
        if self.route_mode:
            behavior.add_child(SwitchWrongDirectionTest(False))
            behavior.add_child(ChangeOppositeBehavior(spawn_dist=self._opposite_frequency))
        behavior.add_child(WaitForever())

        main_behavior.add_child(behavior)

        root.add_child(main_behavior)
        if self.route_mode:
            root.add_child(SwitchWrongDirectionTest(False))
            root.add_child(ChangeOppositeBehavior(spawn_dist=40))
            root.add_child(ChangeRoadBehavior(extra_space=0))

        for actor in self.other_actors:
            root.add_child(ActorDestroy(actor))

        return root
