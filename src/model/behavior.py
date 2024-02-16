import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin

import numpy as np

from model.communication import CommunicationSession
from model.navigation import Location, NavigationTable, Order, Target
from helpers.utils import get_orientation_from_vector, norm
from random import random

class State(Enum):
    INSIDE_DEPOT_AVAILABLE = 1
    INSIDE_DEPOT_CHARGING = 2
    INSIDE_DEPOT_MADE_BID = 3
    ATTEMPTING_DELIVERY = 4
    RETURNING_SUCCESSFUL = 5
    RETURNING_FAILED = 6


def behavior_factory(behavior_params):
    behavior = eval(behavior_params['class'])(**behavior_params['parameters'])

    return behavior


class Behavior(ABC):

    def __init__(self):
        self.color = "blue"
        self.navigation_table = NavigationTable()

    @abstractmethod
    def step(self, api):
        """Simulates 1 step of behavior (= 1 movement)"""
    
    @abstractmethod
    def check_others_bid(self, neighbors):
        pass

    def debug_text(self):
        return ""


class NaiveBehavior(Behavior):
    def __init__(self,high_threshold = 60.0, low_threshold = 20.0, safety_threshold = 5.0):
        super().__init__()
        self.state = State.INSIDE_DEPOT_AVAILABLE
        self.dr = np.array([0, 0]).astype('float64')
        self.id = -1
        self.takeoff_battery_level = 100.0
        self.best_bid = None

# ----->To be specified from config file
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.safety_threshold = safety_threshold

    def step(self, api):
        # self.dr[0], self.dr[1] = 0, 0
        self.id = api.get_id()
        sensors = api.get_sensors()
        self.update_state(sensors, api)
        self.update_movement_based_on_state(api)
        self.check_movement_with_sensors(sensors)
        self.update_nav_table_based_on_dr()

    def check_others_bid(self, session: CommunicationSession):
        self.best_bid = session.get_best_bid()


    def update_state(self, sensors, api):
        
        # print(self.id,self.state)
        # print(self.id,self.state,api.get_battery_level())

        if self.state == State.ATTEMPTING_DELIVERY or self.state == State.RETURNING_FAILED or self.state == State.RETURNING_SUCCESSFUL:
            self.navigation_table.set_relative_position_for_location(Location.DELIVERY_LOCATION, api.get_relative_position_to_location(Location.DELIVERY_LOCATION))
        
        self.navigation_table.set_relative_position_for_location(Location.DEPOT_LOCATION, api.get_relative_position_to_location(Location.DEPOT_LOCATION))

        if self.state == State.INSIDE_DEPOT_CHARGING:

            if api.get_battery_level() >= self.high_threshold:
                self.state = State.INSIDE_DEPOT_AVAILABLE

        elif self.state == State.ATTEMPTING_DELIVERY:            
            if sensors[Location.DELIVERY_LOCATION]:
                api.deliver_package()
                self.state = State.RETURNING_SUCCESSFUL

            elif (api.get_battery_level()-self.safety_threshold) <= self.takeoff_battery_level/2.0:
                self.state = State.RETURNING_FAILED
        
        elif self.state == State.RETURNING_FAILED:
            
            if sensors[Location.DEPOT_LOCATION]:

                self.learn(self.takeoff_battery_level,api.get_package_info(), -1)

                api.return_package()

                self.state = State.INSIDE_DEPOT_CHARGING

        elif self.state == State.RETURNING_SUCCESSFUL:

            if sensors[Location.DEPOT_LOCATION]:
                self.learn(self.takeoff_battery_level,api.get_package_info(), 1)
# ------------->Forget attempted delivery

                if api.get_battery_level() >= self.high_threshold:
                    self.state = State.INSIDE_DEPOT_AVAILABLE
                else:
                    self.state = State.INSIDE_DEPOT_CHARGING

        elif self.state == State.INSIDE_DEPOT_AVAILABLE:
            
            if api.get_battery_level() <= self.low_threshold:
                self.state = State.INSIDE_DEPOT_CHARGING
            
            else:
                order = api.get_order()
                if order != None and order.bid_start_time>=api.clock().tick:
# ------------> communicate bid: id, current battery level, number of fails? maybe we should consider the order arrival time
                    if random() <= self.bidding_policy(api.get_battery_level(), api.get_order()): 
                        api.get_order().bid_start_time = api.clock().tick
                        self.state = State.INSIDE_DEPOT_MADE_BID

        elif self.state == State.INSIDE_DEPOT_MADE_BID:
# ------------> check other robots bids, if the robot has the highest number of failures it wins the bid, if tie with other robots, the robot with the smallest ID wins the bid
            # if self.number_of_failures >= max_peers_failures or (self.number_of_failures == peers_max_failures and self.id < peers_max_failures_id): # robots wins the bid
            if self.id > self.best_bid: # robots wins the bid
                api.pickup_package()

                self.navigation_table.replace_information_entry(Location.DELIVERY_LOCATION, Target(api.get_package_info().location))
                
                self.state = State.ATTEMPTING_DELIVERY

                self.takeoff_battery_level = api.get_battery_level()
            
            else:
                self.state = State.INSIDE_DEPOT_AVAILABLE

        api.update_state(self.state)

    def update_movement_based_on_state(self, api):
        if self.state == State.ATTEMPTING_DELIVERY:
            self.dr = self.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION)
            package_norm = norm(self.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION))
            if package_norm > api.speed():
                self.dr = self.dr * api.speed() / package_norm

        elif self.state == State.RETURNING_SUCCESSFUL or self.state == State.RETURNING_FAILED:
            self.dr = self.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION)
            depot_norm = norm(self.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION))
            if depot_norm > api.speed():
                self.dr = self.dr * api.speed() / depot_norm
        else:
            self.dr = np.array([0, 0]).astype('float64')
        # else:
        #     turn_angle = api.get_levi_turn_angle()
        #     self.dr = api.speed() * np.array([cos(radians(turn_angle)), sin(radians(turn_angle))])

        api.set_desired_movement(self.dr)

    def check_movement_with_sensors(self, sensors):
        if (sensors["FRONT"] and self.dr[0] >= 0) or (sensors["BACK"] and self.dr[0] <= 0):
            self.dr[0] = -self.dr[0]
        if (sensors["RIGHT"] and self.dr[1] <= 0) or (sensors["LEFT"] and self.dr[1] >= 0):
            self.dr[1] = -self.dr[1]

    def update_nav_table_based_on_dr(self):
        self.navigation_table.update_from_movement(self.dr)
        self.navigation_table.rotate_from_angle(-get_orientation_from_vector(self.dr))

    def learn(take_off_battery_level, package_location, package_weight, reward):
#--> TO BE DESIGNED
        pass

    def bidding_policy(self,current_battery_level,order):
#--> TO BE DESIGNED
        return 1.0

# class DecentralisedLearningBehavior(NaiveBehavior):
#     def __init__(self, security_level=3):
#         super(CarefulBehavior, self).__init__()
#         self.color = "deep sky blue"
#         self.security_level = security_level
#         self.pending_information = {location: {} for location in Location}

#     def buy_info(self, session: CommunicationSession):
#         for location in Location:
#             metadata = session.get_metadata(location)
#             metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
#             for bot_id, data in metadata_sorted_by_age:
#                 if data["age"] < self.navigation_table.get_age_for_location(location) and bot_id not in \
#                         self.pending_information[
#                             location]:
#                     try:
#                         other_target = session.make_transaction(neighbor_id=bot_id, location=location)
#                         other_target.set_distance(other_target.get_distance() + session.get_distance_from(bot_id))
#                         if not self.navigation_table.is_information_valid_for_location(location):
#                             self.navigation_table.replace_information_entry(location, other_target)
#                         else:
#                             self.pending_information[location][bot_id] = other_target
#                             if len(self.pending_information[location]) >= self.security_level:
#                                 self.combine_pending_information(location)
#                     except InsufficientFundsException:
#                         pass
#                     except NoInformationSoldException:
#                         pass

#     def combine_pending_information(self, location):
#         distances = [t.get_distance() for t in self.pending_information[location].values()]
#         mean_distance = np.mean(distances, axis=0)
#         best_target = min(self.pending_information[location].values(),
#                           key=lambda t: norm(t.get_distance() - mean_distance))
#         self.navigation_table.replace_information_entry(location, best_target)
#         self.pending_information[location].clear()

#     def step(self, api):
#         super().step(api)
#         self.update_pending_information()

#     def update_pending_information(self):
#         for location in Location:
#             for target in self.pending_information[location].values():
#                 target.update(self.dr)
#                 target.rotate(-get_orientation_from_vector(self.dr))

#     def debug_text(self):
#         return f"size of pending: {[len(self.pending_information[l]) for l in Location]}\n" \
#                f"{self.pending_information[Location.DELIVERY_LOCATION]}\n" \
#                f"{self.pending_information[Location.DEPOT]}"


# class CentralisedLearningBehavior(NaiveBehavior):
#     def __init__(self, threshold=0.25):
#         super(ScepticalBehavior, self).__init__()
#         self.pending_information = {location: {} for location in Location}
#         self.threshold = threshold

#     def buy_info(self, session: CommunicationSession):
#         for location in Location:
#             metadata = session.get_metadata(location)
#             metadata_sorted_by_age = sorted(metadata.items(), key=lambda item: item[1]["age"])
#             for bot_id, data in metadata_sorted_by_age:
#                 if data["age"] < self.navigation_table.get_age_for_location(location) and bot_id not in \
#                         self.pending_information[
#                             location]:
#                     try:
#                         other_target = session.make_transaction(neighbor_id=bot_id, location=location)
#                         other_target.set_distance(other_target.get_distance() + session.get_distance_from(
#                             bot_id))

#                         if not self.navigation_table.is_information_valid_for_location(location) or \
#                                 self.difference_score(
#                                     self.navigation_table.get_relative_position_for_location(location),
#                                     other_target.get_distance()) < self.threshold:
#                             new_target = self.strategy.combine(self.navigation_table.get_information_entry(location),
#                                                                other_target,
#                                                                np.array([0, 0]))
#                             self.navigation_table.replace_information_entry(location, new_target)
#                             self.pending_information[location].clear()
#                         else:
#                             for target in self.pending_information[location].values():
#                                 if self.difference_score(target.get_distance(),
#                                                          other_target.get_distance()) < self.threshold:
#                                     new_target = self.strategy.combine(target,
#                                                                        other_target,
#                                                                        np.array([0, 0]))
#                                     self.navigation_table.replace_information_entry(location, new_target)
#                                     self.pending_information[location].clear()
#                                     break
#                             else:
#                                 self.pending_information[location][bot_id] = other_target
#                     except InsufficientFundsException:
#                         pass
#                     except NoInformationSoldException:
#                         pass

#     @staticmethod
#     def difference_score(current_vector, bought_vector):
#         v_norm = norm(current_vector)
#         score = norm(current_vector - bought_vector) / v_norm if v_norm > 0 else 1000
#         return score

#     def step(self, api):
#         super().step(api)
#         self.update_pending_information()

#     def update_pending_information(self):
#         for location in Location:
#             for target in self.pending_information[location].values():
#                 target.update(self.dr)
#                 target.rotate(-get_orientation_from_vector(self.dr))


# class SaboteurBehavior(NaiveBehavior):
#     def __init__(self, rotation_angle=90):
#         super().__init__()
#         self.color = "red"
#         self.rotation_angle = rotation_angle

#     def sell_info(self, location):
#         t = copy.deepcopy(self.navigation_table.get_information_entry(location))
#         t.rotate(self.rotation_angle)
#         return t


# class GreedyBehavior(NaiveBehavior):
#     def __init__(self):
#         super().__init__()
#         self.color = "green"

#     def sell_info(self, location):
#         t = copy.deepcopy(self.navigation_table.get_information_entry(location))
#         t.age = 1
#         return t


# class FreeRiderBehavior(ScepticalBehavior):
#     def __init__(self):
#         super().__init__()
#         self.color = "pink"

#     def sell_info(self, location):
#         return None


# class ScaboteurBehavior(ScepticalBehavior):
#     def __init__(self, rotation_angle=90, threshold=0.25):
#         super().__init__()
#         self.color = "red"
#         self.rotation_angle = rotation_angle
#         self.threshold = threshold

#     def sell_info(self, location):
#         t = copy.deepcopy(self.navigation_table.get_information_entry(location))
#         t.rotate(self.rotation_angle)
#         return t


# class ScepticalGreedyBehavior(ScepticalBehavior):
#     def __init__(self):
#         super().__init__()
#         self.color = "green"

#     def sell_info(self, location):
#         t = copy.deepcopy(self.navigation_table.get_information_entry(location))
#         t.age = 1
#         return t


# def update_behavior(self, sensors, api):
    #     for location in Location:
    #         if sensors[location]:
    #             try:
    #                 self.navigation_table.set_relative_position_for_location(location,
    #                                                                          api.get_relative_position_to_location(
    #                                                                              location))
    #                 self.navigation_table.set_information_valid_for_location(location, True)
    #                 self.navigation_table.set_age_for_location(location, 0)
    #             except NoLocationSensedException:
    #                 print(f"Sensors do not sense {location}")

    #     if self.state == State.EXPLORING:
    #         if self.navigation_table.is_information_valid_for_location(Location.DELIVERY_LOCATION) and not api.carries_package():
    #             self.state = State.SEEKING_package
    #         if self.navigation_table.is_information_valid_for_location(Location.DEPOT) and api.carries_package():
    #             self.state = State.SEEKING_NEST

    #     elif self.state == State.SEEKING_package:
    #         if api.carries_package():
    #             if self.navigation_table.is_information_valid_for_location(Location.DEPOT):
    #                 self.state = State.SEEKING_NEST
    #             else:
    #                 self.state = State.EXPLORING
    #         elif norm(self.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION)) < api.radius():
    #             self.navigation_table.set_information_valid_for_location(Location.DELIVERY_LOCATION, False)
    #             self.state = State.EXPLORING

    #     elif self.state == State.SEEKING_NEST:
    #         if not api.carries_package():
    #             if self.navigation_table.is_information_valid_for_location(Location.DELIVERY_LOCATION):
    #                 self.state = State.SEEKING_package
    #             else:
    #                 self.state = State.EXPLORING
    #         elif norm(self.navigation_table.get_relative_position_for_location(Location.DEPOT)) < api.radius():
    #             self.navigation_table.set_information_valid_for_location(Location.DEPOT, False)
    #             self.state = State.EXPLORING

    #     if sensors["FRONT"]:
    #         if self.state == State.SEEKING_NEST:
    #             self.navigation_table.set_information_valid_for_location(Location.DEPOT, False)
    #             self.state = State.EXPLORING
    #         elif self.state == State.SEEKING_package:
    #             self.navigation_table.set_information_valid_for_location(Location.DELIVERY_LOCATION, False)
    #             self.state = State.EXPLORING
