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
    def __init__(self,working_threshold = 60.0):
        super().__init__()
        self.state = State.INSIDE_DEPOT_AVAILABLE
        self.dr = np.array([0, 0]).astype('float64')
        self.id = -1
        self.takeoff_battery_level = 100.0
        self.best_bid = None
        self.my_bid = None

# ----->To be specified from config file
        self.working_threshold = working_threshold

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
        
        # print(self.state)
        # print(self.id,self.state)
        # print(self.id,self.state,api.get_battery_level())

        if self.state == State.ATTEMPTING_DELIVERY or self.state == State.RETURNING_FAILED or self.state == State.RETURNING_SUCCESSFUL:
            self.navigation_table.set_relative_position_for_location(Location.DELIVERY_LOCATION, api.get_relative_position_to_location(Location.DELIVERY_LOCATION))
        
        self.navigation_table.set_relative_position_for_location(Location.DEPOT_LOCATION, api.get_relative_position_to_location(Location.DEPOT_LOCATION))

        if self.state == State.INSIDE_DEPOT_CHARGING:

            if api.get_battery_level() >= self.working_threshold:
                self.state = State.INSIDE_DEPOT_AVAILABLE

        elif self.state == State.ATTEMPTING_DELIVERY:            
            if sensors[Location.DELIVERY_LOCATION]:
                api.deliver_package()
                self.state = State.RETURNING_SUCCESSFUL

            elif api.get_battery_level()  <= self.takeoff_battery_level/2.0:
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

                if api.get_battery_level() >= self.working_threshold:
                    self.state = State.INSIDE_DEPOT_AVAILABLE
                else:
                    self.state = State.INSIDE_DEPOT_CHARGING

        elif self.state == State.INSIDE_DEPOT_AVAILABLE:
            
            if api.get_battery_level() < self.working_threshold:
                self.state = State.INSIDE_DEPOT_CHARGING
            
            else:
                order = api.get_order()
                
                # print(">>>>>>>>>>>>><<<<<<<<<<<<<<<-")
                # print(order.location)
                # print(order.weight)
                # print(order.arrival_time)
                # print(order.fulfillment_time)
                # print(order.bid_start_time)

                if order != None and order.bid_start_time>=api.clock().tick:

                    api.get_order().bid_start_time = api.clock().tick
                    # ------------> communicate bid: id, current battery level, number of fails? maybe we should consider the order arrival time
                    self.my_bid = self.bidding_policy(api.get_battery_level(),order)
                    api.make_bid(self.my_bid)

                    self.state = State.INSIDE_DEPOT_MADE_BID

        elif self.state == State.INSIDE_DEPOT_MADE_BID:
# ------------> check other robots bids, if the robot has the highest number of failures it wins the bid, if tie with other robots, the robot with the smallest ID wins the bid
            # if self.number_of_failures >= max_peers_failures or (self.number_of_failures == peers_max_failures and self.id < peers_max_failures_id): # robots wins the bid
            
            # print(self.best_bid,self.my_bid == self.best_bid[0], self.id > self.best_bid[1])
            
            if self.my_bid > self.best_bid[0] or (self.my_bid == self.best_bid[0] and self.id > self.best_bid[1]): # robots wins the bid

                # print(self.id, "*********************** I won bid! *************************")
                api.pickup_package()

                self.navigation_table.replace_information_entry(Location.DELIVERY_LOCATION, Target(api.get_package_info().location))
                
                self.state = State.ATTEMPTING_DELIVERY

                self.takeoff_battery_level = api.get_battery_level()
            
            else:
                self.state = State.INSIDE_DEPOT_AVAILABLE

            self.my_bid = None
            api.make_bid(self.my_bid)

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
        pass

    def bidding_policy(self,current_battery_level,order):
        return current_battery_level

class DecentralisedLearningBehavior(NaiveBehavior):
    def __init__(self, working_threshold = 60.0 ):
        super(DecentralisedLearningBehavior, self).__init__(working_threshold)

    def bidding_policy(self,current_battery_level,order):
#--> TO BE DESIGNED
        return current_battery_level


    def learn(take_off_battery_level, package_location, package_weight, reward):
#--> TO BE DESIGNED
        pass


class CentralisedLearningBehavior(NaiveBehavior):
    def __init__(self, working_threshold = 60.0):
        super(CentralisedLearningBehavior, self).__init__(working_threshold)

    def bidding_policy(self,current_battery_level,order):
#--> TO BE DESIGNED
        return current_battery_level


    def learn(take_off_battery_level, package_location, package_weight, reward):
#--> TO BE DESIGNED
        pass
