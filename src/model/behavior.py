import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

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


def behavior_factory(behavior_params,order_params):
    # if behavior_params['class'] == "DecentralisedLearningBehavior":
    behavior = eval(behavior_params['class'])(**behavior_params['parameters'],**order_params['distances'],**order_params['weights'])
    # else:
        # behavior = eval(behavior_params['class'])(**behavior_params['parameters'])

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
    def __init__(self,working_threshold = 60.0, min_distance= 500,max_distance=8000, min_package_weight=0.5, max_package_weight= 5.0):
        super().__init__()
        self.state = State.INSIDE_DEPOT_AVAILABLE
        self.dr = np.array([0, 0]).astype('float64')
        self.id = -1
        self.takeoff_battery_level = 100.0
        self.best_bid = None
        self.my_bid = None
        self.max_difficulty = 1.0
        self.max_distance = max_distance
        self.max_weight = max_package_weight

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

                self.learn([api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level], 1.0)

                api.deliver_package()
                self.state = State.RETURNING_SUCCESSFUL

            elif api.get_battery_level()  <= self.takeoff_battery_level/2.0:
                self.state = State.RETURNING_FAILED

                self.learn([api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level], 0.0)

                current_difficulty = api.get_package_info().weight
                if api.get_package_info().weight/(self.takeoff_battery_level/100.0) < self.max_weight:
                    self.max_weight = api.get_package_info().weight/(self.takeoff_battery_level/100.0)
                    # print(f'new max weight = {self.max_weight}') # <-----------------------------
                if api.get_package_info().distance/(self.takeoff_battery_level/100.0) < self.max_distance:
                    self.max_distance = api.get_package_info().distance/(self.takeoff_battery_level/100.0)
                    # print(f'new max distance = {self.max_distance}') # <-----------------------------
        
        elif self.state == State.RETURNING_FAILED:
            
            if sensors[Location.DEPOT_LOCATION]:
                api.return_package()
                self.state = State.INSIDE_DEPOT_CHARGING

        elif self.state == State.RETURNING_SUCCESSFUL:

            if sensors[Location.DEPOT_LOCATION]:
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

                    # ------------> communicate bid: id, current battery level, number of fails? maybe we should consider the order arrival time
                    state = [api.get_order().distance,api.get_order().weight,api.get_battery_level()]

                    if self.bidding_policy(state):
                        api.get_order().bid_start_time = api.clock().tick

                        self.my_bid = self.formulate_bid(order,api.get_battery_level())
                        # print("bid", self.id,self.my_bid) # <-----------------------------
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

    def learn(self,state,outcome):
        pass

    def bidding_policy(self,state):
        return True

    def formulate_bid(self,order,battery_level):
        return battery_level

class DecentralisedLearningBehavior(NaiveBehavior):
    def __init__(self, working_threshold = 50.0,exploration_probability = 0.001, min_distance= 500,max_distance=8000, min_package_weight=0.5, max_package_weight= 5.0):
        super(DecentralisedLearningBehavior, self).__init__(working_threshold,min_distance,max_distance, min_package_weight, max_package_weight)
        
        self.epsilon = exploration_probability

        # Initialize the scaler and the online SVM classifier
        self.scaler = StandardScaler()
        
        # Create an SGD classifier with a hinge loss (SVM)
        self.sgd_clf = SGDClassifier(loss='hinge',warm_start=True)

        # Initialise decision model
        X = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,0.0]]
        y = [1.0,0.0]
        X_scaled = self.scaler.fit_transform(X)
        self.sgd_clf.partial_fit(X_scaled, y, classes=np.unique(y))

    def bidding_policy(self,state):
        state_scaled = self.scaler.transform([state])
        predicted_outcome = self.sgd_clf.predict(state_scaled)

        # print("predict",self.id,state,predicted_outcome) # <-----------------------------
        if predicted_outcome == 1.0:
            return True
        else:
            if np.random.uniform() <= self.epsilon:
                # print(self.id,"exploring") # <-----------------------------
                return True
            else:
                return False

    def formulate_bid(self,order,battery_level):
        return (0.5*order.distance/self.max_distance + 0.5*order.weight/self.max_weight)/(battery_level/100.0)

    # def learn(take_off_battery_level, package_location, package_weight, reward):
    def learn(self,state, outcome):
        # print("learn",self.id,state,outcome) # <-----------------------------
        state_scaled = self.scaler.transform([state])
        self.sgd_clf.partial_fit(state_scaled, [outcome])


class CentralisedLearningBehavior(NaiveBehavior):
    def __init__(self, working_threshold = 60.0):
        super(CentralisedLearningBehavior, self).__init__(working_threshold)

    def bidding_policy(self,current_battery_level,order):
#--> TO BE DESIGNED
        return current_battery_level


    def learn(take_off_battery_level, package_location, package_weight, reward):
#--> TO BE DESIGNED
        pass
