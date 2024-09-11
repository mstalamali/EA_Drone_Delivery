from enum import Enum
from helpers.utils import rotation_matrix
import numpy as np
from random import random, uniform

class Location(Enum):
    DELIVERY_LOCATION = 1
    DEPOT_LOCATION = 2

class Order:
    def __init__(self, env_width, env_height, depot, order_id, arrival_time ,order_params):
        # Set order's location
        # while True:
        #     self.location = [ random() * env_width , random() * env_height] 
        #     dist_vector = self.location - np.array([depot[0], depot[1]])
        #     self.distance = np.sqrt(dist_vector.dot(dist_vector))
        #     if self.distance >= order_params["distances"]["min_distance"] and self.distance <= order_params["distances"]["max_distance"]:
        #         break

        self.distance = np.sqrt(uniform(order_params["distances"]["min_distance"]*order_params["distances"]["min_distance"],order_params["distances"]["max_distance"]*order_params["distances"]["max_distance"]))

        theta = uniform(0, 2 * np.pi)

        self.location = [ depot[0] + self.distance * np.cos(theta) , depot[1] + self.distance * np.sin(theta) ] 

        self.weight = uniform(order_params["weights"]["min_package_weight"], order_params["weights"]["max_package_weight"])

        self.radius = order_params["radius"]

        self.id =  order_id
        
        self.arrival_time = arrival_time

        self.fulfillment_time = float('inf')

        self.bid_start_time = float('inf')

        self.in_look_ahead = float('inf')

        self.attempted = 0

        # print(self.location,self.weight)

class Target:
    def __init__(self, location):
        self.location = location
        self.relative_distance = np.array([0, 0]).astype('float64')
        self.age = 0
        self.valid = False

    def is_valid(self):
        return self.valid

    def get_age(self):
        return self.age

    def set_age(self, age):
        self.age = age

    def set_valid(self, valid):
        self.valid = valid

    def get_distance(self):
        return self.relative_distance

    def set_distance(self, distance):
        self.relative_distance = distance

    def update(self, dr):
        self.age += 1
        self.relative_distance -= dr

    def rotate(self, angle):
        rot_mat = rotation_matrix(angle)
        self.relative_distance = rot_mat.dot(self.relative_distance)


class NavigationTable:
    def __init__(self):
        self.entries = dict()
        for location in Location:
            self.entries[location] = Target(location)

    def get_relative_position_for_location(self, location):
        return self.entries[location].get_distance()

    def set_relative_position_for_location(self, location, distance):
        self.entries[location].set_distance(distance)

    def update_from_movement(self, dr):
        for location in self.entries:
            self.entries[location].update(dr)

    def rotate_from_angle(self, angle):
        for location in self.entries:
            self.entries[location].rotate(angle)

    def get_age_for_location(self, location):
        return self.entries[location].age

    def set_age_for_location(self, location, age):
        self.entries[location].age = age

    def get_information_entry(self, location):
        return self.entries[location]

    def replace_information_entry(self, location, new_target):
        self.entries[location] = new_target
