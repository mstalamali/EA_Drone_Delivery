import copy

from helpers import random_walk as rw
from random import random, choices, gauss, uniform
from math import sin, cos, radians, pow
from collections import deque

from model.behavior import State, behavior_factory
from model.communication import CommunicationSession
from model.navigation import Location, NavigationTable
import numpy as np

from helpers.utils import get_orientation_from_vector, rotate, CommunicationState, norm

try:
    from tkinter import LAST
except ModuleNotFoundError:
    print("Tkinter not installed...")

# class implementing robot API (used by the robot controller to access information)
class AgentAPI:
    def __init__(self, agent):
        self.charging = agent.charging
        self.speed = agent.speed
        self.clock = agent.clock
        self.get_sensors = agent.get_sensors
        self.set_desired_movement = agent.set_desired_movement
        self.get_id = agent.get_id
        self.radius = agent.radius
        self.get_relative_position_to_location = agent.get_relative_position_to_location
        self.get_battery_level = agent.get_battery_level
        self.get_order = agent.get_order

        self.update_state = agent.update_state
        self.make_bid=agent.set_bid

        # Delivery related functions
        self.carries_package = agent.carries_package
        self.clear_next_order = agent.clear_next_order
        self.pickup_package = agent.pickup_package
        self.deliver_package = agent.deliver_package
        self.return_package = agent.return_package
        self.get_package_info = agent.get_package_info

        # Logging function
        self.log_data = agent.log_data

        
# Robot class
class Agent:
    colors = {State.EVALUATING: "orange", State.WAITING: "red", State.DECIDING: "green",\
                State.ATTEMPTING: "cyan", State.RETURNING: "magenta"}

    def __init__(self, robot_id, x, y, environment, log_params, behavior_params,order_params, clock, speed, radius, frame_weight, battery_weight,
                 theoritical_battery_capacity, min_battery_health, max_battery_health, noise_sampling_mu, noise_sampling_sigma, noise_sd, fuel_cost,
                 communication_radius):
        
        self._clock = clock


        self.environment = environment

        # Robot variables
        self.id = robot_id
        

        self.pos = np.array([x, y]).astype('float64')
        self._speed = speed
        self._radius = radius
        self.frame_weight = frame_weight
        self.battery_weight = battery_weight
        self.dr = np.array([0, 0]) # direction
        self.sensors = {} # sensor values


        self.color = self.colors[State.WAITING] # robot color used for debugging

        # --> Robot trajectory (only used for visualisation)
        self.trace = deque(self.pos, maxlen=100)
        
        # --> Robot's state variables
        self._in_depot = True
        self._bid = None
        self._charging = False

        # Set battery-related variables
        # --> Initial Values
        self.theoritical_battery_capacity = theoritical_battery_capacity

        if self._clock.tick == 0:
            self.battery_health = uniform(min_battery_health,max_battery_health)
        else:
            data_folder = log_params[2] 
            filename="robots_log_"+str(self._clock.tick)+"_"+log_params[3]
            
            try:
                import csv
                csvfile=open(data_folder+"/"+filename)
                reader = csv.DictReader(csvfile,delimiter='\t')
                data=list(reader)
                self.battery_health = float(data[self.id]['SoH'])
                self.w=[float(data[self.id]['w0']),float(data[self.id]['w1']),float(data[self.id]['w2'])]
                self.b=[float(data[self.id]['b'])]
                behavior_params['parameters']['w']=self.w
                behavior_params['parameters']['b']=self.b
                # print(behavior_params)
                # print(self.id,self.battery_health,self.w,self.b)
            except FileNotFoundError:
                print("Initialisation file not found!")
                raise

        self.actual_battery_capacity = self.theoritical_battery_capacity*self.battery_health

        # --> Time varying values 
        self.current_battery_capacity = self.actual_battery_capacity
        self._battery_level = 100.0

        # Locations of interest
        self.locations = {Location.DELIVERY_LOCATION: tuple(), Location.DEPOT_LOCATION: environment.depot}

        # Delivery variables
        self.items_delivered = 0
        self.failed_deliveries = 0
        self._carries_package = False
        self.pending_orders_list = environment.pending_orders_list
        self.successful_orders_list = environment.successful_orders_list
        self.failed_orders_list = environment.failed_orders_list
        self.charge_level_logging = environment.charge_level_logging
        self.attempted_delivery = None
        self.next_order = None

        # Communication variables
        self.communication_radius = communication_radius
        self.comm_state = CommunicationState.CLOSED


        # Currently not used
        self.orientation = random() * 360  # 360 degree angle
        self.noise_mu = gauss(noise_sampling_mu, noise_sampling_sigma)
        if random() >= 0.5:
            self.noise_mu = -self.noise_mu
        self.noise_sd = noise_sd
        self.fuel_cost = fuel_cost
        self.levi_counter = 1

        # --> robot's controller variables
        self.behavior = behavior_factory(behavior_params,order_params) # robot controller
        self.api = AgentAPI(self) # robot API

        # Simulation constants
        # --> UAV energy model constants
        self.g = 9.81 # Gravity constant (kg/s^2)
        self.n_r = 8  # Number of rotors
        self.rho = 1.2250 # Air density at 15 deg (kg/m^3)
        self.zeta = 0.27 # Area of the spinning blade disc of one rotor (m^2)
        #--> Battery charger constants
        self.charger_power = 100 # (W)
        self.charge_efficiency = 0.95 # (%)

        self.new_nav = self.behavior.navigation_table

        # robot data logging variables
        self.data_logging = log_params[0]
        self.charge_logging = log_params[1]
        self.log_folder = log_params[2]
        self.log_filename_suffix = log_params[3]
        if self.data_logging:
            self.logfile = open(f"{log_params[1]}/robot{str(self.id)}_{self.log_filename_suffix}","w")

        self.state = State.WAITING
    
    # function for debugging during visualisation (by clicking on the robot these information are shown on the right side of the simulation)
    def __str__(self):
        return f"ID: {self.id}\n" \
         f"state: {self.behavior.state}\n" \
         f"communication state: {self.comm_state.name}\n" \
         f"delivery location: ({round(self.pos[0] + rotate(self.behavior.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION), self.orientation)[0])}, {round(self.pos[1] + rotate(self.behavior.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION), self.orientation)[1])}), \n" \
         f"depot at: ({round(self.pos[0] + rotate(self.behavior.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION), self.orientation)[0])}, {round(self.pos[1] + rotate(self.behavior.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION), self.orientation)[1])}), \n" \
         f"carries package: {self._carries_package}\n" \
         f"battery health: {round(self.battery_health, 5)}\n" \
         f"item delivered: {self.items_delivered}\n" \
         f"dr: {np.round(self.dr, 2)}\n" \
         f"{self.behavior.debug_text()}"


    # functions for comparing objects (not used at the moment)
    def __repr__(self):
        return f"bot {self.id}"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    # function that returns the simulation clock object 
    def clock(self):
        return self._clock

    # function that runs the robot's simulation step
    def step(self):
        # print("before", self.clock().tick,self.id,self.behavior.state, self.comm_state, self._bid)
        self.behavior.navigation_table = self.new_nav
        self.sensors = self.environment.get_sensors(self)
        self.behavior.step(AgentAPI(self))
        self.move()
        self.update_battery_state()
        self.update_trace()
        # print("after", self.clock().tick,self.id,self.behavior.state, self.comm_state, self._bid)

    # function that upcate robot communication
    def communicate(self, neighbors):
        self.previous_nav = NavigationTable(self.behavior.navigation_table.entries)

        if self.comm_state == CommunicationState.OPEN:
            session = CommunicationSession(self, neighbors)
            self.behavior.check_others_bid(session)

        self.new_nav = self.behavior.navigation_table
        self.behavior.navigation_table = self.previous_nav

    # def get_info_from_behavior(self, location):
    #     return self.behavior.sell_info(location)

    # function to record robot trajectory
    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    # function that compute the relative position to a robot to a location (distance and orientation)
    def get_relative_position_to_location(self, location: Location):
        # if self.environment.get_sensors(self)[location]:
        return rotate((self.locations[location][0],self.locations[location][1]) - self.pos, -self.orientation)

    # function for moving the robot (updating it's position)
    def move(self):
        wanted_movement = rotate(self.dr, self.orientation) # self.dr is updated earlier based on speed
        self.orientation = get_orientation_from_vector(wanted_movement)
        self.pos = self.pos + wanted_movement

    # function to get robot speed
    def speed(self):
        return self._speed

    # function to get robot radius
    def radius(self):
        return self._radius

    # function to get robot id
    def get_id(self):
        return self.id

    # function to get output from robot sensors
    def get_sensors(self):
        return self.sensors

    # function to get the desired robot motion
    def set_desired_movement(self, dr):
        norm_dr = norm(dr)
        if norm_dr > self._speed:
            dr = self._speed * dr / norm_dr
        self.dr = dr

    # function for setting robot bid
    def set_bid(self, bid):
        self._bid = bid

# ------> Charging related functions
    # function to update robot state of charge
    def update_battery_state(self):

        if self.charging():

            self.current_battery_capacity += (1.0-self.current_battery_capacity/self.actual_battery_capacity) * self.battery_health * self.charger_power * self.charge_efficiency /3600.0

            if self.current_battery_capacity>self.actual_battery_capacity:
                self.current_battery_capacity=self.actual_battery_capacity

            self._battery_level = np.round(self.current_battery_capacity/self.actual_battery_capacity*100.0,1)

        else:

            total_weight = self.frame_weight + self.battery_weight
            if self.carries_package():
                total_weight += self.attempted_delivery.weight

            self.current_battery_capacity-= pow(self.g*total_weight,1.5)/pow(2*self.n_r*self.rho*self.zeta,0.5)/3600
            self._battery_level = np.round(self.current_battery_capacity/self.actual_battery_capacity*100.0,1)

    # function to get robot's state of charge
    def get_battery_level(self):
        return self._battery_level

# ------> State related functions
    # function to check if the robot is charging
    def charging(self):
        return self._charging

    # function to check if the robot is in the depot (not performing a delivery)
    def in_depot(self):
        return self._in_depot

    # function to update robot state (charging state, in depot state, and bidding state)
    def update_state(self,state):

        self.state = state

        if state == State.EVALUATING:
        # if state == State.WAITING or state == State.EVALUATING:
            self.comm_state = CommunicationState.OPEN
        else:
            self.comm_state = CommunicationState.CLOSED

        if state == State.WAITING or state==State.DECIDING or state == State.EVALUATING:
            self._in_depot = True
            self._charging = True
            self.pos=[self.environment.depot[0],self.environment.depot[1]]
        else:
            self._in_depot = False
            self._charging = False

        if state == State.EVALUATING:
            self._made_bid = True
        else:
            self._made_bid = False

        self.color = self.colors[state]

    # function to check if the robot has made a bid
    def made_bid(self):
        return self._made_bid

# ------> Delivery related functions
    # function for the robot to receive (hear) the next order
    def receive_next_order(self,next_order):
        self.next_order = next_order
    
    # function used by the robot to access the current advertised order (if it is none then there is no currently advertised order)
    def get_order(self):
        return self.next_order

    # function used by the robot to forget a previously advertised order
    def clear_next_order(self):
        self.next_order = None

    # function executed by the robot to deliver the package when arriving at delivery location, it unload the 
    def deliver_package(self):
        # print("--------->",self.clock().tick, self.id,"delivered order",self.attempted_delivery.id)
        self._carries_package = False
        self.attempted_delivery.fulfillment_time = self.clock().tick
        self.successful_orders_list.append(self.attempted_delivery)
        self.attempted_delivery = None
        self.items_delivered += 1
        self.environment.number_of_successes += 1
        self.environment.ongoing_attempts -= 1

    # function used by the robot to return package to the queue after a failed attempt (at arrival to the FC)
    def return_package(self):
        self._carries_package = False
        self.attempted_delivery.bid_start_time = float('inf')
        self.attempted_delivery.attempted+=1
        self.pending_orders_list[self.attempted_delivery.id-1]=self.attempted_delivery
        self.attempted_delivery = None
        self.failed_deliveries+=1
        self.environment.failed_delivery_attempts+=1
        self.environment.ongoing_attempts-=1
        self.log_charge("return")

    # function used by the robot to pick up package after winnin the bid
    def pickup_package(self):
        self.attempted_delivery = self.next_order
        self.pending_orders_list[self.environment.current_order] = None
        if self.environment.evaluation_type == "episodes":
            if self.attempted_delivery.arrival_time == float('inf'):
                self.attempted_delivery.arrival_time = self.clock().tick
        self._carries_package = True
        self.locations[Location.DELIVERY_LOCATION] = (self.attempted_delivery.location[0],self.attempted_delivery.location[1],int(self.attempted_delivery.radius))
        self.environment.ongoing_attempts+=1
        self.log_charge("takeoff")

    # function used by the robot to access information (distance and weight) and the package it is carying
    def get_package_info(self):
        return self.attempted_delivery

    # function used to check if the robot carries a package (useful for updating energy model and for visualisation)
    def carries_package(self):
        return self._carries_package

# ------> Drawing functions
    # function to draw robot (without battery level)
    def draw(self, canvas, robot_image):

        if not self.in_depot():

            if robot_image == None:
                circle = canvas.create_oval(self.pos[0] - self._radius,
                                            self.pos[1] - self._radius,
                                            self.pos[0] + self._radius,
                                            self.pos[1] + self._radius,
                                            fill=self.behavior.color,
                                            outline=self.color,
                                            width=3)
            else:
                canvas.create_image(self.pos[0], self.pos[1], image=robot_image, anchor='center')

            self.draw_goal_vector(canvas)
            self.draw_orientation(canvas)

    # function to draw robot (with battery level)
    def draw_advanced(self, canvas, robot_image_empty, robot_image_loaded,pixel_to_m):

        if not self.in_depot():

            if robot_image_empty == None:
                circle = canvas.create_oval(self.pos[0]/pixel_to_m - self._radius/pixel_to_m,
                                            self.pos[1]/pixel_to_m - self._radius/pixel_to_m,
                                            self.pos[0]/pixel_to_m + self._radius/pixel_to_m,
                                            self.pos[1]/pixel_to_m + self._radius/pixel_to_m,
                                            fill=self.behavior.color,
                                            outline=self.color,
                                            width=3)
            else:

                if self._carries_package:
                    canvas.create_image(self.pos[0]/pixel_to_m, self.pos[1]/pixel_to_m, image=robot_image_loaded, anchor='center')
                else:
                    canvas.create_image(self.pos[0]/pixel_to_m, self.pos[1]/pixel_to_m, image=robot_image_empty, anchor='center')

            self.draw_goal_vector(canvas, pixel_to_m)
            self.draw_orientation(canvas, pixel_to_m)
            self.write_battery_level(canvas,pixel_to_m)

    # function to draw robot trajectory 
    def draw_trace(self, canvas):
        tail = canvas.create_line(*self.trace)

    # function to draw robot destination (planned path)
    def draw_goal_vector(self, canvas, pixel_to_m):

        if self.state == State.ATTEMPTING:
            Goal = Location.DELIVERY_LOCATION
            color = "darkgreen"
        else:
            Goal = Location.DEPOT_LOCATION
            if self._carries_package == True:
                color = "darkorange"
            else:
                color = "darkgreen"

        arrow = canvas.create_line(self.pos[0]/pixel_to_m,
                                   self.pos[1]/pixel_to_m,
                                   self.pos[0]/pixel_to_m + rotate(
                                       self.get_relative_position_to_location(Goal)/pixel_to_m,
                                       self.orientation)[0],
                                   self.pos[1]/pixel_to_m + rotate(
                                       self.get_relative_position_to_location(Goal)/pixel_to_m,
                                       self.orientation)[1],
                                   arrow=LAST,
                                   fill=color,
                                   width=2)

    # function to draw robot orientation 
    def draw_orientation(self, canvas, pixel_to_m):
        line = canvas.create_line(self.pos[0]/pixel_to_m,
                                  self.pos[1]/pixel_to_m,
                                  self.pos[0]/pixel_to_m + self._radius/pixel_to_m * cos(radians(self.orientation)),
                                  self.pos[1]/pixel_to_m + self._radius/pixel_to_m * sin(radians(self.orientation)),
                                  fill="white")

    # function to draw battery level
    def write_battery_level(self, canvas, pixel_to_m):
        offset = 20
        text= canvas.create_text(self.pos[0]/pixel_to_m-offset, self.pos[1]/pixel_to_m-offset, fill="black",
                                text=f"{round(self._battery_level)}%", anchor="nw")

# ------> Data logging function
    # function to log robot learning data (updates made to the policy) for low level debugging
    def log_data(self,log_text):
        if self.data_logging:
            self.logfile.write(log_text)

    # function to log robot charge level (done at takeoff and return to FC)
    def log_charge(self,event_type):
        if self.charge_logging:
            self.charge_level_logging.append([self.id, self.clock().tick, event_type, self.get_battery_level()])


