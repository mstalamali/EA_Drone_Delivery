import copy

from helpers import random_walk as rw
from random import random, choices, gauss, uniform
from math import sin, cos, radians, pow
from collections import deque

from model.behavior import State, behavior_factory
from model.communication import CommunicationSession
from model.navigation import Location
import numpy as np

from helpers.utils import get_orientation_from_vector, rotate, CommunicationState, norm

try:
    from tkinter import LAST
except ModuleNotFoundError:
    print("Tkinter not installed...")

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
        self.get_levi_turn_angle = agent.get_levi_turn_angle
        self.get_battery_level = agent.get_battery_level
        self.get_order = agent.get_order

        self.update_state = agent.update_state
        self.make_bid=agent.set_bid

        # Packager related functions
        self.carries_package = agent.carries_package
        self.pickup_package = agent.pickup_package
        self.deliver_package = agent.deliver_package
        self.return_package = agent.return_package
        self.get_package_info = agent.get_package_info

        

class Agent:
    colors = {State.INSIDE_DEPOT_CHARGING: "red", State.INSIDE_DEPOT_MADE_BID: "orange", State.INSIDE_DEPOT_AVAILABLE: "green",\
                State.ATTEMPTING_DELIVERY: "cyan", State.RETURNING_SUCCESSFUL: "magenta", State.RETURNING_FAILED: "gray"}

    def __init__(self, robot_id, x, y, environment, behavior_params, clock, speed, radius, frame_weight, battery_weight,
                 theoritical_battery_capacity, min_battery_health, max_battery_health, noise_sampling_mu, noise_sampling_sigma, noise_sd, fuel_cost,
                 communication_radius):

        self.id = robot_id
        self.pos = np.array([x, y]).astype('float64')
        self._clock = clock

        self._speed = speed
        self._radius = radius
        self.frame_weight = frame_weight
        self.battery_weight = battery_weight

        self.pending_orders_list = environment.pending_orders_list
        self.successful_orders_list = environment.successful_orders_list

        # Set battery-related variables
        # --> Initial Values
        self.theoritical_battery_capacity = theoritical_battery_capacity
        self.battery_health = uniform(min_battery_health,max_battery_health)
        # self.battery_health = 1.0
        self.actual_battery_capacity = self.theoritical_battery_capacity*self.battery_health

        # --> Time varying values 
        self.current_battery_capacity = self.actual_battery_capacity
        self._battery_level = 100.0
        
        self._charging = False

        # Set battery related variables
        self.locations = {Location.DELIVERY_LOCATION: tuple(), Location.DEPOT_LOCATION: environment.depot}
        self.items_delivered = 0
        self.failed_deliveries = 0

        self._carries_package = False

        self.communication_radius = communication_radius
        self.comm_state = CommunicationState.CLOSED

        self.orientation = random() * 360  # 360 degree angle
        self.noise_mu = gauss(noise_sampling_mu, noise_sampling_sigma)
        if random() >= 0.5:
            self.noise_mu = -self.noise_mu
        self.noise_sd = noise_sd

        self.fuel_cost = fuel_cost
        self.environment = environment

        self.levi_counter = 1
        self.trace = deque(self.pos, maxlen=100)

        self.dr = np.array([0, 0])
        self.sensors = {}
        self.behavior = behavior_factory(behavior_params)

        self.attempted_delivery = None

        self._in_depot = True

        self.color = self.colors[State.INSIDE_DEPOT_CHARGING]

        self._bid = None

        self.api = AgentAPI(self)

        self.g = 9.81 # Gravity constant (kg/s^2)
        self.n_r = 8  # Number of rotors
        self.rho = 1.2250 # Air density at 15 deg (kg/m^3)
        self.zeta = 0.27 # Area of the spinning blade disc of one rotor (m^2)
        self.charger_power = 100 # (W)
        self.charge_efficiency = 0.95 # (%)

        self.new_nav = self.behavior.navigation_table
        # total_weight = self.frame_weight + self.battery_weight + 5
        # print((self.theoritical_battery_capacity*0.95/(pow(self.g*total_weight,1.5)/pow(2*self.n_r*self.rho*self.zeta,0.5)/3600.0))*self._speed)

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

    def __repr__(self):
        return f"bot {self.id}"

    def __hash__(self):
        return self.id

    def __eq__(self, other):
        return self.id == other.id

    def __ne__(self, other):
        return not (self == other)

    def clock(self):
        return self._clock

    def step(self):
        # print("before", self.clock().tick,self.id,self.behavior.state, self.comm_state, self._bid)

        self.behavior.navigation_table = self.new_nav
        self.sensors = self.environment.get_sensors(self)
        self.behavior.step(AgentAPI(self))
        self.move()
        self.update_battery_state()
        self.update_trace()

        # print("after", self.clock().tick,self.id,self.behavior.state, self.comm_state, self._bid)

    def communicate(self, neighbors):
        self.previous_nav = copy.deepcopy(self.behavior.navigation_table)

        if self.comm_state == CommunicationState.OPEN:
            session = CommunicationSession(self, neighbors)
            self.behavior.check_others_bid(session)

        self.new_nav = self.behavior.navigation_table
        self.behavior.navigation_table = self.previous_nav

    def get_info_from_behavior(self, location):
        return self.behavior.sell_info(location)

    def update_trace(self):
        self.trace.appendleft(self.pos[1])
        self.trace.appendleft(self.pos[0])

    def get_relative_position_to_location(self, location: Location):
        # if self.environment.get_sensors(self)[location]:
        return rotate((self.locations[location][0],self.locations[location][1]) - self.pos, -self.orientation)

    def move(self):
        wanted_movement = rotate(self.dr, self.orientation)
        noise_angle = gauss(self.noise_mu, self.noise_sd)
        noisy_movement = rotate(wanted_movement, noise_angle)
        self.orientation = get_orientation_from_vector(noisy_movement)
        prev_pos= self.pos
        self.pos = self.clamp_to_map(self.pos + noisy_movement)
        self.travelled_distance = norm(self.pos-prev_pos)

    def clamp_to_map(self, new_position):
        if new_position[0] < self._radius:
            new_position[0] = self._radius
        if new_position[1] < self._radius:
            new_position[1] = self._radius
        if new_position[0] > self.environment.width - self._radius:
            new_position[0] = self.environment.width - self._radius
        if new_position[1] > self.environment.height - self._radius:
            new_position[1] = self.environment.height - self._radius
        return new_position

    def update_levi_counter(self):
        self.levi_counter -= 1
        if self.levi_counter <= 0:
            self.levi_counter = choices(range(1, rw.get_max_levi_steps() + 1), rw.get_levi_weights())[0]

    def get_levi_turn_angle(self):
        angle = 0
        if self.levi_counter <= 1:
            angle = choices(np.arange(0, 360), rw.get_crw_weights())[0]
        self.update_levi_counter()
        return angle

    def speed(self):
        return self._speed

    def radius(self):
        return self._radius

    def get_id(self):
        return self.id

    def reward(self):
        return self.environment.payment_database.get_reward(self.id)

    def get_sensors(self):
        return self.sensors

    def set_desired_movement(self, dr):
        norm_dr = norm(dr)
        if norm_dr > self._speed:
            dr = self._speed * dr / norm_dr
        self.dr = dr

    def set_bid(self, bid):
        self._bid = bid

# ------> Charging related functions

    def update_battery_state(self):

        if self.charging():
            # Formula: charge time = (battery capacity × depth of discharge) ÷ (charge current × charge efficiency)
            # Accuracy: Highest
            # Complexity: Highest

            self.current_battery_capacity += self.actual_battery_capacity*(100.0-self._battery_level) / (self.charger_power*self.charge_efficiency*3600)
            
            if self.current_battery_capacity>self.actual_battery_capacity:
                self.current_battery_capacity=self.actual_battery_capacity

            self._battery_level = self.current_battery_capacity/self.actual_battery_capacity*100.0
        else:
            total_weight = self.frame_weight + self.battery_weight
            if self.carries_package():
                total_weight += self.attempted_delivery.weight

            self.current_battery_capacity-= pow(self.g*total_weight,1.5)/pow(2*self.n_r*self.rho*self.zeta,0.5)/3600
            self._battery_level = self.current_battery_capacity/self.actual_battery_capacity*100.0

    def get_battery_level(self):
        return self._battery_level

# ------> State related functions
    def charging(self):
        return self._charging

    def in_depot(self):
        return self._in_depot

    def update_state(self,state):
        if state == State.INSIDE_DEPOT_MADE_BID:
        # if state == State.INSIDE_DEPOT_AVAILABLE or state == State.INSIDE_DEPOT_MADE_BID:
            self.comm_state = CommunicationState.OPEN
        else:
            self.comm_state = CommunicationState.CLOSED

        if state == State.INSIDE_DEPOT_CHARGING or state == State.INSIDE_DEPOT_AVAILABLE or state == State.INSIDE_DEPOT_MADE_BID:
            self._in_depot = True
            self._charging = True
            self.pos=[self.environment.depot[0],self.environment.depot[1]]
        else:
            self._in_depot = False
            self._charging = False

        self.color = self.colors[state]

# ------> Delivery related functions
    def get_order(self):
        if len(self.pending_orders_list)>0:
            return self.pending_orders_list[0]
        else:
            return None

    def deliver_package(self):
        self._carries_package = False
        self.attempted_delivery.fulfillment_time = self.clock().tick
        self.successful_orders_list.appendleft(self.attempted_delivery)
        self.attempted_delivery = None
        self.items_delivered += 1
        self.environment.number_of_successes += 1
        self.environment.ongoing_attempts -= 1

    # def return_package(self):
    #     self._carries_package = False
    #     self.attempted_delivery.bid_start_time = float('inf')

    #     if self.get_order() == None:
    #         self.pending_orders_list.appendleft(self.attempted_delivery)
    #     else:
    #         if self.pending_orders_list[0].bid_start_time > self.clock().tick:
    #             self.pending_orders_list.appendleft(self.attempted_delivery)
    #         else:
    #             self.pending_orders_list.insert(1,self.attempted_delivery)

    #     self.attempted_delivery = None
    #     self.failed_deliveries+=1
    #     self.environment.failed_delivery_attempts+=1
    #     self.environment.ongoing_attempts-=1


    def return_package(self):
        self._carries_package = False
        self.attempted_delivery.bid_start_time = float('inf')
        
        # Put back the failed order at the end of the queue
        self.pending_orders_list.append(self.attempted_delivery)
        
        self.attempted_delivery = None
        self.failed_deliveries+=1
        self.environment.failed_delivery_attempts+=1
        self.environment.ongoing_attempts-=1

    def pickup_package(self):
        order = self.pending_orders_list.popleft()
        if self.environment.evaluation_type == "episodes":
            if order.arrival_time == float('inf'):
                order.arrival_time = self.clock().tick

        self._carries_package = True
        # self.locations[Location.DELIVERY_LOCATION] = (order.location[0],order.location[1],self.speed())
        self.locations[Location.DELIVERY_LOCATION] = (order.location[0],order.location[1],int(order.radius))
        self.attempted_delivery=order
        self.attempted_delivery.attempted+=1
        self.environment.ongoing_attempts+=1

        # print("GOOOOOOING!!!",len(self.pending_orders_list))
        # print(self.attempted_delivery.location)
        # print(self.attempted_delivery.weight)
        # print(self.attempted_delivery.arrival_time)
        # print(self.attempted_delivery.fulfillment_time)
        # print(self.attempted_delivery.bid_start_time)


    def get_package_info(self):
        return self.attempted_delivery

    def carries_package(self):
        return self._carries_package

# ------> Drawing functions
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

            # self.draw_comm_radius(canvas)
            self.draw_goal_vector(canvas)
            self.draw_orientation(canvas)
            # self.draw_trace(canvas)


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

            # self.draw_comm_radius(canvas)
            self.draw_goal_vector(canvas, pixel_to_m)
            self.draw_orientation(canvas, pixel_to_m)
            self.write_battery_level(canvas,pixel_to_m)

            # self.draw_trace(canvas)

    def draw_trace(self, canvas):
        tail = canvas.create_line(*self.trace)

    def draw_comm_radius(self, canvas):
        circle = canvas.create_oval(self.pos[0] - self.communication_radius,
                                    self.pos[1] - self.communication_radius,
                                    self.pos[0] + self.communication_radius,
                                    self.pos[1] + self.communication_radius,
                                    outline="gray")

    def draw_goal_vector(self, canvas, pixel_to_m):

        arrow = canvas.create_line(self.pos[0]/pixel_to_m,
                                   self.pos[1]/pixel_to_m,
                                   self.pos[0]/pixel_to_m + rotate(
                                       self.behavior.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION)/pixel_to_m,
                                       self.orientation)[0],
                                   self.pos[1]/pixel_to_m + rotate(
                                       self.behavior.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION)/pixel_to_m,
                                       self.orientation)[1],
                                   arrow=LAST,
                                   fill="darkgreen")
        
        arrow = canvas.create_line(self.pos[0]/pixel_to_m,
                                   self.pos[1]/pixel_to_m,
                                   self.pos[0]/pixel_to_m + rotate(
                                       self.behavior.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION)/pixel_to_m,
                                       self.orientation)[0],
                                   self.pos[1]/pixel_to_m + rotate(
                                       self.behavior.navigation_table.get_relative_position_for_location(Location.DEPOT_LOCATION)/pixel_to_m,
                                       self.orientation)[1],
                                   arrow=LAST,
                                   fill="darkorange")

    def draw_orientation(self, canvas, pixel_to_m):
        line = canvas.create_line(self.pos[0]/pixel_to_m,
                                  self.pos[1]/pixel_to_m,
                                  self.pos[0]/pixel_to_m + self._radius/pixel_to_m * cos(radians(self.orientation)),
                                  self.pos[1]/pixel_to_m + self._radius/pixel_to_m * sin(radians(self.orientation)),
                                  fill="white")

    def write_battery_level(self, canvas, pixel_to_m):
        offset = 20
        text= canvas.create_text(self.pos[0]/pixel_to_m-offset, self.pos[1]/pixel_to_m-offset, fill="black",
                                text=f"{round(self._battery_level)}%", anchor="nw")