from math import cos, sin, radians,sqrt
from model.agent import Agent
from model.navigation import Location, Order
from helpers.utils import norm, distance_between
from random import randint, random, expovariate
import numpy as np
from collections import deque
import heapq

try:
    from PIL import ImageTk
except ModuleNotFoundError:
    print("Tkinter not installed...")

# class that implements the environment (including the robots)
class Environment:

    def __init__(self, width, height, pixel_to_m, depot, evaluation_type, log_params, order_params, clock, simulation_steps, agent_params, behavior_params):
        self.population = list()
        self.width = width * pixel_to_m
        self.height = height * pixel_to_m
        self.pixel_to_m = pixel_to_m
        self.clock = clock        
        self.depot = (depot['x']*pixel_to_m, depot['y']*pixel_to_m, depot['radius']*pixel_to_m)
        self.all_orders_list = deque() 
        self.pending_orders_list = []
        self.successful_orders_list = deque()
        self.failed_orders_list = deque()      
        self.order_location_img = None
        self.package_image = None
        self.robot_image = None
        self.timestep = 0
        self.order_params = order_params       
        self.failed_delivery_attempts = 0
        self.ongoing_attempts = 0
        self.number_of_successes = 0        
        self.last_order_arrival = 0.0
        self.next_order_arrival = 0.0
        self.evaluation_type=evaluation_type
        self.simulation_steps = simulation_steps
        self.charge_level_logging = []
        self.lost_uavs = 0 

        if evaluation_type == "episodes": #other option is "continuous":
            self.create_episode_orders_list()
        else:
            self.draw_all_orders(order_params)


        self.create_robots(log_params,agent_params, behavior_params,order_params)

        self.current_order = 0
        self.current_order_advertised = False
        self.next_order_ready = False
        self.last_order_advertisment_time = -self.order_params["times"]["order_processing_interval"]

        # test variables
        self.order_test = 0

        # Images
        self.order_location_img = None
        self.background_img = None
        self.robot_image_empty = None
        self.robot_image_loaded = None
        self.depot_image = None

    def load_images(self):
        self.order_location_img = ImageTk.PhotoImage(file="../assets/house.png")
        self.background_img = ImageTk.PhotoImage(file="../assets/background.png")
        self.robot_image_empty = ImageTk.PhotoImage(file="../assets/drone_wo_load.png")
        self.robot_image_loaded = ImageTk.PhotoImage(file="../assets/drone_w_load.png")
        self.depot_image = ImageTk.PhotoImage(file="../assets/warehouse.png")

    def step(self):
        
        # 1. Update orders' list
        if self.evaluation_type == "continuous":
            self.update_pending_orders_list(self.order_params)


        # 2. Exectue robot step
        for robot in self.population:
            # self.check_locations(robot)
            robot.step()


        # 3. compute neighbors table
        pop_size = len(self.population)
        neighbors_table = [[] for i in range(pop_size)]
        for id1 in range(pop_size):
            if self.population[id1].in_depot(): # if the robot is in depot it will have neighbours
                for id2 in range(id1 + 1, pop_size):
                    if self.population[id2].in_depot(): # the robot's neighbours are all those that are in depot
                        neighbors_table[id1].append(self.population[id2])
                        neighbors_table[id2].append(self.population[id1])
                        
        # 4. update robots communication
        for robot in self.population:
            robot.communicate(neighbors_table[robot.id])


        # 5. move  to the next package if no body bidded for the current
        if len(self.pending_orders_list) > 0:

            if self.current_order_advertised: # if current order advertised, I need to check if it was taken, or skipped

                if self.pending_orders_list[self.current_order] == None: # order was taken 
                    self.current_order = 0
                    if self.pending_orders_list.count(None)<len(self.pending_orders_list):
                        while self.pending_orders_list[self.current_order] == None:
                            self.current_order += 1

                        self.next_order_ready = True

                        self.advertise_next_order();
                    else:
                        self.current_order = 0
                        self.next_order_ready = False
                        self.current_order_advertised = False
                    

                elif self.no_bids() and self.any_UAV_in():

                    if self.pending_orders_list.count(None)<len(self.pending_orders_list):

                        self.current_order += 1

                        while self.current_order<len(self.pending_orders_list) and self.pending_orders_list[self.current_order] == None:
                            self.current_order+=1

                        if self.current_order>=len(self.pending_orders_list):
                            self.current_order=0
                            while self.pending_orders_list[self.current_order] == None:
                                self.current_order += 1
                    
                        self.next_order_ready = True
                        self.advertise_next_order();

                    else:
                        self.current_order = 0
                        self.next_order_ready = False
                        self.current_order_advertised = False
            else:
                if self.next_order_ready: # order ready but it was not advertised, try to advertise it again!
                    self.advertise_next_order();
                else:
                    if self.pending_orders_list.count(None)<len(self.pending_orders_list):
                        while self.current_order<len(self.pending_orders_list) and self.pending_orders_list[self.current_order] == None:
                            self.current_order+=1

                        self.next_order_ready = True
                        self.advertise_next_order()

    def advertise_next_order(self):
        if (self.clock.tick - self.last_order_advertisment_time) >= self.order_params["times"]["order_processing_interval"] and self.any_UAV_in():
            self.last_order_advertisment_time = self.clock.tick
            self.current_order_advertised = True
            # print(self.clock.tick, "adverising order",self.pending_orders_list[self.current_order].id)
            for robot in self.population:
                if robot.in_depot():
                    robot.receive_next_order(self.pending_orders_list[self.current_order])
        else:
            self.current_order_advertised = False

    # function for checking if there are UAVs in the FC (to know if there is no bids because no one is there or because UAVs are not bidding)
    def any_UAV_in(self):
        for robot in self.population:
            if robot.in_depot():
                return True
        return False

    # function imlementing FC listening to bids from robots (it only says if there is a bid or not, does not provide information on the bids, decentralised!)
    def no_bids(self):
        for robot in self.population:
            if robot.made_bid():
                return False
        return True

    # function to create orders in the case of an episode-based testing
    def create_episode_orders_list(self):
        while len(self.pending_orders_list)<self.order_params["times"]['orders_per_episode']:
            self.pending_orders_list.append(Order(self.width, self.height, self.depot,float('inf'), self.order_params))

    # function run at the beggining to draw all orders that will arrive in the future
    def draw_all_orders(self, order_params):
        time = 0
        order_id = 1
        while time <= self.simulation_steps:
            new_order = Order(self.width, self.height, self.depot, order_id, time, order_params)
            # print("new order arrived!",new_order.distance, new_order.location, new_order.weight)
            self.all_orders_list.append(new_order)
            time += expovariate(1.0/order_params["times"]["interval_between_orders_arrivals"])
            order_id+=1

        print(f'Drawing all orders = {len(self.all_orders_list)}')

    # function that implements order arrival as simulation progresses
    def update_pending_orders_list(self, order_params):
        if len(self.all_orders_list)>0:
            if self.clock.tick >= self.all_orders_list[0].arrival_time:
                new_order = self.all_orders_list.popleft()
                # print(self.clock.tick,"new order",new_order.id)
                self.pending_orders_list.append(new_order)

    # function that creates robot objects
    def create_robots(self, log_params, agent_params, behavior_params,order_params):
        robot_id = 0
        for behavior_params in behavior_params:
            for _ in range(behavior_params['population_size']):
                robot = Agent(robot_id=robot_id,
                              # x=randint(agent_params['radius'], self.width - 1 - agent_params['radius']),
                              # y=randint(agent_params['radius'], self.height - 1 - agent_params['radius']),
                              x=self.depot[0],
                              y=self.depot[1],
                              environment=self,
                              log_params=log_params,
                              behavior_params=behavior_params,
                              order_params=order_params,
                              clock=self.clock,
                              **agent_params)
                robot_id += 1
                self.population.append(robot)


    # let the robot sense if they are in depot or at delivery location
    def get_sensors(self, robot):
        sensors = {Location.DELIVERY_LOCATION: self.senses(robot, Location.DELIVERY_LOCATION),
                   Location.DEPOT_LOCATION: self.senses(robot, Location.DEPOT_LOCATION) }
        return sensors

    # function to check if the robot has reached a location
    def senses(self, robot, location):
        if robot.locations[location]!=tuple():
            return sqrt((robot.pos[0]-robot.locations[location][0])**2 + (robot.pos[1]-robot.locations[location][1])**2) < robot.locations[location][2]

    # function that draws different visuation elements (if visualisation is activated)
    def draw(self, canvas):
        self.draw_map(canvas)
        self.draw_zones(canvas)
        self.draw_orders_locations(canvas)
        for robot in self.population:
            # robot.draw(canvas,self.robot_image)
            robot.draw_advanced(canvas,self.robot_image_empty, self.robot_image_loaded,self.pixel_to_m)

    # function that draws the maps 
    def draw_map(self,canvas):
        if self.background_img != None:
            canvas.create_image(self.width/(2*self.pixel_to_m), self.height/(2*self.pixel_to_m), image=self.background_img, anchor='center')


    # function that draws the depot
    def draw_zones(self, canvas):
        if self.depot_image != None:
            canvas.create_image(self.depot[0]/self.pixel_to_m, self.depot[1]/self.pixel_to_m, image=self.depot_image, anchor='center')
        else:
            depot_circle = canvas.create_oval(self.depot[0]/self.pixel_to_m - self.depot[2]/self.pixel_to_m,
                                             self.depot[1]/self.pixel_to_m - self.depot[2]/self.pixel_to_m,
                                             self.depot[0]/self.pixel_to_m + self.depot[2]/self.pixel_to_m,
                                             self.depot[1]/self.pixel_to_m + self.depot[2]/self.pixel_to_m,
                                             fill="orange",
                                             outline="")

    # function that draws pending orders
    def draw_orders_locations(self, canvas):

        # Draw orders with packages in pending queue
        for order in self.pending_orders_list:

            if order!=None:
                if self.order_location_img != None:
                    canvas.create_image(order.location[0]/self.pixel_to_m, order.location[1]/self.pixel_to_m, image=self.order_location_img, anchor='center')
                else:
                    order_circle = canvas.create_oval(order.location[0]/self.pixel_to_m - 10,
                                                 order.location[1]/self.pixel_to_m - 10,
                                                 order.location[0]/self.pixel_to_m + 10,
                                                 order.location[1]/self.pixel_to_m + 10,
                                                 fill="green",
                                                 outline="")

        for robot in self.population:
            if robot.carries_package():
                if self.order_location_img != None:
                    canvas.create_image(robot.attempted_delivery.location[0]/self.pixel_to_m, robot.attempted_delivery.location[1]/self.pixel_to_m, image=self.order_location_img, anchor='center')
                else:
                    order_circle = canvas.create_oval(robot.attempted_delivery.location[0]/self.pixel_to_m - 10,
                                                 robot.attempted_delivery.location[1]/self.pixel_to_m - 10,
                                                 robot.attempted_delivery.location[0]/self.pixel_to_m + 10,
                                                 robot.attempted_delivery.location[1]/self.pixel_to_m + 10,
                                                 fill="green",
                                                 outline="")
    
    #function to check orders being attempted at the moment
    def check_orders_being_attempted(self):
        if self.ongoing_attempts > 0:
            for robot in self.population:
                if robot.carries_package():
                    self.pending_orders_list[robot.attempted_delivery.id -1 ] = robot.attempted_delivery