from math import cos, sin, radians
from model.agent import Agent
from model.navigation import Location, Order
from helpers.utils import norm, distance_between
from random import randint, random, expovariate
import numpy as np
from collections import deque


try:
    from PIL import ImageTk
except ModuleNotFoundError:
    print("Tkinter not installed...")

class Environment:

    def __init__(self, width, height, pixel_to_m, depot, evaluation_type, log_params, order_params, clock, simulation_steps, agent_params, behavior_params):
        self.population = list()
        self.width = width * pixel_to_m
        self.height = height * pixel_to_m
        self.pixel_to_m = pixel_to_m
        self.clock = clock        
        self.depot = (depot['x']*pixel_to_m, depot['y']*pixel_to_m, depot['radius']*pixel_to_m)
        self.all_orders_list = deque() 
        self.lookahead_list = deque()  
        self.pending_orders_list = deque()
        self.successful_orders_list = deque()
        self.failed_orders_list = deque()      
        self.best_bot_id = self.get_best_bot_id()
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

        if evaluation_type == "episodes": #other option is "continuous":
            self.create_episode_orders_list()
        else:
            self.draw_all_orders(order_params)
            # self.update_pending_orders_list(order_params)


        self.create_robots(log_params,agent_params, behavior_params,order_params)

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
        
        # print(self.clock.tick)


        # 2. Update orders' list
        if self.evaluation_type == "continuous":
            self.update_pending_orders_list(self.order_params)


        # 4. Execture robot step
        for robot in self.population:
            # self.check_locations(robot)
            robot.step()


        # 1. compute neighbors
        pop_size = len(self.population)
        neighbors_table = [[] for i in range(pop_size)]
        for id1 in range(pop_size):
            if self.population[id1].in_depot():
                for id2 in range(id1 + 1, pop_size):
                    if self.population[id2].in_depot():
                        neighbors_table[id1].append(self.population[id2])
                        neighbors_table[id2].append(self.population[id1])
                        
        # 3. communication
        for robot in self.population:
            robot.communicate(neighbors_table[robot.id])


        # 4. move packages
        if len(self.pending_orders_list) > 1:
            if self.pending_orders_list[0].bid_start_time>self.clock.tick:
                self.pending_orders_list.rotate(-1)

        # print(len(self.pending_orders_list),len(self.pending_orders_list))
        # if len(self.lookahead_list)>0:
        #     print("time",self.clock.tick,self.lookahead_list[0].in_look_ahead)

        # print(self.clock.tick,len(self.pending_orders_list),len(self.successful_orders_list),len(self.failed_orders_list))


    def create_episode_orders_list(self):
        while len(self.pending_orders_list)<self.order_params["times"]['orders_per_episode']:
            self.pending_orders_list.append(Order(self.width, self.height, self.depot,float('inf'), self.order_params))

        # for i in range(len(self.pending_orders_list)):
        #     print(self.pending_orders_list[i].location)
        #     print(self.pending_orders_list[i].weight)
        #     print(self.pending_orders_list[i].arrival_time)
        #     print(self.pending_orders_list[i].fulfillment_time)
        #     print(self.pending_orders_list[i].bid_start_time)

    def draw_all_orders(self, order_params):
        time = 0
        while time <= self.simulation_steps:
            # print("new order arrived!")
            new_order = Order(self.width, self.height, self.depot, time, order_params)
            self.all_orders_list.append(new_order)
            time += expovariate(1.0/order_params["times"]["interval_between_orders_arrivals"])

        print(f'Drawing all orders = {len(self.all_orders_list)}')


    def update_pending_orders_list(self, order_params):
        
        # Orders arrival
        if len(self.all_orders_list)>0:
            if self.clock.tick >= self.all_orders_list[0].arrival_time:
                new_order = self.all_orders_list.popleft()
                # print(self.clock.tick,new_order.location)
                self.pending_orders_list.append(new_order)

        # Check look ahead queue and remove orders that spent long time in the the look-ahead queue        
        # i = 1
        # while len(self.lookahead_list) > 1 and i < len(self.lookahead_list):
        #     # print("********************* CHECKING",i)
        #     if (self.clock.tick-self.lookahead_list[i].in_look_ahead) > order_params["timeout"]:
        #         # print("********************* DELETING",self.lookahead_list[i].in_look_ahead)
        #         self.failed_orders_list.append(self.lookahead_list[i])
        #         del self.lookahead_list[i]
        #         continue
        #     i+=1

        # # Transfer packages from pending to look-ahead queue:
        # while len(self.lookahead_list) < order_params["look_ahead_size"] and len(self.pending_orders_list)>0:
        #     order = self.pending_orders_list.popleft()
        #     order.in_look_ahead = float(self.clock.tick)
        #     self.lookahead_list.append(order)



        # Check the queue


        # if self.order_test < 1:
        #     self.pending_orders_list.append(Order(self.width, self.height, self.depot, {
        #             "orders_arrival_probability": 0.01,
        #             "interval_between_orders_arrivals": 500.0,
        #             "min_distance": 8000,
        #             "radius": 50,
        #             "min_package_weight": 1,
        #             "max_package_weight": 5
        #           }))
        #     self.order_test+=1

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


    def get_sensors(self, robot):
        orientation = robot.orientation
        speed = robot.speed()
        sensors = {Location.DELIVERY_LOCATION: self.senses(robot, Location.DELIVERY_LOCATION),
                   Location.DEPOT_LOCATION: self.senses(robot, Location.DEPOT_LOCATION),
                   "FRONT": any(self.check_border_collision(robot, robot.pos[0] + speed * cos(radians(orientation)),
                                                            robot.pos[1] + speed * sin(radians(orientation)))),
                   "RIGHT": any(
                       self.check_border_collision(robot, robot.pos[0] + speed * cos(radians((orientation - 90) % 360)),
                                                   robot.pos[1] + speed * sin(radians((orientation - 90) % 360)))),
                   "BACK": any(self.check_border_collision(robot, robot.pos[0] + speed * cos(
                       radians((orientation + 180) % 360)),
                                                           robot.pos[1] + speed * sin(
                                                               radians((orientation + 180) % 360)))),
                   "LEFT": any(
                       self.check_border_collision(robot, robot.pos[0] + speed * cos(radians((orientation + 90) % 360)),
                                                   robot.pos[1] + speed * sin(radians((orientation + 90) % 360)))),
                   }
        return sensors

    def check_border_collision(self, robot, new_x, new_y):
        collide_x = False
        collide_y = False
        if new_x + robot._radius >= self.width or new_x - robot._radius < 0:
            collide_x = True

        if new_y + robot._radius >= self.height or new_y - robot._radius < 0:
            collide_y = True

        return collide_x, collide_y

    def senses(self, robot, location):
        if robot.locations[location]!=tuple():
            dist_vector = robot.pos - np.array([robot.locations[location][0], robot.locations[location][1]])
            dist_from_center = np.sqrt(dist_vector.dot(dist_vector))
            return dist_from_center < robot.locations[location][2]

    def is_on_top_of_spawn(self, robot, location):
        dist_vector = robot.pos - self.foraging_spawns[location].get(robot.id)
        return np.sqrt(dist_vector.dot(dist_vector)) < robot._radius

    # def get_location(self, location, agent):
    #     if agent.id in self.foraging_spawns[location]:
    #         return self.foraging_spawns[location][agent.id]
    #     else:
    #         return np.array([agent.locations[location][0], agent.locations[location][1]])

    def draw(self, canvas):
        self.draw_map(canvas)
        self.draw_zones(canvas)
        self.draw_orders_locations(canvas)
        for robot in self.population:
            # robot.draw(canvas,self.robot_image)
            robot.draw_advanced(canvas,self.robot_image_empty, self.robot_image_loaded,self.pixel_to_m)
        # self.draw_best_bot(canvas)

    def draw_map(self,canvas):
        if self.background_img != None:
            canvas.create_image(self.width/(2*self.pixel_to_m), self.height/(2*self.pixel_to_m), image=self.background_img, anchor='center')

    # def draw_market_stats(self, stats_canvas):
    #     margin = 15
    #     width = stats_canvas.winfo_width() - 2 * margin
    #     height = 20
    #     stats_canvas.create_rectangle(margin, 50, margin + width, 50 + height, fill="light green", outline="")
    #     target_demand = self.market.demand
    #     max_theoretical_supply = self.market.demand/self.demand
    #     demand_pos_x = width*target_demand/max_theoretical_supply
    #     supply_pos_x = width*self.market.get_supply()/max_theoretical_supply
    #     supply_bar_width = 2
    #     stats_canvas.create_rectangle(margin + demand_pos_x, 50, margin + width, 50 + height, fill="salmon", outline="")
    #     stats_canvas.create_rectangle(margin + supply_pos_x - supply_bar_width/2, 48, margin + supply_pos_x + supply_bar_width/2, 52 + height, fill="gray45", outline="")
    #     stats_canvas.create_text(margin + supply_pos_x - 5, 50 + height + 5, fill="gray45", text=f"{round(self.market.get_supply())}", anchor="nw", font="Arial 10")

    def draw_zones(self, canvas):
        # package_circle = canvas.create_oval(self.package[0] - self.package[2],
        #                                  self.package[1] - self.package[2],
        #                                  self.package[0] + self.package[2],
        #                                  self.package[1] + self.package[2],
        #                                  fill="green",
        #                                  outline="")
        if self.depot_image != None:
            canvas.create_image(self.depot[0]/self.pixel_to_m, self.depot[1]/self.pixel_to_m, image=self.depot_image, anchor='center')
        else:
            depot_circle = canvas.create_oval(self.depot[0]/self.pixel_to_m - self.depot[2]/self.pixel_to_m,
                                             self.depot[1]/self.pixel_to_m - self.depot[2]/self.pixel_to_m,
                                             self.depot[0]/self.pixel_to_m + self.depot[2]/self.pixel_to_m,
                                             self.depot[1]/self.pixel_to_m + self.depot[2]/self.pixel_to_m,
                                             fill="orange",
                                             outline="")

    def get_best_bot_id(self):
        best_bot_id = 0
        for bot in self.population:
            if 1 - abs(bot.noise_mu) > 1 - abs(self.population[best_bot_id].noise_mu):
                best_bot_id = bot.id
        return best_bot_id

    def draw_orders_locations(self, canvas):

        # Draw orders with packages in pending queue
        for order in self.pending_orders_list:

            if self.order_location_img != None:
                canvas.create_image(order.location[0]/self.pixel_to_m, order.location[1]/self.pixel_to_m, image=self.order_location_img, anchor='center')
            else:
                order_circle = canvas.create_oval(order.location[0]/self.pixel_to_m - 10,
                                             order.location[1]/self.pixel_to_m - 10,
                                             order.location[0]/self.pixel_to_m + 10,
                                             order.location[1]/self.pixel_to_m + 10,
                                             fill="green",
                                             outline="")

        # Draw orders with packages in look-ahead queue
        for order in self.lookahead_list:

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
                    
        # for bot_id, pos in self.foraging_spawns[Location.DEPOT].items():
        #     canvas.create_image(pos[0] - 8, pos[1] - 8, image=self.img, anchor='nw')

    def draw_best_bot(self, canvas):
        circle = canvas.create_oval(self.population[self.best_bot_id].pos[0] - 4,
                                    self.population[self.best_bot_id].pos[1] - 4,
                                    self.population[self.best_bot_id].pos[0] + 4,
                                    self.population[self.best_bot_id].pos[1] + 4,
                                    fill="red")

    def get_robot_at(self, x, y):
        selected = None
        for bot in self.population:
            if norm(bot.pos - np.array([x*self.pixel_to_m, y*self.pixel_to_m]).astype('float64')) < bot.radius():
                selected = bot
                break

        return selected

    @staticmethod
    def create_spawn_dicts():
        d = dict()
        for location in Location:
            d[location] = dict()
        return d

    def check_locations(self, robot):
        if robot.carries_package():
            if self.senses(robot, Location.DEPOT_LOCATION):
                # Spawn deposit location if needed
                if robot.id not in self.foraging_spawns[Location.DEPOT_LOCATION]:
                    self.add_spawn(Location.DEPOT_LOCATION, robot)
                # Check if robot can deposit package
                if self.is_on_top_of_spawn(robot, Location.DEPOT_LOCATION):
                    self.deposit_package(robot)
        else:
            if self.senses(robot, Location.DELIVERY_LOCATION):
                # Spawn package if needed
                if robot.id not in self.foraging_spawns[Location.DELIVERY_LOCATION]:
                    self.add_spawn(Location.DELIVERY_LOCATION, robot)
                # Check if robot can pickup package
                if self.is_on_top_of_spawn(robot, Location.DELIVERY_LOCATION):
                    self.pickup_package(robot)

    # def add_spawn(self, location, robot):
    #     rand_angle, rand_rad = random() * 360, np.sqrt(random()) * robot.locations[location][2]
    #     pos_in_circle = rand_rad * np.array([cos(radians(rand_angle)), sin(radians(rand_angle))])
    #     self.foraging_spawns[location][robot.id] = np.array([robot.locations[location][0],
    #                                                          robot.locations[location][1]]) + pos_in_circle

    # def deposit_package(self, robot):
    #     robot.drop_package()
    #     self.foraging_spawns[Location.DEPOT_LOCATION].pop(robot.id)

    #     reward = self.market.sell_strawberry(robot.id)

    #     self.payment_database.pay_reward(robot.id, reward=reward)
    #     self.payment_database.pay_creditors(robot.id, total_reward=reward)

    # def pickup_package(self, robot):
    #     robot.pickup_package()
    #     self.foraging_spawns[Location.DELIVERY_LOCATION].pop(robot.id)
    
    def check_orders_being_attempted(self):
        if self.ongoing_attempts > 0:
            for robot in self.population:
                if robot.carries_package():
                    self.pending_orders_list.appendleft(robot.attempted_delivery)