import json

from helpers import random_walk

from model.environment import Environment

from random import seed

from model.navigation import Order

# class to interact with jason config files
class Configuration:
    def __init__(self, config_file):
        self._parameters = self.read_config(config_file)

    def __contains__(self, item):
        return item in self._parameters

    @staticmethod
    def read_config(config_file):
        with open(config_file, "r") as file:
            return json.load(file)

    def save(self, save_path):
        json_object = json.dumps(self._parameters, indent=2)
        with open(save_path, 'w') as file:
            file.write(json_object)

    def value_of(self, parameter):
        return self._parameters[parameter]

    def set(self, parameter, value):
        self._parameters[parameter] = value

# class that implement the clock
class Clock:
    def __init__(self, config: Configuration):
        self._tick = config.value_of("intial_step")
        self.transitory_period = 0
        if "transitory_period" in config:
            self.transitory_period = config.value_of("transitory_period")

    def step(self):
        self._tick += 1

    @property
    def tick(self):
        return self._tick

    def is_in_transitory(self):
        return self._tick < self.transitory_period

# class that implements the controller (object that progresses simulation timesteps)
class MainController:

    # initialisation function, it sets the main controller's viriables and create an environment's object
    def __init__(self, config: Configuration):
        self.config = config
        
        seed(self.config.value_of("seed"))

        self.clock = Clock(config)


        self.environment = Environment(width=self.config.value_of("width"),
                                       height=self.config.value_of("height"),
                                       pixel_to_m=self.config.value_of("pixel_to_m"),
                                       depot=self.config.value_of("depot"),
                                       evaluation_type=self.config.value_of("evaluation_type"),
                                       log_params = [self.config.value_of("data_collection")['agents_data_logging'],self.config.value_of("data_collection")['charge_level_logging'],self.config.value_of("data_collection")["output_directory"],self.config.value_of("data_collection")["filename"]],
                                       order_params=config.value_of("orders"),
                                       clock=self.clock,
                                       simulation_steps =self.config.value_of("simulation_steps"),
                                       agent_params=self.config.value_of("agent"),
                                       behavior_params=self.config.value_of("behaviors"))
        
        if self.config.value_of("evaluation_type") == "episodes":
            self.number_of_episodes = config.value_of("episodes_no")

        self.output_directory = self.config.value_of("data_collection")["output_directory"]
        self.filename = self.config.value_of("data_collection")["filename"]

        self.experiment_running = True

        self.time_evolution_recording_interval = self.config.value_of("data_collection")['recording_interval']
        self.robot_learning_recording_interval = self.config.value_of("data_collection")['learning_evaluation_interval']
        
        if self.filename is not None and self.filename != "":
            self.time_evolution_file = open(self.output_directory + "/time_evolution_" + self.filename,"w")
            self.time_evolution_file.write("Time(s)\tDelivered\tPending\tFailed\tFailed Attempts\n")



    # function that runs one simulation time step
    def step(self):
        self.clock.step()
        self.environment.step()
        if self.filename is not None or self.filename != "":
            self.record_time_evolution_data()
            self.record_robot_learning_data()

    # function that checks weather the end of the simulation has been reached
    def check_end(self):

        if self.clock.tick == self.config.value_of("simulation_steps"):
            self.experiment_running = False

        if self.config.value_of("evaluation_type")=="episodes":

            if self.environment.number_of_successes == self.config.value_of("orders")['orders_per_episode']:
                self.number_of_episodes-=1

                if self.number_of_episodes == 0:  
                    self.experiment_running = False
                else:
                    self.environment.number_of_successes = 0
                    self.environment.create_episode_orders_list()

    # function that save all necessary data that need to be saved at the final timestep
    def save_final_data(self):
        if self.filename is not None or self.filename != "":
            self.record_delivery_time_data()
            self.environment.check_orders_being_attempted()
            self.record_pending_orders_data()
            # self.record_robot_learning_data()
            if self.config.value_of("data_collection")['charge_level_logging']:
                self.record_robot_charge_level_data()

            if (self.clock.tick % self.time_evolution_recording_interval != 0):
                self.record_time_evolution_data(True)

            if (self.clock.tick % self.robot_learning_recording_interval != 0):
                self.record_robot_learning_data(True)

            self.time_evolution_file.close()

            for robot in self.environment.population:
                if hasattr(robot, 'logfile'):
                    robot.logfile.close()

    # function to start and run simulation (it keeps looping until the set number of simulation is reached)
    def start_simulation(self):

        while self.experiment_running:
            self.step()
            self.check_end()

        self.save_final_data()

    # function to get robot at specific location
    def get_robot_at(self, x, y):
        return self.environment.get_robot_at(x, y)

    # function to record  #delivered, #pending, #Failed_Attempts over time
    def record_time_evolution_data(self,end=False):
        if self.clock.tick % self.time_evolution_recording_interval == 0 or end:
            successful = len(self.environment.successful_orders_list)
            failed = self.environment.failed_delivery_attempts
            pending = len(self.environment.pending_orders_list) - self.environment.pending_orders_list.count(None) + self.environment.ongoing_attempts + self.environment.reserved_orders
            failed_attempts = self.environment.failed_delivery_attempts
            self.time_evolution_file.write(str(self.clock.tick)+'\t'+ str(successful)+'\t'+ str(pending)+'\t'+ str(failed)+'\t'+ str(failed_attempts)+'\n')

    # function to record delivered orders data (this is done at the end and records all information about an orders: arrival time, distance, weight, number of attempts...)
    def record_delivery_time_data(self):
        delivery_times_file = open(self.output_directory + "/delivery_times_" + self.filename,"w")
        delivery_times_file.write("Arrived\tDistance\tWeight\tDelivered\tTook\tAttempted\n")
        for order in self.environment.successful_orders_list:
            delivery_times_file.write(str(order.arrival_time)+"\t"+str(order.distance)+"\t"+str(order.weight)+"\t"+str(order.fulfillment_time)+"\t"+str(order.fulfillment_time-order.arrival_time)+"\t"+str(order.attempted)+"\n")
        delivery_times_file.close()

    # function to record pending orders data (this is done at the end and records all information about an orders: arrival time, distance, weight, number of attempts...)
    def record_pending_orders_data(self):
        pending_orders_file = open(self.output_directory + "/pending_orders_" + self.filename,"w")
        pending_orders_file.write("Arrived\tDistance\tWeight\tAttempted\n")
        for order in self.environment.pending_orders_list:
            if order != None:
                pending_orders_file.write(str(order.arrival_time)+"\t"+str(order.distance)+"\t"+str(order.weight)+"\t"+str(order.attempted)+"\n")
        pending_orders_file.close()

    # function to record information about the robots (ID,SoC,SoH,#Delivered,#Attempted,Decision Plane)
    def record_robot_learning_data(self,end=False):

        if self.clock.tick % self.robot_learning_recording_interval == 0 or end:

            robots_log_file = open(self.output_directory + "/robots_log_" + str(self.clock.tick)+ "_" + self.filename ,"w")

            if hasattr(self.environment.population[0].behavior, 'sgd_clf'):
                robots_log_file.write("id\tSoC\tSoH\tDelivered\tFailed\tw0\tw1\tw2\tb\n")
            else:
                robots_log_file.write("id\tSoC\tSoH\tDelivered\tFailed\n")


            for robot in self.environment.population:
                # print(robot.behavior.sgd_clf.coef_.shape)
                robots_log_file.write(str(robot.id)+"\t"+\
                                      str(robot.get_battery_level())+"\t"+\
                                      str(robot.battery_health)+"\t"+\
                                      str(robot.items_delivered)+"\t"+\
                                      str(robot.failed_deliveries))

                if hasattr(robot.behavior, 'sgd_clf'):
                    robots_log_file.write("\t"+str(robot.behavior.sgd_clf.coef_[0,0])+"\t"+\
                                               str(robot.behavior.sgd_clf.coef_[0,1])+"\t"+\
                                               str(robot.behavior.sgd_clf.coef_[0,2])+"\t"+\
                                               str(robot.behavior.sgd_clf.intercept_[0]))

                robots_log_file.write("\n")

            robots_log_file.close()

    # function to record charge level over time (for every robot, at takeoff and return)
    def record_robot_charge_level_data(self):
        cherge_level_log_file = open(self.output_directory + "/charge_level_log_" + self.filename,"w")
        cherge_level_log_file.write("robot\ttime\tcharge_level\n")

        for i in range(len(self.environment.charge_level_logging)):
            cherge_level_log_file.write(str(self.environment.charge_level_logging[i][0])+"\t"+\
                                        str(self.environment.charge_level_logging[i][1])+"\t"+\
                                        str(self.environment.charge_level_logging[i][2])+"\t"+\
                                        str(self.environment.charge_level_logging[i][3])+"\n")
        cherge_level_log_file.close()