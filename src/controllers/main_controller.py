import json

from helpers import random_walk

from model.environment import Environment

from random import seed

from model.navigation import Order


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

class Clock:
    def __init__(self, config: Configuration):
        self._tick = 0
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


class MainController:

    def __init__(self, config: Configuration):
        self.config = config
        
        seed(self.config.value_of("seed"))

        self.clock = Clock(config)

        self.environment = Environment(width=self.config.value_of("width"),
                                       height=self.config.value_of("height"),
                                       pixel_to_m=self.config.value_of("pixel_to_m"),
                                       depot=self.config.value_of("depot"),
                                       evaluation_type=self.config.value_of("evaluation_type"),
                                       order_params=config.value_of("orders"),
                                       clock=self.clock,
                                       agent_params=self.config.value_of("agent"),
                                       behavior_params=self.config.value_of("behaviors"))
        
        self.output_directory = self.config.value_of("data_collection")["output_directory"]
        self.filename = self.config.value_of("data_collection")["filename"]

        if self.filename is not None and self.filename != "":
            self.time_evolution_file = open(self.output_directory + "/time_evolution_" + self.filename,"w")
            self.time_evolution_file.write("Time(s)\tDelivered\tPending\tFailed\n")


    def step(self):

        if self.clock.tick < self.config.value_of("simulation_steps") or ( not (len(self.environment.successful_orders_list) == self.config.value_of("orders")['orders_per_episode']) and (self.config.value_of("evaluation_type")=="episodes")):
            self.clock.step()
            self.environment.step()
            if self.filename is not None or self.filename != "":
                self.record_time_evolution_data()
        # else:
        #     print(len(self.environment.successful_orders_list),self.environment.failed_delivery_attempt)
        #     if self.filename is not None or self.filename != "":
        #         self.record_delivery_time_data()
        #         self.time_evolution_file.close()



    def start_simulation(self):
        for step_nb in range(self.config.value_of("simulation_steps")):
            self.step()

        if self.filename is not None or self.filename != "":
            self.record_delivery_time_data()
            if (self.clock.tick % self.config.value_of("data_collection")['recording_interval'] != 0):
                self.record_time_evolution_data(True)
            self.time_evolution_file.close()

    def get_robot_at(self, x, y):
        return self.environment.get_robot_at(x, y)

    def record_time_evolution_data(self,end=False):
        if self.clock.tick % self.config.value_of("data_collection")['recording_interval'] == 0 or end:

            successful = len(self.environment.successful_orders_list)
            failed = self.environment.failed_delivery_attempts
            pending = len(self.environment.pending_orders_list) + self.environment.ongoing_attempts
            self.time_evolution_file.write(str(self.clock.tick)+'\t'+ str(successful)+'\t'+ str(pending)+'\t'+ str(failed)+'\n')

    def record_delivery_time_data(self):
        delivery_times_file = open(self.output_directory + "/delivery_times_" + self.filename,"w")
        delivery_times_file.write("Arrived\tDelivered\tTook\n")
        for order in self.environment.successful_orders_list:
            delivery_times_file.write(str(order.arrival_time)+"\t"+str(order.fulfillment_time)+"\t"+str(order.fulfillment_time-order.arrival_time)+"\n")
        delivery_times_file.close()



#   Data retrieval functions

    # def get_rewards_evolution(self):
    #     return self.rewards_evolution

    # def get_rewards_evolution_list(self):
    #     return self.rewards_evolution_list

    # def get_items_collected_stats(self):
    #     res = ""
    #     for bot in self.environment.population:
    #         res += str(bot.items_collected) + ","
    #     res = res[:-1]  # remove last comma
    #     res += "\n"
    #     return res

    # def get_rewards(self):
    #     return [bot.reward() for bot in self.environment.population]

    # def get_items_collected(self):
    #     return [bot.items_collected for bot in self.environment.population]

    # def get_sorted_reward_stats(self):
    #     sorted_bots = sorted([bot for bot in self.environment.population], key=lambda bot: abs(bot.noise_mu))
    #     res = ""
    #     for bot in sorted_bots:
    #         res += str(bot.reward()) + ","
    #     res = res[:-1]  # remove last comma
    #     res += "\n"
    #     return res

    # def get_reward_stats(self):
    #     res = ""
    #     for bot in self.environment.population:
    #         res += str(bot.reward()) + ","
    #     res = res[:-1]  # remove last comma
    #     res += "\n"
    #     return res