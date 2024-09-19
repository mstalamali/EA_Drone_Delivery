import copy
from abc import ABC, abstractmethod
from enum import Enum
from math import cos, radians, sin
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

from model.communication import CommunicationSession
from model.navigation import Location, NavigationTable, Order, Target
from helpers.utils import get_orientation_from_vector, norm
from random import random,randint,uniform

class State(Enum):
    WAITING = 1
    DECIDING = 2
    EVALUATING = 3
    ATTEMPTING = 4
    RETURNING = 5

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
        self.state = State.WAITING
        self.dr = np.array([0, 0]).astype('float64')
        self.id = -1
        self.takeoff_battery_level = 100.0
        self.best_bid = None
        self.my_bid = None
        self.max_difficulty = 1.0
        self.max_distance = max_distance
        self.max_weight = max_package_weight
        self.working_threshold = working_threshold
        self.delivery_outcome = 0

    def step(self, api):
        # self.dr[0], self.dr[1] = 0, 0
        self.id = api.get_id()
        sensors = api.get_sensors()
        self.update_state(sensors, api)
        self.update_movement_based_on_state(api)
        # self.check_movement_with_sensors(sensors)
        self.update_nav_table_based_on_dr()

    def check_others_bid(self, session: CommunicationSession):
        self.bids = session.get_bids()

    def update_state(self, sensors, api):
        
        if self.state == State.ATTEMPTING or self.state == State.RETURNING:
            self.navigation_table.set_relative_position_for_location(Location.DELIVERY_LOCATION, api.get_relative_position_to_location(Location.DELIVERY_LOCATION))
        
        self.navigation_table.set_relative_position_for_location(Location.DEPOT_LOCATION, api.get_relative_position_to_location(Location.DEPOT_LOCATION))

        if self.state == State.ATTEMPTING:            
            if sensors[Location.DELIVERY_LOCATION]:

                self.learn([api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level], 1)
                
                if hasattr(self, 'sgd_clf'): 
                    if hasattr(self.sgd_clf, 'coef_'):
                        # api.log_data(api.clock().tick,"learning",[api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level],1,self.sgd_clf.coef_[0,0],self.sgd_clf.coef_[0,1],self.sgd_clf.coef_[0,2],self.sgd_clf.intercept_[0])
                        api.log_data(f"{api.clock().tick}\tlearning\t{[api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level]}\t{1}\t{self.sgd_clf.coef_[0,0]}\t{self.sgd_clf.coef_[0,1]}\t{self.sgd_clf.coef_[0,2]}\t{self.sgd_clf.intercept_[0]}\n")

                api.deliver_package()
                self.delivery_outcome = 1
                self.state = State.RETURNING

            elif api.get_battery_level() <= self.takeoff_battery_level/2.0:
                self.state = State.RETURNING
                self.delivery_outcome = 0

                self.learn([api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level], 0)
                
                if hasattr(self, 'sgd_clf'): 
                    if hasattr(self.sgd_clf, 'coef_'):
                        # api.log_data(api.clock().tick,"learning",[api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level],0,self.sgd_clf.coef_[0,0],self.sgd_clf.coef_[0,1],self.sgd_clf.coef_[0,2],self.sgd_clf.intercept_[0])
                        api.log_data(f"{api.clock().tick}\tlearning\t{[api.get_package_info().distance,api.get_package_info().weight,self.takeoff_battery_level]}\t{0}\t{self.sgd_clf.coef_[0,0]}\t{self.sgd_clf.coef_[0,1]}\t{self.sgd_clf.coef_[0,2]}\t{self.sgd_clf.intercept_[0]}\n")

        
        elif self.state == State.RETURNING:
            
            if sensors[Location.DEPOT_LOCATION]:
                if self.delivery_outcome ==0:
                    api.return_package()
                self.state = State.WAITING

        elif self.state == State.WAITING:
            order = api.get_order()
            if order != None:
                self.state = State.DECIDING
                self.update_state(sensors,api)
            # print(">>>>>>>>>>>>><<<<<<<<<<<<<<<-")
            # print(order.location)
            # print(order.weight)
            # print(order.arrival_time)
            # print(order.fulfillment_time)
            # print(order.bid_start_time)

        elif self.state == State.DECIDING:
            # ------------> communicate bid: id, current battery level, number of fails? maybe we should consider the order arrival time
            state = [api.get_order().distance,api.get_order().weight,api.get_battery_level()]
            
            if self.bidding_policy(state):
                if hasattr(self, 'sgd_clf'): 
                    if hasattr(self.sgd_clf, 'coef_'):
                        api.log_data(f"{api.clock().tick}\tbidding\t{state}\t{1}\t{self.sgd_clf.coef_[0,0]}\t{self.sgd_clf.coef_[0,1]}\t{self.sgd_clf.coef_[0,2]}\t{self.sgd_clf.intercept_[0]}\n")

                self.my_bid = self.formulate_bid(api.get_order(),api.get_battery_level())
                # print("bid", self.id,self.my_bid,order.id) # <-----------------------------
                api.make_bid(self.my_bid)
                self.state = State.EVALUATING
            else:
                self.state = State.WAITING
                api.clear_next_order()
                if hasattr(self, 'sgd_clf'):
                    if hasattr(self.sgd_clf, 'coef_'):
                        api.log_data(f"{api.clock().tick}\tbidding\t{state}\t{0}\t{self.sgd_clf.coef_[0,0]}\t{self.sgd_clf.coef_[0,1]}\t{self.sgd_clf.coef_[0,2]}\t{self.sgd_clf.intercept_[0]}\n")


        elif self.state == State.EVALUATING:
# ------------> check other robots bids, if the robot has the highest number of failures it wins the bid, if tie with other robots, the robot with the smallest ID wins the bid
            # if self.number_of_failures >= max_peers_failures or (self.number_of_failures == peers_max_failures and self.id < peers_max_failures_id): # robots wins the bid
            
            # print(self.best_bid,self.my_bid == self.best_bid[0], self.id > self.best_bid[1])
            
            if api.get_order()!=None and self.evaluate_bids(api.get_order().attempted): # robots wins the bid

                # print(self.id, "*********************** I won bid! *************************",api.get_order().id)
                api.pickup_package()

                self.navigation_table.replace_information_entry(Location.DELIVERY_LOCATION, Target(api.get_package_info().location))
                
                self.state = State.ATTEMPTING

                self.takeoff_battery_level = api.get_battery_level()
            
            else:
                self.state = State.WAITING

            self.my_bid = None
            api.clear_next_order()
            api.make_bid(self.my_bid)

        api.update_state(self.state)

    def update_movement_based_on_state(self, api):
        if self.state == State.ATTEMPTING:
            self.dr = self.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION)
            dr_norm = norm(self.navigation_table.get_relative_position_for_location(Location.DELIVERY_LOCATION))
            if dr_norm > api.speed():
                self.dr = self.dr * api.speed() / dr_norm

        elif self.state == State.RETURNING:
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
        if state[2] >= self.working_threshold:
            return True
        else:
            return False

    def formulate_bid(self,order,battery_level):
        return battery_level

    def evaluate_bids(self,attempted):
        
        if len(self.bids[0])>0:

            bids = self.bids[0]
            bidders = self.bids[1]

            # find the max received bid
            max_value = np.max(bids)

            if self.my_bid > max_value: # If I am better than the max than I am the winner 
                return True

            elif self.my_bid == max_value: # If my bid is equal to the max then I have to compare IDs
                bidders_with_max_bid = [bidders[i] for i in range(len(bidders)) if bids[i] == max_value]
                max_id_max_bid = max(bidders_with_max_bid)

                if self.id > max_id_max_bid: # If my id is highest then I am the winner
                    return True
                else: # If my id is lower than I lost
                    return False

            else: # If my bid is lower than the max of the received then I lost
                return False

        else: # If I am the only bidder than I am the winner
            return True


class DecentralisedLearningBehavior_DistanceBids(NaiveBehavior):
    def __init__(self, working_threshold = 50.0,initial_assumption = 1, exploration_probability = 0.001,initialisation = 0, data_augmentation=0,loss_function = "hinge",learning_rate='optimal', alpha = 0.0001, eta0 =0.01 , scaler_type="standard", bidding_strategy = 'weak_prioritisation', model_initialisation_method = "Assumption",scaler_initialisation_method='KnownMeanVariance', min_distance= 500,max_distance=8000, min_package_weight=0.5, max_package_weight= 5.0):
        super(DecentralisedLearningBehavior_DistanceBids, self).__init__(working_threshold,min_distance,max_distance, min_package_weight, max_package_weight)
        
        self.epsilon = exploration_probability

        # Initialize the scaler and the online SVM classifier
        if scaler_type == "Standard":
            self.scaler = StandardScaler()
        elif scaler_type == "MinMax":
            self.scaler = MinMaxScaler()

        
        # print( loss_function,learning_rate,alpha,eta0)

        # Create an SGD classifier
        self.sgd_clf_random_state = randint(0,2**32 - 1)
        self.sgd_clf = SGDClassifier(loss=loss_function,learning_rate=learning_rate,alpha=alpha,eta0=eta0, warm_start=True, random_state=self.sgd_clf_random_state)

        self.initialisation_pts = initialisation

        self.data_augmentation_pts = data_augmentation

        self.initialised = False

        self.bidding_strategy = bidding_strategy

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.min_package_weight = min_package_weight
        self.max_package_weight = max_package_weight
        self.min_charge = working_threshold
        self.max_charge = 100.0


        # self.X_assumption = [[min_distance,min_package_weight,0.0],[max_distance,max_package_weight,100.0]]
        # self.y_assumption = [0,1]

        #ALL PREVIOUS RESULTS
        if initial_assumption == 1:
            # print("using first assumption")
            self.X_assumption = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,0.0]]
            self.y_assumption = [1,0]

        elif initial_assumption == 2:
            # print("using second assumption")
            self.X_assumption = [[min_distance,min_package_weight,100.0],[min_distance,min_package_weight,0.0]]
            self.y_assumption = [1,0]

        # NEW ASSUMPTION 2
        # self.X_assumption = [[max_distance,max_package_weight,100.0],[min_distance,min_package_weight,0.0]]
        # self.y_assumption = [1,0]


        # self.X_init = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,100.0],[min_distance,min_package_weight,0.0],[max_distance,max_package_weight,0.0]]
        # self.y_init = [1.0,1.0,0.0,0.0]

        self.X_init = []
        self.y_init = []

        self.scaler_initialisation_method = scaler_initialisation_method
        self.model_initialisation_method = model_initialisation_method


        # Scaled Initialisation
        if scaler_initialisation_method == "KnownMeanVariance":

            if scaler_type == "Standard":
                self.scaler.mean_= [(min_distance+max_distance)/2.0,(min_package_weight+max_package_weight)/2.0,(100.0+working_threshold)/2.0]
                self.scaler_mean = self.scaler.mean_

                self.scaler.variance_= [(max_distance-min_distance)*(max_distance-min_distance)/12.0,(max_package_weight-min_package_weight)*(max_package_weight-min_package_weight)/12.0,(100.0-working_threshold)*(100.0-working_threshold)/12.0]
                

                self.scaler.scale_ = np.sqrt(self.scaler.variance_)
                self.scaler_std = self.scaler.scale_

                # print(self.scaler.mean_)
                # print(self.scaler.scale_)
            elif scaler_type == "MinMax":
                self.scaler.data_min_ = [min_distance,min_package_weight,working_threshold]
                self.scaler.data_max_ = [max_distance,max_package_weight,100.0]
                self.scaler.min_ = [-min_distance/(max_distance-min_distance), -min_package_weight/(max_package_weight-min_package_weight), -working_threshold/(100.0-working_threshold) ]
                self.scaler.scale_ = [1.0/(max_distance-min_distance),1.0/(max_package_weight-min_package_weight),1.0/(100.0-working_threshold)]

        elif scaler_initialisation_method == "AssumptionMeanVariance":
            self.scaler.fit(self.X_assumption)

        # Initialise decision model
        if model_initialisation_method == "AssumptionFitting":
            X_scaled = self.scaler.transform(self.X_assumption)
            # self.sgd_clf.partial_fit(X_scaled, self.y_assumption, classes=np.array([0.0,1.0]))
            self.sgd_clf.fit(X_scaled, self.y_assumption)
            self.initialised=True

        elif model_initialisation_method == "CanDoEverything":
            # initial_coef = np.zeros((1, 3)) #np.array([[1.0,10.0,20.0]])
            # initial_intercept = np.zeros(1) #np.array([100.0])
            # self.sgd_clf.coef_ = np.zeros((1, 3))
            # self.sgd_clf.intercept_ = np.zeros(1)
            self.sgd_clf.coef_ = np.array([[1.0,1.0,1.0]])
            self.sgd_clf.intercept_ = np.array([20.0])
            self.sgd_clf.classes_ = np.array([0, 1])
            self.initialised=True
            # self.sgd_clf.fit_status_ = 0

            

    def step(self, api):
        # self.dr[0], self.dr[1] = 0, 0
        self.id = api.get_id()
        sensors = api.get_sensors()
        if api.clock().tick == 1 and hasattr(self, 'sgd_clf'):           
            if hasattr(self.sgd_clf, 'coef_'):
                api.log_data(f"{api.clock().tick}\tinitialisation\t{self.sgd_clf_random_state}\t{self.sgd_clf.coef_[0,0]}\t{self.sgd_clf.coef_[0,1]}\t{self.sgd_clf.coef_[0,2]}\t{self.sgd_clf.intercept_[0]}\n")        
        self.update_state(sensors, api)
        self.update_movement_based_on_state(api)
        # self.check_movement_with_sensors(sensors)
        self.update_nav_table_based_on_dr()



        # if api.clock().tick == 1:
        #     print(self.id,self.sgd_clf.coef_,self.sgd_clf.intercept_)
        #     print(self.id,self.scaler.mean_,self.scaler.scale_)
        
    def bidding_policy(self,state):
        if self.initialised:
            state_scaled = self.transform(state)
            predicted_outcome= self.predict(state_scaled)


            # predicted_outcome = self.sgd_clf.predict([state_scaled])
            # predicted_outcome_my_implementation = self.predict(state_scaled)
            # if predicted_outcome!=predicted_outcome_my_implementation:
            #     print("********", predicted_outcome, predicted_outcome_my_implementation)


            # if self.id == 21:
            #     print(self.sgd_clf.coef_,self.sgd_clf.intercept_)
            #     print(self.sgd_clf.coef_[0][0],self.sgd_clf.coef_[0][1],self.sgd_clf.coef_[0][2],self.sgd_clf.intercept_[0],state,predicted_outcome[0]) # <-----------------------------
            #     print(self.sgd_clf.decision_function(state_scaled)[0])
            #     print()
            if predicted_outcome == 1:
                return True
            else:
                if random() <= self.epsilon:
                    # print(self.id,"exploring") # <-----------------------------
                    return True
                else:
                    return False
        else:
            return True


    def formulate_bid(self,order,battery_level):
        if self.initialised:
            state = [order.distance,order.weight,battery_level]

            state_scaled = self.transform(state)
            
            # TODO FOR HUBER LOSS TRY TO USE PROBABILITY            
            # raw_distance = self.sgd_clf.decision_function([state_scaled])[0]
            # raw_distance_my_implementation = self.decision_function(state_scaled)
            # print(raw_distance,raw_distance_my_implementation)

            raw_distance = self.decision_function(state_scaled)

            weight_norm = np.linalg.norm(self.sgd_clf.coef_)

            normalised_distance = raw_distance/weight_norm

            if self.bidding_strategy == "Random":
                return random()
            else:
                return normalised_distance

        else:
            return battery_level


    # def learn(take_off_battery_level, package_location, package_weight, reward):
    def learn(self,state, outcome):
        if self.initialised:
            # print("learn",self.id,state,outcome) # <-----------------------------
            if self.data_augmentation_pts == 0:
                # state_scaled = self.scaler.transform([state]) # CHANGE: changed transform to fit_transform
                state_scaled = self.transform(state)
                self.sgd_clf.partial_fit([state_scaled], [outcome])
            else:
                states = [state]
                outcomes = [outcome]*(self.data_augmentation_pts+1)

                for aug_st_num in range(0,self.data_augmentation_pts):
                    if outcome == 1:
                        states.append([uniform(self.min_distance,state[0]),uniform(self.min_package_weight,state[1]),uniform(state[2],self.max_charge)])

                    if outcome == 0:
                        states.append([uniform(state[0],self.max_distance),uniform(state[1],self.max_package_weight),uniform(self.min_charge,state[2])])

                # print(outcome,states)
                # print()
                state_scaled = self.scaler.transform(states) # CHANGE: changed transform to fit_transform
                self.sgd_clf.partial_fit(state_scaled, outcomes)

        else:
            self.X_init.append(state)
            self.y_init.append(outcome)
            if len(self.X_init) >= self.initialisation_pts and len(np.unique(self.y_init))==2:
                if self.scaler_initialisation_method == "DataMeanVariance":
                    self.scaler.fit(self.X_init)

                X_scaled = self.scaler.transform(self.X_init)
                # self.sgd_clf.partial_fit(X_scaled, self.y_init, classes=np.array([0,1]))
                self.sgd_clf.fit(X_scaled, self.y_init)

                print(self.id,"initialised!")

                self.initialised=True


    def evaluate_bids(self,attempted):
        
        if len(self.bids[0])>0:

            bids = self.bids[0]
            bidders = self.bids[1]

            max_value = np.max(bids)

            if self.my_bid > max_value:
                max_value = self.my_bid

            normalised_bids = [float(x)/max_value for x in bids]
            my_normalised_bid = float(self.my_bid)/max_value
            
            if self.bidding_strategy == "Weakest":
                A = 0
            elif self.bidding_strategy == "Strongest" or self.bidding_strategy == "Random":
                A = 1
            elif self.bidding_strategy == "Adaptive":
                A = 2/(1+np.exp(-attempted))-1

            bids_distances = [abs(bid-A) for bid in normalised_bids]

            index=np.argmin(bids_distances)
            min_distance_value = bids_distances[index]

            if bids_distances.count(min_distance_value) == 1:
                best_bid = [min_distance_value, bidders[index]]
            else:
                bidders_with_min_bid = [bidders[i] for i in range(len(bidders)) if bids_distances[i] == min_distance_value]
                best_bid = [min_distance_value, max(bidders_with_min_bid)]

            my_distance_value = abs(my_normalised_bid-A)

            if my_distance_value < best_bid[0] or (my_distance_value == best_bid[0] and self.id > best_bid[1]):
                return True
            else:
                return False
        else:
            return True


    def transform(self,state):
        return [(state[0]-self.scaler_mean[0])/self.scaler_std[0], (state[1]-self.scaler_mean[1])/self.scaler_std[1], (state[2]-self.scaler_mean[2])/self.scaler_std[2]]

    def decision_function(self,state):
        return self.sgd_clf.coef_[0,0] * state[0] + self.sgd_clf.coef_[0,1] * state[1] +  self.sgd_clf.coef_[0,2] * state[2] + self.sgd_clf.intercept_[0]      

    def predict(self,state):
        if self.decision_function(state)>=0:
            return 1
        else:
            return 0

# class DecentralisedLearningBehavior_ProbabilityBids(NaiveBehavior):
#     def __init__(self, working_threshold = 50.0,exploration_probability = 0.001, initialisation = 0, bidding_strategy = 'weak_prioritisation' , min_distance= 500,max_distance=8000, min_package_weight=0.5, max_package_weight= 5.0):
#         super(DecentralisedLearningBehavior_ProbabilityBids, self).__init__(working_threshold,min_distance,max_distance, min_package_weight, max_package_weight)
        
#         self.epsilon = exploration_probability

#         # Initialize the scaler and the online SVM classifier
#         self.scaler = StandardScaler()
        
#         # Create an SGD classifier with a hinge loss (SVM)
#         self.sgd_clf = SGDClassifier(loss='log_loss',warm_start=True,random_state=randint(0,2**32 - 1))

#         self.bidding_strategy = bidding_strategy

#         self.initialisation_pts = initialisation

#         self.initialised = False

#         self.X_init = []
#         self.y_init = []

#         # Initialise decision model
#         if self.initialisation_pts == 0: # initialise basd on assumptions    
#             self.X_init = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,0.0]]
#             self.y_init = [1.0,0.0]
#             # X = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,100.0],[min_distance,min_package_weight,0.0],[max_distance,max_package_weight,0.0]]
#             # y = [1.0,1.0,0.0,0.0]
#             X_scaled = self.scaler.fit_transform(self.X_init)
#             self.sgd_clf.partial_fit(X_scaled, self.y_init, classes=np.array([0.0,1.0]))
#             self.initialised=True

#     def bidding_policy(self,state):

#         if self.initialised:
#             state_scaled = self.scaler.transform([state])
#             predicted_outcome = self.sgd_clf.predict(state_scaled)

#             # print("predict",self.id,state,predicted_outcome) # <-----------------------------
#             if predicted_outcome == 1.0:
#                 return True
#             else:
#                 if random() <= self.epsilon:
#                     # print(self.id,"exploring") # <-----------------------------
#                     return True
#                 else:
#                     return False
#         else:
#             return True

#     def formulate_bid(self,order,battery_level):
#         if self.initialised:
#             state = [order.distance,order.weight,battery_level]
#             state_scaled = self.scaler.transform([state])
#             # print(self.id,self.sgd_clf.predict_proba(state_scaled)[0][0])
#             if self.bidding_strategy == "weak_prioritisation":
#                 return self.sgd_clf.predict_proba(state_scaled)[0][0]
#             elif self.bidding_strategy == "strong_prioritisation":
#                 return 1.0-self.sgd_clf.predict_proba(state_scaled)[0][0]
#             elif self.bidding_strategy == "random":
#                 return random()
#         else:
#             return battery_level

#     # def learn(take_off_battery_level, package_location, package_weight, reward):
#     def learn(self,state, outcome):
#         if self.initialised == True:
#             # print("learn",self.id,state,outcome) # <-----------------------------
#             state_scaled = self.scaler.transform([state]) # CHANGE: changed transform to fit_transform
#             self.sgd_clf.partial_fit(state_scaled, [outcome])
#         else:
#             self.X_init.append(state)
#             self.y_init.append(outcome)
#             if len(self.X_init) == self.initialisation_pts:
#                 X_scaled = self.scaler.fit_transform(self.X_init)
#                 self.sgd_clf.partial_fit(X_scaled, self.y_init, classes=np.array([0.0,1.0]))
#                 self.initialised=True

# class CentralisedLearningBehavior(NaiveBehavior):
#     def __init__(self, working_threshold = 60.0):
#         super(CentralisedLearningBehavior, self).__init__(working_threshold)

#     def bidding_policy(self,current_battery_level,order):
# #--> TO BE DESIGNED
#         return current_battery_level


#     def learn(take_off_battery_level, package_location, package_weight, reward):
# #--> TO BE DESIGNED
#         pass

# class DecentralisedLearningBehavior_HeuristicBids(NaiveBehavior):
#     def __init__(self, working_threshold = 50.0,exploration_probability = 0.001, initialisation = 0, min_distance= 500,max_distance=8000, min_package_weight=0.5, max_package_weight= 5.0):
#         super(DecentralisedLearningBehavior_HeuristicBids, self).__init__(working_threshold,min_distance,max_distance, min_package_weight, max_package_weight)
        
#         self.epsilon = exploration_probability

#         # Initialize the scaler and the online SVM classifier
#         self.scaler = StandardScaler()
        
#         # Create an SGD classifier with a hinge loss (SVM)
#         self.sgd_clf = SGDClassifier(loss='hinge',warm_start=True,random_state=randint(0,2**32 - 1))

#         self.initialisation_pts = initialisation

#         self.initialised = False

#         self.X_init = []
#         self.y_init = []

#         # Initialise decision model
#         if self.initialisation_pts == 0: # initialise basd on assumptions    
#             self.X_init = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,0.0]]
#             self.y_init = [1.0,0.0]
#             # X = [[min_distance,min_package_weight,100.0],[max_distance,max_package_weight,100.0],[min_distance,min_package_weight,0.0],[max_distance,max_package_weight,0.0]]
#             # y = [1.0,1.0,0.0,0.0]
#             X_scaled = self.scaler.fit_transform(self.X_init)
#             self.sgd_clf.partial_fit(X_scaled, self.y_init, classes=np.array([0.0,1.0]))
#             self.initialised=True

#     def bidding_policy(self,state):
#         if self.initialised:
#             state_scaled = self.scaler.transform([state])
#             predicted_outcome = self.sgd_clf.predict(state_scaled)

#             # print("predict",self.id,state,predicted_outcome) # <-----------------------------
#             if predicted_outcome == 1.0:
#                 return True
#             else:
#                 if random() <= self.epsilon:
#                     # print(self.id,"exploring") # <-----------------------------
#                     return True
#                 else:
#                     return False
#         else:
#             return True


#     def formulate_bid(self,order,battery_level):
#         if self.initialised:
#             return (0.5*order.distance/self.max_distance + 0.5*order.weight/self.max_weight)/(battery_level/100.0)
#         else:
#             return battery_level

#     # def learn(take_off_battery_level, package_location, package_weight, reward):
#     def learn(self,state, outcome):
#         # print("learn",self.id,state,outcome) # <-----------------------------
#         if self.initialised == True:
#             state_scaled = self.scaler.transform([state]) # CHANGE: changed transform to fit_transform
#             self.sgd_clf.partial_fit(state_scaled, [outcome])
#         else:
#             self.X_init.append(state)
#             self.y_init.append(outcome)
#             if len(self.X_init) == self.initialisation_pts:
#                 X_scaled = self.scaler.fit_transform(self.X_init)
#                 self.sgd_clf.partial_fit(X_scaled, self.y_init, classes=np.array([0.0,1.0]))
#                 self.initialised=True