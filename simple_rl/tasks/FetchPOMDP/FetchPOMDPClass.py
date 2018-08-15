# v 0.5
from simple_rl.pomdp.POMDPClass import POMDP
# from simple_rl.pomdp.BeliefStateClass import FlatDiscreteBeliefState
# from simple_rl.mdp.StateClass import State
import random
random.seed(0)
import copy
import os
import sys
from simple_rl.tasks.FetchPOMDP import file_reader as fr
from simple_rl.tasks.FetchPOMDP.FetchStateClass import FetchPOMDPBeliefState,FetchPOMDPState, FetchPOMDPObservation
import cython
print("In FetchPOMDPClass.py")
print(os.getcwd()+"\config.json")
print("sys.argv: " + str(sys.argv))

import pyximport;
pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff
# from simple_rl.tasks.FetchPOMDP.__init__ import load_json

class FetchPOMDP(POMDP):
	def __init__(self,items = None, desired_item = 0, use_gesture = True, use_language = True, use_look = False):
		if items is None:
			self.items = cstuff.get_items()
		else:
			self.items = items
		self.num_items = len(self.items)
		self.init_state = FetchPOMDPState(*[desired_item,None,None])
		self.curr_state = copy.copy(self.init_state)
		self.init_belief_state = FetchPOMDPBeliefState(desired_item = [1.0 / len(self.items) for i in range(self.num_items)],state = self.init_state)
		self.curr_belief_state = copy.deepcopy(self.init_belief_state)
		self.actions = []
		for i in range(self.num_items):
			self.actions.append("pick " + str(i))
			self.actions.append("point " + str(i))
			if use_look:
				self.actions.append("look " + str(i))
		self.actions.append("wait")

		config = fr.load_json("config.json")
		self.config = config
		self.bag_of_words = config["bag_of_words"]
		self.p_g = config["p_g"]
		self.p_l = config["p_l"]
		self.p_r_match = config["p_r_match"]
		self.p_r_match_look = config["p_r_match_look"]
		# self.alpha = .2
		self.std_theta = config["std_theta"]
		self.std_theta_look = config["std_theta_look"]
		self.std_theta_point = config["std_theta_point"]
		self.gamma = config["gamma"]
		self.point_cost = config["point_cost"]
		self.look_cost = config["look_cost"]
		self.wait_cost = config["wait_cost"]
		self.wrong_pick_cost = config["wrong_pick_cost"] / self.num_items
		self.correct_pick_reward = config["correct_pick_reward"] / self.num_items
		self.min_value = self.wrong_pick_cost
		self.max_value = self.correct_pick_reward

		self.observation_func = cstuff.observation_func
		self.belief_updater_type = "FetchPOMDP_belief_updater"
		self.observation_space_type = "continuous"
		self.state_space_type = "discrete"
		self.action_space_type = "discrete"
		self.use_gesture = use_gesture
		self.use_language = use_language
		self.use_look = use_look
		# POMDP.__init__(self,self.actions,self.transition_func,self.reward_func,cstuff.observation_func, init_belief_state,"custom: FetchPOMDP_belief_updater", self.gamma, 0)
		if use_gesture:
			if use_language:
				self.sample_observation = lambda state: FetchPOMDPObservation(**cstuff.sample_observation(state))
			else:
				self.sample_observation = lambda state: FetchPOMDPObservation(**{"language": None, "gesture": cstuff.sample_gesture(state)})
		elif use_language:
			self.sample_observation = lambda state: FetchPOMDPObservation(**{"language": cstuff.sample_language(state), "gesture": None})
		else:
			self.sample_observation = lambda state: FetchPOMDPObservation(**{"language": None, "gesture": None})
			self.belief_update = lambda belief_state, observation: belief_state
			print("Using neither language nor gesture in FetchPOMDP.")
			print("use_gesture: " + str(use_gesture))
			print("use_language: " + str(use_language))

	def observation_func(self,observation,state,action = None):
		'''
		:param s: state
		:param o: observation
		:param a: action
		:return: probability of receiving observation o when taking action a from state s
		'''
		return cstuff.observation_func(observation,state)
	def sample_observation_from_belief_action(self,belief,action):
		new_belief = copy.deepcopy(belief)
		vals = action.split(" ")
		if vals[0] in ["look", "point"]:
			new_belief["last_referenced_item"] = int(vals[1])
			new_belief["reference_type"] = vals[0]
		state = new_belief.sample()
		return self.sample_observation(state)
	def reward_func(self, state, action):
		vals = action.split(" ")
		if vals[0] == "point":
			return self.point_cost
		if vals[0] == "look":
			return self.look_cost
		if vals[0] == "wait":
			return self.wait_cost
		if vals[0] == "pick":
			if int(vals[1]) == state["desired_item"]:
				return self.correct_pick_reward
			else:
				return self.wrong_pick_cost
	def get_expected_reward(self, belief_state, action):
		vals = action.split(" ")
		if vals[0] == "point":
			return self.point_cost
		if vals[0] == "look":
			return self.look_cost
		if vals[0] == "wait":
			return self.wait_cost
		if vals[0] == "pick":
			correct_prob = belief_state["desired_item"][int(vals[1])]
			reward = correct_prob * self.correct_pick_reward + (1 - correct_prob) * self.wrong_pick_cost
			return reward
		print("Action with unknown reward: " + vals[0])

	def belief_update(self,belief,observation,action = None):
		'''
		:param belief: [desired_item (index), belief distribution]
		:param observation:
		:param action:
		:return:
		'''
		return FetchPOMDPBeliefState(**cstuff.belief_update(belief,observation))
	def belief_updater_func(self, belief, action, observation):
		'''
		:param belief: [last_referenced (index), [P(desired_item == i) i in items]]
		:param action: action as string. ex. "pick 0"
		:param observation: {"language": set of words, "gesture": [dx,dy,dz]
		:return: [last_referenced (index), [P'(desired_item == i) i in items]]
		'''
		return self.belief_update(belief,observation)

	def transition_func(self,state,action):
		vals = action.split(" ")
		if vals[0] in ["look", "point"]:
			s1 = copy.deepcopy(state)
			s1["last_referenced_item"] = int(vals[1])
			s1["reference_type"] = vals[0]
			return s1
		return state
	def transition_prob_func(self,state,action,state2):
		correct_state2 = self.transition_func(state,action)
		if correct_state2 == state2:
			return 1.0
		return 0.0
	def possible_next_states(self,state, action):
		return [self.transition_func(state,action)]
	def execute_action(self, action):
		if type(self.curr_belief_state) is list:
			raise TypeError("curr_belief_state has type list")
		self.curr_state = self.transition_func(self.curr_state,action)
		reward = self.reward_func(self.curr_state,action)
		observation = self.sample_observation(self.curr_state)
		self.update_curr_belief_state(observation)
		if type(self.curr_belief_state) is list:
			raise TypeError("curr_belief_state has type list")
		return (reward, self.curr_belief_state,observation)
	def execute_action_robot(self, action):
		'''
		Same as execute action, but doesn't sample observations since the real world provides observations
		:param action:
		:return: reward, new belief state
		'''
		self.curr_state = self.transition_func(self.curr_state, action)
		reward = self.reward_func(self.curr_state, action)
		self.curr_belief_state.update_from_state(self.curr_state)
		return (reward, self.curr_belief_state)

	def reset(self):
		self.curr_belief_state = copy.deepcopy(self.init_belief_state)
		self.curr_state["desired_item"] = random.sample([i for i in range(len(self.items))],1)[0]
		self.curr_state["last_referenced_item"] = None
		self.curr_state["reference_type"] = None
		self.curr_belief_state.update_from_state(self.curr_state)
	def is_terminal(self, state, a):
		vals = a.split(" ")
		return vals[0] == "pick"
	def get_true_state(self):
		return self.curr_state
	def update_curr_belief_state(self, observation):
		if type(self.curr_belief_state) is list:
			raise TypeError("curr_belief_state has type list")
		self.curr_belief_state.update_from_state(self.curr_state)
		self.curr_belief_state = self.belief_update(self.curr_belief_state,observation)
		if type(self.curr_belief_state) is list:
			raise TypeError("curr_belief_state has type list")
		return self.curr_belief_state

	def get_constants(self):
		c = {"wait_cost": self.wait_cost, "point_cost": self.point_cost, "wrong_pick_cost": self.wrong_pick_cost,
		     "correct_pick_reward": self.correct_pick_reward,"look_cost": self.look_cost, "items": self.items, "gamma": self.gamma,
		     "std_theta": self.std_theta,
		     "bag_of_words": self.bag_of_words}
		return c

	def get_potential_actions(self,state):
		return self.actions
