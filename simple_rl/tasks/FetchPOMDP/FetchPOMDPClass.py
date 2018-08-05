# v 0.5
from simple_rl.pomdp.POMDPClass import POMDP
import numpy as np
# import scipy.stats
import json
import random
import copy
import os
import sys
from simple_rl.tasks.FetchPOMDP import file_reader as fr
from math import log
from collections import defaultdict
from time import time
import timeit
import math
import cython
print("In FetchPOMDPClass.py")
print(os.getcwd()+"\config.json")
print("sys.argv: " + str(sys.argv))

import pyximport;
pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff
# from simple_rl.tasks.FetchPOMDP.__init__ import load_json

# jsonreader.load_json("as")


class FetchPOMDP(POMDP):
	def __init__(self,items = None, desired_item = 0):
		if items is None:
			self.items = cstuff.get_items()
		else:
			self.items = items
		self.num_items = len(self.items)
		self.init_belief = [1.0/len(self.items) for i in range(self.num_items)]
		self.curr_belief = copy.copy(self.init_belief)
		self.init_state = {"desired_item":desired_item, "last_referenced_item":None}
		# self.curr_state["desired_item"] = desired_item
		# self.last_referenced_item = None
		self.curr_state = copy.copy(self.init_state)
		# self.init_mixed_belief = [self.init_state["last_referenced_item"],self.init_belief]
		# self.curr_mixed_belief = copy.copy(self.init_mixed_belief)
		self.actions = []
		for i in range(self.num_items):
			self.actions.append("pick " + str(i))
			self.actions.append("point " + str(i))
		# self.actions.append("look " + str(i))
		self.actions.append("wait")
		self.p_g = .1
		self.p_l = .95
		self.p_r_match = .99
		self.alpha = .2
		self.std_theta = .15
		self.std_theta_look = .3
		self.gamma = .9
		self.point_cost = -1
		self.look_cost = -1 * (1.0/3.0)
		self.wait_cost = -1 * (1.0 / 6.0)
		self.wrong_pick_cost = -20 / self.num_items
		self.correct_pick_reward = 10 / self.num_items
		config = fr.load_json("config.json")
		self.bag_of_words = config["bag_of_words"]

		self.observation_func = cstuff.observation_func
		self.belief_updater_type = "FetchPOMDP_belief_updater"
		self.observation_space_type = "continuous"
		self.state_space_type = "discrete"
		self.action_space_type = "discrete"
		# POMDP.__init__(self,self.actions,self.transition_func,self.reward_func,cstuff.observation_func, init_belief,"custom: FetchPOMDP_belief_updater", self.gamma, 0)



	def observation_func(self,o,s,a = None):
		'''
		:param s: state
		:param o: observation
		:param a: action
		:return: probability of receiving observation o when taking action a from state s
		'''
		return cstuff.observation_func(o,s)

	def reward_func(self,s,a):
		vals = a.split(" ")
		if vals[0] == "point":
			return self.point_cost
		if vals[0] == "look":
			return self.look_cost
		if vals[0] == "wait":
			return self.wait_cost
		if vals[0] == "pick":
			if int(vals[1]) == s["desired_item"]:
				return self.correct_pick_reward
			else:
				return self.wrong_pick_cost
	def sample_observation_from_state(self,state):
		'''
		:param state: state object
		:return:
		'''
		return cstuff.sample_observation(state)


	def belief_update(self,belief,observation,action = None):
		'''
		:param belief: [desired_item (index), belief distribution]
		:param observation:
		:param action:
		:return:
		'''
		return cstuff.belief_update(belief)

	def transition_func(self,state,action):
		vals = action.split(" ")
		if vals[0] in ["look", "point"]:
			s1 = copy.deepcopy(state)
			s1["last_referenced_item"] = int(vals[1])
			return s1
		return state

	def execute_action(self, action):
		vals = action.split(" ")
		if vals[0] in ("point", "look"):
			self.curr_state["last_referenced_item"] = vals[1]
		if vals[0] != "pick":
			reward = self.get_reward_from_state(self.curr_state, action)
		else:
			if int(vals[1]) == self.curr_state["desired_item"]:
				reward = self.correct_pick_reward
			else:
				reward = self.wrong_pick_cost
		# results =  self.generate(self.cur_state, action)
		observation = cstuff.sample_observation(self.curr_state)
		#TODO: Refactor belief_update to return only belief distribution
		self.curr_belief = cstuff.belief_update(self.get_mixed_belief(), observation)[1]
		# print("Observation: " + str(observation))
		# print("Updated belief: " + str(self.cur_state))
		# print("Reward: " + str(reward))
		# print(" ")
		return (reward, self.get_mixed_belief())

	def belief_updater_func(self, belief, action, observation):
		'''
		:param belief: [last_referenced (index), [P(desired_item == i) i in items]]
		:param action: action as string. ex. "pick 0"
		:param observation: {"language": set of words, "gesture": [dx,dy,dz]
		:return: [last_referenced (index), [P'(desired_item == i) i in items]]
		'''
		return cstuff.belief_update(belief,observation)
	def reset(self):
		self.curr_belief = self.init_belief
		self.curr_state["desired_item"] = random.sample([i for i in range(len(self.items))],1)[0]
		self.curr_state["last_referenced_item"] = None
	def is_terminal(self, s, a):
		vals = a.split(" ")
		return vals[0] == "pick"
	def get_reward_from_state(self,s,a):
		vals = a.split(" ")
		if vals[0] == "point":
			return self.point_cost
		if vals[0] == "look":
			return self.look_cost
		if vals[0] == "wait":
			return self.wait_cost
		if vals[0] == "pick":
			if int(vals[1]) == s["desired_item"]:
				return self.correct_pick_reward
			else:
				return self.wrong_pick_cost
	def get_reward(self, b, a):
		vals = a.split(" ")
		if vals[0] == "point":
			return self.point_cost
		if vals[0] == "look":
			return self.look_cost
		if vals[0] == "wait":
			return self.wait_cost
		if vals[0] == "pick":
			correct_prob = b[1][int(vals[1])]
			reward = correct_prob * self.correct_pick_reward + (1 - correct_prob) * self.wrong_pick_cost
			return reward
		print("Action with unknown reward: " + vals[0])
	def get_true_state(self):
		return self.curr_state
	def get_mixed_belief(self):
		return [self.curr_state["last_referenced_item"],self.curr_belief]

	def get_constants(self):
		c = {"wait_cost": self.wait_cost, "point_cost": self.point_cost, "wrong_pick_cost": self.wrong_pick_cost,
		     "correct_pick_reward": self.correct_pick_reward, "items": self.items, "discount": self.gamma,
		     "std_theta": self.std_theta,
		     "bag_of_words": self.bag_of_words}
		return c
print(cstuff.get_items())