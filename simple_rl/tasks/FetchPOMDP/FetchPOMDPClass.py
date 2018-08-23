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
#
import pyximport;
pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff
# from simple_rl.tasks.FetchPOMDP.__init__ import load_json

class FetchPOMDP(POMDP):
	def __init__(self,items = None, desired_item = 0, use_gesture = True, use_language = True, use_look = True):
		if items is None:
			self.items = cstuff.get_items()
		else:
			self.items = items
		#Defined for the sake of BSS, etc
		self.step_cost = 0
		self.in_goal_state = False
		#
		self.num_items = len(self.items)
		self.states = self.get_all_possible_states()
		self.states_by_des = []
		for i in range(self.num_items):
			self.states_by_des.append(self.get_all_possible_states(candidate_desired_items= [i]))


		self.init_state = FetchPOMDPState(*[desired_item,None,None])
		self.cur_state = copy.copy(self.init_state)
		self.init_belief = FetchPOMDPBeliefState(desired_item = [1.0 / len(self.items) for i in range(self.num_items)], state = self.init_state)
		self.cur_belief = copy.deepcopy(self.init_belief)
		self.actions = []
		for i in range(self.num_items):
			self.actions.append("pick " + str(i))
			self.actions.append("point " + str(i))
			if use_look:
				self.actions.append("look " + str(i))
		self.actions.append("wait")
		self.non_terminal_actions = [a for a in self.actions if a.split(" ")[0] != "pick"]
		self.terminal_actions = [a for a in self.actions if a.split(" ")[0] == "pick"]
		config = fr.load_json("config.json")
		self.config = config
		self.bag_of_words = config["bag_of_words"]
		self.p_g = config["p_g"]
		self.p_l = config["p_l_b"]
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

		# self.observation_func = cstuff.observation_func
		self.belief_updater_type = "FetchPOMDP_belief_updater"
		self.observation_space_type = "continuous"
		self.state_space_type = "discrete"
		self.action_space_type = "discrete"
		self.use_gesture = use_gesture
		self.use_language = use_language
		self.use_look = use_look
		# POMDP.__init__(self,self.actions,self.transition_func,self.reward_func,cstuff.observation_func, init_belief,"custom: FetchPOMDP_belief_updater", self.gamma, 0)
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
		'''
		:param belief:
		:param action:
		:return: sampled observation from taking action from belief
		'''
		new_belief = self.belief_transition_func(belief,action)
		state = new_belief.sample()
		return self.sample_observation(state)
	def sample_observation_from_belief(self,belief):
		state = belief.sample()
		return self.sample_observation(state)
	def observation_from_belief_prob(self,observation, belief, action = None):
		if action != None:
			belief = self.belief_transition_func(belief,action)
		states = belief.get_all_plausible_states()
		state_probs = [belief.belief(s) for s in states]
		conditional_probs = [cstuff.observation_func(observation,s) for s in states]
		total_prob = cstuff.dot(state_probs,conditional_probs)
		return total_prob
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
		if action != None:
			belief = self.belief_transition_func(belief,action)
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
		'''
		:param state:
		:param action:
		:return: The resulting state from taking action from state
		'''
		vals = action.split(" ")
		if vals[0] in ["look", "point"]:
			s1 = copy.deepcopy(state)
			s1["last_referenced_item"] = int(vals[1])
			s1["reference_type"] = vals[0]
			return s1
		return state
	def belief_transition_func(self,belief,action):
		vals = action.split(" ")
		if vals[0] in ["look", "point"]:
			b1 = copy.deepcopy(belief)
			b1["last_referenced_item"] = int(vals[1])
			b1["reference_type"] = vals[0]
			return b1
		return belief

	def transition_prob_func(self,state,action,state2):
		correct_state2 = self.transition_func(state,action)
		if correct_state2 == state2:
			return 1.0
		return 0.0
	def possible_next_states(self,state, action):
		return [self.transition_func(state,action)]
	def execute_action(self, action):
		if type(self.cur_belief) is list:
			raise TypeError("cur_belief has type list")
		reward = self.reward_func(self.cur_state, action)
		self.cur_state = self.transition_func(self.cur_state, action)
		observation = self.sample_observation(self.cur_state)
		self.update_cur_belief(observation)
		if action.split(" ")[0] == "pick":
			self.in_goal_state = True
		if type(self.cur_belief) is list:
			raise TypeError("cur_belief has type list")
		return (reward, self.cur_belief, observation)
	def execute_action_robot(self, action, observation):
		'''
		Same as execute action, but doesn't sample observations since the real world provides observations
		:param action:
		:return: reward, new belief state
		'''
		reward = self.reward_func(self.cur_state, action)
		self.cur_state = self.transition_func(self.cur_state, action)
		# self.cur_belief.update_from_state(self.cur_state)
		self.update_cur_belief(observation)
		return (reward, self.cur_belief)

	def reset(self):
		self.cur_belief = copy.deepcopy(self.init_belief)
		self.cur_state["desired_item"] = random.sample([i for i in range(len(self.items))], 1)[0]
		self.cur_state["last_referenced_item"] = None
		self.cur_state["reference_type"] = None
		self.cur_belief.update_from_state(self.cur_state)
	def is_terminal(self, state, a):
		vals = a.split(" ")
		return vals[0] == "pick"
	def get_cur_state(self):
		return self.cur_state
	def update_cur_belief(self, observation):
		if type(self.cur_belief) is list:
			raise TypeError("cur_belief has type list")
		self.cur_belief.update_from_state(self.cur_state)
		self.cur_belief = self.belief_update(self.cur_belief, observation)
		if type(self.cur_belief) is list:
			raise TypeError("cur_belief has type list")
		return self.cur_belief

	def get_constants(self):
		c = {"wait_cost": self.wait_cost, "point_cost": self.point_cost, "wrong_pick_cost": self.wrong_pick_cost,
		     "correct_pick_reward": self.correct_pick_reward,"look_cost": self.look_cost, "items": self.items, "gamma": self.gamma,
		     "std_theta": self.std_theta,
		     "bag_of_words": self.bag_of_words}
		return c

	def get_potential_actions(self,state):
		return self.actions
	def get_all_possible_states(self, candidate_desired_items = None, candidate_last_referenced_items = None, candidate_reference_types = None, allow_no_reference = True):
		if candidate_desired_items == None:
			candidate_desired_items = [i for i in range(self.num_items)]
		if candidate_last_referenced_items == None:
			candidate_last_referenced_items = [i for i in range(self.num_items)]
		if candidate_reference_types == None:
			candidate_reference_types = ["look","point"]
		states = []
		for i in candidate_desired_items:
			if allow_no_reference:
				s = FetchPOMDPState(desired_item= i,last_referenced_item= None, reference_type= None)
				states.append(s)
			for j in candidate_last_referenced_items:
				for ref_type in candidate_reference_types:
					s = FetchPOMDPState(desired_item=i,last_referenced_item=j,reference_type=ref_type)
					states.append(s)
		return states
	def get_non_terminal_actions(self):
		return self.non_terminal_actions
		return self.non_terminal_actions
	def is_in_goal_state(self):
		return self.in_goal_state
	def get_num_belief_updates_and_impossible_observations(self):
		return cstuff.get_num_belief_updates_and_impossible_observations()
	def get_config(self):
		return self.config
	def belief_to_array_of_probs(self,belief):
		return [belief.belief(s) for s in self.states]
def cstuff_test():
	print(cstuff.get_items())
def observation_from_belief_test(n = 10000):
	pomdp = FetchPOMDP()
	pomdp.execute_action("point 1")
	b = pomdp.cur_belief
	observations = [pomdp.sample_observation_from_belief(b) for i in range(n)]
	observation_probs = [pomdp.observation_from_belief_prob(o,b) for o in observations]
	min_prob = min(observation_probs)
	zeros = [i for i in observation_probs if i == 0]
	print("num_zeros: " + str(len(zeros)))
	print(min_prob)
# observation_from_belief_test()