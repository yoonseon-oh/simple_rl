from collections import defaultdict
import copy
import math
import random
import pickle, json
from time import time
from datetime import datetime
import numpy as np
import cython
import pyximport;

pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff
from simple_rl.tasks.FetchPOMDP.FetchPOMDPClass import FetchPOMDP
from simple_rl.tasks.FetchPOMDP.FetchStateClass import FetchPOMDPBeliefState, FetchPOMDPState, FetchPOMDPObservation

# TODO check that alpha["action"] is correctly set
pickle_location = ".\\PBVIPickles\\"


class PBVI2():
	def __init__(self, pomdp, observations_sample_size=3, conservative_lower_bounds = False):
		self.pomdp = pomdp
		self.reward_func = self.pomdp.reward_func
		self.states = self.pomdp.states
		self.actions = self.pomdp.actions
		self.observations_sample_size = observations_sample_size
		self.belief_branching = 1
		self.muted = False
		self.name = "Generic PBVI2"
		self.initialize_reward_vectors()
		self.history = []
		self.conservative_lower_bounds = conservative_lower_bounds
	def initialize_reward_vectors(self):
		self.reward_vectors = {}
		for a in self.actions:
			vals = [self.reward_func(state, a) for state in self.states]
			self.reward_vectors[a] = vals
		return self.reward_vectors

	def get_best_action(self, belief):
		alpha, value = self.get_best_alpha(belief)
		return alpha[1]

	def get_value(self, belief):
		#numpy is much faster for 300 beliefs, 6 items, 500 alphas
		return cstuff.get_value_numpy(self,belief)
	def get_value2(self,belief):
		#equal spee to above. Remove one
		alpha_vals = np.array([alpha[0] for alpha in self.v])
		belief_arr = np.array(self.pomdp.belief_to_array_of_probs(belief))
		values = np.dot(alpha_vals, belief_arr)
		max_value = np.max(values)
		return max_value
	def get_best_alpha(self, belief, v=None):
		'''
		:param belief:
		:return: best_alpha, value
		'''
		if v is None:
			v = self.v
		belief_arr = np.array(self.pomdp.belief_to_array_of_probs(belief))
		alpha_vals = np.array([i[0] for i in v])
		values = np.dot(alpha_vals, belief_arr)
		index = values.argmax()
		return v[index], values[index]
	def get_best_alpha_old(self, belief, v=None):
		'''
		:param belief:
		:return: (alpha,belief . alpha)
		'''
		if v is None:
			v = self.v
		alpha, value = argmax2(v, lambda a: self.alpha_dot_b(a, belief))
		return alpha, value
	def evaluate_alphas_at_beliefs(self,alphas,beliefs):
		#beliefs evaluated at self.v differed from value by at most 10^-14
		alpha_vals = np.array([alpha[0] for alpha in alphas])
		belief_arrs = [np.array(self.pomdp.belief_to_array_of_probs(belief)) for belief in beliefs]
		values = np.dot(belief_arrs, alpha_vals.transpose())
		max_values = values.max(axis = 1)
		return max_values
	def remove_improved_beliefs(self,beliefs, new_alpha,old_alphas, threshold = 0):
		#TODO Test throroughly
		old_values = self.evaluate_alphas_at_beliefs(old_alphas,beliefs)
		new_values = self.evaluate_alphas_at_beliefs([new_alpha],beliefs)
		deltas = new_values - old_values
		filtered_beliefs = [beliefs[i] for i in range(len(beliefs)) if deltas[i] < threshold]
		return filtered_beliefs, deltas.max()
	def alpha_dot_b(self, alpha, belief_state):
		return cstuff.dot(self.pomdp.belief_to_array_of_probs(belief_state),alpha[0])
	def alpha_dot_b_numpy(self, alpha, belief_state):
		#slower (slightly) for 6 items
		belief_arr = np.array(self.pomdp.belief_to_array_of_probs(belief_state))
		alpha_vals = np.array(alpha[0])
		return np.dot(belief_arr,alpha_vals)

	def add_alpha(self, alpha):
		if alpha not in self.v:
			self.v.append(alpha)
		return self.v
	def pullback_alpha(self,alpha,belief,action,observation,observation_prob = None):
		obs_prob_by_state = [self.pomdp.observation_func(observation,s) for s in self.states]
		if observation_prob is None:
			observation_prob = cstuff.dot(obs_prob_by_state,self.pomdp.belief_to_array_of_probs(belief))
		relative_obs_probs = [p / observation_prob for p in obs_prob_by_state]
		vals = []
		for i in range(len(self.pomdp.states)):
			index = self.pomdp.states.index(self.pomdp.transition_func(self.pomdp.states[i],action))
			vals.append(alpha[0][index] * relative_obs_probs[index])
		# vals = [alpha[0][i] * relative_obs_probs[i] for i in range(len(self.pomdp.states))]
		new_alpha = [vals,action]
		return new_alpha

	def backup_fish_version(self, belief, v, num_observations=3):
		# TODO check if it is can be more efficient to compute observation prob and/or new belief while sampling observation
		alphas = []
		for a in self.actions:

			observations = [self.pomdp.sample_observation_from_belief_action(belief, a) for i in
			                range(num_observations)]
			observation_probs = [self.pomdp.observation_from_belief_prob(o, belief, a) for o in observations]
			norm_observation_probs = cstuff.unit_vectorn(observation_probs)
			new_beliefs = [self.pomdp.belief_update(belief, o, a) for o in observations]
			#TODO use * to unpack in one line zip unzip pack unpack google it
			obs_alpha_val_pairs = [self.get_best_alpha(b,v) for b in new_beliefs]
			obs_alphas = [pair[0] for pair in obs_alpha_val_pairs]
			value_of_next_beliefs = [pair[1] for pair in obs_alpha_val_pairs]
			pulled_back_alphas = [self.pullback_alpha(obs_alphas[i],belief,a,observations[i]) for i in range(num_observations)]
			# pulled_back_alphas = [self.pullback_alpha(alpha,belief,a,o) for alpha in obs_alphas]
			value_arrays_by_observation = [alpha[0] for alpha in pulled_back_alphas]
			# value_arrays_by_observation = [argmax(v, lambda alpha: self.alpha_dot_b(alpha, b))[0] for b in new_beliefs]
			alpha_a_b_vals = cstuff.linear_combination_of_lists(value_arrays_by_observation, norm_observation_probs)
			alpha_a_b_vals2 = cstuff.linear_combination_of_lists(value_arrays_by_observation, observation_probs)
			a_val = cstuff.add(self.reward_vectors[a], cstuff.times_list(alpha_a_b_vals, self.pomdp.gamma))
			a_val2 = cstuff.add(self.reward_vectors[a], cstuff.times_list(alpha_a_b_vals2, self.pomdp.gamma))
			new_alpha = [a_val, a]
			new_alpha2 = [a_val2, a]
			if a.split(" ")[0] == "point":
				expected_reward = cstuff.dot(self.pomdp.belief_to_array_of_probs(belief),self.reward_vectors[a])
				bs = [b["desired_item"][2] for b in new_beliefs]
				vs = [b * self.pomdp.correct_pick_reward + (1 - b) * self.pomdp.wrong_pick_cost for b in bs]
				# real_value = self.pomdp.point_cost + self.pomdp.gamma * cstuff.dot(vs, norm_observation_probs)
				belief_value1 = self.alpha_dot_b(new_alpha, belief)
				belief_value2 = expected_reward + self.pomdp.gamma * cstuff.dot(norm_observation_probs,value_of_next_beliefs)

				nonnormal_value = self.alpha_dot_b(new_alpha2, belief)
				pure_values = [expected_reward + self.pomdp.gamma * next_val for next_val in value_of_next_beliefs]

			alphas.append(new_alpha)
		best_alpha, best_value = self.get_best_alpha(belief, alphas)
		return best_alpha
	def backup(self, belief, v, num_observations=3):
		# TODO check if it is can be more efficient to compute observation prob and/or new belief while sampling observation
		#TODO PROBLEM! I am taking alphas that maximize next belief to represent the value of the next belief, but I need to evaluate the alphas against the current belief.
		alphas = []
		for a in self.actions:

			observations = [self.pomdp.sample_observation_from_belief_action(belief, a) for i in
			                range(num_observations)]
			observation_probs = [self.pomdp.observation_from_belief_prob(o, belief, a) for o in observations]
			norm_observation_probs = cstuff.unit_vectorn(observation_probs)
			new_beliefs = [self.pomdp.belief_update(belief, o, a) for o in observations]
			#TODO use * to unpack in one line zip unzip pack unpack google it
			obs_alpha_val_pairs = [self.get_best_alpha(b,v) for b in new_beliefs]
			obs_alphas = [pair[0] for pair in obs_alpha_val_pairs]
			alpha_a_os = [[self.alpha_a_o(alpha,a,o) for alpha in self.v] for o in observations]
			alpha_a_os = self.get_alpha_a_os(a,observations)
			best_alpha_a_os_pairs = [self.get_best_alpha(belief,alphs) for alphs in alpha_a_os]
			best_alpha_a_os = [pair[0] for pair in best_alpha_a_os_pairs]
			best_alpha_a_os_values = [alph[0] for alph in best_alpha_a_os]
			value_of_next_beliefs = [pair[1] for pair in best_alpha_a_os_pairs]
			# value_arrays_by_observation = [alpha[0] for alpha in obs_alphas]
			# value_arrays_by_observation = [argmax(v, lambda alpha: self.alpha_dot_b(alpha, b))[0] for b in new_beliefs]
			alpha_a_b_vals = cstuff.linear_combination_of_lists(best_alpha_a_os_values, norm_observation_probs)
			a_val = cstuff.add(self.reward_vectors[a], cstuff.times_list(alpha_a_b_vals, self.pomdp.gamma))
			new_alpha = [a_val, a]
			if a.split(" ")[0] == "point":
				expected_reward = cstuff.dot(self.pomdp.belief_to_array_of_probs(belief),self.reward_vectors[a])
				bs = [b["desired_item"][2] for b in new_beliefs]
				vs = [b * self.pomdp.correct_pick_reward + (1 - b) * self.pomdp.wrong_pick_cost for b in bs]
				real_value = self.pomdp.point_cost + self.pomdp.gamma * cstuff.dot(vs,norm_observation_probs)
				belief_value1 = self.alpha_dot_b(new_alpha, belief)
				belief_value2 = expected_reward + self.pomdp.gamma * cstuff.dot(norm_observation_probs,
				                                                                value_of_next_beliefs)
				# pure_values = [expected_reward + self.pomdp.gamma * next_val for next_val in value_of_next_beliefs]
			alphas.append(new_alpha)
		best_alpha, best_value = self.get_best_alpha(belief, alphas)
		return best_alpha
	def alpha_a_o(self,alpha,a,o):
		new_alpha = [[alpha[0][i] * self.pomdp.observation_func(o,self.pomdp.transition_func(self.pomdp.states[i],a)) for i in range(len(self.pomdp.states))],None]
		return new_alpha
	def get_alpha_a_os(self,a,observations):
		next_states = [self.pomdp.transition_func(self.pomdp.states[i],a) for i in range(len(self.pomdp.states))]
		obs_probs = [[self.pomdp.observation_func(o,next_states[i]) for i in range(len(next_states))] for o in observations]
		denominators = []
		for s_index in range(len(next_states)):
			sum = 0
			for o_index in range(len(observations)):
				sum += obs_probs[o_index][s_index]
			#if sum is 0, none of the observations are possible from that state, ex if all observations include response but last_referenced is None. In this case, the denominator does not matter, so set it to 1 to avoid /0
			if sum == 0:
				sum = 1
			denominators.append(sum)

		rel_probs = [[obs_probs[o_index][s_index]/denominators[s_index] for s_index in range(len(self.pomdp.states))] for o_index in range(len(observations))]
		ret_alphas = []
		for o_index in range(len(observations)):
			o_alphas = []
			for alpha in self.v:
				new_alpha = [[alpha[0][s_index] * rel_probs[o_index][s_index] for s_index in range(len(self.pomdp.states))],None]
				o_alphas.append(new_alpha)
			ret_alphas.append(o_alphas)
		return ret_alphas
	def get_horizon_0_alpha_from_action(self, action):
		alpha = [[],  action]
		alpha[0].update([self.pomdp.reward_func(state, action) for state in self.pomdp.states])
		return alpha

	def get_lower_bound_alpha_from_action(self, action, conservative_lower_bounds=True):
		# Custom for Fetch
		alpha =[[],  action]
		if action in self.pomdp.terminal_actions:
			values = [self.pomdp.reward_func(s,action) for s in self.pomdp.states]
		else:
			future_value = max(self.pomdp.wait_cost / (1 - self.pomdp.gamma),self.pomdp.wrong_pick_cost)
			if conservative_lower_bounds:
				values = [self.pomdp.reward_func(state, action) +self.pomdp.gamma *  future_value for
				          state in self.pomdp.states]
			else:
				#Currently heuristic for testing
				split_act = action.split(" ")
				if split_act[0] == "point":
					values = []
					for s in self.pomdp.states:
						if s["desired_item"] == split_act[1]:
							values.append(self.pomdp.reward_func(s,action) + self.pomdp.gamma * (0.98 * self.pomdp.correct_pick_reward + 0.02 * self.pomdp.wrong_pick_cost))
						else:
							values.append(self.pomdp.reward_func(s, action) + self.pomdp.gamma * future_value)
				else:
					values = [self.pomdp.reward_func(state, action) + self.pomdp.gamma * future_value for
					          state in self.pomdp.states]
		alpha[0] = values
		return alpha

	def pickle_partial_update(self,alphas, name = None):
		if name is None:
			name = self.name + " partial update " + str(len(alphas)) + " alphas " + str(len(self.beliefs)) + "beliefs " + str(
				datetime.now()).replace(":", ".")[:22]
		p = {"alphas": alphas, "beliefs": self.beliefs, "pomdp config": self.pomdp.config, "num_update_iterations":self.num_update_iterations, "name":self.name, "terminal_action_alphas":self.terminal_action_alphas}
		pickle.dump(p, open(pickle_location + name + ".pickle", "wb"), protocol = 2)
	def pickle_data(self, name=None, n = 100):
		if name is None:
			name = self.name + " pickle " + str(len(self.v)) + " alphas " + str(len(self.beliefs)) + "beliefs " + str(
				datetime.now()).replace(":", ".")[:22]
		p = {"alphas": self.v, "beliefs": self.beliefs, "pomdp config": self.pomdp.config, "num_update_iterations":self.num_update_iterations, "name":self.name, "terminal_action_alphas":self.terminal_action_alphas}
		pickle.dump(p, open(pickle_location + name + ".pickle", "wb"), protocol = 2)
		if len(self.v) > 0 and self.num_update_iterations > 2:
			results1 = self.run(n)
			results1.update({"action_counts": cstuff.get_count_each_action(results1["histories"]), "action_counts_precise":cstuff.get_count_each_action_precise(results1["histories"])})
			p["results"]=results1
			p["max_improvement"]=self.max_improvement
			print("solver1 %" + str(100 * float(results1["num_correct"]) / n))
			print("solver1 action counts: " + str(results1["action_counts"]))
			print("solver1 precise action counts: " + str(results1["action_counts_precise"]))
		pickle.dump(p, open(pickle_location + name + ".pickle", "wb"), protocol = 2)

	def load_from_pickle(self, p):
		self.beliefs = p["beliefs"]
		self.v = p["alphas"]
	def initialize_v(self):
		# currently custom designed for Fetch
		new_alphas = []
		actions = self.pomdp.actions
		self.terminal_action_alphas = {}
		for action in actions:
			alpha = self.get_lower_bound_alpha_from_action(action, self.conservative_lower_bounds)
			new_alphas.append(alpha)
			if action in self.pomdp.terminal_actions:
				self.terminal_action_alphas[action] = alpha
		self.v = new_alphas
		return new_alphas
	def run(self, num_episodes=5):
		# Differes from run by getting reward from mdp state in simulation
		# TODO: Save entire history (not simulation)
		num_correct = 0
		num_wrong = 0
		start_time = time()
		final_scores = []
		counter_plan_from_state = 1
		histories = []
		for episode in range(num_episodes):
			discounted_sum_rewards = 0.0
			num_iter = 0
			if not self.muted:
				print(" ")
				print('Episode {}: '.format(episode))
			self.pomdp.reset()
			curr_belief_state = copy.deepcopy(self.pomdp.get_curr_belief())
			# old_belief = copy.deepcopy(cur_belief)
			if curr_belief_state["reference_type"] in ["point", "look"]:
				raise ValueError("Belief is messed up: " + str(curr_belief_state[0]))
			alpha, value = self.get_best_alpha(self.pomdp.cur_belief)
			action = alpha[1]
			counter_plan_from_state += 1
			history = []
			running = True
			while running:
				if self.pomdp.is_terminal(curr_belief_state, action):
					running = False
				split_action = action.split(" ")
				if split_action[0] == "pick":
					if split_action[1] == str(self.pomdp.cur_state["desired_item"]):
						num_correct += 1
					else:
						num_wrong += 1
				# True state used for record keeping and is NOT used during planning
				true_state = self.pomdp.get_cur_state()
				ret = self.pomdp.execute_action(action)
				# Consider moving belief management to solver
				reward = ret[0]
				next_belief_state = ret[1]
				observation = ret[2]
				print("Action: " + str(action))
				print("Expected Reward: " + str(value))
				print("Actual Reward: " + str(reward))
				print("Observation: " + str(observation))
				if type(curr_belief_state) is list:
					raise TypeError(
						"cur_belief has type list on iteration " + str(num_iter) + " of episode " + str(
							episode) + ": " + str(curr_belief_state))

				history.append({"belief": curr_belief_state.data, "action": action,
				                "observation": make_observation_serializable(observation),
				                "reward": reward, "true state": true_state.data})
				discounted_sum_rewards += ((self.pomdp.gamma ** num_iter) * reward)
				if not self.muted:
					print('({}, {}, {}) -> {} | {}'.format(curr_belief_state, action, next_belief_state, reward,
					                                       discounted_sum_rewards))
					print("")
				curr_belief_state = copy.deepcopy(next_belief_state)
				if type(curr_belief_state) is list:
					raise TypeError(
						"cur_belief has type list on iteration " + str(num_iter) + " of episode " + str(
							episode) + ": " + str(curr_belief_state))
				if running:
					alpha, value = self.get_best_alpha(self.pomdp.cur_belief)
					action = alpha[1]

				# current_history["action"] = action
				num_iter += 1
			histories.append(history)
			final_scores.append(discounted_sum_rewards)
			if not self.muted:
				print("Number of steps in this episode = " + str(num_iter))
				print("counter_plan_from_state = " + str(counter_plan_from_state))
		# print_times()
		total_time = time() - start_time
		ctimes = cstuff.get_times()
		if not self.muted:
			print("Total time: " + str(total_time))
			print("Observation sampling time: " + str(ctimes["obs_sampling_time"]))
			print("sample_gesture_total_time: " + str(ctimes["sample_gesture_total_time"]))
			print("belief update time: " + str(ctimes["belief_update_total_time"]))
			print("observation_func_total_time: " + str(ctimes["observation_func_total_time"]))
			print("gesture_func_total_time: " + str(ctimes["gesture_func_total_time"]))
			print("Total time: " + str(total_time))
			print("Observation sampling time: " + str(ctimes["obs_sampling_time"]))
			print("sample_gesture_total_time: " + str(ctimes["sample_gesture_total_time"]))
			print("belief update time: " + str(ctimes["belief_update_total_time"]))
			print("observation_func_total_time: " + str(ctimes["observation_func_total_time"]))
			print("gesture_func_total_time: " + str(ctimes["gesture_func_total_time"]))
		return {"final_scores": final_scores, "counter_plan_from_state": counter_plan_from_state,
		        "num_correct": num_correct, "num_wrong": num_wrong, "histories": histories}
	def act(self, raw_observation):
		cur_belief = copy.deepcopy(self.pomdp.cur_belief)
		gesture = raw_observation[0]
		if gesture is not None:
			gesture = [gesture[0], gesture[1], gesture[2], gesture[3], gesture[4], gesture[5]]
		language = raw_observation[1]
		if language is not None:
			language = set(raw_observation[1].split(" "))
		else:
			language = set()
		observation = FetchPOMDPObservation(**{"language": language, "gesture": gesture})
		self.pomdp.cur_belief = FetchPOMDPBeliefState(**cstuff.belief_update_robot(self.pomdp.cur_belief, observation))
		next_action = self.get_best_action(self.pomdp.cur_belief)
		self.pomdp.execute_action_robot(next_action)
		self.history.append({"belief": cur_belief.data, "action": next_action,
		                "observation": make_observation_serializable(observation)})
		if next_action in self.pomdp.terminal_actions:
			results = self.history
			with open("RoboFetch history " + str(datetime.now()).replace(":",".")[:22] + ".json","w") as fp:
				json.dump(results, fp, indent=4)
		self.pomdp.reset()
		return next_action


class Perseus2(PBVI2):
	def __init__(self, pomdp, num_beliefs=100, belief_depth=3, observations_sample_size=3, beliefs=None, alphas=None,
	             convergence_threshold=1, conservative_lower_bounds=False, num_update_iterations = 0, name = "Perseus2", pickle = None, shani = True):
		PBVI2.__init__(self, pomdp=pomdp, observations_sample_size=observations_sample_size)
		self.conservative_lower_bounds = conservative_lower_bounds
		if shani:
			self.update_v = lambda: cstuff.update_v_shani(self, convergence_threshold)
		else:
			self.update_v = lambda:  cstuff.update_v(self, convergence_threshold)
		self.backup = cstuff.backup
		self.get_alpha_a_os = cstuff.get_alpha_a_os
		if pickle is not None:
			self.load_from_pickle(pickle)
		else:
			self.name = name
			self.num_update_iterations = num_update_iterations
			if beliefs is None:
				self.beliefs = []
				self.v = []
				self.terminal_action_alphas = []
				self.beliefs = self.initialize_beliefs(num_beliefs, depth=belief_depth)
				self.pickle_data() #change to pickle beliefs
			else:
				self.beliefs = beliefs
			if alphas is None:
				self.v = self.initialize_v()
				# self.update_v()
			else:
				self.v = alphas
			self.name = "Perseus2"
		print("Perseus constructed!")
	def load_from_pickle(self, p):
		self.beliefs = p["beliefs"]
		self.v = p["alphas"]
		self.num_update_iterations = p["num_update_iterations"]
		self.name = p["name"]
		if p["pomdp config"] != self.pomdp.config:
			print("Pickle is from pomdp with different config")
			self.conflicting_config = True
	def initialize_beliefs(self, num_beliefs, depth=3, single_run=False, b0=None):
		# TODO Investigate other methods of generation. Ex. branching
		impossible_configs = []
		self.beliefs = []
		if b0 is None:
			b0 = self.pomdp.cur_belief
		if not single_run:
			num_runs = math.ceil(num_beliefs / depth)
		else:
			num_runs = 1
			depth = num_beliefs
		self.beliefs.append(b0)
		for i in range(num_runs):
			b = b0
			for d in range(depth):
				# Sample a non terminal action since terminal actions do not provide useful belief states
				a = random.sample(self.pomdp.non_terminal_actions, 1)[0]
				b = self.pomdp.belief_transition_func(b, a)
				o = self.pomdp.sample_observation_from_belief_action(b, a)
				imposs = cstuff.is_observation_impossible(b, o)
				if imposs == 1:
					impossible_configs.append((b, o))
					while imposs == 1:
						o = self.pomdp.sample_observation_from_belief_action(b, a)
						imposs = cstuff.is_observation_impossible(b, o)
				b = self.pomdp.belief_update(b, o)
				negs = [i for i in b["desired_item"] if i < 0]
				if len(negs) > 0:
					print("Initialized impossible belief: " + str(b["desired_item"]))
				self.beliefs.append(b)
		return self.beliefs



class PBVIClassic2(PBVI2):
	def __init__(self, pomdp, observations_sample_size=3, name = "Classic PBVI2", convergence_threshold = .2):
		PBVI2.__init__(self,pomdp,observations_sample_size)
		self.v = self.initialize_v()
		self.name = name
		self.backup = cstuff.backup
		self.get_alpha_a_os = cstuff.get_alpha_a_os
		self.beliefs = [pomdp.cur_belief]
		self.observations_sample_size = observations_sample_size
		self.belief_branching = 1
		self.convergence_threshold = convergence_threshold
		self.num_update_iterations = 0
		self.update(convergence_threshold)
	def update(self, convergence_threshold):
		converged = False
		while not converged:
			self.num_update_iterations += 1
			self.collect_beliefs()
			new_vs, max_improvement = self.update_v()
			if max_improvement < convergence_threshold:
				converged = True
			print("max_improvement")
			self.pickle_data()


	def collect_beliefs(self):
		# TODO: FetchPOMDP.observation_func is a function of state only. Generalize.
		new_beliefs = []
		for belief in self.beliefs:
			successor_beliefs = []
			actions = self.pomdp.get_potential_actions(belief)
			for action in actions:
				for i in range(self.observations_sample_size):
					# Can be made faster for large observation number by sampling a number of observations for each state
					# proportional to the state's probability, see alpha_a_b
					observation = self.pomdp.sample_observation_from_belief_action(belief, action)
					new_belief = self.pomdp.belief_update(belief, observation)
					successor_beliefs.append(new_belief)
			# TODO consider using symmetric kl_divergence as metric instead
			farthest_belief = max(successor_beliefs, key = lambda b: cstuff.distance(belief["desired_item"], b["desired_item"]))
			new_beliefs.append(farthest_belief)
		self.beliefs.extend(new_beliefs)
		return self.beliefs

	def update_v(self, convergence_threshold  = .2):
		# converged = False
		# TODO: write convergence test
		# while not converged:
		max_max_improvement = 0
		converged = False
		total_deltas = [0 for b in self.beliefs]
		while not converged:
			max_improvement = 0
			v_prime = []
			cur_deltas = []
			for belief in self.beliefs:
				#TODO refactor backup to return value as well
				old_alpha, old_value = self.get_best_alpha(belief)
				new_alpha = self.backup(self, belief, self.v)
				new_value = self.alpha_dot_b(new_alpha,belief)
				improvement = new_value - old_value
				if improvement < 0:
					# print("Negative improvement: " + str(improvement) + ". Keeping old alpha")
					new_alpha = old_alpha
					improvement = 0
				cur_deltas.append(improvement)
				max_improvement = max(max_improvement,improvement)
				append_if_new(v_prime,new_alpha)
			self.v = v_prime
			total_deltas = [total_deltas[i] + cur_deltas[i] for i in range(len(self.beliefs))]
			print("max_improvement_in update_v: " + str(max_improvement))
			if max_improvement < convergence_threshold:
				converged = False
			#TODO this only accounts for single change. Update to calculate the actual max by storing list of improvements
		max_max_improvement = max(total_deltas)
		print("max_max_improvement in update_v: " + str(max_max_improvement))
		return self.v, max_max_improvement


def argmax(args, fn):
	max_value = fn(args[0])
	maxarg = args[0]
	for i in range(len(args)):
		cur_v = fn(args[i])
		if cur_v > max_value:
			maxarg = args[i]
			max_value = cur_v
	return maxarg
def append_if_new(l,a):
	if a not in l:
		l.append(a)
	return l

def argmax2(args, fn):
	'''
	:param args:
	:param fn:
	:return: (maxarg,max_value)
	'''
	max_value = fn(args[0])
	maxarg = args[0]
	for i in range(len(args)):
		cur_v = fn(args[i])
		if cur_v > max_value:
			maxarg = args[i]
			max_value = cur_v
	return (maxarg, max_value)



def test_dot_dict():
	a = {"cat": 12, "dog": 2}
	b = {"cat": 1, "dog": 2, "mouse": 444}
	print(cstuff.dot_dict(a, b))


def test_arg_max():
	v = [1, 2, 3, 4]
	f = lambda i: -i ** 2
	a = argmax(v, f)
	print(a)
	print(f(a))
	b = argmax(v, lambda i: -i ** 2)
	print(b)
	print((lambda i: -i ** 2)(b))


def times_dict(a, scalar):
	d = cstuff.times_dict(a, scalar)
	# d2 = defaultdict(lambda: scalar * a.default_factory(), d)
	return d


# def add_dict(a, b):
# 	c = cstuff.add_dict(a, b)
# 	# print(c)
# 	# c = defaultdict(lambda: a.default_factory() + b.default_factory(), c)
# 	return c
def add_dict(a, b, keys=None):
	if keys == None:
		keys = a.keys()
	ret = {key: a[key] + b[key] for key in keys}
	return ret


def test_dict_operations():
	a = defaultdict(lambda: -3, {"car": 12})
	b = defaultdict(lambda: -1, {"car": 22})
	c = add_dict(a, b)
	# print(str(c[4]))
	print(str(c[4]) + " = -4")
	d = times_dict(a, 4)
	print(str(d[6]) + " = -12")





# for i in range(100):
# 	print(" ".join([str(x) for x in pb.beliefs]))
# 	print(" ".join([str(x) for x in pb.v]))
# 	print(pb.get_value(pomdp.cur_belief))
# 	alpha, val = pb.get_best_alpha(pomdp.cur_belief)
# 	print(alpha["action"] + " has value " + str(val))
# 	pb.update_v()
# 	pb.collect_beliefs()






def pomdp_to_defaultdict(pomdp):
	return defaultdict(lambda: pomdp.min_value)


def run_perseus2(n=1):
	# Check each each function in PBVI/Perseus from the bottom up
	pomdp = FetchPOMDP()
	pb = Perseus2(pomdp, num_beliefs=100)
	for i in range(n):
		alpha, value = pb.get_best_alpha(pomdp.cur_belief)
		action = alpha[1]
		discounted_sum_rewards = 0.0
		num_iter = 0
		running = True
		while running:
			print("step " + str(num_iter))
			if pomdp.is_terminal(pomdp.cur_belief, action):
				running = False
			print(pomdp.cur_belief)
			ret = pomdp.execute_action(action)
			reward = ret[0]
			next_belief_state = ret[1]
			observation = ret[2]
			discounted_sum_rewards += ((pomdp.gamma ** num_iter) * reward)
			print(action)
			print("empirical reward: " + str(reward))
			print("expected reward: " + str(value))
			if running:
				alpha, value = pb.get_best_alpha(pomdp.cur_belief)
				action = alpha[1]
			num_iter += 1

		print("Took " + str(num_iter) + " steps to get reward " + str(discounted_sum_rewards))
		pomdp.reset()


def test_add_dict():
	ss = [FetchPOMDPState(1, i, "point") for i in range(4)]
	a = {ss[i]: i for i in range(4)}
	b = {ss[i]: i ^ 2 for i in range(4)}
	c = add_dict(a, b)
	print(c[ss[2]])




def make_observation_serializable(o):
	o2 = {"language": list(o["language"]), "gesture": o["gesture"]}


# print(pb.v)
# print(pb.get_value(pomdp.cur_belief))
# def pickle_test():
# 	pomdp = FetchPOMDP()
# 	b = pomdp.init_belief
# 	o = pomdp.sample_observation_from_belief(b)
# 	alpha = [[12 for state in pomdp.states], "wait"]
# 	p = {"b": b, "o": o, "alpha": alpha, "pomdp_config": pomdp.get_config()}
# 	current_time = str(datetime.now()).replace(":", ".")[:22]
# 	pickle.dump(p, open("pickled thing " + current_time, "wb"))
# 	p2 = pickle.load(open("pickled thing " + current_time, "rb"))
# 	print(p2)
# # test_alpha_a_o()
# # test_alpha_a_b()
# # test_get_pick_alpha()
# # test_add_dict()
# # test_get_best_alpha()
# # pickle_test()
# def backup_test():
# 	pomdp = FetchPOMDP(use_look=True)
# 	p = pickle.load(open(pickle_location + "value iteration 40 beliefs updated time 2018-08-21 15.36.31.85.pickle", "rb"))
# 	beliefs = p["beliefs"]
# 	pbvi = Perseus2(pomdp,**{"num_beliefs":100, "belief_depth":3, "observations_sample_size":3, "convergence_threshold":.5}, conservative_lower_bounds=True, beliefs = beliefs)
# 	# pbvi.update_v()
# 	alpha = pbvi.backup(pomdp.cur_belief,pbvi.v)
# 	print("done")
# def pullback_test():
# 	pass
# # backup_test()