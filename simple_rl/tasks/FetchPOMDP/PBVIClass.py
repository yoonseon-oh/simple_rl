from collections import defaultdict
import copy
import math
import random
import pickle
from time import time
from datetime import datetime
import cython
import pyximport;

pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff
from simple_rl.tasks.FetchPOMDP.FetchPOMDPClass import FetchPOMDP
from simple_rl.tasks.FetchPOMDP.FetchStateClass import FetchPOMDPBeliefState, FetchPOMDPState, FetchPOMDPObservation

# TODO check that alpha["action"] is correctly set
pickle_location = ".\\PBVIPickles\\"


class PBVI():
	def __init__(self, pomdp, observations_sample_size=3):
		self.pomdp = pomdp
		self.reward_func = self.pomdp.reward_func
		self.states = self.pomdp.states
		self.actions = self.pomdp.actions
		self.observations_sample_size = observations_sample_size
		self.belief_branching = 1
		self.muted = False
		self.name = "Generic PBVI"
		self.initialize_reward_vectors()

	def initialize_reward_vectors(self):
		self.reward_vectors = {}
		for a in self.actions:
			vals = {state: self.reward_func(state, a) for state in self.states}
			self.reward_vectors[a] = vals
		return self.reward_vectors

	def get_best_action(self, belief):
		alpha = self.get_best_alpha(belief)
		return alpha["action"]

	def get_value(self, belief):
		values = [self.alpha_dot_b(alpha, belief) for alpha in self.v]
		max_value = max(values)
		return max_value

	def get_best_alpha(self, belief, v=None):
		'''
		:param belief:
		:return: (alpha,belief . alpha)
		'''
		if v is None:
			v = self.v
		alpha, value = argmax2(v, lambda a: self.alpha_dot_b(a, belief))
		return alpha, value

	def alpha_dot_b(self, alpha, belief_state):
		# TODONE? Fixed by putting alpha as second argument since dot_dict iterates over keys in first argument
		# TODO check speed, create vectorized version for multiple beliefs and/or alphas
		return cstuff.dot_dict(belief_state.get_explicit_distribution(), alpha["values"])

	def add_alpha(self, alpha):
		if alpha not in self.v:
			self.v.append(alpha)
		return self.v

	def backup_old(self, belief_state, v):
		alphas = [self.alpha_a_b(action, belief_state) for action in self.pomdp.actions]
		return argmax(alphas, lambda alpha: self.alpha_dot_b(alpha, belief_state))

	def backup(self, belief, v, num_observations=3):
		# TODO check if it is can be more efficient to compute observation prob and/or new belief while sampling observation
		alphas = []
		for a in self.actions:
			if a.split(" ")[0] == "point":
				pointing = True
			observations = [self.pomdp.sample_observation_from_belief_action(belief, a) for i in
			                range(num_observations)]
			observation_probs = [self.pomdp.observation_from_belief_prob(o, belief, a) for o in observations]
			norm_observation_probs = cstuff.unit_vectorn(observation_probs)
			new_beliefs = [self.pomdp.belief_update(belief, o, a) for o in observations]
			vals_per_observation = [argmax(v, lambda alpha: self.alpha_dot_b(alpha, b))["values"] for b in new_beliefs]
			alpha_a_b_vals = cstuff.linear_combination_of_dicts(vals_per_observation, norm_observation_probs)
			a_val = cstuff.add_dict(self.reward_vectors[a], cstuff.times_dict(alpha_a_b_vals, self.pomdp.gamma))
			new_alpha = {"values": a_val, "action": a}
			alphas.append(new_alpha)
		best_alpha = self.get_best_alpha(belief, alphas)[0]
		return best_alpha

	def alpha_a_b(self, action, belief):
		'''
		:param action:
		:param belief:
		:return: a (non default) dict with keys in belief.possible_states and values as in shani (34)
		'''
		# TODO perhaps we should evaluate alpha_a_o against b_a_o. Shani and Pineau both say b, but that seems odd :/
		# Not a function of alpha. Shani (33) seems to have a typo, since we are taking argmax over alphas
		# b = self.pomdp.belief_transition_func(belief, action)
		b = belief
		possible_states = b.get_all_plausible_states()
		# possible_states = belief.get_all_possible_states()
		# possible_states = self.pomdp.states
		observations_per_state = {state: math.ceil(self.observations_sample_size * b.belief(state)) for state in
		                          possible_states}
		# Consider storing r_a's based on belief.known for speedup
		r_a = {state: self.reward_func(state, action) for state in self.pomdp.states}
		observations_by_state = {
			state: [self.pomdp.sample_observation(state) for i in range(observations_per_state[state])] for state in
			possible_states}
		# all_sampled_observations might be broken
		all_sampled_observations = [observation for l in observations_by_state.values() for observation in l]
		# TODO Need to take best alpha_a_o, not best alpha (maybe)
		obs_alphas = {observation: [self.alpha_a_o(alpha, action, observation) for alpha in self.v] for observation in
		              all_sampled_observations}
		# TODO remove impossible observation checking code to increase speed
		# possible_configs = []
		# impossible_configs = []
		# for i in range(len(all_sampled_observations)):
		# 	imposs = cstuff.is_observation_impossible(b,all_sampled_observations[i])
		# 	if imposs == 1:
		# 		impossible_configs.append((b,all_sampled_observations[i]))
		# obs_bs = {observation: self.pomdp.belief_update(b, observation, action) for observation in
		#           all_sampled_observations}
		obs_bs = {observation: b for observation in all_sampled_observations}
		max_alphas = {
			observation: argmax(obs_alphas[observation], lambda alpha: self.alpha_dot_b(alpha, obs_bs[observation])) for
			observation in all_sampled_observations}
		# max_alphas = {observation: argmax(self.v, lambda alpha: self.alpha_dot_b(
		# 	self.alpha_a_o(alpha, action, observation, possible_states), belief)) for
		#               observation in
		#               all_sampled_observations}
		# Need to add all max_alphas
		sum_alpha = cstuff.add_list_of_alphas(max_alphas.values(), self.pomdp.states)
		sum_alpha["values"] = add_dict(sum_alpha["values"], r_a, self.pomdp.states)
		sum_alpha["action"] = action
		# sum_alpha["values"] = cstuff.add_dict(sum_alpha,r_a,self.pomdp.states)
		# ret = add_dict(r_a, times_dict(max_alphas, self.pomdp.gamma))
		# ret.default_factory = lambda: a.default_factory() + b.default_factory()
		return sum_alpha

	def alpha_a_b_new(self, action, belief):
		'''
		:param action:
		:param belief:
		:return: a (non default) dict with keys in belief.possible_states and values as in shani (34)
		'''
		# TODO perhaps we should evaluate alpha_a_o against b_a_o. Shani and Pineau both say b, but that seems odd :/
		# Not a function of alpha. Shani (33) seems to have a typo, since we are taking argmax over alphas
		b = self.pomdp.belief_transition_func(belief, action)

		possible_states = b.get_all_plausible_states()
		# possible_states = belief.get_all_possible_states()
		# possible_states = self.pomdp.states
		observations_per_state = {state: math.ceil(self.observations_sample_size * b.belief(state)) for state in
		                          possible_states}
		# Consider storing r_a's based on belief.known for speedup
		r_a = {state: self.reward_func(state, action) for state in self.pomdp.states}
		observations_by_state = {
			state: [self.pomdp.sample_observation(state) for i in range(observations_per_state[state])] for state in
			possible_states}
		# all_sampled_observations might be broken
		all_sampled_observations = [observation for l in observations_by_state.values() for observation in l]
		# TODO Need to take best alpha_a_o, not best alpha (maybe)
		obs_alphas = {observation: [self.alpha_a_o(alpha, action, observation) for alpha in self.v] for observation in
		              all_sampled_observations}
		# TODO remove impossible observation checking code to increase speed
		# possible_configs = []
		# impossible_configs = []
		# for i in range(len(all_sampled_observations)):
		# 	imposs = cstuff.is_observation_impossible(b,all_sampled_observations[i])
		# 	if imposs == 1:
		# 		impossible_configs.append((b,all_sampled_observations[i]))
		obs_bs = {observation: self.pomdp.belief_update(b, observation, action) for observation in
		          all_sampled_observations}
		max_alphas = {
			observation: argmax(obs_alphas[observation], lambda alpha: self.alpha_dot_b(alpha, obs_bs[observation])) for
			observation in all_sampled_observations}
		# max_alphas = {observation: argmax(self.v, lambda alpha: self.alpha_dot_b(
		# 	self.alpha_a_o(alpha, action, observation, possible_states), belief)) for
		#               observation in
		#               all_sampled_observations}
		# Need to add all max_alphas
		sum_alpha = cstuff.add_list_of_alphas(max_alphas.values(), self.pomdp.states)
		sum_alpha["values"] = add_dict(sum_alpha["values"], r_a, self.pomdp.states)
		sum_alpha["action"] = action
		# sum_alpha["values"] = cstuff.add_dict(sum_alpha,r_a,self.pomdp.states)
		# ret = add_dict(r_a, times_dict(max_alphas, self.pomdp.gamma))
		# ret.default_factory = lambda: a.default_factory() + b.default_factory()
		return sum_alpha

	def alpha_a_o(self, alpha, action, observation, states=None):
		'''
		Currently assumes deterministic transitions. TODO: generalize for non deterministic transitions
		TODO cache alpha_a_o to use for multiple belief points
		:param alpha:
		:param action:
		:param observation:
		:param states:
		:return: componentwise product of alpha value at s and observation likelihood at s (since transitions are trivial)
		'''
		# Fixed to take into account transitions 2:07 PM 8/16/2018
		if states == None:
			states = self.pomdp.states
		target_states = [self.pomdp.transition_func(state, action) for state in states]
		new_alpha = {
			"values": {
				states[i]: alpha["values"][target_states[i]] * self.pomdp.observation_func(observation,
				                                                                           target_states[i],
				                                                                           action) for i
				in
				range(len(states))}, "action": action}
		return new_alpha

	def get_pick_alpha(self, des_id):
		alpha = {"values": {}, "action": "pick " + str(des_id)}
		alpha["values"].update({state: self.pomdp.correct_pick_reward for state in self.pomdp.states_by_des[des_id]})
		other_items = [i for i in range(self.pomdp.num_items) if i != des_id]
		for i in other_items:
			alpha["values"].update({state: self.pomdp.wrong_pick_cost for state in self.pomdp.states_by_des[i]})
		return alpha

	def get_horizon_0_alpha_from_action(self, action):
		alpha = {"values": {}, "action": action}
		alpha["values"].update({state: self.pomdp.reward_func(state, action) for state in self.pomdp.states})
		return alpha

	def get_lower_bound_alpha_from_action(self, action, conservative_lower_bounds=False):
		# Custom for Fetch
		alpha = {"values": {}, "action": action}
		if conservative_lower_bounds:
			values = {state: self.pomdp.reward_func(state, action) + self.pomdp.gamma * self.pomdp.wrong_pick_cost for
			          state in self.pomdp.states}
		else:
			values = {state: self.pomdp.reward_func(state, action) + self.pomdp.wait_cost / (1 - self.pomdp.gamma) for
			          state in self.pomdp.states}

		alpha["values"] = values
		return alpha

	def initialize_v(self):
		# currently custom designed for Fetch
		new_alphas = []
		# For each desired item, create an alpha with correct_pick_reward for any state with that desred item
		for des_id in range(self.pomdp.num_items):
			alpha = self.get_pick_alpha(des_id)
			new_alphas.append(alpha)
		return new_alphas

	def pickle_beliefs_and_alphas(self, name=None):
		if name is None:
			name = self.name + " pickle " + str(len(self.v)) + " alphas " + str(len(self.beliefs)) + "beliefs " + str(
				datetime.now()).replace(":", ".")[:22]
		p = {"alphas": self.v, "beliefs": self.beliefs, "pomdp config": self.pomdp.config}
		pickle.dump(p, open(pickle_location + name + ".pickle", "wb"))

	def load_from_pickle(self, p):
		self.beliefs = p["beliefs"]
		self.v = p["alphas"]

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
			action = alpha["action"]
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
					action = alpha["action"]

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


class PBVIClassic(PBVI):
	def __init__(self, pomdp, observations_sample_size=3):
		self.pomdp = pomdp
		self.reward_func = self.pomdp.reward_func
		self.v = self.initialize_v()
		self.beliefs = [pomdp.cur_belief]
		self.observations_sample_size = observations_sample_size
		self.belief_branching = 1
		self.name = "Classic PBVI"

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
			farthest_belief = argmax(successor_beliefs,
			                         lambda b: cstuff.distance(belief["desired_item"], b["desired_item"]))
			new_beliefs.append(farthest_belief)
		self.beliefs.extend(new_beliefs)
		return self.beliefs

	def update_v(self, n=3):
		# converged = False
		# TODO: write convergence test
		# while not converged:
		for i in range(n):
			for belief in self.beliefs:
				self.add_alpha(self.backup(belief, self.v))


class Perseus(PBVI):
	def __init__(self, pomdp, num_beliefs=100, belief_depth=3, observations_sample_size=3, beliefs=None, alphas=None,
	             convergence_threshold=1, conservative_lower_bounds=False):
		PBVI.__init__(self, pomdp=pomdp, observations_sample_size=observations_sample_size)
		self.conservative_lower_bounds = conservative_lower_bounds
		if beliefs is None:
			self.beliefs = []
			self.v = []
			self.beliefs = self.initialize_beliefs(num_beliefs, depth=belief_depth)
			self.pickle_beliefs_and_alphas()
		else:
			self.beliefs = beliefs
		if alphas is None:
			self.v = self.initialize_v()
			self.update_v(convergence_threshold=convergence_threshold)
		else:
			self.v = alphas
		self.name = "Perseus"
		self.pickle_beliefs_and_alphas()
		# else:
		# 	self.beliefs = data["beliefs"]
		# 	self.v = data["alphas"]
		# 	if data["pomdp_config"] != self.pomdp.config:
		# 		print("WARNING: Loaded beliefs and alphas come from pomdp with different configuration.")
		print("Perseus constructed!")

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
				self.beliefs.append(b)
		return self.beliefs

	def initialize_v(self):
		# currently custom designed for Fetch
		new_alphas = []
		# For each desired item, create an alpha with correct_pick_reward for any state with that desred item
		for des_id in range(self.pomdp.num_items):
			alpha = self.get_pick_alpha(des_id)
			new_alphas.append(alpha)
		# For non terminal actions, a lower bound value is the cost of the action + discounted wrong pick next turn.
		# TODO Consider better lower bounds as well
		non_terminal_actions = self.pomdp.get_non_terminal_actions()
		for action in non_terminal_actions:
			alpha = self.get_lower_bound_alpha_from_action(action, self.conservative_lower_bounds)
			new_alphas.append(alpha)
		self.v = new_alphas
		return new_alphas


def argmax(args, fn):
	max_value = fn(args[0])
	maxarg = args[0]
	for i in range(len(args)):
		cur_v = fn(args[i])
		if cur_v > max_value:
			maxarg = args[i]
			max_value = cur_v
	return maxarg


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


def test_alpha_a_b():
	pomdp = FetchPOMDP()
	vi = PBVIClassic(pomdp)
	vi.alpha_a_b(alpha=None, action="point 1", belief=pomdp.cur_belief)


def test_alpha_dot_belief():
	pomdp = FetchPOMDP()
	vi = PBVIClassic(pomdp)


# belief = pomdp.cur_belief
# possible_states = belief.get_all_possible_states()
# print("cur_belief: "  + str(pomdp.cur_belief))
# print("cur_state: " + str(pomdp.cur_state))
# print(possible_states[0])
# print(possible_states[0].data["last_referenced_item"])
# print("test to_state")
# print(pomdp.cur_belief.to_state(0))
# print(pomdp.cur_belief.to_state(0)["last_referenced_item"])


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


# test_dict_operations()
def test_general():
	pomdp = FetchPOMDP()
	pb = PBVIClassic(pomdp=pomdp)
	print(pb.get_value(pomdp.cur_belief))
	bs = copy.deepcopy(pomdp.cur_belief)
	bs["desired_item"] = [0.0 for i in range(len(bs["desired_item"]))]
	bs["desired_item"][pomdp.cur_state["desired_item"]] = 1.0
	possible_states = bs.get_all_possible_states()
	alpha = {"values": defaultdict(lambda: pomdp.min_value, {state: 100 for state in possible_states}),
	         "action": "wait"}
	pb.v.append(alpha)
	print(pb.get_value(pomdp.cur_belief))
	print(pb.alpha_dot_b(alpha, bs))


# for i in range(100):
# 	print(" ".join([str(x) for x in pb.beliefs]))
# 	print(" ".join([str(x) for x in pb.v]))
# 	print(pb.get_value(pomdp.cur_belief))
# 	alpha, val = pb.get_best_alpha(pomdp.cur_belief)
# 	print(alpha["action"] + " has value " + str(val))
# 	pb.update_v()
# 	pb.collect_beliefs()
def test_alpha_dot_b_and_get_value():
	pomdp = FetchPOMDP()
	pb = PBVIClassic(pomdp=pomdp)
	print(pb.get_value(pomdp.cur_belief))
	bs = copy.deepcopy(pomdp.cur_belief)
	bs["desired_item"] = [0.0 for i in range(len(bs["desired_item"]))]
	bs["desired_item"][pomdp.cur_state["desired_item"]] = 1.0
	possible_states = bs.get_all_possible_states()

	alpha = {"values": {state: 100 for state in pomdp.states}, "action": "wait"}
	pb.v.append(alpha)
	print(pb.get_value(pomdp.cur_belief))
	print(pb.alpha_dot_b(alpha, bs))


def test_alpha_a_o():
	pomdp = FetchPOMDP()
	pb = Perseus(pomdp)

	b = pomdp.init_belief
	a = "point 1"
	o = FetchPOMDPObservation({"yes"}, None)
	alpha = pb.get_pick_alpha(1)
	# states = pomdp.get_all_possible_states(candidate_desired_items= [1])
	# alpha["values"].extend({state: pomdp.correct_pick_reward for state in states})
	# s = FetchPOMDPState(1,1,"point")
	# alpha["values"][s] = pomdp.correct_pick_reward
	# alpha["values"][s] = pomdp.correct_pick_reward
	alpha2 = pb.alpha_a_o(alpha, a, o)
	b1 = copy.deepcopy(b)
	b1["last_referenced_item"] = 1
	b1["reference_type"] = "point"
	b1 = pomdp.belief_update(b1, o)
	print(pb.get_value(b))
	print(pb.get_value(b1))


def test_alpha_a_b():
	pomdp = FetchPOMDP()
	pb = Perseus(pomdp)
	b = pomdp.init_belief
	a = "point 1"
	o = pomdp.sample_observation_from_belief_action(b, a)
	bao = pomdp.belief_update(b, o)
	alpha_a_b = pb.alpha_a_b(a, b)
	val = pb.alpha_dot_b(alpha_a_b, b)
	val2 = pb.alpha_dot_b(alpha_a_b, bao)
	print(val)
	print(val2)


def pomdp_to_defaultdict(pomdp):
	return defaultdict(lambda: pomdp.min_value)


def run_perseus(n=1):
	# Check each each function in PBVI/Perseus from the bottom up
	pomdp = FetchPOMDP()
	pb = Perseus(pomdp, num_beliefs=100)
	for i in range(n):
		alpha, value = pb.get_best_alpha(pomdp.cur_belief)
		action = alpha["action"]
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
				action = alpha["action"]
			num_iter += 1

		print("Took " + str(num_iter) + " steps to get reward " + str(discounted_sum_rewards))
		pomdp.reset()


def test_get_pick_alpha():
	pomdp = FetchPOMDP()
	pb = Perseus(pomdp)
	alpha = pb.get_pick_alpha(1)
	print(alpha)


def test_add_dict():
	ss = [FetchPOMDPState(1, i, "point") for i in range(4)]
	a = {ss[i]: i for i in range(4)}
	b = {ss[i]: i ^ 2 for i in range(4)}
	c = add_dict(a, b)
	print(c[ss[2]])


def test_get_best_alpha():
	# pass 4:24 pm 8/17/2018
	pomdp = FetchPOMDP()
	pb = Perseus(pomdp)
	alpha = pb.get_pick_alpha(1)
	b = pomdp.init_belief
	s = b.sample()
	alpha["values"][s] = 1000
	pb.v.append(alpha)
	best_alpha, best_value = pb.get_best_alpha(b)
	print("best_value: " + str(best_value))
	if best_alpha == alpha:
		print("Correct")
	print(pb.get_value(b))
	print(pb.alpha_dot_b(best_alpha, b))
	print(pb.alpha_dot_b(alpha, b))


def make_observation_serializable(o):
	o2 = {"language": list(o["language"]), "gesture": o["gesture"]}


# print(pb.v)
# print(pb.get_value(pomdp.cur_belief))
def pickle_test():
	pomdp = FetchPOMDP()
	b = pomdp.init_belief
	o = pomdp.sample_observation_from_belief(b)
	alpha = {"values": {state: 12 for state in pomdp.states}, "action": "wait"}
	p = {"b": b, "o": o, "alpha": alpha, "pomdp_config": pomdp.get_config()}
	current_time = str(datetime.now()).replace(":", ".")[:22]
	pickle.dump(p, open("pickled thing " + current_time, "wb"))
	p2 = pickle.load(open("pickled thing " + current_time, "rb"))
	print(p2)
# test_alpha_a_o()
# test_alpha_a_b()
# test_get_pick_alpha()
# test_add_dict()
# test_get_best_alpha()
# pickle_test()
