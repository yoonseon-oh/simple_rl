from time import time
import copy
import cython
# import pyximport;
# pyximport.install()
# import simple_rl.tasks.FetchPOMDP.cstuff as cstuff
from simple_rl.tasks.FetchPOMDP import cstuff
from simple_rl.tasks.FetchPOMDP.FetchStateClass import FetchPOMDPObservation
import random

random.seed(0)


class FetchPOMDPSolver(object):
	# Trades off between point and wait, but not point and look when adjusting std_dev
	# TODO Try a small example and check that this works as expected
	def __init__(self, pomdp, horizon=2, qvalue_method="state based", kl_weight=0, observation_branching=1,
	             planner="q estimation", muted=False):
		self.pomdp = pomdp
		self.num_state_samples = pomdp.num_items
		self.horizon = horizon
		self.muted = muted
		self.kl_weight = kl_weight
		if qvalue_method == "state based":
			# self.get_qvalues = self.get_qvalues_from_state
			self.get_reward_sim = lambda b, s, a: self.pomdp.reward_func(s, a)
		elif qvalue_method == "belief based":
			# self.get_qvalues = self.get_qvalues_from_belief
			self.get_reward_sim = lambda b, s, a: self.pomdp.get_expected_reward(b, a)
		# self.belief_update = cstuff.belief_update
		self.belief_update_robot = cstuff.belief_update_robot
		self.sample_observation = self.pomdp.sample_observation
		self.belief_update = self.pomdp.belief_update
		if planner == "q estimation":
			self.plan = self.plan_from_belief
		elif planner == "heuristic":
			self.plan = self.plan_from_belief_heuristically
		# self.get_qvalues = self.get_qvalues_kl
		self.get_qvalues = self.get_qvalues_many_trials
		self.observation_branching = observation_branching

	def plan_from_belief(self, belief_state, num_samples = None):
		if num_samples is None:
			num_samples = self.pomdp.num_items
		#Sampling extraordinarily unlikely states leads to impossible observations (probably)
		# sampled_states = cstuff.sample_distinct_states(belief_state[1], self.num_state_samples)
		# sampled_states = [belief_state.to_state(i) for i in range(self.pomdp.num_items)]
		# sampled_states = [belief_state.sample() for i in range(self.pomdp.num_items)]
		sampled_states = belief_state.get_all_plausible_states()
		list_of_q_lists = [self.get_qvalues(belief_state, state, self.horizon) for state in sampled_states]
		#If states are actually sampled, consider replacing weights
		weights = cstuff.unit_vectorn([belief_state["desired_item"][state["desired_item"]] for state in sampled_states])
		average_q_values = []
		max_q = -10000
		best_action_ids = [0]
		for i in range(len(self.pomdp.actions)):
			q = 0
			for j in range(len(sampled_states)):
				q += weights[j] * list_of_q_lists[j][i]
			average_q_values.append(q)
			if q == max_q:
				best_action_ids.append(i)
			elif q > max_q:
				best_action_ids = [i]
				max_q = q
		next_action_id = random.sample(best_action_ids, 1)[0]
		next_action = self.pomdp.actions[next_action_id]

		if not self.muted:
			qvs = ""
			for i in range(len(self.pomdp.actions)):
				qvs += self.pomdp.actions[i] + ": " + str(round(average_q_values[i], 4)) + ", "
			print("Current belief_state: " + str(belief_state))
			print("q values: " + qvs)
			print("Plan to take action (" + str(next_action) + ") with q value = " + str(
				round(average_q_values[next_action_id], 4)))
			split_action = next_action.split(" ")
			if split_action[0] == "pick":
				print(str(belief_state["desired_item"][int(next_action.split(" ")[1])]) + " sure")
		return next_action

	def plan_from_belief_heuristically(self, belief_state):
		most_likely_item = belief_state.get_most_likely()
		pick_action = "pick " + str(most_likely_item[0])
		point_action = "point " + str(most_likely_item[0])
		pick_reward = self.pomdp.get_expected_reward(belief_state, pick_action)
		if pick_reward > self.pomdp.gamma * self.pomdp.correct_pick_reward:
			return pick_action
		elif most_likely_item[1] >= 1.5 / len(belief_state["desired_item"]):
			return point_action
		else:
			return "wait"

	def get_qvalues(self, belief_state, true_state, horizon):
		actions = self.pomdp.actions
		rewards = [self.get_reward_sim(belief_state, true_state, a) for a in actions]
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		# Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [self.belief_update(belief_state, o) for o in observations]
		next_qvalues = [self.get_qvalues(next_beliefs[i], next_states[i],
		                                 horizon - 1) if i not in terminal_states else 0.0 for i in
		                range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues_many_trials(self, belief_state, true_state, horizon, num_trials=2):
		# Should try only value kl at the deepest level
		actions = self.pomdp.actions
		rewards = [self.get_reward_sim(belief_state, true_state, a) for a in actions]
		if horizon == 0:
			return rewards
		average_next_values = [0 for a in actions]
		# transitions are deterministic, so we can pull the next two lines out of the loop
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		next_beliefs = [copy.deepcopy(belief_state).update_from_state(next_state) for next_state in next_states]
		for trial in range(self.observation_branching):
			# Generalize for general BSS
			terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
			observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
			#Check for impossible observations. Remove to increase speed significantly
			# possible_configs = []
			# impossible_configs = []
			# for i in range(len(observations)):
			# 	imposs = cstuff.is_observation_impossible(next_beliefs[i],observations[i])
			# 	if imposs == 1:
			# 		impossible_configs.append((next_beliefs[i],observations[i],next_states[i]))
			# 	else:
			# 		possible_configs.append((next_beliefs[i],observations[i],next_states[i]))
			# num_impossible = len(impossible_configs)
			next_beliefs = [self.belief_update(next_beliefs[i], observations[i]) for i in range(len(next_states))]
			for b in next_beliefs:
				for i in range(len(b["desired_item"])):
					if b["desired_item"][i] <= 0:
						print(str(b))
			# print(next_beliefs[0]["desired_item"])
			next_qvalues = [self.get_qvalues(next_beliefs[i], next_states[i],
			                                 horizon - 1) if i not in terminal_states else 0.0 for i in
			                range(len(next_states))]
			next_values = [cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]
			average_next_values = cstuff.add(average_next_values, next_values)
		average_next_values = [average_next_values[i] /self.observation_branching for i in range(len(average_next_values))]
		return [
			rewards[i] + self.pomdp.gamma * average_next_values[i] for i in range(len(next_states))]

	def get_qvalues_kl(self, belief_state, true_state, horizon):
		# Should try only value kl at the deepest level
		actions = self.pomdp.actions
		rewards = [self.get_reward_sim(belief_state, true_state, a) for a in actions]
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		next_beliefs = [copy.deepcopy(belief_state).update_from_state(next_state) for next_state in next_states]
		# Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [self.belief_update(next_beliefs[i], observations[i]) for i in range(len(next_states))]
		for b in next_beliefs:
			for i in range(len(b["desired_item"])):
				if b["desired_item"][i] <= 0:
					print(str(b))
		# print(next_beliefs[0]["desired_item"])
		kls = [cstuff.kl_divergence(next_beliefs[i]["desired_item"], belief_state["desired_item"]) for i in
		       range(len(next_states))]
		next_qvalues = [self.get_qvalues(next_beliefs[i], next_states[i],
		                                 horizon - 1) if i not in terminal_states else 0.0 for i in
		                range(len(next_states))]
		return [
			rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) + cstuff.clamp(self.kl_weight * kls[i], 0,
			                                                                              self.pomdp.max_value) for i in
			range(len(next_states))]

	def get_qvalues2Dict(self, b, true_state, horizon):
		'''
		Unfinished. Meant to treat forward search tree as nested dictionaries instead of lists, which would allow
		action set restriction.
		:param b:
		:param true_state:
		:param horizon:
		:return:
		'''
		actions = self.pomdp.actions
		rewards = {a: self.pomdp.get_reward_from_state(true_state, a) for a in self.pomdp.actions}
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		# Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [self.belief_update(b, o) for o in observations]
		next_qvalues = [self.get_qvalues_from_state(next_beliefs[i], next_states[i],
		                                            horizon - 1) if i not in terminal_states else 0.0 for i in
		                range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	# def get_qvalues_kl(self, b, true_state, horizon):
	# 	rewards = [self.pomdp.get_reward_from_state(true_state,a) for a in self.pomdp.actions]
	# 	if horizon == 0:
	# 		return rewards
	# 	actions = self.pomdp.actions
	# 	next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
	# 	#Generalize for general BSS
	# 	terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
	# 	observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
	# 	next_beliefs = [cstuff.belief_update(b,o) for o in observations]
	# 	kl_divs = [cstuff.kl_divergence(b,next_belief) for next_belief in next_beliefs]
	# 	rewards = [rewards[i] + info_value*kl_divs[i] for i in range(len(rewards))]
	# 	next_qvalues = [self.get_qvalues_from_belief(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(len(next_states))]
	# 	return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def run(self, num_episodes=5):
		# Differes from run by getting reward from mdp state in simulation
		# TODO: Save entire history (not simulation)
		num_correct = 0
		num_wrong = 0
		plan = self.plan
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
			action = plan(curr_belief_state)
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
				print("Reward: " + str(reward))
				print("Observation: "  +str(observation))
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
					action = plan(curr_belief_state)
					counter_plan_from_state += 1
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

	def run_robot(self):
		plan = self.plan_from_belief_and_observation
		discounted_sum_rewards = 0.0
		num_iter = 0
		if not self.muted:
			print(" ")
		self.pomdp.reset()
		mixed_belief = self.pomdp.get_curr_belief()
		o = self.pomdp.get_observation()
		action = plan(mixed_belief, o)
		running = True
		while running:
			if self.pomdp.is_terminal(mixed_belief, action):
				running = False
			# print("Terminal action: " + str(action))
			# execute_start_time = time()
			ret = self.pomdp.execute_action(action)
			# print("Execute time = "+ str(time() - execute_start_time))
			reward = ret[0]
			next_mixed_belief = ret[1]
			# print_times()
			discounted_sum_rewards += ((self.pomdp.gamma ** num_iter) * reward)
			if not self.muted:
				print('({}, {}, {}) -> {} | {}'.format(mixed_belief, action, next_mixed_belief, reward,
				                                       discounted_sum_rewards))
				print("")
			# print_times()
			mixed_belief = copy.deepcopy(next_mixed_belief)
			current_history = {"mixed_belief": next_mixed_belief}
			if running:
				action = plan(mixed_belief)
				current_history["action"] = action
			else:
				current_history["action"] = "Fin"

	# def receive_observation(self, o):
	# 	self.pomp.cur_belief = self.belief_update_robot(self.pomdp.cur_belief,o)
	def act(self, raw_observation):
		gesture = raw_observation[0]
		if gesture is not None:
			gesture = [gesture[0], gesture[1], gesture[2], gesture[3], gesture[4], gesture[5]]
		language = raw_observation[1]
		if language is not None:
			language = set(raw_observation[1].split(" "))
		else:
			language = set()
		observation = FetchPOMDPObservation(**{"language": language, "gesture": gesture})
		self.pomdp.curr_belief_state = cstuff.belief_update_robot(self.pomdp.curr_belief_state, observation)
		next_action = self.plan_from_belief(self.pomdp.curr_belief_state)
		self.pomdp.execute_action_robot(next_action)
		return next_action

	def plan_from_belief_and_observation(self, b, o):
		b2 = self.belief_update(b, o)
		return self.plan_from_belief(b2)


def get_most_likely(b):
	pd = b[1]
	most_likely_index = 0
	highest_probability = 0
	for i in range(len(pd)):
		if pd[i] > highest_probability:
			highest_probability = pd[i]
			most_likely_index = i
	return [most_likely_index, highest_probability]


# def test_plan_from_observation
def make_observation_serializable(o):
	o2 = {"language": list(o["language"]), "gesture": o["gesture"]}
