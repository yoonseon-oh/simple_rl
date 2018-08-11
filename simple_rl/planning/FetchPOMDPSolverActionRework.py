from time import time
import copy
import cython
# import pyximport;
# pyximport.install()
# import simple_rl.tasks.FetchPOMDP.cstuff as cstuff
from simple_rl.tasks.FetchPOMDP import cstuff
import random


class FetchPOMDPSolver(object):
	def __init__(self, pomdp, horizon=2, qvalue_method="state based", use_gesture=True, use_language=True,
	             planner="q estimation"):
		self.pomdp = pomdp
		self.num_state_samples = pomdp.num_items
		self.horizon = horizon
		self.muted = True
		if qvalue_method == "state based":
			self.get_qvalues = self.get_qvalues_from_state
		elif qvalue_method == "belief based":
			self.get_qvalues = self.get_qvalues_from_belief
		self.use_gesture = use_gesture
		self.belief_update = cstuff.belief_update
		self.belief_update_robot = cstuff.belief_update_robot
		# if use_gesture:
		# 	if use_language:
		# 		self.sample_observation = cstuff.sample_observation_detailed
		# 	else:
		# 		self.sample_observation = lambda s: {"language": None, "gesture": cstuff.sample_gesture(s)}
		# else:
		# 	if use_language:
		# 		self.sample_observation = lambda s: {"language": cstuff.sample_language(s), "gesture": None}
		# 	else:
		# 		self.sample_observation = lambda s: {"language": None, "gesture": None}
		# 		self.belief_update = lambda b, o: b
		# 		print("Using neither language nor gesture.")
		self.sample_observation = self.pomdp.sample_observation
		self.belief_update = self.pomdp.belief_update
		if planner == "q estimation":
			self.plan = self.plan_from_belief
		elif planner == "heuristic":
			self.plan = self.plan_from_belief_heuristically

	def plan_from_belief(self, b):
		sampled_states = cstuff.sample_distinct_states(b[1], self.num_state_samples)
		list_of_q_lists = [self.get_qvalues(b, [s, b[0][0],b[0][1]], self.horizon) for s in sampled_states]
		weights = cstuff.unit_vectorn([b[1][i] for i in sampled_states])
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
			print("Current b: " + str(b))
			print("q values: " + qvs)
			print("Plan to take action (" + str(next_action) + ") with q value = " + str(
				round(average_q_values[next_action_id], 4)))
			split_action = next_action.split(" ")
			if split_action[0] == "pick":
				print(str(b[1][int(next_action.split(" ")[1])]) + " sure")
		return next_action

	def plan_from_belief_heuristically(self, b):
		most_likely_item = get_most_likely(b)
		pick_action = "pick " + str(most_likely_item[0])
		point_action = "point " + str(most_likely_item[0])
		pick_reward = self.pomdp.get_expected_reward(b, pick_action)
		if pick_reward > self.pomdp.gamma * self.pomdp.correct_pick_reward:
			return pick_action
		elif most_likely_item[1] >= 1.5 / len(b[1]):
			return point_action
		else:
			return "wait"

	def get_qvalues_from_belief(self, b, true_state, horizon):
		actions = self.pomdp.actions
		rewards = [self.pomdp.get_expected_reward(b, a) for a in actions]
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		# Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [self.belief_update(b, o) for o in observations]
		next_qvalues = [self.get_qvalues_from_belief(next_beliefs[i], next_states[i],
		                                             horizon - 1) if i not in terminal_states else 0.0 for i in
		                range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues_from_state(self, b, true_state, horizon):
		actions = self.pomdp.actions
		rewards = [self.pomdp.get_reward_from_state(true_state, a) for a in actions]
		if horizon == 0:
			return rewards
		actions = self.pomdp.actions
		next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
		# Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [self.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [self.belief_update(b, o) for o in observations]
		next_qvalues = [self.get_qvalues_from_state(next_beliefs[i], next_states[i],
		                                            horizon - 1) if i not in terminal_states else 0.0 for i in
		                range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues2Dict(self, b, true_state, horizon):
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
			curr_belief_state = self.pomdp.get_curr_belief()
			if curr_belief_state[0][1] in ["point","look"]:
				raise ValueError("Belief is messed up: " + str(b[0]))
			action = plan(curr_belief_state)
			counter_plan_from_state +=1
			history = []
			running = True
			while running:
				if self.pomdp.is_terminal(curr_belief_state, action):
					running = False
				split_action = action.split(" ")
				if split_action[0] == "pick":
					if split_action[1] == str(self.pomdp.curr_state[0]):
						num_correct += 1
					else:
						num_wrong += 1
				#True state used for record keeping and is NOT used during planning
				true_state = self.pomdp.get_curr_state()
				ret = self.pomdp.execute_action(action)
				# Consider moving belief management to solver
				reward = ret[0]
				next_belief_state = ret[1]
				observation = ret[2]
				if type(curr_belief_state) is list:
					raise TypeError(
						"curr_belief_state has type list on iteration " + str(num_iter) + " of episode " + str(
							episode) + ": " + str(curr_belief_state))

				history.append({"belief": curr_belief_state.data, "action": action,
				                "observation": make_observation_serializable(observation),
				                "reward": reward,"true state":true_state.data})
				discounted_sum_rewards += ((self.pomdp.gamma ** num_iter) * reward)
				if not self.muted:
					print('({}, {}, {}) -> {} | {}'.format(curr_belief_state, action, next_belief_state, reward,
					                                       discounted_sum_rewards))
					print("")
				curr_belief_state = copy.deepcopy(next_belief_state)
				if type(curr_belief_state) is list:
					raise TypeError(
						"curr_belief_state has type list on iteration " + str(num_iter) + " of episode " + str(
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
	# 	self.pomp.curr_belief_state = self.belief_update_robot(self.pomdp.curr_belief_state,o)
	def act(self, raw_observation):
		gesture = raw_observation[0]
		if gesture is not None:
			gesture = [gesture[0], gesture[1], gesture[2], gesture[3], gesture[4], gesture[5]]
		language = raw_observation[1]
		if language is not None:
			language = set(raw_observation[1].split(" "))
		else:
			language = set()
		observation = {"language": language, "gesture": gesture}
		self.pomdp.curr_belief_state = cstuff.belief_update_robot(self.pomdp.curr_belief_state, observation)
		next_action = self.plan_from_belief(self.pomdp.curr_belief_state)
		self.pomdp.execute_action_robot(next_action)
		return next_action

	def plan_from_belief_and_observation(self, b, o):
		b2 = self.belief_update(b, o)
		return self.plan_from_belief(b2)


# def test_blind():
# 	from simple_rl.tasks.FetchPOMDP import FetchPOMDP
# 	pomdp = FetchPOMDP(use_language=False,use_gesture=False)
# 	solver = FetchPOMDPSolver(pomdp,2,"state based", False,False)
# 	o = solver.sample_observation(pomdp.init_state)
# 	b = pomdp.curr_belief_state
# 	b1 = solver.belief_update(b,o)
# 	print("o: " + str(o))
# 	print("b: " + str(b))
# 	print("b1: " + str(b1))
# test_blind()
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
