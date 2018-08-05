from time import time
import copy
import cython
# import pyximport;
# pyximport.install()
# import simple_rl.tasks.FetchPOMDP.cstuff as cstuff
from simple_rl.tasks.FetchPOMDP import cstuff

class FetchPOMDPSolver(object):
	def __init__(self,pomdp, horizon = 2, qvalue_method = "state based"):
		self.pomdp = pomdp
		self.num_state_samples = pomdp.num_items
		self.horizon = horizon
		self.muted = True
		if qvalue_method == "state based":
			self.get_qvalues = self.get_qvalues_from_state
		elif qvalue_method == "belief based":
			self.get_qvalues = self.get_qvalues_from_belief

	def plan_from_belief(self, b):
		sampled_states = cstuff.sample_states(b[1],self.num_state_samples)
		list_of_q_lists = [self.get_qvalues(b, {"desired_item":s, "last_referenced_item":b[0]}, self.horizon) for s in sampled_states]
		weights = cstuff.unit_vectorn([b[1][i] for i in sampled_states])
		average_q_values = []
		max_q = -10000
		best_action_id = 0
		for i in range(len(self.pomdp.actions)):
			q = 0
			for j in range(len(sampled_states)):
				q += weights[j]*list_of_q_lists[j][i]
			average_q_values.append(q)
			if q > max_q:
				best_action_id = i
				max_q = q
		next_action = self.pomdp.actions[best_action_id]

		if not self.muted:
			qvs = ""
			for i in range(len(self.pomdp.actions)):
				qvs += self.pomdp.actions[i] + ": " + str(round(average_q_values[i], 4)) + ", "
			print("Current b: " + str(b))
			print("q values: " + qvs)
			print("Plan to take action (" + str(next_action) + ") with q value = " + str(round(average_q_values[best_action_id], 4)))
			split_action = next_action.split(" ")
			if split_action[0] == "pick":
				print(str(b[1][int(next_action.split(" ")[1])]) + " sure")
		return next_action


	def get_qvalues_from_belief(self, b, true_state, horizon):
		actions = self.pomdp.actions
		rewards = [self.pomdp.get_reward(b,a) for a in actions]
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
		#Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [cstuff.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [cstuff.belief_update(b,o) for o in observations]
		next_qvalues = [self.get_qvalues_from_belief(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues_from_state(self, b, true_state, horizon):
		actions = self.pomdp.actions
		rewards = [self.pomdp.get_reward_from_state(true_state,a) for a in actions]
		if horizon == 0:
			return rewards
		actions = self.pomdp.actions
		next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
		#Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [cstuff.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [cstuff.belief_update(b,o) for o in observations]
		next_qvalues = [self.get_qvalues_from_state(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues2Dict(self, b, true_state, horizon):
		actions = self.pomdp.actions
		rewards = {a:self.pomdp.get_reward_from_state(true_state,a) for a in self.pomdp.actions}
		if horizon == 0:
			return rewards
		next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
		#Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [cstuff.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [cstuff.belief_update(b,o) for o in observations]
		next_qvalues = [self.get_qvalues_from_state(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def get_qvalues_kl(self, b, true_state, horizon):
		rewards = [self.pomdp.get_reward_from_state(true_state,a) for a in self.pomdp.actions]
		if horizon == 0:
			return rewards
		actions = self.pomdp.actions
		next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
		#Generalize for general BSS
		terminal_states = [i for i in range(len(actions)) if actions[i].split(" ")[0] == "pick"]
		observations = [cstuff.sample_observation(next_states[i]) for i in range(len(next_states))]
		next_beliefs = [cstuff.belief_update(b,o) for o in observations]
		kl_divs = [cstuff.kl_divergence(b,next_belief) for next_belief in next_beliefs]
		rewards = [rewards[i] + info_value*kl_divs[i] for i in range(len(rewards))]
		next_qvalues = [self.get_qvalues_from_belief(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(len(next_states))]
		return [rewards[i] + self.pomdp.gamma * cstuff.maxish(next_qvalues[i]) for i in range(len(next_states))]

	def run(self, num_episodes=5):
		#Differes from run by getting reward from mdp state in simulation
		#TODO: Save entire history (not simulation)
		plan = self.plan_from_belief
		start_time = time()
		final_scores = []
		counter_plan_from_state = 1
		history = []
		for episode in range(num_episodes):
			discounted_sum_rewards = 0.0
			num_iter = 0
			if not self.muted:
				print(" ")
				print('Episode {}: '.format(episode))
			self.pomdp.reset()
			mixed_belief = self.pomdp.get_mixed_belief()
			action = plan(mixed_belief)
			current_history = {"mixed_belief":mixed_belief, "action": action}
			history.append(current_history)
			running = True
			while running:
				if self.pomdp.is_terminal(mixed_belief, action):
					running = False
				# print("Terminal action: " + str(action))
				# execute_start_time = time()
				ret = self.pomdp.execute_action(action)
				# print("Execute time = "+ str(time() - execute_start_time))
				reward = ret[0]
				next_state = ret[1]
				# print_times()
				discounted_sum_rewards += ((self.pomdp.gamma ** num_iter) * reward)
				if not self.muted:
					print('({}, {}, {}) -> {} | {}'.format(mixed_belief, action, next_state, reward, discounted_sum_rewards))
					print("")
				# print_times()
				mixed_belief = copy.deepcopy(next_state)
				current_history = {"mixed_belief": next_state}
				if running:
					action = plan(mixed_belief)
					counter_plan_from_state += 1
					current_history["action"] = action
				else:
					current_history["action"] = "Fin"
				# print_times()
				num_iter += 1
				history.append(current_history)
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
		return final_scores, counter_plan_from_state, history