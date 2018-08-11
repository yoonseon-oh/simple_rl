# from config1 import *
import math
from time import time
import json
import random
import numpy as np
# from libcpp.vector cimport vector
import os
import sys
from simple_rl.tasks.FetchPOMDP import file_reader as fr

# import file_reader as fr
# import simple_rl.tasks.FetchPOMDP.file_reader as fr

# from simple_rl.tasks.FetchPOMDP.config_reader import load_json
# '''
# Run "python setup.py build_ext --inplace" to compile. Not currently needed since it is being imported with:
# import pyximport;
# pyximport.install()
# from simple_rl.tasks.FetchPOMDP import cstuff
#
# However, there is another method of import that would require manual compiling.
#
# If you get a build or link error, try deleting the ".pyxbld" folder. On Windows, this is located in "C:\Users\[user]".
# '''

cdef obs_sampling_time = 0
cdef estimate_qs_counter = 0
cdef estimate_qs_total_time = 0
cdef v_total_time = 0
cdef generate_total_time = 0
cdef belief_update_total_time = 0
cdef observation_func_total_time = 0
cdef gesture_func_total_time = 0
cdef sample_gesture_total_time = 0
# print(os.getcwd()+"\config.json")

# cdef load_json(file_name):
# 	# return {"items":["cup"]}
# 	with open(file_name) as json_data:
# 		return json.load(json_data)
# cdef load_json_from_while_imported(file_name):
# 	#Doesn't work - __file__ not defined for c modules.
# 	#Want to get FetchPOMDP directory. Currently gets directory of file that imported this file
# 	with open(os.path.join(sys.path[0], file_name)) as json_data:
# 			return json.load(json_data)

# cdef config = load_json(os.getcwd()+"\config.json")
cdef config = fr.load_json("config.json")
cdef bag_of_words = config["bag_of_words"]
cpdef items = config["items"]
cdef ATTRIBUTES = config["attributes"]
cdef num_items = len(items)
cdef double std_theta_look = config["std_theta_look"]
cdef double point_cost = config["point_cost"]
cdef double p_g = config["p_g"]
cdef double p_l = config["p_l"]  # split into base and response probabilities
cdef double p_r_match = config["p_r_match"]
cdef double p_r_match_look = config["p_r_match_look"]
cdef double alpha = config["alpha"]
cdef double std_theta = config["std_theta"]
cdef double gamma = config["gamma"]
cdef error = 1
import os

cdef positive_responses = set(config["positive_responses"])
cdef negative_responses = set(config["negative_responses"])

# Globals defined for speed
cdef std_theta_1 = 2 * (std_theta ** 2)
cdef std_theta_p_g = p_g / math.sqrt(2 * math.pi * (std_theta ** 2))
cpdef get_items():
	return items
cpdef union_dictionary(dictionary):
	un = set()
	for value in dictionary.values():
		un.update(value)
	return un

cpdef get_relevant_words(item_index, bag):
	words = set()
	item = items[item_index]  # parameter is int, we want the dict
	for att in ATTRIBUTES:
		words.update(bag[item[att]])
	return words

cpdef get_irrelevant_words(item_index, bag):
	cdef dict item = items[item_index]  # fed int, want dict
	words = set()
	keys = set(bag.keys())
	for att in ATTRIBUTES:
		keys.remove(item[att])
	for key in keys:
		words.update(bag[key])
	return words

cdef all_words = union_dictionary(bag_of_words)
cdef relevant_words = [get_relevant_words(i, bag_of_words) for i in range(len(items))]
cdef irrelevant_words = [get_irrelevant_words(i, bag_of_words) for i in range(len(items))]

cpdef double sum(list a):
	cdef double total = 0
	cdef int i
	for i in range(len(a)):
		total += a[i]
	return total
cpdef list subtract(list a, list b):
	return [a[i] - b[i] for i in range(len(a))]

cpdef double dot(list a, list b):
	cdef double sum = 0
	for i in range(len(a)):
		sum += a[i] * b[i]
	return sum

cpdef double dotn(list a, list b, int n):
	cdef double sum = 0
	cdef int i
	for i in range(n):
		sum += a[i] * b[i]
	return sum

cpdef double dot3(list a, list b):
	cdef double sum = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
	return sum

cpdef double dot3v2(list a, list b):
	return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

cdef divide(list lst, double denom):
	cdef int i
	return [lst[i] / denom for i in range(len(lst))]

cpdef unit_vector(list a):
	cdef double den = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
	return [a[0] / den, a[1] / den, a[2] / den]

cpdef unit_vectorn(list a):
	cdef double den = 0
	cdef int i
	for i in range(len(a)):
		den += a[i]
	return [a[i] / den for i in range(len(a))]

cpdef double angle_between(list v1, list v2):
	v1_u = unit_vector(v1)
	v2_u = unit_vector(v2)
	cdef double prod = dot3(v1_u, v2_u)
	#Find faster way to clip
	if prod < -1:
		prod = -1
	elif prod > 1:
		prod = 1
	return math.acos(prod)

cpdef double vec_prob(list ideal, list actual):
	return std_theta_p_g * (math.e ** (-(angle_between(ideal, actual) ** 2) / std_theta_1))

cpdef belief_update(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	global belief_update_total_time
	start = time()
	observation_probs = [
		observation_func(observation,belief_state.to_state(i)) for i
		in
		range(len(belief_state["desired_item"]))]
	denominator = dot(belief_state["desired_item"], observation_probs)
	if denominator == 0:
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		raise ValueError("Received observation with probability 0")
		return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	belief_update_total_time += time() - start
	return ret
cpdef belief_update_old(b, o):
	"""
	:param b: FetchPOMDPBeliefState
	:param o: [set of words, gesture vector]
	:return: [known part of state, belief distribution]
	"""
	global belief_update_total_time
	start = time()
	# if (o["language"] is None or o["language"] == set()) and o["gesture"] is None:
	# 	belief_update_total_time += time() - start
	# 	return b
	observation_probs = [observation_func(o, [i, b[0][0], b[0][1]]) for i in
	                     range(len(b[1]))]
	denominator = dot(b[1], observation_probs)
	if denominator == 0:
		print("b[2] dot observation_probs = 0")
		print("b = " + str(b))
		print("observation_probs = " + str(observation_probs))
		print("o = " + str(o))
		return b
	ret = [[b[0][0], b[0][1]],
	       [b[1][j] * observation_probs[j] / denominator for j in range(len(b[1]))]]  #Replace with c
	belief_update_total_time += time() - start
	return ret
cpdef belief_update_robot(belief_state, observation):
	"""
	:param belief_state: [known part of state, belief distribution over desired item]
	:param desired_item: index of desired item
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief distribution]
	"""
	global belief_update_total_time
	start = time()
	language_probs = [language_func(observation["language"], belief_state.to_state(i)) for i in
	                  range(len(belief_state["desired_item"]))]
	gesture_probs = [gesture_func_robot(observation["gesture"],  belief_state.to_state(i)) for i in
	                 range(len(belief_state["desired_item"]))]
	max_gesture_prob = maxish(gesture_probs)
	if max_gesture_prob < .10:
		gesture_probs = [gesture_func_robot(None, belief_state.to_state(i)) for i in
		                 range(len(belief_state["desired_item"]))]
	observation_probs = [language_probs[i] * gesture_probs[i] for i in
	                     range(len(belief_state["desired_item"]))]
	denominator = dot(belief_state["desired_item"], observation_probs)
	if denominator == 0:
		print("belief_state[\"desired_item\"] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	belief_update_total_time += time() - start
	return ret
cpdef belief_update_robot_old(b, o):
	"""
	:param b: [known part of state, belief distribution over desired item]
	:param desired_item: index of desired item
	:param o: [set of words, gesture vector]
	:return: [known part of state, belief distribution]
	"""
	global belief_update_total_time
	start = time()
	# if (o["language"] is None or o["language"] == set()) and o["gesture"] is None:
	# 	belief_update_total_time += time() - start
	# 	return b
	language_probs = [language_func(o["language"], [i, b[0], b[1]]) for i in
	                  range(len(b[1]))]
	gesture_probs = [gesture_func_robot(o["gesture"], [i, b[0], b[1]]) for i in
	                 range(len(b[1]))]
	max_gesture_prob = maxish(gesture_probs)
	if max_gesture_prob < .10:
		gesture_probs = [gesture_func_robot(None, [i, b[0], b[1]]) for i in
		                 range(len(b[1]))]
	observation_probs = [language_probs[i] * gesture_probs[i] for i in
	                     range(len(b[1]))]
	denominator = dot(b[1], observation_probs)
	if denominator == 0:
		print("b[1] dot observation_probs = 0")
		print("b = " + str(b))
		print("observation_probs = " + str(observation_probs))
		print("o = " + str(o))
		return b
	ret = [[b[0][1], b[0][1]],
	       [b[1][j] * observation_probs[j] / denominator for j in range(len(b[1]))]]  #Replace with c
	belief_update_total_time += time() - start
	return ret

cpdef double observation_func(observation, state):
	global observation_func_total_time
	start_time = time()
	prob = language_func(observation["language"], state) * gesture_func(observation["gesture"], state)
	observation_func_total_time += time() - start_time
	return prob
cpdef double observation_func_robot(observation, state):
	global observation_func_total_time
	start_time = time()
	prob = language_func(observation["language"], state) * gesture_func_robot(observation["gesture"], state)
	observation_func_total_time += time() - start_time
	return prob

cpdef double gesture_func(observation, state):
	global gesture_func_total_time
	start_time = time()
	if observation is None:
		prob = 1 - p_g
	else:
		ideal_vector = items[state["desired_item"]]["location"]
		prob = vec_prob(ideal_vector, observation)
	gesture_func_total_time += time() - start_time
	return prob

cpdef double gesture_func_robot(observation, state):
	global gesture_func_total_time
	start_time = time()
	if observation is None:
		prob = 1 - p_g
	else:
		head = [observation[0], observation[1], observation[2]]
		end_effector = [observation[3], observation[4], observation[5]]
		given_vector = subtract(end_effector, head)
		ideal_vector = subtract(items[state["desired_item"]]["location"], head)
		prob = vec_prob(ideal_vector, given_vector)
	gesture_func_total_time += time() - start_time
	return prob
cpdef double language_func(observation, state):
	# Need to change the way we handle BoW data. Store items with attributes, get descriptors from attributes
	cdef double base_utterance_prob = base_probability(observation, relevant_words[state["desired_item"]], all_words)
	cdef double response_utterance_prob = response_probability(observation, state)
	return base_utterance_prob * response_utterance_prob

cpdef double response_probability(language, state):
	"""
	:param language: language
	:param state: state
	:return: P(language | state)
	TODO: Probability of null response should be higher if state[1] is None
	"""
	cdef int num_positive_included = len(positive_responses.intersection(language))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(language))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included

	if num_positive_included + num_negative_included == 0:
		return 1 - p_l
	if state["last_referenced_item"] is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	match_prob = p_r_match
	if state["reference_type"] == "look":
		match_prob = p_r_match_look
	if state["desired_item"] == state["last_referenced_item"]:
		return match_prob ** (num_positive_included + num_negative_omitted) \
		       * (1 - match_prob) ** (num_positive_omitted + num_negative_omitted)
	return (1 - match_prob) ** (num_positive_included + num_negative_omitted) \
	       * match_prob ** (num_positive_omitted + num_negative_omitted)

cpdef double base_probability(language, vocab, words):
	'''
	:param language: set of words uttered. Consider changing to multiset
	:param words: set of all known words
	:param vocab: set of words related to object in question
	:return: probablity of language
	'''
	# TODO: take into account words ommitted so that probability sums to 1.
	if language is None or language == set():
		return 1 - p_l
	cdef double denominator = len(vocab) + alpha * len(words)
	cdef int num_relevant_words_included = len(set([word for word in language if word in vocab]))
	cdef int num_relevant_words_omitted = len(vocab) - num_relevant_words_included
	cdef int num_irrelevant_words_included = len(language) - num_relevant_words_included
	cdef int num_irrelevant_words_omitted = len(words) - len(vocab) - num_irrelevant_words_included

	cdef double prob_relevant_word_included = (1 + alpha) / denominator
	cdef double prob_irrelevant_word_included = alpha / denominator
	return p_l * (prob_relevant_word_included ** num_relevant_words_included) \
	       * (prob_irrelevant_word_included ** num_irrelevant_words_included) \
	       * (1 - prob_relevant_word_included) ** num_relevant_words_omitted \
	       * (1 - prob_irrelevant_word_included) ** num_irrelevant_words_omitted

cdef sample_gesture(state, allow_none=True):
	global sample_gesture_total_time
	cdef double start_time = time()
	if allow_none and random.random() < p_g:
		return None
	cdef list ideal_vector = items[state["desired_item"]]["location"]
	# The orthogonal vector calculation may be problematic
	cdef list orthogonal_vector = cross(ideal_vector, [random.random() for j in range(3)])
	cdef double angle_off = np.random.normal(0, std_theta)
	cdef double w1 = math.cos(angle_off)
	cdef double w2 = math.sin(angle_off)
	cdef list g = [w1 * ideal_vector[0] + w2 * orthogonal_vector[0], w1 * ideal_vector[1] + w2 * orthogonal_vector[1],
	               w1 * ideal_vector[2] + w2 * orthogonal_vector[2]]
	empirical_angle_off = angle_between(ideal_vector, g)
	if False and abs(angle_off - empirical_angle_off) > .01:
		print("angle_off - empirical_angle_off = " + str(angle_off - empirical_angle_off))
		raise ValueError
	sample_gesture_total_time += time() - start_time
	return g

cpdef sample_language(state):
	language = sample_base_utterance(state)
	# print("base in cstuff: " + str(language))
	language.update(sample_response_utterance(state))
	# print("composite in cstuff: " + str(language))
	return language
# cpdef sample_language_detailed(state):
# 	language = sample_base_utterance_detailed(state)
# 	# print("base in cstuff: " + str(language))
# 	language.update(sample_response_utterance(state))
# 	# print("composite in cstuff: " + str(language))
# 	return language
# Review sample response, base
cpdef sample_response_utterance(state):
	"""
	:param state: state
	:return: single word response utterance
	"""
	if state["last_referenced_item"] is None or random.random() > p_l:
		return set()  # This seems more reasonable than randomly picking yes/no
	match_prob = p_r_match
	if state["reference_type"] == "look":
		match_prob = p_r_match_look
	if state["last_referenced_item"] == state["desired_item"]:
		if random.random() < match_prob:
			return set(["yes"])
		return set(["no"])
	if random.random() < match_prob:
		return set(["no"])
	return set(["yes"])

cdef sample_base_utterance(state):
	"""
	:param relevant_words: words related to desired object
	:param other_words: words unrelated to desired object
	:return: single word response utterance
	"""
	# TODO: Potentially make this more realistic - Currently gives equal weight to all relevant words: shape, color, etc.
	# return nothing if the human doesn't speak
	if random.random() > p_l:
		return set()
	global relevant_words
	global irrelevant_words
	# relevant_words_local = relevant_words[state[0]]
	cdef int item_id = state["desired_item"]
	if item_id >= len(items) or item_id < 0:
		print("item_id is all fouled up: " + str(item_id))
	cdef set relevant_words_local = relevant_words[item_id]
	cdef set other_words_local = irrelevant_words[state["desired_item"]]
	# return a relevant word with probability |id.vocab| * p(w | i_d) for w \in id.vocab
	num_relevant_words = len(relevant_words_local)
	num_other_words = len(other_words_local)
	num_all_words = num_relevant_words + num_other_words
	if random.random() < num_relevant_words * (1 - alpha) / (num_relevant_words + alpha * num_all_words):
		r = random.sample(relevant_words_local, 1)
	else:
		r = random.sample(other_words_local, 1)
	return set(r)

# cdef sample_base_utterance_detailed(s):
# 	"""
# 	:param relevant_words: words related to desired object
# 	:param other_words: words unrelated to desired object
# 	:return: single word response utterance
# 	"""
# 	# TODO: Potentially make this more realistic - Currently gives equal weight to all relevant words: shape, color, etc.
# 	# return nothing if the human doesn't speak
# 	if random.random() > p_l:
# 		return set()
# 	global relevant_words
# 	global irrelevant_words
# 	# relevant_words_local = relevant_words[s[0]]
# 	cdef int item_id = s[0]
# 	if item_id >= len(items) or item_id < 0:
# 		print("item_id is all fouled up: " + str(item_id))
# 	cdef set relevant_words_local = relevant_words[item_id]
# 	cdef set other_words_local = irrelevant_words[s[0]]
# 	# return a relevant word with probability |id.vocab| * p(w | i_d) for w \in id.vocab
# 	num_relevant_words = len(relevant_words_local)
# 	num_other_words = len(other_words_local)
# 	num_all_words = num_relevant_words + num_other_words
#
# 	if random.random() < num_relevant_words * (1 - alpha) / (num_relevant_words + alpha * num_all_words):
# 		# r = random.sample(relevant_words_local, 1)
# 		r = [random.sample(bag_of_words[items[s[0]][attr]], 1)[0] for attr in ATTRIBUTES]
#
# 	else:
# 		r = random.sample(other_words_local, 1)
# 	return set(r)

cpdef sample_observation(state):
	global obs_sampling_time
	cdef double start_time = time()
	language = sample_response_utterance(state)
	language.update(sample_base_utterance(state))
	gesture = sample_gesture(state)
	obs_sampling_time += time() - start_time
	return {"language": language, "gesture": gesture}
# cpdef sample_observation_detailed(s):
# 	global obs_sampling_time
# 	cdef double start_time = time()
# 	language = sample_response_utterance(s)
# 	language.update(sample_base_utterance_detailed(s))
# 	gesture = sample_gesture(s)
# 	obs_sampling_time += time() - start_time
# 	return {"language": language, "gesture": gesture}

cdef list cross(list u, list v):
	return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

def get_times():
	return {"estimate_qs_counter": estimate_qs_counter,
	        "estimate_qs_total_time": estimate_qs_total_time,
	        "v_total_time": v_total_time,
	        "generate_total_time": generate_total_time,
	        "belief_update_total_time": belief_update_total_time,
	        "observation_func_total_time": observation_func_total_time,
	        "gesture_func_total_time": gesture_func_total_time,
	        "sample_gesture_total_time": sample_gesture_total_time,
	        "obs_sampling_time": obs_sampling_time}
#
cpdef sample_distinct_states(belief, int n = 1):
	'''
	Samples n distinct states from belief. Undefined behavior for n > len(belief)
	:param belief: 
	:param n: 
	:return: 
	'''
	cdef states = []
	cdef int j
	cdef int i
	cdef int found_state
	cdef double cumulative_probability
	# while n >= len(belief):
	# 	states.extend([i for i in range(len(belief))])
	# 	n -= len(belief)
	for j in range(n):
		i = 0
		found_state = 0
		cumulative_probability = 0
		while found_state == 0 and i in range(len(belief)):
			if type(belief[i]) is type(None):
				print("belief[i] is None: i = " + str(i))
				print("belief[1]: " + str(belief[1]))
			#Avoid duplicates
			if random.random() < belief[i] / (1 - cumulative_probability) and i not in states:
				states.append(i)
				found_state = 1
			cumulative_probability += belief[i]
			#If we have not selected a state because of floating point error, return a uniformly random state
			i += 1
		if found_state == 1:
			states.append(random.sample([i for i in range(len(belief))], 1)[0])
	return states

cpdef sample_state(list b):
	cdef double cumulative_probability = 0
	cdef int i
	for i in range(len(b)):
		if random.random() < b[i] / (1 - cumulative_probability):
			return i
		cumulative_probability += b[i]

	# In case the distribution added to slightly below 1 and we had bad luck
	return random.sample([i for i in range(len(b))], 1)[0]

# cpdef sample_state(list b):
# 	cdef double cumulative_probability = 0
# 	cdef int i
# 	for i in range(len(b)):
# 		if random.random() < b[i]/(1-cumulative_probability):
# 			return i
# 		cumulative_probability += b[i]
#
# 	# In case the distribution added to slightly below 1 and we had bad luck
# 	return random.sample([i for i in range(len(b))],1)[0]

cpdef states_equal(s1, s2):
	if s1[0] == s2[0] and s1[1] == s2[1] and s1[2] == s2[2]:
		return True
	return False

cpdef kl_divergence(list a, list b):
	cdef int i
	cdef double div = 0
	for i in range(len(a)):
		div += a[i] * math.log(a[i] / b[i])
	return div

cpdef maxish(a):
	if type(a) in (int, float):
		return a
	return max(a)

cpdef dict get_qvalues2(self, list b, dict true_state, int horizon):
	#self may not work. Test
	cdef list actions = self.pomdp.actions
	cdef list rewards = [self.pomdp.get_reward_from_state(true_state, a) for a in actions]
	cdef int i
	cdef int num_actions = len(actions)
	if horizon == 0:
		return rewards
	actions = self.pomdp.actions
	next_states = [self.pomdp.transition_func(true_state, a) for a in actions]
	cdef int num_next_states = len(next_states)
	#Generalize for general BSS
	terminal_states = [i for i in range(num_actions) if actions[i].split(" ")[0] == "pick"]
	observations = [sample_observation(next_states[i]) for i in range(num_next_states)]
	next_beliefs = [belief_update(b, o) for o in observations]
	# next_qvalues = [get_qvalues2(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(num_next_states)]
	next_qvalues = []
	for i in range(num_next_states):
		if i in terminal_states:
			next_qvalues.append(0)
		else:
			next_qvalues.append(self.get_qvalues2(next_beliefs[i], next_states[i], horizon - 1))
	return [rewards[i] + self.pomdp.gamma * maxish(next_qvalues[i]) for i in range(len(next_states))]
