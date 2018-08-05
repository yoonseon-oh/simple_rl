# from config1 import *
import math
from time import time
import json
import random
import numpy as np
# from libcpp.vector cimport vector
import os
import sys
# from simple_rl.tasks.FetchPOMDP.jsonreader import load_json


cdef obs_sampling_time = 0
cdef estimate_qs_counter = 0
cdef estimate_qs_total_time = 0
cdef v_total_time = 0
cdef generate_total_time = 0
cdef belief_update_total_time = 0
cdef observation_func_total_time = 0
cdef gesture_func_total_time = 0
cdef sample_gesture_total_time = 0

cdef double p_g = .1
cdef double p_l = .95  # split into base and response probabilities
cdef double p_r_match = .99
cdef double alpha = .2
cdef double std_theta = .15
cdef double discount = .9
cdef error = 1
cdef load_json(file_name):
	with open(file_name) as json_data:
		return json.load(json_data)
cdef load_json_from_while_imported(file_name):
	#Doesn't work - __file__ not defined for c modules.
	#Want to get FetchPOMDP directory. Currently gets directory of file that imported this file
	with open(os.path.join(sys.path[0], file_name)) as json_data:
			return json.load(json_data)
cdef config = load_json_from_while_imported("config.json")
cdef bag_of_words = config["bag_of_words"]
cpdef items = config["items"]
cdef ATTRIBUTES = config["attributes"]
cdef num_items = len(items)
cdef double std_theta_look = .3
cdef double point_cost = -1
cdef double look_cost = -3
cdef double wait_cost = -1*(1.0/6.0)
cdef double wrong_pick_cost = -20/num_items
cdef double correct_pick_reward = 10/num_items
cdef double info_value = 1

cdef desired_item = 2
cdef last_referenced_item = None
cdef human_pointing_source = [0, 0, 0]
# Include point or pick in last referenced item
cdef state = {"desired_item": desired_item, "last_referenced_item": last_referenced_item}
cdef positive_responses = {"yes", "yeah", "sure", "yup"}
cdef negative_responses = {"no", "nope", "other", "not"}

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

# cdef bag_of_words = load_json('bag of words.json')
cdef all_words = union_dictionary(bag_of_words)
cdef relevant_words = [get_relevant_words(i, bag_of_words) for i in range(len(items))]
cdef irrelevant_words = [get_irrelevant_words(i, bag_of_words) for i in range(len(items))]

cpdef double sum(list a):
	cdef double total = 0
	cdef int i
	for i in range(len(a)):
		total += a[i]
	return total

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
		return [a[i]/den for i in range(len(a))]

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

cpdef belief_update(b, o):
	"""
	:param b: [known part of state, belief distribution over desired item]
	:param desired_item: index of desired item
	:param o: [set of words, gesture vector]
	:return: [known part of state, belief distribution]
	"""
	global belief_update_total_time
	start = time()
	observation_probs = [observation_func(o, {"desired_item": i, "last_referenced_item": b[0]}) for i in
	                     range(len(b[1]))]
	denominator = dot(b[1], observation_probs)
	ret = [b[0], [b[1][j] * observation_probs[j] / denominator for j in range(len(b[1]))]]  #Replace with c
	belief_update_total_time += time() - start
	return ret


def belief_update_c(b, o):
	"""
	:param b: [known part of state, belief distribution over desired item]
	:param desired_item: index of desired item
	:param o: [set of words, gesture vector]
	:return: [known part of state, belief distribution]
	"""
	global belief_update_total_time
	start = time()
	observation_probs = [observation_func(o, {"desired_item": i, "last_referenced_item": b[0]}) for i in
	                     range(len(b[1]))]
	denominator = dot(b[1], observation_probs)
	ret = [b[0], divide(b[1],denominator)]
	belief_update_total_time += time() - start
	return ret


cpdef double observation_func(o, s):
	global observation_func_total_time
	start_time = time()
	prob = language_func(o["language"], s) * gesture_func(o["gesture"], s)
	observation_func_total_time += time() - start_time
	return prob

cpdef double gesture_func(o, s):
	global gesture_func_total_time
	start_time = time()
	if o is None:
		prob = 1 - p_g
	else:
		# target = items[s["desired_item"]]["location"]
		# ideal_vector = [target[0] - human_pointing_source[0],target[1] - human_pointing_source[1],target[2] - human_pointing_source[2]]
		ideal_vector = items[s["desired_item"]]["location"]
		prob = vec_prob(ideal_vector, o)
	gesture_func_total_time += time() - start_time
	return prob

cpdef double language_func(o, s):
	# Need to change the way we handle BoW data. Store items with attributes, get descriptors from attributes
	cdef double base_utterance_prob = base_probability(o, relevant_words[s["desired_item"]], all_words)
	cdef double response_utterance_prob = response_probability(o, s)
	return base_utterance_prob * response_utterance_prob

cpdef double response_probability(l, s):
	"""
	:param l: language
	:param s: state
	:return: P(l | s)
	TODO: Probability of null response should be higher if s["last_referenced_item"] is None
	"""
	cdef int num_positive_included = len(positive_responses.intersection(l))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(l))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included

	if num_positive_included + num_negative_included == 0:
		return 1 - p_l
	if s["last_referenced_item"] is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	if s["desired_item"] == s["last_referenced_item"]:
		return p_r_match ** (num_positive_included + num_negative_omitted) \
		       * (1 - p_r_match) ** (num_positive_omitted + num_negative_omitted)
	return (1 - p_r_match) ** (num_positive_included + num_negative_omitted) \
	       * p_r_match ** (num_positive_omitted + num_negative_omitted)

cpdef double base_probability(l, vocab, words):
	'''
	:param l: set of words uttered. Consider changing to multiset
	:param words: set of all known words
	:param vocab: set of words related to object in question
	:return: probablity of l
	'''
	# TODO: take into account words ommitted so that probability sums to 1.
	if l is None or l == set():
		return 1 - p_l
	cdef double denominator = len(vocab) + alpha * len(words)
	cdef int num_relevant_words_included = len(set([word for word in l if word in vocab]))
	cdef int num_relevant_words_omitted = len(vocab) - num_relevant_words_included
	cdef int num_irrelevant_words_included = len(l) - num_relevant_words_included
	cdef int num_irrelevant_words_omitted = len(words) - len(vocab) - num_irrelevant_words_included

	cdef double prob_relevant_word_included = (1 + alpha) / denominator
	cdef double prob_irrelevant_word_included = alpha / denominator
	return p_l * (prob_relevant_word_included ** num_relevant_words_included) \
	       * (prob_irrelevant_word_included ** num_irrelevant_words_included) \
	       * (1 - prob_relevant_word_included) ** num_relevant_words_omitted \
	       * (1 - prob_irrelevant_word_included) ** num_irrelevant_words_omitted

cdef sample_gesture(s, allow_none=True):
	global sample_gesture_total_time
	cdef double start_time = time()
	if allow_none and random.random() < p_g:
		return None
	cdef list ideal_vector = items[s["desired_item"]]["location"]
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


# Review sample response, base
cdef sample_response_utterance(s):
	"""
	:param s: state
	:return: single word response utterance
	"""
	if s["last_referenced_item"] is None or random.random() > p_l:
		return set()  # This seems more reasonable than randomly picking yes/no
	if s["last_referenced_item"] == s["desired_item"]:
		if random.random() < .99:
			return set(["yes"])
		return set(["no"])
	if random.random() < .99:
		return set(["no"])
	return set(["yes"])


cdef sample_base_utterance(s):
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
	# relevant_words_local = relevant_words[s["desired_item"]]
	cdef int item_id = s["desired_item"]
	if item_id >= len(items) or item_id < 0:
		print("item_id is all fouled up: " + str(item_id))
	cdef set relevant_words_local = relevant_words[item_id]
	cdef set other_words_local = irrelevant_words[s["desired_item"]]
	# return a relevant word with probability |id.vocab| * p(w | i_d) for w \in id.vocab
	num_relevant_words = len(relevant_words_local)
	num_other_words = len(other_words_local)
	num_all_words = num_relevant_words + num_other_words
	if random.random() < num_relevant_words * (1 - alpha) / (num_relevant_words + alpha * num_all_words):
		r = random.sample(relevant_words_local, 1)
	else:
		r = random.sample(other_words_local, 1)
	return r


cpdef sample_observation(s):
	global obs_sampling_time
	cdef double start_time = time()
	language = sample_response_utterance(s)
	language.update(sample_base_utterance(s))
	gesture = sample_gesture(s)
	obs_sampling_time += time() - start_time
	return {"language": language, "gesture": gesture}

cdef list cross(list u, list v):
	return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]

def get_times():
	return {"estimate_qs_counter":estimate_qs_counter,
	"estimate_qs_total_time":estimate_qs_total_time,
	"v_total_time":v_total_time,
	"generate_total_time":generate_total_time,
	"belief_update_total_time":belief_update_total_time,
	"observation_func_total_time":observation_func_total_time,
	"gesture_func_total_time":gesture_func_total_time,
	"sample_gesture_total_time":sample_gesture_total_time,
	"obs_sampling_time": obs_sampling_time}
#
cpdef sample_states(b, int n = 1):
	cdef states = []
	cdef int j
	cdef int i
	cdef int found_state
	cdef double cumulative_probability
	for j in range(n):
		i = 0
		found_state = 0
		cumulative_probability = 0
		while found_state == 0 and i in range(len(b)):
			cumulative_probability += b[i]
			#Avoid duplicates
			if random.random() < cumulative_probability and i not in states:
				states.append(i)
				found_state = 1
			#If we have not selected a state becusae of floating point error, return a uniformly random state
			i += 1
		if found_state ==1:
			states.append(random.sample([i for i in range(len(b))],1)[0])
	return states

cpdef sample_state(list b):
	cdef double cumulative_probability = 0
	cdef int i
	for i in range(len(b)):
		cumulative_probability += b[i]
		if random.random() < cumulative_probability:
			return i
	# In case the distribution added to slightly below 1 and we had bad luck
	return random.sample([i for i in range(len(b))],1)[0]

cpdef states_equal(s1,s2):
	if s1["desired_item"] == s2["desired_item"] and s1["last_referenced_item"] == s2["last_referenced_item"]:
		return True
	return False

cpdef kl_divergence(list a,list b):
	cdef int i
	cdef double div = 0
	for i in range(len(a)):
		div += a[i] * math.log(a[i]/b[i])
	return div

cpdef maxish(a):
	if type(a) in (int, float):
		return a
	return max(a)


cpdef dict get_qvalues2(self, list b, dict true_state, int horizon):
	#self may not work. Test
	cdef list actions = self.pomdp.actions
	cdef list rewards = [self.pomdp.get_reward_from_state(true_state,a) for a in actions]
	cdef int i
	cdef int num_actions = len(actions)
	if horizon == 0:
		return rewards
	actions = self.pomdp.actions
	next_states = [self.pomdp.transition_func(true_state,a) for a in actions]
	cdef int num_next_states = len(next_states)
	#Generalize for general BSS
	terminal_states = [i for i in range(num_actions) if actions[i].split(" ")[0] == "pick"]
	observations = [sample_observation(next_states[i]) for i in range(num_next_states)]
	next_beliefs = [belief_update(b,o) for o in observations]
	# next_qvalues = [get_qvalues2(next_beliefs[i], next_states[i], horizon - 1) if i not in terminal_states else 0.0 for i in range(num_next_states)]
	next_qvalues = []
	for i in range(num_next_states):
		if i in terminal_states:
			next_qvalues.append(0)
		else:
			next_qvalues.append(self.get_qvalues2(next_beliefs[i], next_states[i], horizon - 1))
	return [rewards[i] + self.pomdp.gamma * maxish(next_qvalues[i]) for i in range(len(next_states))]