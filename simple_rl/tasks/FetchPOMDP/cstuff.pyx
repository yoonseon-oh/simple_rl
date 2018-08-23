# from config1 import *
import math
from collections import defaultdict
from datetime import datetime
import copy
from time import time
import json
import random

random.seed(0)
import numpy as np
from scipy.stats import truncnorm
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
#Debugging variables
cdef num_belief_updates = 0
cdef num_impossible_observations = 0
cdef impossible_responses = 0
cdef impossible_bases = 0
cdef impossible_gestures = 0

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
cdef double point_cost = config["point_cost"]
cdef double p_g = config["p_g"]
cdef double p_l_b = config["p_l_b"]  # split into base and response probabilities
cdef double p_l_r = config["p_l_r"]
cdef double p_r_match = config["p_r_match"]
cdef double p_r_match_look = config["p_r_match_look"]
cdef double alpha = config["alpha"]
cdef double std_theta = config["std_theta"]
cdef double std_theta_point = config["std_theta_point"]
cdef double std_theta_look = config["std_theta_look"]
cdef double gamma = config["gamma"]
cdef error = 1
import os

cdef list point_confusion_matrix = calculate_confusion_matrix(items, std_theta_point, -math.pi / 2, math.pi / 2)
cdef list look_confusion_matrix = calculate_confusion_matrix(items, std_theta_look, -math.pi / 2, math.pi / 2)
cpdef dict response_confusion_matrices = {"look": look_confusion_matrix, "point": point_confusion_matrix}
cdef dummy_function():
	return "dummy function"
#Needs to be proper matrix: it matters which item the human incorrectly thinks the agent is pointing at.
cdef calculate_confusion_matrix(items, double std_dev, double min_angle, double max_angle):
	#Indices seem off by one
	#convert item locations to angles (ignore height for now for simplicity)
	item_angles = {i: math.atan2(items[i]["location"][1], items[i]["location"][0]) for i in range(len(items))}
	# sorted_angles = sorted(item_angles.items(), key=lambda kv: kv[1])
	sorted_angles = dict_to_items_sorted_by_value(item_angles)
	midpoints = [(sorted_angles[i][1] + sorted_angles[i + 1][1]) / 2 for i in range(len(sorted_angles) - 1)]
	print(midpoints)
	#Get intervals over which each item is the nearest item (voronoi circle)
	#First and last should wraparound. TODONE
	intervals = {}
	#Set cutoff points for first and last intervals. TODO make sure this works.
	# The if may in fact be unnecessary, since truncnorm.cdf will handle boundary
	pi = math.pi
	print("-math.pi: " + str(-math.pi))
	print("math.pi: " + str(math.pi))
	print("type(min_angle): " + str(type(min_angle)))
	print("min_angle: " + str(min_angle))
	print("max_angle: " + str(max_angle))
	# if min_angle == -pi and max_angle == pi:
	# 	left_mid = ((sorted_angles[-1][1] - 2*math.pi) + sorted_angles[0][1])/2
	# 	right_mid = ((sorted_angles[0][1] + 2*math.pi) + sorted_angles[-1][1])/2
	# else:
	# 	left_mid = min_angle
	# 	right_mid = max_angle
	for i in range(len(items)):
		#zeroth interval is -pi to the zeroth midpoint
		#modify to work with min and max angles
		if i == 0:
			left_mid = ((sorted_angles[-1][1] - 2 * math.pi) + sorted_angles[0][1]) / 2
			intervals[sorted_angles[i][0]] = (left_mid, midpoints[0])
		#last interval is last midpoint to pi
		elif i == len(items) - 1:
			right_mid = ((sorted_angles[0][1] + 2 * math.pi) + sorted_angles[-1][1]) / 2
			intervals[sorted_angles[i][0]] = (midpoints[-1], right_mid)
		#other intervals are from midpoint to midpoint
		else:
			intervals[sorted_angles[i][0]] = (midpoints[i - 1], midpoints[i])
	print(intervals)
	#Assume that the actual location the agent points to is sampled from a gaussian with variance std_theta_look/point.
	#Calculate the probability that pointing to an object is interpreted as pointing to another object
	confusion_matrix = []
	for i in range(len(items)):
		print("i = " + str(i) + " in calculate_confusion_matrix")
		confusion_row = []
		center = item_angles[i]
		left_trunc = (min_angle - center) / std_dev
		right_trunc = (max_angle - center) / std_dev
		for j in range(len(items)):
			print("j = " + str(j) + " in calculate_confusion_matrix")
			#TODO replace min,max with relative min, max to work with truncnorm
			#TODO consider increasing precision
			cdf_left = truncnorm.cdf(intervals[j][0], left_trunc, right_trunc, center, std_dev)
			cdf_right = truncnorm.cdf(intervals[j][1], left_trunc, right_trunc, center, std_dev)
			prob_match = cdf_right - cdf_left
			# if prob_match == 0:
			# 	data = {"i": i, "angles[i]": item_angles[i], "j": j, "angles[j]": item_angles[j],
			# 	        "min_angle": min_angle, "max_angle": max_angle, "intervals[i]": intervals[i],
			# 	        "intervals[j]": intervals[j], "cdf_left": cdf_left, "cdf_right": cdf_right,
			# 	        "left_trunc": left_trunc, "right_trunc": right_trunc, "std_dev": std_dev}
			# 	with open('error calculating confusion matrix ' + str(datetime.now()).replace(":", ".")[:22] + '.json',
			# 	          'w') as fp:
			# 		json.dump(data, fp, indent=4)
			# raise ValueError("Probability 0 while calculating confusion matrix")
			confusion_row.append(prob_match)
		confusion_matrix.append(confusion_row)
	return confusion_matrix
cdef dict_to_items_sorted_by_value(d):
	items = list(copy.deepcopy(d).items())
	sorted_items = []
	num_items = len(items)
	for i in range(num_items):
		min_value = 99999999999999999999999
		min_item = None
		for j in items:
			if j[1] < min_value:
				min_value = j[1]
				min_item = j
		sorted_items.append(j)
		items.remove(j)
	return sorted_items
cpdef sample_interpreted_reference(state):
	#TODO rename
	last_referenced_item = state["last_referenced_item"]
	#TODO consider representing confusion matrices as dicts so that we do not need a special case for None
	if last_referenced_item is None:
		return None
	probs = response_confusion_matrices[state["reference_type"]][last_referenced_item]
	interpreted_item = sample_state(probs)
	return interpreted_item

cdef positive_responses = set(config["positive_responses"])
cdef negative_responses = set(config["negative_responses"])

# Globals defined for speed
cdef std_theta_1 = 2 * (std_theta ** 2)
cdef std_theta_p_g = p_g / math.sqrt(2 * math.pi * (std_theta ** 2))
cpdef get_items():
	return items
cpdef get_num_belief_updates_and_impossible_observations():
	return num_belief_updates, num_impossible_observations
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

cdef get_potential_words(desired_item, last_referenced_item):
	'''
	:param desired_item: 
	:param last_referenced_item: 
	:return: set of words related to desired_item but not last_referenced_item if the resulting set is nonempty, else
	all words related to desired_item
	'''
	words = relevant_words[desired_item]
	if last_referenced_item is None:
		other_words = set()
	else:
		other_words = relevant_words[last_referenced_item]
	potential_words = words.difference(other_words)
	if potential_words == set():
		potential_words = words
	return potential_words
cdef potential_words = [
	[get_potential_words(desired_item, last_referenced_item) for last_referenced_item in range(num_items)] for
	desired_item in range(num_items)]

cpdef double average(list a):
	cdef double total = sum(a)
	return total / len(a)
cpdef double median(list a):
	pass
cpdef double sum(list a):
	cdef double total = 0
	cdef int i
	for i in range(len(a)):
		total += a[i]
	return total

cpdef double sum_dict(a):
	cdef double total = 0
	cdef int i
	for key in a.keys():
		total += a[key]
	return total
cpdef list add(list a, list b):
	return [a[i] + b[i] for i in range(len(a))]

cpdef add_defaultdict(a, b, keys = None):
	#Can't use lambda expressions in cython, so I cannot add the default factories
	if keys == None:
		keys = a.keys()
	if type(a) is defaultdict:
		ret = defaultdict(a.default_factory)
	else:
		ret = {}
	ret.update({key: a[key] + b[key] for key in keys})
	return ret
cpdef add_dict(a, b, keys = None):
	#Can't use lambda expressions in cython, so I cannot add the default factories
	if keys == None:
		keys = a.keys()
	ret = {key: a[key] + b[key] for key in keys}
	return ret
cpdef linear_combination_of_dicts(ds,weights):
	new_dicts = [times_dict(ds[i],weights[i]) for i in range(len(ds))]
	return add_list_of_dicts(new_dicts)
cpdef linear_combination_of_lists(ds,weights):
	new_lists = [times_list(ds[i],weights[i]) for i in range(len(ds))]
	return add_list_of_lists(new_lists)
cpdef linear_combination_of_alphas(alphas,weights):
	ds = [a["values"] for a in alphas]
	vals = linear_combination_of_dicts(ds,weights)
	alpha = {"values":vals, "action":None}
	return alpha
cpdef add_list_of_dicts(ds):
	#TODO check whether swapping order of iteration is faster
	cdef double sum
	cdef dict new_d = {}
	for key in ds[0].keys():
		sum = 0
		for d in ds:
			sum += d[key]
		new_d[key] = sum
	return new_d
cpdef add_list_of_alphas(alphas, list keys):
	new_alpha = {"values": {}, "action": None}
	cdef double sum
	for key in keys:
		sum = 0
		for alpha in alphas:
			sum += alpha["values"][key]
		new_alpha["values"][key] = sum
	return new_alpha
cpdef list subtract(list a, list b):
	return [a[i] - b[i] for i in range(len(a))]
cpdef times_defaultdict(a, scalar):
	#Can't use lambda expressions in cython, so I cannot scale the default factories
	if type(a) is defaultdict:
		ret = defaultdict(a.default_factory)
	else:
		ret = {}
	ret.update({key: scalar * a[key] for key in a.keys()})
	return ret
cpdef times_dict(a, scalar):
	#Can't use lambda expressions in cython, so I cannot scale the default factories
	ret = {key: scalar * a[key] for key in a.keys()}
	return ret
cpdef list times_list(list a, double scalar):
	return [scalar * i for i in a]
cpdef list add_list_of_lists(ls):
	cdef double sum
	cdef int i
	cdef list new_list = []
	for element_index in range(len(ls[0])):
		sum = 0
		for list_index in range(len(ls)):
			sum += ls[list_index][element_index]
		new_list.append(sum)
	return new_list

cpdef double distance(list a, list b):
	cdef double dist = 0
	for i in range(len(a)):
		dist += (a[i] - b[i]) ** 2
cpdef double distance_dict(a, b):
	cdef double distance = 0
	for key in a.keys():
		distance += (a[key] - b[key]) ** 2
	return distance
cpdef double dot_dict(a, b):
	cdef double total = 0
	for key in a.keys():
		total += a[key] * b[key]
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

cpdef double vec_prob(list ideal, list actual, double a = std_theta_p_g, double b = std_theta_1):
	return a * (math.e ** (-(angle_between(ideal, actual) ** 2) / b))
cpdef is_observation_impossible(belief_state,observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: 1 if observation is impossible, otherwise 0
	"""
	global num_belief_updates
	global num_impossible_observations
	num_belief_updates += 1
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [observation_func(observation, state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total.
	#Consider using decimal, may or not work with cython.
	#Consider ignoring observation and returning unaltered belief.
	if denominator == 0:
		return 1
	return 0
cpdef belief_update(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	global num_belief_updates
	global num_impossible_observations
	num_belief_updates += 1
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [observation_func(observation, state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total.
	#Consider using decimal, may or not work with cython.
	#Consider ignoring observation and returning unaltered belief.
	if denominator == 0:
		num_impossible_observations += 1
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
		        "reference_type": belief_state["reference_type"]}
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	return ret
cpdef belief_update_full_debug(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	global num_belief_updates
	global num_impossible_observations
	num_belief_updates += 1
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [observation_func(observation, state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total.
	#Consider using decimal, may or not work with cython.
	#Consider ignoring observation and returning unaltered belief.
	if denominator == 0:
		num_impossible_observations += 1
		print("Full update")
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return 1
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	return 0
cpdef belief_update_gesture_only_debug(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	global num_belief_updates
	global num_impossible_observations
	num_belief_updates += 1
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [gesture_func(observation["gesture"], state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total.
	#Consider using decimal, may or not work with cython.
	#Consider ignoring observation and returning unaltered belief.
	if denominator == 0:
		num_impossible_observations += 1
		print("Gesture only")
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return 1
		return {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
		        "reference_type": belief_state["reference_type"]}
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	return 0
cpdef belief_update_base_only_debug(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	global num_belief_updates
	global num_impossible_observations
	num_belief_updates += 1
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [base_probability(observation["language"], state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total.
	#Consider using decimal, may or not work with cython.
	#Consider ignoring observation and returning unaltered belief.
	if denominator == 0:
		num_impossible_observations += 1
		print("Base only")
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return 1
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	return 0
cpdef belief_update_response_only_debug(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [response_probability(observation["language"], state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total
	if denominator == 0:
		print("Response only")
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return 1
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
	return 0
cpdef belief_update_response_only(belief_state, observation):
	"""
	:param belief_state: FetchPOMDPBeliefState
	:param observation: [set of words, gesture vector]
	:return: [known part of state, belief_state distribution]
	"""
	cdef int i
	global belief_update_total_time
	start = time()
	possible_states = belief_state.get_all_possible_states()
	observation_probs = [response_probability(observation["language"], state) for state in possible_states]
	#remove zeros because anything is possible. TODO: Find out where the zeros are coming from
	# observation_probs = remove_zeros(observation_probs)
	cdef double denominator = dot(belief_state["desired_item"], observation_probs)
	#TODO find better solution for impossible observations. Currently getting .01 impossible/total
	if denominator == 0:
		print("Response only")
		print("Received observation with probability 0. Resetting belief.")
		print("belief_state[2] dot observation_probs = 0")
		print("belief_state = " + str(belief_state))
		print("observation_probs = " + str(observation_probs))
		print("observation = " + str(observation))
		desired_item_distr = [1.0 / len(possible_states) for i in range(len(possible_states))]
		return {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
		        "reference_type": belief_state["reference_type"]}
	# raise ValueError("Received observation with probability 0")
	# return belief_state
	desired_item_distr = [belief_state["desired_item"][j] * observation_probs[j] / denominator for j in
	                      range(len(belief_state["desired_item"]))]
	#Probability 0 breaks things (ex. KL divergence), so we'll replace 0 with the smallest positive float
	desired_item_distr = remove_zeros(desired_item_distr)
	ret = {"desired_item": desired_item_distr, "last_referenced_item": belief_state["last_referenced_item"],
	       "reference_type": belief_state["reference_type"]}
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
	gesture_probs = [gesture_func_robot(observation["gesture"], belief_state.to_state(i)) for i in
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
	#TODO: remove zeros
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
cpdef double observation_func_response_only(observation, state):
	global observation_func_total_time
	start_time = time()
	prob = response_probability(observation["language"], state)
	observation_func_total_time += time() - start_time
	return prob
cpdef double observation_func(observation, state):
	global observation_func_total_time
	start_time = time()
	prob = language_func(observation["language"], state) * gesture_func(observation["gesture"], state)
	# prob = language_func(observation["language"], state) * gesture_func(observation["gesture"], state)
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
	cdef double base_utterance_prob = base_probability(observation, state)
	cdef double response_utterance_prob = response_probability(observation, state)
	return base_utterance_prob * response_utterance_prob
cpdef double response_probability(language, state):
	"""
	:param language: language
	:param state: state
	:return: P(language | state)
	TODO: Probability of null response should be higher if state[1] is None
	"""
	#TODO fix
	cdef int num_positive_included = len(positive_responses.intersection(language))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(language))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included

	if num_positive_included + num_negative_included == 0:
		return 1 - p_l_r
	reference_type = state["reference_type"]
	last_referenced_item = state["last_referenced_item"]
	desired_item = state["desired_item"]
	if last_referenced_item is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	#Return probability that the human thinks the agent pointed at the desired item
	cdef double pos_prob = response_confusion_matrices[reference_type][last_referenced_item][desired_item]
	#p_r_match < 1 to represent error in speech detection
	# pos_prob *= p_r_match
	if num_positive_included > num_negative_included:
		return pos_prob
	if num_negative_included > num_positive_included:
		return 1 - pos_prob
cpdef double response_probability_state(observation, state):
	"""
	:param language: language
	:param state: state
	:return: P(language | state)
	TODO: Probability of null response should be higher if state[1] is None
	"""
	#TODO fix
	language = observation["language"]
	cdef int num_positive_included = len(positive_responses.intersection(language))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(language))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included
	if num_positive_included + num_negative_included == 0:
		return 1 - p_l_r
	reference_type = state["reference_type"]
	last_referenced_item = state["last_referenced_item"]
	desired_item = state["desired_item"]
	if last_referenced_item is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	#Return probability that the human thinks the agent pointed at the desired item
	cdef double pos_prob = response_confusion_matrices[reference_type][last_referenced_item][desired_item]
	if num_positive_included > num_negative_included:
		return pos_prob
	if num_negative_included > num_positive_included:
		return 1 - pos_prob

cpdef double response_probability_old(language, state):
	"""
	:param language: language
	:param state: state
	:return: P(language | state)
	TODO: Probability of null response should be higher if state[1] is None
	"""
	#TODO fix
	cdef int num_positive_included = len(positive_responses.intersection(language))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(language))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included

	if num_positive_included + num_negative_included == 0:
		return 1 - p_l_b
	if state["last_referenced_item"] is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	if state["reference_type"] == "point":
		match_prob = p_r_match
	elif state["reference_type"] == "look":
		match_prob = p_r_match_look
	if (state["desired_item"] == state["last_referenced_item"]) == (num_positive_included > num_negative_included):
		return match_prob
	return 1 - match_prob
# if state["desired_item"] == state["last_referenced_item"]:
# 	return match_prob ** (num_positive_included + num_negative_omitted) \
# 	       * (1 - match_prob) ** (num_positive_omitted + num_negative_omitted)
# return (1 - match_prob) ** (num_positive_included + num_negative_omitted) \
#        * match_prob ** (num_positive_omitted + num_negative_omitted)

cpdef double response_probability_2(language, state):
	"""
	:param language: language
	:param state: state
	:return: P(language | state)
	TODO: Probability of null response should be higher if state[1] is None
	"""
	#TODO fix
	cdef int num_positive_included = len(positive_responses.intersection(language))
	cdef int num_positive_omitted = len(positive_responses) - num_positive_included
	cdef int num_negative_included = len(negative_responses.intersection(language))
	cdef int num_negative_omitted = len(negative_responses) - num_negative_included

	if num_positive_included + num_negative_included == 0:
		return 1 - p_l_b
	if state["last_referenced_item"] is None:
		return .5 ** (len(positive_responses) + len(negative_responses))
	if state["reference_type"] == "point":
		match_prob = p_r_match
	elif state["reference_type"] == "look":
		match_prob = p_r_match_look
	if (state["desired_item"] == state["last_referenced_item"]) == (num_positive_included > num_negative_included):
		return match_prob
	return 1 - match_prob

cpdef double base_probability_old_params(language, vocab, words):
	'''
	:param language: set of words uttered. Consider changing to multiset
	:param words: set of all known words
	:param vocab: set of words related to object in question
	:return: probablity of language
	'''
	# TODO: take into account words ommitted so that probability sums to 1.
	if language is None or language == set():
		return 1 - p_l_b
	cdef double denominator = len(vocab) + alpha * len(words)
	cdef int num_relevant_words_included = len(set([word for word in language if word in vocab]))
	cdef int num_relevant_words_omitted = len(vocab) - num_relevant_words_included
	cdef int num_irrelevant_words_included = len(language) - num_relevant_words_included
	cdef int num_irrelevant_words_omitted = len(words) - len(vocab) - num_irrelevant_words_included

	cdef double prob_relevant_word_included = (1 + alpha) / denominator
	cdef double prob_irrelevant_word_included = alpha / denominator
	return p_l_b * (prob_relevant_word_included ** num_relevant_words_included) \
	       * (prob_irrelevant_word_included ** num_irrelevant_words_included) \
	       * (1 - prob_relevant_word_included) ** num_relevant_words_omitted \
	       * (1 - prob_irrelevant_word_included) ** num_irrelevant_words_omitted

cpdef double base_probability(observation, state):
	'''
	:param language: set of words uttered. Consider changing to multiset
	:param words: set of all known words
	:param vocab: set of words related to object in question
	:return: probablity of language
	'''
	# TODO: take into account words ommitted so that probability sums to 1.
	# language = observation["language"]
	language = observation
	vocab = relevant_words[state["desired_item"]]
	words = all_words
	if language is None or language == set():
		return 1 - p_l_b
	cdef double denominator = len(vocab) + alpha * len(words)
	cdef int num_relevant_words_included = len(set([word for word in language if word in vocab]))
	cdef int num_relevant_words_omitted = len(vocab) - num_relevant_words_included
	cdef int num_irrelevant_words_included = len(language) - num_relevant_words_included
	cdef int num_irrelevant_words_omitted = len(words) - len(vocab) - num_irrelevant_words_included

	cdef double prob_relevant_word_included = (1 + alpha) / denominator
	cdef double prob_irrelevant_word_included = alpha / denominator
	return p_l_b * (prob_relevant_word_included ** num_relevant_words_included) \
	       * (prob_irrelevant_word_included ** num_irrelevant_words_included) \
	       * (1 - prob_relevant_word_included) ** num_relevant_words_omitted \
	       * (1 - prob_irrelevant_word_included) ** num_irrelevant_words_omitted

cpdef sample_gesture(state, allow_none=True):
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
	#Note: this assumes the only source of error is the robot miscommunicating to the human, not the human misspeaking
	"""
	:param state: state
	:return: single word response utterance
	"""
	if state["last_referenced_item"] is None or random.random() > p_l_r :
		return set()  # This seems more reasonable than randomly picking yes/no
	yes_prob = response_confusion_matrices[state["reference_type"]][state["last_referenced_item"]][
		state["desired_item"]]
	if random.random() < yes_prob:
		return {"yes"}
	else:
		return {"no"}
# cpdef sample_response_utterance_old2(state):
# 	"""
# 	:param state: state
# 	:return: single word response utterance
# 	"""
# 	if state["last_referenced_item"] is None or random.random() > p_l_b:
# 		return set()  # This seems more reasonable than randomly picking yes/no
# 	match_prob = response_probs[state["reference_type"]][state["last_referenced_item"]]
# 	if state["last_referenced_item"] == state["desired_item"]:
# 		if random.random() < match_prob:
# 			return set(["yes"])
# 		return set(["no"])
# 	if random.random() < match_prob:
# 		return set(["no"])
# 	return set(["yes"])
cpdef sample_response_utterance_old(state):
	"""
	:param state: state
	:return: single word response utterance
	"""
	if state["last_referenced_item"] is None or random.random() > p_l_b:
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
cpdef sample_response_utterance_2(state):
	"""
	:param state: state
	:return: single word response utterance
	"""
	if state["last_referenced_item"] is None or random.random() > p_l_b:
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

cpdef sample_base_utterance(state):
	"""
	:param relevant_words: words related to desired object
	:param other_words: words unrelated to desired object
	:return: single word response utterance, related to desired item but not referenced item if possible
	"""
	# TODO: Potentially make this more realistic - Currently gives equal weight to all relevant words: shape, color, etc.
	# return nothing if the human doesn't speak
	if random.random() > p_l_b:
		return set()
	global relevant_words
	global irrelevant_words
	global potential_words
	# relevant_words_local = relevant_words[state[0]]
	cdef int item_id = state["desired_item"]
	#TODO turn response_confusion_matrices to dicts so that
	# if item_id >= len(items) or item_id < 0:
	# 	print("item_id is all fouled up: " + str(item_id))
	cdef set relevant_words_local = relevant_words[item_id]
	cdef set other_words_local = irrelevant_words[item_id]
	cdef set potential_words_local
	if state["last_referenced_item"] is None:
		potential_words_local = relevant_words_local
	else:
		assumed_referenced_item = sample_interpreted_reference(state)
		potential_words_local = potential_words[item_id][assumed_referenced_item]
	# return a relevant word with probability |id.vocab| * p(w | i_d) for w \in id.vocab
	num_potential_words = len(potential_words_local)
	num_relevant_words = len(relevant_words_local)
	num_other_words = len(other_words_local)
	num_all_words = num_relevant_words + num_other_words
	#TODO think about adjusting these probabilities to reflect the new model. Maybe split into three cases
	if random.random() < num_relevant_words * (1 - alpha) / (num_relevant_words + alpha * num_all_words):
		r = random.sample(potential_words_local, 1)
	else:
		r = random.sample(other_words_local, 1)
	return set(r)

cdef sample_base_utterance_old(state):
	"""
	:param relevant_words: words related to desired object
	:param other_words: words unrelated to desired object
	:return: single word response utterance
	"""
	# TODO: Potentially make this more realistic - Currently gives equal weight to all relevant words: shape, color, etc.
	# return nothing if the human doesn't speak
	if random.random() > p_l_b:
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
# 	if random.random() > p_l_b:
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
cpdef entropy(a):
	cdef int i
	cdef double ent = 0
	for i in range(len(a)):
		ent -= a[i] * math.log(a[i])
	return ent
cpdef kl_divergence2(a, b):
	return kl_divergence(a["desired_item"], b["desired_item"])
cpdef kl_divergence_bounded(a, b, bound):
	cdef int i
	cdef double div = 0
	for i in range(len(a)):
		try:
			div += a[i] * math.log(a[i] / b[i])
		except:
			print("a: " + str(a))
			print("b: " + str(b))
			raise ValueError("a[" + str(i) + "] = " + str(a[i]) + "; b[" + str(i) + "] = " + str(b[i]))
	return div
cpdef kl_divergence(list a, list b):
	cdef int i
	cdef double div = 0
	for i in range(len(a)):
		# if a[i] == 0.0:
		# 	a[i] = np.nextafter(0,1)
		# if b[i] == 0.0:
		# 	b[i] = np.nextafter(0,1)
		try:
			div += a[i] * math.log(a[i] / b[i])
		except:
			print("a: " + str(a))
			print("b: " + str(b))
			raise ValueError("a[" + str(i) + "] = " + str(a[i]) + "; b[" + str(i) + "] = " + str(b[i]))
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
cpdef remove_zeros(b):
	for i in range(len(b)):
		if b[i] == 0.0:
			b[i] = np.nextafter(0, 1)
	return b

def get_response_confusion_matrices():
	return response_confusion_matrices
cpdef get_max_angle(items):
	max_angle = 0
	for item1 in items():
		for item2 in items():
			max_angle = max(angle_between(item1["location"], item2["location"]), max_angle)
	print(max_angle)
cpdef clamp(value, min_value, max_value):
	if value > max_value:
		return max_value
	elif value < min_value:
		return min_value
	else:
		return value
cpdef argmax(args, fn):
	cdef double max_value = fn(args[0])
	cdef double cur_v
	maxarg = args[0]
	for i in range(len(args)):
		cur_v = fn(args[i])
		if cur_v > max_value:
			maxarg = args[i]
			max_value = cur_v
	return maxarg


cpdef argmax2(args, fn):
	'''
	:param args:
	:param fn:
	:return: (maxarg,max_value)
	'''
	cdef double max_value = fn(args[0])
	cdef double cur_v
	maxarg = args[0]
	for i in range(len(args)):
		cur_v = fn(args[i])
		if cur_v > max_value:
			maxarg = args[i]
			max_value = cur_v
	return (maxarg, max_value)


cpdef update_v(self, double convergence_threshold=1):
	'''
	Taken from algorithm 4 from Shani et al
	:return:
	'''
	# for i in range(depth):
	cdef double max_improvement = convergence_threshold + 0.1
	cdef int iterations = self.num_update_iterations
	cdef int num_updated = 0
	cdef list beliefs_to_update
	cdef list v_prime
	while max_improvement > convergence_threshold:
		beliefs_to_update = copy.deepcopy(self.beliefs)
		print("max_improvement: " + str(max_improvement))
		max_improvement = 0
		v_prime = []
		while len(beliefs_to_update) > 0:
			b = random.sample(beliefs_to_update, 1)[0]
			new_alpha = backup(self, b, self.v)
			current_best_alpha, current_value = self.get_best_alpha(b)
			new_value = self.alpha_dot_b(new_alpha, b)
			# Changed from >= to >
			if new_value > current_value:
				# TODO vectorize! This is currently a bottleneck and seems easy to parallelize
				beliefs_to_update.remove(b)
				alpha_b = new_alpha
				max_improvement = max(max_improvement, new_value - current_value)
			else:
				beliefs_to_update.remove(b)
				alpha_b = current_best_alpha
			# self.add_alpha(alpha_b)
			v_prime.append(alpha_b)
			num_updated += 1
			if num_updated % 100 == 0:
				self.pickle_partial_update(alphas = v_prime)
				# self.pickle_data(
				# 	name=self.name + " value iteration " + str(num_updated) + " beliefs updated time " + str(
				# 		datetime.now()).replace(":", ".")[:22])
		self.v = v_prime
		iterations += 1
		self.max_improvement = max_improvement
		self.pickle_data(
			name=self.name + " value iteration " + str(iterations) + " time " + str(datetime.now()).replace(":", ".")[:22])
		self.num_update_iterations = iterations
	return self.v
cpdef update_v_shani(self, double convergence_threshold=1):
	'''
	Taken from algorithm 4 from Shani et al
	:return:
	'''
	#TODO reread and fix this
	cdef double max_improvement = convergence_threshold + 0.1
	cdef int iterations = self.num_update_iterations
	cdef int num_updated = 0
	cdef list beliefs_to_update
	cdef list v_prime
	while max_improvement > convergence_threshold:
		beliefs_to_update = copy.deepcopy(self.beliefs)
		if iterations > 0:
			print("max_improvement: " + str(max_improvement))
		max_improvement = 0
		v_prime = []
		while len(beliefs_to_update) > 0:
			b = random.sample(beliefs_to_update, 1)[0]
			new_alpha = backup(self, b, self.v)
			current_best_alpha, current_value = self.get_best_alpha(b)
			new_value = self.alpha_dot_b(new_alpha, b)
			# Changed from >= to >
			if new_value > current_value:
				#Consider
				beliefs_to_update.remove(b)
				if len(beliefs_to_update) > 0:# and iterations > 0: #TODO Iteration restriction is hacky, find the correct solution
					beliefs_to_update, max_delta = self.remove_improved_beliefs(beliefs_to_update,new_alpha,self.v,convergence_threshold)
				else:
					max_delta = new_value - current_value
				# TODO vectorize! This is currently a bottleneck and seems easy to parallelize
				alpha_b = new_alpha
				max_improvement = max(max_improvement, max_delta)
			else:
				beliefs_to_update.remove(b)
				alpha_b = current_best_alpha
			# self.add_alpha(alpha_b)
			v_prime.append(alpha_b)
			num_updated += 1
			if num_updated % 100 == 0:
				self.pickle_partial_update(alphas = v_prime)
				# self.pickle_data(
				# 	name=self.name + " value iteration " + str(num_updated) + " beliefs updated time " + str(
				# 		datetime.now()).replace(":", ".")[:22])
		check_for_actions_in_alphas(self.pomdp.actions, v_prime)
		self.v = v_prime
		#TODO fix root cause instead of hacking pick actions back in. Tests show that each item has at least 47 beliefs which are > .95 confident in it, 30-70 > .99
		#Tests also show that all 69 pick actions pick 0 :/
		iterations += 1
		self.max_improvement = max_improvement
		self.pickle_data(
			name=self.name + " value iteration " + str(iterations) + " time " + str(datetime.now()).replace(":", ".")[:22])
		self.num_update_iterations = iterations
	return self.v
cpdef check_for_actions_in_alphas(actions,alphas):
	alphas_by_actions = {}
	missing_actions = []
	for a in actions:
		rel_alphas = [al for al in alphas if al[1] == a]
		alphas_by_actions[a] = rel_alphas
		if len(rel_alphas) == 0 and a.split(" ")[0] == "pick":
			print(a + " has " + str(len(rel_alphas)) + " alphas")
			missing_actions.append(a)
	if len(missing_actions) > 0:
		s = ""
		for a in missing_actions:
			s += a + "; "
		raise ValueError("Missing alphas for actions :" + s)
cpdef backup(self, belief, v, num_observations=3):
	# TODO check if it is can be more efficient to compute observation prob and/or new belief while sampling observation
	#TODO PROBLEM! I am taking alphas that maximize next belief to represent the value of the next belief, but I need to evaluate the alphas against the current belief.
	cdef list alphas = []
	cdef list observations
	cdef list observation_probs, norm_observation_probs
	cdef list new_beliefs
	cdef list alpha_a_os
	cdef list best_alpha_a_os_pairs
	for a in self.actions:
		if a not in self.pomdp.terminal_actions:
			observations = [self.pomdp.sample_observation_from_belief_action(belief, a) for i in
			                range(num_observations)]
			observation_probs = [self.pomdp.observation_from_belief_prob(o, belief, a) for o in observations]
			norm_observation_probs = unit_vectorn(observation_probs)
			new_beliefs = [self.pomdp.belief_update(belief, o, a) for o in observations]
			#TODO use * to unpack in one line zip unzip pack unpack google it
			obs_alpha_val_pairs = [self.get_best_alpha(b,v) for b in new_beliefs]
			obs_alphas = [pair[0] for pair in obs_alpha_val_pairs]
			alpha_a_os = get_alpha_a_os(self,a,observations)
			best_alpha_a_os_pairs = [self.get_best_alpha(belief,alphs) for alphs in alpha_a_os]
			best_alpha_a_os = [pair[0] for pair in best_alpha_a_os_pairs]
			best_alpha_a_os_values = [alph[0] for alph in best_alpha_a_os]
			value_of_next_beliefs = [pair[1] for pair in best_alpha_a_os_pairs]
			alpha_a_b_vals = linear_combination_of_lists(best_alpha_a_os_values, norm_observation_probs)
			a_val = add(self.reward_vectors[a], times_list(alpha_a_b_vals, self.pomdp.gamma))
			new_alpha = [a_val, a]
			alphas.append(new_alpha)
		else:
			#TODO replace with dictionary lookup for speed
			new_alpha = self.terminal_action_alphas[a]
	best_alpha, best_value = self.get_best_alpha(belief, alphas)
	return best_alpha
cpdef get_alpha_a_os(self,a,observations):
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
cpdef get_value_numpy(self,belief):
	alpha_vals = np.array([alpha[0] for alpha in self.v])
	belief_arr = np.array(self.pomdp.belief_to_array_of_probs(belief))
	values = np.dot(alpha_vals,belief_arr)
	max_value = np.max(values)
	return max_value

cpdef get_count_each_action(histories):
	counts = {"look": 0, "wait": 0, "point": 0, "pick": 0}
	for history in histories:
		for i in history:
			vals = i["action"].split(" ")
			counts[vals[0]] += 1
	return counts

cpdef get_count_each_action_precise(histories):
	counts = defaultdict(int)
	for history in histories:
		for i in history:
			counts[i["action"]] += 1
	keys = list(counts.keys())
	keys.sort()
	a = {}
	for key in keys:
		a[key] = counts[key]
	return a