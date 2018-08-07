#!/usr/bin/env python

# Python imports.
import sys
from time import time
from datetime import datetime
import json
# Other imports.
from simple_rl.tasks.FetchPOMDP import FetchPOMDP, FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver

# TODO: Surprisingly long time between FetchPOMDPClass printing cstuff.get_items() and trials starting. Why?
# Running solvers with horizon 2 had poor (60%) results for 6 items (100 trials). Horizon 3 gave 90% without gestures, 70% with (10 trials)
# The old version was much better. Why?
# output_directory = "C:\\Users\\Brolly\\Documents\\simple_rl-FetchPOMDP\\simple_rl\\tasks\\FetchPOMDP\\FetchPOMDP Trials"
output_directory = ".\\FetchPOMDP Trials\\"


def get_full_path(file_name="Test results", ext=".json"):
	return output_directory + file_name + str(datetime.now()).replace(":", ".") + ext


def average(a):
	avg = 0
	for i in a:
		avg += i
	avg /= len(a)
	return avg


def count_positive(a):
	return len([i for i in a if i > 0])


def compare_gesture_no_gesture(n=100, horizon=2):
	pomdp = FetchPOMDP()
	# create solver
	start = time()
	solver_with_gesture = FetchPOMDPSolver(pomdp, horizon=horizon)
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver_with_gesture.muted = False
	# solve
	start = time()
	with_gesture_results = solver_with_gesture.run(num_episodes=n)
	with_gesture_time_elapsed = time() - start
	print(" ")

	# create solver
	start = time()
	solver_sans_gesture = FetchPOMDPSolver(pomdp, horizon=horizon, use_gesture=False)
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver_sans_gesture.muted = False
	# solve
	start = time()
	sans_gesture_results = solver_sans_gesture.run(num_episodes=n)
	sans_gesture_time_elapsed = time() - start

	# print("with_gesture_state: "+ str(with_gesture_time_elapsed) + "; "  + str(scores_with_gesture))
	# print("no_gesture: " + str(sans_gesture_time_elapsed) + "; " + str(scores_no_gesture))
	print("sans_gesture_results:")
	print(str(sans_gesture_results))
	print("with_gesture_results:")
	print(str(with_gesture_results))
	results = {"horizon": horizon,
	           "no_gesture_scores": {"average": average(sans_gesture_results["final_scores"]),
	                                 "num_correct": sans_gesture_results["num_correct"],
	                                 "all": sans_gesture_results["final_scores"]},
	           "with_gesture_scores": {"average": average(with_gesture_results["final_scores"]),
	                                   "num_correct": with_gesture_results["num_correct"],
	                                   "all": with_gesture_results["final_scores"]},
	           "no_gesture_time": sans_gesture_time_elapsed,
	           "with_gesture_time": with_gesture_time_elapsed,
	           "average_actions_no_gesture": float(sans_gesture_results["counter_plan_from_state"]) / n,
	           "average_actions_with_gesture": float(with_gesture_results["counter_plan_from_state"]) / n}
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(output_directory + 'with_gesture v no_gesture' + str(time()) + '.json', 'w') as fp:
		json.dump(results, fp)


def test(obs_mod1=(True, True), n=100, horizon=2):
	'''
	:param obs_mod1: (use_gesture, use_language)
	:param obs_mod2: (use_gesture, use_language)
	:param n: Number of trials
	:param horizon: depth of forward search
	:return: None. prints and writes results in a json file.
	'''
	solve1_name = "(" + ""

	pomdp = FetchPOMDP()
	# create solver
	start = time()
	solver1 = FetchPOMDPSolver(pomdp, horizon=horizon, use_gesture=obs_mod1[0], use_language=obs_mod1[1])
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver1.muted = False
	# solve
	start = time()
	solver1_results = solver1.run(num_episodes=n)
	solver1_time_elapsed = time() - start
	print(" ")


	results = {"horizon": horizon,
	           "solver1": {"use_gesture":obs_mod1[0], "use_language":obs_mod1[1],
	                       "time": solver1_time_elapsed,
	                       "average_actions": float(solver1_results["counter_plan_from_state"]) / n,
	                       "average": average(solver1_results["final_scores"]),
	                       "num_correct": solver1_results["num_correct"], "all": solver1_results["final_scores"]}}
	print("Results:" + "\n" +str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("compare observation models "), 'w') as fp:
		json.dump(results, fp)

def compare_observation_models(obs_mod1=(True, True), obs_mod2=(True, True), n=100, horizon=2):
	'''
	:param obs_mod1: (use_gesture, use_language)
	:param obs_mod2: (use_gesture, use_language)
	:param n: Number of trials
	:param horizon: depth of forward search
	:return: None. prints and writes results in a json file.
	'''
	solve1_name = "(" + ""

	pomdp = FetchPOMDP()
	# create solver
	start = time()
	solver1 = FetchPOMDPSolver(pomdp, horizon=horizon, use_gesture=obs_mod1[0], use_language=obs_mod1[1])
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver1.muted = False
	# solve
	start = time()
	solver1_results = solver1.run(num_episodes=n)
	solver1_time_elapsed = time() - start
	print(" ")

	# create solver
	start = time()
	solver2 = FetchPOMDPSolver(pomdp, horizon=horizon, use_gesture=obs_mod2[0], use_language=obs_mod2[1])
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver2.muted = False
	# solve
	start = time()
	solver2_results = solver2.run(num_episodes=n)
	solver2_time_elapsed = time() - start

	results = {"horizon": horizon,
	           "solver1": {"use_gesture":obs_mod1[0], "use_language":obs_mod1[1],
	                       "time": solver1_time_elapsed,
	                       "average_actions": float(solver1_results["counter_plan_from_state"]) / n,
	                       "average": average(solver1_results["final_scores"]),
	                       "num_correct": solver1_results["num_correct"], "all": solver1_results["final_scores"]},
	           "solver2": {"use_gesture":obs_mod2[0], "use_language":obs_mod2[1],
	                       "time": solver2_time_elapsed,
	                       "average_actions": float(solver2_results["counter_plan_from_state"]) / n,
	                       "average": average(solver2_results["final_scores"]),
	                       "num_correct": solver2_results["num_correct"],
	                       "all": solver2_results["final_scores"]}}
	print("Results:" + "\n" +str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("compare observation models "), 'w') as fp:
		json.dump(results, fp)

def test_gestureless_pomdp(n = 10, horizon  = 2):
	pomdp = FetchPOMDP(use_gesture=False)
	pomdp.point_cost = pomdp.wait_cost
	solver  = FetchPOMDPSolver(pomdp,use_gesture=False, use_language=True, qvalue_method="belief based", horizon= horizon)
	solver.muted = False
	# solve
	start = time()
	solver_results = solver.run(num_episodes=n)
	solver_time_elapsed = time() - start
	results = {"horizon": horizon,
	           "solver": {"time": solver_time_elapsed,
	                       "average_actions": float(solver_results["counter_plan_from_state"]) / n,
	                       "average": average(solver_results["final_scores"]),
	                       "num_correct": solver_results["num_correct"], "all": solver_results["final_scores"]}}
	print("Results:" + "\n" +str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("gestureless results "), 'w') as fp:
		json.dump(results, fp)


def test_belief_state_class():
	pomdp = FetchPOMDP()
	solver = FetchPOMDPSolver(pomdp, horizon=2)
	print("pomdp.curr_belief_state[0]: " + str(pomdp.curr_belief_state[0]))
	print("pomdp.curr_belief_state[1]: " + str(pomdp.curr_belief_state[1]))
	solver.plan_from_belief(pomdp.curr_belief_state)

def test_arguments():
	pomdp = FetchPOMDP(use_gesture=False)
	for i in range(100):
		o = pomdp.sample_observation(pomdp.curr_state)
		if type(o['gesture']) is not type(None):
			print(o)
def test_heuristic_planner(n = 100, horizon = 2, obs_mod = (True,True)):
	pomdp = FetchPOMDP(use_gesture=obs_mod[0],use_language=obs_mod[1])
	pomdp.point_cost = pomdp.wait_cost
	solver = FetchPOMDPSolver(pomdp, use_gesture=obs_mod[0], use_language=obs_mod[1], qvalue_method="belief based",
	                          horizon=horizon, planner="heuristic")
	solver.muted = False
	start = time()
	solver_results = solver.run(num_episodes=n)
	solver_time_elapsed = time() - start
	results = {"horizon": horizon,
	           "solver": {"time": solver_time_elapsed,
	                       "average_actions": float(solver_results["counter_plan_from_state"]) / n,
	                       "average": average(solver_results["final_scores"]),
	                       "solver_results":solver_results}}
	print("Results:" + "\n" +str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("heuristic results "), 'w') as fp:
		json.dump(results, fp, indent=4)
def format_results(results,time_elapsed, horizon,n):
	f_results = {"horizon": horizon,
	           "solver": {"time": time_elapsed,
	                       "average_actions": float(results["counter_plan_from_state"]) / n,
	                       "average": average(results["final_scores"]),
	                       "solver_results":results}}
def main(open_plot=True):
	# compare_observation_models((True, True), (False, False), n=10)
	# test_gestureless_pomdp(10)
	# test_arguments()
	# test_gestureless_pomdp(10, horizon=3)
	test_heuristic_planner(n =100)

	# test((True,True), n=10)

if __name__ == "__main__":    main(open_plot=not sys.argv[-1] == "no_plot")
