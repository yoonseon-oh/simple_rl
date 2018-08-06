#!/usr/bin/env python

# Python imports.
import sys
from time import time
import json
# Other imports.
from simple_rl.tasks.FetchPOMDP import FetchPOMDP, FetchPOMDP, cstuff
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver
#TODO: Surprisingly long time between FetchPOMDPClass printing cstuff.get_items() and trials starting. Why?
#Running solvers with horizon 2 had poor (60%) results for 6 items (100 trials). Horizon 3 gave 90% without gestures, 70% with (10 trials)
#The old version was much better. Why?
def average(a):
	avg = 0
	for i in a:
		avg += i
	avg /= len(a)
	return avg
def count_positive(a):
	return len([i for i in a if i > 0])

def compare_gesture_no_gesture(n = 100, horizon = 2):
	pomdp = FetchPOMDP()
	#create solver
	start = time()
	solver_with_gesture = FetchPOMDPSolver(pomdp, horizon= horizon)
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver_with_gesture.muted = False
	#solve
	start = time()
	with_gesture_results = solver_with_gesture.run(num_episodes=n)
	with_gesture_time_elapsed = time() - start
	print(" ")

	#create solver
	start = time()
	solver_sans_gesture = FetchPOMDPSolver(pomdp,horizon=horizon,use_gesture=False)
	solver_creation_time = time() -start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver_sans_gesture.muted = False
	#solve
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
		"no_gesture_scores": {"average": average(sans_gesture_results["final_scores"]), "num_correct": sans_gesture_results["num_correct"],
		                   "all": sans_gesture_results["final_scores"]},
		"with_gesture_scores": {"average": average(with_gesture_results["final_scores"]), "num_correct": with_gesture_results["num_correct"],"all": with_gesture_results["final_scores"]},
		"no_gesture_time": sans_gesture_time_elapsed,
		"with_gesture_time": with_gesture_time_elapsed, "average_actions_no_gesture": float(sans_gesture_results["counter_plan_from_state"]) / n,
		"average_actions_with_gesture": float(with_gesture_results["counter_plan_from_state"]) / n}
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open('with_gesture v no_gesture' + str(time()) + '.json', 'w') as fp:
		json.dump(results, fp)

def compare_observation_models(obs_mod1 =(True,True), obs_mod2 = (True,True), n = 100, horizon = 2):
	'''
	:param obs_mod1: (use_gesture, use_language)
	:param obs_mod2: (use_gesture, use_language)
	:param n: Number of trials
	:param horizon: depth of forward search
	:return: None. prints and writes results in a json file.
	'''
	solve1_name = "(" + ""

	pomdp = FetchPOMDP()
	#create solver
	start = time()
	solver1 = FetchPOMDPSolver(pomdp, horizon= horizon, use_gesture=obs_mod1[0], use_language= obs_mod1[1])
	solver_creation_time = time() - start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver1.muted = False
	#solve
	start = time()
	solver1_results = solver1.run(num_episodes=n)
	solver1_time_elapsed = time() - start
	print(" ")

	#create solver
	start = time()
	solver2 = FetchPOMDPSolver(pomdp,horizon=horizon,use_gesture=obs_mod2[0], use_language=obs_mod2[1])
	solver_creation_time = time() -start
	print("Created solver in " + str(solver_creation_time) + " seconds")
	solver2.muted = False
	#solve
	start = time()
	solver2_results = solver2.run(num_episodes=n)
	solver2_time_elapsed = time() - start

	# print("solver1_state: "+ str(solver1_time_elapsed) + "; "  + str(scores_solver1))
	# print("no_gesture: " + str(solver2_time_elapsed) + "; " + str(scores_no_gesture))
	print("solver2_results:")
	print(str(solver2_results))
	print("solver1_results:")
	print(str(solver1_results))
	results = {"horizon": horizon,
		"no_gesture_scores": {"average": average(solver2_results["final_scores"]), "num_correct": solver2_results["num_correct"],
		                   "all": solver2_results["final_scores"]},
		"solver1_scores": {"average": average(solver1_results["final_scores"]), "num_correct": solver1_results["num_correct"],"all": solver1_results["final_scores"]},
		"no_gesture_time": solver2_time_elapsed,
		"solver1_time": solver1_time_elapsed, "average_actions_no_gesture": float(solver2_results["counter_plan_from_state"]) / n,
		"average_actions_solver1": float(solver1_results["counter_plan_from_state"]) / n}
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open('solver1 v no_gesture' + str(time()) + '.json', 'w') as fp:
		json.dump(results, fp)

def test_belief_state_class():
	pomdp = FetchPOMDP()
	solver = FetchPOMDPSolver(pomdp, horizon= 2)
	print("pomdp.curr_belief_state[0]: " + str(pomdp.curr_belief_state[0]))
	print("pomdp.curr_belief_state[1]: " + str(pomdp.curr_belief_state[1]))
	solver.plan_from_belief(pomdp.curr_belief_state)
def main(open_plot=True):
	compare_observation_models((True,True),(False,True),n=100)
if __name__ == "__main__":    main(open_plot=not sys.argv[-1] == "no_plot")
