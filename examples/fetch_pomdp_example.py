#!/usr/bin/env python

# Python imports.
import sys
from time import time
import json
# Other imports.
from simple_rl.tasks.FetchPOMDP import FetchPOMDP, cstuff
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
	with open('with_gesture v no_gesture unambiguous' + str(time()) + '.json', 'w') as fp:
		json.dump(results, fp)


def main(open_plot=True):
	# Setup MDP, Agents.
	# pomdp = FetchPOMDP()
	# solver = FetchPOMDPSolver(pomdp, horizon= 2)
	# solver.muted = False

    # Run experiment and make plot.
	# results = solver.run(num_episodes=100)
	# num_positive = len([i for i in results["final_scores"] if i > 0])
	# print("num_positive: " + str(num_positive))
	compare_gesture_no_gesture(1000)
	# pomdp = FetchPOMDP()
	# solver_sans_gesture = FetchPOMDPSolver(pomdp, horizon=2, use_gesture=False)
	# solver_sans_gesture.test_no_gesture()
if __name__ == "__main__":    main(open_plot=not sys.argv[-1] == "no_plot")
