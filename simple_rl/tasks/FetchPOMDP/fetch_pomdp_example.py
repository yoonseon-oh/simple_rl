#!/usr/bin/env python

# Python imports.
import sys
from time import time
from datetime import datetime
import json
import pickle
# Other imports.
from simple_rl.tasks.FetchPOMDP import FetchPOMDP, FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver
from simple_rl.tasks.FetchPOMDP import file_reader as fr
from simple_rl.planning.BeliefSparseSamplingClass import BeliefSparseSampling
from simple_rl.pomdp.BeliefMDPClass import BeliefMDP
from simple_rl.tasks.FetchPOMDP.PBVIClass import Perseus
from simple_rl.tasks.FetchPOMDP.PBVIClassVectorized import Perseus2, PBVIClassic2
# TODO try bss
# TODO: Surprisingly long time between FetchPOMDPClass printing cstuff.get_items() and trials starting. Why?
# Running solvers with horizon 2 had poor (60%) results for 6 items (100 trials). Horizon 3 gave 90% without gestures, 70% with (10 trials)
# The old version was much better. Why?
# output_directory = "C:\\Users\\Brolly\\Documents\\simple_rl-FetchPOMDP\\simple_rl\\tasks\\FetchPOMDP\\FetchPOMDP Trials"
output_directory = ".\\FetchPOMDP Trials\\"
pickle_directory = ".\\PBVIPickles\\"

def get_full_path(file_name="Test results", ext=".json"):
	return output_directory + file_name + str(datetime.now()).replace(":", ".")[:22] + ext


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





def gestureless_pomdp(n=10, horizon=2):
	pomdp = FetchPOMDP(use_gesture=False)
	pomdp.point_cost = pomdp.wait_cost
	solver = FetchPOMDPSolver(pomdp, use_gesture=False, use_language=True, qvalue_method="belief based",
	                          horizon=horizon)
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
	print("Results:" + "\n" + str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("gestureless results "), 'w') as fp:
		json.dump(results, fp)


def heuristic_planner(n=100, horizon=2, obs_mod=(True, True)):
	pomdp = FetchPOMDP(use_gesture=obs_mod[0], use_language=obs_mod[1])
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
	                      "solver_results": solver_results}}
	print("Results:" + "\n" + str(results))
	results.update(pomdp.get_constants())
	# print(get_constants)
	print(results)
	with open(get_full_path("heuristic results "), 'w') as fp:
		json.dump(results, fp, indent=4)


def format_results(results, time_elapsed, horizon, n):
	f_results = {"horizon": horizon,
	             "solver": {"time": time_elapsed,
	                        "average_actions": float(results["counter_plan_from_state"]) / n,
	                        "average": average(results["final_scores"]),
	                        "solver_results": results}}


def custom(pomdp1_args, pomdp2_args, solver1_args, solver2_args, n=10):
	pomdp1 = FetchPOMDP(**pomdp1_args)
	solver1 = FetchPOMDPSolver(pomdp1, **solver1_args)
	pomdp2 = FetchPOMDP(**pomdp2_args)
	solver2 = FetchPOMDPSolver(pomdp2, **solver2_args)
	results1 = solver1.run(num_episodes=n)
	results2 = solver2.run(num_episodes=n)
	results1.update({"args": {"pomdp": pomdp1_args, "solver": solver1_args}, "config": pomdp1.config,
	                 "action_counts": get_count_each_action(results1["histories"])})
	results2.update({"args": {"pomdp": pomdp2_args, "solver": solver2_args}, "config": pomdp2.config,
	                 "action_counts": get_count_each_action(results2["histories"])})
	results = {"1": results1, "2": results2}
	print("solver1 %" + str(100 * float(results1["num_correct"]) / n))
	print("solver1 action counts: " + str(results1["action_counts"]))
	print("solver2 %" + str(100 * float(results2["num_correct"]) / n))
	print("solver2 action counts: " + str(results2["action_counts"]))

	with open(get_full_path("custom test "), 'w') as fp:
		json.dump(results, fp, indent=4)
	return results


def custom_run(pomdp1_args, solver1_args, n=10):
	pomdp1 = FetchPOMDP(**pomdp1_args)
	solver1 = FetchPOMDPSolver(pomdp1, **solver1_args)
	results1 = solver1.run(num_episodes=n)
	num_belief_updates, num_impossible_observations = pomdp1.get_num_belief_updates_and_impossible_observations()
	impossible_observations_rate = float(num_impossible_observations) / float(num_belief_updates)
	results1.update({"args": {"pomdp": pomdp1_args, "solver": solver1_args}, "config": pomdp1.config,
	                 "action_counts": get_count_each_action(results1["histories"]),
	                 "impossible_observation_rate": impossible_observations_rate})
	results = {"1": results1}
	print("solver1 %" + str(100 * float(results1["num_correct"]) / n))
	print("solver1 action counts: " + str(results1["action_counts"]))
	print("impossible_observations_rate: " + str(impossible_observations_rate))
	with open(get_full_path("custom test "), 'w') as fp:
		json.dump(results, fp, indent=4)
	return results

def perseus_run(pomdp1_args,solver1_args, n = 10):
	pomdp1 = FetchPOMDP(**pomdp1_args)
	# p = pickle.load(open(pickle_directory + "value iteration 1time 2018-08-21 21.00.15.60.pickle", "rb"))
	# p = pickle.load(open(pickle_directory + "Perseus lab items value iteration 4 time 2018-08-22 09.39.27.67.pickle", "rb"))
	# solver1 = Perseus2(pomdp1, **solver1_args, pickle = p)
	solver1 = Perseus2(pomdp1, **solver1_args, name="Perseus new lab 1")
	solver1.update_v()
	# values1 = solver1.evaluate_alphas_at_beliefs(solver1.v,solver1.beliefs)
	# values2 = [solver1.get_value(b) for b in solver1.beliefs]
	# print("cat")
	results1 = solver1.run(num_episodes=n)
	num_belief_updates, num_impossible_observations = pomdp1.get_num_belief_updates_and_impossible_observations()
	impossible_observations_rate = float(num_impossible_observations) / float(num_belief_updates)
	results1.update({"args": {"pomdp": pomdp1_args, "solver": solver1_args}, "config": pomdp1.config,
	                 "action_counts": get_count_each_action(results1["histories"]),
	                 "impossible_observation_rate": impossible_observations_rate})
	results = {"1": results1}
	print("solver1 %" + str(100 * float(results1["num_correct"]) / n))
	print("solver1 action counts: " + str(results1["action_counts"]))
	print("impossible_observations_rate: " + str(impossible_observations_rate))
	with open(get_full_path("Perseus test "), 'w') as fp:
		json.dump(results, fp, indent=4)
	return results
def pbvi_run(pomdp1_args,solver1_args, n = 10):
	pomdp1 = FetchPOMDP(**pomdp1_args)
	# p = pickle.load(open(pickle_directory + "value iteration 1time 2018-08-21 21.00.15.60.pickle", "rb"))
	# p = pickle.load(open(pickle_directory + "Perseus lab items value iteration 4 time 2018-08-22 09.39.27.67.pickle", "rb"))
	# solver1 = Perseus2(pomdp1, **solver1_args, pickle = p)
	solver1 = PBVIClassic2(pomdp1, **solver1_args, name="PBVI useless look")
	# solver1.update_v()
	# values1 = solver1.evaluate_alphas_at_beliefs(solver1.v,solver1.beliefs)
	# values2 = [solver1.get_value(b) for b in solver1.beliefs]
	# print("cat")
	results1 = solver1.run(num_episodes=n)
	num_belief_updates, num_impossible_observations = pomdp1.get_num_belief_updates_and_impossible_observations()
	impossible_observations_rate = float(num_impossible_observations) / float(num_belief_updates)
	results1.update({"args": {"pomdp": pomdp1_args, "solver": solver1_args}, "config": pomdp1.config,
	                 "action_counts": get_count_each_action(results1["histories"]),
	                 "impossible_observation_rate": impossible_observations_rate})
	results = {"1": results1}
	print("solver1 %" + str(100 * float(results1["num_correct"]) / n))
	print("solver1 action counts: " + str(results1["action_counts"]))
	print("impossible_observations_rate: " + str(impossible_observations_rate))
	with open(get_full_path("Perseus test "), 'w') as fp:
		json.dump(results, fp, indent=4)
	return results

def balance_point_and_look():
	for i in range(10):
		config = fr.load_json("config.json")
		config["look_cost"] -= 0.1
		config["point_cost"] += 0.1
		with open('config.json', 'w') as fp:
			json.dump(config, fp)
		res = custom_run({"use_look": True},
		                 {"horizon": 2, "qvalue_method": "belief based", "muted": False}, n=10)
		action_counts = res["1"]["action_counts"]
		if action_counts["look"] > 0 and action_counts["point"] > 0:
			print("look and point")
			print("look cost: " + str(config["look_cost"]))
			print("point cost: " + str(config["point_cost"]))
			with open('config look and point' + str(datetime.now()).replace(":", ".") + '.json', 'w') as fp:
				json.dump(config, fp, indent=4)


def get_count_each_action(histories):
	counts = {"look": 0, "wait": 0, "point": 0, "pick": 0}
	for history in histories:
		for i in history:
			vals = i["action"].split(" ")
			counts[vals[0]] += 1
	return counts


def bss(pomdp1_args, num_episodes=5):
	pomdp1 = FetchPOMDP(**pomdp1_args)
	bmdp = BeliefMDP(pomdp1)
	solver = BeliefSparseSampling(bmdp, pomdp1.gamma, 5.0, pomdp1.correct_pick_reward, bmdp.init_state)
	scores, policies = solver.run(num_episodes=num_episodes, verbose=True)


# def main(open_plot=True):
# 	# compare_observation_models((True, True), (False, False), n=10)
# 	# test_gestureless_pomdp(10)
# 	# test_arguments()
# 	# test_gestureless_pomdp(10, horizon=3)
# 	# test_heuristic_planner(n =1000)
# 	# balance_point_and_look()
# 	# test((True,True), n=10)
# 	custom_run({"use_look": True},
# 	           {"horizon": 2, "qvalue_method": "belief based", "observation_branching": 10, "muted": False}, n = 10)
#
# 	# custom_test({"use_look": True},{"use_look": True},
# 	#            {"horizon": 2, "qvalue_method": "belief based", "muted": False, "kl_weight":0},
# 	#             {"horizon": 2, "qvalue_method": "belief based", "muted": False,"kl_weight":0}, n =10)
# 	# bss({"use_look": True})
# 	if __name__ == "__main__":    main(open_plot=not sys.argv[-1] == "no_plot")
# custom_run({"use_look": True},
# 	           {"horizon": 2, "qvalue_method": "belief based", "observation_branching": 1, "muted": False}, n = 10)

perseus_run({"use_look": True},
            {"num_beliefs":500, "belief_depth":3, "observations_sample_size":3, "convergence_threshold":.2}, n = 100)
# pbvi_run({"use_look": True},
#             {"observations_sample_size":3, "convergence_threshold":.2}, n = 100)