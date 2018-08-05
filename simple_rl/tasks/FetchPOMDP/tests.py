from simple_rl.tasks.FetchPOMDP.FetchPOMDPClass import *
from time import time
import json

def average(a):
	avg = 0
	for i in a:
		avg += i
	avg /= len(a)
	return avg


def count_positive(a):
	return len([i for i in a if i > 0])


def count_negative(a):
	return len([i for i in a if i < 0])


def compare_classic_and_static_run(n=10):
	start_static = time()
	static_results = bs.run(num_episodes=n)
	time_static = time() - start_static

	start_classic = time()
	classic_results = bs.run(num_episodes=n)
	time_classic = time() - start_classic
	# print("static_state: "+ str(time_static) + "; "  + str(scores_static))
	# print("classic: " + str(time_classic) + "; " + str(scores_classic))

	results = {
		"classic_scores": {"average": average(classic_results[0]), "num_positive": count_positive(classic_results[0]),
		                   "all": classic_results[0]},
		"static_scores": {"average": average(static_results[0]), "num_positive":count_positive(static_results[0]),"all": static_results[0]},
		"classic_time": time_classic,
		"static_time": time_static, "average_actions_classic": float(classic_results[1]) / n,
		"average_actions_static": float(static_results[1]) / n}
	results = results.update(get_constants())
	with open('classic v static.json', 'w') as fp:
		json.dump(results, fp)

def get_constants():
	c = {"wait_cost":wait_cost,"point_cost":point_cost,"wrong_pick_cost":wrong_pick_cost,"correct_pick_reward":correct_pick_reward,"items":items,"discount":discount,"std_theta":std_theta,"bag_of_words":bag_of_words}
	return c
def compare_static_and_static2(n = 10):
	bs.static_horizon = 20
	start_static = time()
	static_results = bs.run(num_episodes=n)
	time_static = time() - start_static

	start_static2 = time()
	static2_results = bs.run2(num_episodes=n)
	time_static2 = time() - start_static2
	# print("static_state: "+ str(time_static) + "; "  + str(scores_static))
	# print("static2: " + str(time_static2) + "; " + str(scores_static2))

	results = {
		"static2_scores": {"average": average(static2_results[0]), "num_positive": count_positive(static2_results[0]),
		                   "all": static2_results[0]},
		"static_scores": {"average": average(static_results[0]), "num_positive":count_positive(static_results[0]),"all": static_results[0]},
		"static2_time": time_static2,
		"static_time": time_static, "average_actions_static2": float(static2_results[1]) / n,
		"average_actions_static": float(static_results[1]) / n}
	results.update(get_constants())
	# print(get_constants)
	print(results)
	with open('static v static2.json', 'w') as fp:
		json.dump(results, fp)

def compare_horizons(n = 50):
	bs.static_horizon = 2
	start_static = time()
	horizon_2_results = bs.run(num_episodes=n)
	time_static = time() - start_static

	bs.static_horizon = 10
	start_4 = time()
	horizon_4_results = bs.run(num_episodes=n)
	time_4 = time() - start_4


	results = {
		"horizon4_scores": {"average": average(horizon_4_results[0]), "num_positive": count_positive(horizon_4_results[0]),
		                   "all": horizon_4_results[0]},
		"horizon2_scores": {"average": average(horizon_2_results[0]), "num_positive":count_positive(horizon_2_results[0]),"all": horizon_2_results[0]},
		"time_4": time_4,
		"time_2": time_static, "average_actions_classic": float(horizon_4_results[1]) / n,
		"average_actions_static": float(horizon_2_results[1]) / n}
	with open('horizon_test.json', 'w') as fp:
		json.dump(results, fp)

def compare_gesture_no_gesture(n = 100):
	start_static = time()
	static_results = bs.run(num_episodes=n)
	time_static = time() - start_static

	start_static2 = time()
	static2_results = bs.run2(num_episodes=n)
	time_static2 = time() - start_static2
	# print("static_state: "+ str(time_static) + "; "  + str(scores_static))
	# print("static2: " + str(time_static2) + "; " + str(scores_static2))

	results = {
		"static2_scores": {"average": average(static2_results[0]), "num_positive": count_positive(static2_results[0]),
		                   "all": static2_results[0]},
		"static_scores": {"average": average(static_results[0]), "num_positive":count_positive(static_results[0]),"all": static_results[0]},
		"static2_time": time_static2,
		"static_time": time_static, "average_actions_static2": float(static2_results[1]) / n,
		"average_actions_static": float(static_results[1]) / n}
	results.update(get_constants())
	# print(get_constants)
	print(results)
	with open('static v static2.json', 'w') as fp:
		json.dump(results, fp)


compare_static_and_static2(100)
# compare_classic_and_static_run(100)