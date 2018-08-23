import pickle, os
from simple_rl.tasks.FetchPOMDP import FetchPOMDP
# from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver
from simple_rl.tasks.FetchPOMDP.PBVIClassVectorized import Perseus2
class RoboFetch(object):
	def __init__(self, use_look = True):
		self.pomdp = FetchPOMDP(use_look=use_look)
		# self.solver = FetchPOMDPSolver(self.pomdp, muted = True)
		pickle_directory = os.path.dirname(os.path.realpath(__file__)) + "/special pickles/"
		pickle_name = "6 items/Perseus2 value iteration 5 time 2018-08-22 13.06.46.72.pickle2"
		print("RoboFetch CWD:")
		print(os.getcwd())
		p = pickle.load(open(pickle_directory + pickle_name, "rb"))
		self.solver = Perseus2(self.pomdp, pickle = p)
		self.solver.name = "Perseus2 On Baxter"
	def act(self,observation):
		return self.solver.act(observation)