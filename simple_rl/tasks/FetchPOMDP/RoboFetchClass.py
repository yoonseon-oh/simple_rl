from simple_rl.tasks.FetchPOMDP import FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver

class RoboFetch(object):
	def __init__(self):
		self.pomdp = FetchPOMDP()
		self.solver = FetchPOMDPSolver(self.pomdp)
	def act(self,observation):
		return self.solver.act(observation)