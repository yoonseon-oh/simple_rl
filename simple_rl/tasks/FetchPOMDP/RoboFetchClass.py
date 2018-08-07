from simple_rl.tasks.FetchPOMDP import FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver

class RoboFetch(object):
	def __init__(self, use_look = True):
		self.pomdp = FetchPOMDP(use_look=use_look)
		self.solver = FetchPOMDPSolver(self.pomdp, muted = True)
	def act(self,observation):
		return self.solver.act(observation)