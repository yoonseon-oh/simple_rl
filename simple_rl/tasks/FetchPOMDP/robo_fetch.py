from simple_rl.tasks.FetchPOMDP import FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver

pomdp = FetchPOMDP()
solver = FetchPOMDPSolver(pomdp)
# print(solver.act([0,0,0,1,0,0,"blue ball"]))
