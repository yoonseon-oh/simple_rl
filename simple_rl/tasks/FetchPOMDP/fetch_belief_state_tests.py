from simple_rl.pomdp.BeliefStateClass import FlatFiniteBeliefState
from simple_rl.tasks.FetchPOMDP.FetchPOMDPClass import FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver
pomdp =

belief_state = FlatFiniteBeliefState(self.init_state[1], [1.0 / len(self.items) for i in range(self.num_items)])