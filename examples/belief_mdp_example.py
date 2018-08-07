from simple_rl.tasks.maze_1d.Maze1DPOMDPClass import Maze1DPOMDP
from simple_rl.pomdp.BeliefMDPClass import BeliefMDP
from simple_rl.planning.BeliefSparseSamplingClass import BeliefSparseSampling

pomdp = Maze1DPOMDP()
belief_mdp = BeliefMDP(pomdp)

bss = BeliefSparseSampling(gen_model=belief_mdp, gamma=0.6, tol=1.0, max_reward=1.0, state=belief_mdp.init_state)
scores, policies = bss.run(num_episodes=3, verbose=True)
