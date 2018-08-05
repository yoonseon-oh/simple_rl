#!/usr/bin/env python

# Python imports.
import sys
# Other imports.
from simple_rl.tasks.FetchPOMDP.FetchPOMDPClass import FetchPOMDP
from simple_rl.planning.FetchPOMDPSolver import FetchPOMDPSolver


def main(open_plot=True):
	# Setup MDP, Agents.
	pomdp = FetchPOMDP()
	solver = FetchPOMDPSolver(pomdp, horizon= 2)
	solver.muted = False

    # Run experiment and make plot.
	results = solver.run(num_episodes=5)
	print(results)
if __name__ == "__main__":    main(open_plot=not sys.argv[-1] == "no_plot")
