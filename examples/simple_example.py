#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PredicateClass import Predicate

def create_grid_word_option():
    init_predicate = Predicate(func=lambda s: s.x < 3. and s.y < 3.)
    term_predicate = Predicate(func=lambda s: s.x == 3. and s.y == 3.)
    policy = {GridWorldState(1., 1.): 'right', GridWorldState(2., 1.): 'right', GridWorldState(3., 1.): 'up',
              GridWorldState(1., 2.): 'right', GridWorldState(2., 2.): 'right', GridWorldState(2., 3.): 'up',
              GridWorldState(1., 3.): 'right', GridWorldState(2., 2.): 'right'}
    o = Option(init_predicate, term_predicate=term_predicate, policy=policy, name='GridWorldOption1')
    return o

def main(open_plot=True):
    # Setup MDP.
    mdp = GridWorldMDP(width=3, height=3, init_loc=(1, 1), goal_locs=[(3, 3)], lava_locs=[], gamma=0.99, walls=[])

    # For testing purposes, lets cheat and create a great option that any agent should take
    mdp.actions.append(create_grid_word_option())

    # Make agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions())
    rand_agent = RandomAgent(actions=mdp.get_actions())

    # Run experiment and make plot.
    run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=5, episodes=50, steps=25, open_plot=open_plot,
                      track_disc_reward=False, verbose=True)

if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
