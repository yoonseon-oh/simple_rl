#!/usr/bin/env python

# Python imports.
import sys

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.tasks import TaxiOOMDP
from simple_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.tasks.taxi.taxi_helpers import *
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.aa_helpers import *


def main(open_plot=True):
    # Taxi initial state attributes..
    agent = {"x":1, "y":1, "has_passenger":0}
    passengers = [{"x":3, "y":2, "dest_x":2, "dest_y":3, "in_taxi":0}]
    walls = []
    mdp = TaxiOOMDP(width=4, height=4, agent=agent, walls=walls, passengers=passengers)

    # Agents.
    ql_agent = QLearningAgent(actions=mdp.get_actions()) 
    rand_agent = RandomAgent(actions=mdp.get_actions())

    viz = False
    if viz:
        # Visualize Taxi.
        run_single_agent_on_mdp(ql_agent, mdp, episodes=50, steps=1000)
        mdp.visualize_agent(ql_agent)
    else:
        # Run experiment and make plot.
        run_agents_on_mdp([ql_agent, rand_agent], mdp, instances=10, episodes=1, steps=500, reset_at_terminal=True, open_plot=open_plot)

# ----------------------------
# -- Option Implementations --
# ----------------------------

def passenger_pickup(mdp, passenger):
    '''
    Args:
        mdp (TaxiOOMDP): MDP with the reward function leading to passenger pickup
        passenger (passenger): should have location and destination attributes
    Returns:
        option (Option): representing seq of actions to take to get to agent's location
    '''
    vi = ValueIteration(mdp)
    vi.run_vi()
    o_policy_dict = make_dict_from_lambda(vi.policy, vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)
    policy = o_policy.get_action

    init_predicate = Predicate(func=lambda state: not state.get_first_obj_of_class('agent')['has_passenger'])
    term_predicate = Predicate(func=lambda state: state.get_first_obj_of_class('agent')['has_passenger'])

    return Option(init_predicate=init_predicate, term_predicate=term_predicate, policy=policy, term_prob=0.0,
                  name='pickup_option')

def passenger_dropoff(mdp, passenger):
    '''
    Args:
        mdp (TaxiOOMDP): MDP with the reward function leading to passenger being dropped off a their destination
        passenger (passenger): should have location and destination attributes
    Returns:
        option (Option): representing seq of actions to take to get to agent's location
    '''
    vi = ValueIteration(mdp)
    vi.run_vi()
    o_policy_dict = make_dict_from_lambda(vi.policy, vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)
    policy = o_policy.get_action

    # pdb.set_trace()

    init_predicate = Predicate(func=lambda state: state.get_first_obj_of_class('agent')['has_passenger'])
    term_predicate = Predicate(func=lambda state: state.get_first_obj_of_class('agent')['x'] == passenger['dest_x'] and
                                                  state.get_first_obj_of_class('agent')['y'] == passenger['dest_y'] and
                                                  state.get_first_obj_of_class('agent')['has_passenger'] == False)

    return Option(init_predicate=init_predicate, term_predicate=term_predicate, policy=policy, term_prob=0.0,
                  name='dropoff_option')

def setup_taxi_mdp():
    agent = {"x": 1, "y": 1, "has_passenger": 0}
    passengers = [{"x": 3, "y": 2, "dest_x": 2, "dest_y": 3, "in_taxi": 0}]
    passenger = passengers[0]

    def passenger_pickup_reward_function(state, action):
        passenger = state.get_first_obj_of_class('passenger')
        passenger_location = (passenger['x'], passenger['y'])
        taxi_location = (state.get_agent_x(), state.get_agent_y())
        if taxi_location == passenger_location and action == 'pickup':
            return 1.
        return 0.

    def passenger_dropoff_reward_function(state, action):
        passenger = state.get_first_obj_of_class('passenger')
        agent = state.get_first_obj_of_class('agent')
        passenger_destination = (passenger['dest_x'], passenger['dest_y'])
        agent_location = (state.get_agent_x(), state.get_agent_y())
        if agent['has_passenger'] and agent_location == passenger_destination and action == 'dropoff':
            return 1.
        return 0.

    print '\nPickup Option:\n'
    pickup_mdp = TaxiOOMDP(width=4, height=4, agent=agent, walls=[], passengers=passengers,
                           reward_func=passenger_pickup_reward_function, terminal_func=passenger_pickup_terminal_state)
    pickup_option = passenger_pickup(pickup_mdp, passenger)
    pickup_end_state, _ = pickup_option.act_until_terminal(pickup_mdp.init_state, pickup_mdp.transition_func)

    obj_to_dict = lambda obj: obj.__dict__['attributes']
    passengers = map(obj_to_dict, pickup_end_state.get_objects_of_class('passenger'))
    agent = obj_to_dict(pickup_end_state.get_first_obj_of_class('agent'))

    print '\nDropoff Option:\n'
    dropoff_mdp = TaxiOOMDP(width=4, height=4, agent=agent, walls=[], passengers=passengers,
                            reward_func=passenger_dropoff_reward_function)
    dropoff_option = passenger_dropoff(dropoff_mdp, passenger)
    dropoff_option.act_until_terminal(dropoff_mdp.init_state, dropoff_mdp.transition_func, verbose=True)

    return pickup_mdp, pickup_option, dropoff_mdp, dropoff_option

def options_for_taxi_domain():
    _, pickup_option, _, dropoff_option = setup_taxi_mdp()
    return [pickup_option, dropoff_option]

if __name__ == "__main__":
    pickup_mdp, pickup_option, dropoff_mdp, dropoff_option = setup_taxi_mdp()
