from simple_rl.mdp.MDPClass import MDP
from simple_rl.pomdp.POMDPClass import POMDP

from collections import defaultdict

class BeliefMDP(MDP):
    def __init__(self, pomdp):
        '''
        Convert given POMDP to a Belief State MDP
        Args:
            pomdp (POMDP)
        '''
        self.state_transition_func = pomdp.transition_func
        self.state_reward_func = pomdp.reward_func
        self.state_observation_func = pomdp.observation_func
        self.belief_updater_func = pomdp.belief_updater_func

        MDP.__init__(self, pomdp.actions, self._belief_transition_function, self._belief_reward_function,
                     pomdp.init_belief, pomdp.gamma, pomdp.step_cost)

    def _belief_transition_function(self, belief, action):
        '''
        The belief MDP transition function T(b, a) --> b' is a generative function that given a belief state and an
        action taken from that belief state, returns the most likely next belief state
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            new_belief (defaultdict)
        '''
        observation = self._belief_observation_function(belief, action)
        return self.belief_updater_func(belief, action, observation)

    def _belief_reward_function(self, belief, action):
        '''
        The belief MDP reward function R(b, a) is the expected reward from the POMDP reward function
        over the belief state distribution.
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            reward (float)
        '''
        reward = 0.
        for state in belief:
            reward += belief[state] * self.state_reward_func(state, action)
        return reward

    def _belief_observation_function(self, belief, action):
        '''
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            observation (str): most probable observation given (b, a)
        '''
        most_probable_state = max(belief, key=belief.get)
        return self.state_observation_func(most_probable_state, action)
