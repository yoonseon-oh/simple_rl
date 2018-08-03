from simple_rl.pomdp.BeliefUpdaterClass import BeliefUpdater
from simple_rl.mdp.MDPClass import MDP

from collections import defaultdict

class POMDP(MDP):
    ''' Abstract class for a Partially Observable Markov Decision Process. '''

    def __init__(self, actions, transition_func, reward_func, observation_func,
                 init_belief, belief_updater_type, gamma=0.99, step_cost=0):
        '''
        In addition to the input parameters needed to define an MDP, the POMDP
        definition requires an observation function, a way to update the belief
        state and an initial belief.
        Args:
            actions (list)
            transition_func: T(s, a) -> s'
            reward_func: R(s, a) -> float
            observation_func: O(s) -> z
            init_belief (defaultdict): initial probability distribution over states
            belief_updater_type (str): discrete/kalman/particle
            gamma (float)
            step_cost (int)
        '''
        self.observation_func = observation_func
        self.belief_updater = BeliefUpdater(transition_func, reward_func, observation_func, belief_updater_type).updater
        self.curr_belief = init_belief

        # init_belief_state = BeliefState(data=init_belief.values())
        sampled_init_state = max(init_belief, key=init_belief.get)
        MDP.__init__(self, actions, transition_func, reward_func, sampled_init_state, gamma, step_cost)

    def get_curr_belief(self):
        return self.curr_belief

    def get_observation_func(self):
        '''
        Returns:
            observation_function: O(s, a) -> o
        '''
        return self.observation_func

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            reward (float)
            next_belief (defaultdict)
        '''
        reward = self.belief_reward_func(self.curr_belief, action)
        observation = self.belief_observation_func(self.curr_belief, action)
        new_belief = self.belief_updater(self.curr_belief, observation, action)

        self.curr_belief = new_belief
        return reward, new_belief

    def belief_reward_func(self, belief, action):
        '''
        Convert R(s, a) to R(b, a) by taking an expectation over the belief states
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            reward (float): R(b, a)
        '''
        reward = 0.
        for state in belief:
            reward += belief[state] * self.reward_func(state, action)
        return reward

    def belief_observation_func(self, belief, action):
        '''
        Simulate the POMDP providing the agent an observation by sampling from the
        domain's observation function
        Args:
            belief (defaultdict)
            action (str)

        Returns:
            observation (str)
        '''
        most_probable_state = max(belief, key=belief.get)
        return self.observation_func(most_probable_state, action)


