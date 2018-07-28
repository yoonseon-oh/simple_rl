from simple_rl.pomdp.BeliefStateClass import BeliefState
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP

class POMDP(MDP):
    def __init__(self, actions, transition_func, reward_func, observation_func, init_belief_state, gamma=0.99, step_cost=0):
        '''
        Args:
            actions (list)
            transition_func: T(s, a) -> s'
            reward_func: R(s, a) -> float
            observation_func: o(s, a) -> s
            init_belief_state (BeliefState)
            gamma (float)
            step_cost (int)
        '''
        sampled_init_state = init_belief_state.sample()
        self.observation_func = observation_func
        MDP.__init__(self, actions, transition_func, reward_func, sampled_init_state, gamma, step_cost)

    def update_belief(self, current_belief, observation, action):
        '''
        Args:
            current_belief (BeliefState)
            observation (State)
            action (str)

        Returns:
            next_belief (BeliefState)
        '''
        pass

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)

        Returns:
            reward (float)
            next_belief_state (BeliefState)
        '''
        pass

