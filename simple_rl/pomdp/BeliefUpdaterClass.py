from collections import defaultdict

class BeliefUpdater(object):
    ''' Wrapper class for different methods for belief state updates in POMDPs. '''

    def __init__(self, transition_func, reward_func, observation_func, updater_type='discrete'):
        '''
        Args:
            transition_func: T(s, a) --> s'
            reward_func: R(s, a) --> float
            observation_func: O(s) --> z
            updater_type (str)
        '''
        self.reward_func = reward_func
        self.updater_type = updater_type

        self.transition_probs = self.construct_transition_matrix(transition_func)
        self.observation_probs = self.construct_observation_matrix(observation_func)

        if updater_type == 'discrete':
            self.updater = self.discrete_filter_updater
        elif updater_type == 'kalman':
            self.updater = self.kalman_filter_updater
        elif updater_type == 'particle':
            self.updater = self.particle_filter_updater
        else:
            raise AttributeError('updater_type {} did not conform to expected type'.format(updater_type))

    def discrete_filter_updater(self, belief, action, observation):
        pass

    def kalman_filter_updater(self, belief, action, observation):
        pass

    def particle_filter_updater(self, belief, action, observation):
        pass

    def construct_transition_matrix(self, transition_func):
        '''
        Create an MLE of the transition probabilities by sampling from the transition_func
        multiple times.
        Args:
            transition_func: T(s, a) -> s'

        Returns:
            transition_probabilities (defaultdict): T(s, a, s') --> float
        '''
        pass

    def construct_observation_matrix(self, observation_func):
        '''
        Create an MLE of the observation probabilities by sampling from the observation_func
        multiple times.
        Args:
            observation_func: O(s) -> z

        Returns:
            observation_probabilities (defaultdict): O(s, z) --> float
        '''
        pass