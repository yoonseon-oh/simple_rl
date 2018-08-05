from collections import defaultdict
from simple_rl.planning.ValueIterationClass import ValueIteration

import pdb

class BeliefUpdater(object):
    ''' Wrapper class for different methods for belief state updates in POMDPs. '''

    def __init__(self, mdp, transition_func, reward_func, observation_func, updater_type='discrete'):
        '''
        Args:
            mdp (POMDP)
            transition_func: T(s, a) --> s'
            reward_func: R(s, a) --> float
            observation_func: O(s) --> z
            updater_type (str)
        '''
        self.reward_func = reward_func
        self.updater_type = updater_type

        # We use the ValueIteration class to construct the transition and observation probabilities
        self.vi = ValueIteration(mdp, sample_rate=500)

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
        def _update(bel, s, sp, T, O, a, z):
            print 'BeliefUpdate::bel[{}]={}\tO[s][{}][{}]={}\tT[s][a][{}]={}'.format(s,bel[s],a,z,O[s][a][z],sp,T[s][a][sp])
            return bel[s] * O[s][a][z] * T[s][a][sp]

        def _compute_normalization_factor(bel, T, O, a, z):
            normalization_factor = sum([_update(bel, s, sp, T, O, a, z) for s in bel for sp in bel])
            return normalization_factor

        def _update_belief_for_state(b, s, T, O, a, z):
            num = sum([_update(b, s, sp, T, O, a, z) for sp in b])
            norm = _compute_normalization_factor(b, T, O, a, z)
            print 'BeliefUpdate::s={},Numerator={},Normalization={}'.format(s, num, norm)
            return num / norm if norm > 0 else 0.

        new_belief = defaultdict()
        for state in belief:
            new_belief[state] = _update_belief_for_state(belief, state, self.transition_probs, self.observation_probs, action, observation)

        return new_belief

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
        self.vi._compute_matrix_from_trans_func()
        return self.vi.trans_dict

    def construct_observation_matrix(self, observation_func):
        '''
        Create an MLE of the observation probabilities by sampling from the observation_func
        multiple times.
        Args:
            observation_func: O(s) -> z

        Returns:
            observation_probabilities (defaultdict): O(s, a, z) --> float
        '''
        obs_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
        for state in self.vi.get_states():
            for action in self.vi.mdp.actions:
                for sample in range(self.vi.sample_rate):
                    observation = observation_func(state, action)
                    obs_dict[state][action][observation] += 1. / self.vi.sample_rate
        return obs_dict
