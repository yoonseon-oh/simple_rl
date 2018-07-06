from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pdb

class Symbol(object):
    def __init__(self, name='', grounding_classifier=None, states_set=set()):
        '''
        Args:
            name (str)
            grounding_classifier (DecisionTreeClassifier)
            states_set (set): all the states in the underlying MDP
        '''
        self.name = name
        self.grounding_classifier = grounding_classifier
        self.universal_set = states_set
        self.grounding_set = self._construct_grounding_set()

    def _construct_grounding_set(self):
        grounding_set = set()
        for state in self.universal_set:
            state_features = np.array(state.get_state_vector()).reshape(1, -1)
            if self.grounding_classifier.predict(state_features) == 1:
                grounding_set.add(state)
            elif 'effect' in self.name:
                print 'For {}, did not add {} to grounding set'.format(self.name, state)
        return grounding_set

    def symbolic_and(self, symbol):
        '''
        Corresponds to the intersection of the corresponding grounding sets
        Args:
            symbol (Symbol): operand
        Returns:
            another_symbol (Symbol)
        '''
        return self.grounding_set.intersection(symbol.grounding_set)

    def symbolic_or(self, symbol):
        '''
        Corresponds to the union of the corresponding grounding sets
        Args:
            symbol (Symbol): operand
        Returns:
            another_symbol (Symbol)
        '''
        return self.grounding_set.union(symbol.grounding_set)

    def symbolic_not(self):
        '''
        Returns:
            symbol (Symbol): Corresponds to the complement of the current grounding set
        '''
        return self.universal_set.difference(self.grounding_set)

    def symbolic_null(self):
        '''
        Corresponds to whether or not a grounding set is empty
        Returns:
            is_null (bool)
        '''
        return len(self.grounding_set) == 0

    def symbolic_subset(self, symbol):
        '''
        Corresponds to test whether current grounding set is a subset of another grounding set
        Args:
            symbol (Symbol)
        Returns:
            is_subset (bool)
        '''
        return self.grounding_set.issubset(symbol.grounding_set)


class ProbabilisticSymbol(object):
    def __init__(self, name='', grounding_classifier=None, states_set=set([])):
        self.name = name
        self.grounding_classifier = grounding_classifier
        self.states = states_set
        self.possible_states = self._construct_grounding_set()

    def _construct_grounding_set(self):
        grounding_set = set()
        for state in self.states:
            state_features = np.array(state.get_state_vector()).reshape(1, -1)
            # pdb.set_trace()
            if 'effect' in self.name:
                effect_probability = np.exp(self.grounding_classifier.score_samples(state_features))
                # print 'For {}, {}, effect_prob is {}'.format(self.name, state, effect_probability)
                if effect_probability > 0:
                    grounding_set.add(state)
            else:
                precond_probability = self.grounding_classifier.predict_proba(state_features)
                # print 'For {}, {}, precond_prob is {}'.format(self.name, state, precond_probability)
                if precond_probability[0][1] > 0:
                    grounding_set.add(state)

        return grounding_set

    def feasibility(self, other):
        probability_feasible = 0.0
        for state in self.states:
            state_features = np.array(state.get_state_vector()).reshape(1, -1)
            effects_probability = self.grounding_classifier.score_samples(state_features)
            precond_probability = other.grounding_classifier.predict(state_features)
            probability_feasible += (effects_probability * precond_probability)
        return probability_feasible
