from sklearn.tree import DecisionTreeClassifier
import numpy as np

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
