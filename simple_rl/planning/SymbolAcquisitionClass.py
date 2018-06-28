from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.tasks.taxi.TaxiOOMDPClass import TaxiOOMDP, TaxiState
from examples.oomdp_example import options_for_taxi_domain
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.planning.SymbolicAlgebraClass import Symbol

from collections import defaultdict
import numpy as np
from sklearn import tree
import pdb
import networkx as nx

class SymbolAcquisition(object):
    def __init__(self, name='taxi_set_symbols'):
        self.mdp = construct_taxi_domain()
        self.options = options_for_taxi_domain()
        self.states = get_states(self.mdp)
        self.name = name

    def __str__(self):
        return self.name

    def generate_symbol_set(self):
        self._interact()
        self._learn_symbols()

    def _learn_symbols(self):
        for option in self.options:
            self._learn_precondition_symbol(option)
            self._learn_effects_symbol(option)

    def _learn_precondition_symbol(self, option):
        option.precond_classifier.fit(option.precond_features, option.precond_labels)
        option.construct_precondition_symbol(states=self.states, trained_classifier=option.precond_classifier)

    def _learn_effects_symbol(self, option):
        option.effects_classifier.fit(option.effect_features, option.effect_labels)
        option.construct_effects_symbol(states=self.states, trained_classifier=option.effects_classifier)

    def _interact(self, num_times=1):
        '''
        Args:
            num_times (int)

        Discussion:
            For each option, we are going to train 2 classifiers:
            (a) precondition classifier. Training samples of form: (s, I_o(s))
            (b) effects classifier. Training samples of form: (s')
        '''
        for _ in range(num_times):
            pickup_mdp = construct_taxi_domain()
            self._collect_data_(pickup_mdp)
        for option in self.options:
            option.effect_features = np.array(option.effect_features)
            option.precond_features = np.array(option.precond_features)
            option.effect_labels = np.array(option.effect_labels)
            option.precond_labels = np.array(option.precond_labels)

    def _collect_data_(self, mdp):
        for state in get_states(mdp):
            self._collect_data_for_effects_classifiers(mdp, state)

    def _collect_data_for_effects_classifiers(self, mdp, state):
        options_for_state = self._option_space(state)
        for option in options_for_state:
            end_state, intermediate_states = option.act_until_terminal(state, mdp.transition_func, verbose=False)
            option.effect_features.append(np.array(end_state.get_state_vector()))
            option.effect_labels.append(1)
            for intermediate_state in intermediate_states:
                intermediate_state_features = intermediate_state.get_state_vector()
                option.effect_features.append(np.array(intermediate_state_features))
                option.effect_labels.append(0)

    def _option_space(self, state):
        option_space = set([])
        state_features = state.get_state_vector()
        for option in self.options:
            option.precond_features.append(np.array(state_features))
            if option.is_init_true(state):
                option_space.add(option)
                option.precond_labels.append(1)
            else:
                option.precond_labels.append(0)
        return option_space

def construct_taxi_domain():
    taxi = {"x": 1, "y": 1, "has_passenger": 0}
    passengers = [{"x": 3, "y": 2, "dest_x": 2, "dest_y": 3, "in_taxi": 0}]
    walls = []
    pickup_mdp = TaxiOOMDP(width=4, height=4, agent=taxi, walls=walls, passengers=passengers)

    return pickup_mdp

def get_states(mdp):
    vi = ValueIteration(mdp)
    return vi.get_states()

def draw_classifiers(options):
    from sklearn.tree import export_graphviz
    feature_names = ['py', 'px', 'in_taxi', 'desty', 'destx', 'ty', 'tx', 'has_passenger']
    for option in options:
        classifier = option.precond_classifier
        export_graphviz(classifier, feature_names=feature_names,
                        out_file='{}_PrecondClassifier.dot'.format(option.name),
                        class_names=['false', 'true'])

        classifier = option.effects_classifier
        export_graphviz(classifier, feature_names=feature_names,
                        out_file='{}_EffectsClassifier.dot'.format(option.name),
                        class_names=['false', 'true'])


def construct_plan_graph(symbolic_planner):
    plan_graph = nx.DiGraph()

    for option in symbolic_planner.options:
        plan_graph.add_node(option)

    for option1 in symbolic_planner.options:
        for option2 in symbolic_planner.options:
            if option1 != option2 and option1.effects_symbol.symbolic_subset(option2.precond_symbol):
                plan_graph.add_edge(option1, option2)
    return plan_graph


if __name__ == '__main__':
    symbolic_planner = SymbolAcquisition()
    symbolic_planner.generate_symbol_set()
    draw_classifiers(symbolic_planner.options)
    plan_graph = construct_plan_graph(symbolic_planner)
