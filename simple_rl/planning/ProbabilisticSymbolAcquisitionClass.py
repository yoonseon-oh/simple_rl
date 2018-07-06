from simple_rl.planning.SymbolAcquisitionClass import *

from sklearn import svm
import networkx as nx
import pdb

class ProbabilisticSymbolAcquisition(SymbolAcquisition):
    def __init__(self, name='taxi_probabilistic_symbols'):
        # pdb.set_trace()
        SymbolAcquisition.__init__(self, name=name)

    def _collect_data_for_effects_classifiers(self, mdp, state):
        options_for_state = self._option_space(state)
        for option in options_for_state:
            end_state, reward = option.rollout(state, mdp.reward_func, mdp.transition_func)
            option.effect_features.append(np.array(end_state.get_state_vector()))
            option.reward_features.append(np.array(state.get_state_vector()))
            option.reward_labels.append(reward)

    def _learn_symbols(self):
        for option in self.options:
            self._learn_effects_symbol(option)
            self._learn_precondition_symbol(option)
            self._learn_reward_model(option)

    def _learn_reward_model(self, option):
        option.reward_classifier.fit(option.reward_features, option.reward_labels)

    def _learn_precondition_symbol(self, option):
        option.precond_classifier.fit(option.precond_features, option.precond_labels)
        option.construct_prob_precond_symbol(states=self.states, trained_classifier=option.precond_classifier)

    def _learn_effects_symbol(self, option):
        option.effects_classifier.fit(option.effect_features)
        option.construct_prob_effects_symbol(states=self.states, trained_classifier=option.effects_classifier)

    def construct_taxi_domain(self):
        print 'constructing taxi domain with some slip probability'
        taxi = {"x": 1, "y": 1, "has_passenger": 0}
        passengers = [{"x": 3, "y": 2, "dest_x": 2, "dest_y": 3, "in_taxi": 0}]
        walls = []
        pickup_mdp = TaxiOOMDP(width=4, height=4, agent=taxi, walls=walls, passengers=passengers, slip_prob=0.2)

        return pickup_mdp

def construct_probabilistic_plan_graph(symbolic_planner):
    plan_graph = nx.DiGraph()

    for option in symbolic_planner.options:
        plan_graph.add_node(option)

    for option1 in symbolic_planner.options:
        for option2 in symbolic_planner.options:
            if option1 != option2 and option1.effects_symbol.feasibility(option2.precond_symbol) > 0:
                o1_end_states = option1.effects_symbol.possible_states
                print 'Option1 possible end states: ', o1_end_states
                rewards = option2.reward_classifier.predict(o1_end_states)
                plan_graph.add_edge(option1, option2, np.sum(rewards))
    return plan_graph


if __name__ == '__main__':
    symbolic_planner = ProbabilisticSymbolAcquisition()
    symbolic_planner.generate_symbol_set()
    plan_graph = construct_probabilistic_plan_graph(symbolic_planner)
