# Python imports.
from collections import defaultdict
import random
from sklearn import tree

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.planning.SymbolicAlgebraClass import Symbol

class Option(object):

	def __init__(self, init_predicate, term_predicate, policy, name="o", term_prob=0.01):
		'''
		Args:
			init_func (S --> {0,1})
			init_func (S --> {0,1})
			policy (S --> A)
		'''
		self.init_predicate = init_predicate
		self.term_predicate = term_predicate
		self.term_flag = False
		self.name = name
		self.term_prob = term_prob

		self.effect_features = []
		self.effect_labels = []
		self.precond_features = []
		self.precond_labels = []

		self.effects_classifier = tree.DecisionTreeClassifier()
		self.precond_classifier = tree.DecisionTreeClassifier()

		self.effects_symbol = None
		self.precond_symbol = None

		if type(policy) is defaultdict or type(policy) is dict:
			self.policy_dict = dict(policy)
			self.policy = self.policy_from_dict
		else:
			self.policy = policy

	def __eq__(self, other):
		return isinstance(other, type(self)) \
			   and ((self.init_predicate, self.term_predicate, self.term_prob, self.policy) ==
					(other.init_predicate, other.term_predicate, other.term_prob, other.policy))

	def __hash__(self):
		return hash((self.init_predicate, self.term_predicate, self.term_prob, self.policy))

	def is_init_true(self, ground_state):
		return self.init_predicate.is_true(ground_state)

	def is_term_true(self, ground_state):
		return self.term_predicate.is_true(ground_state) or self.term_flag or self.term_prob > random.random()

	def act(self, ground_state):
		return self.policy(ground_state)

	def set_policy(self, policy):
		self.policy = policy

	def set_name(self, new_name):
		self.name = new_name

	def act_until_terminal(self, cur_state, transition_func, verbose=False):
		'''
		Summary:
			Executes the option until termination.
		'''
		if verbose: print 'Starting option execution from {}\nWith Action {}'.format(cur_state, self.policy(cur_state))
		intermediate_states_encountered = [cur_state]
		if self.is_init_true(cur_state):
			cur_state = transition_func(cur_state, self.act(cur_state))
			while not self.is_term_true(cur_state):
				intermediate_states_encountered.append(cur_state)
				if verbose: print 'action = {}'.format(self.policy(cur_state))
				cur_state = transition_func(cur_state, self.act(cur_state))

		return cur_state, intermediate_states_encountered

	def rollout(self, cur_state, reward_func, transition_func, step_cost=0):
		'''
		Summary:
			Executes the option until termination.

		Returns:
			(tuple):
				1. (State): state we landed in.
				2. (float): Reward from the trajectory.
		'''
		total_reward = 0
		if self.is_init_true(cur_state):
			# First step.
			total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost
			cur_state = transition_func(cur_state, self.act(cur_state))

			# Act until terminal.
			while not self.is_term_true(cur_state):
				cur_state = transition_func(cur_state, self.act(cur_state))
				total_reward += reward_func(cur_state, self.act(cur_state)) - step_cost

		return cur_state, total_reward

	def policy_from_dict(self, state):
		if state not in self.policy_dict.keys():
			self.term_flag = True
			return random.choice(list(set(self.policy_dict.values())))
		else:
			self.term_flag = False
			return self.policy_dict[state]

	def term_func_from_list(self, state):
		return state in self.term_list

	def __str__(self):
		return "option." + str(self.name)

	def construct_effects_symbol(self, states, trained_classifier):
		name = self.name + '_' + 'effects_symbol'
		self.effects_symbol = Symbol(name=name, grounding_classifier=trained_classifier, states_set=states)

	def construct_precondition_symbol(self, states, trained_classifier):
		name = self.name + '_' + 'precondition_symbol'
		self.precond_symbol = Symbol(name=name, grounding_classifier=trained_classifier, states_set=states)

	def __repr__(self):
		return self.__str__()