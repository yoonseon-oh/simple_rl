from simple_rl.planning import Planner
from simple_rl.planning import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.utils.additional_datastructures import Stack

from Queue import  PriorityQueue
import numpy as np
from collections import defaultdict
import copy
import pdb


class BoundedRTDP(Planner):
    def __init__(self, mdp, name='BRTDP', tau=1.5):
        Planner.__init__(self, mdp, name)
        self.lower_values = MonotoneLowerBound(mdp).lower_values
        self.upper_values = MonotoneUpperBound(mdp).upper_values
        self.vi = ValueIteration(mdp)
        self.states = self.vi.get_states()
        self.vi._compute_matrix_from_trans_func()
        self.trans_dict = self.vi.trans_dict
        self.max_diff = (self.upper_values[self.mdp.init_state] - self.lower_values[self.mdp.init_state]) / tau

    def plan(self, state):
        pass

    def policy(self, state):
        pass

    def __str__(self):
        return self.name

    def _run_sample_trial(self):
        init_state = self.mdp.init_state

        state = copy.deepcopy(init_state)
        trajectory = Stack()
        while not (state.x, state.y) in self.mdp.goal_locs:
            trajectory.push(state)
            self.upper_values[state] = self._best_qvalue(state, self.upper_values)
            action = self._greedy_action(state, self.lower_values)
            self.lower_values[state] = self._qvalue(state, action, self.lower_values)
            expected_gap_distribution = self._expected_gap_distribution(state, action)
            expected_gap = sum(expected_gap_distribution.values())
            print 'State: ({}, {})\tAction: {}\tExpectedGap: {}\tMaxDiff: {}'.format(state.x, state.y, action, expected_gap, self.max_diff)
            if expected_gap < self.max_diff:
                break
            state = self._pick_next_state(expected_gap_distribution, expected_gap)
        while not trajectory.isEmpty():
            state = trajectory.pop()
            self.upper_values[state] = self._best_qvalue(state, self.upper_values)
            self.lower_values[state] = self._best_qvalue(state, self.lower_values)

    def _greedy_action(self, state, values):
        return max([(self._qvalue(state, action, values), action) for action in self.actions])[1]

    def _qvalue(self, state, action, values):
        return self.mdp.reward_func(state, action) + sum([self.trans_dict[state][action][next_state] * values[next_state] \
                                                          for next_state in self.states])

    def _best_qvalue(self, state, values):
        return max([self._qvalue(state, action, values) for action in self.actions])

    def _update(self, state, values):
        action = self._greedy_action(state)
        values[state] = self._qvalue(state, action, values)

    def _expected_gap_distribution(self, state, action):
        expected_gaps = defaultdict()
        for next_state in self.states:
            gap = self.upper_values[next_state] - self.lower_values[next_state]
            expected_gaps[next_state] = self.trans_dict[state][action][next_state] * gap
        return expected_gaps

    def _pick_next_state(self, distribution, expected_gap):
        def _scale_distribution(distribution, scaling_factor):
            for state in distribution:
                distribution[state] *= scaling_factor
        _scale_distribution(distribution, expected_gap)
        return max(distribution, key=distribution.get)

    def _residual(self, state):
        pass

    def run(self):
        self._run_sample_trial()


class MonotoneLowerBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        relaxed_mdp = self._constructDeterministicRelaxationMDP(mdp)

        Planner.__init__(self, relaxed_mdp, name)
        self.vi = ValueIteration(relaxed_mdp)
        self.states = self.vi.get_states()
        self.actions = relaxed_mdp.get_actions()
        self.vi._compute_matrix_from_trans_func()
        self.trans_dict = self.vi.trans_dict
        self.vi.run_vi()
        self.lower_values = self._construct_lower_values()

    def _constructDeterministicRelaxationMDP(self, mdp):
        relaxed_mdp = copy.deepcopy(mdp)
        relaxed_mdp.set_slip_prob(0.0)
        return relaxed_mdp

    def _construct_lower_values(self):
        values = defaultdict()
        for state in self.states:
            values[state] = self.vi.get_value(state)
        return values

    def __str__(self):
        return self.name


class MonotoneUpperBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        Planner.__init__(self, mdp, name)
        self.vi = ValueIteration(mdp)
        self.states = self.vi.get_states()
        self.actions = self.mdp.get_actions()
        self.upper_values = self._construct_upper_values()

    def _construct_upper_values(self):
        values = defaultdict()
        for state in self.states:
            if (state.x, state.y) in self.mdp.goal_locs:
                values[state] = 1.0
            else:
                values[state] = 2.5
        return values

    def __str__(self):
        return self.name


if __name__ == '__main__':
    mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.25)
    bounded_rtdp = BoundedRTDP(mdp)
    bounded_rtdp.run()
