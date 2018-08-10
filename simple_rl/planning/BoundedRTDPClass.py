from simple_rl.planning import Planner
from simple_rl.planning import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.utils.additional_datastructures import Stack

from collections import defaultdict
import copy

class BoundedRTDP(Planner):
    def __init__(self, mdp, name='BRTDP', tau=10.):
        Planner.__init__(self, mdp, name)
        self.lower_values = MonotoneLowerBound(mdp).lower_values
        self.upper_values = MonotoneUpperBound(mdp).upper_values
        self.vi = ValueIteration(mdp, sample_rate=500)
        self.states = self.vi.get_states()
        self.vi._compute_matrix_from_trans_func()
        self.trans_dict = self.vi.trans_dict
        self.max_diff = (self.upper_values[self.mdp.init_state] - self.lower_values[self.mdp.init_state]) / tau

    def plan(self, state=None, horizon=100):
        state = self.mdp.get_init_state() if state is None else state
        policy = defaultdict()
        steps = 0
        while (not state.is_terminal()) and steps < horizon:
            next_action = self.policy(state)
            policy[state] = next_action
            state = self.transition_func(state, next_action)
            steps += 1
        return policy

    def policy(self, state):
        return self._greedy_action(state, self.lower_values)

    def _run_sample_trial(self):
        init_state = self.mdp.init_state

        state = copy.deepcopy(init_state)
        trajectory = Stack()
        while not state.is_terminal():
            trajectory.push(state)
            self.upper_values[state] = self._best_qvalue(state, self.upper_values)
            action = self._greedy_action(state, self.lower_values)
            self.lower_values[state] = self._qvalue(state, action, self.lower_values)
            expected_gap_distribution = self._expected_gap_distribution(state, action)
            expected_gap = sum(expected_gap_distribution.values())
            print '{}\tAction: {}\tExpectedGap: {}\tMaxDiff: {}'.format(state, action, expected_gap, self.max_diff)
            if expected_gap < self.max_diff:
                print 'Ending rollouts with gap {} and max_diff {}'.format(expected_gap, self.max_diff)
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

    def _expected_gap_distribution(self, state, action):
        expected_gaps = defaultdict()
        for next_state in self.states:
            gap = self.upper_values[next_state] - self.lower_values[next_state]
            expected_gaps[next_state] = self.trans_dict[state][action][next_state] * gap
        return expected_gaps

    def _pick_next_state(self, distribution, expected_gap):
        def _scale_distribution(_distribution, scaling_factor):
            for state in _distribution:
                _distribution[state] *= scaling_factor
            return _distribution
        scaled_distribution = _scale_distribution(distribution, expected_gap)
        return max(scaled_distribution, key=scaled_distribution.get)

    def run(self):
        self._run_sample_trial()

class MonotoneLowerBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        relaxed_mdp = self._construct_deterministic_relaxation_mdp(mdp)

        Planner.__init__(self, relaxed_mdp, name)
        self.vi = ValueIteration(relaxed_mdp)
        self.states = self.vi.get_states()
        self.actions = relaxed_mdp.get_actions()
        self.vi._compute_matrix_from_trans_func()
        self.trans_dict = self.vi.trans_dict
        self.vi.run_vi()
        self.lower_values = self._construct_lower_values()

    def _construct_deterministic_relaxation_mdp(self, mdp):
        relaxed_mdp = copy.deepcopy(mdp)
        relaxed_mdp.set_slip_prob(0.0)
        return relaxed_mdp

    def _construct_lower_values(self):
        values = defaultdict()
        for state in self.states:
            values[state] = self.vi.get_value(state)
        return values

class MonotoneUpperBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        Planner.__init__(self, mdp, name)
        self.vi = ValueIteration(mdp)
        self.states = self.vi.get_states()
        self.upper_values = self._construct_upper_values()

    def _construct_upper_values(self):
        values = defaultdict()
        for state in self.states:
            values[state] = 1. / (1. - self.gamma)
        return values

if __name__ == '__main__':
    test_mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.2)
    bounded_rtdp = BoundedRTDP(test_mdp)
    bounded_rtdp.run()
    policy = bounded_rtdp.plan()
