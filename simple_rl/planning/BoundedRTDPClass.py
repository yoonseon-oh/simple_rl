from simple_rl.planning import Planner
from simple_rl.planning import ValueIteration
from simple_rl.tasks import GridWorldMDP
from simple_rl.utils.additional_datastructures import Stack

from Queue import  PriorityQueue
import numpy as np

class BoundedRTDP(Planner):
    def __init__(self, mdp, name='BRTDP', tol=0.01):
        Planner.__init__(self, mdp, name)
        self.lower_bound = MonotoneLowerBound(mdp)
        self.upper_bound = MonotoneUpperBound(mdp)
        self.tol = tol

    def plan(self, state):
        pass

    def policy(self, state):
        pass

    def __str__(self):
        return self.name

    def _run_sample_trial(self):
        state = self.mdp.init_state
        trajectory = Stack()

    def run(self):
        start = self.mdp.init_state
        while self.upper_bound.get_value(start) - self.lower_bound.get_value(start) > self.tol:
            self._run_sample_trial()


class MonotoneLowerBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        Planner.__init__(self, mdp, name)
        self.relaxed_mdp = self._constructDeterministicRelaxationMDP()
        self.vi = ValueIteration(mdp)
        self.vi.run_vi()

    def plan(self, state):
        return self.vi.plan(state)

    def policy(self, state):
        return self.vi.policy(state)

    def get_value(self, state):
        return self.vi.get_value(state)

    def _constructDeterministicRelaxationMDP(self):
        '''
        Returns:
             mdp (MDP): a deterministic relaxation of self.mdp
        '''
        return self.mdp

    def __str__(self):
        return self.name


class MonotoneUpperBound(Planner):
    def __init__(self, mdp, name='MonotoneUpperBound'):
        Planner.__init__(self, mdp, name)

    def plan(self, state):
        pass

    def policy(self, state):
        pass

    def get_value(self, state):
        pass

    def sweep(self):
        pass

    def __str__(self):
        return self.name


if __name__ == '__main__':
    mdp = GridWorldMDP(width=6, height=6, goal_locs=[(6, 6)], slip_prob=0.0)
    bounded_rtdp = BoundedRTDP(mdp)
    bounded_rtdp.run()
