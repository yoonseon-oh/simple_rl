# Python imports
from collections import defaultdict
import re

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupL1StateClass import *
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupStateMapper import AbstractCleanupL1StateMapper

class CleanupL1GroundedAction(NonPrimitiveAbstractTask):
    pass

class CleanupRootGroundedAction(RootTaskNode):
    pass

class CleanupL1MDP(MDP):
    LIFTED_ACTIONS = ['toDoor', 'toRoom', 'toObject', 'objectToDoor', 'objectToRoom']

    @classmethod
    def ground_actions(cls, l1_state):
        '''
        Given a list of lifted/parameterized actions and the L0 cleanup domain,
        generate a list of grounded actions based on the attributes of the objects
        instantiated in the L0 domain.
        Args:
            l1_state (CleanupL1State): underlying ground level MDP

        Returns:
            actions (list): grounded actions
        '''
        grounded_actions = []

        for door in l1_state.doors: # type: CleanupL1Door
            grounded_actions.append(cls.LIFTED_ACTIONS[0] + '(' + str(door) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[3] + '(' + str(door) + ')')

        for room in l1_state.rooms: # type: CleanupL1Room
            grounded_actions.append(cls.LIFTED_ACTIONS[1] + '(' + str(room) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[4] + '(' + str(room) + ')')

        for block in l1_state.blocks: # type: CleanupL1Block
            grounded_actions.append(cls.LIFTED_ACTIONS[2] + '(' + str(block.block_color) + ')')

        return grounded_actions

    def __init__(self, l0_domain):
        state_mapper = AbstractCleanupL1StateMapper(l0_domain)
        l1_init_state = state_mapper.map_state(l0_domain.init_state)
        grounded_actions = CleanupL1MDP.ground_actions(l1_init_state)
        MDP.__init__(self, grounded_actions, self._transition_function, self._reward_function, l1_init_state)

    def _reward_function(self, state, action):
        pass

    def _transition_function(self, state, action):
        pass

    def _terminal_function(self, state):
        pass

def get_l1_policy():
    pass

if __name__ == '__main__':
    get_l1_policy()
