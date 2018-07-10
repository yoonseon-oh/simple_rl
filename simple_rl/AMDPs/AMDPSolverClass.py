from simple_rl.mdp.StateClass import State

from collections import defaultdict
import re
import pdb

class AMDPAgent(object):
    def __init__(self, root_grounded_task, policy_generators, base_mdp):
        self.root_grounded_task = root_grounded_task
        self.policy_generators = policy_generators
        self.base_mdp = base_mdp
        self.state_stack = []
        self.policy_stack = []
        for i in range(len(policy_generators)):
            self.state_stack.append(State())
            self.policy_stack.append(defaultdict())
        self.max_level = len(self.policy_generators) - 1

    def plan(self):
        base_state = self.base_mdp.init_state
        self.state_stack[0] = base_state

        for i in range(1, len(self.policy_generators)):
            pg = self.policy_generators[i]
            base_state = pg.generateAbstractState(base_state)
            self.state_stack[i] = base_state

        self.decompose(self.root_grounded_task, self.max_level)

    def decompose(self, grounded_task, level):
        print 'Decomposing action {} at level {}'.format(grounded_task, level)
        state = self.state_stack[level]

        # TODO: Hack: Decide if this should be a string or a TaskNode
        if not type(grounded_task) is str: grounded_task = grounded_task.name

        policy = self.policy_generators[level].generatePolicy(state, grounded_task)
        if level > 0:
            while not self._is_terminal(state, grounded_task, level):
                action = policy[state]
                self.policy_stack[level][state] = action
                self.decompose(action, level-1)
                state = self.state_stack[level]
        else:
            while not self._is_terminal(state, grounded_task, level) and not self._env_is_terminal(state):
                action = policy[state]
                self.policy_stack[level][state] = action
                reward, state = self.base_mdp.execute_agent_action(action)
                self.state_stack[level] = state
        if level < self.max_level:
            projected_state = self.policy_generators[level+1].generateAbstractState(self.state_stack[level])
            self.state_stack[level+1] = projected_state

    # TODO: Hack: Somehow use TaskNodes to determine if a grounded task is terminal
    def _is_terminal(self, state, grounded_task, level):
        destination_room = int(re.findall(r'\d+', grounded_task)[0])
        if level == 0:
            destination_location = self.base_mdp._get_single_location_for_room(destination_room)
            current_location = (int(state.x), int(state.y))
            return  current_location == destination_location
        elif level == 1:
            return state.agent_in_room_number == destination_room
        raise NotImplementedError('level {} not supported yet'.format(level))

    def _env_is_terminal(self, state):
        return (state.x, state.y) in self.base_mdp.goal_locs
