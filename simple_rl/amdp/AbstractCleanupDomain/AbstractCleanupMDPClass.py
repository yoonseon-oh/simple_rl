# Python imports
from collections import defaultdict
import copy

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupL1StateClass import *

class CleanupL1GroundedAction(NonPrimitiveAbstractTask):
    def __init__(self, l1_action_string, subtasks, lower_domain):
        '''
        Args:
            l1_action_string (str)
            subtasks (list)
            lower_domain (CleanUpMDP)
        '''
        self.action = l1_action_string
        self.l0_domain = lower_domain

        tf, rf = self._terminal_function, self._reward_function
        NonPrimitiveAbstractTask.__init__(self, l1_action_string, subtasks, tf, rf)

    def _terminal_function(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            is_terminal (bool)
        '''
        def _robot_door_terminal_func(s, door_color):
            return s.robot.current_door == door_color
        def _robot_room_terminal_func(s, room_color):
            return s.robot.current_room == room_color
        def _robot_to_block_terminal_func(s, block_color):
            return s.robot.adjacent_block == block_color
        def _block_to_door_terminal_func(s, block_color, door_color):
            for block in s.blocks:
                if block.block_color == block_color and block.current_door == door_color:
                    return True
            return False
        def _block_to_room_terminal_func(s, block_color, room_color):
            for block in s.blocks:
                if block.block_color == block_color and block.current_room == room_color:
                    return True
            return False

        state_mapper = AbstractCleanupL1StateMapper(self.l0_domain)
        projected_state = state_mapper.map_state(state)

    def _reward_function(self, state):
        pass

    # -------------------------------
    # L1 Action Helper Functions
    # -------------------------------

    @staticmethod
    def grounded_to_lifted_action(grounded_action_str):
        return grounded_action_str.split('(')[0]

    @staticmethod
    def grounded_to_action_parameter(grounded_action_str):
        return grounded_action_str.split('(')[1].split(')')[0]

    @staticmethod
    def door_name_to_room_colors(door_name):
        return door_name.split('_')

class CleanupRootGroundedAction(RootTaskNode):
    pass

class CleanupL1MDP(MDP):
    LIFTED_ACTIONS = ['toDoor', 'toRoom', 'toObject', 'objectToDoor', 'objectToRoom']

    # -------------------------------
    # Level 1 MDP description
    # -------------------------------

    def __init__(self, l0_domain):
        '''
        Args:
            l0_domain (CleanUpMDP)
        '''
        self.l0_domain = l0_domain
        state_mapper = AbstractCleanupL1StateMapper(l0_domain)
        l1_init_state = state_mapper.map_state(l0_domain.init_state)
        grounded_actions = CleanupL1MDP.ground_actions(l1_init_state)

        print 'Grounded actions: ', grounded_actions

        MDP.__init__(self, grounded_actions, self._transition_function, self._reward_function, l1_init_state)

    def _is_goal_state(self, state):
        for block in state.blocks: # type: CleanupL1Block
            if block.block_color == self.l0_domain.task.block_color:
                return block.current_room == self.l0_domain.task.goal_room_color and \
                       state.robot.current_room == self.l0_domain.task.goal_room_color
        raise ValueError('Did not find an L1 Block object with color {}'.format(self.l0_domain.task.block_color))

    def _reward_function(self, state, action):
        '''
        Args:
            state (CleanupL1State)
            action (str)

        Returns:
            reward (float)
        '''
        next_state = self._transition_function(state, action)
        return 1. if self._is_goal_state(next_state) else 0.

    def _transition_function(self, state, action):
        '''
        Args:
            state (CleanupL1State)
            action (str): grounded action

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        lifted_action = CleanupL1GroundedAction.grounded_to_lifted_action(action)

        if lifted_action == 'toDoor':
            target_door_name = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_door(state, target_door_name)

        if lifted_action == 'toRoom':
            destination_room = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_room(state, destination_room)

        if lifted_action == 'toObject':
            block_color = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_block(state, block_color)

        if lifted_action == 'objectToDoor':
            target_door_name = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_door(state, target_door_name)
            next_state = self._move_block_to_door(next_state, target_door_name)

        if lifted_action == 'objectToRoom':
            destination_room = CleanupL1GroundedAction.grounded_to_action_parameter(action)
            next_state = self._move_agent_to_room(state, destination_room)
            next_state = self._move_block_to_room(next_state, destination_room)

        next_state.set_terminal(self._is_goal_state(next_state))

        return next_state

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

        for door in l1_state.doors:  # type: CleanupL1Door
            grounded_actions.append(cls.LIFTED_ACTIONS[0] + '(' + str(door) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[3] + '(' + str(door) + ')')

        for room in l1_state.rooms:  # type: CleanupL1Room
            grounded_actions.append(cls.LIFTED_ACTIONS[1] + '(' + str(room) + ')')
            grounded_actions.append(cls.LIFTED_ACTIONS[4] + '(' + str(room) + ')')

        for block in l1_state.blocks:  # type: CleanupL1Block
            grounded_actions.append(cls.LIFTED_ACTIONS[2] + '(' + str(block.block_color) + ')')

        return grounded_actions

    # -------------------------------
    # Navigation Helper functions
    # -------------------------------

    @staticmethod
    def _move_agent_to_door(state, door_name):
        next_state = copy.deepcopy(state)
        connecting_rooms = CleanupL1GroundedAction.door_name_to_room_colors(door_name)
        for door in state.doors:  # type: CleanupL1Door
            if connecting_rooms[0] in door.connected_rooms and connecting_rooms[1] in door.connected_rooms:
                next_state.robot.current_door = door.connected_rooms[0] + '_' + door.connected_rooms[1]
                next_state.robot.current_room = ''
        return next_state

    @staticmethod
    def _move_block_to_door(state, door_name):
        '''
        Args:
            state (CleanupL1State)
            door_name (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        connecting_rooms = CleanupL1GroundedAction.door_name_to_room_colors(door_name)
        for door in state.doors:  # type: CleanupL1Door
            if connecting_rooms[0] in door.connected_rooms and connecting_rooms[1] in door.connected_rooms:
                for block in next_state.blocks:
                    if block.block_color == state.robot.adjacent_block:
                        block.current_door = door
                        block.current_room = ''
        return next_state

    @staticmethod
    def _move_agent_to_room(state, destination_room_color):
        '''
        Args:
            state (CleanupL1State)
            destination_room_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        if destination_room_color in state.robot.current_door:
            next_state.robot.current_room = destination_room_color
            next_state.robot.current_door = ''
        return next_state

    @staticmethod
    def _move_block_to_room(state, destination_room_color):
        '''
        Args:
            state (CleanupL1State)
            destination_room_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        if state.robot.adjacent_block is None or state.robot.adjacent_block == '':
            return state

        next_state = copy.deepcopy(state)
        block = next_state.get_l1_block_for_color(next_state.robot.adjacent_block)
        block.current_room = destination_room_color
        block.current_door = ''
        return next_state

    @staticmethod
    def _move_agent_to_block(state, block_color):
        '''
        Args:
            state (CleanupL1State)
            block_color (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        target_block = state.get_l1_block_for_color(block_color)
        if target_block:
            if target_block.current_room == state.robot.current_room:
                next_state.robot.adjacent_block = target_block.block_color
                next_state.robot.current_door = ''
        return next_state

def get_l1_policy(domain):
    vi = ValueIteration(domain, sample_rate=1)
    vi.run_vi()

    policy = defaultdict()
    action_seq, state_seq = vi.plan(domain.init_state)

    print 'Plan for {}:'.format(domain)
    for i in range(len(action_seq)):
        print "\tpi[{}] -> {}\n".format(state_seq[i], action_seq[i])
        policy[state_seq[i]] = action_seq[i]

    return policy

if __name__ == '__main__':
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
    from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask
    from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP
    from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupStateMapper import AbstractCleanupL1StateMapper

    task = CleanUpTask("purple", "red")
    room1 = CleanUpRoom("room1", [(x, y) for x in range(5) for y in range(3)], "blue")
    block1 = CleanUpBlock("block1", 1, 1, color="green")
    block2 = CleanUpBlock("block2", 2, 4, color="purple")
    block3 = CleanUpBlock("block3", 8, 1, color="orange")
    room2 = CleanUpRoom("room2", [(x, y) for x in range(5, 10) for y in range(3)], color="red")
    room3 = CleanUpRoom("room3", [(x, y) for x in range(0, 10) for y in range(3, 6)], color="yellow")
    rooms = [room1, room2, room3]
    blocks = [block1, block2, block3]
    doors = [CleanUpDoor(4, 0), CleanUpDoor(3, 2)]
    mdp = CleanUpMDP(task, rooms=rooms, doors=doors, blocks=blocks)

    amdp = CleanupL1MDP(mdp)

    get_l1_policy(amdp)


