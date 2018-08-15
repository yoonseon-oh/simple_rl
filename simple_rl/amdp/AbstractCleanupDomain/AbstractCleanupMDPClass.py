# Python imports
from collections import defaultdict
import copy

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupL1StateClass import *
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupStateMapper import AbstractCleanupL1StateMapper
from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP

import pdb

class CleanupL1GroundedAction(NonPrimitiveAbstractTask):
    pass

class CleanupRootGroundedAction(RootTaskNode):
    pass

class CleanupL1MDP(MDP):
    LIFTED_ACTIONS = ['toDoor', 'toRoom', 'toObject', 'objectToDoor', 'objectToRoom']

    def __init__(self, l0_domain):
        '''
        Args:
            l0_domain (CleanUpMDP)
        '''
        state_mapper = AbstractCleanupL1StateMapper(l0_domain)
        l1_init_state = state_mapper.map_state(l0_domain.init_state)
        grounded_actions = CleanupL1MDP.ground_actions(l1_init_state)
        MDP.__init__(self, grounded_actions, self._transition_function, self._reward_function, l1_init_state)

    def _reward_function(self, state, action):
        pass

    def _transition_function(self, state, action):
        '''
        Args:
            state (CleanupL1State)
            action (str): grounded action

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        lifted_action = self._grounded_to_lifted_action(action)

        if lifted_action == 'toDoor':
            target_door_name = self._grounded_to_action_parameter(action)
            next_state = self._move_agent_to_door(state, target_door_name)

        if lifted_action == 'toRoom':
            room_action_parameter = self._grounded_to_action_parameter(action)
            destination_room = self._action_parameter_to_room(room_action_parameter)
            next_state = self._move_agent_to_room(state, destination_room)

        if lifted_action == 'toObject':
            block_color = self._grounded_to_action_parameter(action)
            next_state = self._move_agent_to_block(state, block_color)

        if lifted_action == 'objectToDoor':
            target_door_name = self._grounded_to_action_parameter(action)
            next_state = self._move_agent_to_door(state, target_door_name)
            next_state = self._move_block_to_door(next_state, target_door_name)

        if lifted_action == 'objectToRoom':
            room_action_parameter = self._grounded_to_action_parameter(action)
            destination_room = self._action_parameter_to_room(room_action_parameter)
            next_state = self._move_agent_to_room(state, destination_room)
            next_state = self._move_block_to_room(next_state, destination_room)

        return next_state


    def _terminal_function(self, state):
        pass

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

    @staticmethod
    def _move_agent_to_door(state, door_name):
        next_state = copy.deepcopy(state)
        connecting_rooms = CleanupL1MDP._action_parameter_to_rooms(door_name)
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

        '''
        next_state = copy.deepcopy(state)
        connecting_rooms = CleanupL1MDP._action_parameter_to_rooms(door_name)
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
        block = CleanupL1MDP._get_l1_block_for_color(next_state, next_state.robot.adjacent_block)
        block.current_room = destination_room_color
        block.current_door = ''
        return next_state

    @staticmethod
    def _move_agent_to_block(state, block_color):
        '''
        Args:
            state (CleanupL1State)
            block (str)

        Returns:
            next_state (CleanupL1State)
        '''
        next_state = copy.deepcopy(state)
        target_block = CleanupL1MDP._get_l1_block_for_color(state, block_color)
        if target_block:
            if target_block.current_room == state.robot.current_room:
                next_state.robot.adjacent_block = target_block.block_color
                next_state.robot.current_door = ''
        return next_state

    @staticmethod
    def _grounded_to_lifted_action(grounded_action):
        '''
        Convenience method to extract the lifted action part of the string
        Args:
            grounded_action (str)

        Returns:
            lifted_action (str)
        '''
        return grounded_action.split('(')[0]

    @staticmethod
    def _grounded_to_action_parameter(grounded_action):
        '''
        Args:
            grounded_action (str)

        Returns:
            action_param (str)
        '''
        return grounded_action.split('(')[1].split(')')[0]

    @staticmethod
    def _action_parameter_to_rooms(action_param):
        '''
        Args:
            action_param (str)

        Returns:
            rooms (list): of colors of the two rooms
        '''
        return action_param.split('::')[1].split('_')

    @staticmethod
    def _action_parameter_to_room(action_param):
        '''
        Args:
            action_param (str)

        Returns:
            room (str): color of the destination room
        '''
        return action_param.split(':')[-1]

    @staticmethod
    def _determine_destination_room(state, connecting_rooms):
        '''
        Args:
            state (CleanupL1State)
            connecting_rooms (list)

        Returns:
            destination_room (str)
        '''
        if state.robot.current_room == connecting_rooms[0]:
            return connecting_rooms[1]
        if state.robot.current_room == connecting_rooms[1]:
            return connecting_rooms[0]
        return None

    @staticmethod
    def _get_l1_block_for_color(state, block_color):
        '''
        Args:
            state (CleanupL1State)
            block_color (str)

        Returns:
            block (CleanupL1Block)
        '''
        for block in state.blocks:
            if block.block_color == block_color:
                return block
        return None

    @staticmethod
    def _get_l1_room_for_color(state, room_color):
        '''
        Args:
            state (CleanupL1State)
            room_color (str)

        Returns:
            room (CleanupL1Room)
        '''
        for room in state.rooms: # type: CleanupL1Room
            if room.room_color == room_color:
                return room
        return None

def get_l1_policy():
    pass

if __name__ == '__main__':
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
    from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask
    from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP
    from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupStateMapper import AbstractCleanupL1StateMapper

    task = CleanUpTask("green", "red")
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
