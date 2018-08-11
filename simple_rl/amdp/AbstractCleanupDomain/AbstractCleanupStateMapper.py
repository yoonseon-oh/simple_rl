from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.tasks.cleanup.cleanup_state import CleanUpState
from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupL1StateClass import *

class AbstractCleanupL1StateMapper(AMDPStateMapper):
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)
        self.l0_domain = l0_domain

    def map_state(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            projected_state (CleanupL1State)
        '''
        l1_robot = self._derive_l1_robot(state)
        l1_doors = self._derive_l1_doors(state)
        l1_rooms = self._derive_l1_rooms(state)
        l1_blocks = self._derive_l1_blocks(state)
        return CleanupL1State(l1_robot, l1_doors, l1_rooms, l1_blocks)

    # -----------------------------
    # ----- Helper Methods --------
    # -----------------------------

    def _derive_l1_blocks(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_blocks (list): list of CleanupL1Block objects
        '''
        l1_blocks = []
        for block in state.blocks:  # type: CleanUpBlock
            room_color = self._position_to_room_color(state.rooms, (block.x, block.y))
            block_color = block.color
            l1_blocks.append(CleanupL1Block(room_color, block_color))
        return l1_blocks

    @staticmethod
    def _derive_l1_rooms(state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_rooms (list)
        '''
        return [CleanupL1Room(room.color) for room in state.rooms]

    def _derive_l1_doors(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            l1_doors (list): list of CleanupL1Door Objects
        '''
        l1_doors = []
        for door in state.doors: # type: CleanUpDoor
            connecting_rooms = []
            left_room = self._position_to_room_color(state.rooms, (door.x - 1, door.y))
            right_room = self._position_to_room_color(state.rooms, (door.x + 1, door.y))
            above_room = self._position_to_room_color(state.rooms, (door.x, door.y + 1))
            below_room = self._position_to_room_color(state.rooms, (door.x, door.y - 1))
            if left_room and right_room and left_room != right_room:
                connecting_rooms = [left_room, right_room]
            elif above_room and below_room and above_room != below_room:
                connecting_rooms = [above_room, below_room]
            if connecting_rooms:
                l1_doors.append(CleanupL1Door(connecting_rooms))
        return l1_doors

    def _derive_l1_robot(self, state):
        '''
        Args:
            state (CleanUpState)

        Returns:
            robot (CleanupL1Robot)
        '''
        robot_room = self._position_to_room_color(state.rooms, (state.x, state.y))
        adjacent_block_color = None
        for block in state.blocks:
            if AbstractCleanupL1StateMapper._is_block_adjacent_to_robot(block, state):
                adjacent_block_color = block.color
        return CleanupL1Robot(robot_room, adjacent_block_color)

    @staticmethod
    def _position_to_room_color(rooms, position):
        '''
        Args:
            rooms (list) of CleanupRoom objects
            position (tuple)

        Returns:
            room_color (str)
        '''
        for room in rooms: # type: CleanUpRoom
            if position in room.points_in_room:
                return room.color
        return None

    @staticmethod
    def _is_block_adjacent_to_robot(state, block):
        '''
        Args:
            state (CleanUpState)
            block (CleanUpBlock)

        Returns:
            is_adjacent (bool): true if the agent is horizontally or vertically adjacent to the block
        '''
        manhattan_distance = abs(state.x - block.x) + abs(state.y - block.y)
        return manhattan_distance <= 1
