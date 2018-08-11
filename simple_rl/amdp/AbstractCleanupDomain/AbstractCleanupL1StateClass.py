from simple_rl.mdp.StateClass import State

class CleanupL1State(State):
    def __init__(self, robot, doors, rooms, blocks):
        '''
        Args:
            robot (CleanupL1Robot)
            doors (list): list of all the CleanupL1Door objects
            rooms (list): list of all the CleanupL1Room objects
            blocks (list): list of all the CleanupL1Block objects
        '''
        self.robot = robot
        self.doors = doors
        self.rooms = rooms
        self.blocks = blocks

        State.__init__(self, data=[robot, doors, rooms, blocks])

    def __str__(self):
        return str(self.robot) + '\t' + str(self.doors) + '\t' + str(self.rooms) + '\t' + str(self.blocks)

    def __repr__(self):
        return self.__str__()

class CleanupL1Robot(object):
    def __init__(self, current_room, adjacent_block=None):
        '''
        Args:
            current_room (str): color of the agent's current room
            adjacent_block (str): color of the block next to the agent
        '''
        self.current_room = current_room
        self.adjacent_block = adjacent_block

    def __str__(self):
        block = self.adjacent_block if self.adjacent_block is not None else 'NoBlock'
        return 'Robot::room:' + self.current_room + '  adjacent_block:' + block

    def __repr__(self):
        return self.__str__()

class CleanupL1Door(object):
    def __init__(self, connected_rooms):
        '''
        Args:
            connected_rooms (list): list of strings representing the colors of the 2 rooms connected
            by the current door
        '''
        self.connected_rooms = connected_rooms

    def __str__(self):
        return 'Door::' + str(self.connected_rooms[0]) + '_' + str(self.connected_rooms[1])

    def __repr__(self):
        return self.__str__()

class CleanupL1Room(object):
    def __init__(self, room_color):
        '''
        Args:
            room_color (str): color of the current room
        '''
        self.room_color = room_color

    def __str__(self):
        return 'Room::color:' + self.room_color

    def __repr__(self):
        return self.__str__()

class CleanupL1Block(object):
    def __init__(self, current_room, block_color):
        '''
        Args:
            current_room (str): color of the room in which the current block is placed
            block_color (str): color of the current block
        '''
        self.current_room = current_room
        self.block_color = block_color

    def __str__(self):
        return 'Block::color:' + self.block_color + ' in_room:' + str(self.current_room)

    def __repr__(self):
        return self.__str__()
