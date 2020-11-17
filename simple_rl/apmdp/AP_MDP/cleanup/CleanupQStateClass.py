''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class CleanupQStateClass(State):
    ''' Class for Grid World States '''
    # Object, Room List
    # objlist = {pos:[[x,y],[x,y], ...], color: ['r','g','b',..]}
    def __init__(self, x, y, obj_id, q, objlist, roomlist):
        State.__init__(self, data=[x, y, obj_id, q, objlist, roomlist])
        self.x = x
        self.y = y
        self.obj_id = obj_id # ID of the object a robot is carrying
        self.q = q
        self.objlist = objlist
        self.roomlist = roomlist

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.obj_id)+")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y and self.obj_id == other.obj_id
