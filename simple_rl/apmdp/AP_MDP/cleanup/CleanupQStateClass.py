''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class CleanupQState(State):
    ''' Class for Grid World States '''
    # Object, Room List
    # obj_loc = [(1,3),(1,5),(5,6),(6,8)], color:{0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}}
    def __init__(self, x, y, obj_id, q, obj_loc, obj_color, roomlist):

        obj_loc_list = []
        for ii in range(0, len(obj_loc)):
            obj_loc_list.extend(obj_loc[ii])

        State.__init__(self, data=[x, y, obj_id, q, objlist])
        self.x = x
        self.y = y
        self.obj_id = obj_id # ID of the object a robot is carrying
        self.q = q
        self.objloc = objlist['pos']
        self.roomlist = roomlist

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.obj_id)+")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y and self.obj_id == other.obj_id
