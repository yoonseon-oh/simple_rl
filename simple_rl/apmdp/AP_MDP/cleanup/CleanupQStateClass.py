''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class CleanupQState(State):
    ''' Class for Grid World States '''
    # Object
    # init_state = [x,y, obj_id]
    # obj_id = -1 if a robot is not carrying an object
    # obj_loc = [(1,3),(1,5),(5,6),(6,8)]
    # #obj_att:{0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}

    def __init__(self, init_state, q, obj_loc):


        State.__init__(self, data=list(init_state) + [q] + obj_loc)
        self.x = init_state[0]
        self.y = init_state[1]
        self.obj_id = init_state[2] # ID of the object a robot is carrying
        self.q = q
        self.obj_loc = obj_loc
        #0self.obj_att = obj_att

    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.obj_id) + ","+ str(self.q)+","+ str(self.obj_loc)+ ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y and self.q==other.q and self.obj_id == other.obj_id

    def update_data(self):
        self.data = [self.x, self.y, self.obj_id, self.q] + self.obj_loc

