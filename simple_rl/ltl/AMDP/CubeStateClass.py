''' GridWorldStateClass.py: Contains the GridWorldState class. '''

# Other imports.
from simple_rl.mdp.StateClass import State

class CubeState(State):
    ''' Class for Grid World States '''

    def __init__(self, x, y, z):
        State.__init__(self, data=[x, y, z])
        self.x = round(x, 5)
        self.y = round(y, 5)
        self.z = round(z, 5)


    def __hash__(self):
        return hash(tuple(self.data))

    def __str__(self):
        return "s: (" + str(self.x) + "," + str(self.y) + "," + str(self.z)+")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, State) and self.x == other.x and self.y == other.y and self.z == other.z
