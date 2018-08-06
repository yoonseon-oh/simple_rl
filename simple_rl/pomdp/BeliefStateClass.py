from simple_rl.mdp.StateClass import State
import pyximport;
pyximport.install()
from simple_rl.pomdp import cstuff

class BeliefState(State):
    '''
     Abstract class defining a belief state, i.e a probability distribution over states
    '''
    def __init__(self,data):
        State.__init__(self, data)

    def belief(self, state):
        '''
        Args:
            state (State)
        Returns:
            belief[state] (float): probability that agent is in state
        '''
        pass

    def sample(self):
        '''
        Returns:
            sampled_state (State)
        '''
        pass

class FlatFiniteBeliefState(BeliefState):
    def __init__(self, data):
        BeliefState.__init__(self,data)
    def belief(self, state):
        return self.data[1][state]
    def sample(self):
        return cstuff.sample_state(self.data[1])
    @staticmethod
    def generate(length, type = "uniform"):
        if type == "uniform":
            return [1.0/float(length) for i in range(length)]