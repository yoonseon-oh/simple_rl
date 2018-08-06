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

class FlatDiscreteBeliefState(BeliefState):
    '''

    '''
    def __init__(self, data):
        '''
        :param data: [known part of state, 1D probability distribution (list) over the unknown part of state]
        '''
        BeliefState.__init__(self,data)
    def belief(self, state):
       '''
       :param state: Index of state you wish to test
       :return: Probability that state is true
       '''
       return self.data[1][state]
    def sample(self):
        '''
        :return: A sample for the unknown part of the state.
        '''
        return cstuff.sample_state(self.data[1])
    @staticmethod
    def generate(length,known_data = [], type = "uniform"):
        if type == "uniform":
            return [known_data,[1.0/float(length) for i in range(length)]]
    def get_most_likely(self):
        pd = self.data[1]
        most_likely_index = 0
        highest_probability = 0
        for i in range(len(pd)):
            if pd[i] > highest_probability:
                highest_probability = pd[i]
                most_likely_index = i
        return [most_likely_index,highest_probability]