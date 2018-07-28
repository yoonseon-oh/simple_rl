from simple_rl.mdp.StateClass import State

class BeliefState(State):
    '''
     Abstract class defining a belief state, i.e a probability distribution over states
    '''
    def __init__(self, data=[]):
        State.__init__(self, data=data)

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