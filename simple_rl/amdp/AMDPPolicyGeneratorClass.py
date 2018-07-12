from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.mdp.StateClass import State
from simple_rl.amdp.AMDPTaskNodesClass import AbstractTask

from collections import defaultdict

class AMDPPolicyGenerator(object):
    '''
     This is a Policy Generating Interface for AMDPs. The purpose of such policy generators is to
     generate a policy for a lower level state abstraction in AMDPs given an AMDP action from a higher
     state abstraction and a lower level state. It also has access to the state mapper allowing generation
     of abstract states
    '''
    def generatePolicy(self, state, grounded_task):
        '''
        Args:
            state (State)
            grounded_task (AbstractTask)
        Returns:
            policy (defaultdict)
        '''
        pass

    def  generateAbstractState(self, state):
        '''
        Args:
            state (State): state in the lower level MDP
        Returns:
            state (State): state in the current (higher) level of the MDP
        '''
        pass

    def getPolicy(self, mdp, verbose=False):
        '''
        Args:
            mdp (MDP): MDP (same level as the current Policy Generator)
        Returns:
            policy (defaultdict): optimal policy in mdp
        '''
        vi = ValueIteration(mdp)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(mdp.init_state)

        if verbose: print 'Plan for {}:'.format(mdp)
        for i in range(len(action_seq)):
            if verbose: print "\tpi[{}] -> {}".format(state_seq[i], action_seq[i])
            policy[state_seq[i]] = action_seq[i]
        return policy
