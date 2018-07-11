from simple_rl.planning.ValueIterationClass import ValueIteration

from collections import defaultdict

class AMDPPolicyGenerator(object):
    '''
     This is a Policy Generating Interface for AMDPs. The purpose of such policy generators is to
     generate a policy for a lower level state abstraction in AMDPs given an AMDP action from a higher
     state abstraction and a lower level state. It also has access to the state mapper allowing generation
     of abstract states
    '''
    def generatePolicy(self, state, grounded_task):
        pass

    def  generateAbstractState(self, state):
        pass

    def getPolicy(self, mdp, verbose=False):
        vi = ValueIteration(mdp)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(mdp.init_state)

        if verbose: print 'Plan for {}:'.format(mdp)
        for i in range(len(action_seq)):
            if verbose: print "\tpi[{}] -> {}".format(state_seq[i], action_seq[i])
            policy[state_seq[i]] = action_seq[i]
        return policy
