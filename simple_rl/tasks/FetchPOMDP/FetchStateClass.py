from simple_rl.mdp.StateClass import State
class FetchPOMDPState(State,dict):
	def __init__(self, data):
		State.__init__(self, data)

f = FetchPOMDPState([0,0])
print(f[1])
