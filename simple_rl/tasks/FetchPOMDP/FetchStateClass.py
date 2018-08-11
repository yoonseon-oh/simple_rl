import copy
from simple_rl.mdp.StateClass import State
from simple_rl.pomdp.BeliefStateClass import FlatDiscreteBeliefState
import pyximport;
pyximport.install()
from simple_rl.tasks.FetchPOMDP import cstuff

class FetchPOMDPState(State, dict):
	def __init__(self, desired_item, last_referenced_item=None, reference_type=None):
		State.__init__(self, {"desired_item": desired_item, "last_referenced_item": last_referenced_item,
		                      "reference_type": reference_type})


#Deciding whether to store a dict with {"known":{}, "unknown":{}} or a single dict and then a list of all known,unknown variables.
class FetchPOMDPBeliefState(FlatDiscreteBeliefState):
	'''
	b[i] = self["unknown"]["desired_item"][i]. self["known"] is dict of known variables
	'''
	def __init__(self, desired_item, last_referenced_item = None, reference_type = None, state = None):
		self.known = ["last_referenced_item","reference_type"]
		self.unknown = ["desired_item"]
		FlatDiscreteBeliefState.__init__(self,
			{"desired_item": desired_item, "last_referenced_item": last_referenced_item,
			 "reference_type": reference_type})
		if state is not None:
			self.update_from_state(state)
	def sample(self):
		sample_dict = {key:self[key] for key in self.known}
		sample_dict.update({"desired_item":cstuff.sample_state(self["desired_item"])})
		return FetchPOMDPState(**sample_dict)

	def belief(self, state):
		'''
		:param state: FetchState you wish to test. To test a desired item only, use bs["desired_item"]["index"]
		:return: Probability that state is true
		'''
		for key in self.known:
			if self[key] != state[key]:
				return 0
		return self["desired_item"][state["desired_item"]]
	def get_most_likely(self):
		pd = self["desired_item"]
		most_likely_index = 0
		highest_probability = 0
		for i in range(len(pd)):
			if pd[i] > highest_probability:
				highest_probability = pd[i]
				most_likely_index = i
		return [most_likely_index, highest_probability]
	def update_from_state(self,state):
		for key in self.known:
			self[key] = state[key]
	def to_state(self, desired_item):
		sample_dict = {key: self[key] for key in self.known}
		sample_dict.update({"desired_item":desired_item})
		return FetchPOMDPState(**sample_dict)
# def get_counts(samples):
# 	counts = {i:0 for i in range(4)}
# 	for s in samples:
# 		counts[s["desired_item"]] += 1
# 	averages = {i: counts[i] / len(samples) for i in range(4)}
# 	return (counts, averages)
# fb = FetchPOMDPBeliefState([0.25 for i in range(4)],None,None)
# samples = [fb.sample() for i in range(100000)]
# (counts,averages) = get_counts(samples)
# print(counts)
# print(averages)
# for s in samples:
# 	print(s)
# print(fb.get_most_likely())
def test():
	st = FetchPOMDPState(0,1,"point")
	bs = FetchPOMDPBeliefState([.25 for i in range(4)], state = st)
	print(bs)
	st["last_referenced_item"] = 3
	print(bs)
	bs.update_from_state(st)
	print(bs)
	print(bs.belief(2))
	st2 = bs.to_state(3)
	print(st2)
def cstuff_belief_update_test():
	st = FetchPOMDPState(0, 1, "point")
	bs = FetchPOMDPBeliefState([1.0/6 for i in range(6)], state=st)
	print(bs)
	o = {"language":set("blue"),"gesture":None}
	bs2 = FetchPOMDPBeliefState(**cstuff.belief_update(bs,o))
	print(bs2)
cstuff_belief_update_test()