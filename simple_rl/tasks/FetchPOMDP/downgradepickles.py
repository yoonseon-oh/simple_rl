import pickle
def downgrade_pickle(location):
	p = pickle.load(open(location, "rb"))
	pickle.dump(p,open(location + "2","wb"), protocol = 2)

downgrade_pickle("./special pickles/look stddev of 1.6/Perseus2 value iteration 9 time 2018-08-25 20.22.42.44.pickle")