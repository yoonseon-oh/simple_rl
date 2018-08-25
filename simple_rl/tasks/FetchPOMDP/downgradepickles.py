import pickle
def downgrade_pickle(location):
	p = pickle.load(open(location, "rb"))
	pickle.dump(p,open(location + "2","wb"), protocol = 2)

downgrade_pickle("./special pickles/6 items/Perseus2 value iteration 5 time 2018-08-22 13.06.46.72.pickle")