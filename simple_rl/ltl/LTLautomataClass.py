from sympy import *
import spot

class LTLautomata():
    def __init__(self, ltlformula = 'Fr'):
        #self.APs = ap_dict # atomic propositions, dictionary = {'x': 'L1_1', 'y': 'L1_2'}, 'x': level1, room1
        self.formula = ltlformula # ltl formula ex. Always(x)
        self.ltl2automaton(ltlformula) # table


    def ltl2automaton(self, formula): # automata={state s: transition(s)}, transition(s)={ltl: next state(s')}
        # translate LTL formula to automata
        A = spot.translate(formula,'BA','complete')
        self.aut_spot = A
        self.num_sets = A.num_sets()
        self.num_states = A.num_states()
        self.init_state = A.get_init_state_number()

        # read APs
        self.APs = []
        for ii in range(0,len(A.ap())):
            self.APs.append(A.ap()[ii].ap_name())


        # make transition dictionary
        self.trans_dict = {}
        bdict = A.get_dict()
        for s in range(0, A.num_states()):
            dict_sub = {}
            for t in A.out(s):

                dict_sub[spot.bdd_format_formula(bdict,t.cond).replace('!','~')] = t.dst

            self.trans_dict[s] = dict_sub


    def transition_func(self, q, evaluated_APs):
        # evaluate aps
        # ex. evaluated_APs = {'r': True, 'b': False}
        # evaluated_APs: dict --> string

        # define symbols
        for ap in self.APs:
            exec('%s = symbols(\'%s\')' % (ap,ap))

        # return the next state
        for key in self.trans_dict[q].keys():
            if key == '1':
                return self.trans_dict[q][key]
            flag = (eval(key)).subs(evaluated_APs)

            if flag: # if ltl of the edge is true,
                return self.trans_dict[q][key]    # transition occurs

        return q

    def reward_func(self, q):
        # manually first
        if self.aut_spot.state_is_accepting(q): # get positive reward if it visits an accepting state
            return 100
        elif (~self.aut_spot.state_is_accepting(q)) & ('1' in self.trans_dict[q].keys()): # if the agent cannot reach the accepting state
            if self.trans_dict[q]['1'] == q:
                return -100
        else:
            return -1 # cost of actions

    def terminal_func(self,q):
        if self.aut_spot.state_is_accepting(q): # get positive reward if it visits an accepting state
            return True
        elif (~self.aut_spot.state_is_accepting(q)) & ('1' in self.trans_dict[q].keys()): # if the agent cannot reach the accepting state
            if self.trans_dict[q]['1'] == q:
                return True
        else:
            return False

if __name__ == '__main__':
    #ap_dict = {'x': 'L1_1', 'y':'L1_2'}
    A = LTLautomata('Fr')

    print(A.transition_func(A.init_state,{'r':True}))

