import sympy
import sys
sys.path.append('../spot-2.8.7')
import time
from simple_rl.apmdp.LTLautomataClass import LTLautomata
import spot

# Generic AMDP imports.
from simple_rl.apmdp.AP_MDP.AMDPSolver2Class import AMDPAgent
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask

# Abstract grid world imports.
from simple_rl.apmdp.AP_MDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.apmdp.AP_MDP.cleanup.CleanupQStateClass import CleanupQState
from simple_rl.apmdp.AP_MDP.cleanup.AbstractCleanupMDPClass import *
from simple_rl.apmdp.AP_MDP.cleanup.AbstractCleanupPolicyGeneratorClass import *
from simple_rl.apmdp.AP_MDP.cleanup.AbstractCleanupStateMapperClass import *

from simple_rl.apmdp.settings.build_cleanup_env_1 import build_cube_env

from simple_rl.run_experiments import run_agents_on_mdp

class LTLAMDP():
    def __init__(self, ltlformula, ap_maps, env_file=[], slip_prob=0.01, verbose=False):
        '''

        :param ltlformula: string, ltl formulation ex) a & b
        :param ap_maps: atomic propositions are denoted by alphabets. It should be mapped into states or actions
                        ex) {a:[(int) level, 'action' or 'state', value], b: [0,'action', 'south']
        '''
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.ap_maps = ap_maps
        self.cube_env = env_file[0] #build_cube_env() #define environment
        #self._generate_AP_tree() # relationship between atomic propositions-TODO:
        # simplify automata
        #self.automata._simplify_dict(self.relation_TF) # TODO:
        self.slip_prob = slip_prob
        self.verbose = verbose

    def solve(self, init_loc=(1, 1, -1), FLAG_LOWEST=False):
        Q_init = self.automata.init_state
        Q_goal = self.automata.get_accepting_states()
        Paths_saved = {}
        backup_num = 0

        [q_paths, q_words]=self.automata.findpath(Q_init, Q_goal[0])   # Find a path of states of automata

        n_path = len(q_paths) # the number of paths

        len_action_opt = 1000
        state_seq_opt = []
        action_seq_opt = []
        # Find a path in the environment
        for np in range(0, n_path):
            flag_success = True
            cur_path = q_paths[np] # current q path
            cur_words = q_words[np] # current q words
            cur_loc = init_loc

            action_seq = []
            state_seq = []
            cur_stay= []

            len_action = 0
            for tt in range(0, len(cur_words)):
                # do not find a solution again if the problem is solved once,
                if (cur_path[tt], cur_path[tt+1], cur_loc) in Paths_saved.keys():

                    state_seq_sub = Paths_saved[(cur_path[tt], cur_path[tt+1], cur_loc)]['state_seq_sub']
                    action_seq_sub = Paths_saved[(cur_path[tt], cur_path[tt + 1], cur_loc)]['action_seq_sub']
                    backup_num_sub = 0
                    cur_stay = Paths_saved[(cur_path[tt], cur_path[tt + 1], cur_loc)]['cur_stay']

                else:
                    trans_fcn = self.automata.trans_dict[cur_path[tt]]
                    # 1. extract constraints
                    constraints = {}
                    constraints['goal'] = cur_words[tt]
                    constraints['stay'] = [s for s in trans_fcn.keys() if trans_fcn[s] == cur_path[tt]][0]
                    cur_stay.append(constraints['stay'])

                    # 2. Parse: Which level corresponds to the current sub - problem
                    sub_ap_maps = {}
                    sub_level = 2
                    for ap in self.ap_maps.keys():
                        if ap in constraints['goal'] or ap in constraints['stay']:
                            sub_ap_maps[ap] = self.ap_maps[ap]
                            if sub_ap_maps[ap][0] == 'In':
                                sub_level = min(sub_level, 2)
                            elif sub_ap_maps[ap][0] in ['RobotIn', 'RobotAt', 'On']:
                                sub_level = min(sub_level, 1)

                    print(sub_level, sub_ap_maps)
                    # solve at the lowest level
                    if FLAG_LOWEST:
                        sub_level = 0

                    if self.verbose:
                        print("----- Solve in level {} MDP : goal {}, stay {} -----".format(sub_level,constraints['goal'], constraints['stay']))
                    # 3. Solve AMDP
                    if sub_level == 0:
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L0(init_locs=cur_loc, constraints=constraints, ap_maps =sub_ap_maps)

                    elif sub_level == 1:
                        # solve
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L1(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps)
                    elif sub_level == 2:
                        # solve
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L2(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps)


                    # Save solution
                    Paths_saved[(cur_path[tt], cur_path[tt+1], cur_loc)] = {'state_seq_sub': state_seq_sub,
                                                                        'action_seq_sub': action_seq_sub, 'backup_num_sub': backup_num_sub,
                                                                            'cur_words': cur_words, 'cur_stay': cur_stay}
                # update
                backup_num = backup_num + backup_num_sub
                state_seq.append(state_seq_sub)
                action_seq.append(action_seq_sub)
                len_action = len_action + len(action_seq_sub)


                #cur_loc = (state_seq_sub[-1].x, state_seq_sub[-1].y, state_seq_sub[-1].z)
                if state_seq_sub[-1].q != 1:
                    flag_success = False
                    break

            if flag_success:
                if len_action_opt > len_action:
                    state_seq_opt = state_seq
                    action_seq_opt = action_seq
                    len_action_opt = len_action
                if self.verbose:    # Show results
                    print("=====================================================")
                    if flag_success:
                        print("[Success] Plan for a path {} in DBA".format(np))
                    else:
                        print("[Fail] Plan for a path {} in DBA".format(np))
                    for k in range(len(action_seq)):
                        print("Goal: {}, Stay: {}".format(cur_words[k], cur_stay[k]))
                        for i in range(len(action_seq[k])):
                            print("state", state_seq[k][i], action_seq[k][i])
                            #room_number, floor_number = self._get_abstract_number(state_seq[k][i])  #TODO: ???

                            #print("\t {} in room {} on the floor {}, {}".format(state_seq[k][i], room_number, floor_number, action_seq[k][i]))
                        print('\t----------------------------------------')
                    #room_number, floor_number = self._get_abstract_number(state_seq[k][-1])
                    #print("\t {} in room {} on the floor {}".format(state_seq[k][-1], room_number, floor_number))
                    print("state", state_seq[k][-1])
                    print("=====================================================")

        return state_seq_opt, action_seq_opt, len_action_opt, backup_num

    def _solve_subproblem_L0(self, init_locs=(1, 1, -1), constraints={},
                             ap_maps={}, verbose=False):
        mdp = CleanupQMDP(init_robot=init_locs, env_file = [self.cube_env], constraints = constraints, ap_maps = ap_maps,
                          slip_prob=self.slip_prob)
        value_iter = ValueIteration(mdp, sample_rate = 1, max_iterations=50)
        value_iter.run_vi()
        num_backup = value_iter.get_num_backups_in_recent_run()

        # Value Iteration.
        action_seq, state_seq = value_iter.plan(mdp.get_init_state())

        if verbose:
            print("Plan for", mdp)
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, num_backup


    def _solve_subproblem_L1(self, init_locs=(1, 1, 1), constraints={}, ap_maps={},
                             verbose=False):

        # define l0 domain
        l0Domain = CleanupQMDP(init_robot=init_locs, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                               slip_prob=self.slip_prob)
        backup_num = 0
        # if the current state satisfies the constraint already, we don't have to solve it.
        if l0Domain.init_state.q == 1:
            action_seq = []
            state_seq = [l0Domain.init_state]
        else:
            # define l1
            l1_state_mapper = AbstractCleanupL1StateMapper(l0Domain)
            l1_state = l1_state_mapper.map_state(l0Domain.init_state)
            l1Domain = CleanupL1MDP(l1_state, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
                                 slip_prob=self.slip_prob)

            policy_generators = []
            l0_policy_generator = CleanupL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
            l1_policy_generator = CleanupL1PolicyGenerator(l0Domain, l1_state_mapper,
                                                           env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

            policy_generators.append(l0_policy_generator)
            policy_generators.append(l1_policy_generator)

            # 2 levels
            # list of primitive actions (l0 domain)
            l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
            # list of nonprimitive actions (l1 domain)
            여기여기
            a2rt = [CleanupL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
            l1Root = CleanupRootL1GroundedAction(l1Domain.ACTIONS[0], a2rt, l1Domain,
                                              l1Domain.terminal_func, l1Domain.reward_func,
                                                 constraints=constraints, ap_maps=ap_maps)

            agent = AMDPAgent(l1Root, policy_generators, l0Domain)
            agent.solve()
            backup_num = agent.backup_num

            # Plan
            state = l0Domain.init_state
            action_seq = []
            state_seq = [state]
            while state in agent.policy_stack[0].keys():
                action = agent.policy_stack[0][state]
                state = l0Domain._transition_func(state, action)

                action_seq.append(action)
                state_seq.append(state)

        if verbose:
            print("Plan")
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, backup_num

    def _solve_subproblem_L2(self, init_locs=(1, 1, -1), constraints={},
                             ap_maps={}, verbose=False):
        # define l0 domain
        l0Domain = CleanupQMDP(init_robot=init_locs, env_file=[self.cube_env], constraints=constraints,
                               ap_maps=ap_maps, slip_prob= self.slip_prob)
        backup_num = 0
        # if the current state satisfies the constraint already, we don't have to solve it.
        if l0Domain.init_state.q == 1:
            action_seq = []
            state_seq = [l0Domain.init_state]
        else:
            # define l1 domain
            l1_state_mapper = AbstractCleanupL1StateMapper(l0Domain)
            l1_state = l1_state_mapper.map_state(l0Domain.init_state)
            l1Domain = CleanupL1MDP(l1_state=l1_state, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

            # define l2 domain
            l2_state_mapper = AbstractCleanupL2StateMapper(l1Domain)
            l2_state = l2_state_mapper.map_state(l1_state)
            l2Domain = CleanupL2MDP(l2_state, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

            policy_generators = []
            l0_policy_generator = CleanupL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
            l1_policy_generator = CleanupL1PolicyGenerator(l0Domain, l1_state_mapper,
                                                        env_file=[self.cube_env], constraints=constraints,
                                                        ap_maps=ap_maps)
            l2_policy_generator = CleanupL2PolicyGenerator(l1Domain, l2_state_mapper,
                                                        env_file=[self.cube_env], constraints=constraints,
                                                        ap_maps=ap_maps)

            policy_generators.append(l0_policy_generator)
            policy_generators.append(l1_policy_generator)
            policy_generators.append(l2_policy_generator)

            # 2 levels
            l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
            a2rt = [CleanupL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
            a2rt2 = [CleanupL2GroundedAction(a, a2rt, l1Domain) for a in l2Domain.ACTIONS]

            l2Root = CleanupRootL2GroundedAction(l2Domain.Actions[0], a2rt2, l2Domain,
                                              l2Domain.terminal_func, l2Domain.reward_func, constraints=constraints,
                                              ap_maps=ap_maps)

            agent = AMDPAgent(l2Root, policy_generators, l0Domain)

            # Test - base, l1 domain
            l2Subtasks = [PrimitiveAbstractTask(action) for action in l1Domain.ACTIONS]

            agent.solve()
            backup_num = agent.backup_num

            # Plan: Extract action seq, state_seq
            state = l0Domain.init_state
            action_seq = []
            state_seq = [state]
            while state in agent.policy_stack[0].keys():
                action = agent.policy_stack[0][state]
                state = l0Domain._transition_func(state, action)

                action_seq.append(action)
                state_seq.append(state)

        # Debuging
        if verbose:
            print("Plan")
            for i in range(len(action_seq)):
                print("\t", state_seq[i], action_seq[i])
            print("\t", state_seq[-1])

        return action_seq, state_seq, backup_num


    def _generate_AP_tree(self): # return the relationship between atomic propositions
        # TODO: WRONG CHECK!
        relation_TF = {}
        for key in self.ap_maps.keys():
            level = self.ap_maps[key][0]  # current level
            lower_list = []
            notlower_list = []
            samelevel_list = []
            higher_list = []
            nothigher_list = []

            ap = self.ap_maps[key]

            if level == 0:   # the current level
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0:  # level 0
                        samelevel_list.append(key2)
                    if ap2[0] == 1:  # level 1
                        if ap2[1] == 'state' and ap[2] in self.cube_env['room_to_locs'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)
                    if ap2[0] == 2:  # level 2
                        if ap2[1] == 'state' and ap[2] in self.cube_env['floor_to_locs'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)

            if level == 1:
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0 and ap2[1] == 'state':  # lower
                        if ap2[2] in self.cube_env['room_to_locs'][ap[2]]:
                            lower_list.append(key2)
                        else:
                            notlower_list.append(key2)

                    if self.ap_maps[key2][0] == 1: # same level
                        samelevel_list.append(key2)

                    if ap2[0] == 2 and ap2[1] == 'state': # higher level
                        if ap[2] in self.cube_env['floor_to_rooms'][ap2[2]]:
                            higher_list.append(key2)
                        else:
                            nothigher_list.append(key2)

            if level == 2:
                for key2 in self.ap_maps.keys():
                    ap2 = self.ap_maps[key2]
                    if ap2[0] == 0 and ap2[1] == 'state':  # lower
                        if ap2[2] in self.cube_env['floor_to_locs'][ap[2]]:
                            lower_list.append(key2)
                        else:
                            notlower_list.append(key2)

                    if ap2[0] == 1 and ap2[2] in self.cube_env['floor_to_rooms'][self.ap_maps[key][2]]:
                        lower_list.append(key2)
                    elif self.ap_maps[key2][0] == 1:
                        notlower_list.append(key2)

                    if ap2[0] == 2:
                        samelevel_list.append(key2)

            relation_TF[key] = {'lower': lower_list, 'same': samelevel_list, 'lower_not': notlower_list,
                                'higher': higher_list, 'higher_not': nothigher_list}

        self.relation_TF = relation_TF


    def format_output(self, state_seq, action_seq):
        sseq = []
        aseq = []
        room_seq = []
        floor_seq = []
        for k in range(len(action_seq)):

            for i in range(len(action_seq[k])):
                room_number, floor_number = self._get_abstract_number(state_seq[k][i])

                sseq.append(state_seq[k][i].data[0:3])
                aseq.append(action_seq[k][i])
                room_seq.append(room_number)
                floor_seq.append(floor_number)

        room_number, floor_number = self._get_abstract_number(state_seq[k][-1])

        sseq.append(state_seq[k][-1].data[0:3])
        room_seq.append(room_number)
        floor_seq.append(floor_number)

        return sseq, aseq, room_seq, floor_seq

if __name__ == '__main__':

    cube_env = build_cube_env()

    init_loc = (1,1,-1)
    ltl_formula = 'F a'  # ex) 'F(a & F( b & Fc))', 'F a', '~a U b'
    ap_maps = {'a':('RobotIn', 1)}
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=True)

    sseq, aseq, len_actions, backup = ltl_amdp.solve(init_loc, FLAG_LOWEST=False)

    computing_time = time.time() - start_time

    # make the prettier output
    s_seq, a_seq, r_seq, f_seq = ltl_amdp.format_output(sseq, aseq)

#    for t in range(0, len(a_seq)):
#        print("\t {} in room {} on the floor {}, {}".format(s_seq[t], r_seq[t], f_seq[t], a_seq[t]))
#    print("\t {} in room {} on the floor {}".format(s_seq[-1], r_seq[-1], f_seq[-1]))

    print("Summary")
    print("\t Time: {} seconds, the number of actions: {}, backup: {}"
          .format(round(computing_time, 3), len_actions, backup))














