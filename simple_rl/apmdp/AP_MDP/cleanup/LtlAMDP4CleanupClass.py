import sympy
import sys
sys.path.append('../spot-2.8.7')
import time
import glob
from datetime import datetime
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
from simple_rl.apmdp.AP_MDP.cleanup.CleanupDrawing import *

from simple_rl.run_experiments import run_agents_on_mdp

class LTLAMDP():
    def __init__(self, ltlformula, ap_maps, env_file=[], slip_prob=0.01, verbose=False):
        '''

        :param ltlformula: string, ltl formulation ex) a & b
        :param ap_maps: atomic propositions are denoted by alphabets. It should be mapped into states or actions
                        ex) {a:[(int) level, 'action' or 'state', value], b: [0,'action', 'south']
        '''
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.ap_maps = {}
        for ap in self.automata.APs:
            self.ap_maps[ap] = ap_maps[ap]

        self.cube_env = env_file[0] #build_cube_env() #define environment
        # simplify automata
        self._simplify_dict_()
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
            print('solve: buchi path', np)
            flag_success = True
            cur_path = q_paths[np] # current q path
            cur_words = q_words[np] # current q words
            cur_state = init_state = CleanupQState(init_state=init_loc, q=-1, obj_loc=self.cube_env['obj_to_locs'])

            action_seq = []
            state_seq = []
            cur_stay= []

            len_action = 0
            for tt in range(0, len(cur_words)):
                # do not find a solution again if the problem is solved once,
                if (cur_path[tt], cur_path[tt+1], cur_state) in Paths_saved.keys():

                    state_seq_sub = Paths_saved[(cur_path[tt], cur_path[tt+1], cur_state)]['state_seq_sub']
                    action_seq_sub = Paths_saved[(cur_path[tt], cur_path[tt + 1], cur_state)]['action_seq_sub']
                    backup_num_sub = 0
                    cur_stay = Paths_saved[(cur_path[tt], cur_path[tt + 1], cur_state)]['cur_stay']

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
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L0(init_state=cur_state, constraints=constraints, ap_maps =sub_ap_maps)

                    elif sub_level == 1:
                        # solve
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L1(init_state=cur_state, constraints=constraints, ap_maps=sub_ap_maps)
                    elif sub_level == 2:
                        # solve
                        action_seq_sub, state_seq_sub, backup_num_sub = self._solve_subproblem_L2(init_state=cur_state, constraints=constraints, ap_maps=sub_ap_maps)


                    # Save solution
                    Paths_saved[(cur_path[tt], cur_path[tt+1], cur_state)] = {'state_seq_sub': state_seq_sub,
                                                                        'action_seq_sub': action_seq_sub, 'backup_num_sub': backup_num_sub,
                                                                            'cur_words': cur_words, 'cur_stay': cur_stay}
                # update
                backup_num = backup_num + backup_num_sub
                state_seq.append(state_seq_sub)
                action_seq.append(action_seq_sub)
                len_action = len_action + len(action_seq_sub)

                # update initial state
                cur_state = copy.deepcopy(state_seq_sub[-1])
                cur_state.set_terminal(False)
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

                        print('\t----------------------------------------')
                    print("state", state_seq[k][-1])
                    print("=====================================================")

        return state_seq_opt, action_seq_opt, len_action_opt, backup_num

    def _solve_subproblem_L0(self, init_state=(1, 1, -1), constraints={},
                             ap_maps={}, verbose=False):
        mdp = CleanupQMDP(init_state=init_state, env_file = [self.cube_env], constraints = constraints, ap_maps = ap_maps,
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


    def _solve_subproblem_L1(self, init_state, constraints={}, ap_maps={},
                             verbose=True):

        # define l0 domain
        l0Domain = CleanupQMDP(init_state=init_state, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps,
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

    def _solve_subproblem_L2(self, init_state, constraints={},
                             ap_maps={}, verbose=False):

        # define l0 domain
        l0Domain = CleanupQMDP(init_state=init_state, env_file=[self.cube_env], constraints=constraints,
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

            l2Root = CleanupRootL2GroundedAction(l2Domain.ACTIONS[0], a2rt2, l2Domain,
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


    def _simplify_dict_(self):
        trans_dict_simplified = {}

        for key in self.automata.trans_dict.keys():
            cur_dict = {}
            for key2 in self.automata.trans_dict[key].keys():
                if key2 == '1':
                    cur_dict[key2] = self.automata.trans_dict[key][key2]
                elif not self._check_contradiction(logical_formula=key2):
                    cur_dict[key2] = self.automata.trans_dict[key][key2]

            trans_dict_simplified[key] = cur_dict

        self.automata.trans_dict = trans_dict_simplified

    def _check_contradiction(self, logical_formula):
        # define symbols
        aps = [ap for ap in self.automata.APs if ap in logical_formula]
        for ap in aps:
            exec('%s = symbols(\'%s\')' % (ap, ap))

        num_ap = len(aps)

        tf_table = [[True], [False]]  # compute possible tf_table

        aps = self.automata.APs
        for ii in range(1, num_ap):
            ap1 = aps[ii]
            flag_determined = False
            related_list = []  # the index in the list and ap1 cannot be both True

            for jj in range(0, ii):
                ap2 = aps[jj]
                if self.ap_maps[ap1] == self.ap_maps[ap2]:
                    related_list.append((jj, 'same'))
                elif self.ap_maps[ap1][0] == self.ap_maps[ap2][0]:
                    if self.ap_maps[ap1][0] == 'In':
                        if self.ap_maps[ap1][1][0] == self.ap_maps[ap2][1][0]:
                            related_list.append((jj, 'not'))
                    else:
                        related_list.append((jj, 'not'))
                elif self.ap_maps[ap1][0] == 'In' and self.ap_maps[ap2][0] == 'On' and self.ap_maps[ap2][1] == -1:
                    related_list.append((jj, 'not'))
                elif self.ap_maps[ap2][0] == 'In' and self.ap_maps[ap1][0] == 'On' and self.ap_maps[ap1][1] == -1:
                    related_list.append((jj, 'not'))

            # update tf table
            tf_table_new = []
            for tf in tf_table:
                tf_cur = {True: True, False: True}
                for rel in related_list:
                    if rel[1] == 'same':
                        tf_cur[tf[rel[0]]] = True
                        tf_cur[~tf[rel[0]]] = False
                    elif tf_cur[rel[0]]:
                        tf_cur[True] = False
                        tf_cur[False] = True

                if tf_cur[True]:
                    tf_table_new.append(tf+[True])
                if tf_cur[False]:
                    tf_table_new.append(tf+[False])

            tf_table = copy.deepcopy(tf_table_new)

        # Check if ltl formula can be true
        Flag_true = False
        for tf in tf_table:
            Flag_true = Flag_true or (eval(logical_formula)).subs(dict(zip(aps,tf)))

        return not Flag_true



    def format_output(self, state_seq, action_seq, env):
        # compute a room id and an object id of objects
        sseq = []
        aseq = []
        state1_seq = []
        for k in range(len(action_seq)):
            for i in range(len(action_seq[k])):

                state1_seq.append(self.state0_to_1(state_seq[k][i],env))
                #sseq.append(state_seq[k][i].data)
                sseq.append(state_seq[k][i])
                aseq.append(action_seq[k][i])

        state1_seq.append(self.state0_to_1(state_seq[k][-1], env))
        sseq.append(state_seq[k][-1])

        return sseq, aseq, state1_seq

    def state0_to_1(selfe,state, env):
        num_obj = len(state.obj_loc)
        obj_room = [-1] * num_obj
        robot_high = [-1, -1, -1]
        for ii in range(0, num_obj):  # object id
            for jj in range(0, env['num_room']):
                if state.obj_loc[ii] in env['room_to_locs'][jj]:
                    obj_room[ii] = jj

                if (state.x, state.y) in env['room_to_locs'][jj]:
                    robot_high[1] = jj

                if state.x == state.obj_loc[ii][0] and state.y == state.obj_loc[ii][1]:
                    robot_high[0] = ii

        robot_high[2] = state.obj_id
        return robot_high + obj_room


if __name__ == '__main__':

    # Settings
    init_loc = (5,8,-1)
    ltl_formula = '~c U (a & b) '  # ex) 'F(a & F( b & Fc))', 'F a', '~a U b'
    ap_maps = {'a':('On', 1), 'b':('RobotIn', 1),'c':('RobotAt',0)}
    save_dir = '/media/ys/SharedData/Research/AP-MDP-result'

    # initialize
    start_time = time.time()
    cube_env = build_cube_env()
    # figure name
    now = datetime.now()
    datestr = now.strftime('%Y%m%d-%H-%M')
    listfig = glob.glob('{}/{}*.png'.format(save_dir,datestr))
    save_name = '{}-{}.png'.format(datestr,len(listfig))



    # Solve
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=True)
    sseq, aseq, len_actions, backup = ltl_amdp.solve(init_loc, FLAG_LOWEST=False)

    computing_time = time.time() - start_time

    # make the prettier output
    s_seq, a_seq, s1_seq = ltl_amdp.format_output(sseq, aseq, cube_env)

    draw_state_seq(cube_env, s_seq, save_fig ='{}/{}'.format(save_dir,save_name), title=str(ltl_formula) + str(ltl_amdp.ap_maps))
    plt.pause(5)
    for t in range(0, len(a_seq)):
        print(s1_seq[t], a_seq[t])
    print(s1_seq[-1])

    print("Summary")
    print("\t Time: {} seconds, the number of actions: {}, backup: {}"
          .format(round(computing_time, 3), len_actions, backup))













