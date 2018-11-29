import sympy
import spot
from simple_rl.ltl.LTLautomataClass import LTLautomata

# Generic AMDP imports.
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask

# Abstract grid world imports.
from simple_rl.ltl.AMDP.RoomCubeMDPClass import RoomCubeMDP
from simple_rl.ltl.AMDP.RoomCubeStateClass import RoomCubeState
from simple_rl.ltl.AMDP.AbstractCubeMDPClass import *
from simple_rl.ltl.AMDP.AbstractCubePolicyGeneratorClass import *
from simple_rl.ltl.AMDP.AbstractCubeStateMapperClass import *

from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env


class LTLAMDP():
    def __init__(self, ltlformula, ap_maps):
        '''

        :param ltlformula: string, ltl formulation ex) a & b
        :param ap_maps: atomic propositions are denoted by alphabets. It should be mapped into states or actions
                        ex) {a:[(int) level, 'action' or 'state', value], b: [0,'action', 'south']
        '''
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.ap_maps = ap_maps
        self.cube_env = build_cube_env() #define environment

    def solve_debug(self):
        constraints = {'goal': 'a', 'stay': '~b'}
        sub_ap_maps = {'a': (1, 'state', 3), 'b': (1, 'state', 2), 'c': (0, 'state', (1, 4, 1))}

        # 2. Parse: Which level corresponds to the current sub - problem
        sub_level = 2
        for ap in sub_ap_maps.keys():
            if ap in constraints['goal'] or ap in constraints['stay']:

                sub_level = min(sub_level, sub_ap_maps[ap][0])

        # 3. Solve AMDP

        if sub_level == 0:
            self._solve_subproblem_L0(constraints=constraints, ap_maps=sub_ap_maps)

        elif sub_level == 1:
            # solve
            self._solve_subproblem_L1(constraints=constraints, ap_maps=sub_ap_maps)

    def solve(self, init_loc=(1,1,1)):
        Q_init = self.automata.init_state
        Q_goal = self.automata.get_accepting_states()

        [q_paths, q_words]=self.automata.findpath(Q_init, Q_goal[0])   # Find a path of states of automata

        n_path = len(q_paths) # the number of paths

        # Find a path in the environment
        for np in [3]:#range(1,n_path):
            cur_path = q_paths[np] # current q path
            cur_words = q_words[np] # current q words
            cur_loc = init_loc

            action_seq = []
            state_seq = []
            for tt in range(0, len(cur_words)):
                trans_fcn = self.automata.trans_dict[cur_path[tt]]
                # 1. extract constraints
                constraints = {}
                constraints['goal'] = cur_words[tt]
                constraints['stay'] = [s for s in trans_fcn.keys() if trans_fcn[s] == cur_path[tt]][0]

                # 2. Parse: Which level corresponds to the current sub - problem
                sub_ap_maps = {}
                sub_level = 2
                for ap in self.ap_maps.keys():
                    if ap in constraints['goal'] or ap in constraints['stay']:
                        sub_ap_maps[ap] = ap_maps[ap]
                        sub_level = min(sub_level, sub_ap_maps[ap][0])

                # 3. Solve AMDP
                if sub_level == 0:
                    action_seq_sub, state_seq_sub = self._solve_subproblem_L0(init_locs=cur_loc, constraints=constraints, ap_maps =sub_ap_maps)

                elif sub_level == 1:
                    # solve
                    action_seq_sub, state_seq_sub = self._solve_subproblem_L1(init_locs=cur_loc, constraints=constraints, ap_maps=sub_ap_maps)

                # update
                state_seq.append(state_seq_sub)
                action_seq.append(action_seq_sub)
                cur_loc = (state_seq_sub[-1].x, state_seq_sub[-1].y, state_seq_sub[-1].z)

            print('Plan')
            for k in range(len(action_seq)):
                for i in range(len(action_seq[k])):
                    room_number = self._get_room_number(state_seq[k][i])

                    print("\t {} in room {}, {}".format(state_seq[k][i], room_number, action_seq[k][i]))
            print("\t {} in room {}".format(state_seq[k][-1], self._get_room_number(state_seq[k][-1])))

    def _get_room_number(self, state):
        room_number = 0
        for r in range(1, self.cube_env['num_room'] + 1):
            if (state.x, state.y, state.z) in self.cube_env['room_to_locs'][r]:
                room_number = r

        return room_number


    def _solve_subproblem_L0(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}): #TODO
        mdp = RoomCubeMDP(init_loc=init_locs, env_file = [self.cube_env], constraints = constraints, ap_maps = ap_maps)
        value_iter = ValueIteration(mdp, sample_rate = 5)
        value_iter.run_vi()

        # Value Iteration.
        action_seq, state_seq = value_iter.plan(mdp.get_init_state())

        print("Plan for", mdp)
        for i in range(len(action_seq)):
            print("\t", state_seq[i], action_seq[i])
        print("\t", state_seq[-1])

        return action_seq, state_seq


    def _solve_subproblem_L1(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}):

        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        # define l1 domain
        start_room = l0Domain.get_room_numbers(init_locs)
        l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        policy_generators = []
        l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
        l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain), env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        policy_generators.append(l0_policy_generator)
        policy_generators.append(l1_policy_generator)

        # 2 levels
        l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
        a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
        l1Root = CubeRootL1GroundedAction(l1Domain.action_for_room_number(0), a2rt, l1Domain,
                                          l1Domain.terminal_func, l1Domain.reward_func, constraints=constraints, ap_maps=ap_maps)

        agent = AMDPAgent(l1Root, policy_generators, l0Domain)
        agent.solve()

        state = RoomCubeState(init_locs[0], init_locs[1], init_locs[2], 0)
        action_seq = []
        state_seq = [state]
        while state in agent.policy_stack[0].keys():
            action = agent.policy_stack[0][state]
            state = l0Domain._transition_func(state, action)

            action_seq.append(action)
            state_seq.append(state)

        print("Plan")
        for i in range(len(action_seq)):
            print("\t", state_seq[i], action_seq[i])
        print("\t", state_seq[-1])
        return action_seq, state_seq

    def _solve_subproblem_L2(self, init_locs=(1, 1, 1), constraints={}, ap_maps={}):
        # define l0 domain
        l0Domain = RoomCubeMDP(init_loc=init_locs, env_file=[self.cube_env], constraints=constraints,
                               ap_maps=ap_maps)

        # define l1 domain
        start_room = l0Domain.get_room_numbers(init_locs)
        start_floor = l0Domain.get_floor_numbers(init_locs)

        l1Domain = CubeL1MDP(start_room, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)
        l2Domain = CubeL2MDP(start_floor, env_file=[self.cube_env], constraints=constraints, ap_maps=ap_maps)

        policy_generators = []
        l0_policy_generator = CubeL0PolicyGenerator(l0Domain, env_file=[self.cube_env])
        l1_policy_generator = CubeL1PolicyGenerator(l0Domain, AbstractCubeL1StateMapper(l0Domain),
                                                    env_file=[self.cube_env], constraints=constraints,
                                                    ap_maps=ap_maps)
        l2_policy_generator = CubeL1PolicyGenerator(l1Domain, AbstractCubeL2StateMapper(l1Domain),
                                                    env_file=[self.cube_env], constraints=constraints,
                                                    ap_maps=ap_maps)

        policy_generators.append(l0_policy_generator)
        policy_generators.append(l1_policy_generator)
        policy_generators.append(l2_policy_generator)

        # 2 levels
        l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
        l2Subtasks = [NonPrimitiveAbstractTask(action) for action in l1Domain.ACTIONS]
        a2rt = [CubeL1GroundedAction(a, l1Subtasks, l0Domain) for a in l1Domain.ACTIONS]
        a2rt2 = [CubeL2GroundedAction(a, l2Subtasks, l1Domain) for a in l2Domain.ACTIONS]

        l1Root = CubeRootL1GroundedAction(l1Domain.action_for_room_number(0), a2rt, l1Domain,
                                          l1Domain.terminal_func, l1Domain.reward_func, constraints=constraints,
                                          ap_maps=ap_maps)
        l2Root = CubeRootL2GroundedAction(l2Domain.action_for_floor_number(0), a2rt2, l2Domain,
                                          l1Domain.terminal_func, l1Domain.reward_func, constraints=constraints,
                                          ap_maps=ap_maps)

        agent = AMDPAgent(l1Root, policy_generators, l0Domain)
        agent.solve()

        return True
        #return action_seq, state_seq




if __name__ == '__main__':
    ltl_formula = 'F (d & F (a & F b))'
    ap_maps = {'a': [1, 'state', 3], 'b': [1, 'state', 5], 'c': [1, 'state', 2], 'd': [0, 'action', 'north']}
    ltl_amdp = LTLAMDP(ltl_formula , ap_maps)
    ltl_amdp.solve()












