import time
import os
from simple_rl.ltl.AMDP.RoomCubePlainMDPClass import RoomCubePlainMDP
from simple_rl.ltl.AMDP.LtlAMDPClass import LTLAMDP
from simple_rl.ltl.settings.build_cube_env_1 import build_cube_env
from simple_rl.planning import ValueIteration

def run_plain_pMDP(ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    mdp = RoomCubePlainMDP(ltl_formula=ltl_formula, env_file=[cube_env],
                           ap_maps=ap_maps)

    value_iter = ValueIteration(mdp, sample_rate=1)
    value_iter.run_vi()

    # Value Iteration
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())

    computing_time = time.time() - start_time

    # Print
    if verbose:
        print("=====================================================")
        print("Plain: Plan for ", ltl_formula)
        for i in range(len(action_seq)):
            room_number, floor_number = mdp._get_abstract_number(state_seq[i])

            print(
                "\t {} in room {} on the floor {}, {}".format(state_seq[i], room_number, floor_number, action_seq[i]))
        room_number, floor_number = mdp._get_abstract_number(state_seq[-1])
        print("\t {} in room {} on the floor {}".format(state_seq[-1], room_number, floor_number))

    return computing_time, len(action_seq)

def run_aMDP(ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=verbose)

    # ltl_amdp.solve_debug()
    sseq, aseq, len_actions = ltl_amdp.solve()

    computing_time = time.time() - start_time

    return computing_time, len_actions

def run_aMDP_lowest(ltl_formula, cube_env, ap_maps, verbose=False):
    start_time = time.time()
    ltl_amdp = LTLAMDP(ltl_formula, ap_maps, env_file=[cube_env], slip_prob=0.0, verbose=verbose)

    # ltl_amdp.solve_debug()
    sseq, aseq, len_actions = ltl_amdp.solve(FLAG_LOWEST=True)

    computing_time = time.time() - start_time

    return computing_time, len_actions


if __name__ == '__main__':

    cube_env1 = build_cube_env()

    # define scenarios for a large environment
    formula_set1 = ['Fa', 'F (a & F b)',  'F(a & F( b & Fc))', '~a U b']
    ap_maps_set1 = {}
    ap_maps_set1[0] = {'a': [2, 'state', 3]}
    ap_maps_set1[1] = {'a': [0, 'state', (2,4,1)], 'b': [1,'state', 7]}
    ap_maps_set1[2] = {'a': [1, 'state', 9], 'b': [2, 'state', 3], 'c': [1, 'state', 18]}
    ap_maps_set1[3] = {'a': [1, 'state', 2], 'b': [2, 'state', 3]}

    # define scenarios for a large environment
    formula_set2 = ['Fa', 'Fa', 'F (a & F b)', 'F(a & F( b & Fc))', '~a U b']
    ap_maps_set2 = {}
    ap_maps_set2[0] = {'a': [1, 'state', 8]}
    ap_maps_set2[1] = {'a': [2, 'state', 4]}
    ap_maps_set2[2] = {'a': [1, 'state', 8], 'b': [2, 'state', 4]}
    ap_maps_set2[3] = {'a': [1, 'state', 9], 'b': [2, 'state', 3], 'c': [1, 'state', 18]}

    # simulation settings
    run_num = 100.0   #the number of run
    flag_verbose = False  # Show result paths

    # select the world (1: small, 2: large cube world)
    formula_set = formula_set1
    ap_maps_set = ap_maps_set1

    for num_case in [0,1,3]:
        file = open("{}/results/result_time.txt".format(os.getcwd()), "a")

        ltl_formula = formula_set[num_case]
        ap_maps = ap_maps_set[num_case]

        #initialize
        run_time_plain = 0.0
        run_time_amdp = 0.0
        run_time_amdp_lowest = 0.0
        run_len_plain = 0.0
        run_len_amdp = 0.0
        run_len_amdp_lowest = 0.0

        for i in range(int(run_num)):
            print("* Trial {}".format(i))

            # Experiment: AMDP
            print("[Trial {}] AP-MDP ----------------------------------------".format(i))
            t, l = run_aMDP(ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_amdp = run_time_amdp + t
            run_len_amdp = run_len_amdp + l
            print("  [AP-MDP]  Time: {} seconds, the number of actions: {}"
                  .format(round(t, 3), l))

            # Experiment: decomposed LTL and solve it at the lowest level
            print("[Trial {}] AP-MDP at level 0 ----------------------------------------".format(i))
            t, l = run_aMDP_lowest(ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_amdp_lowest = run_time_amdp_lowest + t
            run_len_amdp_lowest = run_len_amdp_lowest + l
            print("  [AP-MDP at level 0]  Time: {} seconds, the number of actions: {}"
                  .format(round(t, 3), l))

            # Experiment: Plain MDP

            print("[Trial {}] Plain ----------------------------------------".format(i))
            t, l = run_plain_pMDP(ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
            run_time_plain = run_time_plain + t
            run_len_plain = run_len_plain + l
            print("  [Plain] Time: {} seconds, the number of actions: {}"
                  .format(round(t, 3), l))


        print("* Summary: " + ltl_formula)
        print("  AP-MDP: {}s, {}".format(round(run_time_amdp / run_num, 3), run_len_amdp / run_num))
        print("  AP-MDP at level 0: {}s, {}".format(round(run_time_amdp_lowest / run_num, 3), run_len_amdp_lowest / run_num))
        print("  Product-MDP: {}s, {}".format(round(run_time_plain / run_num, 3), run_len_plain / run_num))

        file.write("=================================================\n")
        file.write("Run {} times\n".format(run_num))
        file.write("Task:\t"+ltl_formula+"\n")
        file.write("AP:\t{}\n".format(ap_maps))
        file.write("AP-MDP:\t{}s, {}\n".format(round(run_time_amdp / run_num, 3), run_len_amdp / run_num))
        file.write("AP-MDP at level 0:\t{}s, {}\n".format(round(run_time_amdp_lowest / run_num, 3),
                                                  run_len_amdp_lowest / run_num))
        file.write("Product-MDP:\t{}s, {}\n".format(round(run_time_plain / run_num, 3), run_len_plain / run_num))

        file.close()




