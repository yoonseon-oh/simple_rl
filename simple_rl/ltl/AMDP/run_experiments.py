import time
from simple_rl.ltl.AMDP.RoomCubePlainMDPClass import RoomCubePlainMDP
from simple_rl.ltl.AMDP.LtlAMDPClass import LTLAMDP
from simple_rl.ltl.settings.build_cube_env_2 import build_cube_env
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

if __name__ == '__main__':

    cube_env1 = build_cube_env()
    ltl_formula = 'F (a & F b )'
    ap_maps = {'a': [1, 'state', 32], 'b': [2, 'state', 4]} #, 'c': [1, 'state', 5]}

    run_num = 10.0
    run_time_plain = 0.0
    run_time_amdp = 0.0
    run_len_plain = 0.0
    run_len_amdp = 0.0
    flag_verbose = False

    for i in range(int(run_num)):
        print("* Trial {}".format(i))

        # Experiment: AMDP
        t, l = run_aMDP(ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
        run_time_amdp = run_time_amdp + t
        run_len_amdp = run_len_amdp + l
        print("  [AMDP]  Time: {} seconds, the number of actions: {}"
              .format(round(t, 3), l))

        # Experiment: Plain MDP
        t, l = run_plain_pMDP(ltl_formula, cube_env1, ap_maps, verbose=flag_verbose)
        run_time_plain = run_time_plain + t
        run_len_plain = run_len_plain + l
        print("  [Plain] Time: {} seconds, the number of actions: {}"
              .format(round(t, 3), l))


    print("* Summary: " + ltl_formula)
    print("  AMDP: {}s, {}".format(round(run_time_amdp / run_num, 3), run_len_amdp / run_num))
    print("  Plain: {}s, {}".format(round(run_time_plain / run_num, 3), run_len_plain / run_num))





