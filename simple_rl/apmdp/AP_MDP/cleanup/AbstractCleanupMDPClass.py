# Python imports.
from __future__ import print_function
from collections import defaultdict
import re as re1

# Other imports.
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning import ValueIteration
from simple_rl.amdp.AMDPTaskNodesClass import NonPrimitiveAbstractTask, RootTaskNode
from simple_rl.apmdp.settings.build_cleanup_env_1 import build_cube_env

from sympy import *
import random
import copy

class CleanupL2State(State):
    def __init__(self,  obj_room, q, obj_id = -1, is_terminal=False): # obj_room = [obj0_room, obj1_room, ..., objn_room]
        State.__init__(self, data= [obj_id] + obj_room + [q], is_terminal=is_terminal)
        self.obj_id = obj_id
        self.obj_room = obj_room
        self.q = q # logic state

    def __hash__(self):
        #return hash(tuple(self.data))
        return hash(tuple([self.obj_id] +self.obj_room + [self.q]))

    def __str__(self):
        return 'active {} room numbers of objects {}, Q: {}'.format(self.obj_id, self.obj_room, self.q)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, CleanupL2State) and self.obj_room == other.obj_room and self.q == other.q

class CleanupL2GroundedAction(NonPrimitiveAbstractTask): # relation between L2 task and L1 state
    # l2_action: [ObjID, RoomID]
    def __init__(self, l2_action_str, subtasks, lowerDomain):
        self.action_name = l2_action_str
        self.l1_domain = lowerDomain
        l2_action_parse = l2_action_str.split('_')
        self.goal_constraints = {'goal': 'a', 'stay': '~a', 'lowest': True}  # TODO: lowest?

        if l2_action_parse[0] == "Deactivate":
            self.ap_maps = {'a': ('On', -1)}
        else:
            l2_action = [int(l2_action_parse[1]), int(l2_action_parse[2])]
            self.target_obj = l2_action[0]
            self.goal_room = l2_action[1]
            self.goal_constraints = {'goal': 'a & b', 'stay': '~a | ~b', 'lowest': True}  # TODO: lowest?
            self.ap_maps = {'a': ('RobotIn', self.goal_room), 'b':('On', self.target_obj)} #

        tf, rf = self._terminal_function, self._reward_function

        NonPrimitiveAbstractTask.__init__(self, self.action_name, subtasks, tf, rf)

    def _terminal_function(self, state): # state: L1
        if self.ap_maps['a'][0]== 'On':
            return state.obj_id == -1
        else:
            return state.obj_room[self.target_obj] == self.goal_room and state.obj_id == self.target_obj

    def _reward_function(self, state):
        if state.q == 1:
            return 100
        elif state.q == 0:
            return -1
        else:
            return -100

class CleanupL1State(State):
    def __init__(self, robot_at, robot_in, obj_id, obj_room, q, is_terminal=False):
        State.__init__(self, data=[robot_at, robot_in, obj_id]+obj_room+[q], is_terminal=is_terminal)
        self.robot_at = robot_at
        self.robot_in = robot_in
        self.obj_id = obj_id
        self.obj_room = obj_room
        self.q = q # logic state

    def __hash__(self):
        #return hash(tuple(self.data))
        return hash(tuple([self.robot_at, self.robot_in, self.obj_id]+self.obj_room+[self.q]))

    def __str__(self):
        return 'Agent At {}, In {}, object_ID {}, obj_room {}, Q: {}'.format(self.robot_at, self.robot_in, self.obj_id,
                                                                                          self.obj_room, self.q)

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return isinstance(other, CleanupL1State) and self.robot_at == other.robot_at and self.robot_in == other.robot_in \
                    and self.obj_id == other.obj_id and self.obj_room == other.obj_room and self.q == other.q


class CleanupL1GroundedAction(NonPrimitiveAbstractTask): # L1_action: ['PICKUP', obj_id], ['PLACE'], ['NavRoom', roomid], ['NavObj', objid]
    def __init__(self, l1_action, subtasks, lowerDomain):
        self.action_name = l1_action
        self.l0_domain = lowerDomain
        self.goal_constraints, self.ap_maps = self.extract_goal(l1_action)
        tf, rf = self._terminal_function, self._reward_function

        NonPrimitiveAbstractTask.__init__(self, self.action_name, subtasks, tf, rf)

    def extract_goal(self, l1_action_str):
        l1_action = l1_action_str.split("_")
        constraints = {'goal': 'a', 'stay': '~a', 'lowest': True}  # lowest:
        state = self.l0_domain.cur_state
        high_pose = self.l0_domain.get_highlevel_pose((state.x, state.y, state.obj_id), state.obj_loc)

        if l1_action[0] == 'PLACE':
            constraints = {'goal': 'a', 'stay': '~a', 'lowest': True}  # lowest:
            ap_maps = {'a': ('On', -1)}
        elif l1_action[0] == 'NavRoomAdj':
            ap_maps  = {'a': ('RobotIn', self.l0_domain.transition_table[high_pose['room']][int(l1_action[1])])}
        else:
            l1_action[1] = int(l1_action[1])
            action2predicate = {'PICKUP': 'On', 'NavRoom':'RobotIn', 'NavObj': 'RobotAt', 'NavRoomAdj': 'RobotIn'}
            arg = {'PICKUP': l1_action[1],  'NavRoom': l1_action[1], 'NavObj': l1_action[1]}
            ap_maps = {'a': (action2predicate[l1_action[0]], arg[l1_action[0]])}

        return constraints, ap_maps

    def _terminal_function(self, state): # state : the lowest level state
        goal = self.ap_maps['a']
        flag_terminal = False
        if goal[0] == 'On':
            flag_terminal = (state.obj_id == goal[1])
        elif goal[0] == 'RobotIn':
            flag_terminal = ((state.x,state.y) in self.l0_domain.room_to_locs[goal[1]])
        elif goal[0] == 'RobotAt':
            flag_terminal = ((state.x, state.y) == state.obj_loc[goal[1]])

        return flag_terminal

    def _reward_function(self, state):
        if state.q == 1:
            return 100
        elif state.q == 0:
            return -1
        else:
            return -100

class CleanupRootL2GroundedAction(RootTaskNode):
    def __init__(self, action, subtasks, l2_domain, terminal_func, reward_func, constraints, ap_maps):
        self.action_str = 'Root_' + action
        self.goal_constraints = constraints
        self.ap_maps = ap_maps

        RootTaskNode.__init__(self, self.action_str, subtasks, l2_domain, terminal_func, reward_func)

class CleanupRootL1GroundedAction(RootTaskNode):
    def __init__(self, action, subtasks, l1_domain, terminal_func, reward_func, constraints, ap_maps):
        self.action_str = 'Root_' + action
        #self.goal_state = CubeL1State(CubeL1GroundedAction.extract_goal_room(action_str), is_terminal=True)
        self.goal_constraints = constraints
        self.ap_maps = ap_maps

        RootTaskNode.__init__(self, self.action_str, subtasks, l1_domain, terminal_func, reward_func)

class CleanupL2MDP(MDP):
    ACTIONS = []#[list(x) for x in itertools.product(0,)] #[obj_id, room_id]
    def __init__(self,init_state = [], gamma=0.99, env_file=[], constraints={}, ap_maps={}):
        # state_init: [obj_room1, obj_room2, ..., obj_roomn]
        self.terminal_func = lambda state: state.q != 0
        self.constraints = constraints
        self.ap_maps = ap_maps

        if len(env_file) != 0:
            self.env = env_file[0]
            CleanupL2MDP.ACTIONS = self.env['L2ACTIONS']
        else:
            print("Input: env_file")
        print("MDPL2:", self.constraints, self.ap_maps)
        # initial state
        initial_state = init_state
        initial_state.q = self._evaluate_q(initial_state)
        if initial_state.q != 0:
            initial_state.set_terminal(True)

        MDP.__init__(self, CleanupL2MDP.ACTIONS, self._transition_func, self._reward_func, init_state=initial_state,
                     gamma=gamma)

    def _reward_func(self, state, action):

        next_state = self._transition_func(state, action)

        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100

        return reward

    def _transition_func(self, state, action_str): #[target_object, goal_room]
        action_tmp = action_str.split("_")
        if state.is_terminal():
            return state
        next_state = copy.deepcopy(state)

        if action_str == "Deactivate":
            next_state.obj_id = -1
        else:
            if state.obj_id == -1 or state.obj_id == int(action_tmp[1]):
                action = [int(action_tmp[1]), int(action_tmp[2])]
                # the target object can be moved to only adjacent room
                if action[1] in self.env['transition_table'][state.obj_room[action[0]]]:
                    next_state.obj_room[action[0]] = action[1]
                    next_state.obj_id = action[0]

        next_state.q = self._evaluate_q(next_state)
        if next_state.q != 0:
            next_state.set_terminal(True)

        #print("MDP2-Trans:",state, action_str, next_state)
        return next_state

    def _evaluate_q(self, state):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(state)

        # q state transition
        # define symbols
        for ap in evaluated_APs.keys():
            exec('%s = symbols(\'%s\')' % (ap, ap))
        # evaluation
        if eval(self.constraints['goal']).subs(evaluated_APs):  # goal
            next_q = 1
        elif eval(self.constraints['stay']).subs(evaluated_APs):  # keep planning
            next_q = 0
        else:  # fail
            next_q = -1

        return next_q


    def _evaluate_APs(self, state):
        evaluated_APs = {}

        for ap in self.ap_maps.keys():
            if self.ap_maps[ap][0] == "In":  # an object is in a room
                evaluated_APs[ap] = (state.obj_room[self.ap_maps[ap][1][0]] == self.ap_maps[ap][1][1]) and (state.obj_id == -1)

        return evaluated_APs

    def __str__(self):
        return 'AbstractCleanupL2MDP: InitState: {}, Goal: {}'.format(self.init_state, self.constraints['goal'])

#    @classmethod
#    def action_for_floor_number(self, floor_number):
#        for action in CleanupL2MDP.ACTIONS:
#            if str(floor_number) in action:
#                return action
#        raise ValueError('unable to find action corresponding to floor {}'.format(floor_number))

class CleanupL1MDP(MDP):
    ACTIONS = ["toRoom%d" %ii for ii in range(1, 11)]  # actions??
    def __init__(self, l1_state, gamma=0.99, slip_prob=0.0, env_file=[], constraints = {}, ap_maps = {}):
        # TODO: work
        self.terminal_func = lambda state: state.q != 0
        self.slip_prob = slip_prob
        self.constraints = constraints
        self.ap_maps = ap_maps

        if len(env_file) != 0:
            self.env = env_file[0]
            CleanupL1MDP.ACTIONS = self.env['L1ACTIONS']

        else:
            print("Input: env_file")

        if 'lowest' in self.constraints.keys():  # if L1 is a subproblem, reduce action space
            if self.ap_maps['a'][0] == 'On':
                CleanupL1MDP.ACTIONS = ['PLACE']
            else:
                CleanupL1MDP.ACTIONS = ["NavRoom_"+ str(ii) for ii in range(0, self.env['num_room'])] \
                            +["NavObj_"+ str(self.ap_maps['b'][1]), "PICKUP_"+ str(self.ap_maps['b'][1]) ] \

        else:
            predicates = [val[0] for val  in self.ap_maps.values()]
            predicates = []
            CleanupL1MDP.ACTIONS = ["NavRoom_"+ str(ii) for ii in range(0, self.env['num_room'])] + ["PLACE"]
            for ap in self.ap_maps.values():
                if ap[0] == 'RobotAt':
                    str_act = 'NavObj_' + str(ap[1])
                    if str_act not in CleanupL1MDP.ACTIONS:
                        CleanupL1MDP.ACTIONS.append(str_act)

                elif ap[0] == 'On':
                    if ap[1] != -1:
                        str_act = 'NavObj_' + str(ap[1])
                        if str_act not in CleanupL1MDP.ACTIONS:
                            CleanupL1MDP.ACTIONS.append(str_act)

                        str_act = 'PICKUP_' + str(ap[1])
                        if str_act not in CleanupL1MDP.ACTIONS:
                            CleanupL1MDP.ACTIONS.append(str_act)
                elif ap[0] == 'In':
                    str_act = 'NavObj_' + str(ap[1][0])
                    if str_act not in CleanupL1MDP.ACTIONS:
                        CleanupL1MDP.ACTIONS.append(str_act)

                    str_act = 'PICKUP_' + str(ap[1][0])
                    if str_act not in CleanupL1MDP.ACTIONS:
                        CleanupL1MDP.ACTIONS.append(str_act)

        print("MDPL1:", self.constraints, self.ap_maps, l1_state)
        initial_state = l1_state
        initial_state.q = self.evaluate_q(initial_state)
        if initial_state.q != 0:
            initial_state.set_terminal(True)

        MDP.__init__(self, CleanupL1MDP.ACTIONS, self._transition_func, self._reward_func, init_state=initial_state,
                     gamma=gamma)

        #print("MDP1", self.constraints, self.ap_maps, initial_state)

    def _reward_func(self, state, action):

        next_state = self._transition_func(state, action)

        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100
        #print(state, action, reward)
        return reward


    def _transition_func(self, state, action_str): # action # L1_action: ['PICKUP', obj_id], ['PLACE'], ['NavRoom', roomid], ['NavObj', objid]
        action = action_str.split("_")
        if len(action)>1 : action[1] = int(action[1])
        if state.is_terminal():
            return state

        next_state = copy.deepcopy(state)
        if action[0] == 'PICKUP':
            if state.robot_at == action[1] and state.obj_id ==-1:
                next_state.obj_id = action[1]
                next_state.robot_at = -1

        elif action[0] == 'PLACE':
            next_state.obj_id = -1
            next_state.robot_at = state.obj_id

        elif action[0] == 'NavRoom':
            if action[1] in self.env['transition_table'][state.robot_in]:
                next_state.robot_in = action[1]
                if state.obj_id !=-1: # robot is carrying an object
                    next_state.obj_room[state.obj_id] = action[1]
                next_state.robot_at = -1

        elif action[0] == 'NavRoomAdj':
            if action[1] < len(self.env['transition_table'][state.robot_in]):
                next_state.robot_in = self.env['transition_table'][state.robot_in][action[1]]
                if state.obj_id !=-1: # robot is carrying an object
                    next_state.obj_room[state.obj_id] =next_state.robot_in
                next_state.robot_at = -1

        elif action[0] == 'NavObj':
            if state.robot_in == state.obj_room[action[1]]:
                next_state.robot_at = action[1]

        if next_state.obj_id != -1:
            next_state.obj_room[next_state.obj_id] = next_state.robot_in

        next_state.q = self.evaluate_q(next_state)
        if next_state.q != 0:
            next_state.set_terminal(True)

        #if state == next_state: # if the transition does not work,
        #    next_state.q = -1
        #    next_state.set_terminal(True)

        #print("MDP1 transition: ", state, action, next_state)

        return next_state

    def evaluate_q(self, state):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(state)

        # q state transition
        # define symbols
        for ap in evaluated_APs.keys():
            exec('%s = symbols(\'%s\')' % (ap, ap))
        # evaluation
        if eval(self.constraints['goal']).subs(evaluated_APs):  # goal
            next_q = 1
        elif eval(self.constraints['stay']).subs(evaluated_APs):  # keep planning
            next_q = 0
        else:  # fail
            next_q = -1

        return next_q

    def _evaluate_APs(self, state):
        evaluated_APs = {}
        for ap in self.ap_maps.keys():
            if self.ap_maps[ap][0] == "In": # an object is in a room
               evaluated_APs[ap] = state.obj_room[self.ap_maps[ap][1][0]] == self.ap_maps[ap][1][1] and state.obj_id == -1

            elif self.ap_maps[ap][0] == "On":
               evaluated_APs[ap] = state.obj_id == self.ap_maps[ap][1]

            elif self.ap_maps[ap][0] == "RobotIn": # a robot is in a room
               evaluated_APs[ap] = state.robot_in == self.ap_maps[ap][1]

            elif self.ap_maps[ap][0] == "RobotAt":
                evaluated_APs[ap] = state.robot_at == self.ap_maps[ap][1]

        return evaluated_APs

    def __str__(self):
        return 'AbstractFourRoomMDP: InitState: {}, GoalState: {}'.format(self.init_state, self.constraints['goal'])

#    @classmethod
#    def action_for_room_number(self, room_number):
#        for action in CubeL1MDP.ACTIONS:
#            if str(room_number) in action:
#                return action
#        raise ValueError('unable to find action corresponding to room {}'.format(room_number))

# -----------------------------------
# Debug functions
# -----------------------------------

"""
def debug_l1_grid_world():
    def get_l1_policy(start_room=None, goal_room=None, mdp=None):
        if mdp is None:
            mdp = CubeL1MDP(start_room, goal_room)
        vi = ValueIteration(mdp)
        vi.run_vi()

        policy = defaultdict()
        action_seq, state_seq = vi.plan(mdp.init_state)

        print('Plan for {}:'.format(mdp))
        for i in range(len(action_seq)):
            print("\tpi[{}] -> {}".format(state_seq[i], action_seq[i]))
            policy[state_seq[i]] = action_seq[i]
        return policy
    policy = get_l1_policy(1, 4)
"""
if __name__ == '__main__':
    mdp = CleanupL1MDP(env_file=[build_cube_env()])
    print('done')