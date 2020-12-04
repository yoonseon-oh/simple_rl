''' FourRoomMDPClass.py: Contains the FourRoom class. '''

# Python imports.
import math
import os
from collections import defaultdict
import numpy as np

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.apmdp.AP_MDP.RoomupQStateClass import RoomupQState

from simple_rl.apmdp.AP_MDP.CubeStateClass import CubeState
from simple_rl.apmdp.settings.build_cleanup_env_1 import build_cube_env

from sympy import *

class CleanupQMDP(MDP):
    ''' Class for a Cube World with Rooms '''

    ACTIONS = ["north","south","east","west","pickup","place"]
    def __init__(self, len_x=9, len_y=9, len_z=5, init_loc=(1,1,-1),
                 goal_locs=[(9,9,3)], env_file = [],
                 gamma=0.99, slip_prob=0.00, name="cleanup",
                 is_goal_terminal=True, #rand_init=False,
                 step_cost=0.0, constraints={'goal':[],'stay':[]}, ap_maps = {}):
        '''
        Args:
            len_x, len_y, len_z (int)
            init_loc (tuple: (int, int,int))
            goal_locs (list of tuples: [(int, int,int)...]
            env_file: specify environment)
            constraints: logic formula of 'goal' and 'stay' for the reward function
                        - goal (large positive), stay (zero), otherwise (large negative)
            ap_maps: dictionary {ap_symbol: (category, state), ...} ex) {a: ('r', [1]), b:('a',west)}
                    category: floor(f), room(r), lowest level action(a), grid cells (c)
        '''

        # Load environment file

        if len(env_file)==0:
            print('Fail to initialize RoomCubeMDP')

        else:
            cube_env = env_file[0]
            len_x = cube_env['len_x']
            len_y = cube_env['len_y']
            walls = cube_env['walls']
            self.num_room = cube_env['num_room']
            self.room_to_locs = cube_env['room_to_locs']
            self.obj_to_locs = cube_env['obj_to_locs']
            self.num_obj = cube_env['num_obj']
            self.obj_color = cube_env['color']

        init_state = CleanupQState(init_loc)

        MDP.__init__(self, CleanupQMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state, gamma=gamma)

                     #len_x, len_y, len_z, init_loc, goal_locs=goal_locs, walls=walls, gamma=gamma, slip_prob=slip_prob, name=name,
                         #is_goal_terminal=is_goal_terminal, rand_init=rand_init, step_cost=step_cost)

        if 'lowest' in constraints.keys():
            self.constraints = {'goal': 'a', 'stay': 'b'}
            self.ap_maps = {'a': ap_maps['a'], 'b': [1, 'state', self.get_room_numbers(init_loc)[0]]}  # AP --> real world
        else:
            self.constraints = constraints  # constraints for LTL
            self.ap_maps = ap_maps

        init_state = RoomCubeState(init_loc[0], init_loc[1], init_loc[2], self._transition_q(init_loc, ""))
        if init_state.q != 0:
            init_state.set_terminal(True)

        MDP.__init__(self, RoomCubeMDP.ACTIONS, self._transition_func, self._reward_func, init_state=init_state,
                     gamma=gamma)



    def _transition_func(self, state, action):
        next_state_xyz = super()._transition_func(state, action)

        next_q = self._transition_q((next_state_xyz.x, next_state_xyz.y, next_state_xyz.z), action)

        next_state = RoomCubeState(next_state_xyz.x, next_state_xyz.y, next_state_xyz.z, next_q)

        if next_q != 0:
            next_state.set_terminal(True)
            #next_state._is_terminal = (next_q == 1)

        return next_state


    def is_loc_in_room(self, loc, room_number):
        return loc in self.room_to_locs[room_number]


    def is_loc_on_floor(self, loc, floor_number):
        return loc in self.floor_to_locs[floor_number]

    def get_room_numbers(self, loc):
        room_numbers = []
        for i in range(1, self.num_room+1):
            if loc in self.room_to_locs[i]:
                room_numbers.append(i)
        return room_numbers

    def get_floor_numbers(self, loc):
        room_number = self.get_room_numbers(loc)[0]
        floor_numbers = []
        for i in range(1, self.num_floor+1):
            if room_number in self.floor_to_rooms[i]:
                floor_numbers.append(i)
        return floor_numbers

    def _reward_func(self, state, action): # TODO: Complete
        next_state = self._transition_func(state, action)
        #next_state = state
        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100

        return reward

    def _transition_q(self, loc, action):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(loc, action)

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

    def _evaluate_APs(self, loc, action): # TODO: Complete
        evaluated_APs ={}

        for ap in self.ap_maps.keys():
            if (self.ap_maps[ap][0] == 0) and (self.ap_maps[ap][1] == 'state'): # level 0
                evaluated_APs[ap] = (loc[0] == self.ap_maps[ap][2][0]) & (loc[1] == self.ap_maps[ap][2][1]) & (loc[2] == self.ap_maps[ap][2][2])

            elif (self.ap_maps[ap][0] == 0 ) and (self.ap_maps[ap][1] == 'action'):
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

            elif self.ap_maps[ap][0] == 1 and (self.ap_maps[ap][1] == 'state'):  # level 1
                evaluated_APs[ap] = self.is_loc_in_room(loc, self.ap_maps[ap][2])

            elif self.ap_maps[ap][0] == 1 and (self.ap_maps[ap][1] == 'action'):  # level 1
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

            elif self.ap_maps[ap][0] == 2 and (self.ap_maps[ap][1] == 'state'):  # level 2
                evaluated_APs[ap] = self.is_loc_on_floor(loc, self.ap_maps[ap][2])

            elif self.ap_maps[ap][0] == 2 and (self.ap_maps[ap][1] == 'action'):  # level 2
                evaluated_APs[ap] = self.ap_maps[ap][2] in action

        return evaluated_APs


if __name__ == '__main__':
    cube_env1 = build_cube_env()
    mdp = RoomCubeMDP(env_file=[cube_env1])
