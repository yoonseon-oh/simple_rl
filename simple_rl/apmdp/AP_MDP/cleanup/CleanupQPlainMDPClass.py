''' The lowest level of AP-MDP for clean-up domain '''

# Python imports.
import math
import os
from collections import defaultdict
import numpy as np
from datetime import datetime
import time
import copy
import glob

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.apmdp.LTLautomataClass import LTLautomata
from simple_rl.apmdp.AP_MDP.cleanup.CleanupQStateClass import CleanupQState
from simple_rl.apmdp.AP_MDP.cleanup.LtlAMDP4CleanupClass import ltl2str

from simple_rl.apmdp.AP_MDP.CubeStateClass import CubeState
from simple_rl.apmdp.settings.build_cleanup_env_1 import build_cube_env, draw_cleanup_env
from simple_rl.apmdp.AP_MDP.cleanup.CleanupDrawing import *
from simple_rl.planning import ValueIteration
import random
import matplotlib.pyplot as plt

from sympy import *

class CleanupQPlainMDP(MDP):
    ''' Class for a cleanup domain - lowest level of AP-MDP'''

    ACTIONS = ["north","south","east","west","pickup","place"]
    def __init__(self, init_state=(1,1,-1,0), env_file = [], ltlformula="",
                 gamma=0.99, name="cleanup",  slip_prob=0.0,
                 is_goal_terminal=True, step_cost=0.0, constraints={'goal':[],'stay':[]}, ap_maps = {}):
        '''
        Args:
            len_x, len_y, len_z (int)
            init_state: CleanupQState
            env_file: specify environment)
            constraints: logic formula of 'goal' and 'stay' for the reward function
                        - goal (large positive), stay (zero), otherwise (large negative)
            ap_maps: dictionary {ap_symbol: (predicate, argument), ...}
                ex) {a: ('In', [obj_id, room_id]), b:('On',obj_id), c:('RobotIn',room_id),, d:('RobotAt',obj_id)}
        '''

        # Load environment file

        if len(env_file)==0:
            print('Fail to initialize RoomCubeMDP')

        else:
            cube_env = env_file[0]
            self.len_x = cube_env['len_x']
            self.len_y = cube_env['len_y']
            self.walls = cube_env['walls']
            self.num_room = cube_env['num_room']
            self.room_to_locs = cube_env['room_to_locs']
            self.obj_loc_init = init_state.obj_loc
            self.obj_loc = init_state.obj_loc
            self.num_obj = cube_env['num_obj']
            self.obj_color = cube_env['obj_color']
            self.slip_prob = slip_prob
            self.transition_table = cube_env['transition_table']
            self.notblocked = cube_env['notblock']

        self.actions = ["north", "south", "east", "west", "pickup", "place"]
        #TODO:Avoid object when moving around

        # constraints
        self.automata = LTLautomata(ltlformula) # Translate LTL into the automata
        self.constraints = constraints
        self.ap_maps = {}
        for ap in self.automata.APs:
            self.ap_maps[ap] = ap_maps[ap]

        # extract related objects
        obj_list = {}
        for ap, val in self.ap_maps.items():
            if val[0] =="In":
                obj_list[ap] = val[1][0]
            elif val[0] !='RobotIn':
                obj_list[ap] = val[1]

        self.related_obj={}
        for key,val in self.automata.trans_dict.items():
            related_objlist = []
            for ap in obj_list.keys():
                if ap in "".join(val.keys()):
                    related_objlist.append(obj_list[ap])
            self.related_obj[key] = related_objlist

        # initialize
        self.init_state = init_state
        init_state.q = self.automata.init_state
        if self.automata.aut_spot.state_is_accepting(init_state.q):
            init_state.set_terminal(True)

        # initialize MDP
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=init_state,
                     gamma=gamma)

        print("Plain MDP: ", self.ap_maps, self.constraints, self.init_state)
        '''
        if 'lowest' in constraints.keys():
            self.constraints = {'goal': 'a', 'stay': 'b'}
            self.ap_maps = {'a': ap_maps['a'], 'b': [1, 'state', self.get_room_numbers(init_loc)[0]]}  # AP --> real world
        else:
            self.constraints = constraints  # constraints for LTL
            self.ap_maps = ap_maps
        '''


    def _transition_func(self, state, action):

        next_state = copy.deepcopy(state)

        r = random.random() # probability of transition error
        if self.slip_prob > r:
            # Flip dir.
            if action == "north":
                action = random.choice(["west", "east"])
            elif action == "south":
                action = random.choice(["west", "east"])
            elif action == "west":
                action = random.choice(["north", "south"])
            elif action == "east":
                action = random.choice(["north", "south"])
            elif action == "pickup":
                action = -1
            elif action == "place":
                action = -1

        # Compute action
        if action == "north" and state.y < self.len_y and not self.is_wall(state.x, state.y + 1):
            next_state.y = state.y + 1
        elif action == "south" and state.y > 1 and not self.is_wall(state.x, state.y - 1):
            next_state.y = state.y - 1
        elif action == "east" and state.x < self.len_x and not self.is_wall(state.x + 1, state.y):
            next_state.x = state.x + 1
        elif action == "west" and state.x > 1 and not self.is_wall(state.x - 1, state.y):
            next_state.x = state.x - 1
        elif action == "pickup":
            if state.obj_id == -1:
                try:
                    obj_id = state.obj_loc.index((state.x, state.y))
                    next_state.obj_id = obj_id
                except ValueError:
                    pass

        elif action == "place":
            if state.obj_id !=- 1:
                obj_list = state.obj_loc[0:state.obj_id]+state.obj_loc[(state.obj_id+1)::]
                if (next_state.x, next_state.y) not in self.notblocked:
                    if (next_state.x, next_state.y) not in obj_list:
                        next_state.obj_id = -1
                        next_state.obj_loc[state.obj_id]=(next_state.x, next_state.y)


        # if an object is on the robot, the robot and the object should be at the same location.
        if next_state.obj_id !=-1:
            next_state.obj_loc[next_state.obj_id] = (next_state.x, next_state.y)

        high_pose = self.get_highlevel_pose(robot_init=(next_state.x, next_state.y,next_state.obj_id),obj_loc=next_state.obj_loc)
        next_state.q = self._transition_q(next_state)

        # TODO: Wrong, if On() or In(), the robot should be at ().
        # if q state does not change, though the robot at the object, the robot does not move to the object
        if high_pose['object'] not in [-1]+ self.related_obj[state.q]:
            return state
        else:
            next_state.set_terminal(is_term=False)
            if self.automata.aut_spot.state_is_accepting(next_state.q):
                next_state.set_terminal(True)
            next_state.update_data()

        return next_state

    def is_loc_in_room(self, loc, room_number):
        return loc in self.room_to_locs[room_number]

    def is_wall(self,x,y):
        return (x,y) in self.walls

    def get_room_numbers(self, loc):
        room_numbers = []
        for i in range(1, self.num_room+1):
            if loc in self.room_to_locs[i]:
                room_numbers.append(i)
        return room_numbers

    def is_goal_state(self, state):
        return state.q == 1

    def _reward_func(self, state, action): # TODO: Complete
        next_state = self._transition_func(state, action)
        return self.automata.reward_func(next_state.q)

    def _transition_q(self, state):
        # evaluate APs
        evaluated_APs = self._evaluate_APs(state)
        next_q = self.automata.transition_func(state.q, evaluated_APs)

        return next_q

    def _evaluate_APs(self, state):
        evaluated_APs ={}

        #ap_maps: ex) {a: ('in', [obj_id, room_id]), b: ('on', obj_id), c: ('RobotIn', room_id),, d: ('RobotAt', obj_id)}
        for ap in self.ap_maps.keys():
            if self.ap_maps[ap][0] == "In": # an object is in a room
               evaluated_APs[ap] = (state.obj_loc[self.ap_maps[ap][1][0]] in self.room_to_locs[self.ap_maps[ap][1][1]]) \
                                   and (state.obj_id == -1)
            elif self.ap_maps[ap][0] == "On":
               evaluated_APs[ap] = state.obj_id == self.ap_maps[ap][1]

            elif self.ap_maps[ap][0] == "RobotIn": # a robot is in a room
               evaluated_APs[ap] = (state.x, state.y) in self.room_to_locs[self.ap_maps[ap][1]]

            elif self.ap_maps[ap][0] == "RobotAt":
                evaluated_APs[ap] = (state.x, state.y) == state.obj_loc[self.ap_maps[ap][1]] and (state.obj_id != self.ap_maps[ap][1])

        return evaluated_APs

    def get_highlevel_pose(self,robot_init,obj_loc):
        high_pose = {'room': -1, 'object':-1}
        for ii in range(0,len(obj_loc)):
            if ii != robot_init[2] and (robot_init[0], robot_init[1]) == obj_loc[ii]:
                high_pose['object']= ii

        high_pose ['room']=self.xy_to_room((robot_init[0], robot_init[1]))

        return high_pose
    def canGo_object(self, xy, obj_id,obj_loc): #return True if a robot can go considering object distribution
        for ii in self.object_avoid:
            if ii != obj_id and xy == obj_loc[ii]:
                return False
        return True


    def xy_to_room(self, xy):
        for ii in range(0, self.num_room):
            if xy in self.room_to_locs[ii]:
                return ii
        return -1


if __name__ == '__main__':
    # Settings
    init_loc = (4, 5, -1)
    ltl_formula = 'F b  '  # ex) 'F(a & F( b & Fc))', 'F a', '~a U b'
    ap_maps = {'a': ('On', 1), 'b': ('In', (2, 4)), 'c': ('In', (1, 4))}
    save_dir = '/media/ys/SharedData/Research/AP-MDP-result/PlainMDP'

    # initialize
    start_time = time.time()
    cube_env = build_cube_env()
    cube_env['num_obj'] = 4
    cube_env['obj_to_locs'] =cube_env['obj_to_locs'][0:cube_env['num_obj']]
    # figure name
    now = datetime.now()
    datestr = now.strftime('%Y%m%d-%H-%M')
    listfig = glob.glob('{}/{}*.png'.format(save_dir, datestr))
    save_name = '{}-{}.png'.format(datestr, len(listfig))

    # Solve
    init_state = CleanupQState(init_state=init_loc, q=-1, obj_loc=cube_env['obj_to_locs'])
    mdp = CleanupQPlainMDP(ltlformula = ltl_formula, init_state=init_state, env_file=[cube_env], ap_maps=ap_maps)
    value_iter = ValueIteration(mdp, sample_rate=1, max_iterations=50)
    value_iter.run_vi()
    num_backup = value_iter.get_num_backups_in_recent_run()

    # Value Iteration.
    a_seq, s_seq = value_iter.plan(mdp.get_init_state())

    computing_time = time.time() - start_time

    draw_state_seq(cube_env, s_seq, save_fig='{}/{}'.format(save_dir, save_name), title=ltl2str(ltl_formula, ap_maps))
    plt.pause(5)
    for t in range(0, len(a_seq)):
        print(s_seq[t], a_seq[t])
    print(s_seq[-1])

    print("Summary")
    print("\t Time: {} seconds, the number of actions: {}, backup: {}"
          .format(round(computing_time, 3), len(a_seq), num_backup))




