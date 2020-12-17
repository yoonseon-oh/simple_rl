''' The lowest level of AP-MDP for clean-up domain '''

# Python imports.
import math
import os
from collections import defaultdict
import numpy as np
import copy

# Other imports
from simple_rl.mdp.MDPClass import MDP
from simple_rl.apmdp.AP_MDP.cleanup.CleanupQStateClass import CleanupQState

from simple_rl.apmdp.AP_MDP.CubeStateClass import CubeState
from simple_rl.apmdp.settings.build_cleanup_env_1 import build_cube_env, draw_cleanup_env
from simple_rl.planning import ValueIteration
import random
import matplotlib.pyplot as plt

from sympy import *

class CleanupQMDP(MDP):
    ''' Class for a cleanup domain - lowest level of AP-MDP'''

    ACTIONS = ["north","south","east","west","pickup","place"]
    def __init__(self, len_x=9, len_y=9, init_robot=(1,1,-1), q_init=0, env_file = [],
                 gamma=0.99, name="cleanup",  slip_prob=0.0,
                 is_goal_terminal=True, step_cost=0.0, constraints={'goal':[],'stay':[]}, ap_maps = {}):
        '''
        Args:
            len_x, len_y, len_z (int)
            init_robot (tuple: (int, int,int)): the state of a robot
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
            self.obj_loc_init = cube_env['obj_to_locs']
            self.obj_loc = cube_env['obj_to_locs']
            self.num_obj = cube_env['num_obj']
            self.obj_color = cube_env['obj_color']
            self.slip_prob = slip_prob

        self.actions = ["north", "south", "east", "west", "pickup", "place"]
        # initialize constraints
        if 'lowest' in constraints.keys():  # Todo
            high_state = self.get_highlevel_pose(init_robot,self.obj_loc)
            if 'RobotIn' == ap_maps['a'][0]:  # goal condition is about navigation
                self.constraints = {'goal': 'a', 'stay': 'b'}
                self.ap_maps = {'a': ap_maps['a'],
                                'b': ('RobotIn', high_state['room'])}  # AP --> real world
                self.actions = ['north','south','east','west']

            elif 'RobotAt' == ap_maps['a'][0]:  # goal condition is about navigation
                obj_room = self.xy_to_room(self.obj_loc_init[ap_maps['a'][1]])
                self.constraints = {'goal': 'a', 'stay': 'b | c'}
                self.ap_maps = {'a': ap_maps['a'],
                                'b': ('RobotIn', high_state['room']), 'c': ('RobotIn', obj_room)}  # AP --> real world


                if high_state[0] == 'room':
                    stay_condition = ('RobotIn', high_state[1])
                elif high_state[0] == 'object':
                    stay_condition = ('RobotAt', high_state[1])
                else:
                    print("error")
                    stay_condition = 'None'

                self.constraints = {'goal': 'a', 'stay': 'b'}
                self.ap_maps = {'a': ap_maps['a'],
                                'b': stay_condition}  # AP --> real world
                self.actions = ['north','south','west','east']

            elif ap_maps['a'][0] == 'On':
                self.constraints = {'goal': 'a', 'stay': '~a & b'}
                if ap_maps['a'][1] != -1:
                    self.ap_maps = {'a': ap_maps['a'], 'b': ('RobotAt',ap_maps['a'][1])}
                    self.actions = ['pickup']
                else:
                    self.ap_maps = {'a': ap_maps['a'], 'b': ('RobotAt', init_robot[2])}
                    self.actions = ['place']
        else:
            self.constraints = constraints  # constraints for LTL
            self.ap_maps = ap_maps

        print(self.ap_maps, self.constraints)

        # initialize the state
        init_state = CleanupQState(init_robot, q=-1, obj_loc=self.obj_loc_init)
        init_state.q = self._transition_q(init_state)

        if init_state.q != 0:
            init_state.set_terminal(True)

        self.init_state = init_state
        # initialize MDP
        MDP.__init__(self, self.actions, self._transition_func, self._reward_func, init_state=init_state,
                     gamma=gamma)

        '''
        if 'lowest' in constraints.keys():
            self.constraints = {'goal': 'a', 'stay': 'b'}
            self.ap_maps = {'a': ap_maps['a'], 'b': [1, 'state', self.get_room_numbers(init_loc)[0]]}  # AP --> real world
        else:
            self.constraints = constraints  # constraints for LTL
            self.ap_maps = ap_maps
        '''

    def _transition_func(self, state, action):
        next_state = self._transition_env(state, action) # compute the transition of the environment
        next_state.set_terminal(is_term=False)
        next_state.q = self._transition_q(next_state, action)

        if next_state.q != 0:
            next_state.set_terminal(True)
            #next_state._is_terminal = (next_q == 1)
        print(state, next_state)
        next_state.update_data()

        return next_state

    def _transition_env(self, state, action):

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
            try:
                obj_id = state.obj_loc.index((state.x, state.y))
                next_state.obj_id = obj_id
            except ValueError:
                pass

        elif action == "place":
            if state.obj_id !=- 1:
                next_state.obj_id = -1
                next_state.obj_loc[state.obj_id]=(next_state.x, next_state.y)

        # if an object is on the robot, the robot and the object should be at the same location.
        if next_state.obj_id !=-1:
            next_state.obj_loc[next_state.obj_id] = (next_state.x, next_state.y)

        next_state.update_data()
        #print(state, action, next_state)
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
        #next_state = state
        if next_state.q == 0: # stay
            reward = -1
        elif next_state.q == 1:  # success
            reward = 100
        elif next_state.q == -1:  # fail
            reward = -100

        return reward

    def _transition_q(self, state):
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

    def _evaluate_APs(self, state): # TODO: Complete
        evaluated_APs ={}

        #ap_maps: ex) {a: ('in', [obj_id, room_id]), b: ('on', obj_id), c: ('RobotIn', room_id),, d: ('RobotAt', obj_id)}
        for ap in self.ap_maps.keys():
            if self.ap_maps[ap][0] == "In": # an object is in a room
               evaluated_APs[ap] = state.obj_loc[self.ap_maps[ap][1][0]] in self.room_to_locs[self.ap_maps[ap][1][1]]
            elif self.ap_maps[ap][0] == "On":
               evaluated_APs[ap] = state.obj_id == self.ap_maps[ap][1]

            elif self.ap_maps[ap][0] == "RobotIn": # a robot is in a room
               evaluated_APs[ap] = (state.x, state.y) in self.room_to_locs[self.ap_maps[ap][1]]

            elif self.ap_maps[ap][0] == "RobotAt":
                evaluated_APs[ap] = (state.x, state.y) == state.obj_loc[self.ap_maps[ap][1]]

        return evaluated_APs

    def get_highlevel_pose(self,robot_init,obj_loc):
        high_pose = {'room': -1, 'object':-1}
        for ii in range(0,len(obj_loc)):
            if ii != robot_init[2] and (robot_init[0], robot_init[1]) == obj_loc[ii]:
                high_pose['object']= ii

        high_pose ['room']=self.xy_to_room((robot_init[0], robot_init[1]))

        return high_pose

    def xy_to_room(self, xy):
        for ii in range(0, self.num_room):
            if xy in self.room_to_locs[ii]:
                return ii
        return -1




if __name__ == '__main__':
    constraints = {'goal': 'a', 'stay': 'b','lowest':[]}
    ap_maps = {'b': ('On', -1), 'a': ('In', (1,1))}
    init_robot = (1,3,1)
    #ex) ap_maps = {'a': ('In', [1, 1]), 'b':('On',1), 'c':('RobotIn',1), 'd':('RobotAt',1),'e':('RobotIn',0)}

    env = build_cube_env()
    mdp = CleanupQMDP(env_file=[env], constraints=constraints,ap_maps=ap_maps,init_robot = init_robot)
    value_iter = ValueIteration(mdp, sample_rate=3, max_iterations=20)
    value_iter.run_vi()


    # run value interation
    print("plan start")
    action_seq, state_seq = value_iter.plan(mdp.get_init_state())


    # draw the result
    for tt in range(0,len(state_seq)):
        draw_cleanup_env(env)
        print(state_seq[tt])
        plt.plot(state_seq[tt].x, state_seq[tt].y, mec='black',marker='o', mfc = 'r')
        plt.draw()
        plt.pause(1)

    print('end')


