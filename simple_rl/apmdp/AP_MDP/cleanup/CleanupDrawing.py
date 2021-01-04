# Packages to draw the result
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

def draw_env(env):
    img = np.zeros((env['len_y'],env['len_x'],3))
    # draw rooms
    color_set = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0.4] ]
    for x in range(0, env['len_x']):
        for y in range(0, env['len_y']):
            if env['map'][x][y] == "w":
                color= [0,0,0]
            else:
                color = color_set[env['map'][x][y]]
            img[y][x][0] = color[0]
            img[y][x][1] = color[1]
            img[y][x][2] = color[2]

    plt.imshow(img)

    # draw objects
    for ii in range(0, env['num_obj']):
        plt.plot(env['obj_to_locs'][ii][0],env['obj_to_locs'][ii][1],'^',mec='black',
                 mfc= env['obj_color'][ii], ms=8.0)

    plt.draw()

def draw_state_seq(env,state_seq, save_fig = ""):
    # state_seq: sequence of CleanupQstates
    img = np.zeros((env['len_y'], env['len_x'], 3))
    # draw rooms
    color_set = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 0.4]]
    for x in range(0, env['len_x']):
        for y in range(0, env['len_y']):
            if env['map'][x][y] == "w":
                color = [0, 0, 0]
            else:
                color = color_set[env['map'][x][y]]
            img[y][x][0] = color[0]
            img[y][x][1] = color[1]
            img[y][x][2] = color[2]

    plt.imshow(img)

    # Extract trajectories
    robot_seq = {'x': [], 'y': []}
    obj_seq = [{'x':[], 'y':[]} for i in range(0, env['num_obj'])]
    for tt in range(0, len(state_seq)):
        robot_seq['x'].append(state_seq[tt].x)
        robot_seq['y'].append(state_seq[tt].y)

        for ii in range(0,env['num_obj']):
            obj_seq[ii]['x'].append(state_seq[tt].obj_loc[ii][0])
            obj_seq[ii]['y'].append(state_seq[tt].obj_loc[ii][1])

    # Draw trajectories
    for ii in range(0, env['num_obj']):
        plt.plot(obj_seq[ii]['x'], obj_seq[ii]['y'], lw = 2, c=env['obj_color'][ii])
        plt.plot(obj_seq[ii]['x'][0], obj_seq[ii]['y'][0], '^',mec='black',
                 mfc= env['obj_color'][ii], ms=8.0)
        plt.plot(obj_seq[ii]['x'][-1], obj_seq[ii]['y'][-1], '^', mec='black',
                 mfc=env['obj_color'][ii], ms=8.0)
    print(env['room_color'],env['obj_color'])

    plt.plot(robot_seq['x'], robot_seq['y'], lw=1, c='black')

    plt.draw()

    # save figure
    if save_fig!="":
        plt.savefig(save_fig)


if __name__ == '__main__':
    plt.pause(10)

    print("done")