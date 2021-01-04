import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import itertools

def build_cube_env():
    cube_env = {} # Define settings as a dictionary
    len_x, cube_env['len_x'] = 8, 8   # the number of grids (x-axis)
    len_y, cube_env['len_y'] = 11, 11  # the number of grids (y-axis)
    num_room, cube_env['num_room'] = 4, 4 # the number of rooms
    #room_len, cube_env['room_len'] = 2, 2

    # Define a map : room number, w (wall), 0: door
    map = [] # map[z][y][x]
    map = [[0]*len_y, [0]*len_y, [0]*len_y]
    map.append(['w',0,'w','w','w','w','w','w','w',0,'w'])
    map.extend([[1, 1, 1,'w',2 ,2 ,2 ,'w',3 ,3 ,3], [1 ,1 ,1 ,1 ,2 ,2 ,2 ,'w',3 ,3 ,3],
                [1,1,1, 'w', 2,2,2,2, 3, 3, 3], [1, 1, 1, 'w', 2, 2, 2, 'w', 3, 3, 3]])

    cube_env['map'] = map

    # extract (x,y) in each room
    room_to_locs = defaultdict()
    loc_to_room = {}
    for r in range(0,cube_env['num_room']+1):
        locs = []
        for x in range(0, cube_env['len_x']):
            for y in range(0,cube_env['len_y']):
                    if cube_env['map'][x][y] == r:
                        locs.append((x,y))
                        loc_to_room[(x,y)] = r

        room_to_locs[r] = locs

    cube_env['room_to_locs'] = room_to_locs
    cube_env['loc_to_room'] = loc_to_room

    # extract (x,y) ᅟin walls
    walls = []
    for x in range(0, cube_env['len_x']):
        for y in range(0, cube_env['len_y']):
                if cube_env['map'][x][y] == 'w':
                    walls.append((x, y))

    cube_env['walls'] = walls

    # Define transition table (connectivity between rooms)
    cube_env['transition_table'] = {0:[1, 3], 1: [0, 2], 2:[1,3], 3:[0, 2]}
    cube_env['maxnum_adjroom'] = 2
    # A robot can go to the object's location if it is in the current room

    # Define attributes
    cube_env['room_color'] = {0: 'red', 1: 'blue', 2: 'green', 3:'yellow'}

    # Define objects
    cube_env['obj_to_locs']=[(1,3),(1,5) ]#,(5,6),(6,8)]
    cube_env['obj_color'] = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow'}
    cube_env['num_obj'] = len(cube_env['obj_to_locs'])



    #cube_env['L2ACTIONS'] = ["Activate_" + str(ii) for ii in range(0, cube_env['num_obj'])]\
    #                        + ["NavRoom2_"+ str(ii) for ii in range(0, cube_env['num_room'])] + ["Deactivate"]
    # Define Actions
    cube_env['L2ACTIONS'] = ["MoveObj_"+ str(x[0])+"_"+ str(x[1]) for x in
                             itertools.product(list(range(0,cube_env['num_obj'])),
                                               list(range(0,cube_env['num_room'])))] + ["Deactivate"]
    cube_env['L1ACTIONS'] = ["NavRoom_"+ str(ii) for ii in range(0, cube_env['num_room'])] \
                            +["NavObj_"+ str(ii) for ii in range(0, cube_env['num_obj'])] \
                            +["PICKUP_" + str(ii) for ii in range(0, cube_env['num_obj'])] \
                            + ["PLACE"]


# cube_env['L1ACTIONS'] = ["NavRoomAdj_" + str(ii) for ii in range(0, cube_env['maxnum_adjroom'])] \

    cube_env['L0ACTIONS'] = ["north", "south", "east", "west", "pickup", "place"]

    # save
    #np.save('cube_env_1.npy',cube_env)

    return cube_env

def draw_cleanup_env(env):
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


if __name__ == '__main__':
    env = build_cube_env()
    draw_cleanup_env(env)
    plt.pause(10)

    print("done")