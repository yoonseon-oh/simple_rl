import numpy as np
from collections import defaultdict

def build_cube_env():
    cube_env = {} # Define settings as a dictionary
    cube_env['len_x'] = 6  # the number of grids (x-axis)
    cube_env['len_y'] = 4  # the number of grids (y-axis)
    cube_env['len_z'] = 3  # the number of grids (z-axis)
    cube_env['num_floor']= 3 # the number of floors
    cube_env['num_room'] = 18 # the number of rooms

    # Define a map : room number, w (wall)
    map = [] # map[z][y][x]
    map.append([[int(np.ceil(ii / 2)) for ii in range(1, 7)],
                [int(np.ceil(ii / 2)) for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+3 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+3 for ii in range(1, 7)]])  # the first floor
    map.append([[int(np.ceil(ii / 2))+6 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+6 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+9 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+9 for ii in range(1, 7)]])  # the second floor
    map.append([[int(np.ceil(ii / 2))+12 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+12 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+15 for ii in range(1, 7)],
                [int(np.ceil(ii / 2))+15 for ii in range(1, 7)]])  # the third floor

    cube_env['map'] = map

    # extract (x,y,z) in each room
    room_to_locs = defaultdict()
    for r in range(1,cube_env['num_room']+1):
        locs = []
        for x in range(1, cube_env['len_x']+1):
            for y in range(1,cube_env['len_y']+1):
                for z in range(1,cube_env['len_z']+1):
                    if cube_env['map'][z-1][y-1][x-1] == r:
                        locs.append((x,y,z))
        room_to_locs[r] = locs

    cube_env['room_to_locs'] = room_to_locs

    # extract (x,y,z) ᅟin walls
    walls = []
    for x in range(1, cube_env['len_x'] + 1):
        for y in range(1, cube_env['len_y'] + 1):
            for z in range(1, cube_env['len_z'] + 1):
                if cube_env['map'][z - 1][y - 1][x - 1] == 'w':
                    walls.append((x, y, z))

    cube_env['walls'] = walls

    # Extract room numbers and locations in each floor
    floor_to_room = defaultdict()
    floor_to_locs = defaultdict()
    for f in range(0, cube_env['num_floor']):
        rooms = []
        locs = []
        for x in range(1, cube_env['len_x'] + 1):
            for y in range(1, cube_env['len_y'] + 1):
                if cube_env['map'][f][y-1][x-1] not in rooms:
                    rooms.append(cube_env['map'][f][y-1][x-1])
                locs.append((x, y, f+1))

        floor_to_room[f+1] = rooms
        floor_to_locs[f+1] = locs

    cube_env['floor_to_rooms'] = floor_to_room
    cube_env['floor_to_locs'] = floor_to_locs

    # Define transition table (connectivity between rooms)
    cube_env['transition_table'] = {}

    for r in range(1,cube_env['num_room']+1):
        connected_rooms = []
        for x,y,z in cube_env['room_to_locs'][r]:
            near = [(max(x-2,0), y-1, z-1), (min(x, cube_env['len_x']-1), y-1, z-1),
                    (x-1, max(y-2,0), z-1), (x-1, min(y, cube_env['len_y']-1), z-1)]
            for i,j,k in near:
                next_r = cube_env['map'][k][j][i]
                if next_r not in connected_rooms and next_r != r:
                    connected_rooms.append(next_r)
        if r==6:
            connected_rooms.append(12)
        elif r==12:
            connected_rooms.append(6)
            connected_rooms.append(18)
        elif r==18:
            connected_rooms.append(12)

        cube_env['transition_table'][r] = connected_rooms

    # Define attributes
    cube_env['attribute_color'] = {1: 'red', 6: 'blue', 12: 'blue', 18: 'blue',
                                   8: 'yellow', 10: 'purple', 15: 'green'
                                   }

    # Define Actions
    cube_env['L2ACTIONS'] = ["toFloor%d" %ii for ii in range(1, cube_env['num_floor']+1)]
    cube_env['L1ACTIONS'] = ["toRoom%d" % ii for ii in range(1, cube_env['num_room'] + 1)]
    cube_env['L0ACTIONS'] = ["north", "south", "east", "west", "up", "down"]

    # save
    #np.save('cube_env_1.npy',cube_env)

    return cube_env

if __name__ == '__main__':
    env = build_cube_env()
    print("done")