from simple_rl.AMDPs.AbstractGridWorld.GridWorldAMDPClass import FourRoomL1GroundedAction
from simple_rl.AMDPs.AMDPPolicyGeneratorClass import AMDPPolicyGenerator
from simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from simple_rl.AMDPs.AbstractGridWorld.GridWorldAMDPClass import FourRoomL1MDP, FourRoomL1State

import re

class L1PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0MDP, verbose=False):
        self.domain = l0MDP
        self.verbose = verbose

    def generatePolicy(self, l1_state, grounded_action_str):
        grounded_action = FourRoomL1GroundedAction(grounded_action_str)
        mdp = FourRoomL1MDP(l1_state.agent_in_room_number, grounded_action.goal_state.agent_in_room_number)

        return self.getPolicy(mdp)

    # TODO: Figure out a way to project l0 state to l1 w/o having access to l0Domain in l1
    def generateAbstractState(self, l0_state):
        l0_location = (l0_state.x, l0_state.y)
        room = self.domain.get_room_numbers(l0_location)[0]
        return FourRoomL1State(room)

class L0PolicyGenerator(AMDPPolicyGenerator):
    def __init__(self, l0Domain, verbose=False):
        self.domain = l0Domain
        self.verbose = verbose

    def generatePolicy(self, state, grounded_task):
        # TODO: There should be a better way to get the destination_room from the grounded_task (create GT class)
        destination_room = int(re.findall(r'\d+', grounded_task)[0])
        destination_location = self.domain._get_single_location_for_room(destination_room)
        init_location = (state.x, state.y)
        mdp = FourRoomMDP(self.domain.width, self.domain.height, init_loc=init_location, goal_locs=[destination_location])
        return self.getPolicy(mdp)

if __name__ == '__main__':
    start_location = (1, 1)
    goal_location = (5, 5)

    l0Domain = FourRoomMDP(width=5, height=5, init_loc=start_location, goal_locs=[goal_location])

    start_room = l0Domain.get_room_numbers(start_location)[0]
    goal_room = l0Domain.get_room_numbers(goal_location)[0]

    l1Domain = FourRoomL1MDP(start_room, goal_room, l0Domain.gamma)

    l0State = l0Domain.init_state
    l1room = l0Domain.get_room_numbers((l0State.x, l0State.y))[0]
    l1State = FourRoomL1State(l1room)

    pg = L1PolicyGenerator(l0Domain)
