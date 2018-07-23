from simple_rl.tasks.four_room.FourRoomMDPClass import FourRoomMDP
from simple_rl.amdp.AbstractGridWorld.AbstractGridWorldMDPClass import FourRoomL1MDP, FourRoomL1GroundedAction, FourRoomRootGroundedAction
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask
from simple_rl.amdp.AbstractGridWorld.AbstractGridWorldPolicyGenerators import L0PolicyGenerator, L1PolicyGenerator
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AbstractGridWorld.AbstractGridWorldStateMapper import AbstractGridWorldL1StateMapper

if __name__ == '__main__':
    start_location = (1, 1)
    goal_location = (3, 4)

    l0Domain = FourRoomMDP(width=5, height=5, init_loc=start_location, goal_locs=[goal_location])

    start_room = l0Domain.get_room_numbers(start_location)[0]
    goal_room = l0Domain.get_room_numbers(goal_location)[0]

    l1Domain = FourRoomL1MDP(start_room, goal_room)
    l1tf = l1Domain.terminal_func
    l1rf = l1Domain.reward_func

    policy_generators = []
    l0_policy_generator = L0PolicyGenerator(l0Domain)
    l1_policy_generator = L1PolicyGenerator(l0Domain, AbstractGridWorldL1StateMapper(l0Domain))
    policy_generators.append(l0_policy_generator)
    policy_generators.append(l1_policy_generator)

    l0State = l0Domain.init_state
    l1State = l1_policy_generator.generateAbstractState(l0State)

    l1Subtasks = [PrimitiveAbstractTask(action) for action in l0Domain.ACTIONS]
    a2rt = [FourRoomL1GroundedAction(a, l1Subtasks, l0Domain) for a in FourRoomL1MDP.ACTIONS]
    l1Root = FourRoomRootGroundedAction(FourRoomL1MDP.action_for_room_number(goal_room), a2rt, l1Domain,
                                        l1Domain.terminal_func, l1Domain.reward_func)

    agent = AMDPAgent(l1Root, policy_generators, l0Domain)
    agent.plan()