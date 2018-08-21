from simple_rl.tasks.cleanup.CleanupMDPClass import CleanUpMDP
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupMDPClass import CleanupL1MDP, CleanupL1GroundedAction, CleanupRootGroundedAction
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupPolicyGenerators import CleanupL0PolicyGenerator, CleanupL1PolicyGenerator
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AbstractCleanupDomain.AbstractCleanupStateMapper import AbstractCleanupL1StateMapper

def create_l0_cleanup_domain():
    from simple_rl.tasks.cleanup.cleanup_block import CleanUpBlock
    from simple_rl.tasks.cleanup.cleanup_door import CleanUpDoor
    from simple_rl.tasks.cleanup.cleanup_room import CleanUpRoom
    from simple_rl.tasks.cleanup.cleanup_task import CleanUpTask

    task = CleanUpTask("purple", "yellow")
    room1 = CleanUpRoom("room1", [(x, y) for x in range(5) for y in range(3)], "blue")
    block1 = CleanUpBlock("block1", 3, 1, color="purple")
    room2 = CleanUpRoom("room2", [(x, y) for x in range(5) for y in range(3, 6)], color="yellow")
    rooms = [room1, room2]
    blocks = [block1]
    doors = [CleanUpDoor(3, 2)]
    return CleanUpMDP(task, rooms=rooms, doors=doors, blocks=blocks)

def main():
    l0_domain = create_l0_cleanup_domain()
    l1_domain = CleanupL1MDP(l0_domain)

    l0_policy_generator = CleanupL0PolicyGenerator(l0_domain, verbose=True)
    l1_policy_generator = CleanupL1PolicyGenerator(l0_domain, AbstractCleanupL1StateMapper(l0_domain), verbose=True)
    policy_generators = [l0_policy_generator, l1_policy_generator]

    l0_init_state = l0_domain.init_state
    l1_init_state = l1_policy_generator.generateAbstractState(l0_init_state)

    l0_actions = [PrimitiveAbstractTask(action) for action in l0_domain.ACTIONS]
    l1_actions = [CleanupL1GroundedAction(action, l0_actions, l0_domain) for action in CleanupL1MDP.ground_actions(l1_init_state)]
    root_action = CleanupRootGroundedAction(str(l0_domain.task), l1_actions, l1_domain, l1_domain.terminal_func, l1_domain.reward_func)

    agent = AMDPAgent(root_action, policy_generators, l0_domain)
    agent.plan()

if __name__ == '__main__':
    main()


