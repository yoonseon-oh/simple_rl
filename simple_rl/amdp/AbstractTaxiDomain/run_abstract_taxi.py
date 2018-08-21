# Other imports.
from simple_rl.tasks.taxi.TaxiOOMDPClass import TaxiOOMDP
from simple_rl.amdp.AbstractTaxiDomain.AbstractTaxiPolicyGeneratorClass import L0PolicyGenerator, L1PolicyGenerator
from simple_rl.amdp.AMDPTaskNodesClass import PrimitiveAbstractTask
from simple_rl.amdp.AMDPSolverClass import AMDPAgent
from simple_rl.amdp.AbstractTaxiDomain.AbstractTaxiMDPClass import TaxiL1GroundedAction, TaxiL1OOMDP, TaxiRootGroundedAction
from simple_rl.amdp.AbstractTaxiDomain.AbstractTaxiStateMapperClass import AbstractTaxiL1StateMapper

def main():
    agent = {"x": 1, "y": 1, "has_passenger": 0}
    passengers = [{"x": 5, "y": 1, "dest_x": 5, "dest_y": 5, "in_taxi": 0}]
    passenger = passengers[0]
    l0_domain = TaxiOOMDP(width=5, height=5, agent=agent, walls=[], passengers=passengers)

    agent_init_color = l0_domain.color_for_location((agent['x'], agent['y']))
    passenger_init_color = l0_domain.color_for_location((passenger['x'], passenger['y']))
    passenger_dest_color = l0_domain.color_for_location((passenger['dest_x'], passenger['dest_y']))
    l1_domain = TaxiL1OOMDP(agent_init_color, passenger_init_color, passenger_dest_color)

    policy_generators = []
    l0_policy_generator = L0PolicyGenerator(l0_domain)
    l1_policy_generator = L1PolicyGenerator(l0_domain, AbstractTaxiL1StateMapper(l0_domain))
    policy_generators.append(l0_policy_generator)
    policy_generators.append(l1_policy_generator)

    l1_subtasks = [PrimitiveAbstractTask(action) for action in TaxiOOMDP.ACTIONS]
    l1_tasks = [TaxiL1GroundedAction(a, l1_subtasks, l0_domain) for a in TaxiL1OOMDP.ACTIONS]

    l1_root = TaxiRootGroundedAction('ride', l1_tasks, l1_domain, l1_domain.terminal_func,
                                     l1_domain.reward_func)

    agent = AMDPAgent(l1_root, policy_generators, l0_domain)
    agent.plan()

if __name__ == '__main__':
    main()
