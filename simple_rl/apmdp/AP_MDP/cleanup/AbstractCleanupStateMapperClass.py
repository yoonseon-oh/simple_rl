# Other imports.
from simple_rl.amdp.AMDPStateMapperClass import AMDPStateMapper
from simple_rl.apmdp.AP_MDP.cleanup.AbstractCleanupMDPClass import CleanupL1State, CleanupL2State


class AbstractCleanupL1StateMapper(AMDPStateMapper):
    def __init__(self, l0_domain):
        AMDPStateMapper.__init__(self, l0_domain)

    def map_state(self, l0_state):
        '''
        Args:
            l0_state (MDPState): L0 CleanupMDP
        Returns:
            projected_state : Mapping of state into L1 space
        '''
        obj_room = []
        for obj_xy in l0_state.obj_loc:
            obj_room.append(self.lower_domain.xy_to_room(obj_xy))
        robot_highlevel= self.lower_domain.get_highlevel_pose((l0_state.x,l0_state.y,l0_state.obj_id),l0_state.obj_loc)

        return CleanupL1State(robot_at=robot_highlevel['object'], robot_in=robot_highlevel['room'], obj_id=l0_state.obj_id, obj_room=obj_room, q=l0_state.q)

class AbstractCleanupL2StateMapper(AMDPStateMapper): # TODO: Modify
    def __init__(self, l1_domain):
        AMDPStateMapper.__init__(self, l1_domain)

    def map_state(self, l1_state):
        '''
        Args:
            l1_state (OOMDPState): L1 CleanupL1MDP
        Returns:
            projected_state (TaxiL1State): Mapping of state into L2 space
        '''

        return CleanupL2State(l1_state.obj_room, l1_state.q)


