import numpy as np

from simple_rl.abstraction.action_abs.OptionClass import Option


class StripsObject(object):
    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class TaxiProposition(StripsObject):
    def __init__(self):
        self.taxi_at_passenger = False
        self.taxi_at_passenger_destination = False
        self.passenger_at_destination = False
        self.taxi_has_passenger = False
        self.passenger_in_taxi = False

    def vector(self):
        return np.array(self.__dict__.values())


class TaxiAction(StripsObject):
    def __init__(self):
        self.option = Option()
        self.precondition = TaxiProposition()
        self.positive_effect = TaxiProposition()
        self.negative_effect = TaxiProposition()

    def action_tuple(self):
        return self.precondition.vector(), self.positive_effect.vector(), self.negative_effect.vector()


class STRIPSFormulation(object):
    def __init__(self, symbolic_planner):
        self.symbolic_planner = symbolic_planner
        self.actions = []

    def get_proposition(self, state):
        agent = state.get_first_obj_of_class('agent')
        passenger = state.get_first_obj_of_class('passenger')

        proposition = TaxiProposition()
        proposition.taxi_at_passenger = agent['x'] == passenger['x'] and agent['y'] == passenger['y']
        proposition.taxi_at_passenger_destination = agent['x'] == passenger['dest_x'] and agent['y'] == passenger['dest_y']
        proposition.passenger_at_destination = passenger['x'] == passenger['dest_x'] and passenger['y'] == passenger['dest_y']
        proposition.taxi_has_passenger = agent['has_passenger']
        proposition.passenger_in_taxi = passenger['in_taxi']

        return proposition

    def get_action(self, option):
        action = TaxiAction()
        action.option = option

