class AbstractTask(object):
    def __init__(self, name):
        self.action_name = name
        self.subtasks = []

    def __str__(self):
        pass

    def __repr__(self):
        pass

    def get_name(self):
        pass

    def is_task_primitive(self):
        pass

    def is_terminal(self, current_state):
        pass

    def get_reward_function(self, action):
        pass

class PrimitiveAbstractTask(AbstractTask):
    def __init__(self, action_name):
        AbstractTask.__init__(self, name=action_name)

    def __str__(self):
        return 'PrimitiveNode::{}'.format(self.get_name())

    def __repr__(self):
        return self.__str__()

    def get_name(self):
        return self.action_name

    def is_task_primitive(self):
        return True

    def is_terminal(self, current_state):
        return True

class NonPrimitiveAbstractTask(AbstractTask):
    def __init__(self, action_name, subtasks, terminal_func, reward_func):
        AbstractTask.__init__(self, name=action_name)
        self.subtasks = subtasks
        self.terminal_func = terminal_func
        self.reward_func = reward_func

    def get_name(self):
        return self.action_name

    def __str__(self):
        return 'NonPrimitiveNode::{}'.format(self.get_name())

    def __repr__(self):
        return self.__str__()

    def is_task_primitive(self):
        return False

    def is_terminal(self, current_state):
        return self.terminal_func(current_state)

class RootTaskNode(AbstractTask):
    def __init__(self, name, children, domain, terminal_func, reward_func):
        AbstractTask.__init__(self, name=name)
        self.subtasks = children
        self.domain = domain
        self.terminal_func = terminal_func
        self.reward_func = reward_func

    def __str__(self):
        return 'RootNode::{}'.format(self.action_name)

    def __repr__(self):
        return self.__str__()

    def is_task_primitive(self):
        return False

    def is_terminal(self, current_state):
        return self.terminal_func(current_state)
