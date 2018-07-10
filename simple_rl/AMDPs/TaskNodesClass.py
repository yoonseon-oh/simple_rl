class AbstractTask(object):
    def __init__(self):
        pass

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
        AbstractTask.__init__(self)
        self.action_name = action_name

    def __str__(self):
        return '{}'.format(self.get_name())

    def __repr__(self):
        return self.__str__()

    def get_name(self):
        return self.action_name

    def is_task_primitive(self):
        return True

    def is_terminal(self, current_state):
        return True

class NonPrimitiveAbstractTask(AbstractTask):
    def __init__(self, action_name, subtasks):
        AbstractTask.__init__(self)
        self.action_name = action_name
        self.subtasks = subtasks

    def get_name(self):
        return self.action_name

    def __str__(self):
        return '{}'.format(self.get_name())

    def __repr__(self):
        return self.__str__()

    def is_task_primitive(self):
        return False

class RootTaskNode(AbstractTask):
    def __init__(self, name, children, domain, terminal_func, reward_func):
        AbstractTask.__init__(self)
        self.name = name
        self.children = children
        self.domain = domain
        self.terminal_func = terminal_func
        self.reward_func = reward_func

    def is_terminal(self, current_state):
        return self.terminal_func(current_state)

    def __str__(self):
        return '{}'.format(self.name)

    def __repr__(self):
        return self.__str__()

    def is_task_primitive(self):
        return False
