import pyomo.environ as pyomo

class Edge:

    __base__ = 'Edge'

    def __init__(self, handle, **kwargs):

        self.handle = handle
        self.handles = []

        self.lines = {}

    def parameters(self, model):

        return model

    def variables(self, model):

        return model

    def constraints(self, model):

        return model

    def objective(self, model):
        """Base objective function returns zero cost"""

        return 0.  # Default to no cost for nodes

    def solution(self, model):

        solution = {}

        for handle in self.handles:

            value = list(getattr(model, handle).extract_values().values())
            solution[handle] = value

        return solution 