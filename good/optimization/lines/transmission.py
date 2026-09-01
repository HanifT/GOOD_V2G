# from ..base.line import Line
# import pyomo.environ as pyomo

# class Transmission(Line):
#     '''
#     Links enable transfer of energy between nodes. Each link has exactly one source node
#     and exactly one target node with energy transferred from source to target.
#     '''
#     def __init__(self, handle, **kwargs):

#         super().__init__(handle, **kwargs)
        
#         self.installed_capacity = kwargs.get('installed_capacity', 0)
#         self.operating_cost = kwargs.get('operating_cost', 2.222222222222222e-09)
#         self.efficiency = kwargs.get('efficiency', 0.90)

#         # Can capacity be expanded
#         self.capex_limit = kwargs.get('capex_limit', kwargs.get('capex_capacity', 0))
#         self.capex_cost = kwargs.get('capex_cost', 0)
#         self.extensible = self.capex_limit > 0

#     def parameters(self, model):

#         # Capacity Expansion
#         if not self.extensible:

#             handle = f"{self.handle}::capex"
#             self.handles.append(handle)
#             setattr(
#                 model, handle,
#                 pyomo.Param(initialize = 0),
#             )

#         return model

#     def variables(self, model):

#         # Production (transmission flow)
#         handle = f"{self.handle}::transmission"
#         self.handles.append(handle)
#         setattr(
#             model, handle,
#             pyomo.Var(
#                 model.steps,
#                 initialize = [0] * len(model.steps),
#                 within = pyomo.NonNegativeReals
#                 ),
#             )

#         # Capacity Expansion
#         if self.extensible:

#             handle = f"{self.handle}::capex"
#             self.handles.append(handle)
#             setattr(
#                 model, handle,
#                 pyomo.Var(
#                     initialize = 0,
#                     bounds = (0, self.capex_limit), within = pyomo.NonNegativeReals,
#                     ),
#                 )

#         return model

#     def constraints(self, model):

#         # Capacity constraint
#         transmission = getattr(model, f"{self.handle}::transmission")
#         capex = getattr(model, f"{self.handle}::capex")
        
#         setattr(
#             model, f"{self.handle}::capacity_constraint",
#             pyomo.Constraint(
#                 model.steps,
#                 rule=lambda m, t: transmission[t] <= self.installed_capacity + capex
#             )
#         )
        
#         return model

#     def transmit(self, model, step=None):
#         transmission = getattr(model, f"{self.handle}::transmission")

#         if step is None:
#             energy = pyomo.quicksum(
#                 transmission[i] * model.time_step / self.efficiency for i in model.steps
#             )
#         else:
#             energy = transmission[step] * model.time_step / self.efficiency

#         return energy

#     def receive(self, model, step=None):
#         transmission = getattr(model, f"{self.handle}::transmission")

#         if step is None:
#             energy = pyomo.quicksum(
#                 transmission[i] * model.time_step for i in model.steps
#             )
#         else:
#             energy = transmission[step] * model.time_step

#         return energy

#     def capacity(self, model, step = None):

#         capex = getattr(model, f"{self.handle}::capex")

#         capacity = self.installed_capacity + capex

#         return capacity

#     def objective(self, model):
#         """
#         Calculate the cost of transmission
#         """

#         transmission = getattr(model, f"{self.handle}::transmission")

#         transmission_cost =  pyomo.quicksum(
#             transmission[t] * model.time_step * self.operating_cost for t in model.steps
#         )

#         capex = getattr(model, f"{self.handle}::capex")

#         expansion_cost = capex * self.capex_cost * model.amortization

#         cost = transmission_cost + expansion_cost

#         return cost

#     def solution(self, model):

#         solution = {}

#         for handle in self.handles:

#             value = list(getattr(model, handle).extract_values().values())
#             solution[handle.split('::')[1]] = value

#         return solution

from ..base.line import Line

import pyomo.environ as pyomo


class Transmission(Line):
    """
    Directed transmission line between two regions.

    A line is inactive when it has:
        installed_capacity <= 0
        and
        capex_limit <= 0

    An inactive line cannot transmit electricity and cannot be expanded.
    Therefore, no Pyomo parameters, variables, or constraints are created
    for that line.
    """

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        self.installed_capacity = float(
            kwargs.get("installed_capacity", 0) or 0
        )

        self.operating_cost = float(
            kwargs.get(
                "operating_cost",
                2.222222222222222e-09,
            ) or 0
        )

        self.efficiency = float(
            kwargs.get("efficiency", 0.90) or 0.90
        )

        # Transmission expansion settings
        self.capex_limit = float(
            kwargs.get(
                "capex_limit",
                kwargs.get("capex_capacity", 0),
            ) or 0
        )

        self.capex_cost = float(
            kwargs.get("capex_cost", 0) or 0
        )

        if self.installed_capacity < 0:
            raise ValueError(
                f"{self.handle}: installed_capacity cannot be negative. "
                f"Received {self.installed_capacity}."
            )

        if self.capex_limit < 0:
            raise ValueError(
                f"{self.handle}: capex_limit cannot be negative. "
                f"Received {self.capex_limit}."
            )

        if self.efficiency <= 0 or self.efficiency > 1:
            raise ValueError(
                f"{self.handle}: efficiency must be greater than 0 "
                f"and less than or equal to 1. "
                f"Received {self.efficiency}."
            )

        self.extensible = self.capex_limit > 0

        # No flow or expansion is mathematically possible when both are zero.
        self.active = (
            self.installed_capacity > 0
            or self.capex_limit > 0
        )

    def parameters(self, model):

        # Do not create any Pyomo component for an inactive line.
        if not self.active:
            return model

        # A fixed zero capex parameter is needed for active,
        # non-expandable lines so the remaining methods can use
        # the same capacity expression.
        if not self.extensible:

            handle = f"{self.handle}::capex"
            self.handles.append(handle)

            setattr(
                model,
                handle,
                pyomo.Param(
                    initialize=0.0,
                ),
            )

        return model

    def variables(self, model):

        # An inactive line has no hourly flow variables.
        if not self.active:
            return model

        # Hourly transmission flow
        handle = f"{self.handle}::transmission"
        self.handles.append(handle)

        setattr(
            model,
            handle,
            pyomo.Var(
                model.steps,
                initialize=0.0,
                within=pyomo.NonNegativeReals,
            ),
        )

        # Capacity expansion
        if self.extensible:

            handle = f"{self.handle}::capex"
            self.handles.append(handle)

            setattr(
                model,
                handle,
                pyomo.Var(
                    initialize=0.0,
                    bounds=(0.0, self.capex_limit),
                    within=pyomo.NonNegativeReals,
                ),
            )

        return model

    def constraints(self, model):

        # An inactive line requires no capacity constraints.
        if not self.active:
            return model

        transmission = getattr(
            model,
            f"{self.handle}::transmission",
        )

        capex = getattr(
            model,
            f"{self.handle}::capex",
        )

        setattr(
            model,
            f"{self.handle}::capacity_constraint",
            pyomo.Constraint(
                model.steps,
                rule=lambda m, t: (
                    transmission[t]
                    <= self.installed_capacity + capex
                ),
            ),
        )

        return model

    def transmit(self, model, step=None):

        # Inactive lines contribute exactly zero to the source balance.
        if not self.active:
            return 0.0

        transmission = getattr(
            model,
            f"{self.handle}::transmission",
        )

        if step is None:
            return pyomo.quicksum(
                transmission[t]
                * model.time_step
                / self.efficiency
                for t in model.steps
            )

        return (
            transmission[step]
            * model.time_step
            / self.efficiency
        )

    def receive(self, model, step=None):

        # Inactive lines contribute exactly zero to the destination balance.
        if not self.active:
            return 0.0

        transmission = getattr(
            model,
            f"{self.handle}::transmission",
        )

        if step is None:
            return pyomo.quicksum(
                transmission[t] * model.time_step
                for t in model.steps
            )

        return transmission[step] * model.time_step

    def capacity(self, model, step=None):

        if not self.active:
            return 0.0

        capex = getattr(
            model,
            f"{self.handle}::capex",
        )

        return self.installed_capacity + capex

    def objective(self, model):
        """
        Calculate transmission operating and expansion costs.
        """

        if not self.active:
            return 0.0

        transmission = getattr(
            model,
            f"{self.handle}::transmission",
        )

        transmission_cost = pyomo.quicksum(
            transmission[t]
            * model.time_step
            * self.operating_cost
            for t in model.steps
        )

        capex = getattr(
            model,
            f"{self.handle}::capex",
        )

        expansion_cost = (
            capex
            * self.capex_cost
            * model.amortization
        )

        return transmission_cost + expansion_cost

    def solution(self, model):

        # No results exist for an inactive line.
        if not self.active:
            return {
                "transmission": [],
                "capex": [0.0],
            }

        solution = {}

        for handle in self.handles:

            component = getattr(model, handle)

            if component.is_indexed():
                values = [
                    pyomo.value(component[t])
                    for t in component
                ]
            else:
                values = [
                    pyomo.value(component)
                ]

            solution[
                handle.split("::")[1]
            ] = values

        return solution