from ..base.policy import Policy

import numpy as np
import pyomo.environ as pyomo
import logging

class Portfolio_Standard(Policy):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        # Terms of the policy
        self.ratio = kwargs.get('generation_portion', 0)

        self.non_compliance_capacity = kwargs.get('non_compliance_capacity', 0)
        self.non_compliance_cost = kwargs.get('non_compliance_cost', 1)

        # Criteria for assigning assets to sets
        inclusion_criteria = kwargs.get('inclusion_criteria', {})
        exclusion_criteria = kwargs.get('exclusion_criteria', {})

        # Truning srings into functions if needed
        self.inclusion_criteria = self.interpret(inclusion_criteria)
        self.exclusion_criteria = self.interpret(exclusion_criteria)

        # Building the incuded and excluded sets
        self.assets = kwargs.get('assets', [])
        self.included = self.build_set(self.assets, self.inclusion_criteria)
        self.excluded = self.build_set(self.assets, self.exclusion_criteria)


    def build_set(self, assets, criteria):

        included = []

        for asset in assets:

            include = True

            for fun in criteria:

                include *= fun(asset)

            if include:

                included.append(asset)

        return included

    def interpret(self, criteria):

        interpreted_criteria = []

        for fun in criteria:
            if isinstance(fun, str):

                fun = eval(fun)
            
            interpreted_criteria.append(fun)

        return interpreted_criteria

    def variables(self, model):

        # Shortfall - avoids infeasibility due to insufficient supply from included set
        handle = f"{self.handle}::non_compliance"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Var(
                model.steps,
                initialize = 0,
                bounds = (0, self.non_compliance_capacity),
                ),
            )

    def constraints(self, model):

        included_generation = sum(
                asset['object'].energy(model) for asset in self.included 
            )

        excluded_generation = sum(
                asset['object'].energy(model) for asset in self.excluded 
            )

        non_compliance = getattr(model, f"{self.handle}::non_compliance")

        setattr(
            model, f"{self.handle}::compliance",
            pyomo.Constraint(
                expr = (
                    included_generation + non_compliance >=
                    included_generation * self.generation_portion +
                    excluded_generation * self.generation_portion
                    )
                )
            )
    
        return model

    def objective(self, model):

        non_compliance = getattr(model, f"{self.handle}::non_compliance")

        cost = non_compliance * self.non_compliance_cost
        
        return cost