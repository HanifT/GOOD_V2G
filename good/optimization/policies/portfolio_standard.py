from ..base.policy import Policy

import numpy as np
import pyomo.environ as pyomo
import logging

class Portfolio_Standard(Policy):

    def __init__(self, handle, **kwargs):

        super().__init__(handle, **kwargs)

        # Terms of the policy
        self.ratio = kwargs.get('ratio', 0)

        self.non_compliance_capacity = kwargs.get('non_compliance_capacity', 0)
        self.non_compliance_cost = kwargs.get('non_compliance_cost', 1)

        # Criteria for assigning assets to sets
        inclusion_criteria = kwargs.get('inclusion_criteria', {})
        exclusion_criteria = kwargs.get('exclusion_criteria', {})
        self.region_handles = kwargs.get('region_handles', [])

        # Truning srings into functions if needed
        self.inclusion_criteria = self.interpret(inclusion_criteria)
        self.exclusion_criteria = self.interpret(exclusion_criteria)

        # Building the incuded and excluded sets
        self.assets = kwargs.get('assets', {})
        self.included = self.build_set(self.assets, self.inclusion_criteria)
        self.excluded = self.build_set(self.assets, self.exclusion_criteria)

        # print(handle)
        # print(len(self.included))
        # print(len(self.excluded))


    def build_set(self, assets, criteria):

        included = {}

        for key, asset in assets.items():

            include = True

            for fun in criteria.values():

                include *= fun(asset)

            if include:

                included[key] = asset

        return included

    def interpret(self, criteria):

        interpreted_criteria = {}

        for key, fun in criteria.items():
            if isinstance(fun, str):

                fun = eval(fun)
            
            interpreted_criteria[key] = fun

        return interpreted_criteria

    def parameters(self, model):

        handle = f"{self.handle}::ratio"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Param(initialize = self.ratio, mutable = True),
        )

        return model

    def variables(self, model):

        # Shortfall - avoids infeasibility due to insufficient supply from included set
        handle = f"{self.handle}::non_compliance"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Var(
                initialize = 0,
                bounds = (0, self.non_compliance_capacity),
                ),
            )

        return model

    def _total_wastage_energy(self, model):
        """
        Sum regional wastage across the regions covered by this policy.
        Wastage is defined in Region as f'{region_handle}::wastage'[t].
        """
        if not self.region_handles:
            return 0

        total_wastage = 0

        for region_handle in self.region_handles:
            wastage_name = f"{region_handle}::wastage"

            if hasattr(model, wastage_name):
                wastage_var = getattr(model, wastage_name)

                total_wastage += pyomo.quicksum(
                    wastage_var[t] * model.time_step # we add * model.time_step
                    for t in model.steps
                )

        return total_wastage

    def constraints(self, model):

        ratio = getattr(model, f"{self.handle}::ratio")
        if self.included:
            included_generation = sum(
                    asset['object'].energy(model) for asset in self.included.values()
                )

        else:
            included_generation = 0
        if self.excluded:
            excluded_generation = sum(
                    asset['object'].energy(model) for asset in self.excluded.values()
                )

        else:
            excluded_generation = 0
        non_compliance = getattr(model, f"{self.handle}::non_compliance")

        # Find all unique regions from assets
        regions = set()
        for asset_data in self.assets.values():
            node = asset_data.get('node', None)
            if node:
                regions.add(node)


        # Approximate renewable curtailment using regional wastage
        total_wastage = self._total_wastage_energy(model)

        # Net included generation after subtracting curtailment proxy
        included_generation_net = included_generation - total_wastage

        setattr(
            model, f"{self.handle}::compliance",
            pyomo.Constraint(
                expr=(
                        included_generation_net + non_compliance >=
                        ratio * (included_generation_net + excluded_generation)
                )
            )
        )
    
        return model

    def objective(self, model):

        non_compliance = getattr(model, f"{self.handle}::non_compliance")

        cost = non_compliance * self.non_compliance_cost
        
        return cost