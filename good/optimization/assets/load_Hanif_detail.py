import numpy as np

from ..base.asset import Asset
import pyomo.environ as pyomo


class Load(Asset):

    def __init__(self, handle, **kwargs):
        '''
        A Load is a grid asset which adds or subtracts energy based on a profile
        and do not receive a dispacth signal from the grid. This includes end user loads
        and certaint types of renewables.
        '''

        super().__init__(handle, **kwargs)

        self.installed_capacity = kwargs.get('installed_capacity', 0)
        self.operating_cost = kwargs.get('operating_cost', 0)

        # Curtailment flag (only True for renewables modeled as negative load)
        self.curtailable = kwargs.get("curtailable", False)

        # Optional curtailment penalty ($/J) if you want (often 0)

        # print(self.handle, self.installed_capacity / 1e6)

        # Can capacity be expanded
        self.capex_capacity = kwargs.get('capex_capacity', 0)
        self.capex_cost = kwargs.get('capex_cost', 0)
        self.extensible = self.capex_capacity > 0

        self.shift_capacity = kwargs.get('shift_capacity', 0)
        self.shiftable = self.shift_capacity > 0

        self.shift_window = kwargs.get('shift_window', None)

        self.profile = kwargs.get('profile', None)

        # Shifting Cost
        self.shift_cost = kwargs.get("shift_cost", 0.0)  # $/J (or $/MWh if your time_step is hours and shift is MW)
        if self.shift_cost is not None and not np.isscalar(self.shift_cost):
            self.shift_cost = np.array(self.shift_cost)

        # Fixed cost for having shiftable load available
        self.fixed_shift_cost = kwargs.get("fixed_shift_cost", 0.0)

        if self.profile is not None:
            self.profile = np.array(self.profile)

    def parameters(self, model):

        if self.shift_window is None:
            self.shift_window = len(model.steps)

        if self.profile is None:
            self.profile = [0] * len(model.steps)

        handle = f"{self.handle}::profile"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Param(model.steps,
                        initialize=self.profile[int(model.start):int(model.stop)]
                        )
        )

        # Capacity Expansion
        if not self.extensible:
            handle = f"{self.handle}::capex"
            self.handles.append(handle)
            setattr(
                model, handle,
                pyomo.Param(initialize=0),
            )

        if not self.shiftable:
            handle = f"{self.handle}::shift"
            self.handles.append(handle)
            setattr(
                model, handle,
                pyomo.Param(
                    model.steps, initialize=[0] * len(model.steps),
                )
            )

        if np.isscalar(self.shift_cost):
            shift_cost_profile = [self.shift_cost] * len(model.steps)
        else:
            shift_cost_profile = self.shift_cost[int(model.start):int(model.stop)]

        handle = f"{self.handle}::shift_cost"
        self.handles.append(handle)
        setattr(
            model, handle,
            pyomo.Param(
                model.steps,
                initialize=shift_cost_profile
            )
        )
        return model

    def variables(self, model):

        # Capacity Expansion
        if self.extensible:
            handle = f"{self.handle}::capex"
            self.handles.append(handle)
            setattr(
                model, handle,
                pyomo.Var(
                    initialize=0,
                    bounds=(0, self.capex_capacity), within=pyomo.NonNegativeReals,
                ),
            )

        # if self.shiftable:
        #
        #     handle = f"{self.handle}::shift"
        #     self.handles.append(handle)
        #     setattr(
        #         model, handle,
        #         pyomo.Var(
        #             model.steps,
        #             initialize = [0] * len(model.steps),
        #             within = pyomo.Reals,
        #             bounds = (-self.shift_capacity, self.shift_capacity)
        #             ),
        #         )

        if self.shiftable:
            # signed net shift (can be +/-)
            handle = f"{self.handle}::shift"
            self.handles.append(handle)
            setattr(
                model, handle,
                pyomo.Var(
                    model.steps,
                    initialize=0,
                    within=pyomo.Reals,
                )
            )

            # nonnegative components for absolute value
            handle_up = f"{self.handle}::shift_up"
            self.handles.append(handle_up)
            setattr(
                model, handle_up,
                pyomo.Var(model.steps, initialize=0, within=pyomo.NonNegativeReals)
            )

            handle_dn = f"{self.handle}::shift_down"
            self.handles.append(handle_dn)
            setattr(
                model, handle_dn,
                pyomo.Var(model.steps, initialize=0, within=pyomo.NonNegativeReals)
            )

        return model

    def constraints(self, model):

        # -------------------------
        # Load shifting constraints
        # -------------------------
        if self.shiftable:

            shift = getattr(model, f"{self.handle}::shift")
            shift_up = getattr(model, f"{self.handle}::shift_up")
            shift_down = getattr(model, f"{self.handle}::shift_down")
            profile = getattr(model, f"{self.handle}::profile")
            capex = getattr(model, f"{self.handle}::capex")

            # 1) Link signed shift to up/down parts
            setattr(
                model, f"{self.handle}::shift_link",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: shift[t] == shift_up[t] - shift_down[t]
                ),
            )

            # 2) Magnitude limit (implements |shift[t]| <= shift_capacity in a linear way)
            # For V1G: shift capacity at hour t should be proportional to the actual load at that hour
            # If profile[t] = 0.5 and capacity = -1000 MW, base_load = -500 MW
            # Then we can shift by at most 500 MW (can go from 0 to -1000 MW or vice versa)
            def shift_mag_rule(m, t):
                # Time-varying shift capacity based on actual load magnitude at hour t
                base_power_magnitude = abs(profile[t] * (self.installed_capacity + capex))
                return shift_up[t] + shift_down[t] <= base_power_magnitude

            setattr(
                model, f"{self.handle}::shift_mag_limit",
                pyomo.Constraint(model.steps, rule=shift_mag_rule)
            )

            # 3) Net-zero shift in each window
            steps_list = list(model.steps)

            # If shift_window is None, parameters() should set it to len(model.steps),
            # but guard anyway:
            window = int(self.shift_window) if self.shift_window is not None else len(steps_list)
            window = max(window, 1)

            for i_start in range(0, len(steps_list), window):
                window_steps = steps_list[i_start:i_start + window]

                shift_sum = pyomo.quicksum(shift[t] for t in window_steps)

                setattr(
                    model, f"{self.handle}::shift_sum_constraint_{i_start}",
                    pyomo.Constraint(expr=(0, shift_sum, 0)),
                )

            # 4) Physical lower bound:
            # shifted load cannot exceed the maximum charging magnitude
            setattr(
                model, f"{self.handle}::shift_lower_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: profile[t] * (self.installed_capacity + capex) + shift[t]
                                      >= (self.installed_capacity + capex)
                ),
            )

            # 5) Physical upper bound:
            # shifted load cannot become positive (cannot turn load into generation)
            setattr(
                model, f"{self.handle}::shift_upper_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: profile[t] * (self.installed_capacity + capex) + shift[t]
                                      <= 0
                ),
            )

        return model

    def energy(self, model, step=None):

        profile = getattr(model, f"{self.handle}::profile")
        shift = getattr(model, f"{self.handle}::shift")
        capex = getattr(model, f"{self.handle}::capex")

        capacity = self.installed_capacity + capex

        if step is None:
            energy = pyomo.quicksum(
                (profile[t] * capacity + shift[t]) * model.time_step
                for t in model.steps
            )
        else:
            energy = (profile[step] * capacity + shift[step]) * model.time_step

        return energy

    def power(self, model, step=None):

        profile = getattr(model, f"{self.handle}::profile")
        shift = getattr(model, f"{self.handle}::shift")
        capex = getattr(model, f"{self.handle}::capex")

        capacity = self.installed_capacity + capex

        if step is None:

            power = pyomo.quicksum(
                profile[t] * capacity + shift[t] for t in model.steps
            )

        else:

            power = profile[step] * capacity + shift[step]

        return power

    def capacity(self, model, step=None):

        capex = getattr(model, f"{self.handle}::capex")

        capacity = self.installed_capacity + capex

        return capacity

    def objective(self, model):

        profile = getattr(model, f"{self.handle}::profile")
        shift = getattr(model, f"{self.handle}::shift")
        capex = getattr(model, f"{self.handle}::capex")
        capacity = self.installed_capacity + capex

        expansion_cost = capex * self.capex_cost * model.amortization

        cost = expansion_cost

        if self.shiftable:
            shift_up = getattr(model, f"{self.handle}::shift_up")
            shift_down = getattr(model, f"{self.handle}::shift_down")
            shift_cost = getattr(model, f"{self.handle}::shift_cost")
            # original
            cost += pyomo.quicksum(
                shift_down[t] * model.time_step * shift_cost[t]
                for t in model.steps
            )

            # Penalize TOTAL shifting magnitude, not just one direction
            # cost += pyomo.quicksum(
            #     (shift_up[t] + shift_down[t]) * model.time_step * shift_cost[t]
            #     for t in model.steps
            # )

            # Fixed cost on available shiftable capacity
            cost += self.shift_capacity * self.fixed_shift_cost * model.amortization
        return cost

    def solution(self, model):

        solution = {}

        for handle in self.handles:
            value = list(getattr(model, handle).extract_values().values())
            solution[handle.split('::')[1]] = value

        # Net Contribution
        profile = list(
            getattr(model, f"{self.handle}::profile").extract_values().values()
        )
        shift = list(getattr(model, f"{self.handle}::shift").extract_values().values())
        capex = list(getattr(model, f"{self.handle}::capex").extract_values().values())
        capacity = self.installed_capacity + capex[0]

        solution["net"] = (
            [profile[i] * capacity + shift[i] for i in model.steps]
        )

        return solution