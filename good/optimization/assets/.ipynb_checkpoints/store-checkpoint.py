from ..base.asset import Asset
import pyomo.environ as pyomo
import numpy as np
import math


class Store(Asset):
    """
    Storage asset used by GOOD.

    This version keeps the original Store structure, but adds two optional
    speed improvements that are especially useful for V2G cohorts:

    1) active_steps:
       Lets the runner pass the active model steps directly instead of passing
       a full availability profile for every cohort.

    2) direct_throughput_cost:
       Removes the extra throughput variable and throughput_link constraint.
       The objective uses charge_power + discharge_power directly instead.
       This is algebraically equivalent when operating_cost is nonnegative.

    The defaults preserve the old behavior for normal storage assets. For
    type == "ev_v2g", the faster path is enabled by default.
    """

    def __init__(self, handle, **kwargs):
        super().__init__(handle, **kwargs)

        self.type = kwargs.get("type", "store")

        # Existing energy capacity [J]
        self.installed_capacity = kwargs.get("installed_capacity", 0.0)

        # Existing power capacity [W]
        self.installed_power_capacity = kwargs.get("installed_power_capacity", None)

        self.operating_cost = kwargs.get("operating_cost", 0.0)
        self.fixed_operating_cost = kwargs.get("fixed_operating_cost", 0.0)

        if self.operating_cost is not None and not np.isscalar(self.operating_cost):
            self.operating_cost = np.array(self.operating_cost)

        self.availability_profile = kwargs.get("availability_profile", None)
        if self.availability_profile is None:
            self.availability_profile = None
        elif isinstance(self.availability_profile, dict):
            pass
        else:
            self.availability_profile = np.array(self.availability_profile)

        # Optional compact active-step input.
        # This avoids creating a full availability Param over all model steps
        # for every V2G cohort.
        self.active_steps_input = kwargs.get("active_steps", None)

        # If True, do not create a Pyomo availability Param. The availability
        # profile is only used to decide which steps receive variables.
        self.skip_availability_param = kwargs.get(
            "skip_availability_param",
            self.type == "ev_v2g"
        )

        # If True, skip the throughput variable and throughput_link constraint.
        # The objective charges operating cost directly on charge + discharge.
        self.direct_throughput_cost = kwargs.get(
            "direct_throughput_cost",
            self.type == "ev_v2g"
        )

        if self.direct_throughput_cost:
            if np.isscalar(self.operating_cost):
                if float(self.operating_cost) < -1e-15:
                    raise ValueError(
                        f"{self.handle}: direct_throughput_cost requires "
                        "nonnegative operating_cost."
                    )
            else:
                if np.any(np.asarray(self.operating_cost, dtype=float) < -1e-15):
                    raise ValueError(
                        f"{self.handle}: direct_throughput_cost requires "
                        "nonnegative operating_cost."
                    )

        # Efficiencies
        # If only round trip efficiency is given, split it symmetrically.
        self.efficiency = kwargs.get("efficiency", 1.0)
        self.charge_efficiency = kwargs.get("charge_efficiency", None)
        self.discharge_efficiency = kwargs.get("discharge_efficiency", None)

        if self.charge_efficiency is None or self.discharge_efficiency is None:
            eta = float(self.efficiency)
            eta = max(min(eta, 1.0), 1e-8)
            root_eta = math.sqrt(eta)
            self.charge_efficiency = root_eta
            self.discharge_efficiency = root_eta

        self.production_rate = kwargs.get("production_rate", 1.0)     # 1/s
        self.consumption_rate = kwargs.get("consumption_rate", 1.0)   # 1/s

        self.initial = kwargs.get("initial", 0.0)

        self.reset_each_window = kwargs.get(
            "reset_each_window",
            self.type == "ev_v2g"
        )

        # Expansion bounds
        self.capex_capacity = kwargs.get("capex_capacity", 0.0)   # energy bound [J]
        self.capex_power_capacity = kwargs.get("capex_power_capacity", None)  # power bound [W]

        # Costs
        self.capex_cost = kwargs.get("capex_cost", 0.0)  # legacy energy cost [$ / J]
        self.capex_cost_energy = kwargs.get("capex_cost_energy", self.capex_cost)
        self.capex_cost_power = kwargs.get("capex_cost_power", 0.0)

        # Optional fixed duration [hours]
        self.duration_hours = kwargs.get("duration_hours", None)

        # Extensibility flags
        self.extensible_energy = self.capex_capacity > 0
        self.extensible_power = (
            self.capex_power_capacity is not None and self.capex_power_capacity > 0
        )
        self.extensible = self.extensible_energy or self.extensible_power

        # Filled later.
        self._all_steps = []
        self._active_steps = []

    def _existing_power_capacity(self):
        if self.installed_power_capacity is not None:
            return float(self.installed_power_capacity)
        return float(self.installed_capacity) * float(self.production_rate)

    def _resolve_active_steps_from_input(self, model):
        """
        Convert self.active_steps_input to model step labels.

        The GOOD model may use either local steps, such as 0..719, or absolute
        steps, such as 5376..6095. This helper accepts either form and also
        tries a model.start shift as a fallback.
        """
        model_steps = list(model.steps)
        model_step_set = set(model_steps)

        if self.active_steps_input is None:
            return None

        raw_active_steps = [int(t) for t in self.active_steps_input]

        active_steps = [
            t for t in raw_active_steps
            if t in model_step_set
        ]

        if active_steps:
            return active_steps

        # Safety fallback if the user passed local steps but model.steps are
        # absolute.
        model_start = int(getattr(model, "start", 0))
        shifted_steps = [
            model_start + int(t)
            for t in raw_active_steps
        ]

        active_steps = [
            t for t in shifted_steps
            if t in model_step_set
        ]

        return active_steps

    def _resolve_availability_profile_and_active_steps(self, model):
        """
        Return availability_profile, active_steps.

        availability_profile is returned only when needed for a Pyomo Param.
        If skip_availability_param is True, this method avoids building a large
        full availability dictionary where possible.
        """
        model_steps = list(model.steps)

        # Fast compact path.
        active_steps = self._resolve_active_steps_from_input(model)
        if active_steps is not None:
            if self.skip_availability_param:
                return None, active_steps

            active_set = set(active_steps)
            availability_profile = {
                t: 1.0 if t in active_set else 0.0
                for t in model_steps
            }
            return availability_profile, active_steps

        # Original behavior, with optional no-Param path.
        if self.availability_profile is None:
            active_steps = model_steps

            if self.skip_availability_param:
                return None, active_steps

            availability_profile = {
                t: 1.0
                for t in model_steps
            }
            return availability_profile, active_steps

        if isinstance(self.availability_profile, dict):
            if self.skip_availability_param:
                active_steps = [
                    t for t in model_steps
                    if float(self.availability_profile.get(t, 0.0)) > 0.0
                ]

                # Safety fallback if availability keys are local but model
                # steps are absolute.
                if not active_steps and len(self.availability_profile) > 0:
                    model_start = int(getattr(model, "start", 0))
                    active_steps = [
                        t for t in model_steps
                        if float(
                            self.availability_profile.get(
                                int(t) - model_start,
                                0.0
                            )
                        ) > 0.0
                    ]

                return None, active_steps

            availability_profile = {
                t: float(self.availability_profile.get(t, 0.0))
                for t in model_steps
            }

            active_steps = [
                t for t in model_steps
                if availability_profile[t] > 0.0
            ]

            return availability_profile, active_steps

        # Numpy/list profile path.
        availability_slice = self.availability_profile[int(model.start):int(model.stop)]

        if self.skip_availability_param:
            active_steps = [
                t for i, t in enumerate(model_steps)
                if float(availability_slice[i]) > 0.0
            ]
            return None, active_steps

        availability_profile = {
            t: float(availability_slice[i])
            for i, t in enumerate(model_steps)
        }

        active_steps = [
            t for t in model_steps
            if availability_profile[t] > 0.0
        ]

        return availability_profile, active_steps

    def parameters(self, model):
        self._all_steps = list(model.steps)

        # Backward-compatible zero params when not extensible.
        if not self.extensible_energy:
            handle = f"{self.handle}::capex_energy"
            self.handles.append(handle)
            setattr(model, handle, pyomo.Param(initialize=0.0))

        if not self.extensible_power:
            handle = f"{self.handle}::capex_power"
            self.handles.append(handle)
            setattr(model, handle, pyomo.Param(initialize=0.0))

        # Legacy alias.
        handle = f"{self.handle}::capex"
        if handle not in self.handles:
            self.handles.append(handle)
        if not hasattr(model, handle):
            if not self.extensible_energy:
                setattr(model, handle, pyomo.Param(initialize=0.0))

        # Operating cost.
        if np.isscalar(self.operating_cost):
            operating_cost_profile = [self.operating_cost] * len(model.steps)
        else:
            operating_cost_profile = list(
                self.operating_cost[int(model.start):int(model.stop)]
            )

        handle = f"{self.handle}::operating_cost"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Param(model.steps, initialize=operating_cost_profile)
        )

        availability_profile, active_steps = (
            self._resolve_availability_profile_and_active_steps(model)
        )

        self._active_steps = active_steps

        if not self.skip_availability_param:
            handle = f"{self.handle}::availability"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(model.steps, initialize=availability_profile)
            )

        return model

    def variables(self, model):
        active_steps = self._active_steps or []

        # Throughput for variable-cost accounting.
        # For V2G, direct_throughput_cost=True skips this variable and uses
        # charge + discharge directly in the objective.
        if not self.direct_throughput_cost:
            handle = f"{self.handle}::throughput"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(active_steps, initialize=0.0, within=pyomo.NonNegativeReals)
            )

        # Separate charge and discharge variables.
        handle = f"{self.handle}::charge_power"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Var(active_steps, initialize=0.0, within=pyomo.NonNegativeReals)
        )

        handle = f"{self.handle}::discharge_power"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Var(active_steps, initialize=0.0, within=pyomo.NonNegativeReals)
        )

        # Net power: positive = discharge to grid, negative = charge from grid.
        handle = f"{self.handle}::power"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Var(active_steps, initialize=0.0, within=pyomo.Reals)
        )

        handle = f"{self.handle}::level"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Var(active_steps, initialize=0.0, within=pyomo.NonNegativeReals)
        )

        if self.extensible_energy:
            handle = f"{self.handle}::capex_energy"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(
                    initialize=0.0,
                    bounds=(0.0, self.capex_capacity),
                    within=pyomo.NonNegativeReals
                )
            )

        if self.extensible_power:
            handle = f"{self.handle}::capex_power"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(
                    initialize=0.0,
                    bounds=(0.0, self.capex_power_capacity),
                    within=pyomo.NonNegativeReals
                )
            )

        # Legacy alias: capex = capex_energy.
        if self.extensible_energy:
            handle = f"{self.handle}::capex"
            setattr(
                model,
                handle,
                pyomo.Expression(expr=getattr(model, f"{self.handle}::capex_energy"))
            )

        return model

    def constraints(self, model):
        active_steps = self._active_steps or []

        if len(active_steps) == 0:
            return model

        throughput = None
        if not self.direct_throughput_cost:
            throughput = getattr(model, f"{self.handle}::throughput")

        charge = getattr(model, f"{self.handle}::charge_power")
        discharge = getattr(model, f"{self.handle}::discharge_power")
        power = getattr(model, f"{self.handle}::power")
        level = getattr(model, f"{self.handle}::level")

        capex_energy = getattr(model, f"{self.handle}::capex_energy")
        capex_power = getattr(model, f"{self.handle}::capex_power")

        existing_energy = float(self.installed_capacity)
        existing_power = self._existing_power_capacity()

        eta_c = float(self.charge_efficiency)
        eta_d = float(self.discharge_efficiency)

        last_step = active_steps[-1]

        prev_step = {}
        for i, t in enumerate(active_steps):
            prev_step[t] = None if i == 0 else active_steps[i - 1]

        # Net power definition.
        setattr(
            model,
            f"{self.handle}::net_power_definition",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: power[t] == discharge[t] - charge[t]
            )
        )

        # State-of-charge balance with efficiency.
        def level_rule(m, t):
            pt = prev_step[t]

            inflow = charge[t] * eta_c * model.time_step
            outflow = discharge[t] / eta_d * model.time_step

            if pt is None:
                return level[t] == self.initial + inflow - outflow
            return level[t] == level[pt] + inflow - outflow

        setattr(
            model,
            f"{self.handle}::level_constraint",
            pyomo.Constraint(active_steps, rule=level_rule)
        )

        # Final level reset.
        setattr(
            model,
            f"{self.handle}::level_final_constraint",
            pyomo.Constraint(expr=level[last_step] == self.initial)
        )

        # Energy capacity limit.
        setattr(
            model,
            f"{self.handle}::storage_constraint",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: level[t] <= existing_energy + capex_energy
            )
        )

        # Separate power limits.
        setattr(
            model,
            f"{self.handle}::discharge_limit",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: discharge[t] <= existing_power + capex_power
            )
        )

        setattr(
            model,
            f"{self.handle}::charge_limit",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: charge[t] <= existing_power + capex_power
            )
        )

        # Throughput for variable cost.
        # Skip this for V2G when objective uses charge + discharge directly.
        if not self.direct_throughput_cost:
            setattr(
                model,
                f"{self.handle}::throughput_link",
                pyomo.Constraint(
                    active_steps,
                    rule=lambda m, t: throughput[t] >= charge[t] + discharge[t]
                )
            )

        # Optional duration link.
        if self.duration_hours is not None and (self.extensible_energy or self.extensible_power):
            duration_seconds = float(self.duration_hours) * 3600.0
            setattr(
                model,
                f"{self.handle}::duration_link",
                pyomo.Constraint(
                    expr=capex_energy == capex_power * duration_seconds
                )
            )

        return model

    def objective(self, model):
        active_steps = self._active_steps or []

        operating_cost = getattr(model, f"{self.handle}::operating_cost")

        capex_energy = getattr(model, f"{self.handle}::capex_energy")
        capex_power = getattr(model, f"{self.handle}::capex_power")

        existing_energy = float(self.installed_capacity)

        if self.direct_throughput_cost:
            charge = getattr(model, f"{self.handle}::charge_power")
            discharge = getattr(model, f"{self.handle}::discharge_power")

            variable_cost = pyomo.quicksum(
                (charge[t] + discharge[t]) * model.time_step * operating_cost[t]
                for t in active_steps
            )
        else:
            throughput = getattr(model, f"{self.handle}::throughput")

            variable_cost = pyomo.quicksum(
                throughput[t] * model.time_step * operating_cost[t]
                for t in active_steps
            )

        fixed_cost = (
            (existing_energy + capex_energy)
            * self.fixed_operating_cost
            * model.amortization
        )

        expansion_cost_energy = (
            capex_energy * self.capex_cost_energy * model.amortization
        )

        expansion_cost_power = (
            capex_power * self.capex_cost_power * model.amortization
        )

        return variable_cost + fixed_cost + expansion_cost_energy + expansion_cost_power

    def energy(self, model, step=None):
        power = getattr(model, f"{self.handle}::power")

        if step is None:
            return pyomo.quicksum(
                power[t] * model.time_step
                for t in self._active_steps
            )

        if step not in self._active_steps:
            return 0.0

        return power[step] * model.time_step

    def power(self, model, step=None):
        power = getattr(model, f"{self.handle}::power")

        if step is None:
            return pyomo.quicksum(power[t] for t in self._active_steps)

        if step not in self._active_steps:
            return 0.0

        return power[step]

    def energy_capacity(self, model):
        capex_energy = getattr(model, f"{self.handle}::capex_energy")
        return self.installed_capacity + capex_energy

    def power_capacity(self, model):
        capex_power = getattr(model, f"{self.handle}::capex_power")
        return self._existing_power_capacity() + capex_power

    def solution(self, model):
        solution = {}

        all_steps = list(model.steps)
        active_set = set(self._active_steps)

        for handle in self.handles:
            comp = getattr(model, handle)

            if hasattr(comp, "extract_values"):
                values_dict = comp.extract_values()

                full_values = []
                for t in all_steps:
                    if t in values_dict:
                        full_values.append(values_dict[t])
                    else:
                        full_values.append(0.0)

                solution[handle.split("::")[1]] = full_values

        charge_comp = getattr(model, f"{self.handle}::charge_power")
        discharge_comp = getattr(model, f"{self.handle}::discharge_power")
        power_comp = getattr(model, f"{self.handle}::power")

        full_charge = []
        full_discharge = []
        full_power = []

        for t in all_steps:
            if t in active_set:
                full_charge.append(pyomo.value(charge_comp[t]))
                full_discharge.append(pyomo.value(discharge_comp[t]))
                full_power.append(pyomo.value(power_comp[t]))
            else:
                full_charge.append(0.0)
                full_discharge.append(0.0)
                full_power.append(0.0)

        solution["charge"] = full_charge
        solution["discharge"] = full_discharge
        solution["net"] = full_power
        solution["active_mask"] = [1 if t in active_set else 0 for t in all_steps]

        return solution