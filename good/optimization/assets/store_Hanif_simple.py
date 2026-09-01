from ..base.asset import Asset
import pyomo.environ as pyomo
import numpy as np
import math


class Store(Asset):
    """
    Storage asset used by GOOD.

    This version keeps the existing GOOD Store interface, but adds a compact
    aggregate-EV-battery formulation for V2G.

    Main idea for EV V2G:
        Use one Store per region or node, not one Store per EV charging window.
        The Store can now have hourly energy and power availability limits, so
        the aggregate EV battery is large when many vehicles are plugged in and
        small when fewer vehicles are connected.

    New optional inputs:
        energy_capacity_profile:
            Hourly available energy capacity [J]. This is an absolute bound.
            Example: connected_EV_count[t] * usable_kWh * 3.6e6.

        power_capacity_profile:
            Hourly available charge and discharge power capacity [W].
            Used for both charge and discharge if separate profiles are not
            provided.

        charge_power_capacity_profile:
            Hourly available charge power capacity [W].

        discharge_power_capacity_profile:
            Hourly available discharge power capacity [W].

        capacity_availability_profile:
            Hourly multiplier between 0 and 1. If this is supplied, and no
            absolute energy/power profile is supplied, it scales the installed
            energy and power capacity. This is useful when you know the total EV
            fleet battery size and an hourly connected-vehicle fraction.

        availability_scales_capacity:
            If True, availability_profile is also used as the capacity
            availability multiplier. Defaults to True for type == "ev_v2g".

        minimum_level_profile:
            Optional hourly minimum storage level [J].

        reset_interval_hours:
            Optional reset interval. For example, 24 forces the aggregate EV
            battery level back to initial at the end of each day. This prevents
            the model from using EVs as seasonal storage.

        reset_steps:
            Optional explicit list of model steps where level[t] == initial.

    Existing behavior is preserved when these new profiles are not supplied.
    For type == "ev_v2g", the fast path still skips the throughput variable by
    default and can skip inactive time steps.
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

        # Make EV flexibility optional and cost-safe by default.
        # For type == "ev_v2g", fixed availability cost is not charged unless
        # explicitly requested. This means charge=discharge=0 is a cost-neutral
        # feasible option, so V2G cannot make the model worse than the same EV
        # load without V2G. Normal stationary storage keeps the old behavior.
        self.charge_fixed_operating_cost = kwargs.get(
            "charge_fixed_operating_cost",
            self.type != "ev_v2g",
        )

        # Linear relaxation to reduce simultaneous charge/discharge in aggregate
        # EV batteries. Exact no-simultaneity would require binaries, but this
        # keeps total charge+discharge within one power budget for V2G.
        self.combined_charge_discharge_limit = kwargs.get(
            "combined_charge_discharge_limit",
            self.type == "ev_v2g",
        )

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

        # Efficiencies.
        # If only round-trip efficiency is given, split it symmetrically.
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

        # Kept for compatibility with older runner code.
        self.reset_each_window = kwargs.get(
            "reset_each_window",
            self.type == "ev_v2g"
        )

        # Optional reset controls for aggregate EV batteries.
        self.enforce_final_level = kwargs.get("enforce_final_level", True)
        self.reset_interval_hours = kwargs.get("reset_interval_hours", None)
        self.reset_steps_input = kwargs.get("reset_steps", None)

        # Expansion bounds.
        self.capex_capacity = kwargs.get("capex_capacity", 0.0)  # energy bound [J]
        self.capex_power_capacity = kwargs.get("capex_power_capacity", None)  # power bound [W]

        # Costs.
        self.capex_cost = kwargs.get("capex_cost", 0.0)  # legacy energy cost [$ / J]
        self.capex_cost_energy = kwargs.get("capex_cost_energy", self.capex_cost)
        self.capex_cost_power = kwargs.get("capex_cost_power", 0.0)

        # Optional fixed duration [hours].
        self.duration_hours = kwargs.get("duration_hours", None)

        # Hourly aggregate-EV-battery limits.
        self.energy_capacity_profile = self._clean_profile(
            kwargs.get("energy_capacity_profile", None)
        )

        self.power_capacity_profile = self._clean_profile(
            kwargs.get("power_capacity_profile", None)
        )

        self.charge_power_capacity_profile = self._clean_profile(
            kwargs.get("charge_power_capacity_profile", None)
        )

        self.discharge_power_capacity_profile = self._clean_profile(
            kwargs.get("discharge_power_capacity_profile", None)
        )

        self.minimum_level_profile = self._clean_profile(
            kwargs.get("minimum_level_profile", None)
        )

        # Optional multiplier for hourly connected EV share.
        self.availability_scales_capacity = kwargs.get(
            "availability_scales_capacity",
            self.type == "ev_v2g"
        )

        self.capacity_availability_profile = self._clean_profile(
            kwargs.get(
                "capacity_availability_profile",
                kwargs.get("availability_multiplier_profile", None)
            )
        )

        if (
            self.capacity_availability_profile is None
            and self.availability_scales_capacity
            and self.availability_profile is not None
        ):
            self.capacity_availability_profile = self.availability_profile

        # Extensibility flags.
        self.extensible_energy = self.capex_capacity > 0
        self.extensible_power = (
            self.capex_power_capacity is not None and self.capex_power_capacity > 0
        )
        self.extensible = self.extensible_energy or self.extensible_power

        # Filled later.
        self._all_steps = []
        self._active_steps = []
        self._reset_steps = []

        self._has_energy_capacity_profile = False
        self._has_power_capacity_profile = False
        self._has_charge_power_capacity_profile = False
        self._has_discharge_power_capacity_profile = False
        self._has_capacity_availability_profile = False
        self._has_minimum_level_profile = False

    @staticmethod
    def _clean_profile(profile):
        if profile is None:
            return None
        if isinstance(profile, dict):
            return profile
        if np.isscalar(profile):
            return profile
        return np.array(profile)

    def _existing_power_capacity(self):
        if self.installed_power_capacity is not None:
            return float(self.installed_power_capacity)
        return float(self.installed_capacity) * float(self.production_rate)

    def _profile_to_dict(self, profile, model, default=0.0, name="profile"):
        """
        Convert scalar, dict, list, or numpy profile to a dict keyed by model.steps.

        GOOD sometimes uses local model steps such as 0..167 and sometimes
        absolute steps such as 5376..5543. This helper accepts either full-year
        profiles indexed by absolute step or already-sliced profiles whose
        length matches len(model.steps).
        """
        model_steps = list(model.steps)

        if profile is None:
            return {t: float(default) for t in model_steps}

        if np.isscalar(profile):
            return {t: float(profile) for t in model_steps}

        model_start = int(getattr(model, "start", 0))
        model_stop = int(getattr(model, "stop", model_start + len(model_steps)))

        if isinstance(profile, dict):
            out = {}
            for t in model_steps:
                if t in profile:
                    out[t] = float(profile[t])
                    continue

                local_t = int(t) - model_start
                if local_t in profile:
                    out[t] = float(profile[local_t])
                    continue

                out[t] = float(default)
            return out

        arr = np.asarray(profile, dtype=float)

        if len(arr) == len(model_steps):
            return {
                t: float(arr[i])
                for i, t in enumerate(model_steps)
            }

        if model_stop <= len(arr):
            sliced = arr[model_start:model_stop]
            if len(sliced) == len(model_steps):
                return {
                    t: float(sliced[i])
                    for i, t in enumerate(model_steps)
                }

        raise ValueError(
            f"{self.handle}: {name} length {len(arr)} cannot be aligned with "
            f"model steps start={model_start}, stop={model_stop}, "
            f"len(model.steps)={len(model_steps)}."
        )

    def _make_param(self, model, short_name, values):
        handle = f"{self.handle}::{short_name}"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Param(model.steps, initialize=values)
        )
        return getattr(model, handle)

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

        Note:
            For aggregate EV batteries, capacity_availability_profile controls
            hourly storage size. availability_profile still controls whether a
            time step is active.
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

            availability_profile = self._profile_to_dict(
                self.availability_profile,
                model,
                default=0.0,
                name="availability_profile"
            )

            active_steps = [
                t for t in model_steps
                if availability_profile[t] > 0.0
            ]

            return availability_profile, active_steps

        # Numpy/list profile path.
        availability_profile = self._profile_to_dict(
            self.availability_profile,
            model,
            default=0.0,
            name="availability_profile"
        )

        if self.skip_availability_param:
            active_steps = [
                t for t in model_steps
                if availability_profile[t] > 0.0
            ]
            return None, active_steps

        active_steps = [
            t for t in model_steps
            if availability_profile[t] > 0.0
        ]

        return availability_profile, active_steps

    def _resolve_reset_steps(self, model, active_steps):
        active_set = set(active_steps)
        reset_steps = set()

        if self.reset_steps_input is not None:
            raw_reset_steps = [int(t) for t in self.reset_steps_input]
            model_start = int(getattr(model, "start", 0))

            for t in raw_reset_steps:
                if t in active_set:
                    reset_steps.add(t)
                elif model_start + t in active_set:
                    reset_steps.add(model_start + t)

        if self.reset_interval_hours is not None:
            dt = float(pyomo.value(model.time_step))
            interval_seconds = float(self.reset_interval_hours) * 3600.0
            interval_steps = int(round(interval_seconds / dt))

            if interval_steps <= 0:
                raise ValueError(
                    f"{self.handle}: reset_interval_hours produces "
                    f"invalid interval_steps={interval_steps}."
                )

            model_start = int(getattr(model, "start", active_steps[0]))

            for t in active_steps:
                local_index = int(t) - model_start
                if local_index >= 0 and (local_index + 1) % interval_steps == 0:
                    reset_steps.add(t)

        return sorted(reset_steps)

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
        operating_cost_profile = self._profile_to_dict(
            self.operating_cost,
            model,
            default=0.0,
            name="operating_cost"
        )

        self._make_param(model, "operating_cost", operating_cost_profile)

        availability_profile, active_steps = (
            self._resolve_availability_profile_and_active_steps(model)
        )

        self._active_steps = active_steps
        self._reset_steps = (
            self._resolve_reset_steps(model, active_steps)
            if len(active_steps) > 0
            else []
        )

        if not self.skip_availability_param:
            handle = f"{self.handle}::availability"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(model.steps, initialize=availability_profile)
            )

        # Hourly available energy capacity [J].
        self._has_energy_capacity_profile = self.energy_capacity_profile is not None
        if self._has_energy_capacity_profile:
            values = self._profile_to_dict(
                self.energy_capacity_profile,
                model,
                default=0.0,
                name="energy_capacity_profile"
            )
            self._make_param(model, "energy_capacity_available", values)

        # Hourly available power capacity [W], used for both charge and discharge
        # unless a separate profile is provided.
        self._has_power_capacity_profile = self.power_capacity_profile is not None
        if self._has_power_capacity_profile:
            values = self._profile_to_dict(
                self.power_capacity_profile,
                model,
                default=0.0,
                name="power_capacity_profile"
            )
            self._make_param(model, "power_capacity_available", values)

        self._has_charge_power_capacity_profile = (
            self.charge_power_capacity_profile is not None
        )
        if self._has_charge_power_capacity_profile:
            values = self._profile_to_dict(
                self.charge_power_capacity_profile,
                model,
                default=0.0,
                name="charge_power_capacity_profile"
            )
            self._make_param(model, "charge_power_capacity_available", values)

        self._has_discharge_power_capacity_profile = (
            self.discharge_power_capacity_profile is not None
        )
        if self._has_discharge_power_capacity_profile:
            values = self._profile_to_dict(
                self.discharge_power_capacity_profile,
                model,
                default=0.0,
                name="discharge_power_capacity_profile"
            )
            self._make_param(model, "discharge_power_capacity_available", values)

        # Hourly connected-fleet multiplier. This is only used when the
        # corresponding absolute capacity profile is not provided.
        self._has_capacity_availability_profile = (
            self.capacity_availability_profile is not None
        )
        if self._has_capacity_availability_profile:
            values = self._profile_to_dict(
                self.capacity_availability_profile,
                model,
                default=0.0,
                name="capacity_availability_profile"
            )
            self._make_param(model, "capacity_availability", values)

        self._has_minimum_level_profile = self.minimum_level_profile is not None
        if self._has_minimum_level_profile:
            values = self._profile_to_dict(
                self.minimum_level_profile,
                model,
                default=0.0,
                name="minimum_level_profile"
            )
            self._make_param(model, "level_minimum", values)

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

    def _energy_capacity_expr(self, model, t, existing_energy, capex_energy):
        if self._has_energy_capacity_profile:
            energy_capacity_available = getattr(
                model,
                f"{self.handle}::energy_capacity_available"
            )
            return energy_capacity_available[t] + capex_energy

        if self._has_capacity_availability_profile:
            capacity_availability = getattr(
                model,
                f"{self.handle}::capacity_availability"
            )
            return (existing_energy + capex_energy) * capacity_availability[t]

        return existing_energy + capex_energy

    def _charge_power_capacity_expr(self, model, t, existing_power, capex_power):
        if self._has_charge_power_capacity_profile:
            charge_power_capacity_available = getattr(
                model,
                f"{self.handle}::charge_power_capacity_available"
            )
            return charge_power_capacity_available[t] + capex_power

        if self._has_power_capacity_profile:
            power_capacity_available = getattr(
                model,
                f"{self.handle}::power_capacity_available"
            )
            return power_capacity_available[t] + capex_power

        if self._has_capacity_availability_profile:
            capacity_availability = getattr(
                model,
                f"{self.handle}::capacity_availability"
            )
            return (existing_power + capex_power) * capacity_availability[t]

        return existing_power + capex_power

    def _discharge_power_capacity_expr(self, model, t, existing_power, capex_power):
        if self._has_discharge_power_capacity_profile:
            discharge_power_capacity_available = getattr(
                model,
                f"{self.handle}::discharge_power_capacity_available"
            )
            return discharge_power_capacity_available[t] + capex_power

        if self._has_power_capacity_profile:
            power_capacity_available = getattr(
                model,
                f"{self.handle}::power_capacity_available"
            )
            return power_capacity_available[t] + capex_power

        if self._has_capacity_availability_profile:
            capacity_availability = getattr(
                model,
                f"{self.handle}::capacity_availability"
            )
            return (existing_power + capex_power) * capacity_availability[t]

        return existing_power + capex_power

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

        # Optional daily, weekly, or user-defined resets.
        reset_steps = list(self._reset_steps or [])
        if len(reset_steps) > 0:
            setattr(
                model,
                f"{self.handle}::level_periodic_reset_constraint",
                pyomo.Constraint(
                    reset_steps,
                    rule=lambda m, t: level[t] == self.initial
                )
            )

        # Final level reset. This keeps old Store behavior by default.
        if self.enforce_final_level and last_step not in set(reset_steps):
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
                rule=lambda m, t: level[t] <= self._energy_capacity_expr(
                    m, t, existing_energy, capex_energy
                )
            )
        )

        # Optional minimum level.
        if self._has_minimum_level_profile:
            level_minimum = getattr(model, f"{self.handle}::level_minimum")
            setattr(
                model,
                f"{self.handle}::minimum_level_constraint",
                pyomo.Constraint(
                    active_steps,
                    rule=lambda m, t: level[t] >= level_minimum[t]
                )
            )

        # Separate power limits.
        setattr(
            model,
            f"{self.handle}::discharge_limit",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: discharge[t] <= self._discharge_power_capacity_expr(
                    m, t, existing_power, capex_power
                )
            )
        )

        setattr(
            model,
            f"{self.handle}::charge_limit",
            pyomo.Constraint(
                active_steps,
                rule=lambda m, t: charge[t] <= self._charge_power_capacity_expr(
                    m, t, existing_power, capex_power
                )
            )
        )

        # Optional combined power-budget limits.
        # These are mainly for aggregate EV batteries. They do not add binaries,
        # but they prevent the model from using the full charge limit and full
        # discharge limit at the same time.
        if self.combined_charge_discharge_limit:
            setattr(
                model,
                f"{self.handle}::combined_charge_power_limit",
                pyomo.Constraint(
                    active_steps,
                    rule=lambda m, t: charge[t] + discharge[t]
                    <= self._charge_power_capacity_expr(
                        m, t, existing_power, capex_power
                    )
                )
            )

            setattr(
                model,
                f"{self.handle}::combined_discharge_power_limit",
                pyomo.Constraint(
                    active_steps,
                    rule=lambda m, t: charge[t] + discharge[t]
                    <= self._discharge_power_capacity_expr(
                        m, t, existing_power, capex_power
                    )
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

        if self.charge_fixed_operating_cost:
            fixed_cost = (
                (existing_energy + capex_energy)
                * self.fixed_operating_cost
                * model.amortization
            )
        else:
            fixed_cost = 0.0

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

    def energy_capacity(self, model, step=None):
        capex_energy = getattr(model, f"{self.handle}::capex_energy")

        if step is not None and self._has_energy_capacity_profile:
            energy_capacity_available = getattr(
                model,
                f"{self.handle}::energy_capacity_available"
            )
            return energy_capacity_available[step] + capex_energy

        if step is not None and self._has_capacity_availability_profile:
            capacity_availability = getattr(
                model,
                f"{self.handle}::capacity_availability"
            )
            return (self.installed_capacity + capex_energy) * capacity_availability[step]

        return self.installed_capacity + capex_energy

    def power_capacity(self, model, step=None):
        capex_power = getattr(model, f"{self.handle}::capex_power")
        existing_power = self._existing_power_capacity()

        if step is not None and self._has_power_capacity_profile:
            power_capacity_available = getattr(
                model,
                f"{self.handle}::power_capacity_available"
            )
            return power_capacity_available[step] + capex_power

        if step is not None and self._has_capacity_availability_profile:
            capacity_availability = getattr(
                model,
                f"{self.handle}::capacity_availability"
            )
            return (existing_power + capex_power) * capacity_availability[step]

        return existing_power + capex_power

    def capacity(self, model, step=None):
        # Legacy alias used by older GOOD code.
        return self.energy_capacity(model, step=step)

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
                        full_values.append(pyomo.value(values_dict[t]))
                    else:
                        full_values.append(0.0)

                solution[handle.split("::")[1]] = full_values

        charge_comp = getattr(model, f"{self.handle}::charge_power")
        discharge_comp = getattr(model, f"{self.handle}::discharge_power")
        power_comp = getattr(model, f"{self.handle}::power")
        level_comp = getattr(model, f"{self.handle}::level")

        full_charge = []
        full_discharge = []
        full_power = []
        full_level = []

        for t in all_steps:
            if t in active_set:
                full_charge.append(pyomo.value(charge_comp[t]))
                full_discharge.append(pyomo.value(discharge_comp[t]))
                full_power.append(pyomo.value(power_comp[t]))
                full_level.append(pyomo.value(level_comp[t]))
            else:
                full_charge.append(0.0)
                full_discharge.append(0.0)
                full_power.append(0.0)
                full_level.append(0.0)

        solution["charge"] = full_charge
        solution["discharge"] = full_discharge
        solution["net"] = full_power
        solution["level_full"] = full_level
        solution["active_mask"] = [1 if t in active_set else 0 for t in all_steps]

        return solution
