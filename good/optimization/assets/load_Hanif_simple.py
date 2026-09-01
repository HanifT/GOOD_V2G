import numpy as np

from ..base.asset import Asset
import pyomo.environ as pyomo


class Load(Asset):
    """
    Grid load asset used by GOOD.

    Sign convention
    ---------------
    GOOD uses the same sign convention as the existing Load class:

        load power < 0  means demand
        load power > 0  means supply or negative demand

    For normal non-shiftable loads, the asset contributes:

        profile[t] * (installed_capacity + capex)

    For V1G / shiftable EV load, this version uses a compact cumulative
    formulation instead of an origin-destination shifting matrix.

    The baseline EV load is split into:

        inflexible load + flexible charging requirement

    The optimization chooses only the aggregate flexible charging power in
    each hour. Cumulative arrival/deadline constraints make sure the flexible
    energy is charged within the allowed shift_window.

    This reduces the V1G formulation from approximately:

        hours * shift_window * nodes

    to approximately:

        2 * hours * nodes

    because each node only needs:

        flexible_charge[t]
        cumulative_flexible_charge[t]

    Main new options
    ----------------
    shift_formulation:
        "cumulative"      default for shiftable load. Fast V1G formulation.
        "legacy_window"   keeps the previous signed-shift window formulation.

    shift_fraction / flexible_fraction:
        Fraction of the baseline load that is flexible. Example: 0.30.
        If not supplied and 0 < shift_capacity <= 1, shift_capacity is treated
        as this fraction.

    flexible_power_profile:
        Optional hourly flexible charging requirement [W, positive magnitude].
        If supplied, it overrides shift_fraction-based calculation.

    shift_power_capacity / shift_power_capacity_profile:
        Maximum optimized flexible charging power [W, positive magnitude].
        This should represent connected EV charging availability. For example,
        more vehicles connected at night means a larger value at night.

    shift_window:
        Maximum delay in model steps. For hourly data, 24 means 24 hours.
    """

    def __init__(self, handle, **kwargs):
        super().__init__(handle, **kwargs)

        self.installed_capacity = kwargs.get("installed_capacity", 0.0)
        self.operating_cost = kwargs.get("operating_cost", 0.0)

        # Curtailment flag. Kept for compatibility with the previous class.
        self.curtailable = kwargs.get("curtailable", False)

        # Capacity expansion.
        self.capex_capacity = kwargs.get("capex_capacity", 0.0)
        self.capex_cost = kwargs.get("capex_cost", 0.0)
        self.extensible = self.capex_capacity > 0

        # Shiftable load controls.
        self.shift_capacity = kwargs.get("shift_capacity", 0.0)
        self.shiftable = self.shift_capacity > 0

        self.shift_window = kwargs.get("shift_window", None)
        self.shift_formulation = kwargs.get(
            "shift_formulation",
            "cumulative" if self.shiftable else "none",
        )

        # Hourly load profile. Existing GOOD convention usually uses a
        # positive profile multiplied by a negative installed_capacity for load.
        self.profile = kwargs.get("profile", None)
        if self.profile is not None:
            self.profile = np.asarray(self.profile, dtype=float)

        # Fraction of baseline demand that is flexible.
        self.shift_fraction = kwargs.get(
            "shift_fraction",
            kwargs.get("flexible_fraction", None),
        )

        if self.shift_fraction is None and 0.0 < float(self.shift_capacity) <= 1.0:
            self.shift_fraction = float(self.shift_capacity)

        if self.shift_fraction is None:
            # Backward-compatible default:
            # If shift_capacity is used only as a flag, treat all baseline
            # load as flexible unless a flexible_power_profile is supplied.
            self.shift_fraction = 1.0

        self.shift_fraction = float(self.shift_fraction)

        # Optional direct flexible-energy arrival profile [W, positive].
        # This is the cleanest input for EV V1G:
        # flexible_power_profile[t] = X% of baseline EV demand at hour t.
        self.flexible_power_profile = kwargs.get("flexible_power_profile", None)
        if self.flexible_power_profile is not None:
            self.flexible_power_profile = np.asarray(self.flexible_power_profile, dtype=float)

        # Optional maximum flexible charging power profile [W, positive].
        # This is where connected-vehicle availability enters V1G.
        self.shift_power_capacity = kwargs.get(
            "shift_power_capacity",
            kwargs.get("flex_charge_capacity", None),
        )

        self.shift_power_capacity_profile = kwargs.get(
            "shift_power_capacity_profile",
            kwargs.get("flex_charge_capacity_profile", None),
        )
        if self.shift_power_capacity_profile is not None:
            self.shift_power_capacity_profile = np.asarray(
                self.shift_power_capacity_profile,
                dtype=float,
            )

        self.shift_power_multiplier = float(kwargs.get("shift_power_multiplier", 1.0))

        # Make flexibility optional and cost-safe by default.
        # If True, the no-flexibility baseline schedule is always feasible:
        #     flex_charge[t] = flexible_power[t]
        #     shift[t] = 0
        # This prevents V1G from being forced to change load in a way that can
        # make the scenario worse than the baseline EV-load case.
        self.ensure_baseline_feasible = kwargs.get("ensure_baseline_feasible", True)

        # Fixed availability costs for flexible EV load should not be charged
        # unless the user explicitly wants to price the program itself. When
        # False, V1G is an optional operational capability and can always choose
        # the baseline schedule at no additional cost.
        self.charge_fixed_shift_cost = kwargs.get("charge_fixed_shift_cost", False)

        # Horizon-end balance for flexible charging.
        # For sliced horizons such as 7-day or 30-day model windows, forcing
        # every flexible kWh that arrives near the final hour to be charged
        # before the slice ends creates artificial infeasibility. By default,
        # only energy whose deadline has passed must be charged.
        self.enforce_final_flex_balance = kwargs.get(
            "enforce_final_flex_balance",
            False,
        )

        # Whether the optimized load should remain between zero and nameplate
        # demand. For the cumulative EV formulation this is not needed because
        # flex_charge is nonnegative and limited by shift_power_capacity. Leaving
        # these old bounds on can over-constrain V1G when profiles are sliced or
        # not perfectly normalized.
        self.enforce_physical_shift_bounds = kwargs.get(
            "enforce_physical_shift_bounds",
            False if self.shift_formulation == "cumulative" else True,
        )

        # Shifting cost.
        self.shift_cost = kwargs.get("shift_cost", 0.0)
        if self.shift_cost is not None and not np.isscalar(self.shift_cost):
            self.shift_cost = np.asarray(self.shift_cost, dtype=float)

        # How to apply shift_cost in the cumulative formulation.
        # "none"   no variable shift cost, fastest and recommended for V1G
        # "charge" charges all optimized flexible charging energy
        self.shift_cost_mode = kwargs.get("shift_cost_mode", "none")

        # Fixed cost for having shiftable load available.
        self.fixed_shift_cost = kwargs.get("fixed_shift_cost", 0.0)

        # Internal arrays filled in parameters().
        self._steps = []
        self._profile_slice = None
        self._flexible_power_slice = None
        self._shift_power_capacity_slice = None
        self._cumulative_arrived = None
        self._cumulative_due = None

    # ------------------------------------------------------------------
    # Helper methods for profiles and model-step indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _as_sliced_array(values, model, default_value=0.0):
        """Return an array aligned with list(model.steps)."""
        n = len(model.steps)

        if values is None:
            return np.full(n, float(default_value), dtype=float)

        if np.isscalar(values):
            return np.full(n, float(values), dtype=float)

        arr = np.asarray(values, dtype=float)

        if len(arr) == n:
            return arr.copy()

        # GOOD often uses model.start/model.stop to slice an annual profile.
        start = int(getattr(model, "start", 0))
        stop = int(getattr(model, "stop", start + n))

        if len(arr) >= stop:
            return arr[start:stop].copy()

        raise ValueError(
            f"Profile length {len(arr)} cannot be aligned with model steps "
            f"length {n} and slice [{start}:{stop}]."
        )

    def _build_baseline_power_magnitude(self, model):
        """
        Build positive baseline demand magnitude [W] from profile and installed capacity.

        Existing GOOD load convention commonly uses:
            installed_capacity < 0
            profile[t] >= 0

        so baseline load is negative and the magnitude is:
            -profile[t] * installed_capacity
        """
        profile = self._profile_slice

        baseline_power = profile * float(self.installed_capacity)

        # Positive demand magnitude. Negative values are clipped to zero so the
        # flexible-load logic is not accidentally applied to generation.
        return np.maximum(-baseline_power, 0.0)

    def _build_flexible_power_profile(self, model):
        """
        Flexible charging energy arrival profile [W, positive].

        This is the hourly amount of EV charging energy that is allowed to move
        within the shift window.
        """
        if self.flexible_power_profile is not None:
            flex = self._as_sliced_array(self.flexible_power_profile, model)
            return np.maximum(flex, 0.0)

        baseline_mag = self._build_baseline_power_magnitude(model)

        if float(self.shift_capacity) > 1.0:
            # Backward-compatible interpretation:
            # shift_capacity is an absolute maximum flexible power [W].
            flex = np.minimum(baseline_mag * self.shift_fraction, float(self.shift_capacity))
        else:
            flex = baseline_mag * self.shift_fraction

        return np.maximum(flex, 0.0)

    def _build_shift_power_capacity_profile(self, model):
        """
        Maximum optimized flexible charging power [W, positive].

        For EVs, this should be based on connected vehicles and charger power.
        Example:
            connected_EVs[t] * participation * charger_power_W

        Important:
            If ensure_baseline_feasible=True, the returned pmax is never below
            flexible_power[t]. This guarantees that the optimizer can always
            choose flex_charge[t] = flexible_power[t], which gives shift[t] = 0
            and reproduces the baseline EV-load profile exactly.
        """
        if self.shift_power_capacity_profile is not None:
            pmax = self._as_sliced_array(self.shift_power_capacity_profile, model)
            pmax = np.maximum(pmax, 0.0)

        elif self.shift_power_capacity is not None:
            pmax = np.full(
                len(model.steps),
                max(float(self.shift_power_capacity), 0.0),
                dtype=float,
            )

        else:
            baseline_mag = self._build_baseline_power_magnitude(model)

            # Default: allow flexible charging up to the baseline demand magnitude.
            # If the user wants load moved into hours with low baseline charging,
            # pass shift_power_capacity_profile from connected-EV availability.
            pmax = baseline_mag * self.shift_power_multiplier

            # If shift_capacity is an absolute W value, make sure pmax is at least
            # that value in hours with flexible arrivals. This keeps backward
            # compatibility with old calls that used shift_capacity as a W bound.
            if float(self.shift_capacity) > 1.0:
                pmax = np.maximum(pmax, float(self.shift_capacity))

        # Cost-safe fallback:
        # The no-flexibility baseline schedule requires flex_charge[t] equal to
        # the flexible baseline charging requirement. If a connected-availability
        # profile is lower than that in some hours, the old formulation could
        # force shifting or become infeasible. This line guarantees that doing
        # nothing remains feasible.
        if self.ensure_baseline_feasible and self._flexible_power_slice is not None:
            pmax = np.maximum(pmax, self._flexible_power_slice)

        return np.maximum(pmax, 0.0)

    def _build_cumulative_arrival_deadline_profiles(self, model):
        """
        Build cumulative arrived and cumulative due flexible energy [J].

        arrived[t] is all flexible EV energy that has become available through
        hour t.

        due[t] is all flexible EV energy that must have been charged by hour t.
        For shift_window = 24, energy arriving at hour h must be completed no
        later than h + 24.
        """
        flex = self._flexible_power_slice
        dt = float(model.time_step)
        n = len(flex)

        arrived = np.cumsum(flex * dt)

        if self.shift_window is None:
            window = n
        else:
            window = int(self.shift_window)
        window = max(window, 0)

        due = np.zeros(n, dtype=float)
        for i in range(n):
            due_index = i - window
            if due_index >= 0:
                due[i] = arrived[due_index]
            else:
                due[i] = 0.0

        return arrived, due

    @staticmethod
    def _param_dict(steps, values):
        return {t: float(v) for t, v in zip(list(steps), list(values))}

    def _scalar_component_value(self, component):
        try:
            return float(pyomo.value(component))
        except Exception:
            values = component.extract_values()
            if None in values:
                return float(values[None])
            return float(next(iter(values.values())))

    def _indexed_component_values(self, component, steps):
        values = []
        for t in steps:
            try:
                values.append(float(pyomo.value(component[t])))
            except Exception:
                values_dict = component.extract_values()
                values.append(float(values_dict.get(t, 0.0)))
        return values

    # ------------------------------------------------------------------
    # Pyomo construction methods
    # ------------------------------------------------------------------

    def parameters(self, model):
        self._steps = list(model.steps)

        if self.shift_window is None:
            self.shift_window = len(self._steps)

        self._profile_slice = self._as_sliced_array(self.profile, model, default_value=0.0)

        handle = f"{self.handle}::profile"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Param(
                model.steps,
                initialize=self._param_dict(model.steps, self._profile_slice),
            ),
        )

        if not self.extensible:
            handle = f"{self.handle}::capex"
            self.handles.append(handle)
            setattr(model, handle, pyomo.Param(initialize=0.0))

        if not self.shiftable:
            handle = f"{self.handle}::shift"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(
                    model.steps,
                    initialize=self._param_dict(model.steps, np.zeros(len(self._steps))),
                ),
            )

        if np.isscalar(self.shift_cost):
            shift_cost_profile = np.full(len(self._steps), float(self.shift_cost), dtype=float)
        else:
            shift_cost_profile = self._as_sliced_array(self.shift_cost, model)

        handle = f"{self.handle}::shift_cost"
        self.handles.append(handle)
        setattr(
            model,
            handle,
            pyomo.Param(
                model.steps,
                initialize=self._param_dict(model.steps, shift_cost_profile),
            ),
        )

        if self.shiftable and self.shift_formulation == "cumulative":
            self._flexible_power_slice = self._build_flexible_power_profile(model)
            self._shift_power_capacity_slice = self._build_shift_power_capacity_profile(model)
            self._cumulative_arrived, self._cumulative_due = (
                self._build_cumulative_arrival_deadline_profiles(model)
            )

            handle = f"{self.handle}::flexible_power"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(
                    model.steps,
                    initialize=self._param_dict(model.steps, self._flexible_power_slice),
                ),
            )

            handle = f"{self.handle}::shift_power_capacity"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(
                    model.steps,
                    initialize=self._param_dict(model.steps, self._shift_power_capacity_slice),
                ),
            )

            handle = f"{self.handle}::cumulative_arrived"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(
                    model.steps,
                    initialize=self._param_dict(model.steps, self._cumulative_arrived),
                ),
            )

            handle = f"{self.handle}::cumulative_due"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Param(
                    model.steps,
                    initialize=self._param_dict(model.steps, self._cumulative_due),
                ),
            )

        return model

    def variables(self, model):
        if self.extensible:
            handle = f"{self.handle}::capex"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(
                    initialize=0.0,
                    bounds=(0.0, self.capex_capacity),
                    within=pyomo.NonNegativeReals,
                ),
            )

        if self.shiftable and self.shift_formulation == "legacy_window":
            # Previous signed-shift formulation. Kept only for compatibility.
            handle = f"{self.handle}::shift"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(model.steps, initialize=0.0, within=pyomo.Reals),
            )

            handle_up = f"{self.handle}::shift_up"
            self.handles.append(handle_up)
            setattr(
                model,
                handle_up,
                pyomo.Var(model.steps, initialize=0.0, within=pyomo.NonNegativeReals),
            )

            handle_dn = f"{self.handle}::shift_down"
            self.handles.append(handle_dn)
            setattr(
                model,
                handle_dn,
                pyomo.Var(model.steps, initialize=0.0, within=pyomo.NonNegativeReals),
            )

        elif self.shiftable and self.shift_formulation == "cumulative":
            # Optimized aggregate flexible charging power [W, positive demand magnitude].
            handle = f"{self.handle}::flex_charge"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(model.steps, initialize=0.0, within=pyomo.NonNegativeReals),
            )

            # Cumulative flexible energy charged [J].
            handle = f"{self.handle}::cumulative_flex_charge"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Var(model.steps, initialize=0.0, within=pyomo.NonNegativeReals),
            )

            # Keep the old interface:
            # shift[t] is added to baseline load in energy()/power().
            #
            # baseline load includes the flexible baseline charging.
            # actual load = baseline load + flexible_baseline - optimized_charge
            #
            # Therefore:
            # shift[t] = flexible_baseline[t] - flex_charge[t]
            flexible_power = getattr(model, f"{self.handle}::flexible_power")
            flex_charge = getattr(model, f"{self.handle}::flex_charge")

            handle = f"{self.handle}::shift"
            self.handles.append(handle)
            setattr(
                model,
                handle,
                pyomo.Expression(
                    model.steps,
                    rule=lambda m, t: flexible_power[t] - flex_charge[t],
                ),
            )

        return model

    def constraints(self, model):
        if not self.shiftable:
            return model

        if self.shift_formulation == "legacy_window":
            return self._legacy_window_constraints(model)

        if self.shift_formulation != "cumulative":
            raise ValueError(
                f"{self.handle}: unknown shift_formulation={self.shift_formulation!r}. "
                "Use 'cumulative' or 'legacy_window'."
            )

        return self._cumulative_shift_constraints(model)

    def _legacy_window_constraints(self, model):
        shift = getattr(model, f"{self.handle}::shift")
        shift_up = getattr(model, f"{self.handle}::shift_up")
        shift_down = getattr(model, f"{self.handle}::shift_down")
        profile = getattr(model, f"{self.handle}::profile")
        capex = getattr(model, f"{self.handle}::capex")

        setattr(
            model,
            f"{self.handle}::shift_link",
            pyomo.Constraint(
                model.steps,
                rule=lambda m, t: shift[t] == shift_up[t] - shift_down[t],
            ),
        )

        def shift_mag_rule(m, t):
            # Existing GOOD convention: demand is negative.
            # This avoids nonlinear abs(profile[t] * capacity).
            base_power_magnitude = -profile[t] * (self.installed_capacity + capex)

            if float(self.shift_capacity) > 1.0:
                return shift_up[t] + shift_down[t] <= float(self.shift_capacity)

            return shift_up[t] + shift_down[t] <= self.shift_fraction * base_power_magnitude

        setattr(
            model,
            f"{self.handle}::shift_mag_limit",
            pyomo.Constraint(model.steps, rule=shift_mag_rule),
        )

        steps_list = list(model.steps)
        window = int(self.shift_window) if self.shift_window is not None else len(steps_list)
        window = max(window, 1)

        for i_start in range(0, len(steps_list), window):
            window_steps = steps_list[i_start:i_start + window]
            shift_sum = pyomo.quicksum(shift[t] for t in window_steps)

            setattr(
                model,
                f"{self.handle}::shift_sum_constraint_{i_start}",
                pyomo.Constraint(expr=(0.0, shift_sum, 0.0)),
            )

        if self.enforce_physical_shift_bounds:
            setattr(
                model,
                f"{self.handle}::shift_lower_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: (
                        profile[t] * (self.installed_capacity + capex) + shift[t]
                        >= self.installed_capacity + capex
                    ),
                ),
            )

            setattr(
                model,
                f"{self.handle}::shift_upper_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: (
                        profile[t] * (self.installed_capacity + capex) + shift[t]
                        <= 0.0
                    ),
                ),
            )

        return model

    def _cumulative_shift_constraints(self, model):
        steps = list(model.steps)

        profile = getattr(model, f"{self.handle}::profile")
        capex = getattr(model, f"{self.handle}::capex")
        flexible_power = getattr(model, f"{self.handle}::flexible_power")
        shift_power_capacity = getattr(model, f"{self.handle}::shift_power_capacity")
        cumulative_arrived = getattr(model, f"{self.handle}::cumulative_arrived")
        cumulative_due = getattr(model, f"{self.handle}::cumulative_due")
        flex_charge = getattr(model, f"{self.handle}::flex_charge")
        cumulative_flex_charge = getattr(model, f"{self.handle}::cumulative_flex_charge")
        shift = getattr(model, f"{self.handle}::shift")

        prev_step = {}
        for i, t in enumerate(steps):
            prev_step[t] = None if i == 0 else steps[i - 1]

        def cumulative_balance_rule(m, t):
            pt = prev_step[t]
            charged_energy = flex_charge[t] * model.time_step

            if pt is None:
                return cumulative_flex_charge[t] == charged_energy

            return cumulative_flex_charge[t] == cumulative_flex_charge[pt] + charged_energy

        setattr(
            model,
            f"{self.handle}::cumulative_flex_charge_balance",
            pyomo.Constraint(model.steps, rule=cumulative_balance_rule),
        )

        # Cannot charge flexible EV energy before it has arrived.
        setattr(
            model,
            f"{self.handle}::cumulative_flex_charge_arrived_limit",
            pyomo.Constraint(
                model.steps,
                rule=lambda m, t: cumulative_flex_charge[t] <= cumulative_arrived[t],
            ),
        )

        # Must charge flexible EV energy by its deadline.
        setattr(
            model,
            f"{self.handle}::cumulative_flex_charge_due_limit",
            pyomo.Constraint(
                model.steps,
                rule=lambda m, t: cumulative_flex_charge[t] >= cumulative_due[t],
            ),
        )

        # End-of-horizon treatment.
        #
        # For full-year runs, the user may require all flexible energy that
        # arrived inside the modeled horizon to be served by the final step.
        # For 7-day, 30-day, or seasonal slices, that strict equality creates a
        # fake boundary problem: EV energy that arrives near the final hour
        # would normally have up to shift_window hours after the slice to charge.
        #
        # The default therefore enforces only cumulative_due at the final step.
        # This keeps the no-shift baseline feasible and prevents V1G from
        # causing artificial infeasibility.
        last_step = steps[-1]
        if self.enforce_final_flex_balance:
            final_expr = cumulative_flex_charge[last_step] == cumulative_arrived[last_step]
        else:
            final_expr = cumulative_flex_charge[last_step] >= cumulative_due[last_step]

        setattr(
            model,
            f"{self.handle}::cumulative_flex_charge_final",
            pyomo.Constraint(expr=final_expr),
        )

        # Hourly connected-EV charging power limit.
        setattr(
            model,
            f"{self.handle}::flex_charge_power_limit",
            pyomo.Constraint(
                model.steps,
                rule=lambda m, t: flex_charge[t] <= shift_power_capacity[t],
            ),
        )

        if self.enforce_physical_shift_bounds:
            # Keep the realized load between zero and installed load capacity.
            #
            # actual_load = profile[t] * capacity + shift[t]
            #
            # Because demand is negative, this says:
            #     installed_capacity <= actual_load <= 0
            setattr(
                model,
                f"{self.handle}::shift_lower_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: (
                        profile[t] * (self.installed_capacity + capex) + shift[t]
                        >= self.installed_capacity + capex
                    ),
                ),
            )

            setattr(
                model,
                f"{self.handle}::shift_upper_bound",
                pyomo.Constraint(
                    model.steps,
                    rule=lambda m, t: (
                        profile[t] * (self.installed_capacity + capex) + shift[t]
                        <= 0.0
                    ),
                ),
            )

        return model

    def energy(self, model, step=None):
        profile = getattr(model, f"{self.handle}::profile")
        shift = getattr(model, f"{self.handle}::shift")
        capex = getattr(model, f"{self.handle}::capex")

        capacity = self.installed_capacity + capex

        if step is None:
            return pyomo.quicksum(
                (profile[t] * capacity + shift[t]) * model.time_step
                for t in model.steps
            )

        return (profile[step] * capacity + shift[step]) * model.time_step

    def power(self, model, step=None):
        profile = getattr(model, f"{self.handle}::profile")
        shift = getattr(model, f"{self.handle}::shift")
        capex = getattr(model, f"{self.handle}::capex")

        capacity = self.installed_capacity + capex

        if step is None:
            return pyomo.quicksum(
                profile[t] * capacity + shift[t]
                for t in model.steps
            )

        return profile[step] * capacity + shift[step]

    def capacity(self, model, step=None):
        capex = getattr(model, f"{self.handle}::capex")
        return self.installed_capacity + capex

    def objective(self, model):
        capex = getattr(model, f"{self.handle}::capex")
        expansion_cost = capex * self.capex_cost * model.amortization

        cost = expansion_cost

        if self.shiftable:
            shift_cost = getattr(model, f"{self.handle}::shift_cost")

            if self.shift_formulation == "legacy_window":
                shift_down = getattr(model, f"{self.handle}::shift_down")
                cost += pyomo.quicksum(
                    shift_down[t] * model.time_step * shift_cost[t]
                    for t in model.steps
                )

            elif self.shift_formulation == "cumulative" and self.shift_cost_mode == "charge":
                flex_charge = getattr(model, f"{self.handle}::flex_charge")
                cost += pyomo.quicksum(
                    flex_charge[t] * model.time_step * shift_cost[t]
                    for t in model.steps
                )

            if self.charge_fixed_shift_cost:
                cost += self.shift_capacity * self.fixed_shift_cost * model.amortization

        return cost

    def solution(self, model):
        solution = {}
        steps = list(model.steps)

        for handle in self.handles:
            comp = getattr(model, handle)

            if comp.is_indexed():
                solution[handle.split("::")[1]] = self._indexed_component_values(comp, steps)
            else:
                solution[handle.split("::")[1]] = [self._scalar_component_value(comp)]

        profile_comp = getattr(model, f"{self.handle}::profile")
        shift_comp = getattr(model, f"{self.handle}::shift")
        capex_comp = getattr(model, f"{self.handle}::capex")

        profile = self._indexed_component_values(profile_comp, steps)
        shift = self._indexed_component_values(shift_comp, steps)
        capex_value = self._scalar_component_value(capex_comp)
        capacity = float(self.installed_capacity) + capex_value

        solution["net"] = [
            profile[i] * capacity + shift[i]
            for i in range(len(steps))
        ]

        if self.shiftable and self.shift_formulation == "cumulative":
            flex_charge_comp = getattr(model, f"{self.handle}::flex_charge")
            flexible_power_comp = getattr(model, f"{self.handle}::flexible_power")
            cumulative_comp = getattr(model, f"{self.handle}::cumulative_flex_charge")

            solution["optimized_flexible_charge"] = self._indexed_component_values(
                flex_charge_comp,
                steps,
            )
            solution["baseline_flexible_charge"] = self._indexed_component_values(
                flexible_power_comp,
                steps,
            )
            solution["cumulative_flexible_charge"] = self._indexed_component_values(
                cumulative_comp,
                steps,
            )

        return solution
