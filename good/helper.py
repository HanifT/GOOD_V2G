import numpy as np
import json
import copy
from pathlib import Path


def enhance_transmission_with_distance(graph):
    """
    Add distance-based transmission efficiency and costs.
    Also add small distance penalty to renewable capex for tie-breaking.
    """
    from good.utilities import haversine
    import numpy as np

    # California load centers (for renewable plant distance calculations)
    LOAD_CENTERS = {
        "NY_Z_J": (-74.0060, 40.7128),  # NYC
        "NY_Z_K": (-73.7949, 40.7282),  # Long Island
        "NY_Z_G-I": (-73.7562, 42.6526),  # Albany
        "NY_Z_C&E": (-75.1652, 43.0481),  # Central NY
        "NY_Z_B": (-78.8784, 42.8864),  # Buffalo
        "NY_Z_D": (-76.1474, 43.0481),  # Syracuse-ish
        "NY_Z_F": (-77.6109, 43.1566),  # Rochester
        "NY_Z_A": (-79.7624, 42.1292),  # Far west
    }

    # Get node coordinates
    NODE_COORDS = {}
    for node_id, node_data in graph._node.items():
        NODE_COORDS[node_id] = (
            node_data.get('x', LOAD_CENTERS.get(node_id, (-119, 36))[0]),
            node_data.get('y', LOAD_CENTERS.get(node_id, (-119, 36))[1])
        )

    # print("\n" + "="*80)
    # print("ENHANCING TRANSMISSION WITH DISTANCE-BASED PARAMETERS")
    # print("="*80)

    # 1. Add efficiency and enhanced costs to transmission lines
    for source, _adj in graph._adj.items():
        for target, edge in _adj.items():

            source_coords = NODE_COORDS.get(source, (-119, 36))
            target_coords = NODE_COORDS.get(target, (-119, 36))

            distance_km = haversine(
                source_coords[0], source_coords[1],
                target_coords[0], target_coords[1]
            ) / 1000

            for line_key, line in edge.get('lines', {}).items():
                # Add transmission losses: ~0.5% per 100km
                efficiency = max(0.90, 1.0 - (distance_km / 100) * 0.005)
                line['efficiency'] = efficiency

                # Scale operating cost with distance
                base_cost = line.get('operating_cost', 2.22e-9)
                distance_multiplier = 1.0 + (distance_km / 500)
                line['operating_cost'] = base_cost * distance_multiplier

    #             print(f"\n{line_key}: {source} → {target}")
    #             print(f"  Distance: {distance_km:.1f} km")
    #             print(f"  Efficiency: {efficiency:.4f} ({(1-efficiency)*100:.2f}% loss)")
    #             print(f"  Operating cost multiplier: {distance_multiplier:.3f}x")
    #
    # print("\n" + "="*80)
    # print("ADDING RENEWABLE LOCATION TIE-BREAKERS")
    # print("="*80)

    # 2. Add distance-based tie-breaker to renewable plants
    renewable_count = 0
    skipped_count = 0

    for source, node in graph._node.items():
        load_center = LOAD_CENTERS.get(source, (-119, 36))

        for handle, asset in node["assets"].items():
            fuel = asset.get("fuel", "")

            if fuel in {"solar", "wind"}:
                # Check if capex_cost exists and is a number
                capex_cost = asset.get("capex_cost")

                # Skip if no capex_cost or if it's not a valid number
                if capex_cost is None:
                    skipped_count += 1
                    continue

                # Convert to float if it's a string
                try:
                    if isinstance(capex_cost, str):
                        capex_cost = float(capex_cost)
                        asset["capex_cost"] = capex_cost
                    elif not isinstance(capex_cost, (int, float)):
                        skipped_count += 1
                        continue
                except (ValueError, TypeError):
                    skipped_count += 1
                    continue

                # Only process if capex_cost > 0
                if capex_cost <= 0:
                    skipped_count += 1
                    continue

                renewable_count += 1

                # Get plant coordinates
                x = asset.get("x", load_center[0])
                y = asset.get("y", load_center[1])

                # Distance to regional load center
                distance_km = haversine(x, y, load_center[0], load_center[1]) / 1000

                # Calculate capacity factor
                profile = asset.get("profile", [])
                if profile and len(profile) > 0:
                    if isinstance(profile, (list, np.ndarray)):
                        avg_cf = float(np.mean(profile))
                    else:
                        avg_cf = 0.3
                else:
                    avg_cf = asset.get("capacity_factor", 0.3)

                # Penalty 1: Distance (prefer local generation)
                distance_penalty = (distance_km / 100) * 1e-14

                # Penalty 2: Low capacity factor (prefer better sites)
                cf_penalty = max(0, 0.5 - avg_cf) * 1e-14

                # Apply combined penalty
                asset["capex_cost"] = capex_cost * (1.0 + distance_penalty + cf_penalty)

    print(f"\nProcessed {renewable_count} renewable plants with capex")
    print(f"Skipped {skipped_count} renewable plants (no valid capex_cost)")
    print("=" * 80 + "\n")

    return graph

def make_cohort_availability(model_steps, start_idx, window_hours):
    availability = {}
    for i, t in enumerate(model_steps):
        if start_idx <= i < start_idx + window_hours:
            availability[t] = 1.0
        else:
            availability[t] = 0.0

    # DEBUG: Print first few keys
    sample_keys = list(availability.keys())[:5]
    # print(f"DEBUG make_cohort_availability: sample keys = {sample_keys}")

    return availability

def scale_base_load_to_target_peak(graph, target_peak_gw, tx_regions):
    """
    Scale all Texas base load assets so the combined peak reaches target_peak_gw.

    Assumes:
    - base load installed_capacity is negative
    - asset["profile"] is a string like "ERC_REST:load"
    - actual profile values are stored in graph._node[region]["profiles"]
    """

    target_peak_w = target_peak_gw * 1e9
    current_system_profile = None
    base_assets = []

    for region in tx_regions:
        node = graph._node.get(region, {})
        assets = node.get("assets", {})
        node_profiles = node.get("profiles", {})

        for handle, asset in assets.items():
            if not handle.startswith("base_load_"):
                continue
            if asset.get("_class") != "Load":
                continue
            if asset.get("type") != "load":
                continue

            installed = float(asset.get("installed_capacity", 0.0))
            profile_name = asset.get("profile", None)

            if installed == 0 or profile_name is None:
                continue

            if profile_name not in node_profiles:
                raise KeyError(
                    f"Profile '{profile_name}' not found in node['profiles'] for region '{region}'."
                )

            profile = np.array(node_profiles[profile_name], dtype=float)

            if profile.size == 0:
                continue

            # Magnitude only for peak calculation
            hourly_load_w = np.abs(installed) * np.abs(profile)

            base_assets.append((region, handle, asset))

            if current_system_profile is None:
                current_system_profile = hourly_load_w.copy()
            else:
                if len(current_system_profile) != len(hourly_load_w):
                    raise ValueError(
                        f"Profile length mismatch for {handle}: "
                        f"{len(hourly_load_w)} vs {len(current_system_profile)}"
                    )
                current_system_profile += hourly_load_w

    if current_system_profile is None:
        raise ValueError("No Texas base load assets found.")

    current_peak_w = float(current_system_profile.max())
    current_peak_gw = current_peak_w / 1e9

    if current_peak_w <= 0:
        raise ValueError("Current Texas peak load is zero or invalid.")

    scale_factor = target_peak_w / current_peak_w

    # Keep load negative after scaling
    for region, handle, asset in base_assets:
        old_mag = abs(float(asset["installed_capacity"]))
        asset["installed_capacity"] = -old_mag * scale_factor

    print("\nTexas load scaling summary")
    print(f"Current TX base peak : {current_peak_gw:.2f} GW")
    print(f"Target TX base peak  : {target_peak_gw:.2f} GW")
    print(f"Applied scale factor : {scale_factor:.4f}")

    return graph, scale_factor, current_peak_gw

def retire_fraction_of_fuel(graph, fuel_name, retire_fraction):
    """
    Retire a fraction of existing capacity for all assets with the given fuel.

    Parameters
    ----------
    graph : networkx graph
    fuel_name : str
        Example: "coal"
    retire_fraction : float
        Example: 1/3 means retire one-third and keep two-thirds
    """
    if not (0 <= retire_fraction <= 1):
        raise ValueError("retire_fraction must be between 0 and 1.")

    keep_fraction = 1.0 - retire_fraction
    total_before_mw = 0.0
    total_after_mw = 0.0

    for region, node in graph._node.items():
        assets = node.get("assets", {})

        for handle, asset in assets.items():
            if str(asset.get("fuel", "")).lower() != fuel_name.lower():
                continue

            old_cap = float(asset.get("installed_capacity", 0.0))
            new_cap = old_cap * keep_fraction
            asset["installed_capacity"] = new_cap

            total_before_mw += old_cap / 1e6
            total_after_mw += new_cap / 1e6

    print(f"{fuel_name} retirement summary")
    print(f"Retired fraction : {retire_fraction:.2%}")
    print(f"Capacity before  : {total_before_mw:,.2f} MW")
    print(f"Capacity after   : {total_after_mw:,.2f} MW")

    return graph

def retire_positive_load_modeled_renewable_mw(graph, fuel_name, retire_mw, regions):
    """
    Retire a fixed MW amount of solar/wind that are modeled as positive Load assets.

    Parameters
    ----------
    graph : networkx graph
    fuel_name : str
        "solar" or "wind"
    retire_mw : float
        MW to retire
    regions : list[str]
        Regions to search
    """
    retire_w = retire_mw * 1e6

    candidates = []
    total_existing_w = 0.0

    for region in regions:
        node = graph._node.get(region, {})
        for handle, asset in node.get("assets", {}).items():
            asset_type = str(asset.get("type", "")).lower()
            asset_class = str(asset.get("_class", ""))

            if asset_type != fuel_name.lower():
                continue
            if asset_class != "Load":
                continue

            old_cap = float(asset.get("installed_capacity", 0.0))

            # positive renewable load
            if old_cap <= 0:
                continue

            candidates.append((region, handle, asset, old_cap))
            total_existing_w += old_cap

    if total_existing_w == 0:
        raise ValueError(f"No positive load-modeled {fuel_name} assets found.")

    if retire_w > total_existing_w:
        raise ValueError(
            f"Requested retirement of {retire_mw:,.0f} MW exceeds total existing "
            f"{fuel_name} capacity of {total_existing_w / 1e6:,.0f} MW."
        )

    retire_fraction = retire_w / total_existing_w
    total_after_w = 0.0

    for region, handle, asset, old_cap in candidates:
        new_cap = old_cap * (1 - retire_fraction)
        asset["installed_capacity"] = new_cap
        total_after_w += new_cap

    print(f"{fuel_name} retirement summary")
    print(f"Capacity before   : {total_existing_w / 1e6:,.2f} MW")
    print(f"Retired amount    : {retire_mw:,.2f} MW")
    print(f"Capacity after    : {total_after_w / 1e6:,.2f} MW")
    print(f"Retire fraction   : {retire_fraction:.4f}")

    return graph

def get_ipm_ev_profiles(ev_data, year, adoption, market, ipm_regions, charging_scenario, ):
    """
    Read IPM-level EV profiles from the new EV JSON structure.

    Expected structure:
    ev_data[str(year)][adoption][market][ipm_region][charging_scenario]

    Returns
    -------
    dict
        region -> {
            "hourly_profile": np.array,
            "hourly_total_kw": np.array,
            "peak_w": float
        }
    """

    year_key = str(year)

    if year_key not in ev_data:
        raise KeyError(f"Year {year_key} not found in EV data.")

    if adoption not in ev_data[year_key]:
        raise KeyError(f"Adoption {adoption} not found for year {year_key}.")

    if market not in ev_data[year_key][adoption]:
        raise KeyError(
            f"Market {market} not found for year {year_key}, adoption {adoption}."
        )

    out = {}
    missing_regions = []

    for region in ipm_regions:
        try:
            entry = ev_data[year_key][adoption][market][region][charging_scenario]
        except KeyError:
            missing_regions.append(region)
            continue

        hourly_profile = np.array(entry["hourly_profile"], dtype=float)
        hourly_total_kw = np.array(entry["hourly_total_kw"], dtype=float)

        if len(hourly_profile) != len(hourly_total_kw):
            raise ValueError(
                f"Length mismatch for {region}, {charging_scenario}. "
                f"profile={len(hourly_profile)}, total_kw={len(hourly_total_kw)}"
            )

        if len(hourly_profile) != 8760:
            print(
                f"Warning: {region}, {charging_scenario} has "
                f"{len(hourly_profile)} hours, not 8760."
            )

        out[region] = {
            "hourly_profile": hourly_profile,
            "hourly_total_kw": hourly_total_kw,
            "peak_w": float(hourly_total_kw.max()) * 1000.0,
        }

    if missing_regions:
        print("Missing EV profiles for these regions:")
        print(missing_regions)

    if len(out) == 0:
        raise ValueError("No EV profiles were loaded. Check market, regions, and scenario name.")

    return out

def apply_retirement_policy_to_graph(graph, retirement_policy=None, verbose=False, ):
    """
    Apply fuel retirement assumptions to the GOOD graph.

    Parameters
    ----------
    graph : GOOD graph
        The graph before building the network.

    retirement_policy : dict or None
        Example:
        {
            "oil": 0.70,
            "nuclear": 1.00,
            "natural gas turbine": 0.20,
            "natural gas combined cycle": 0.04,
            "coal": 1.00,
        }

    verbose : bool
        Print applied retirements.

    Returns
    -------
    graph
        Updated graph.
    """

    if retirement_policy is None:
        if verbose:
            print("No retirement policy applied.")
        return graph

    for fuel_name, retire_fraction in retirement_policy.items():

        if retire_fraction is None:
            continue

        retire_fraction = float(retire_fraction)

        if retire_fraction <= 0:
            continue

        if verbose:
            print(
                f"Retiring {retire_fraction:.2%} of fuel type: {fuel_name}"
            )

        graph = retire_fraction_of_fuel(
            graph=graph,
            fuel_name=fuel_name,
            retire_fraction=retire_fraction,
        )

    return graph

def apply_asset_constraints_to_network(network, asset_constraint_policy=None, verbose=False, ):
    """
    Apply fuel-specific must-run and ramp-rate assumptions to network assets.

    Parameters
    ----------
    network : GOOD network
        Network after:
        network = good.optimization.network.Network(...).from_graph(...)

    asset_constraint_policy : dict or None
        Example:
        {
            "natural gas combined cycle": {
                "must_run_fraction": 0.05,
                "ramp_rate": 0.20,
            },
            "natural gas turbine": {
                "ramp_rate": 0.80,
            },
            "oil": {
                "must_run_fraction": 0.05,
                "ramp_rate": 0.30,
            },
            "coal": {
                "must_run_fraction": 0.05,
                "ramp_rate": 0.05,
            },
        }

    verbose : bool
        Print applied settings.

    Returns
    -------
    network
        Updated network.
    """

    if asset_constraint_policy is None:
        if verbose:
            print("No asset constraint policy applied.")
        return network

    for asset_handle, asset in network.assets.items():

        fuel = str(asset.get("fuel", "")).lower().strip()

        if fuel not in asset_constraint_policy:
            continue

        settings = asset_constraint_policy[fuel]
        asset_object = asset["object"]

        if "must_run_fraction" in settings:
            setattr(
                asset_object,
                "must_run_fraction",
                settings["must_run_fraction"],
            )

        if "ramp_rate" in settings:
            setattr(
                asset_object,
                "ramp_rate",
                settings["ramp_rate"],
            )

        if verbose:
            print(
                f"Applied constraints to {asset_handle}: "
                f"fuel={fuel}, settings={settings}"
            )

    return network

def deep_update_dict(base, updates):
    """
    Recursively update a nested dictionary.
    """

    result = base.copy()

    for key, value in updates.items():
        if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
        ):
            result[key] = deep_update_dict(result[key], value)
        else:
            result[key] = value

    return result

def get_flexibility_policy(state, flexibility_policy=None, use_state_default_flexibility=True, ):
    """
    Build the final flexibility policy for one run.

    Priority:
    1. Start from FLEXIBILITY_POLICIES["default"]
    2. Add state-specific settings if available
    3. Add custom flexibility_policy if passed
    """

    policy = flexibility_policy["default"]

    if use_state_default_flexibility:
        state_policy = flexibility_policy.get(state, {})
        policy = deep_update_dict(policy, state_policy)

    if flexibility_policy is not None:
        policy = deep_update_dict(policy, flexibility_policy)

    return policy

def build_flexibility_settings(cfg, state, n_hours, n_ev_hours, flexibility_policy=None,
                               use_state_default_flexibility=True, ):
    """
    Convert V1G, V2G, and battery policy settings into model-ready values.

    Parameters
    ----------
    cfg : dict
        Scenario configuration. Must contain:
        - v1g_share
        - v2g_share
        - batt_capex_cost

    state : str
        State or market label.

    n_hours : int
        Number of model hours in the optimization window.

    n_ev_hours : int
        Full length of EV profile, usually 8760.

    flexibility_policy : dict, optional
        Custom settings for one run.

    use_state_default_flexibility : bool
        If True, use FLEXIBILITY_POLICIES[state] when available.

    Returns
    -------
    dict
        Model-ready flexibility settings.
    """

    policy = get_flexibility_policy(
        state=state,
        flexibility_policy=flexibility_policy,
        use_state_default_flexibility=use_state_default_flexibility,
    )

    v1g_policy = policy["v1g"]
    v2g_policy = policy["v2g"]
    battery_policy = policy["battery"]

    # -----------------------------
    # Participation shares
    # -----------------------------
    v1g_participation = float(cfg["v1g_share"])
    v2g_participation = float(cfg["v2g_share"])

    total_participation = v1g_participation + v2g_participation
    fixed_share = 1.0 - total_participation

    if fixed_share < -1e-9:
        raise ValueError(
            f"V1G + V2G participation exceeds 1. "
            f"v1g={v1g_participation}, v2g={v2g_participation}"
        )

    fixed_share = max(fixed_share, 0.0)

    # -----------------------------
    # Cost profiles
    # -----------------------------
    hourly_scaler = np.ones(n_hours)
    hourly_scaler_full = np.ones(n_ev_hours)

    ev_shift_cost_profile = (
            float(v1g_policy["base_shift_cost"]) * hourly_scaler_full
    )

    v2g_shift_cost_profile = (
            float(v2g_policy["base_shift_cost"]) * hourly_scaler_full
    )

    # -----------------------------
    # V1G fixed cost
    # $/kW-year to $/W-year
    # -----------------------------
    v1g_fixed_om_per_kw_year = float(v1g_policy["fixed_om_per_kw_year"])
    v1g_fix_cost = v1g_fixed_om_per_kw_year / 1000.0

    # -----------------------------
    # V2G settings
    # -----------------------------
    hours_ev = float(v2g_policy["energy_duration_hours"])
    eta_ev = float(v2g_policy["roundtrip_efficiency"])

    v2g_fixed_om_per_kw_year = float(v2g_policy["fixed_om_per_kw_year"])
    v2g_fix_cost = v2g_fixed_om_per_kw_year / (hours_ev * 3600 * 1000)

    # -----------------------------
    # Stationary battery settings
    # -----------------------------
    hours_st = float(battery_policy["duration_hours"])
    eta_charge_st = float(battery_policy["charge_efficiency"])
    eta_discharge_st = float(battery_policy["discharge_efficiency"])

    fixed_om_per_kw_year = float(battery_policy["fixed_om_per_kw_year"])
    batt_fix_cost = fixed_om_per_kw_year / (hours_st * 3600 * 1000)

    cycling_cost_per_mwh = float(battery_policy["cycling_cost_per_mwh"])
    batt_opt_cost = cycling_cost_per_mwh / 3.6e9

    batt_capex_cost = float(cfg["batt_capex_cost"])

    return {
        # Shares
        "V1G_PARTICIPATION": v1g_participation,
        "V2G_PARTICIPATION": v2g_participation,
        "TOTAL_PARTICIPATION": total_participation,
        "FIXED_SHARE": fixed_share,

        # V1G
        "EV_SHIFT_WINDOW": int(v1g_policy["shift_window_hours"]),
        "EV_SHIFT_COST": ev_shift_cost_profile,
        "V1G_FIX_COST": v1g_fix_cost,

        # V2G
        "V2G_WINDOW_HOURS": int(v2g_policy["window_hours"]),
        "HOURS_EV": hours_ev,
        "ETA_EV": eta_ev,
        "SHIFT_COST": v2g_shift_cost_profile,
        "V2G_FIX_COST": v2g_fix_cost,

        # Stationary battery
        "HOURS_ST": hours_st,
        "ETA_CHARGE_ST": eta_charge_st,
        "ETA_DISCHARGE_ST": eta_discharge_st,
        "P_TOTAL_MW_ST": float(battery_policy["total_power_mw"]),
        "BATT_FIX_COST": batt_fix_cost,
        "BATT_OPT_COST": batt_opt_cost,
        "BATT_CAPEX_COST": batt_capex_cost,
        "BATTERY_INITIAL_SOC_FRACTION": float(battery_policy["initial_soc_fraction"]),

        # Useful diagnostics
        "policy": policy,
    }

def deep_update_dict(base, updates):
    """
    Recursively update a nested dictionary without modifying the original.
    """

    result = copy.deepcopy(base)

    for key, value in updates.items():
        if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
        ):
            result[key] = deep_update_dict(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result

def compute_crf(discount_rate, lifetime_years):
    """
    Compute capital recovery factor.
    """

    r = float(discount_rate)
    n = float(lifetime_years)

    if r == 0:
        return 1.0 / n

    return r * (1 + r) ** n / ((1 + r) ** n - 1)

def get_economic_policy(state, discount_rate=None, lifetime=None, economic_policy=None,
                        use_state_default_economic_policy=True, ):
    """
    Build final economic policy for one run.

    Priority:
    1. ECONOMIC_POLICIES["default"]
    2. ECONOMIC_POLICIES[state]
    3. economic_policy passed into run_one_scenario
    4. discount_rate and lifetime passed directly into run_one_scenario
    """

    policy = copy.deepcopy(economic_policy["default"])

    if use_state_default_economic_policy:
        policy = deep_update_dict(
            policy,
            economic_policy.get(state, {}),
        )

    if economic_policy is not None:
        policy = deep_update_dict(policy, economic_policy)

    if discount_rate is not None:
        policy["discount_rate"] = discount_rate

    if lifetime is not None:
        policy["lifetime_years"] = lifetime

    policy["crf"] = compute_crf(
        discount_rate=policy["discount_rate"],
        lifetime_years=policy["lifetime_years"],
    )

    return policy

def apply_economic_policy_to_graph(graph, economic_policy, verbose=False, ):
    """
    Apply CRF and import operating cost assumptions to the graph.
    """

    crf = economic_policy["crf"]

    renewable_fuels_for_crf = set(
        economic_policy.get("renewable_fuels_for_crf", {"solar", "wind"})
    )

    apply_crf_to_renewables = economic_policy.get(
        "apply_crf_to_renewables",
        True,
    )

    apply_crf_to_storage = economic_policy.get(
        "apply_crf_to_storage",
        True,
    )

    import_operating_cost = economic_policy.get(
        "import_operating_cost",
        None,
    )

    renewable_assets_updated = 0
    storage_assets_updated = 0
    import_assets_updated = 0

    for source, node in graph._node.items():
        for handle, asset in node.get("assets", {}).items():

            fuel = str(asset.get("fuel", "")).lower().strip()

            # Apply CRF to solar and wind capex
            if (
                    apply_crf_to_renewables
                    and fuel in renewable_fuels_for_crf
                    and asset.get("capex_cost") is not None
                    and not asset.get("_crf_applied", False)
            ):
                asset["capex_cost"] *= crf
                asset["_crf_applied"] = True
                renewable_assets_updated += 1

            # Apply CRF to storage capex
            if (
                    apply_crf_to_storage
                    and asset.get("_class") == "Store"
                    and not asset.get("_crf_applied", False)
            ):
                if asset.get("capex_cost_energy") is not None:
                    asset["capex_cost_energy"] *= crf

                if asset.get("capex_cost_power") is not None:
                    asset["capex_cost_power"] *= crf

                asset["_crf_applied"] = True
                storage_assets_updated += 1

            # Apply import cost
            if import_operating_cost is not None and fuel == "import":
                asset["operating_cost"] = import_operating_cost
                import_assets_updated += 1

    if verbose:
        print("Economic policy applied.")
        print(f"Discount rate: {economic_policy['discount_rate']}")
        print(f"Lifetime years: {economic_policy['lifetime_years']}")
        print(f"CRF: {crf:.6f}")
        print(f"Renewable assets updated: {renewable_assets_updated}")
        print(f"Storage assets updated: {storage_assets_updated}")
        print(f"Import assets updated: {import_assets_updated}")

    return graph

def get_transmission_policy(state, transmission_policy=None, use_state_default_transmission_policy=True, ):
    """
    Build final transmission policy for one run.
    """

    policy = copy.deepcopy(transmission_policy["default"])

    if use_state_default_transmission_policy:
        policy = deep_update_dict(
            policy,
            transmission_policy.get(state, {}),
        )

    if transmission_policy is not None:
        policy = deep_update_dict(policy, transmission_policy)

    return policy

def apply_transmission_policy_to_graph(graph, transmission_policy, verbose=False, ):
    """
    Apply transmission distance enhancement, line operating cost, and efficiency.
    """

    if transmission_policy.get("apply_distance_enhancement", True):
        graph = enhance_transmission_with_distance(graph)

    default_operating_cost = transmission_policy.get(
        "default_operating_cost",
        None,
    )

    default_efficiency = transmission_policy.get(
        "default_efficiency",
        None,
    )

    overwrite_existing_operating_cost = transmission_policy.get(
        "overwrite_existing_operating_cost",
        False,
    )

    overwrite_existing_efficiency = transmission_policy.get(
        "overwrite_existing_efficiency",
        False,
    )

    line_cost_updates = 0
    line_efficiency_updates = 0

    for source, _adj in graph._adj.items():
        for target, edge in _adj.items():
            for line_name, line_data in edge.get("lines", {}).items():

                if default_operating_cost is not None:
                    current_cost = line_data.get("operating_cost", 0)

                    if overwrite_existing_operating_cost or current_cost == 0:
                        line_data["operating_cost"] = default_operating_cost
                        line_cost_updates += 1

                if default_efficiency is not None:
                    if overwrite_existing_efficiency or "efficiency" not in line_data:
                        line_data["efficiency"] = default_efficiency
                        line_efficiency_updates += 1

    if verbose:
        print("Transmission policy applied.")
        print(f"Distance enhancement: {transmission_policy.get('apply_distance_enhancement', True)}")
        print(f"Line cost updates: {line_cost_updates}")
        print(f"Line efficiency updates: {line_efficiency_updates}")

    return graph

def load_scenarios(region, year=2030, scenarios_folder="Examples"):
    """
    Load scenario JSON file for a selected region and year.

    Parameters
    ----------
    region : str
        Region name. Examples: "CA", "NY", "TX", "PJM".
    year : int
        Scenario year.
    scenarios_folder : str
        Folder where scenario JSON files are saved.

    Returns
    -------
    dict
        Scenario dictionary with integer scenario IDs.
    """

    region = region.upper()
    file_path = Path(scenarios_folder) / f"scenarios_{year}_{region}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Scenario file not found: {file_path}")

    with open(file_path, "r") as f:
        scenarios = json.load(f)

    scenarios = {int(k): v for k, v in scenarios.items()}

    return scenarios