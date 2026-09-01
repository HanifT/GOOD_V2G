import os
import copy
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pyomo.opt import SolverStatus, TerminationCondition
warnings.filterwarnings("ignore")

import good
from good.reload import deep_reload
import numpy as np
import pandas as pd
import pyomo.environ as pe

from good.save_tools import save_scenario_results
from good.debug_tools import (
    run_full_debug_suite,
    save_capacity_and_capex_tables,
    save_ev_summary,
)

from . import helper

import hashlib
import numpy as np
from collections import defaultdict

def diagnose_exact_renewable_groups(graph):
    """
    Find expandable solar and wind assets that may be aggregated exactly.

    This function only diagnoses groups. It does not modify the graph.

    It supports both:
        1. Numeric profile arrays
        2. String profile references such as "PJM_AP:wind:7"
    """

    groups = defaultdict(list)
    renewable_count = 0

    for region_handle, node in graph._node.items():

        for asset_handle, asset in node.get("assets", {}).items():

            fuel = str(
                asset.get("fuel", "")
            ).strip().lower()

            asset_type = str(
                asset.get("type", "")
            ).strip().lower()

            if (
                fuel not in {"solar", "wind"}
                and asset_type not in {"solar", "wind"}
            ):
                continue

            capex_capacity_raw = asset.get(
                "capex_capacity",
                0,
            )

            try:
                capex_capacity = float(
                    capex_capacity_raw or 0
                )
            except (TypeError, ValueError):
                print(
                    f"WARNING: invalid capex_capacity for "
                    f"{region_handle}/{asset_handle}: "
                    f"{capex_capacity_raw!r}"
                )
                continue

            if capex_capacity <= 0:
                continue

            # -------------------------------------------------
            # Profile identity
            # -------------------------------------------------
            profile_raw = asset.get(
                "profile",
                None,
            )

            if isinstance(profile_raw, str):

                # GOOD can store profiles as references such as:
                # "PJM_AP:wind:7"
                profile_identity = (
                    "reference",
                    profile_raw,
                )

            elif profile_raw is None:

                profile_identity = (
                    "none",
                    None,
                )

            else:

                try:
                    profile_array = np.asarray(
                        profile_raw,
                        dtype=np.float64,
                    )

                    profile_hash = hashlib.sha1(
                        profile_array.tobytes()
                    ).hexdigest()

                    profile_identity = (
                        "array",
                        profile_hash,
                    )

                except (TypeError, ValueError):

                    # Safe fallback for unusual profile objects
                    profile_identity = (
                        "other",
                        repr(profile_raw),
                    )

            # -------------------------------------------------
            # Numeric cost fields
            # -------------------------------------------------
            try:
                capex_cost = float(
                    asset.get("capex_cost", 0) or 0
                )
            except (TypeError, ValueError):
                capex_cost = repr(
                    asset.get("capex_cost", 0)
                )

            try:
                operating_cost = float(
                    asset.get("operating_cost", 0) or 0
                )
            except (TypeError, ValueError):
                operating_cost = repr(
                    asset.get("operating_cost", 0)
                )

            key = (
                region_handle,
                str(asset.get("jurisdiction", "")),
                fuel,
                asset_type,
                bool(asset.get("rps_eligible", False)),
                capex_cost,
                operating_cost,
                profile_identity,
            )

            groups[key].append(
                asset_handle
            )

            renewable_count += 1

    mergeable_groups = {
        key: handles
        for key, handles in groups.items()
        if len(handles) > 1
    }

    original_assets_in_groups = sum(
        len(handles)
        for handles in mergeable_groups.values()
    )

    resulting_group_assets = len(
        mergeable_groups
    )

    removable_assets = (
        original_assets_in_groups
        - resulting_group_assets
    )

    print("=" * 80)
    print("EXACT RENEWABLE AGGREGATION DIAGNOSTIC")
    print("=" * 80)
    print(
        "Expandable solar/wind assets:",
        renewable_count,
    )
    print(
        "Exact duplicate groups:",
        resulting_group_assets,
    )
    print(
        "Assets contained in duplicate groups:",
        original_assets_in_groups,
    )
    print(
        "Assets potentially removable:",
        removable_assets,
    )
    print(
        "CAPEX variables after exact aggregation:",
        renewable_count - removable_assets,
    )
    print("=" * 80)

    largest_groups = sorted(
        mergeable_groups.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )[:20]

    print("\nLargest exact groups:")

    for key, handles in largest_groups:

        (
            region,
            jurisdiction,
            fuel,
            asset_type,
            rps_eligible,
            capex_cost,
            operating_cost,
            profile_identity,
        ) = key

        print(
            f"{region:12s} | "
            f"{jurisdiction:3s} | "
            f"{fuel:6s} | "
            f"assets={len(handles):3d} | "
            f"profile={profile_identity[1]}"
        )

    return mergeable_groups

def add_fixed_datacenter_load_to_graph(
    graph,
    load_profiles_mw_by_region,
    start_hour,
    n_ev_hours,
    load_name_prefix="datacenter",
):
    """
    Add data-center demand as fixed regional load.

    Input:
        load_profiles_mw_by_region:
            {
                "PJM_Dom": [MW, MW, ..., MW],
                "PJM_AP":  [MW, MW, ..., MW],
                ...
            }

    The input profile can be:
        1) only the modeled window length, e.g. 168 hours, or
        2) full-year length.

    GOOD Load profiles are indexed by the model step, so if the input is only
    168 hours, we place it inside a full-year profile using start_hour.
    """

    if load_profiles_mw_by_region is None:
        return graph

    if not load_profiles_mw_by_region:
        return graph

    total_added_peak_mw = 0.0
    total_added_avg_mw = 0.0

    for region, profile_mw in load_profiles_mw_by_region.items():

        if region not in graph._node:
            print(f"WARNING: data-center region {region} is not in graph. Skipping.")
            continue

        arr = np.array(profile_mw, dtype=float)

        if arr.ndim != 1:
            raise ValueError(f"Data-center profile for {region} must be one-dimensional.")

        if np.any(arr < -1e-9):
            raise ValueError(f"Data-center load for {region} has negative values.")

        # Case 1: input is already full-year length.
        if len(arr) == n_ev_hours:
            full_profile_mw = arr.copy()

        # Case 2: input is only the modeled window, e.g. 168 hours.
        else:
            full_profile_mw = np.zeros(n_ev_hours, dtype=float)

            end_hour = start_hour + len(arr)

            if end_hour > n_ev_hours:
                raise ValueError(
                    f"Data-center profile for {region} is too long. "
                    f"start_hour={start_hour}, len(profile)={len(arr)}, "
                    f"n_ev_hours={n_ev_hours}."
                )

            full_profile_mw[start_hour:end_hour] = arr

        peak_mw = float(np.max(full_profile_mw))

        if peak_mw <= 0:
            continue

        # GOOD uses W for installed_capacity.
        peak_w = peak_mw * 1e6

        # Load profile is normalized to peak.
        normalized_profile = (full_profile_mw / peak_mw).tolist()

        asset_name = f"{load_name_prefix}_load_{region}"

        graph._node[region].setdefault("assets", {})[asset_name] = {
            "_class": "Load",
            "type": "load",
            "fuel": "datacenter",

            # Negative because load is demand.
            "installed_capacity": -peak_w,

            # Unitless normalized shape.
            "profile": normalized_profile,

            # Important: data centers are fixed load here, not V1G.
            "shift_capacity": 0.0,
            "shift_window": 0,
            "capex_capacity": 0,
            "shift_cost": 0.0,
            "fixed_shift_cost": 0.0,
        }

        total_added_peak_mw += peak_mw
        total_added_avg_mw += float(np.mean(full_profile_mw[start_hour:start_hour + len(arr)]))

        print(
            f"Added data-center load to {region}: "
            f"peak={peak_mw:.2f} MW, "
            f"window_avg={np.mean(arr):.2f} MW"
        )

    print(
        f"Total data-center load added: "
        f"peak_sum={total_added_peak_mw:.2f} MW, "
        f"avg_sum={total_added_avg_mw:.2f} MW"
    )

    return graph


# =============================================================================
# EV aggregate-flexibility helper functions
# =============================================================================

def resolve_ev_connected_fraction_profile(
    region_data,
    n_ev_hours,
    fallback_profile=None,
    min_connected_fraction=0.05,
):
    """
    Return an hourly connected-EV fraction profile aligned with the EV data.

    Preferred input is an explicit connected/plugged-in fraction profile inside
    region_data. If that is not available, use the EV charging profile as a
    conservative fallback shape. The fallback is not a perfect plug-in profile,
    but it lets the aggregate V1G/V2G formulation run until a true connected-EV
    profile is added to the EV data.
    """

    candidate_keys = [
        "connected_fraction_profile",
        "plugged_in_fraction_profile",
        "availability_fraction_profile",
        "ev_availability_profile",
        "connected_profile",
        "plugged_in_profile",
        "availability_profile",
    ]

    arr = None
    source_key = None

    if isinstance(region_data, dict):
        for key in candidate_keys:
            if key in region_data and region_data[key] is not None:
                arr = np.asarray(region_data[key], dtype=float)
                source_key = key
                break

    used_fallback = False

    if arr is None:
        if fallback_profile is None:
            arr = np.ones(n_ev_hours, dtype=float)
        else:
            arr = np.asarray(fallback_profile, dtype=float)
        source_key = "fallback_hourly_profile"
        used_fallback = True

    if arr.ndim != 1:
        raise ValueError(
            f"Connected-EV availability profile must be one-dimensional. "
            f"Source key: {source_key}."
        )

    if len(arr) != n_ev_hours:
        raise ValueError(
            f"Connected-EV availability profile length mismatch. "
            f"Expected {n_ev_hours}, got {len(arr)}. Source key: {source_key}."
        )

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.maximum(arr, 0.0)

    max_val = float(np.max(arr)) if len(arr) else 0.0

    if max_val <= 0.0:
        arr = np.ones(n_ev_hours, dtype=float)
    elif max_val > 1.0:
        # If the profile is counts, MW, or another positive magnitude, convert
        # it to a 0-1 relative connected-fleet fraction.
        arr = arr / max_val

    arr = np.clip(arr, 0.0, 1.0)

    if used_fallback and min_connected_fraction is not None:
        # Keep a small connected floor so V1G can move energy into hours where
        # the baseline charging profile is low. Replace this fallback with a
        # true plug-in availability profile when available.
        arr = np.maximum(arr, float(min_connected_fraction))
        arr = np.clip(arr, 0.0, 1.0)

    return arr


def build_aggregate_ev_capacity_profiles(
    region_data,
    peak_power_w,
    hours_ev,
    n_ev_hours,
    fallback_profile=None,
    min_connected_fraction=0.05,
):
    """
    Build hourly aggregate EV energy and power availability profiles.

    This is the runner-side input to the updated Store class:
        energy_capacity_profile[t]          [J]
        charge_power_capacity_profile[t]    [W]
        discharge_power_capacity_profile[t] [W]

    The simple conversion keeps the same duration logic as the older GOOD EV
    storage approximation:
        available_energy[t] = available_power[t] * HOURS_EV
    """

    connected_fraction = resolve_ev_connected_fraction_profile(
        region_data=region_data,
        n_ev_hours=n_ev_hours,
        fallback_profile=fallback_profile,
        min_connected_fraction=min_connected_fraction,
    )

    available_power_w = float(peak_power_w) * connected_fraction
    available_energy_j = available_power_w * float(hours_ev) * 3600.0

    return connected_fraction, available_power_w, available_energy_j



def build_detail_equivalent_v2g_capacity_profiles(
    region_profile_full,
    peak_power_w,
    hours_ev,
    window_hours,
    n_ev_hours,
    start_hour,
    end_hour,
):
    """
    Build aggregate V2G power/energy profiles that match the old detailed
    cohort logic as closely as possible, but without creating one Store per
    hour.

    Old detailed logic:
        for each charging hour tau:
            P_tau = peak_power_w * EV_profile[tau]
            E_tau = P_tau * hours_ev
            that cohort is available for V2G_WINDOW_HOURS

    This helper sums all active cohorts at each model hour:
        available_power[t] = sum_tau_active_at_t P_tau
        available_energy[t] = sum_tau_active_at_t E_tau

    This is much less conservative than using the instantaneous connected
    fraction only, and is the main fix for restoring the value seen in the
    detailed formulation.
    """
    profile = np.asarray(region_profile_full, dtype=float)

    if profile.ndim != 1:
        raise ValueError("region_profile_full must be one-dimensional.")

    if len(profile) != n_ev_hours:
        raise ValueError(
            f"V2G profile length mismatch. Expected {n_ev_hours}, got {len(profile)}."
        )

    n_hours = int(end_hour - start_hour)
    if n_hours <= 0:
        raise ValueError("end_hour must be greater than start_hour.")

    window_profile = profile[start_hour:end_hour].copy()
    window_profile = np.nan_to_num(window_profile, nan=0.0, posinf=0.0, neginf=0.0)
    window_profile = np.maximum(window_profile, 0.0)

    # Match the detailed runner behavior exactly.
    if window_profile.size > 0 and float(np.max(window_profile)) > 1.0:
        window_profile = window_profile / float(np.max(window_profile))

    window = int(window_hours) if window_hours is not None else n_hours
    window = max(window, 1)

    power_profile_w = np.zeros(n_ev_hours, dtype=float)

    for tau_idx in range(n_hours):
        p_tau = float(peak_power_w) * float(window_profile[tau_idx])

        if p_tau <= 1e-9:
            continue

        active_start = start_hour + tau_idx
        active_stop = min(end_hour, active_start + window)

        power_profile_w[active_start:active_stop] += p_tau

    energy_profile_j = power_profile_w * float(hours_ev) * 3600.0

    return power_profile_w, energy_profile_j


def print_rps_asset_membership(graph, policies_input):
    """
    Print which asset types are included and excluded by each RPS policy.
    This evaluates the same inclusion_criteria and exclusion_criteria that
    Portfolio_Standard will use later.
    """

    rows = []

    for policy_key, policy in policies_input.items():

        if not policy_key.startswith("rps_"):
            continue

        inclusion_criteria = policy.get("inclusion_criteria", {})
        exclusion_criteria = policy.get("exclusion_criteria", {})

        inclusion_functions = [
            eval(fun_string) if isinstance(fun_string, str) else fun_string
            for fun_string in inclusion_criteria.values()
        ]

        exclusion_functions = [
            eval(fun_string) if isinstance(fun_string, str) else fun_string
            for fun_string in exclusion_criteria.values()
        ]

        for node_name, node in graph._node.items():

            for asset_key, asset in node.get("assets", {}).items():

                include = all(fun(asset) for fun in inclusion_functions)
                exclude = all(fun(asset) for fun in exclusion_functions)

                if not include and not exclude:
                    continue

                rows.append({
                    "policy": policy_key,
                    "status": "included" if include else "excluded",
                    "node": node_name,
                    "asset_key": asset_key,
                    "type": asset.get("type"),
                    "fuel": asset.get("fuel"),
                    "_class": asset.get("_class"),
                    "jurisdiction": asset.get("jurisdiction"),
                    "rps_eligible": asset.get("rps_eligible"),
                    "renewable": asset.get("renewable"),
                    "capacity_mw": abs(float(asset.get("installed_capacity", 0.0))) / 1e6,
                })

    if not rows:
        print("\nRPS asset membership check: no matching assets found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("RPS ASSET MEMBERSHIP CHECK")
    print("=" * 80)

    summary = (
        df
        .groupby(
            ["policy", "status", "type", "fuel", "jurisdiction", "rps_eligible"],
            dropna=False
        )
        .agg(
            asset_count=("asset_key", "count"),
            capacity_mw=("capacity_mw", "sum"),
        )
        .reset_index()
        .sort_values(["policy", "status", "type", "fuel", "jurisdiction"])
    )

    print(summary.to_string(index=False))

    print("=" * 80 + "\n")

    return df


def get_model_region_policy_inputs(model_region, region_run_configs):
    """
    Return the policy dictionaries that should be passed to run_one_scenario().

    Use this when you run one model region at a time.

    Important for PJM:
        state_rps_policies is intentionally None because each PJM scenario
        already carries its own cfg["state_rps_policies"]. This lets the same
        run notebook execute rps_minus10, rps_base, and rps_plus10 without
        manually changing a PJM policy dictionary.

    Example:
        MODEL_REGION = "PJM"

        policy_inputs = get_model_region_policy_inputs(
            MODEL_REGION,
            REGION_RUN_CONFIGS,
        )

        result = s_runner.run_one_scenario(
            scenario_id=scenario_id,
            ctx=ctx,
            year=2030,
            adoption=adoption,
            charging=charging,
            month=7,
            day_duration=day_duration,
            model_region=MODEL_REGION,
            state_rps_policies=policy_inputs["state_rps_policies"],
            state_retirement_policies=policy_inputs["state_retirement_policies"],
        )
    """

    if model_region not in region_run_configs:
        raise ValueError(
            f"Unknown model_region={model_region}. "
            f"Available model regions are: {list(region_run_configs.keys())}"
        )

    cfg = region_run_configs[model_region]

    return {
        "model_region": model_region,
        "rps_mode": cfg.get("rps_mode"),
        "output_json": cfg.get("output_json"),

        # For PJM this should be None because each scenario has its own
        # cfg["state_rps_policies"].
        "state_rps_policies": cfg.get("state_rps_policies"),

        # For PJM this returns PJM_STATE_RETIREMENT_POLICIES.
        # For CAISO, NYISO, ERCOT this is usually None unless you define it.
        "state_retirement_policies": cfg.get("state_retirement_policies"),
    }



def fill_empty_solution_lists(solution, n_steps, fields_to_fill=None):
    """
    Fill empty solution lists before creating the solution dataframe.

    This avoids pandas length errors when a solution field exists
    but has no values. For PJM, this happens for node clearing_price.
    """

    if fields_to_fill is None:
        fields_to_fill = {"clearing_price"}

    filled_count = 0

    for source, node in solution._node.items():

        for key, value in list(node.items()):
            if key in fields_to_fill and isinstance(value, list) and len(value) == 0:
                node[key] = [np.nan] * n_steps
                filled_count += 1

        for asset_handle, asset in node.get("assets", {}).items():
            for key, value in list(asset.items()):
                if key in fields_to_fill and isinstance(value, list) and len(value) == 0:
                    asset[key] = [np.nan] * n_steps
                    filled_count += 1

        for target, edge in solution._adj[source].items():

            for key, value in list(edge.items()):
                if key in fields_to_fill and isinstance(value, list) and len(value) == 0:
                    edge[key] = [np.nan] * n_steps
                    filled_count += 1

            for line_handle, line in edge.get("lines", {}).items():
                for key, value in list(line.items()):
                    if key in fields_to_fill and isinstance(value, list) and len(value) == 0:
                        line[key] = [np.nan] * n_steps
                        filled_count += 1

    if filled_count > 0:
        print(
            f"Filled {filled_count} empty solution fields with NaN "
            f"for {n_steps} model steps."
        )

    return solution


def resolve_model_keys(cfg, state_to_regions, model_region=None):
    """
    Resolve EV state, model region, and GOOD node regions.

    Example for NYISO:
        ev_state = "NY"
        model_region = "NYISO"
        regions_for_model = STATE_TO_REGIONS["NYISO"]

    Example for PJM:
        ev_state = "PJM"
        model_region = "PJM"
        regions_for_model = STATE_TO_REGIONS["PJM"]
    """

    ev_state = cfg.get("state")

    resolved_model_region = (
        model_region
        or cfg.get("model_region")
        or cfg.get("ipm_region")
        or ev_state
    )

    lookup_candidates = [
        resolved_model_region,
        cfg.get("ipm_region"),
        ev_state,
    ]

    lookup_candidates = [x for x in lookup_candidates if x is not None]

    for key in lookup_candidates:
        if key in state_to_regions:
            return {
                "ev_state": ev_state,
                "model_region": resolved_model_region,
                "region_lookup_key": key,
                "regions_for_model": state_to_regions[key],
            }

    raise ValueError(
        f"Could not find a valid region key for scenario.\n"
        f"Candidates checked: {lookup_candidates}\n"
        f"Available STATE_TO_REGIONS keys: {list(state_to_regions.keys())}"
    )


def get_policy_with_fallback(policy_dict, model_region, ev_state):
    """
    Try model region first, then EV state.
    This supports both NYISO based and NY based policy dictionaries.
    """

    if policy_dict is None:
        return None

    if model_region in policy_dict:
        return policy_dict[model_region]

    if ev_state in policy_dict:
        return policy_dict[ev_state]

    return None


def remove_batteries_only(graph):
    """
    Remove battery storage, but keep pumped hydro.
    """

    battery_types = {"storage", "battery"}
    total_removed = 0

    for source, node in graph._node.items():
        assets = node.get("assets", {})
        assets_to_remove = []

        for key, asset in assets.items():
            asset_type = asset.get("type", "").lower()
            asset_fuel = asset.get("fuel", "").lower()

            is_battery = (
                (asset_type in battery_types or asset_fuel in battery_types)
                and "pump" not in asset_type
                and "pump" not in asset_fuel
                and "optional_storage" not in key.lower()
            )

            is_optional_storage = "optional_storage" in key.lower()

            if is_battery or is_optional_storage:
                assets_to_remove.append(key)

        for key in assets_to_remove:
            del assets[key]
            total_removed += 1

    print(f"Total battery assets removed: {total_removed}")
    print("=" * 80 + "\n")

    return graph

@dataclass
class ScenarioRunContext:
    """
    Container for objects that used to live as notebook globals.
    """

    scenarios: Dict[int, dict]
    base_graph: Any
    base_policies: dict
    state_to_regions: dict
    ev_data: Any
    battery_weights: dict
    results_dir: str

    discount_rate: float
    lifetime: int

    retirement_policies: Optional[dict] = None
    asset_constraint_policies: Optional[dict] = None
    state_rps_policies: Optional[dict] = None
    state_retirement_policies: Optional[dict] = None

    solver_executable: str = "/ocean/projects/ele250005p/htayarani/CPLEX_Studio222/cplex/bin/x86-64_linux/cplex"
    solver_threads: int = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))

def run_one_scenario(
    scenario_id,
    ctx: ScenarioRunContext,
    year=2030,
    adoption="mid",
    charging="load_leveling",
    month=7,
    day_duration=30,
    policies_input=None,
    graph_input=None,
    model_region=None,
    discount_rate=None,
    lifetime=None,
    fix_peak=False,
    target_peak_gw=None,
    peak_region_mode="state",
    state_rps_policies=None,
    peak_regions=None,
    retirement_policy=None,
    asset_constraint_policy=None,
    use_state_default_retirement=True,
    use_state_default_asset_constraints=True,
    flexibility_policy=None,
    use_state_default_flexibility=True,
    economic_policy=None,
    use_state_default_economic_policy=True,
    # New: fixed additional load, such as data centers.
    additional_fixed_load_mw_by_region=None,
    transmission_policy=None,
    use_state_default_transmission_policy=True,
):
    # ========================================
    # Fresh graph and fresh policies
    # ========================================
    deep_reload(good)
    # ========================================
    # Default inputs
    # ========================================
    if scenario_id not in ctx.scenarios:
        raise KeyError(
            f"Scenario ID {scenario_id} is not in ctx.scenarios. "
            f"Available IDs are: {list(ctx.scenarios.keys())[:10]}..."
        )

    cfg = ctx.scenarios[scenario_id]

    keys = resolve_model_keys(
        cfg=cfg,
        state_to_regions=ctx.state_to_regions,
        model_region=model_region,
    )

    ev_state = keys["ev_state"]
    model_region = keys["model_region"]
    region_lookup_key = keys["region_lookup_key"]
    regions_for_model = keys["regions_for_model"]

    if graph_input is None:
        graph_input = copy.deepcopy(ctx.base_graph)

    if policies_input is None:
        policies_input = copy.deepcopy(ctx.base_policies)

    if discount_rate is None:
        discount_rate = ctx.discount_rate

    if lifetime is None:
        lifetime = ctx.lifetime

    print(f"EV state: {ev_state}")
    print(f"Model region: {model_region}")
    print(f"STATE_TO_REGIONS key used: {region_lookup_key}")
    print(f"GOOD regions used: {regions_for_model}")
    # ========================================
    # Optional base load peak scaling
    # ========================================
    if fix_peak:
        if target_peak_gw is None:
            raise ValueError(
                "fix_peak=True, so you must pass target_peak_gw. "
                "Example: target_peak_gw=32.0"
            )
        if peak_region_mode == "state":
            regions_for_peak = regions_for_model
        elif peak_region_mode == "all":
            regions_for_peak = list(graph_input._node.keys())
        elif peak_region_mode == "custom":
            if peak_regions is None:
                raise ValueError(
                    "peak_region_mode='custom', so you must pass peak_regions. "
                    "Example: peak_regions=['NY_Z_J', 'NY_Z_K']"
                )
            regions_for_peak = peak_regions
        else:
            raise ValueError(
                "peak_region_mode must be one of: 'state', 'all', or 'custom'."
            )
        graph, load_scale_factor, current_peak_gw = helper.scale_base_load_to_target_peak(
            graph=graph_input,
            target_peak_gw=target_peak_gw,
            tx_regions=regions_for_peak,
        )
        # print(
        #     f"Base load peak scaling is ON. "
        #     f"Mode = {peak_region_mode}. "
        #     f"Original peak = {current_peak_gw:.2f} GW. "
        #     f"Target peak = {target_peak_gw:.2f} GW. "
        #     f"Scale factor = {load_scale_factor:.4f}."
        # )
    else:
        graph = graph_input
        print("Base load peak scaling is OFF. Using original graph load.")
    # Update scenario-specific RPS
    # ========================================
    # Update CA RPS policy after loading old policy file
    # ========================================
    if scenario_id not in ctx.scenarios:
        raise KeyError(
            f"Scenario ID {scenario_id} is not in ctx.scenarios. "
            f"Available IDs are: {list(ctx.scenarios.keys())[:10]}..."
        )

    cfg = ctx.scenarios[scenario_id]
    # ========================================
    # Remove old state-specific policies that will be rebuilt
    # ========================================
    keys_to_remove = [
        k for k in policies_input.keys()
        if k.startswith("rps_") or k.startswith("rm_")
    ]

    for k in keys_to_remove:
        del policies_input[k]
    # ========================================
    # Add RPS policy
    # ========================================
    policy_region = model_region
    region_type = cfg.get("region_type", "single_state_region")

    if region_type == "multi_state_region":

        if state_rps_policies is None:
            state_rps_policies = cfg.get("state_rps_policies", None)

        if state_rps_policies is None:
            raise ValueError(
                "This scenario is a multi state region, but state_rps_policies is None. "
                "Pass state_rps_policies to run_one_scenario."
            )

        included_states = cfg.get("included_states", list(state_rps_policies.keys()))

        for state_jurisdiction in included_states:

            state_policy = state_rps_policies.get(state_jurisdiction, {})
            state_rps_ratio = state_policy.get("rps_ratio", 0.0)

            if state_rps_ratio <= 0:
                continue

            rps_policy_key = f"rps_{policy_region}_{state_jurisdiction}"

            policies_input[rps_policy_key] = {
                "type": "rps",
                "_class": "Portfolio_Standard",
                "ratio": state_rps_ratio,
                "non_compliance_capacity": state_policy.get(
                    "non_compliance_capacity",
                    cfg["rps_non_compliance_capacity"],
                ),
                "non_compliance_cost": state_policy.get(
                    "non_compliance_cost",
                    cfg["rps_non_compliance_cost"],
                ),
                "region_handles": regions_for_model,
                "inclusion_criteria": {
                    # 0: "lambda a: a.get('renewable', False)",
                    0: "lambda a: a.get('rps_eligible', False)",
                    1: "lambda a: a.get('_class', '') != 'Store'",
                    2: "lambda a: a.get('type', '') != 'load'",
                    3: f"lambda a: a.get('jurisdiction', '') == '{state_jurisdiction}'",
                },
                "exclusion_criteria": {
                    # 0: "lambda a: not a.get('renewable', False)",
                    0: "lambda a: not a.get('rps_eligible', False)",
                    1: "lambda a: a.get('_class', '') != 'Store'",
                    2: "lambda a: a.get('type', '') != 'load'",
                    3: f"lambda a: a.get('jurisdiction', '') == '{state_jurisdiction}'",
                },
            }

    else:

        policy_jurisdiction = cfg.get("jurisdiction", ev_state)
        rps_policy_key = f"rps_{policy_region}"

        policies_input[rps_policy_key] = {
            "type": "rps",
            "_class": "Portfolio_Standard",
            "ratio": cfg["rps_ratio"],
            "non_compliance_capacity": cfg["rps_non_compliance_capacity"],
            "non_compliance_cost": cfg["rps_non_compliance_cost"],
            "region_handles": regions_for_model,
            "inclusion_criteria": {
                # 0: "lambda a: a.get('renewable', False)",
                0: "lambda a: a.get('rps_eligible', False)",
                1: "lambda a: a.get('_class', '') != 'Store'",
                2: "lambda a: a.get('type', '') != 'load'",
                3: f"lambda a: a.get('jurisdiction', '') == '{policy_jurisdiction}'",
            },
            "exclusion_criteria": {
                # 0: "lambda a: not a.get('renewable', False)",
                0: "lambda a: not a.get('rps_eligible', False)",
                1: "lambda a: a.get('_class', '') != 'Store'",
                2: "lambda a: a.get('type', '') != 'load'",
                3: f"lambda a: a.get('jurisdiction', '') == '{policy_jurisdiction}'",
            },
        }

    # ========================================
    # Add reserve margin policy
    # ========================================

    rm_policy_key = f"rm_{policy_region}"

    policies_input[rm_policy_key] = {
        "_class": "Reserve_Margin",
        "margin": cfg["reserve_margin"],
        "sign": cfg["reserve_margin_sign"],
        "non_compliance_capacity": cfg["reserve_non_compliance_capacity"],
        "non_compliance_cost": cfg["reserve_non_compliance_cost"],
        "demand_criteria": {
            2: "lambda a: a.get('type', '') == 'load'",
        },
        "supply_criteria": {
            2: "lambda a: a.get('type', '') != 'load'",
        },
    }

    # ========================================
    # Print policy check
    # ========================================

    created_rps_policies = [
        k for k in policies_input.keys()
        if k.startswith("rps_")
    ]

    print("Created RPS policies:")
    for k in created_rps_policies:
        print(
            f"  {k}: ratio={policies_input[k]['ratio']}, "
            f"regions={policies_input[k]['region_handles']}"
        )

        # print("    INCLUSION:", policies_input[k]["inclusion_criteria"])
        # print("    EXCLUSION:", policies_input[k]["exclusion_criteria"])

    # print(f"Created reserve margin policy: {rm_policy_key}")

    def remove_batteries_only(graph):
        """Remove only battery storage, keep pumped hydro."""
        battery_types = {'storage', 'battery'}

        total_removed = 0

        # print("\n" + "="*80)
        # print("REMOVING BATTERY STORAGE (keeping pumped hydro)")
        # print("="*80)

        for source, node in graph._node.items():
            assets = node.get('assets', {})
            assets_to_remove = []

            for key, asset in assets.items():
                asset_type = asset.get('type', '').lower()
                asset_fuel = asset.get('fuel', '').lower()

                # Remove batteries but NOT pumped hydro
                is_battery = (
                        (asset_type in battery_types or asset_fuel in battery_types) and
                        'pump' not in asset_type and
                        'pump' not in asset_fuel and
                        'optional_storage' not in key.lower()
                )

                # Also remove optional_storage entries
                is_optional_storage = 'optional_storage' in key.lower()

                if is_battery or is_optional_storage:
                    assets_to_remove.append(key)

            for key in assets_to_remove:
                # print(f"  {source}: Removed '{key}'")
                del assets[key]
                total_removed += 1

        print(f"Total battery assets removed: {total_removed}")
        print("=" * 80 + "\n")

        return graph
    # Clean the graph
    graph = remove_batteries_only(graph)
    # ========================================
    # Import EV load
    # ========================================
    ev_market = cfg.get("ev_market", ev_state)

    print(f"EV profile market key: {ev_market}")
    ev_region_inputs = helper.get_ipm_ev_profiles(
        ev_data=ctx.ev_data,
        year=year,
        adoption=adoption,
        market=model_region,
        ipm_regions=regions_for_model,
        charging_scenario=charging,
    )
    if ev_region_inputs is None:
        raise ValueError(
            "helper.get_ipm_ev_profiles returned None. "
            f"Check ev_data keys, year={year}, adoption={adoption}, "
            f"ev_market={ev_market}, charging_scenario={charging}."
        )

    if not ev_region_inputs:
        raise ValueError(
            "helper.get_ipm_ev_profiles returned an empty dictionary. "
            f"Check ipm_regions={regions_for_model}."
        )

    # print("EV regions returned:", list(ev_region_inputs.keys())[:10])

    # Check that each EV region has valid data
    bad_ev_regions = []

    for region, region_data in ev_region_inputs.items():

        if region_data is None:
            bad_ev_regions.append((region, "region_data is None"))
            continue

        if not isinstance(region_data, dict):
            bad_ev_regions.append((region, f"region_data is {type(region_data)}"))
            continue

        if "hourly_profile" not in region_data:
            bad_ev_regions.append((region, "missing hourly_profile"))
            continue

        if "peak_w" not in region_data:
            bad_ev_regions.append((region, "missing peak_w"))
            continue

        if region_data["hourly_profile"] is None:
            bad_ev_regions.append((region, "hourly_profile is None"))
            continue

    if bad_ev_regions:
        raise ValueError(
            "Invalid EV region inputs returned by helper.get_ipm_ev_profiles:\n"
            + "\n".join([f"{region}: {problem}" for region, problem in bad_ev_regions])
        )

    example_region = next(iter(ev_region_inputs))
    # print("Example EV region:", example_region)
    # print("Example EV data keys:", ev_region_inputs[example_region].keys())
    # print("Example EV profile length:", len(ev_region_inputs[example_region]["hourly_profile"]))
    # print("Example EV peak_w:", ev_region_inputs[example_region]["peak_w"])


    # Use one region only to define the time length
    example_region = next(iter(ev_region_inputs))
    n_ev_hours = len(ev_region_inputs[example_region]["hourly_profile"])
    start_hour = month * 32 * 24
    end_hour = start_hour + 24 * day_duration
    if end_hour > n_ev_hours:
        raise ValueError(
            f"Requested time window ends at {end_hour}, "
            f"but EV profile has only {n_ev_hours} hours."
        )
    n_hours = end_hour - start_hour
    # ========================================
    # Add additional fixed load, e.g. data centers
    # ========================================
    graph = add_fixed_datacenter_load_to_graph(
        graph=graph,
        load_profiles_mw_by_region=additional_fixed_load_mw_by_region,
        start_hour=start_hour,
        n_ev_hours=n_ev_hours,
        load_name_prefix="datacenter",
    )
    # ========================================
    # V1G, V2G, and stationary battery settings
    # ========================================
    flex = helper.build_flexibility_settings(
        cfg=cfg,
        state=model_region,
        n_hours=n_hours,
        n_ev_hours=n_ev_hours,
        flexibility_policy=flexibility_policy,
        use_state_default_flexibility=use_state_default_flexibility,
    )

    if flex is None and ev_state != model_region:
        print(
            f"No flexibility settings found for model_region={model_region}. "
            f"Retrying with ev_state={ev_state}."
        )

        flex = helper.build_flexibility_settings(
            cfg=cfg,
            state=ev_state,
            n_hours=n_hours,
            n_ev_hours=n_ev_hours,
            flexibility_policy=flexibility_policy,
            use_state_default_flexibility=use_state_default_flexibility,
        )

    if flex is None:
        raise ValueError(
            "helper.build_flexibility_settings returned None. "
            f"Checked model_region={model_region} and ev_state={ev_state}. "
            "Check the default flexibility policy dictionary or return statement "
            "inside helper.build_flexibility_settings()."
        )

    required_flex_keys = [
        "V1G_PARTICIPATION",
        "V2G_PARTICIPATION",
        "TOTAL_PARTICIPATION",
        "FIXED_SHARE",
        "EV_SHIFT_WINDOW",
        "EV_SHIFT_COST",
        "V1G_FIX_COST",
        "V2G_WINDOW_HOURS",
        "HOURS_EV",
        "ETA_EV",
        "SHIFT_COST",
        "V2G_FIX_COST",
        "HOURS_ST",
        "ETA_CHARGE_ST",
        "ETA_DISCHARGE_ST",
        "P_TOTAL_MW_ST",
        "BATT_FIX_COST",
        "BATT_OPT_COST",
        "BATT_CAPEX_COST",
        "BATTERY_INITIAL_SOC_FRACTION",
    ]

    missing_flex_keys = [key for key in required_flex_keys if key not in flex]

    if missing_flex_keys:
        raise KeyError(
            f"Flexibility settings are missing these keys: {missing_flex_keys}"
        )


    V1G_PARTICIPATION = flex["V1G_PARTICIPATION"]
    V2G_PARTICIPATION = flex["V2G_PARTICIPATION"]
    TOTAL_PARTICIPATION = flex["TOTAL_PARTICIPATION"]
    FIXED_SHARE = flex["FIXED_SHARE"]
    EV_SHIFT_WINDOW = flex["EV_SHIFT_WINDOW"]
    EV_SHIFT_COST = flex["EV_SHIFT_COST"]
    V1G_FIX_COST = flex["V1G_FIX_COST"]
    V2G_WINDOW_HOURS = flex["V2G_WINDOW_HOURS"]
    HOURS_EV = flex["HOURS_EV"]
    ETA_EV = flex["ETA_EV"]
    SHIFT_COST = flex["SHIFT_COST"]
    V2G_FIX_COST = flex["V2G_FIX_COST"]
    HOURS_ST = flex["HOURS_ST"]
    ETA_CHARGE_ST = flex["ETA_CHARGE_ST"]
    ETA_DISCHARGE_ST = flex["ETA_DISCHARGE_ST"]
    P_TOTAL_MW_ST = flex["P_TOTAL_MW_ST"]
    BATT_FIX_COST = flex["BATT_FIX_COST"]
    BATT_OPT_COST = flex["BATT_OPT_COST"]
    BATT_CAPEX_COST = flex["BATT_CAPEX_COST"]
    BATTERY_INITIAL_SOC_FRACTION = flex["BATTERY_INITIAL_SOC_FRACTION"]
    # ========================================
    # Add EV demand
    # ========================================
    for source, node in graph._node.items():
        if source not in ev_region_inputs:
            continue
        regional_profile = ev_region_inputs[source]["hourly_profile"]
        regional_capacity_w = ev_region_inputs[source]["peak_w"]
        if regional_capacity_w <= 0:
            continue
        # 1) Fixed EV load
        if FIXED_SHARE > 0:
            node["assets"][f"ev_load_fixed_{source}"] = {
                "_class": "Load",
                "type": "load",
                "installed_capacity": -regional_capacity_w * FIXED_SHARE,
                "profile": regional_profile,
            }
        # 2) V1G EV load
        if V1G_PARTICIPATION > 0:
            node["assets"][f"ev_load_v1g_{source}"] = {
                "_class": "Load",
                "type": "load",
                "installed_capacity": -regional_capacity_w * V1G_PARTICIPATION,
                "profile": regional_profile,
            }
        # 3) V2G EV load
        if V2G_PARTICIPATION > 0:
            node["assets"][f"ev_load_v2g_{source}"] = {
                "_class": "Load",
                "type": "load",
                "installed_capacity": -regional_capacity_w * V2G_PARTICIPATION,
                "profile": regional_profile,
            }
    # ========================================
    # Add V1G load shifting
    # ========================================
    for source, node in graph._node.items():
        for handle, asset in node["assets"].items():
            if asset.get("_class") != "Load" or asset.get("type") != "load":
                continue

            # Match the detailed formulation for savings:
            #   - signed shift variable
            #   - daily/window net-zero energy
            #   - physical bounds on realized load
            #
            # Important difference from the previous simple version:
            # the cumulative formulation was causal delay-only. That is more
            # physically strict, but it is NOT equivalent to the detailed
            # model. The detailed model allows two-sided reshaping inside each
            # window, which is why it produced larger savings.
            if handle.startswith("ev_load_v1g_") or handle.startswith("ev_load_v2g_"):
                asset["shift_capacity"] = 1.0
                asset["shift_fraction"] = 1.0
                asset["shift_formulation"] = "legacy_window"
                asset["shift_window"] = EV_SHIFT_WINDOW
                asset["capex_capacity"] = 0

                # Keep smart charging as an operational option. With the
                # current policy this is normally zero, but this keeps the
                # field connected to the existing settings.
                asset["shift_cost"] = EV_SHIFT_COST
                asset["shift_cost_mode"] = "none"

                if handle.startswith("ev_load_v1g_"):
                    asset["fixed_shift_cost"] = V1G_FIX_COST
                else:
                    asset["fixed_shift_cost"] = 0.0

                # Do not charge a fixed program cost unless explicitly studied.
                asset["charge_fixed_shift_cost"] = False

                # Keep the same physical realized-load bounds as the detailed
                # Load class.
                asset["enforce_physical_shift_bounds"] = True

                # Remove cumulative-only fields if this graph was copied from
                # an earlier run context.
                asset.pop("flexible_power_profile", None)
                asset.pop("shift_power_capacity_profile", None)
                asset.pop("ensure_baseline_feasible", None)
                asset.pop("enforce_final_flex_balance", None)

            elif handle.startswith("ev_load_fixed_"):
                asset["shift_capacity"] = 0.0
                asset["shift_window"] = 0
                asset["shift_formulation"] = "none"
                asset["capex_capacity"] = 0
                asset["shift_cost"] = 0.0
                asset["fixed_shift_cost"] = 0.0

            elif handle.startswith("base_load_"):
                asset["shift_capacity"] = 0.0
                asset["shift_window"] = 0
                asset["shift_formulation"] = "none"
                asset["capex_capacity"] = 0
                asset["shift_cost"] = 0.0
                asset["fixed_shift_cost"] = 0.0

            else:
                asset["shift_capacity"] = 0.0
                asset["shift_window"] = 0
                asset["shift_formulation"] = "none"
                asset["capex_capacity"] = 0
                asset["shift_cost"] = 0.0
                asset["fixed_shift_cost"] = 0.0

    # ========================================
    # Add stationary battery
    # ========================================
    assert abs(sum(ctx.battery_weights.values()) - 1.0) < 1e-6
    for source, node in graph._node.items():
        if source not in ctx.battery_weights:
            continue
        # Existing battery power in MW for this region
        P_MW = P_TOTAL_MW_ST * ctx.battery_weights[source]
        P_W = P_MW * 1e6
        # Existing battery energy in J
        E_J = P_W * (HOURS_ST * 3600)
        # If you want battery expansion, define max expansion separately.
        # For now this keeps expansion disabled.
        CAPEX_POWER_W = P_W * 100
        CAPEX_ENERGY_J = CAPEX_POWER_W * (HOURS_ST * 3600)
        node["assets"][f"battery_{source}"] = {
            "_class": "Store",
            "type": "battery",
            "fuel": "battery",
            # Existing capacities
            "installed_capacity": E_J,  # energy capacity [J]
            "installed_power_capacity": P_W,  # power capacity [W]
            # Expansion bounds
            "capex_capacity": CAPEX_ENERGY_J,  # energy expansion bound [J]
            "capex_power_capacity": CAPEX_POWER_W,  # power expansion bound [W]
            # Costs
            # BATT_CAPEX_COST should be $/J if it is battery pack cost
            "capex_cost_energy": BATT_CAPEX_COST,
            "capex_cost_power": 0.0,
            # Fixed O&M
            # Keep this only if BATT_FIX_COST is in $/J-year.
            # If your fixed O&M is actually $/W-year, then it should be a separate power-based term,
            # not this field.
            "fixed_operating_cost": BATT_FIX_COST,
            # Variable throughput cost ($/J throughput)
            "operating_cost": BATT_OPT_COST,
            # Battery physics
            "charge_efficiency": ETA_CHARGE_ST,
            "discharge_efficiency": ETA_DISCHARGE_ST,
            "production_rate": 1 / (HOURS_ST * 3600),
            "consumption_rate": 1 / (HOURS_ST * 3600),
            "duration_hours": HOURS_ST,
            # Start full
            "initial": BATTERY_INITIAL_SOC_FRACTION * E_J,
            "reset_each_window": False,
        }
    # ========================================
    # Add V2G storage
    # ========================================
    for source, node in graph._node.items():
        ev_handles = [h for h in node["assets"] if h.startswith("ev_load_v2g_")]
        if not ev_handles:
            continue

        ev_handle = ev_handles[0]
        ev_asset = node["assets"][ev_handle]
        P_ev_total = abs(float(ev_asset.get("installed_capacity", 0.0)))

        if P_ev_total <= 1e-9:
            continue

        region_data = ev_region_inputs[source]
        region_profile_full = np.asarray(region_data["hourly_profile"], dtype=float)

        power_capacity_profile_w, energy_capacity_profile_j = (
            build_detail_equivalent_v2g_capacity_profiles(
                region_profile_full=region_profile_full,
                peak_power_w=P_ev_total,
                hours_ev=HOURS_EV,
                window_hours=V2G_WINDOW_HOURS,
                n_ev_hours=n_ev_hours,
                start_hour=start_hour,
                end_hour=end_hour,
            )
        )

        max_power_w = float(np.max(power_capacity_profile_w))
        max_energy_j = float(np.max(energy_capacity_profile_j))

        if max_power_w <= 1e-9 or max_energy_j <= 1e-9:
            continue

        node["assets"][f"ev_v2g_store_{source}"] = {
            "_class": "Store",
            "type": "ev_v2g",
            "fuel": "ev_v2g",

            # Reporting/nameplate capacities.
            "installed_capacity": max_energy_j,
            "installed_power_capacity": max_power_w,

            # Detail-equivalent aggregate envelope:
            # at each hour this equals the sum of all detailed V2G cohorts
            # that would have been active in the old runner.
            "energy_capacity_profile": energy_capacity_profile_j.tolist(),
            "charge_power_capacity_profile": power_capacity_profile_w.tolist(),
            "discharge_power_capacity_profile": power_capacity_profile_w.tolist(),

            # No EV battery expansion. This is driver-provided flexibility.
            "capex_capacity": 0,
            "capex_power_capacity": 0,
            "capex_cost": 0.0,
            "capex_cost_energy": 0.0,
            "capex_cost_power": 0.0,

            # Costs and physics.
            "efficiency": ETA_EV,
            "fixed_operating_cost": V2G_FIX_COST,
            "operating_cost": SHIFT_COST,
            "production_rate": 1 / (HOURS_EV * 3600),
            "consumption_rate": 1 / (HOURS_EV * 3600),
            "duration_hours": HOURS_EV,

            # This aggregate envelope already carries the V2G window through
            # the rolling cohort-sum profile. A fixed midnight-to-midnight reset
            # would make it artificially more restrictive than the detailed
            # cohort model, so only the final level is reset.
            "initial": 0.0,
            "reset_each_window": False,
            "reset_interval_hours": None,
            "enforce_final_level": True,

            # Match the detailed model's feasible set more closely. The earlier
            # simple version added a combined charge/discharge budget, which
            # was more restrictive than the detailed Store class.
            "combined_charge_discharge_limit": False,
            "charge_fixed_operating_cost": False,

            # Faster Store path.
            "direct_throughput_cost": True,
            "skip_availability_param": True,
        }

        print(
            f"Added detail-equivalent aggregate V2G store to {source}: "
            f"max_power={max_power_w / 1e6:.2f} MW, "
            f"max_energy={max_energy_j / 3.6e9:.2f} MWh, "
            f"window_hours={V2G_WINDOW_HOURS}"
        )
    # ========================================
    # Build network
    # ========================================
    deep_reload(good)
    build_kw = {
        "verbose": True,
        "steps": (month * 32 * 24, month * 32 * 24 + 24 * day_duration),
        "amortization_period": 31536000,
        "shortfall_capacity": np.inf,
        "shortfall_cost": 1e-4,
        "wastage_capacity": np.inf,
        "wastage_cost": 1e-9*0,
    }
    # ========================================
    # Apply economic assumptions
    # ========================================
    economic_settings = helper.get_economic_policy(
        state=model_region,
        discount_rate=discount_rate,
        lifetime=lifetime,
        economic_policy=economic_policy,
        use_state_default_economic_policy=use_state_default_economic_policy,
    )
    graph = helper.apply_economic_policy_to_graph(
        graph=graph,
        economic_policy=economic_settings,
        verbose=False,
    )
    # ========================================
    # Apply transmission assumptions
    # ========================================
    transmission_settings = helper.get_transmission_policy(
        state=model_region,
        transmission_policy=transmission_policy,
        use_state_default_transmission_policy=use_state_default_transmission_policy,
    )
    graph = helper.apply_transmission_policy_to_graph(
        graph=graph,
        transmission_policy=transmission_settings,
        verbose=False,
    )
    # ========================================
    # Apply retirement policy
    # Priority:
    # 1. Function argument retirement_policy
    # 2. Scenario-specific retirement_policy from cfg
    # 3. Context-level default retirement policy
    # ========================================
    if retirement_policy is None:
        retirement_policy = cfg.get("retirement_policy", None)

    if retirement_policy is None and use_state_default_retirement:
        retirement_policy = get_policy_with_fallback(
            ctx.retirement_policies,
            model_region=model_region,
            ev_state=ev_state,
        )

    graph = helper.apply_retirement_policy_to_graph(
        graph=graph,
        retirement_policy=retirement_policy,
        verbose=False,
    )
    exact_renewable_groups = diagnose_exact_renewable_groups(
    graph
    )
    #test small hydro
    # for node in graph._node.values():
    #     for asset in node.get("assets", {}).values():
    #         if asset.get("type") == "small_hydro":
    #             asset["installed_capacity"] *= 10
    rps_membership_df = print_rps_asset_membership(
        graph=graph,
        policies_input=policies_input,
    )
    network = good.optimization.network.Network(**build_kw).from_graph(graph, policies_input)
    # ========================================
    # Apply ramp rate and must-run policy
    # ========================================
    if asset_constraint_policy is None and use_state_default_asset_constraints:
        asset_constraint_policy = get_policy_with_fallback(
            ctx.asset_constraint_policies,
            model_region=model_region,
            ev_state=ev_state,
        )
    network = helper.apply_asset_constraints_to_network(
        network=network,
        asset_constraint_policy=asset_constraint_policy,
        verbose=False,
    )
    # print("model step range example:", start_hour, start_hour + 15)
    network.build()
    # ========================================
    # Solve
    # ========================================
    solve_kw = {
        "solver": {
            "_name": "cplex",
            "executable": ctx.solver_executable,

            # "threads": 10,
            "threads": int(getattr(ctx, "solver_threads", 1)),
            "randomseed": 12345,
            "lpmethod": 4,

            "output clonelog": -1,

            "barrier convergetol": 1e-8,
            "simplex tolerances feasibility": 1e-8,
            "simplex tolerances optimality": 1e-8,
            "emphasis numerical": 1,
        },
    }
    network.solve(**solve_kw)
    solution = network.solution_graph()

    solver_status = network.result.solver.status
    termination = network.result.solver.termination_condition

    print("=" * 80)
    print("SOLVER STATUS CHECK")
    print("=" * 80)
    print("Solver status:", solver_status)
    print("Termination condition:", termination)
    print("=" * 80)

    valid_terminations = {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
    }

    if solver_status != SolverStatus.ok or termination not in valid_terminations:
        raise RuntimeError(
            f"Solver did not return a valid solution.\n"
            f"Status: {solver_status}\n"
            f"Termination condition: {termination}"
        )

    ### test
    n_steps = len(list(network.model.steps))

    solution = fill_empty_solution_lists(
        solution=solution,
        n_steps=n_steps,
        fields_to_fill={"clearing_price"},
    )
    # ========================================
    # Build dataframe only after the check passes
    # ========================================
    dataframe = network.solution_dataframe(solution)
    objective_value = pe.value(network.model.objective)
    # ========================================
    # Debug outputs
    # ========================================
    debug_results = run_full_debug_suite(
    network=network,
    scenario_id=scenario_id,
    cfg=cfg,
    battery_hours=4,
    check_simultaneous=True,
    run_energy_balance=True,
    run_ev_debug=True,
    run_generation_table=True,
    )
    # ========================================
    # Save compact cost components for plotting
    # ========================================
    cost_breakdown = debug_results["cost_breakdown"]
    cost_components_row = pd.DataFrame([{
        "scenario_id": scenario_id,
        "scenario_tag": cfg.get("scenario_tag", ""),
        "objective_value": float(objective_value),
        "capex_total": float(cost_breakdown.get("capex_total", 0.0)),
        "asset_opex_total": float(cost_breakdown.get("opex_total", 0.0)),
        "line_opex_total": float(cost_breakdown.get("line_opex_total", 0.0)),
        "fixed_total": float(cost_breakdown.get("fixed_total", 0.0)),
        "penalty_total": float(cost_breakdown.get("penalty_total", 0.0)),
        "other_gap": float(cost_breakdown.get("other_gap", 0.0)),
        "reconstructed_total": float(cost_breakdown.get("reconstructed_total", 0.0)),
        "model_total": float(cost_breakdown.get("model_total", objective_value)),
    }])

    scenario_tag = cfg.get("scenario_tag", f"s{scenario_id:02d}")
    scenario_output_dir = os.path.join(
        ctx.results_dir,
        f"s{scenario_id:02d}_{scenario_tag}_{adoption}_m{month}_d{day_duration}"
    )
    os.makedirs(scenario_output_dir, exist_ok=True)
    cost_components_path = os.path.join(
        scenario_output_dir,
        f"s{scenario_id:02d}_{scenario_tag}_cost_components.csv"
    )
    cost_components_row.to_csv(cost_components_path, index=False)
    print(f"Cost components CSV: {cost_components_path}")

    capex_paths = save_capacity_and_capex_tables(
        debug_results=debug_results,
        scenario_id=scenario_id,
        cfg=cfg,
        scenario_output_dir=scenario_output_dir,
        scenario_tag=scenario_tag,
        adoption=adoption,
        charging=charging,
        month=month,
        day_duration=day_duration,
    )
    ev_paths = save_ev_summary(
        debug_results=debug_results,
        scenario_id=scenario_id,
        cfg=cfg,
        scenario_output_dir=scenario_output_dir,
        scenario_tag=scenario_tag,
    )
    # ========================================
    # Save outputs
    # ========================================
    results = save_scenario_results(
        scenario_id=scenario_id,
        cfg=cfg,
        solution=solution,
        dataframe=dataframe,
        objective_value=objective_value,
        network=network,
        RESULTS_DIR=ctx.results_dir,
        year=year,
        adoption=adoption,
        charging=charging,
        month=month,
        day_duration=day_duration,
    )
    results["cost_components_csv"] = cost_components_path
    results.update(capex_paths)
    results.update(ev_paths)
    return {**results,"debug_results": debug_results}

def run_one_scenario_parallel_task(task):
    """
    Parallel-safe wrapper for one GOOD scenario.

    This function is defined in s_runner.py, not in the notebook, so
    ProcessPoolExecutor can import it cleanly on macOS/Jupyter.
    """

    import traceback

    scenario_id = task["scenario_id"]
    ctx = task["ctx"]
    cfg = ctx.scenarios[scenario_id]

    try:
        result = run_one_scenario(
            scenario_id=scenario_id,
            ctx=ctx,
            year=task["year"],
            adoption=task["adoption"],
            charging=task["charging"],
            month=task["month"],
            day_duration=task["day_duration"],
            model_region=task["model_region"],
            discount_rate=task["discount_rate"],
            lifetime=task["lifetime"],
            fix_peak=task["fix_peak"],
            target_peak_gw=task["target_peak_gw"],
            peak_region_mode=task["peak_region_mode"],
            peak_regions=task["peak_regions"],
            retirement_policy=task["retirement_policy"],
            asset_constraint_policy=task["asset_constraint_policy"],
            use_state_default_retirement=task["use_state_default_retirement"],
            use_state_default_asset_constraints=task["use_state_default_asset_constraints"],
            flexibility_policy=task["flexibility_policy"],
            use_state_default_flexibility=task["use_state_default_flexibility"],
            economic_policy=task["economic_policy"],
            additional_fixed_load_mw_by_region=task.get("additional_fixed_load_mw_by_region", None),
            use_state_default_economic_policy=task["use_state_default_economic_policy"],
            transmission_policy=task["transmission_policy"],
            use_state_default_transmission_policy=task["use_state_default_transmission_policy"],
        )

        row = {
            "scenario_id": result["scenario_id"],
            "tag": result["tag"],
            "group": cfg["group"],
            "rps_ratio": cfg["rps_ratio"],
            "v1g_share": cfg["v1g_share"],
            "v2g_share": cfg["v2g_share"],
            "batt_capex_cost": cfg["batt_capex_cost"],
            "batt_capex_label": cfg["batt_capex_label"],
            "objective_value": result["objective_value"],
            "out_dir": result["out_dir"],
            "solution_json_path": result["solution_json_path"],
            "solution_csv_path": result["solution_csv_path"],
            "summary_csv_path": result["summary_csv_path"],
            "total_cost_csv_path": result["total_cost_csv_path"],
            "adoption_level": task["adoption"],
            "charging_scenario": task["charging_name"],
            "charging_description": task["charging_description"],
        }

        return {
            "ok": True,
            "scenario_id": scenario_id,
            "row": row,
            "error": None,
            "traceback": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "scenario_id": scenario_id,
            "row": None,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }