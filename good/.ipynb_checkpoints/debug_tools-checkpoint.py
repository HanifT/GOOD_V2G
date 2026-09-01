import pyomo.environ as pyo
import pandas as pd

def debug_objective_gap(network):
    print("\n" + "=" * 100)
    print("OBJECTIVE GAP CHECK")
    print("=" * 100)

    model = network.model

    asset_obj = 0.0
    for name, asset in network.assets.items():
        if isinstance(asset, dict) and "object" in asset:
            asset_obj += pyo.value(asset["object"].objective(model))

    # Sum edge objectives (Links containing transmission lines)
    line_obj = 0.0
    for source, _adj in network.graph._adj.items():
        for target, edge in _adj.items():
            if "object" in edge:
                line_obj += pyo.value(edge["object"].objective(model))

    policy_obj = 0.0
    for name, policy in network.policies.items():
        if isinstance(policy, dict) and "object" in policy:
            policy_obj += pyo.value(policy["object"].objective(model))

    model_obj = pyo.value(model.objective)
    gap = model_obj - (asset_obj + line_obj + policy_obj)

    # print(f"asset objective = {asset_obj:,.2f}")
    # print(f"line objective  = {line_obj:,.2f}")
    # print(f"policy objective= {policy_obj:,.2f}")
    # print(f"model objective = {model_obj:,.2f}")
    # print(f"gap             = {gap:,.8f}")

    return {
        "asset_objective": asset_obj,
        "line_objective": line_obj,
        "policy_objective": policy_obj,
        "model_objective": model_obj,
        "gap": gap,
    }

def debug_capacity_and_cost_summary(network, scenario_id, cfg, battery_hours=4):
    model = network.model

    print("\n" + "=" * 140)
    print(f"SCENARIO {scenario_id} CLEAN CAPACITY / CAPEX DEBUG")
    print(f"RPS Target: {cfg['rps_ratio'] * 100:.1f}%")
    print("=" * 140)

    # -----------------------------
    # RPS compliance
    # -----------------------------
    rps_results = {}
    rps_policy = network.policies.get("rps_CA")
    if rps_policy:
        included_gen = sum(
            pyo.value(asset["object"].power(model))
            for asset in rps_policy["object"].included.values()
        )
        excluded_gen = sum(
            pyo.value(asset["object"].power(model))
            for asset in rps_policy["object"].excluded.values()
        )
        total_gen = included_gen + excluded_gen
        actual_rps = included_gen / total_gen if total_gen > 0 else 0.0

        non_compliance = pyo.value(model.component("rps_CA::non_compliance"))

        # print("\nRPS Compliance:")
        # print(f"  Target: {cfg['rps_ratio'] * 100:.2f}%")
        # print(f"  Actual: {actual_rps * 100:.4f}%")
        # print(f"  Non-compliance: {non_compliance:.6f}")
        # print(f"  Binding: {'YES' if abs(actual_rps - cfg['rps_ratio']) < 1e-4 else 'NO'}")

        rps_results = {
            "target": cfg["rps_ratio"],
            "actual": actual_rps,
            "non_compliance": non_compliance,
        }

    # -----------------------------
    # Gas generation
    # -----------------------------
    gas_gen_MWh = 0.0
    for asset_key, asset_data in network.assets.items():
        if "natural gas" in asset_data.get("fuel", "").lower():
            if hasattr(model, f"{asset_key}::production"):
                prod = getattr(model, f"{asset_key}::production")
                gas_gen_MWh += sum(
                    pyo.value(prod[t]) * pyo.value(model.time_step) / 3.6e9
                    for t in model.steps
                )

    print(f"\nGas Generation: {gas_gen_MWh:,.0f} MWh")

    # -----------------------------
    # Capacity / capex table
    # -----------------------------
    rows = []

    for asset_key, asset_data in network.assets.items():
        asset = asset_data["object"]
        fuel = asset_data.get("fuel", "unknown")
        asset_class = asset.__class__.__name__
        is_store = asset_class == "Store"

        capex_val = 0.0
        if hasattr(model, f"{asset_key}::capex"):
            capex_val = pyo.value(getattr(model, f"{asset_key}::capex"))

        installed_capacity_raw = getattr(asset, "installed_capacity", 0.0)

        if is_store:
            duration_hours = asset_data.get("duration_hours", battery_hours)
            duration_sec = duration_hours * 3600
            installed_capacity_MW = installed_capacity_raw / duration_sec / 1e6
            capex_capacity_MW = capex_val / duration_sec / 1e6
        else:
            installed_capacity_MW = installed_capacity_raw / 1e6
            capex_capacity_MW = capex_val / 1e6

        total_capacity_MW = installed_capacity_MW + capex_capacity_MW

        capex_cost_coeff = asset_data.get("capex_cost", asset_data.get("capex_cost_energy", 0.0))

        amort = pyo.value(model.amortization) if hasattr(model, "amortization") else 1.0
        capex_cost_term = capex_val * capex_cost_coeff * amort

        if is_store:
            duration_hours = asset_data.get("duration_hours", battery_hours)
            duration_sec = duration_hours * 3600
            avg_capex_cost_per_MW = capex_cost_coeff * duration_sec * 1e6
        else:
            avg_capex_cost_per_MW = capex_cost_coeff * 1e6

        rows.append({
            "asset": asset_key,
            "fuel": fuel,
            "class": asset_class,
            "is_store": is_store,
            "installed_MW": installed_capacity_MW,
            "capex_added_MW": capex_capacity_MW,
            "total_MW": total_capacity_MW,
            "capex_cost_coeff_raw": capex_cost_coeff,
            "avg_capex_cost_per_MW": avg_capex_cost_per_MW,
            "capex_cost_term_dollar": capex_cost_term,
        })

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("fuel", dropna=False)
        .agg(
            installed_MW=("installed_MW", "sum"),
            capex_added_MW=("capex_added_MW", "sum"),
            total_MW=("total_MW", "sum"),
            avg_capex_cost_per_MW=("avg_capex_cost_per_MW", "mean"),
            total_capex_cost=("capex_cost_term_dollar", "sum"),
            n_assets=("asset", "count"),
        )
        .reset_index()
    )

    total_wastage_MWh = 0.0
    total_shortfall_MWh = 0.0

    for source, node in network.graph._node.items():
        if hasattr(model, f"{source}::wastage"):
            wastage = getattr(model, f"{source}::wastage")
            total_wastage_MWh += sum(
                pyo.value(wastage[t]) * pyo.value(model.time_step) / 3.6e9
                for t in model.steps
            )

        if hasattr(model, f"{source}::shortfall"):
            shortfall = getattr(model, f"{source}::shortfall")
            total_shortfall_MWh += sum(
                pyo.value(shortfall[t]) * pyo.value(model.time_step) / 3.6e9
                for t in model.steps
            )

    print(
        f"{'Fuel':<30}"
        f"{'Installed MW':>15}"
        f"{'Capex MW':>15}"
        f"{'Total MW':>15}"
        f"{'Avg Capex $/MW':>20}"
        f"{'Capex Cost ($M)':>18}"
        f"{'#Assets':>10}"
    )
    print("-" * 140)

    for _, r in summary.sort_values("capex_added_MW", ascending=False).iterrows():
        print(
            f"{str(r['fuel']):<30}"
            f"{r['installed_MW']:>15,.2f}"
            f"{r['capex_added_MW']:>15,.2f}"
            f"{r['total_MW']:>15,.2f}"
            f"{r['avg_capex_cost_per_MW']:>20,.2f}"
            f"{r['total_capex_cost'] / 1e6:>18,.2f}"
            f"{int(r['n_assets']):>10}"
        )

    print("-" * 140)
    print(f"{'Total Wastage (MWh)':<30}{total_wastage_MWh:>15,.2f}")
    print(f"{'Total Shortfall (MWh)':<30}{total_shortfall_MWh:>15,.2f}")

    return {
        "rps": rps_results,
        "summary_table": summary,
        "detail_table": df,
        "wastage_MWh": total_wastage_MWh,
        "shortfall_MWh": total_shortfall_MWh,
        "gas_gen_MWh": gas_gen_MWh,
    }

def detailed_cost_breakdown(network, scenario_id, cfg):
    model = network.model

    print("\n" + "=" * 100)
    print(f"DETAILED COST BREAKDOWN - SCENARIO {scenario_id}")
    print("=" * 100 + "\n")

    total_cost = pyo.value(model.objective)
    print(f"Total Objective: ${total_cost:,.2f}\n")

    capex_breakdown = {}
    opex_breakdown = {}
    line_opex_breakdown = {}
    fixed_breakdown = {}
    penalty_costs = {}

    # -----------------------------
    # CAPEX
    # -----------------------------
    for asset_key, asset_data in network.assets.items():
        asset = asset_data["object"]
        asset_class = asset.__class__.__name__
        is_store = asset_class == "Store"
        duration_hours = asset_data.get("duration_hours", 4.0)
        duration_sec = duration_hours * 3600

        if hasattr(model, f"{asset_key}::capex"):
            capex_val = pyo.value(getattr(model, f"{asset_key}::capex"))
            amort = pyo.value(model.amortization) if hasattr(model, "amortization") else 1.0

            if is_store:
                capex_cost_energy = asset_data.get("capex_cost_energy", 0.0) or 0.0
                capex_cost_power = asset_data.get("capex_cost_power", 0.0) or 0.0

                # If your model only expands energy capacity for stores:
                capex_cost_term = capex_val * capex_cost_energy * amort

            else:
                capex_cost_coeff = asset_data.get("capex_cost", 0.0) or 0.0
                capex_cost_term = capex_val * capex_cost_coeff * amort

            fuel = asset_data.get("fuel", "unknown")
            if abs(capex_cost_term) > 0:
                capex_breakdown[fuel] = capex_breakdown.get(fuel, 0.0) + capex_cost_term

    # -----------------------------
    # ASSET OPEX
    # -----------------------------
    for asset_key, asset_data in network.assets.items():
        if hasattr(model, f"{asset_key}::production"):
            prod = getattr(model, f"{asset_key}::production")
            operating_cost = asset_data.get("operating_cost", 0.0) or 0.0

            cost = sum(
                pyo.value(prod[t]) * pyo.value(model.time_step) * operating_cost
                for t in model.steps
            )

            fuel = asset_data.get("fuel", "unknown")
            if abs(cost) > 0:
                opex_breakdown[fuel] = opex_breakdown.get(fuel, 0.0) + cost

    # Store cycling cost
    for asset_key, asset_data in network.assets.items():
        if asset_data.get("_class") == "Store":
            fuel = asset_data.get("fuel", "storage")

            if hasattr(model, f"{asset_key}::throughput"):
                throughput = getattr(model, f"{asset_key}::throughput")
                operating_cost_param = getattr(model, f"{asset_key}::operating_cost")
                cost = sum(
                    pyo.value(throughput[t]) * pyo.value(model.time_step) * pyo.value(operating_cost_param[t])
                    for t in model.steps
                )
            elif hasattr(model, f"{asset_key}::production"):
                production = getattr(model, f"{asset_key}::production")
                operating_cost = asset_data.get("operating_cost", 0.0) or 0.0
                cost = sum(
                    pyo.value(production[t]) * pyo.value(model.time_step) * operating_cost
                    for t in model.steps
                )
            else:
                cost = 0.0

            if abs(cost) > 0:
                opex_breakdown[fuel] = opex_breakdown.get(fuel, 0.0) + cost

    # -----------------------------
    # LINE OPEX
    # -----------------------------
    for source, _adj in network.graph._adj.items():
        for target, edge in _adj.items():
            line_name = edge.get("id", f"{source}->{target}")
            line_obj = edge.get("object", None)
            if line_obj is None:
                continue

            # try common variable names
            if hasattr(model, f"{line_name}::production"):
                flow = getattr(model, f"{line_name}::production")
                operating_cost = edge.get("operating_cost", 0.0) or 0.0
                cost = sum(
                    pyo.value(flow[t]) * pyo.value(model.time_step) * operating_cost
                    for t in model.steps
                )
            elif hasattr(model, f"{line_name}::flow"):
                flow = getattr(model, f"{line_name}::flow")
                operating_cost = edge.get("operating_cost", 0.0) or 0.0
                cost = sum(
                    abs(pyo.value(flow[t])) * pyo.value(model.time_step) * operating_cost
                    for t in model.steps
                )
            else:
                # fallback to object objective if available
                try:
                    cost = pyo.value(line_obj.objective(model))
                except Exception:
                    cost = 0.0

            if abs(cost) > 0:
                line_opex_breakdown[line_name] = line_opex_breakdown.get(line_name, 0.0) + cost

    # -----------------------------
    # FIXED
    # -----------------------------
    for asset_key, asset_data in network.assets.items():
        asset = asset_data["object"]

        if asset_data.get("_class") == "Store":
            capex_val = 0.0
            if hasattr(model, f"{asset_key}::capex"):
                capex_val = pyo.value(getattr(model, f"{asset_key}::capex"))

            installed = getattr(asset, "installed_capacity", 0.0)
            fixed_om = getattr(asset, "fixed_operating_cost", 0.0)
            amort = pyo.value(model.amortization)

            cost = (installed + capex_val) * fixed_om * amort
            fuel = asset_data.get("fuel", "storage_fixed_om")
            if abs(cost) > 0:
                fixed_breakdown[fuel] = fixed_breakdown.get(fuel, 0.0) + cost

    # -----------------------------
    # PENALTIES
    # -----------------------------
    for source, node in network.graph._node.items():
        if hasattr(model, f"{source}::shortfall"):
            shortfall = getattr(model, f"{source}::shortfall")
            cost = sum(
                pyo.value(shortfall[t]) * pyo.value(model.time_step)
                for t in model.steps
            ) * node["object"].shortfall_cost
            if cost > 0:
                penalty_costs["shortfall"] = penalty_costs.get("shortfall", 0.0) + cost

        if hasattr(model, f"{source}::wastage"):
            wastage = getattr(model, f"{source}::wastage")
            cost = sum(
                pyo.value(wastage[t]) * pyo.value(model.time_step)
                for t in model.steps
            ) * node["object"].wastage_cost
            if cost > 0:
                penalty_costs["wastage"] = penalty_costs.get("wastage", 0.0) + cost

    if "rps_CA" in network.policies:
        non_compliance = pyo.value(model.component("rps_CA::non_compliance"))
        cost = non_compliance * network.policies["rps_CA"]["object"].non_compliance_cost
        if cost > 0:
            penalty_costs["rps_non_compliance"] = cost

    def _print_block(title, data):
        print(f"\n{title}")
        print("-" * 100)
        for k, v in sorted(data.items(), key=lambda x: -x[1]):
            print(f"  {k:<30} ${v:>15,.2f}")
        print(f"  {'TOTAL':<30} ${sum(data.values()):>15,.2f}")

    print("=" * 100)
    print("COST SUMMARY")
    print("=" * 100)

    _print_block("CAPEX", capex_breakdown)
    _print_block("ASSET OPEX", opex_breakdown)
    _print_block("LINE OPEX", line_opex_breakdown)
    _print_block("FIXED", fixed_breakdown)
    _print_block("PENALTIES", penalty_costs)

    reconstructed = (
        sum(capex_breakdown.values())
        + sum(opex_breakdown.values())
        + sum(line_opex_breakdown.values())
        + sum(fixed_breakdown.values())
        + sum(penalty_costs.values())
    )

    other_gap = total_cost - reconstructed

    print("\n" + "=" * 100)
    print(f"{'Reconstructed Total':<30} ${reconstructed:>15,.2f}")
    print(f"{'Model Objective':<30} ${total_cost:>15,.2f}")
    print(f"{'Other Gap':<30} ${other_gap:>15,.2f}")
    print("=" * 100)

    return {
        "capex": capex_breakdown,
        "opex": opex_breakdown,
        "line_opex": line_opex_breakdown,
        "fixed": fixed_breakdown,
        "penalty": penalty_costs,
        "capex_total": sum(capex_breakdown.values()),
        "opex_total": sum(opex_breakdown.values()),
        "line_opex_total": sum(line_opex_breakdown.values()),
        "fixed_total": sum(fixed_breakdown.values()),
        "penalty_total": sum(penalty_costs.values()),
        "reconstructed_total": reconstructed,
        "model_total": total_cost,
        "other_gap": other_gap,
    }

def check_simultaneous_charge_discharge(network, tol=1e-6):
    """
    Check whether any Store asset is charging and discharging at the same time.

    tol is in W.
    """
    model = network.model

    print("\n" + "=" * 100)
    print("SIMULTANEOUS CHARGE / DISCHARGE CHECK")
    print("=" * 100)

    violations = []

    for asset_key, asset_data in network.assets.items():
        if asset_data.get("_class") != "Store":
            continue

        charge_name = f"{asset_key}::charge_power"
        discharge_name = f"{asset_key}::discharge_power"

        if not hasattr(model, charge_name) or not hasattr(model, discharge_name):
            continue

        charge = getattr(model, charge_name)
        discharge = getattr(model, discharge_name)

        local_count = 0
        local_max_charge = 0.0
        local_max_discharge = 0.0

        for t in charge:
            c = pyo.value(charge[t])
            d = pyo.value(discharge[t])

            if c > tol and d > tol:
                violations.append({
                    "asset": asset_key,
                    "t": t,
                    "charge_W": c,
                    "discharge_W": d,
                })
                local_count += 1
                local_max_charge = max(local_max_charge, c)
                local_max_discharge = max(local_max_discharge, d)

        # if local_count > 0:
            # print(
            #     f"{asset_key:<40} simultaneous hours = {local_count:>5}   "
            #     f"max charge = {local_max_charge/1e6:>10.3f} MW   "
            #     f"max discharge = {local_max_discharge/1e6:>10.3f} MW"
            # )

    if len(violations) == 0:
        print("No simultaneous charge/discharge detected.")
    else:
        print(f"\nTotal simultaneous violations: {len(violations)}")

    return violations

def debug_store_dispatch_summary(network):
    import pyomo.environ as pyo

    model = network.model

    print("\n" + "=" * 100)
    print("STORE DISPATCH SUMMARY")
    print("=" * 100)

    results = {}
    found_any = False

    for asset_key, asset_data in network.assets.items():
        asset_obj = asset_data.get("object", None)
        if asset_obj is None or asset_obj.__class__.__name__ != "Store":
            continue

        found_any = True

        out = {
            "total_charge_GWh": 0.0,
            "total_discharge_GWh": 0.0,
            "max_charge_MW": 0.0,
            "max_discharge_MW": 0.0,
        }

        if hasattr(model, f"{asset_key}::charge_power"):
            charge = getattr(model, f"{asset_key}::charge_power")
            charge_vals = [pyo.value(charge[t]) for t in charge]
            out["total_charge_GWh"] = sum(charge_vals) * pyo.value(model.time_step) / 3.6e12
            out["max_charge_MW"] = max(charge_vals) / 1e6 if charge_vals else 0.0

        if hasattr(model, f"{asset_key}::discharge_power"):
            discharge = getattr(model, f"{asset_key}::discharge_power")
            discharge_vals = [pyo.value(discharge[t]) for t in discharge]
            out["total_discharge_GWh"] = sum(discharge_vals) * pyo.value(model.time_step) / 3.6e12
            out["max_discharge_MW"] = max(discharge_vals) / 1e6 if discharge_vals else 0.0

        results[asset_key] = out

        # print(
        #     f"{asset_key:<40}"
        #     f" charge={out['total_charge_GWh']:>10.3f} GWh"
        #     f" discharge={out['total_discharge_GWh']:>10.3f} GWh"
        #     f" max_charge={out['max_charge_MW']:>10.3f} MW"
        #     f" max_discharge={out['max_discharge_MW']:>10.3f} MW"
        # )

    if not found_any:
        print("No Store assets found in network.assets.")

    return results

def debug_energy_balance(network, hours=48, start_hour=None):
    """
    Print hourly energy balance using actual model step labels.
    All signs are from the grid perspective:
      generation  -> positive
      load        -> negative
      charge      -> negative
      discharge   -> positive
      batt_net    -> positive if net discharge, negative if net charge
    """
    import pyomo.environ as pyo

    model = network.model

    print("\n" + "=" * 100)
    print("ENERGY BALANCE DIAGNOSTIC")
    print("=" * 100)

    all_steps = list(model.steps)
    if len(all_steps) == 0:
        print("No model steps found.")
        return []

    steps_to_check = all_steps[:min(hours, len(all_steps))]

    print(f"\nSolar assets found: {len([k for k, v in network.assets.items() if v.get('fuel') == 'solar'])}")
    print(f"Wind assets found : {len([k for k, v in network.assets.items() if v.get('fuel') == 'wind'])}")

    def _safe_value(comp, idx=None):
        try:
            if idx is None:
                return pyo.value(comp)
            if idx in comp:
                return pyo.value(comp[idx])
            return 0.0
        except Exception:
            return 0.0

    results = []

    for i, t in enumerate(steps_to_check):
        solar_gen_mw = 0.0
        wind_gen_mw = 0.0
        gas_gen_mw = 0.0
        other_gen_mw = 0.0

        batt_net_mw = 0.0
        batt_charge_mw = 0.0     # negative from grid perspective
        batt_discharge_mw = 0.0  # positive from grid perspective

        load_mw = 0.0            # negative from grid perspective
        curtail_mw = 0.0
        shortfall_mw = 0.0

        # --------------------------------------------------
        # Assets
        # --------------------------------------------------
        for asset_key, asset_data in network.assets.items():
            asset_obj = asset_data.get("object", None)
            fuel = asset_data.get("fuel", "")
            asset_type = asset_data.get("type", "")

            # Producer-style assets
            if hasattr(model, f"{asset_key}::production"):
                prod = getattr(model, f"{asset_key}::production")
                mw = _safe_value(prod, t) / 1e6

                fuel_l = fuel.lower()
                if fuel == "solar":
                    solar_gen_mw += mw
                elif fuel == "wind":
                    wind_gen_mw += mw
                elif "natural gas" in fuel_l:
                    gas_gen_mw += mw
                else:
                    other_gen_mw += mw

            # Load-style assets
            elif hasattr(model, f"{asset_key}::profile"):
                profile = getattr(model, f"{asset_key}::profile")

                shift_val = 0.0
                if hasattr(model, f"{asset_key}::shift"):
                    shift = getattr(model, f"{asset_key}::shift")
                    shift_val = _safe_value(shift, t)

                capex_val = 0.0
                if hasattr(model, f"{asset_key}::capex"):
                    capex = getattr(model, f"{asset_key}::capex")
                    capex_val = _safe_value(capex)

                installed_capacity = getattr(asset_obj, "installed_capacity", 0.0)
                profile_val = _safe_value(profile, t)
                power_w = profile_val * (installed_capacity + capex_val) + shift_val

                # Renewable negative loads -> convert to positive generation
                if fuel in ["solar", "wind"] or asset_type in ["solar", "wind"]:
                    gen_mw = power_w / 1e6
                    if fuel == "solar" or asset_type == "solar":
                        solar_gen_mw += gen_mw
                    elif fuel == "wind" or asset_type == "wind":
                        wind_gen_mw += gen_mw

                # Actual load -> negative from grid perspective
                elif asset_type == "load" and not asset_data.get("renewable", False):
                    load_mw += -power_w / 1e6

            # Store-style assets
            if asset_obj is not None and asset_obj.__class__.__name__ == "Store":
                if hasattr(model, f"{asset_key}::power"):
                    power = getattr(model, f"{asset_key}::power")
                    batt_net_mw += _safe_value(power, t) / 1e6

                if hasattr(model, f"{asset_key}::charge_power"):
                    charge = getattr(model, f"{asset_key}::charge_power")
                    batt_charge_mw += -_safe_value(charge, t) / 1e6

                if hasattr(model, f"{asset_key}::discharge_power"):
                    discharge = getattr(model, f"{asset_key}::discharge_power")
                    batt_discharge_mw += _safe_value(discharge, t) / 1e6

        # --------------------------------------------------
        # Node-level slack
        # --------------------------------------------------
        for source in network.graph.nodes():
            if hasattr(model, f"{source}::wastage"):
                wastage = getattr(model, f"{source}::wastage")
                curtail_mw += _safe_value(wastage, t) / 1e6

            if hasattr(model, f"{source}::shortfall"):
                shortfall = getattr(model, f"{source}::shortfall")
                shortfall_mw += _safe_value(shortfall, t) / 1e6

        display_hour = (start_hour + i) % 24 if start_hour is not None else i % 24

        # print(
        #     f"t={int(t):4d}  h={display_hour:02d}  "
        #     f"Solar={solar_gen_mw:9,.1f}  "
        #     f"Wind={wind_gen_mw:9,.1f}  "
        #     f"Gas={gas_gen_mw:9,.1f}  "
        #     f"Other={other_gen_mw:9,.1f}  "
        #     f"BattNet={batt_net_mw:8,.1f}  "
        #     f"Charge={batt_charge_mw:8,.1f}  "
        #     f"Disch={batt_discharge_mw:8,.1f}  "
        #     f"Load={load_mw:9,.1f}  "
        #     f"Curt={curtail_mw:8,.1f}  "
        #     f"Short={shortfall_mw:8,.1f}"
        # )

        results.append({
            "step": int(t),
            "hour_of_day": display_hour,
            "solar_mw": solar_gen_mw,
            "wind_mw": wind_gen_mw,
            "gas_mw": gas_gen_mw,
            "other_mw": other_gen_mw,
            "battery_net_mw": batt_net_mw,
            "battery_charge_mw": batt_charge_mw,
            "battery_discharge_mw": batt_discharge_mw,
            "load_mw": load_mw,
            "curtailment_mw": curtail_mw,
            "shortfall_mw": shortfall_mw,
        })

    return results

def debug_ev_assets(network):
    """
    Debug EV-related assets including:
    - fixed EV load
    - V1G load
    - V2G load
    - V2G storage

    Returns
    -------
    ev_summary : dict
    """
    import pyomo.environ as pyo

    model = network.model

    print("\n" + "=" * 100)
    print("EV ASSET DIAGNOSTIC")
    print("=" * 100)

    ev_summary = {}

    def _safe_value(x):
        try:
            return pyo.value(x)
        except Exception:
            return 0.0

    def _extract_region(asset_key):
        if asset_key.startswith("ev_v2g_store_"):
            parts = asset_key.split("_")
            if len(parts) > 4:
                return "_".join(parts[3:-1])
            return asset_key
        return asset_key.split("_")[-1]

    for asset_key, asset_data in network.assets.items():
        if not any(x in asset_key for x in [
            "ev_load_fixed",
            "ev_load_v1g",
            "ev_load_v2g",
            "ev_v2g_store"
        ]):
            continue

        asset = asset_data.get("object", None)
        if asset is None:
            continue

        region = _extract_region(asset_key)

        if region not in ev_summary:
            ev_summary[region] = {
                "fixed": None,
                "v1g": None,
                "v2g_load": None,
                "v2g_store": None,
            }

        # -----------------------------
        # Fixed EV load
        # -----------------------------
        if "ev_load_fixed" in asset_key:
            installed_mw = abs(getattr(asset, "installed_capacity", 0.0)) / 1e6
            shift_capacity_mw = getattr(asset, "shift_capacity", 0.0) / 1e6

            ev_summary[region]["fixed"] = {
                "capacity_MW": installed_mw,
                "shift_capacity_MW": shift_capacity_mw,
                "shiftable": getattr(asset, "shiftable", False),
            }

            print(f"\n{asset_key}:")
            print(f"  Installed Capacity: {installed_mw:,.2f} MW")
            print(f"  Shift Capacity: {shift_capacity_mw:,.2f} MW")
            print(f"  Shiftable: {getattr(asset, 'shiftable', False)}")

        # -----------------------------
        # V1G EV load
        # -----------------------------
        elif "ev_load_v1g" in asset_key:
            installed_mw = abs(getattr(asset, "installed_capacity", 0.0)) / 1e6
            shift_capacity_mw = getattr(asset, "shift_capacity", 0.0) / 1e6
            shift_window = getattr(asset, "shift_window", None)

            max_shift_up = 0.0
            max_shift_down = 0.0
            total_shift_mwh = 0.0

            if hasattr(model, f"{asset_key}::shift"):
                shift = getattr(model, f"{asset_key}::shift")
                shift_vals_mw = [_safe_value(shift[t]) / 1e6 for t in model.steps]
                if len(shift_vals_mw) > 0:
                    max_shift_up = max(shift_vals_mw)
                    max_shift_down = min(shift_vals_mw)
                    total_shift_mwh = sum(abs(v) for v in shift_vals_mw)

            ev_summary[region]["v1g"] = {
                "capacity_MW": installed_mw,
                "shift_capacity_MW": shift_capacity_mw,
                "max_shift_up_MW": max_shift_up,
                "max_shift_down_MW": max_shift_down,
                "total_shift_MWh": total_shift_mwh,
                "shift_window": shift_window,
            }

            print(f"\n{asset_key}:")
            print(f"  Installed Capacity: {installed_mw:,.2f} MW")
            print(f"  Shift Capacity: {shift_capacity_mw:,.2f} MW")
            print(f"  Shift Window: {shift_window}")
            print(f"  Max Shift Up: {max_shift_up:,.2f} MW")
            print(f"  Max Shift Down: {max_shift_down:,.2f} MW")
            print(f"  Total Shifting: {total_shift_mwh:,.2f} MWh")

        # -----------------------------
        # V2G EV load
        # -----------------------------
        elif "ev_load_v2g" in asset_key:
            installed_mw = abs(getattr(asset, "installed_capacity", 0.0)) / 1e6
            shift_capacity_mw = getattr(asset, "shift_capacity", 0.0) / 1e6

            ev_summary[region]["v2g_load"] = {
                "capacity_MW": installed_mw,
                "shift_capacity_MW": shift_capacity_mw,
                "shiftable": getattr(asset, "shiftable", False),
            }

            print(f"\n{asset_key}:")
            print(f"  Installed Capacity: {installed_mw:,.2f} MW")
            print(f"  Shift Capacity: {shift_capacity_mw:,.2f} MW")
            print(f"  Shiftable: {getattr(asset, 'shiftable', False)}")

        # -----------------------------
        # V2G Store
        # -----------------------------
        elif "ev_v2g_store" in asset_key:
            installed_capacity_j = getattr(asset, "installed_capacity", 0.0)
            installed_power_capacity_w = getattr(asset, "installed_power_capacity", None)
            production_rate = getattr(asset, "production_rate", 0.0)
            charge_eff = getattr(asset, "charge_efficiency", getattr(asset, "efficiency", None))
            discharge_eff = getattr(asset, "discharge_efficiency", getattr(asset, "efficiency", None))

            if installed_power_capacity_w is not None:
                installed_mw = installed_power_capacity_w / 1e6
            else:
                installed_mw = installed_capacity_j * production_rate / 1e6 if production_rate else 0.0

            installed_mwh = installed_capacity_j / 3.6e9

            max_discharge_mw = 0.0
            max_charge_mw = 0.0
            total_discharge_mwh = 0.0
            total_charge_mwh = 0.0
            simultaneous_hours = 0

            if hasattr(model, f"{asset_key}::discharge_power"):
                discharge = getattr(model, f"{asset_key}::discharge_power")
                d_vals = [_safe_value(discharge[t]) / 1e6 for t in discharge]
                if len(d_vals) > 0:
                    max_discharge_mw = max(d_vals)
                    total_discharge_mwh = sum(d_vals)

            if hasattr(model, f"{asset_key}::charge_power"):
                charge = getattr(model, f"{asset_key}::charge_power")
                c_vals = [_safe_value(charge[t]) / 1e6 for t in charge]
                if len(c_vals) > 0:
                    max_charge_mw = max(c_vals)
                    total_charge_mwh = sum(c_vals)

            if hasattr(model, f"{asset_key}::charge_power") and hasattr(model, f"{asset_key}::discharge_power"):
                charge = getattr(model, f"{asset_key}::charge_power")
                discharge = getattr(model, f"{asset_key}::discharge_power")
                common_idx = set(charge.keys()) & set(discharge.keys())
                simultaneous_hours = sum(
                    1 for t in common_idx
                    if _safe_value(charge[t]) > 1e-6 and _safe_value(discharge[t]) > 1e-6
                )

            min_soc_mwh = 0.0
            max_soc_mwh = 0.0
            if hasattr(model, f"{asset_key}::level"):
                level = getattr(model, f"{asset_key}::level")
                level_vals = [_safe_value(level[t]) / 3.6e9 for t in level]
                if len(level_vals) > 0:
                    min_soc_mwh = min(level_vals)
                    max_soc_mwh = max(level_vals)

            ev_summary[region]["v2g_store"] = {
                "capacity_MW": installed_mw,
                "capacity_MWh": installed_mwh,
                "max_discharge_MW": max_discharge_mw,
                "max_charge_MW": max_charge_mw,
                "total_discharge_MWh": total_discharge_mwh,
                "total_charge_MWh": total_charge_mwh,
                "min_soc_MWh": min_soc_mwh,
                "max_soc_MWh": max_soc_mwh,
                "charge_efficiency": charge_eff,
                "discharge_efficiency": discharge_eff,
                "simultaneous_hours": simultaneous_hours,
            }

            # print(f"\n{asset_key}:")
            # print(f"  Installed Capacity: {installed_mw:,.2f} MW / {installed_mwh:,.2f} MWh")
            # print(f"  Charge efficiency: {charge_eff}")
            # print(f"  Discharge efficiency: {discharge_eff}")
            # print(f"  Max Discharge: {max_discharge_mw:,.2f} MW")
            # print(f"  Max Charge: {max_charge_mw:,.2f} MW")
            # print(f"  Total Discharge: {total_discharge_mwh:,.2f} MWh")
            # print(f"  Total Charge: {total_charge_mwh:,.2f} MWh")
            # print(f"  SOC Range: {min_soc_mwh:,.2f} - {max_soc_mwh:,.2f} MWh")
            # print(f"  Simultaneous charge/discharge hours: {simultaneous_hours}")

    # -----------------------------
    # Regional summary
    # -----------------------------
    print("\n" + "=" * 100)
    print("REGIONAL EV SUMMARY")
    print("=" * 100)

    # for region in sorted(ev_summary.keys()):
    #     data = ev_summary[region]
    #     print(f"\n{region}:")

    #     if data["fixed"]:
    #         # print(f"  Fixed EV Load: {data['fixed']['capacity_MW']:,.2f} MW")

    #     if data["v1g"]:
    #         v1g = data["v1g"]
    #         # print(f"  V1G Load: {v1g['capacity_MW']:,.2f} MW")
    #         # print(f"    Shift Capacity: {v1g['shift_capacity_MW']:,.2f} MW")
    #         # print(f"    Actual Shifting: {v1g['total_shift_MWh']:,.2f} MWh")

    #     if data["v2g_load"]:
    #         # print(f"  V2G Load: {data['v2g_load']['capacity_MW']:,.2f} MW")

    #     if data["v2g_store"]:
    #         v2g = data["v2g_store"]
    #         # print(f"  V2G Storage: {v2g['capacity_MW']:,.2f} MW / {v2g['capacity_MWh']:,.2f} MWh")
    #         # print(f"    Max Discharge: {v2g['max_discharge_MW']:,.2f} MW")
    #         # print(f"    Max Charge: {v2g['max_charge_MW']:,.2f} MW")
    #         # print(f"    Total Discharge: {v2g['total_discharge_MWh']:,.2f} MWh")
    #         # print(f"    Total Charge: {v2g['total_charge_MWh']:,.2f} MWh")
    #         # print(f"    Simultaneous Hours: {v2g['simultaneous_hours']}")

    # -----------------------------
    # Statewide totals
    # -----------------------------
    print("\n" + "=" * 100)
    print("STATEWIDE EV TOTALS")
    print("=" * 100)

    total_fixed = sum(d["fixed"]["capacity_MW"] for d in ev_summary.values() if d["fixed"])
    total_v1g = sum(d["v1g"]["capacity_MW"] for d in ev_summary.values() if d["v1g"])
    total_v1g_shift = sum(d["v1g"]["total_shift_MWh"] for d in ev_summary.values() if d["v1g"])
    total_v2g_load = sum(d["v2g_load"]["capacity_MW"] for d in ev_summary.values() if d["v2g_load"])
    total_v2g_store_mw = sum(d["v2g_store"]["capacity_MW"] for d in ev_summary.values() if d["v2g_store"])
    total_v2g_store_mwh = sum(d["v2g_store"]["capacity_MWh"] for d in ev_summary.values() if d["v2g_store"])
    total_v2g_discharge = sum(d["v2g_store"]["total_discharge_MWh"] for d in ev_summary.values() if d["v2g_store"])
    total_v2g_charge = sum(d["v2g_store"]["total_charge_MWh"] for d in ev_summary.values() if d["v2g_store"])

    # print(f"Fixed EV Load: {total_fixed:,.2f} MW")
    # print(f"V1G Load: {total_v1g:,.2f} MW")
    # print(f"  Total V1G shifting: {total_v1g_shift:,.2f} MWh")
    # print(f"V2G Load: {total_v2g_load:,.2f} MW")
    # print(f"V2G Storage: {total_v2g_store_mw:,.2f} MW / {total_v2g_store_mwh:,.2f} MWh")
    # print(f"  Total V2G discharge: {total_v2g_discharge:,.2f} MWh")
    # print(f"  Total V2G charge: {total_v2g_charge:,.2f} MWh")
    # print(f"Total EV Load: {total_fixed + total_v1g + total_v2g_load:,.2f} MW")

    return ev_summary

def debug_generation_table(network, battery_hours=4):
    """
    Build and print a generation table in MWh/GWh by fuel.

    Convention:
    - normal generators: positive generation
    - solar/wind modeled as negative loads: flip sign
    - storage: count only discharge as generation
    - actual loads are excluded from generation table
    """
    import pyomo.environ as pyo
    import pandas as pd

    model = network.model
    dt = pyo.value(model.time_step)

    rows = []

    def _safe_value(comp, idx=None):
        try:
            if idx is None:
                return pyo.value(comp)
            if idx in comp:
                return pyo.value(comp[idx])
            return 0.0
        except Exception:
            return 0.0

    for asset_key, asset_data in network.assets.items():
        asset_obj = asset_data.get("object", None)
        fuel = asset_data.get("fuel", "unknown")
        asset_type = asset_data.get("type", "unknown")
        asset_class = asset_obj.__class__.__name__ if asset_obj is not None else "unknown"

        gen_mwh = 0.0

        # --------------------------------------------------
        # Producer-style assets
        # --------------------------------------------------
        if hasattr(model, f"{asset_key}::production"):
            prod = getattr(model, f"{asset_key}::production")
            gen_mwh = sum(
                _safe_value(prod, t) * dt / 3.6e9
                for t in model.steps
            )

        # --------------------------------------------------
        # Load-style renewables
        # --------------------------------------------------
        elif hasattr(model, f"{asset_key}::profile"):
            profile = getattr(model, f"{asset_key}::profile")

            shift_name = f"{asset_key}::shift"
            capex_name = f"{asset_key}::capex"

            shift = getattr(model, shift_name) if hasattr(model, shift_name) else None
            capex = getattr(model, capex_name) if hasattr(model, capex_name) else None

            installed_capacity = getattr(asset_obj, "installed_capacity", 0.0)
            capex_val = _safe_value(capex) if capex is not None else 0.0
            total_capacity = installed_capacity + capex_val

            # Renewable negative loads only
            if fuel in ["solar", "wind"] or asset_type in ["solar", "wind"]:
                gen_mwh = 0.0
                for t in model.steps:
                    shift_val = _safe_value(shift, t) if shift is not None else 0.0
                    power_w = _safe_value(profile, t) * total_capacity + shift_val
                    gen_mwh += (-power_w) * dt / 3.6e9   # flip sign to grid perspective

            else:
                # actual loads are not generation
                continue

        # --------------------------------------------------
        # Store assets: count discharge only
        # --------------------------------------------------
        if asset_obj is not None and asset_obj.__class__.__name__ == "Store":
            gen_mwh = 0.0
            if hasattr(model, f"{asset_key}::discharge_power"):
                discharge = getattr(model, f"{asset_key}::discharge_power")
                gen_mwh = sum(
                    _safe_value(discharge, t) * dt / 3.6e9
                    for t in discharge
                )
            elif hasattr(model, f"{asset_key}::power"):
                power = getattr(model, f"{asset_key}::power")
                gen_mwh = sum(
                    max(0.0, _safe_value(power, t)) * dt / 3.6e9
                    for t in power
                )

        # Capacity values for display
        installed_capacity_raw = getattr(asset_obj, "installed_capacity", 0.0) if asset_obj is not None else 0.0

        capex_val = 0.0
        if hasattr(model, f"{asset_key}::capex"):
            capex_val = _safe_value(getattr(model, f"{asset_key}::capex"))

        is_store = asset_class == "Store"

        if is_store:
            installed_mw = installed_capacity_raw / (battery_hours * 3600) / 1e6
            capex_mw = capex_val / (battery_hours * 3600) / 1e6
        else:
            installed_mw = installed_capacity_raw / 1e6
            capex_mw = capex_val / 1e6

        rows.append({
            "asset": asset_key,
            "fuel": fuel,
            "generation_mwh": gen_mwh,
            "generation_gwh": gen_mwh / 1000.0,
            "installed_mw": installed_mw,
            "capex_mw": capex_mw,
            "total_mw": installed_mw + capex_mw,
        })

    df = pd.DataFrame(rows)

    if df.empty:
        print("\n" + "=" * 120)
        print("GENERATION TABLE")
        print("=" * 120)
        print("No generation assets found.")
        return {
            "detail_table": df,
            "summary_table": df,
            "total_wastage_mwh": 0.0,
            "total_shortfall_mwh": 0.0,
        }

    summary = (
        df.groupby("fuel", dropna=False)
        .agg(
            generation_MWh=("generation_mwh", "sum"),
            generation_GWh=("generation_gwh", "sum"),
            installed_MW=("installed_mw", "sum"),
            capex_MW=("capex_mw", "sum"),
            total_MW=("total_mw", "sum"),
            n_assets=("asset", "count"),
        )
        .reset_index()
        .sort_values("generation_MWh", ascending=False)
    )

    total_wastage_mwh = 0.0
    total_shortfall_mwh = 0.0

    for source in network.graph.nodes():
        if hasattr(model, f"{source}::wastage"):
            wastage = getattr(model, f"{source}::wastage")
            total_wastage_mwh += sum(
                _safe_value(wastage, t) * dt / 3.6e9
                for t in model.steps
            )

        if hasattr(model, f"{source}::shortfall"):
            shortfall = getattr(model, f"{source}::shortfall")
            total_shortfall_mwh += sum(
                _safe_value(shortfall, t) * dt / 3.6e9
                for t in model.steps
            )

    print("\n" + "=" * 140)
    print("GENERATION TABLE")
    print("=" * 140)
    print(
        f"{'Fuel':<30}"
        f"{'Generation (GWh)':>20}"
        f"{'Installed MW':>15}"
        f"{'Capex MW':>15}"
        f"{'Total MW':>15}"
        f"{'#Assets':>10}"
    )
    print("-" * 140)

    for _, r in summary.iterrows():
        print(
            f"{str(r['fuel']):<30}"
            f"{r['generation_GWh']:>20,.2f}"
            f"{r['installed_MW']:>15,.2f}"
            f"{r['capex_MW']:>15,.2f}"
            f"{r['total_MW']:>15,.2f}"
            f"{int(r['n_assets']):>10}"
        )

    print("-" * 140)
    print(f"{'Total Wastage (MWh)':<30}{total_wastage_mwh:>20,.2f}")
    print(f"{'Total Shortfall (MWh)':<30}{total_shortfall_mwh:>20,.2f}")

    return {
        "detail_table": df,
        "summary_table": summary,
        "total_wastage_mwh": total_wastage_mwh,
        "total_shortfall_mwh": total_shortfall_mwh,
    }

def compute_slack(solution, time_step):

    total_waste = 0.0
    total_short = 0.0

    for region in solution["regions"].values():

        waste = region.get("wastage", {})
        short = region.get("shortfall", {})

        total_waste += sum(v * time_step / 3.6e9 for v in waste.values())
        total_short += sum(v * time_step / 3.6e9 for v in short.values())

    return total_waste, total_short

def debug_shortfall_by_region(network, tol_mw=1e-3):
    import pyomo.environ as pyo

    model = network.model
    print("\n" + "=" * 100)
    print("SHORTFALL BY REGION AND HOUR")
    print("=" * 100)

    hits = []

    for region in network.graph.nodes():
        var_name = f"{region}::shortfall"
        if not hasattr(model, var_name):
            continue

        shortfall = getattr(model, var_name)

        for t in model.steps:
            mw = pyo.value(shortfall[t]) / 1e6
            if mw > tol_mw:
                hits.append((region, int(t), mw))
                # print(f"Region={region:<15} t={int(t):>5} shortfall={mw:>10.3f} MW")

    if not hits:
        print("No shortfall found.")

    return hits

def debug_wastage_by_region(network, tol_mw=1e-3):
    import pyomo.environ as pyo

    model = network.model
    print("\n" + "=" * 100)
    print("WASTAGE BY REGION AND HOUR")
    print("=" * 100)

    hits = []

    for region in network.graph.nodes():
        var_name = f"{region}::wastage"
        if not hasattr(model, var_name):
            continue

        wastage = getattr(model, var_name)

        for t in model.steps:
            mw = pyo.value(wastage[t]) / 1e6
            if mw > tol_mw:
                hits.append((region, int(t), mw))
                # print(f"Region={region:<15} t={int(t):>5} wastage={mw:>10.3f} MW")

    if not hits:
        print("No wastage found.")

    return hits

def debug_import_congestion_at_shortfall(network, tol_shortfall_mw=1e-3, binding_frac=0.98):
    import pyomo.environ as pyo

    model = network.model

    print("\n" + "=" * 100)
    print("IMPORT CONGESTION AT SHORTFALL HOURS")
    print("=" * 100)

    all_results = []

    for region in network.graph.nodes():
        shortfall_name = f"{region}::shortfall"
        if not hasattr(model, shortfall_name):
            continue

        shortfall = getattr(model, shortfall_name)

        for t in model.steps:
            short_mw = pyo.value(shortfall[t]) / 1e6
            if short_mw <= tol_shortfall_mw:
                continue

            # print(f"\nSHORTFALL at region={region}, t={int(t)}, shortfall={short_mw:.3f} MW")

            hour_result = {
                "region": region,
                "t": int(t),
                "shortfall_mw": short_mw,
                "incoming_lines": []
            }

            found_incoming = False

            for source, node in network.graph._node.items():
                for target, export_edge in node["object"].exports.items():
                    if target != region:
                        continue

                    for line_name, line_obj in export_edge["object"].lines.items():
                        obj = line_obj["object"]
                        var_name = f"{obj.handle}::transmission"

                        if not hasattr(model, var_name):
                            continue

                        transmission = getattr(model, var_name)
                        flow_mw = pyo.value(transmission[t]) / 1e6
                        cap_mw = obj.installed_capacity / 1e6
                        loading = flow_mw / cap_mw if cap_mw > 0 else 0.0
                        binding = loading >= binding_frac

                        found_incoming = True

                        # print(
                        #     f"  {source:>12} -> {region:<12} "
                        #     f"flow={flow_mw:>10.2f} MW   "
                        #     f"cap={cap_mw:>10.2f} MW   "
                        #     f"loading={loading:>7.2%}   "
                        #     f"{'BINDING' if binding else ''}"
                        # )

                        hour_result["incoming_lines"].append({
                            "source": source,
                            "target": region,
                            "line_name": line_name,
                            "flow_mw": flow_mw,
                            "cap_mw": cap_mw,
                            "loading": loading,
                            "binding": binding,
                        })

            if not found_incoming:
                print("  No incoming transmission lines found for this region.")

            all_results.append(hour_result)

    if not all_results:
        print("No shortfall hours found.")

    return all_results

def debug_transmission_summary(network):
    import pyomo.environ as pyo

    model = network.model

    print("\n" + "=" * 100)
    print("TRANSMISSION LINE MAPPING")
    print("=" * 100)

    for source, node in network.graph._node.items():
        for target, export_edge in node["object"].exports.items():
            for line_name, line_obj in export_edge["object"].lines.items():
                obj = line_obj["object"]
                # print(f"{line_name}: {source} -> {target}")
                # print(f"  Capacity: {obj.installed_capacity / 1e6:.2f} MW")
                # print(f"  Efficiency: {obj.efficiency:.4f}")

    print("\n" + "=" * 100)
    print("TRANSMISSION FLOW CHECK")
    print("=" * 100)

    total_transmission_MWh = 0.0

    for source, node in network.graph._node.items():
        for export_name, export_edge in node["object"].exports.items():
            for line_name, line_obj in export_edge["object"].lines.items():
                var_name = f"{line_obj['object'].handle}::transmission"
                if not hasattr(model, var_name):
                    continue

                transmission_var = getattr(model, var_name)

                total_energy_J = sum(pyo.value(transmission_var[t]) for t in model.steps)
                total_energy_MWh = total_energy_J / 3.6e9
                total_transmission_MWh += total_energy_MWh

                if abs(total_energy_MWh) > 0.001:
                    print(f"{source} -> {export_name} via {line_name}: {total_energy_MWh:,.2f} MWh")

    print(f"\nTotal transmission across all lines: {total_transmission_MWh:,.2f} MWh")

    print("\n" + "=" * 100)
    print("TRANSMISSION LINE PARAMETERS")
    print("=" * 100)

    for source, node in network.graph._node.items():
        for export_name, export_edge in node["object"].exports.items():
            for line_name, line_obj in export_edge["object"].lines.items():
                obj = line_obj["object"]
                # print(f"{line_name}:")
                # print(f"  Operating cost: {obj.operating_cost:.2e} $/J")
                # print(f"  Operating cost: {obj.operating_cost * 3.6e9:.2f} $/MWh")
                # print(f"  Efficiency: {obj.efficiency}")
                # print(f"  Installed capacity: {obj.installed_capacity / 1e6:.2f} MW")
                # print(f"  Capex limit: {obj.capex_limit / 1e6:.2f} MW")

def save_capacity_and_capex_tables(
    debug_results,
    scenario_id,
    cfg,
    scenario_output_dir,
    scenario_tag,
    adoption,
    charging,
    month,
    day_duration,
):
    """
    Save the same capacity and capex tables produced by debug_capacity_and_cost_summary().

    Outputs:
        *_capex_detail.csv
        *_capex_summary.csv
    """

    import os

    capacity_debug = debug_results["capacity_and_cost"]

    capex_detail_df = capacity_debug["detail_table"].copy()
    capex_summary_df = capacity_debug["summary_table"].copy()

    # Add scenario metadata
    for df_out in [capex_detail_df, capex_summary_df]:
        df_out["scenario_id"] = scenario_id
        df_out["scenario_tag"] = cfg.get("scenario_tag", "")
        df_out["adoption"] = adoption
        df_out["charging"] = charging
        df_out["month"] = month
        df_out["day_duration"] = day_duration
        df_out["batt_capex_cost"] = cfg.get("batt_capex_cost", None)
        df_out["rps_ratio"] = cfg.get("rps_ratio", None)
        df_out["v1g_share"] = cfg.get("v1g_share", None)
        df_out["v2g_share"] = cfg.get("v2g_share", None)

    capex_detail_path = os.path.join(
        scenario_output_dir,
        f"s{scenario_id:02d}_{scenario_tag}_capex_detail.csv"
    )

    capex_summary_path = os.path.join(
        scenario_output_dir,
        f"s{scenario_id:02d}_{scenario_tag}_capex_summary.csv"
    )

    capex_detail_df.to_csv(capex_detail_path, index=False)
    capex_summary_df.to_csv(capex_summary_path, index=False)

    print(f"Capex detail CSV: {capex_detail_path}")
    print(f"Capex summary CSV: {capex_summary_path}")

    return {
        "capex_detail_csv": capex_detail_path,
        "capex_summary_csv": capex_summary_path,
    }

def run_full_debug_suite(
    network,
    scenario_id,
    cfg,
    battery_hours=4,
    run_generation_table=True,
    check_simultaneous=True,
    run_energy_balance=True,
    run_ev_debug=True,
    run_transmission_debug=True,
    run_shortfall_debug=True,
):
    results = {}

    results["objective_gap"] = debug_objective_gap(network)

    results["capacity_and_cost"] = debug_capacity_and_cost_summary(
        network=network,
        scenario_id=scenario_id,
        cfg=cfg,
        battery_hours=battery_hours,
    )

    results["cost_breakdown"] = detailed_cost_breakdown(
        network=network,
        scenario_id=scenario_id,
        cfg=cfg,
    )

    if run_transmission_debug:
        results["transmission_summary"] = debug_transmission_summary(network)

    if run_shortfall_debug:
        results["shortfall_by_region"] = debug_shortfall_by_region(network)
        results["wastage_by_region"] = debug_wastage_by_region(network)
        results["import_congestion_at_shortfall"] = debug_import_congestion_at_shortfall(network)

    if run_generation_table:
        results["generation_table"] = debug_generation_table(
            network=network,
            battery_hours=battery_hours,
        )

    if check_simultaneous:
        results["simultaneous_charge_discharge"] = check_simultaneous_charge_discharge(network)

    if run_energy_balance:
        results["energy_balance"] = debug_energy_balance(network, hours=48)

    if run_ev_debug:
        results["ev_assets"] = debug_ev_assets(network)

    return results

def save_ev_summary(
    debug_results,
    scenario_id,
    cfg,
    scenario_output_dir,
    scenario_tag,
):
    import os
    import pandas as pd

    ev_data = debug_results["ev_assets"]

    # -------- STATE TOTALS --------
    state_totals = ev_data.get("state_totals", {})

    ev_summary_df = pd.DataFrame([{
        "scenario_id": scenario_id,
        "scenario_tag": scenario_tag,

        "fixed_ev_mw": state_totals.get("fixed_ev_load_mw", 0.0),
        "v1g_ev_mw": state_totals.get("v1g_load_mw", 0.0),
        "v2g_ev_mw": state_totals.get("v2g_load_mw", 0.0),

        "v1g_shift_mwh": state_totals.get("total_v1g_shifting_mwh", 0.0),
        "v2g_discharge_mwh": state_totals.get("total_v2g_discharge_mwh", 0.0),
        "v2g_charge_mwh": state_totals.get("total_v2g_charge_mwh", 0.0),
    }])

    ev_summary_path = os.path.join(
        scenario_output_dir,
        f"s{scenario_id:02d}_{scenario_tag}_ev_summary.csv"
    )

    ev_summary_df.to_csv(ev_summary_path, index=False)
    print(f"EV summary CSV: {ev_summary_path}")

    return {
        "ev_summary_csv": ev_summary_path
    }