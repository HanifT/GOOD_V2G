import os
import pandas as pd


def save_scenario_results(
    scenario_id,
    cfg,
    solution,
    dataframe,
    objective_value,
    network,
    RESULTS_DIR,
    year,
    adoption,
    charging,
    month,
    day_duration,
):
    """
    Save all outputs for a scenario in a structured folder.

    Returns a dictionary with paths and metadata.
    """

    # -----------------------------
    # Build scenario tag
    # -----------------------------
    bcapex_label_clean = (
        cfg["batt_capex_label"]
        .replace("$", "")
        .replace("/", "")
        .replace("kWh", "")
    )

    scenario_tag = (
        f"s{scenario_id:02d}_"
        f"{cfg['group'].replace(' ', '_')}_"
        f"rps{int(cfg['rps_ratio'] * 100)}_"
        f"v1g{int(cfg['v1g_share'] * 100)}_"
        f"v2g{int(cfg['v2g_share'] * 100)}_"
        f"bcapex{bcapex_label_clean}_"
        f"{adoption}_"
        f"m{month}_"
        f"d{day_duration}"
    )

    # -----------------------------
    # Create output directory
    # -----------------------------
    out_dir = os.path.join(RESULTS_DIR, scenario_tag)
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # File paths
    # -----------------------------
    solution_json_path = os.path.join(out_dir, f"{scenario_tag}_solution.json")
    solution_csv_path = os.path.join(out_dir, f"{scenario_tag}_solution.csv")
    summary_csv_path = os.path.join(out_dir, f"{scenario_tag}_summary.csv")
    total_cost_csv_path = os.path.join(out_dir, f"{scenario_tag}_total_cost.csv")

    # -----------------------------
    # Save solution graph
    # -----------------------------
    import good.graph  # keep local import to avoid circular deps

    good.graph.graph_to_json(solution, solution_json_path)

    # -----------------------------
    # Save dataframe
    # -----------------------------
    dataframe.to_csv(solution_csv_path, index=False)

    # -----------------------------
    # Save summary
    # -----------------------------
    summary = pd.DataFrame([{
        "scenario_id": scenario_id,
        "scenario_tag": scenario_tag,
        "group": cfg["group"],
        "rps_ratio": cfg["rps_ratio"],
        "v1g_share": cfg["v1g_share"],
        "v2g_share": cfg["v2g_share"],
        "batt_capex_cost": cfg["batt_capex_cost"],
        "batt_capex_label": cfg["batt_capex_label"],
        "objective_value": objective_value,
        "year": year,
        "adoption": adoption,
        "charging": charging,
        "month": month,
        "day_duration": day_duration,
    }])

    summary.to_csv(summary_csv_path, index=False)

    # -----------------------------
    # Save total cost
    # -----------------------------
    total_cost = pd.DataFrame([{
        "scenario_id": scenario_id,
        "scenario_tag": scenario_tag,
        "objective_value": objective_value,
    }])

    total_cost.to_csv(total_cost_csv_path, index=False)

    # -----------------------------
    # Print summary
    # -----------------------------
    print(f"Finished scenario {scenario_id}: {scenario_tag}")
    print(f"Solution JSON: {solution_json_path}")
    print(f"Solution CSV: {solution_csv_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Total cost CSV: {total_cost_csv_path}")
    print(f"Objective = {objective_value:,.2f}")

    # -----------------------------
    # Return results
    # -----------------------------
    return {
        "scenario_id": scenario_id,
        "tag": scenario_tag,
        "objective_value": objective_value,
        "solution": solution,
        "dataframe": dataframe,
        "network": network,
        "out_dir": out_dir,
        "solution_json_path": solution_json_path,
        "solution_csv_path": solution_csv_path,
        "summary_csv_path": summary_csv_path,
        "total_cost_csv_path": total_cost_csv_path,
    }