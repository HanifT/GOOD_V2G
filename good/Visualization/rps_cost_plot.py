from matplotlib.colors import to_rgb, to_hex
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import geopandas as gpd
import matplotlib as mpl
import os
import seaborn as sns
import re
import json
import glob
import numpy as np
from matplotlib.patches import Patch
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================================================
# Region aliases
# =========================================================

# =========================================================
# Supported regions
# =========================================================
# =========================================================
# Region definitions
# =========================================================

REGION_ALIASES = {
    "CAISO": "CAISO",
    "ERCOT": "ERCOT",
    "FRCC": "FRCC",
    "NYISO": "NYISO",
    "PJM": "PJM",
    "MAPP": "MAPP",
    "MISO": "MISO",
    "ISO-NE": "ISO-NE",
    "SERC-E": "SERC-E",
    "SERC-N": "SERC-N",
    "SERC-SE": "SERC-SE",
    "SPP": "SPP",
    "NWPP": "NWPP",
    "RMRG": "RMRG",
    "SRSG": "SRSG",
}


REGION_DISPLAY_NAMES = {
    "CAISO": "CAISO",
    "ERCOT": "ERCOT",
    "FRCC": "FRCC",
    "ISO-NE": "ISO-NE",
    "MAPP": "MAPP",
    "MISO": "MISO",
    "NWPP": "NWPP",
    "NYISO": "NYISO",
    "PJM": "PJM",
    "RMRG": "RMRG",
    "SERC-E": "SERC-E",
    "SERC-N": "SERC-N",
    "SERC-SE": "SERC-SE",
    "SPP": "SPP",
    "SRSG": "SRSG",
}


MULTI_STATE_REGIONS = {
    "ISO-NE",
    "MAPP",
    "MISO",
    "NWPP",
    "PJM",
    "RMRG",
    "SERC-E",
    "SERC-N",
    "SERC-SE",
    "SPP",
    "SRSG",
}

GROUP_ORDER = ["Base only", "V1G", "V2G"]

GROUP_COLORS = {
    "Base only": "black",
    "V1G": "#4E79A7",   # blue
    "V2G": "#E15759",   # old V1G+V2G red
}

MARKER_MAP = {
    "0%": "o",
    "10%": "D",
    "25%": "s",
    "50%": "^",
}


# =========================================================
# Helper functionsa
# =========================================================
def resolve_duplicate_generation_cases(
    df,
    strategy="min_scenario_id",
    verbose=True,
):
    """
    Remove duplicated generation scenario folders.

    Duplicate means:
        same region, RPS, group, participation, battery capex,
        but more than one scenario folder.

    strategy:
        "min_scenario_id" -> keeps the lowest sXX folder.
                            This is usually correct when the current RPS order
                            starts later, e.g. [60, 70, 80].
        "max_scenario_id" -> keeps the highest sXX folder.
        "error"           -> raises an error.
    """

    key_cols = [
        "case_type",
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
    ]

    folder_cols = key_cols + [
        "scenario_id",
        "scenario_folder",
        "solution_json_path",
    ]

    folder_keys = df[folder_cols].drop_duplicates().copy()

    duplicate_summary = (
        folder_keys
        .groupby(key_cols, as_index=False)
        .agg(
            n_folders=("scenario_folder", "nunique"),
            folders=("scenario_folder", lambda x: "\n".join(map(str, x))),
        )
    )

    duplicated_cases = duplicate_summary[
        duplicate_summary["n_folders"] > 1
    ].copy()

    if duplicated_cases.empty:
        return df

    if strategy == "error":
        raise ValueError(
            "Duplicate generation scenario folders were found.\n\n"
            + duplicated_cases.to_string(index=False)
        )

    if verbose:
        print("\nDuplicate generation scenario folders were found.")
        print("The reader will keep only one folder per case.")
        print("Duplicate cases:")
        print(duplicated_cases.to_string(index=False))

    if strategy == "min_scenario_id":
        keep_keys = (
            folder_keys
            .sort_values(key_cols + ["scenario_id", "scenario_folder"])
            .drop_duplicates(subset=key_cols, keep="first")
        )

    elif strategy == "max_scenario_id":
        keep_keys = (
            folder_keys
            .sort_values(key_cols + ["scenario_id", "scenario_folder"])
            .drop_duplicates(subset=key_cols, keep="last")
        )

    else:
        raise ValueError(
            "Unknown duplicate strategy. Use: "
            "'min_scenario_id', 'max_scenario_id', or 'error'."
        )

    keep_folders = set(keep_keys["scenario_folder"].tolist())

    if verbose:
        print("\nKeeping these folders:")
        print(
            keep_keys[
                key_cols + ["scenario_id", "scenario_folder"]
            ]
            .sort_values(key_cols)
            .to_string(index=False)
        )

    df = df[df["scenario_folder"].isin(keep_folders)].copy()

    return df.reset_index(drop=True)
def validate_rps_coverage(df, region_codes, rps_axis_by_region, selected_participation):
    """
    Make sure each region has the expected RPS levels.

    Base only:
        should exist for every expected RPS level.

    V1G / V2G:
        should exist for every expected RPS level
        at each selected participation level.
    """

    problems = []

    for region in region_codes:
        expected_levels = list(rps_axis_by_region[region]["order"])
        sub_region = df[df["region"] == region].copy()

        # -----------------------------
        # Base only
        # -----------------------------
        base_levels = sorted(
            sub_region.loc[
                sub_region["group"] == "Base only",
                "rps_plot_value"
            ].dropna().unique().tolist()
        )

        missing_base = sorted(set(expected_levels) - set(base_levels))
        if missing_base:
            problems.append(
                f"{region} | Base only is missing RPS levels: {missing_base}"
            )

        # -----------------------------
        # V1G / V2G
        # -----------------------------
        for group in ["V1G", "V2G"]:
            for part in selected_participation:
                sub = sub_region[
                    (sub_region["group"] == group)
                    & (sub_region["participation"] == part)
                ].copy()

                levels = sorted(sub["rps_plot_value"].dropna().unique().tolist())
                missing = sorted(set(expected_levels) - set(levels))

                if missing:
                    problems.append(
                        f"{region} | {group}, {part} is missing RPS levels: {missing}"
                    )

    if problems:
        msg = "\n".join(problems)
        raise ValueError(
            "Some expected RPS scenarios are missing from the plot data:\n" + msg
        )

def add_direct_line_labels(
    ax,
    label_records,
    min_gap=0.9,
    x_pad_frac=0.018,
    fontsize=9.0,
    connector_line=True,
):
    """
    Add direct labels near the actual line endpoints.

    Parameters
    ----------
    ax : matplotlib axis
    label_records : list of dicts
        Each item should have:
            {
                "label": str,
                "x": float,
                "y": float,
                "color": str
            }

    min_gap : float
        Minimum vertical separation between labels, in y-axis data units.

    x_pad_frac : float
        Horizontal padding as a fraction of x-axis range.

    fontsize : float
        Text size for direct labels.

    connector_line : bool
        If True, draw a short connector from the real endpoint to the shifted label.
    """

    if not label_records:
        return

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_range = x_max - x_min
    x_pad = x_pad_frac * x_range

    # Sort by y so we can stack labels without overlap
    records = sorted(label_records, key=lambda d: d["y"])

    adjusted_y = []
    lower_bound = y_min + 0.6
    upper_bound = y_max - 0.2

    # Forward pass: enforce minimum spacing
    for i, rec in enumerate(records):
        target_y = rec["y"]

        if i == 0:
            new_y = max(target_y, lower_bound)
        else:
            new_y = max(target_y, adjusted_y[-1] + min_gap)

        adjusted_y.append(new_y)

    # If top label goes too high, shift all labels downward
    overflow = adjusted_y[-1] - upper_bound
    if overflow > 0:
        adjusted_y = [y - overflow for y in adjusted_y]

    # If bottom label goes too low after shifting, shift all upward
    underflow = lower_bound - adjusted_y[0]
    if underflow > 0:
        adjusted_y = [y + underflow for y in adjusted_y]

    # Draw labels
    for rec, y_lab in zip(records, adjusted_y):
        x0 = rec["x"]
        y0 = rec["y"]
        color = rec["color"]
        label = rec["label"]

        x_text = x0 + x_pad

        if connector_line:
            ax.plot(
                [x0, x_text - 0.10 * x_pad],
                [y0, y_lab],
                color=color,
                linewidth=1.0,
                alpha=0.9,
                zorder=25,
            )

        ax.text(
            x_text,
            y_lab,
            label,
            ha="left",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.72,
                pad=0.4,
            ),
            zorder=30,
        )

def parse_rps_info(folder_name, region=None):
    """
    Parse RPS information from scenario folder names.

    Supports normal numeric RPS:
        rps0, rps25, rps50, rps60, rps70, rps80

    Supports PJM relative RPS:
        rps-10
        rps+10
        rps_minus10
        rps_plus10
        rps_base
        rps0
        rps10
    """

    name = folder_name.lower()
    region = str(region).upper() if region is not None else None

    # =====================================================
    # PJM text-based relative RPS cases
    # =====================================================

    if re.search(r"rps[_-]?minus[_-]?10", name):
        rps_value = -10

    elif re.search(r"rps[_-]?base", name):
        rps_value = 0

    elif re.search(r"rps[_-]?plus[_-]?10", name):
        rps_value = 10

    # =====================================================
    # Signed numeric cases
    # Important:
    # This must come BEFORE normal numeric RPS.
    # Otherwise rps-10 can be incorrectly read as +10.
    # =====================================================

    else:
        signed_match = re.search(r"rps([+-]\d+)(?=_|$)", name)

        if signed_match:
            rps_value = int(signed_match.group(1))

        else:
            normal_match = re.search(r"rps[_]?(\d+)(?=_|$)", name)

            if normal_match:
                rps_value = int(normal_match.group(1))
            else:
                return {
                    "rps_case": None,
                    "rps_plot_value": None,
                    "rps_display_label": None,
                    "rps_ratio": None,
                }

    # =====================================================
    # Region-specific labeling
    # =====================================================

    if region == "PJM":
        if rps_value == -10:
            return {
                "rps_case": "minus10",
                "rps_plot_value": -10,
                "rps_display_label": "Base - 10",
                "rps_ratio": None,
            }

        if rps_value == 0:
            return {
                "rps_case": "base",
                "rps_plot_value": 0,
                "rps_display_label": "Base",
                "rps_ratio": None,
            }

        if rps_value == 10:
            return {
                "rps_case": "plus10",
                "rps_plot_value": 10,
                "rps_display_label": "Base + 10",
                "rps_ratio": None,
            }

        return {
            "rps_case": str(rps_value),
            "rps_plot_value": rps_value,
            "rps_display_label": f"Base {rps_value:+d}",
            "rps_ratio": None,
        }

    # Other regions use actual RPS percentages.
    return {
        "rps_case": str(rps_value),
        "rps_plot_value": rps_value,
        "rps_display_label": f"{rps_value}%",
        "rps_ratio": rps_value / 100.0,
    }

def normalize_region(region):
    """
    Convert state-style names to GOOD model-region names.

    Examples:
        CA -> CAISO
        NY -> NYISO
        TX -> ERCOT
        FL -> FRCC
        PJM -> PJM
    """
    region = str(region).upper()

    if region not in REGION_ALIASES:
        raise ValueError(
            f"Unknown region/state code: {region}. "
            f"Allowed values are: {sorted(REGION_ALIASES)}"
        )

    return REGION_ALIASES[region]

def normalize_label_map(label_map, regions, map_name):
    """
    Allows the user to provide either state keys or region keys.

    Example:
        {"CA": "flex", "NY": "flex", "TX": "midnight"}
    or:
        {"CAISO": "flex", "NYISO": "flex", "ERCOT": "midnight"}
    """

    if label_map is None:
        raise ValueError(f"{map_name} cannot be None.")

    out = {}

    for region in regions:
        region_code = normalize_region(region)

        if region in label_map:
            out[region_code] = label_map[region]

        elif region_code in label_map:
            out[region_code] = label_map[region_code]

        else:
            raise ValueError(
                f"Missing {map_name} for {region}. "
                f"Please add either key '{region}' or '{region_code}'."
            )

    return out

def parse_result_dir_name(folder_name):
    """
    Parse both old and new result folder formats.

    Old:
        scenario_results_2030_mid_even_CA_flex

    New:
        scenario_results_2030_mid_PJM_flex
        scenario_results_2030_mid_FRCC_flex
    """

    pattern = re.compile(
        r"^scenario_results_"
        r"(?P<year>\d{4})_"
        r"(?P<adoption>slow|mid|fast)"
        r"(?:_even)?_"
        r"(?P<region>[A-Za-z0-9-]+)_"
        r"(?P<scenario_label>[A-Za-z0-9]+)$",
        re.IGNORECASE,
    )

    match = pattern.match(folder_name)

    if match is None:
        return None

    region_code = normalize_region(match.group("region"))

    return {
        "year": int(match.group("year")),
        "adoption": match.group("adoption").lower(),
        "region": region_code,
        "scenario_label": match.group("scenario_label").lower(),
    }

def find_result_dirs(
    output_root,
    region,
    year=2030,
    adoption_levels=("slow", "mid", "fast"),
    scenario_labels=None,
):
    """
    Find valid GOOD scenario result folders.

    If both old and new folder formats exist for the same case,
    keep only one folder.

    Preference:
        1. New canonical region folder:
           scenario_results_2030_mid_NYISO_flex
           scenario_results_2030_mid_ERCOT_midnight

        2. Non-even folder

        3. Old _even_ folder
    """

    output_root = Path(output_root)
    region_code = normalize_region(region)
    region_dir = output_root / region_code

    if not region_dir.exists():
        raise FileNotFoundError(f"Region output folder does not exist: {region_dir}")

    adoption_levels = {a.lower() for a in adoption_levels}

    if scenario_labels is not None:
        scenario_labels = {s.lower() for s in scenario_labels}

    records = []

    for child in sorted(region_dir.iterdir()):
        if not child.is_dir():
            continue

        if not child.name.startswith("scenario_results_"):
            continue

        meta = parse_result_dir_name(child.name)

        if meta is None:
            continue

        if meta["region"] != region_code:
            continue

        if meta["year"] != year:
            continue

        if meta["adoption"] not in adoption_levels:
            continue

        if scenario_labels is not None and meta["scenario_label"] not in scenario_labels:
            continue

        expected_new_name = (
            f"scenario_results_"
            f"{meta['year']}_"
            f"{meta['adoption']}_"
            f"{region_code}_"
            f"{meta['scenario_label']}"
        ).lower()

        folder_name_lower = child.name.lower()

        # Lower value = higher priority
        if folder_name_lower == expected_new_name:
            folder_priority = 0
        elif "_even_" not in folder_name_lower:
            folder_priority = 1
        else:
            folder_priority = 2

        meta["results_dir"] = child
        meta["_folder_priority"] = folder_priority
        meta["_folder_name"] = child.name

        records.append(meta)

    # -----------------------------------------------------
    # Keep one folder per region-year-adoption-scenario_label
    # -----------------------------------------------------
    best = {}

    for rec in records:
        key = (
            rec["region"],
            rec["year"],
            rec["adoption"],
            rec["scenario_label"],
        )

        if key not in best:
            best[key] = rec
        else:
            old = best[key]

            old_rank = (
                old["_folder_priority"],
                len(str(old["results_dir"])),
                str(old["results_dir"]),
            )

            new_rank = (
                rec["_folder_priority"],
                len(str(rec["results_dir"])),
                str(rec["results_dir"]),
            )

            if new_rank < old_rank:
                best[key] = rec

    out = list(best.values())

    for rec in out:
        rec.pop("_folder_priority", None)
        rec.pop("_folder_name", None)

    return sorted(
        out,
        key=lambda d: (
            d["region"],
            d["year"],
            d["adoption"],
            d["scenario_label"],
            str(d["results_dir"]),
        ),
    )

def parse_scenario_id(text):
    """
    Extract scenario id from strings like:
        s01_Base_only_rps50...
        s16_V1G_rps70...
    """
    match = re.search(r"(?:^|_)s0*(\d+)(?:_|$)", text, re.IGNORECASE)

    if match is None:
        return None

    return int(match.group(1))

def get_objective_value(row, file_path):
    """
    Read objective value robustly.
    """
    possible_columns = [
        "objective_value",
        "total_cost",
        "system_cost",
    ]

    for col in possible_columns:
        if col in row and pd.notna(row[col]):
            return float(row[col])

    raise ValueError(
        f"No objective cost column found in {file_path}. "
        f"Tried columns: {possible_columns}"
    )

def is_policy_result_folder(folder_name):
    """
    Keep only folders that contain policy metadata.

    This avoids reading duplicate/simple folders like:
        s01_s01_fast_m7_d7

    and keeps folders like:
        s01_Base_only_rps50_v1g0_v2g0_bcapex150_fast_m7_d7
    """
    name = folder_name.lower()

    has_rps = "rps" in name
    has_bcapex = "bcapex" in name

    return has_rps and has_bcapex

def parse_model_days_from_folder(folder_name):
    """
    Extract model duration from folder names such as:
        ..._m0_d7
        ..._m0_d30
        ..._m0_d90
        ..._m0_d360
    """

    match = re.search(
        r"(?:^|_)d(\d+)(?:_|$)",
        str(folder_name),
        re.IGNORECASE,
    )

    if match is None:
        return None

    return int(match.group(1))

def load_total_costs_from_results(
    results_dir,
    scenario_label,
    region,
    year,
    adoption,
    model_days_per_run=None,
):
    """
    Load total-cost files from one scenario_results directory.

    When model_days_per_run is provided, only folders matching that
    model duration are loaded.
    """

    results_dir = Path(results_dir)

    total_files = list(results_dir.rglob("*_total_cost.csv"))
    summary_files = list(results_dir.rglob("*_summary.csv"))

    candidate_files = sorted(total_files + summary_files)

    selected_files = {}

    for file_path in candidate_files:
        folder = file_path.parent.name

        if not is_policy_result_folder(folder):
            continue

        folder_model_days = parse_model_days_from_folder(folder)

        # Critical fix: only read the requested model duration.
        if model_days_per_run is not None:
            if folder_model_days != int(model_days_per_run):
                continue

        scenario_id = parse_scenario_id(folder)

        if scenario_id is None:
            scenario_id = parse_scenario_id(file_path.name)

        if scenario_id is None:
            continue

        # Prefer total_cost over summary within the same folder.
        priority = (
            0
            if file_path.name.endswith("_total_cost.csv")
            else 1
        )

        key = (scenario_id, folder)

        if key not in selected_files:
            selected_files[key] = (
                priority,
                file_path,
                folder_model_days,
            )
        else:
            old_priority, _, _ = selected_files[key]

            if priority < old_priority:
                selected_files[key] = (
                    priority,
                    file_path,
                    folder_model_days,
                )

    records = []

    for (
        scenario_id,
        folder,
    ), (
        _,
        file_path,
        folder_model_days,
    ) in selected_files.items():

        try:
            temp = pd.read_csv(file_path)

            if temp.empty:
                continue

            row = temp.iloc[0].to_dict()

            match_v1g = re.search(
                r"v1g(\d+)",
                folder,
                re.IGNORECASE,
            )

            match_v2g = re.search(
                r"v2g(\d+)",
                folder,
                re.IGNORECASE,
            )

            match_bcapex = re.search(
                r"bcapex(\d+)",
                folder,
                re.IGNORECASE,
            )

            rps_info = parse_rps_info(
                folder,
                region=region,
            )

            v1g_val = (
                int(match_v1g.group(1))
                if match_v1g
                else 0
            )

            v2g_val = (
                int(match_v2g.group(1))
                if match_v2g
                else 0
            )

            # Skip old combined V1G + V2G cases.
            if v1g_val > 0 and v2g_val > 0:
                continue

            if v1g_val > 0:
                group = "V1G"
                participation_val = v1g_val

            elif v2g_val > 0:
                group = "V2G"
                participation_val = v2g_val

            else:
                group = "Base only"
                participation_val = 0

            objective_value = get_objective_value(
                row,
                file_path,
            )

            record = {
                "region": region,
                "year": year,
                "adoption": adoption.capitalize(),
                "scenario_label": scenario_label,
                "scenario_id": scenario_id,
                "scenario_tag": row.get("scenario_tag", ""),
                "objective_value": objective_value,
                "objective_value_mil": objective_value / 1e6,
                "file_path": str(file_path),
                "folder": folder,

                # Model duration
                "model_days_in_folder": folder_model_days,

                # V1G / V2G
                "v1g_share": v1g_val / 100.0,
                "v2g_share": v2g_val / 100.0,
                "participation": f"{participation_val}%",
                "group": group,

                # RPS
                "rps_case": rps_info["rps_case"],
                "rps_ratio": rps_info["rps_ratio"],
                "rps_percent": rps_info["rps_plot_value"],
                "rps_plot_value": rps_info["rps_plot_value"],
                "rps_display_label": rps_info[
                    "rps_display_label"
                ],

                # Battery CAPEX
                "batt_capex_num": (
                    int(match_bcapex.group(1))
                    if match_bcapex
                    else None
                ),
                "batt_capex_label": (
                    f"${int(match_bcapex.group(1))}/kWh"
                    if match_bcapex
                    else None
                ),
            }

            records.append(record)

        except Exception as exc:
            print(f"Could not read {file_path}: {exc}")

    return pd.DataFrame(records)

def load_region_total_costs(
    output_root,
    region,
    year=2030,
    adoption_level="mid",
    batt_capex=None,
    scenario_labels=None,
    model_days_per_run=None,
):
    """
    Load all total costs for one model region.
    """

    region_code = normalize_region(region)

    result_dirs = find_result_dirs(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption_levels=(adoption_level,),
        scenario_labels=scenario_labels,
    )

    df_list = []

    for item in result_dirs:
        temp = load_total_costs_from_results(
            results_dir=item["results_dir"],
            scenario_label=item["scenario_label"],
            region=region_code,
            year=item["year"],
            adoption=item["adoption"],
            model_days_per_run=model_days_per_run,
        )

        if not temp.empty:
            df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError(
            f"No valid total-cost data found for {region_code}. "
            f"Check Output/{region_code}/ and make sure the folder starts with scenario_results_."
        )

    df = pd.concat(df_list, ignore_index=True)

    if batt_capex is not None:
        df = df[df["batt_capex_num"] == batt_capex].copy()

    return df.reset_index(drop=True)

def load_one_region_for_rps_plot(
    output_root,
    region,
    year,
    adoption_level,
    batt_capex,
    program_scenario_label,
    baseline_scenario_label,
):
    """
    Load the exact data needed for the RPS same-cost plot.

    Base only:
        loaded from baseline_scenario_label

    V1G, V2G, V1G+V2G:
        loaded from program_scenario_label
    """

    needed_labels = sorted({
        program_scenario_label.lower(),
        baseline_scenario_label.lower(),
    })

    df = load_region_total_costs(
        output_root=output_root,
        region=region,
        year=year,
        adoption_level=adoption_level,
        batt_capex=batt_capex,
        scenario_labels=needed_labels,
    )

    base = df[
        (df["group"] == "Base only")
        & (df["scenario_label"] == baseline_scenario_label.lower())
    ].copy()

    programs = df[
        (df["group"].isin(["V1G", "V2G", "V1G+V2G"]))
        & (df["scenario_label"] == program_scenario_label.lower())
    ].copy()

    out = pd.concat([base, programs], ignore_index=True)

    if out.empty:
        raise ValueError(
            f"No plot data found for {region}. "
            f"Program={program_scenario_label}, "
            f"baseline={baseline_scenario_label}, "
            f"batt_capex={batt_capex}, "
            f"adoption={adoption_level}."
        )

    return out


def summarize_available_result_dirs(output_root, regions=None, year=2030):
    """
    Quick diagnostic table for the notebook.
    Shows which scenario_results folders exist.
    """

    output_root = Path(output_root)

    if regions is None:
        regions = ["CAISO", "ERCOT", "FRCC", "NYISO", "PJM"]

    records = []

    for region in regions:
        region_code = normalize_region(region)
        region_dir = output_root / region_code

        if not region_dir.exists():
            continue

        for child in sorted(region_dir.iterdir()):
            if not child.is_dir():
                continue

            meta = parse_result_dir_name(child.name)

            if meta is None:
                continue

            if year is not None and meta["year"] != year:
                continue

            records.append({
                "region": region_code,
                "year": meta["year"],
                "adoption": meta["adoption"],
                "scenario_label": meta["scenario_label"],
                "folder": child.name,
            })

    return pd.DataFrame(records)

def add_ca_direct_line_labels(ax, ca_label_records):
    """
    Add labels directly next to the actual California scenario lines.

    This is direct line labeling, not a legend/key.
    """

    if len(ca_label_records) == 0:
        return

    # Manual offsets to avoid overlap.
    label_offsets = {
        "Base": (0.35, 0.00),
        "V1G, 25%": (0.35, -0.45),
        "V1G, 50%": (0.35, 0.45),
        "V2G, 25%": (0.35, -0.70),
        "V2G, 50%": (0.35, 0.70),
    }

    for item in ca_label_records:
        label = item["label"]
        x = item["x"]
        y = item["y"]
        color = item["color"]

        dx, dy = label_offsets.get(label, (0.35, 0.0))

        ax.text(
            x + dx,
            y + dy,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
            color=color,
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=0.8,
            ),
            zorder=30,
        )

import matplotlib.colors as mcolors


def get_contrast_text_color(facecolor, dark_threshold=0.48):
    """
    Return white text for dark bar colors and black text for light bar colors.
    """

    try:
        r, g, b = mcolors.to_rgb(facecolor)
    except ValueError:
        return "black"

    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if luminance < dark_threshold:
        return "white"

    return "black"


def get_resource_label_threshold(
    label_threshold_by_region,
    region,
    resource,
    default_threshold=20,
):
    """
    Allows either:

    label_threshold_by_region = {
        "TX": 500,
        "NY": 300,
    }

    or:

    label_threshold_by_region = {
        "TX": {
            "default": 1000,
            "Solar": 2000,
            "Wind": 2000,
            "Nuclear": 800,
        }
    }
    """

    region_setting = label_threshold_by_region.get(region, default_threshold)

    if isinstance(region_setting, dict):
        return region_setting.get(
            resource,
            region_setting.get("default", default_threshold),
        )

    return region_setting


def get_max_labels_for_region(max_labels_per_bar_by_region, region, default_value=4):
    """
    Limit the number of labels inside each stacked bar.
    This prevents crowded panels like NYISO, ERCOT, and FRCC.
    """

    if max_labels_per_bar_by_region is None:
        return default_value

    return max_labels_per_bar_by_region.get(region, default_value)


def validate_generation_rps_coverage(
    df,
    region_codes,
    rps_order_by_region,
    selected_group,
    selected_participation,
    strict=True,
):
    """
    Check whether every requested RPS level exists for the selected group
    and participation level.

    This catches cases like:
        ERCOT requested [0, 60, 70],
        but V2G 50% only exists for [0, 60].
    """

    problems = []
    diagnostics = []

    for region in region_codes:
        expected_rps = list(rps_order_by_region[region])

        for part in selected_participation:
            sub = df[
                (df["region"] == region)
                & (df["group"] == selected_group)
                & (df["participation"] == part)
            ].copy()

            available_rps = sorted(
                sub["rps"].dropna().unique().tolist()
            )

            missing_rps = [
                rps for rps in expected_rps
                if rps not in available_rps
            ]

            if missing_rps:
                problems.append(
                    f"{region}, {selected_group}, {part}: "
                    f"missing requested RPS {missing_rps}; "
                    f"available RPS = {available_rps}"
                )

            for rps in expected_rps:
                sub_rps = sub[sub["rps"] == rps].copy()

                if sub_rps.empty:
                    diagnostics.append({
                        "region": region,
                        "group": selected_group,
                        "participation": part,
                        "rps": rps,
                        "status": "missing",
                        "total_abs_delta_gwh": None,
                    })
                else:
                    total_abs_delta = sub_rps["delta_gwh"].abs().sum()

                    diagnostics.append({
                        "region": region,
                        "group": selected_group,
                        "participation": part,
                        "rps": rps,
                        "status": "present",
                        "total_abs_delta_gwh": total_abs_delta,
                    })

    diagnostic_df = pd.DataFrame(diagnostics)

    if problems:
        message = (
            "Some requested RPS cases are missing from the generation plot data:\n"
            + "\n".join(problems)
        )

        if strict:
            raise ValueError(message)
        else:
            print("\nWARNING:")
            print(message)

    return diagnostic_df
# =========================================================
# Main plotting function
# =========================================================
def plot_rps_reached_at_same_cost(
    output_root,
    regions=("CA", "NY", "TX"),
    year=2030,
    adoption_level="mid",
    batt_capex=150,
    program_scenario_by_region=None,
    baseline_scenario_by_region=None,
    rps_axis_by_region=None,
    selected_participation=("30%", "50%"),
    direct_label_regions=None,
    direct_label_min_gap=0.9,
    direct_label_fontsize=9.0,
    save_path=None,
    log_x=False,
):
    """
    Plot RPS target reached at the same total system cost.

    rps_axis_by_region example:
    {
        "CA":  {"order": [50, 60, 70], "labels": ["50%", "60%", "70%"]},
        "NY":  {"order": [60, 70, 80], "labels": ["60%", "70%", "80%"]},
        "TX":  {"order": [50, 60, 70], "labels": ["50%", "60%", "70%"]},
        "FL":  {"order": [25, 50],     "labels": ["25%", "50%"]},
        "PJM": {"order": [-10, 0, 10], "labels": ["Base - 10", "Base", "Base + 10"]},
    }
    """

    region_codes = [normalize_region(r) for r in regions]
    if direct_label_regions is None:
        direct_label_regions = []

    direct_label_regions = {normalize_region(r) for r in direct_label_regions}


    program_map = normalize_label_map(
        program_scenario_by_region,
        regions,
        "program_scenario_by_region",
    )

    baseline_map = normalize_label_map(
        baseline_scenario_by_region,
        regions,
        "baseline_scenario_by_region",
    )

    all_data = []

    for region in region_codes:
        temp = load_one_region_for_rps_plot(
            output_root=output_root,
            region=region,
            year=year,
            adoption_level=adoption_level,
            batt_capex=batt_capex,
            program_scenario_label=program_map[region],
            baseline_scenario_label=baseline_map[region],
        )
        all_data.append(temp)

    df = pd.concat(all_data, ignore_index=True)

    # -----------------------------------
    # default RPS axis if user does not pass it
    # -----------------------------------
    if rps_axis_by_region is None:
        rps_axis_by_region = {}
        for region in region_codes:
            vals = (
                df.loc[df["region"] == region, "rps_plot_value"]
                .dropna()
                .astype(float)
                .sort_values()
                .unique()
                .tolist()
            )
            labels = []
            for v in vals:
                if region == "PJM":
                    if v == -10:
                        labels.append("Base - 10")
                    elif v == 0:
                        labels.append("Base")
                    elif v == 10:
                        labels.append("Base + 10")
                    else:
                        labels.append(str(int(v)))
                else:
                    labels.append(f"{int(v)}%")

            rps_axis_by_region[region] = {
                "order": vals,
                "labels": labels,
            }
    else:
        normalized = {}
        for key, value in rps_axis_by_region.items():
            normalized[normalize_region(key)] = value
        rps_axis_by_region = normalized

    # -----------------------------------
    # keep only selected RPS scenarios
    # -----------------------------------
    filtered_parts = []
    missing_rps_messages = []

    # Build lookup so plotted lines follow the user-selected order,
    # not just numeric sorting.
    rps_order_lookup = {}

    for region in region_codes:
        if region not in rps_axis_by_region:
            raise ValueError(
                f"Missing rps_axis_by_region entry for {region}. "
                f"Please add an order/labels entry for this region."
            )

        selected_rps_values = list(rps_axis_by_region[region]["order"])

        if len(selected_rps_values) == 0:
            raise ValueError(f"Empty RPS order for {region}.")

        if len(rps_axis_by_region[region]["labels"]) != len(selected_rps_values):
            raise ValueError(
                f"RPS labels length does not match order length for {region}.\n"
                f"order={selected_rps_values}\n"
                f"labels={rps_axis_by_region[region]['labels']}"
            )

        for idx, rps_value in enumerate(selected_rps_values):
            rps_order_lookup[(region, rps_value)] = idx

        sub = df[
            (df["region"] == region)
            & (df["rps_plot_value"].isin(selected_rps_values))
            ].copy()

        available_rps_values = (
            sub["rps_plot_value"]
            .dropna()
            .unique()
            .tolist()
        )

        missing_rps = [
            rps for rps in selected_rps_values
            if rps not in available_rps_values
        ]

        if missing_rps:
            missing_rps_messages.append(
                f"{region}: requested {missing_rps}, "
                f"available after filtering = {sorted(available_rps_values)}"
            )

        filtered_parts.append(sub)

    if missing_rps_messages:
        raise ValueError(
            "Some selected RPS scenarios are missing from the loaded data:\n"
            + "\n".join(missing_rps_messages)
        )

    df = pd.concat(filtered_parts, ignore_index=True)

    df["rps_order_index"] = df.apply(
        lambda row: rps_order_lookup.get(
            (row["region"], row["rps_plot_value"]),
            999,
        ),
        axis=1,
    )

    df = df[
        (df["group"] == "Base only")
        | (df["participation"].isin(selected_participation))
    ].copy()

    # remove any leftover V1G+V2G just in case
    df = df[df["group"].isin(["Base only", "V1G", "V2G"])].copy()
    # -----------------------------------
    # remove duplicated plot rows
    # -----------------------------------
    dedupe_cols = [
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps_plot_value",
        "batt_capex_num",
    ]

    duplicate_rows = df[df.duplicated(dedupe_cols, keep=False)].copy()

    if not duplicate_rows.empty:
        print("\nDuplicate plot rows were found and removed.")
        print("This usually means both old and new result folders exist.")
        print(
            duplicate_rows[
                dedupe_cols + ["objective_value_mil", "folder", "file_path"]
                ]
            .sort_values(dedupe_cols + ["objective_value_mil"])
            .to_string(index=False)
        )

        # Keep one point per region / RPS / group / participation.
        # If the new canonical-folder fix above is active, this should rarely be needed.
        df = (
            df
            .sort_values(
                dedupe_cols + ["objective_value_mil", "file_path"]
            )
            .drop_duplicates(
                subset=dedupe_cols,
                keep="last",
            )
            .reset_index(drop=True)
        )
    # -----------------------------------
    # layout
    # -----------------------------------
    n_regions = len(region_codes)

    if n_regions <= 3:
        nrows = 1
        ncols = n_regions
        figsize = (5.3 * ncols, 5.4)
    else:
        nrows = 2
        ncols = 3
        figsize = (16.5, 9.0)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
    )

    axes = axes_grid.flatten()

    # -----------------------------------
    # plot panels
    # -----------------------------------
    for plot_index, region in enumerate(region_codes):
        ax = axes[plot_index]

        sub_region = df[df["region"] == region].copy()
        axis_info = rps_axis_by_region[region]
        region_order = axis_info["order"]
        region_labels = axis_info["labels"]

        ax.set_facecolor("white")
        direct_label_records = []
        # baseline rows
        base_rows = (
            sub_region[sub_region["group"] == "Base only"]
            .sort_values("rps_order_index")
            .reset_index(drop=True)
        )
        ca_label_records = []
        for base_index, row in base_rows.iterrows():
            base_cost = row["objective_value_mil"]
            base_y = row["rps_plot_value"]

            ax.axvline(
                base_cost,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                zorder=1,
            )

            label_y = max(region_order) + 0.7 + 0.8 * (base_index % 2)

            ax.text(
                base_cost,
                label_y,
                f"Base\n{row['rps_display_label']}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="dimgray",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.0),
                zorder=5,
            )

        # lines

        for group in GROUP_ORDER:
            sub_group = sub_region[sub_region["group"] == group].copy()

            if sub_group.empty:
                continue

            if group == "Base only":
                plot_data = sub_group.sort_values("rps_order_index")

                ax.plot(
                    plot_data["objective_value_mil"],
                    plot_data["rps_plot_value"],
                    color=GROUP_COLORS[group],
                    marker=MARKER_MAP["0%"],
                    linewidth=2.4,
                    markersize=7.5,
                    zorder=4,
                )

                if region in direct_label_regions and not plot_data.empty:
                    last_row = plot_data.iloc[-1]
                    direct_label_records.append({
                        "label": "Base",
                        "x": last_row["objective_value_mil"],
                        "y": last_row["rps_plot_value"],
                        "color": GROUP_COLORS[group],
                    })

            else:
                for part in selected_participation:
                    plot_data = (
                        sub_group[sub_group["participation"] == part]
                        .sort_values("rps_order_index")
                    )

                    if plot_data.empty:
                        continue

                    ax.plot(
                        plot_data["objective_value_mil"],
                        plot_data["rps_plot_value"],
                        color=GROUP_COLORS[group],
                        marker=MARKER_MAP.get(part, "o"),
                        linewidth=2.1,
                        markersize=7.5,
                        zorder=3,
                    )

                    if region in direct_label_regions and not plot_data.empty:
                        last_row = plot_data.iloc[-1]
                        direct_label_records.append({
                            "label": f"{group}, {part}",
                            "x": last_row["objective_value_mil"],
                            "y": last_row["rps_plot_value"],
                            "color": GROUP_COLORS[group],
                        })

        if log_x:
            ax.set_xscale("log")

        ax.set_title(
            REGION_DISPLAY_NAMES.get(region, region),
            fontsize=18,
            color="black",
            pad=8,
        )

        ax.set_xlabel(
            "Total system cost (Million $)",
            fontsize=14,
            color="black",
        )

        ax.set_ylabel(
            "RPS target reached",
            fontsize=14,
            color="black",
        )

        ax.set_yticks(region_order)
        ax.set_yticklabels(region_labels, fontsize=11, color="black")
        ax.set_ylim(min(region_order) - 3, max(region_order) + 4)

        ax.tick_params(axis="x", colors="black", labelsize=11)
        ax.tick_params(axis="y", colors="black", labelsize=11)

        ax.grid(True, which="both", linestyle="--", alpha=0.35, color="gray")

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)
        # Add direct labels to the actual California lines
        if region in direct_label_regions:
            add_direct_line_labels(
                ax,
                direct_label_records,
                min_gap=direct_label_min_gap,
                fontsize=direct_label_fontsize,
            )

    # -----------------------------------
    # use empty panel for legend if available
    # -----------------------------------
    if len(axes) > n_regions:
        legend_ax = axes[n_regions]
        legend_ax.set_facecolor("white")
        legend_ax.axis("off")

        legend_handles = [
            Line2D([0], [0], color="black", marker="o", lw=2.4, markersize=7.5, label="Base"),
            Line2D([0], [0], color=GROUP_COLORS["V1G"], marker=MARKER_MAP["25%"], lw=2.1, markersize=7.5, label="V1G, 25%"),
            Line2D([0], [0], color=GROUP_COLORS["V1G"], marker=MARKER_MAP["50%"], lw=2.1, markersize=7.5, label="V1G, 50%"),
            Line2D([0], [0], color=GROUP_COLORS["V2G"], marker=MARKER_MAP["25%"], lw=2.1, markersize=7.5, label="V2G, 25%"),
            Line2D([0], [0], color=GROUP_COLORS["V2G"], marker=MARKER_MAP["50%"], lw=2.1, markersize=7.5, label="V2G, 50%"),
        ]

        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=14,
            ncol=1,
        )

        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)

        for txt in legend.get_texts():
            txt.set_color("black")

        # turn off any other unused panel
        for extra_ax in axes[n_regions + 1:]:
            extra_ax.axis("off")
    else:
        # fallback: put legend inside first panel
        handles = [
            Line2D([0], [0], color="black", marker="o", lw=2.4, markersize=7.5, label="Base"),
            Line2D([0], [0], color=GROUP_COLORS["V1G"], marker=MARKER_MAP["25%"], lw=2.1, markersize=7.5, label="V1G, 25%"),
            Line2D([0], [0], color=GROUP_COLORS["V1G"], marker=MARKER_MAP["50%"], lw=2.1, markersize=7.5, label="V1G, 50%"),
            Line2D([0], [0], color=GROUP_COLORS["V2G"], marker=MARKER_MAP["25%"], lw=2.1, markersize=7.5, label="V2G, 25%"),
            Line2D([0], [0], color=GROUP_COLORS["V2G"], marker=MARKER_MAP["50%"], lw=2.1, markersize=7.5, label="V2G, 50%"),
        ]
        axes[0].legend(handles=handles, loc="best", frameon=True, fontsize=11)

    # -----------------------------------
    # turn off unused panels
    # legend is drawn inside California only
    # -----------------------------------
    # if len(axes) > n_regions:
    #     for extra_ax in axes[n_regions:]:
    #         extra_ax.axis("off")

    scale_label = "log scale" if log_x else "linear scale"

    fig.suptitle(
        f"RPS target reached under different EV charging strategies\n"
        f"Battery capex = ${batt_capex}/kWh, x axis = {scale_label}",
        fontsize=19,
        color="black",
        y=0.975,
    )

    if nrows == 1:
        fig.subplots_adjust(
            top=0.80,
            bottom=0.16,
            left=0.07,
            right=0.98,
            wspace=0.24,
        )
    else:
        fig.subplots_adjust(
            top=0.84,
            bottom=0.08,
            left=0.06,
            right=0.98,
            hspace=0.42,
            wspace=0.25,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return df, fig, axes

def compute_total_cost_delta_vs_rps_for_region(
    output_root,
    region,
    year,
    adoption_level,
    batt_order,
    baseline_scenario_label,
    program_scenario_label,
    rps_order=None,
    selected_participation=("30%", "50%"),
):
    """
    Compute delta total system cost relative to the baseline scenario
    at the same RPS level and battery capex, for one region.
    """

    region_code = normalize_region(region)

    needed_labels = sorted({
        baseline_scenario_label.lower(),
        program_scenario_label.lower(),
    })

    df = load_region_total_costs(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption_level=adoption_level,
        batt_capex=None,
        scenario_labels=needed_labels,
    )

    df = df[df["batt_capex_num"].isin(batt_order)].copy()
    df = df[df["group"].isin(["Base only", "V1G", "V2G"])].copy()

    if rps_order is not None:
        df = df[df["rps_plot_value"].isin(rps_order)].copy()

    df = df[
        (df["group"] == "Base only")
        | (df["participation"].isin(selected_participation))
    ].copy()

    # Baseline rows
    base = df[
        (df["group"] == "Base only")
        & (df["scenario_label"] == baseline_scenario_label.lower())
    ][
        ["region", "batt_capex_num", "rps_plot_value", "objective_value_mil"]
    ].copy()

    base = base.rename(
        columns={"objective_value_mil": "baseline_objective_value_mil"}
    )

    # Program rows
    prog = df[
        (df["group"].isin(["V1G", "V2G"]))
        & (df["scenario_label"] == program_scenario_label.lower())
    ].copy()

    merged = prog.merge(
        base,
        on=["region", "batt_capex_num", "rps_plot_value"],
        how="left",
    )

    if merged["baseline_objective_value_mil"].isna().any():
        missing = merged[merged["baseline_objective_value_mil"].isna()][
            ["group", "participation", "batt_capex_num", "rps_plot_value"]
        ]
        raise ValueError(
            f"Missing baseline rows for {region_code}. "
            f"Examples:\n{missing.head()}"
        )

    merged["delta_cost_mil"] = (
        merged["objective_value_mil"] - merged["baseline_objective_value_mil"]
    )

    merged["series_label"] = (
        merged["group"] + ", " + merged["participation"]
    )

    return merged.reset_index(drop=True)

def plot_total_cost_delta_vs_rps_regions(
    output_root,
    regions=("CA", "NY", "TX"),
    year=2030,
    adoption_level="mid",
    batt_order=(150, 250),
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    rps_order_by_region=None,
    selected_participation=("30%", "50%"),
    panel_title_by_region=None,
    figure_title=None,
    save_path=None,
):
    """
    Plot delta total system cost vs RPS for one or more regions.

    Delta cost is computed relative to the baseline scenario at the same:
        - region
        - RPS level
        - battery capex
    """

    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    if panel_title_by_region is None:
        panel_title_by_region = {}

    panel_title_by_region = {
        normalize_region(k): v
        for k, v in panel_title_by_region.items()
    }

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    # -----------------------------------------------------
    # Load all region data
    # -----------------------------------------------------
    df_list = []

    for region in region_codes:
        temp = compute_total_cost_delta_vs_rps_for_region(
            output_root=output_root,
            region=region,
            year=year,
            adoption_level=adoption_level,
            batt_order=batt_order,
            baseline_scenario_label=baseline_map[region],
            program_scenario_label=program_map[region],
            rps_order=rps_order_by_region.get(region, None),
            selected_participation=selected_participation,
        )
        df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError("No region data found for plotting.")

    df_delta_all = pd.concat(df_list, ignore_index=True)

    # -----------------------------------------------------
    # Build default RPS axis labels if not provided
    # -----------------------------------------------------
    rps_axis_by_region = {}

    for region in region_codes:
        if region in rps_order_by_region:
            order_vals = rps_order_by_region[region]
        else:
            order_vals = (
                df_delta_all.loc[df_delta_all["region"] == region, "rps_plot_value"]
                .dropna()
                .astype(float)
                .sort_values()
                .unique()
                .tolist()
            )

        labels = []
        for v in order_vals:
            if region == "PJM":
                if v == -10:
                    labels.append("Base - 10")
                elif v == 0:
                    labels.append("Base")
                elif v == 10:
                    labels.append("Base + 10")
                else:
                    labels.append(str(int(v)))
            else:
                labels.append(f"{int(v)}%")

        rps_axis_by_region[region] = {
            "order": order_vals,
            "labels": labels,
        }

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    n_regions = len(region_codes)

    if n_regions == 1:
        nrows = 1
        ncols = 2
        figsize = (11.0, 5.5)
    elif n_regions <= 3:
        nrows = 1
        ncols = n_regions
        figsize = (5.3 * ncols, 5.4)
    else:
        nrows = 2
        ncols = 3
        figsize = (16.5, 9.0)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
    )

    axes = axes_grid.flatten()

    # -----------------------------------------------------
    # Style settings
    # -----------------------------------------------------
    series_order = [
        ("V1G", "25%"),
        ("V1G", "50%"),
        ("V2G", "25%"),
        ("V2G", "50%"),
    ]

    batt_order_sorted = tuple(sorted(batt_order))

    # -----------------------------------------------------
    # Plot each region
    # -----------------------------------------------------
    for plot_index, region in enumerate(region_codes):
        ax = axes[plot_index]
        sub = df_delta_all[df_delta_all["region"] == region].copy()

        axis_info = rps_axis_by_region[region]
        x_vals = axis_info["order"]
        x_labels = axis_info["labels"]

        ax.set_facecolor("white")

        # baseline = 0 line
        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=1,
        )

        for group, part in series_order:
            sub_series = sub[
                (sub["group"] == group)
                & (sub["participation"] == part)
            ].copy()

            if sub_series.empty:
                continue

            pivot = sub_series.pivot_table(
                index="rps_plot_value",
                columns="batt_capex_num",
                values="delta_cost_mil",
                aggfunc="first",
            )

            pivot = pivot.reindex(x_vals)

            available_batts = [b for b in batt_order_sorted if b in pivot.columns]
            if len(available_batts) == 0:
                continue

            low = pivot[available_batts].min(axis=1)
            high = pivot[available_batts].max(axis=1)
            mid = pivot[available_batts].mean(axis=1)

            color = GROUP_COLORS[group]
            marker = MARKER_MAP.get(part, "o")

            # band between battery capex cases
            if len(available_batts) >= 2:
                ax.fill_between(
                    x_vals,
                    low.values,
                    high.values,
                    color=color,
                    alpha=0.12,
                    zorder=2,
                )

                # lower and upper band edges
                ax.plot(
                    x_vals,
                    pivot[available_batts[0]].values,
                    color=color,
                    linewidth=1.0,
                    alpha=0.45,
                    marker=marker,
                    markersize=4.5,
                    zorder=3,
                )

                ax.plot(
                    x_vals,
                    pivot[available_batts[-1]].values,
                    color=color,
                    linewidth=1.0,
                    alpha=0.45,
                    marker=marker,
                    markersize=4.5,
                    zorder=3,
                )

            # mid line
            ax.plot(
                x_vals,
                mid.values,
                color=color,
                linewidth=2.2,
                marker=marker,
                markersize=6.5,
                zorder=4,
            )

        panel_title = panel_title_by_region.get(
            region,
            REGION_DISPLAY_NAMES.get(region, region),
        )

        ax.set_title(
            panel_title,
            fontsize=18,
            color="black",
            pad=8,
        )

        ax.set_xlabel("RPS target reached", fontsize=14, color="black")
        ax.set_ylabel("Δ total system cost (Million $)", fontsize=14, color="black")

        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=11, color="black")

        ax.tick_params(axis="x", colors="black", labelsize=11)
        ax.tick_params(axis="y", colors="black", labelsize=11)

        ax.grid(True, which="both", linestyle="--", alpha=0.35, color="gray")

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color=GROUP_COLORS["V1G"], lw=2.2, label="V1G"),
        Line2D([0], [0], color=GROUP_COLORS["V2G"], lw=2.2, label="V2G"),
        Line2D([0], [0], color="black", marker=MARKER_MAP["25%"], lw=0, markersize=7, label="25% participation"),
        Line2D([0], [0], color="black", marker=MARKER_MAP["50%"], lw=0, markersize=7, label="50% participation"),
        Line2D([0], [0], color="black", linestyle="--", lw=1.2, label="Baseline = 0"),
    ]

    if len(batt_order_sorted) >= 2:
        legend_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="none",
                alpha=0.15,
                label=f"Battery capex band (${batt_order_sorted[0]}–${batt_order_sorted[-1]}/kWh)",
            )
        )
    else:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="gray",
                lw=2.0,
                label=f"Battery capex = ${batt_order_sorted[0]}/kWh",
            )
        )

    if len(axes) > n_regions:
        legend_ax = axes[n_regions]
        legend_ax.set_facecolor("white")
        legend_ax.axis("off")

        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=13,
            ncol=1,
        )

        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)

        for txt in legend.get_texts():
            txt.set_color("black")

        for extra_ax in axes[n_regions + 1:]:
            extra_ax.axis("off")
    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.02),
            ncol=min(3, len(legend_handles)),
            frameon=True,
            fontsize=11,
        )

    # -----------------------------------------------------
    # Figure title
    # -----------------------------------------------------
    if figure_title is None:
        if len(batt_order_sorted) >= 2:
            batt_text = f"${batt_order_sorted[0]}–${batt_order_sorted[-1]}/kWh"
        else:
            batt_text = f"${batt_order_sorted[0]}/kWh"

        figure_title = (
            "Delta total system cost vs RPS target reached\n"
            f"Battery capex = {batt_text}"
        )

    fig.suptitle(
        figure_title,
        fontsize=19,
        color="black",
        y=0.975,
    )

    if n_regions == 1:
        fig.subplots_adjust(
            top=0.82,
            bottom=0.13,
            left=0.08,
            right=0.97,
            wspace=0.22,
        )
    elif n_regions <= 3:
        fig.subplots_adjust(
            top=0.82,
            bottom=0.16,
            left=0.07,
            right=0.98,
            wspace=0.24,
        )
    else:
        fig.subplots_adjust(
            top=0.84,
            bottom=0.08,
            left=0.06,
            right=0.98,
            hspace=0.42,
            wspace=0.25,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return df_delta_all, fig, axes

def plot_total_cost_delta_vs_rps_one_region(
    output_root,
    region,
    year=2030,
    adoption_level="mid",
    batt_order=(150, 250),
    baseline_scenario_label="arrive",
    program_scenario_label="flex",
    rps_order=None,
    panel_title=None,
    figure_title=None,
    selected_participation=("30%", "50%"),
    save_path=None,
):
    """
    Convenience wrapper to plot only one region.
    """

    region_code = normalize_region(region)

    return plot_total_cost_delta_vs_rps_regions(
        output_root=output_root,
        regions=(region_code,),
        year=year,
        adoption_level=adoption_level,
        batt_order=batt_order,
        baseline_scenario_label_by_region={region_code: baseline_scenario_label},
        program_scenario_label_by_region={region_code: program_scenario_label},
        rps_order_by_region={region_code: rps_order} if rps_order is not None else None,
        selected_participation=selected_participation,
        panel_title_by_region={region_code: panel_title} if panel_title is not None else None,
        figure_title=figure_title,
        save_path=save_path,
    )

def compute_total_cost_delta_for_region_battery_adoption_range(
    output_root,
    region,
    year,
    adoption_levels=("slow", "mid", "fast"),
    batt_capex=150,
    baseline_scenario_label="arrive",
    program_scenario_label="flex",
    rps_order=None,
    selected_participation=("30%", "50%"),
):
    """
    Compute delta total system cost relative to the baseline scenario
    for one region, one battery capex value, and multiple adoption levels.

    This supports:
        main line = Mid adoption
        shaded band = Slow to Fast adoption range
    """

    region_code = normalize_region(region)

    needed_labels = sorted({
        baseline_scenario_label.lower(),
        program_scenario_label.lower(),
    })

    df_list = []

    for adoption_level in adoption_levels:
        temp = load_region_total_costs(
            output_root=output_root,
            region=region_code,
            year=year,
            adoption_level=adoption_level,
            batt_capex=batt_capex,
            scenario_labels=needed_labels,
        )

        if not temp.empty:
            df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError(
            f"No data found for {region_code}, batt_capex={batt_capex}, "
            f"adoption_levels={adoption_levels}"
        )

    df = pd.concat(df_list, ignore_index=True)

    df = df[df["group"].isin(["Base only", "V1G", "V2G"])].copy()

    if rps_order is not None:
        df = df[df["rps_plot_value"].isin(rps_order)].copy()

    df = df[
        (df["group"] == "Base only")
        | (df["participation"].isin(selected_participation))
    ].copy()

    # -----------------------------
    # Baseline rows by adoption
    # -----------------------------
    base = df[
        (df["group"] == "Base only")
        & (df["scenario_label"] == baseline_scenario_label.lower())
    ][
        [
            "region",
            "adoption",
            "rps_plot_value",
            "objective_value_mil",
        ]
    ].copy()

    base = base.rename(
        columns={"objective_value_mil": "baseline_objective_value_mil"}
    )

    # -----------------------------
    # Program rows by adoption
    # -----------------------------
    program = df[
        (df["group"].isin(["V1G", "V2G"]))
        & (df["scenario_label"] == program_scenario_label.lower())
    ].copy()

    merged = program.merge(
        base,
        on=["region", "adoption", "rps_plot_value"],
        how="left",
    )

    if merged["baseline_objective_value_mil"].isna().any():
        missing = merged[merged["baseline_objective_value_mil"].isna()][
            [
                "region",
                "adoption",
                "group",
                "participation",
                "rps_plot_value",
                "batt_capex_num",
            ]
        ]

        raise ValueError(
            f"Missing baseline rows for {region_code}, batt_capex={batt_capex}.\n"
            f"{missing.head(30)}"
        )

    merged["delta_cost_mil"] = (
        merged["objective_value_mil"]
        - merged["baseline_objective_value_mil"]
    )

    merged["delta_cost_percent"] = 100 * (
        merged["objective_value_mil"]
        - merged["baseline_objective_value_mil"]
    ) / merged["baseline_objective_value_mil"]

    merged["batt_capex_num"] = batt_capex

    return merged.reset_index(drop=True)

def audit_generation_resource_mapping(solution_json_path, graph):
    """
    Show raw asset metadata and mapped resource categories.

    Use this when Hydro or Import looks missing.
    """

    import json
    import pandas as pd

    with open(solution_json_path) as f:
        data = json.load(f)

    rows = []

    for node in data["nodes"]:
        region = node["id"]

        if region not in graph._node:
            continue

        for handle, a_sol in node.get("assets", {}).items():
            if handle not in graph._node[region]["assets"]:
                continue

            meta = graph._node[region]["assets"][handle]

            cls = meta.get("_class", "")
            typ = meta.get("type", "")
            fuel = meta.get("fuel", None)

            mapped = map_fuel_for_plot_extended(
                fuel=fuel,
                typ=typ,
                cls=cls,
                handle=handle,
            )

            rows.append({
                "region": region,
                "handle": handle,
                "_class": cls,
                "type": typ,
                "fuel": fuel,
                "mapped_resource": mapped,
            })

    return pd.DataFrame(rows)

def generation_summary_from_solution_json(solution_json_path, graph, time_step_h=1.0):
    import json
    import numpy as np
    import pandas as pd

    with open(solution_json_path) as f:
        data = json.load(f)

    rows = []

    for node in data["nodes"]:
        region = node["id"]

        if region not in graph._node:
            continue

        for handle, a_sol in node.get("assets", {}).items():

            if handle not in graph._node[region]["assets"]:
                continue

            meta = graph._node[region]["assets"][handle]

            cls = str(meta.get("_class", ""))
            typ = str(meta.get("type", ""))
            fuel = meta.get("fuel", None)

            cls_lower = cls.lower()
            typ_lower = typ.lower()
            fuel_lower = str(fuel or "").lower()
            handle_lower = str(handle).lower()

            # -------------------------
            # 1. Skip real demand only
            # -------------------------
            if cls_lower == "load" and typ_lower == "load":
                continue

            # -------------------------
            # 2. Store assets
            # -------------------------
            if cls_lower == "store":
                prod = np.array(a_sol.get("production", []), dtype=float)

                if prod.size == 0:
                    continue

                is_v2g = (
                    typ_lower == "ev_v2g"
                    or fuel_lower == "ev_v2g"
                    or "ev_v2g" in handle_lower
                    or "v2g" in handle_lower
                )

                is_pumped_hydro = (
                    "hydro" in typ_lower
                    or "hydro" in fuel_lower
                    or "hydro" in handle_lower
                    or "pump" in typ_lower
                    or "pump" in fuel_lower
                    or "pump" in handle_lower
                )

                if is_v2g:
                    resource = "V2G Discharge"
                elif is_pumped_hydro:
                    resource = "Hydro"
                else:
                    resource = "Battery Discharge"

                energy_gwh = prod.sum() * time_step_h / 1e9

                rows.append({
                    "region": region,
                    "handle": handle,
                    "resource": resource,
                    "fuel": fuel,
                    "type": typ,
                    "_class": cls,
                    "energy_gwh": energy_gwh,
                })

                continue

            # -------------------------
            # 3. Producer assets
            # -------------------------
            if cls_lower == "producer":
                # Producer.solution() stores both production and net.
                # Use production first because it is the direct generator output.
                values = a_sol.get("production", a_sol.get("net", []))
                prod = np.array(values, dtype=float)

                if prod.size == 0:
                    continue

                gen = np.clip(prod, 0, None)
                energy_gwh = gen.sum() * time_step_h / 1e9

                resource = map_fuel_for_plot_extended(
                    fuel=fuel,
                    typ=typ,
                    cls=cls,
                    handle=handle,
                )

                rows.append({
                    "region": region,
                    "handle": handle,
                    "resource": resource,
                    "fuel": fuel,
                    "type": typ,
                    "_class": cls,
                    "energy_gwh": energy_gwh,
                })

                continue

            # -------------------------
            # 4. Other positive-output assets
            # Example: renewables modeled as positive Load with type small_hydro.
            # -------------------------
            values = a_sol.get("net", [])
            net = np.array(values, dtype=float)

            if net.size == 0:
                continue

            gen = np.clip(net, 0, None)
            energy_gwh = gen.sum() * time_step_h / 1e9

            resource = map_fuel_for_plot_extended(
                fuel=fuel,
                typ=typ,
                cls=cls,
                handle=handle,
            )

            rows.append({
                "region": region,
                "handle": handle,
                "resource": resource,
                "fuel": fuel,
                "type": typ,
                "_class": cls,
                "energy_gwh": energy_gwh,
            })

    if len(rows) == 0:
        raise ValueError(f"No generation data found in {solution_json_path}")

    detail = pd.DataFrame(rows)

    summary = (
        detail
        .groupby("resource", as_index=False)["energy_gwh"]
        .sum()
    )

    return summary
def map_fuel_for_plot_extended(fuel=None, typ=None, cls=None, handle=None):
    """
    Robust resource mapper for generation-mix plots.

    Important:
        - Import can be a Producer asset with fuel='import'.
        - Small/large hydro may appear in type or handle, not only fuel.
        - Pumped hydro may appear as a Store and should not be labeled as Battery.
    """

    raw_text = " ".join([
        str(fuel or ""),
        str(typ or ""),
        str(cls or ""),
        str(handle or ""),
    ]).lower()

    text = (
        raw_text
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("\\", " ")
    )

    # -----------------------------
    # Imports
    # -----------------------------
    if "import" in text or "imported" in text:
        return "Import"

    # -----------------------------
    # EV / storage resources
    # -----------------------------
    if "v2g" in text and "charge" not in text:
        return "V2G Discharge"

    if "battery discharge" in text or "battery_discharge" in raw_text:
        return "Battery Discharge"

    if "battery charge" in text or "battery_charge" in raw_text:
        return "Battery Charge"

    # -----------------------------
    # Hydro
    # Must come before generic non-fossil.
    # -----------------------------
    if (
        "hydro" in text
        or "hydroelectric" in text
        or "hydropower" in text
        or "small hydro" in text
        or "large hydro" in text
        or "pumped hydro" in text
        or "pump hydro" in text
    ):
        return "Hydro"

    # -----------------------------
    # Renewables / clean resources
    # -----------------------------
    if "solar" in text or "pv" in text:
        return "Solar"

    if "wind" in text:
        return "Wind"

    if "geothermal" in text:
        return "Geothermal"

    if "biomass" in text or "bio mass" in text:
        return "Biomass"

    if "waste" in text:
        return "Waste"

    if "nuclear" in text:
        return "Nuclear"

    if "non fossil" in text or "nonfossil" in text or "non-fossil" in raw_text:
        return "Non-fossil"

    # -----------------------------
    # Fossil resources
    # -----------------------------
    if "coal" in text:
        return "Coal"

    if (
        "natural gas combined cycle" in text
        or "combined cycle" in text
        or "ccng" in text
        or "ngcc" in text
    ):
        return "CCNG"

    if (
        "natural gas turbine" in text
        or "gas turbine" in text
        or "combustion turbine" in text
        or "ngct" in text
    ):
        return "Gas\nTurbine"

    if "oil" in text or "steam" in text:
        return "Steam"

    # -----------------------------
    # Generic storage fallback
    # -----------------------------
    if "battery" in text or "storage" in text or "store" in text:
        return "Battery Discharge"

    return "Other"

def plot_total_cost_delta_vs_rps_regions_by_battery(
    output_root,
    regions=("CA", "NY", "TX"),
    year=2030,
    adoption_levels=("slow", "mid", "fast"),
    main_adoption_level="Mid",
    band_adoption_levels=("Slow", "Fast"),
    batt_order=(150, 250),
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    rps_order_by_region=None,
    selected_participation=("30%", "50%"),
    panel_title_by_region=None,
    figure_title=None,

    # Left y-axis stays percentage reduction.
    delta_metric="percent",

    # New right y-axis.
    show_second_y_axis=True,
    second_y_axis_label="Total system cost\nreduction (million $)",

    show_adoption_band=True,
    save_path=None,

    # Font controls
    font_size=12,
    figure_title_fontsize=None,
    panel_title_fontsize=None,
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    legend_fontsize=None,
):
    """
    Plot total system cost reduction vs RPS.

    Left y-axis:
        Percent total system cost reduction relative to Base.

    Right y-axis:
        Absolute total system cost reduction in million dollars.

    Positive values mean the program lowers total system cost relative to Base.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # =====================================================
    # Font controls
    # =====================================================
    if figure_title_fontsize is None:
        figure_title_fontsize = font_size + 6

    if panel_title_fontsize is None:
        panel_title_fontsize = font_size + 4

    if axis_label_fontsize is None:
        axis_label_fontsize = font_size + 1

    if tick_label_fontsize is None:
        tick_label_fontsize = font_size - 1

    if legend_fontsize is None:
        legend_fontsize = font_size

    # =====================================================
    # Helpers
    # =====================================================
    def _find_matching_column(columns, target):
        target_clean = str(target).strip().lower()

        for col in columns:
            if str(col).strip().lower() == target_clean:
                return col

        return None

    def _format_rps_labels(
            region,
            x_order,
        ):
            labels = []
        
            for x in x_order:
                if region in MULTI_STATE_REGIONS:
                    if x == -10:
                        labels.append(
                            "RPS −10 pp"
                        )
        
                    elif x == 0:
                        labels.append(
                            "RPS baseline"
                        )
        
                    elif x == 10:
                        labels.append(
                            "RPS +10 pp"
                        )
        
                    else:
                        labels.append(
                            f"RPS {int(x):+d} pp"
                        )
        
                else:
                    labels.append(
                        f"{int(x)}%"
                    )
    
            return labels

    def _add_second_y_axis(ax, panel_df):
        """
        Add right y-axis showing absolute cost reduction.

        The plotted data on the left axis is percent reduction.
        The right axis converts that percent scale to million dollars
        using the panel-specific median ratio:

            million dollars per percentage point

        This keeps the visual line positions unchanged while adding
        the absolute-value interpretation on the right.
        """

        if not show_second_y_axis:
            return None

        valid = panel_df[
            ["cost_reduction_percent", "cost_reduction_mil"]
        ].copy()

        valid["cost_reduction_percent"] = pd.to_numeric(
            valid["cost_reduction_percent"],
            errors="coerce",
        )

        valid["cost_reduction_mil"] = pd.to_numeric(
            valid["cost_reduction_mil"],
            errors="coerce",
        )

        valid = valid.replace([np.inf, -np.inf], np.nan).dropna()

        valid = valid[
            valid["cost_reduction_percent"].abs() > 1e-9
        ].copy()

        if valid.empty:
            return None

        valid["mil_per_percent"] = (
            valid["cost_reduction_mil"]
            / valid["cost_reduction_percent"]
        )

        valid = valid.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["mil_per_percent"]
        )

        if valid.empty:
            return None

        mil_per_percent = valid["mil_per_percent"].median()

        if not np.isfinite(mil_per_percent) or abs(mil_per_percent) < 1e-12:
            return None

        ymin, ymax = ax.get_ylim()

        ax2 = ax.twinx()

        ax2.set_ylim(
            ymin * mil_per_percent,
            ymax * mil_per_percent,
        )

        ax2.set_ylabel(
            second_y_axis_label,
            fontsize=axis_label_fontsize,
            color="black",
        )

        ax2.tick_params(
            axis="y",
            colors="black",
            labelsize=tick_label_fontsize,
        )

        for spine in ax2.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

        ax2.grid(False)

        return ax2

    # =====================================================
    # Normalize inputs
    # =====================================================
    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): list(v)
            for k, v in rps_order_by_region.items()
        }

    if panel_title_by_region is None:
        panel_title_by_region = {}
    else:
        panel_title_by_region = {
            normalize_region(k): v
            for k, v in panel_title_by_region.items()
        }

    # =====================================================
    # Load data
    # =====================================================
    df_list = []

    for region in region_codes:
        for batt in batt_order:
            temp = compute_total_cost_delta_for_region_battery_adoption_range(
                output_root=output_root,
                region=region,
                year=year,
                adoption_levels=adoption_levels,
                batt_capex=batt,
                baseline_scenario_label=baseline_map[region],
                program_scenario_label=program_map[region],
                rps_order=rps_order_by_region.get(region, None),
                selected_participation=selected_participation,
            )

            df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError("No data found for plotting.")

    df_delta_all = pd.concat(df_list, ignore_index=True)

    # =====================================================
    # Build reduction metrics
    # =====================================================
    required_cols = [
        "delta_cost_mil",
        "delta_cost_percent",
    ]

    missing_cols = [
        col for col in required_cols
        if col not in df_delta_all.columns
    ]

    if missing_cols:
        raise ValueError(
            "The dataframe returned by "
            "compute_total_cost_delta_for_region_battery_adoption_range "
            "is missing these columns:\n"
            + "\n".join(missing_cols)
        )

    df_delta_all["delta_cost_mil"] = pd.to_numeric(
        df_delta_all["delta_cost_mil"],
        errors="coerce",
    )

    df_delta_all["delta_cost_percent"] = pd.to_numeric(
        df_delta_all["delta_cost_percent"],
        errors="coerce",
    )

    # Existing delta is assumed to be:
    #     program cost - baseline cost
    #
    # Therefore, cost reduction is the negative of delta.
    # Positive means the program reduces total system cost.
    df_delta_all["cost_reduction_mil"] = -df_delta_all["delta_cost_mil"]
    df_delta_all["cost_reduction_percent"] = -df_delta_all["delta_cost_percent"]

    # Left y-axis is always percentage reduction.
    metric_col = "cost_reduction_percent"
    y_label = "Total system cost reduction (%)"

    # =====================================================
    # Plot settings
    # =====================================================
    series_order = [
        ("V1G", "25%"),
        ("V1G", "50%"),
        ("V2G", "25%"),
        ("V2G", "50%"),
    ]

    linestyle_map = {
        "25%": "-",
        "50%": "-",
    }

    # =====================================================
    # Figure layout: region rows, battery columns
    # =====================================================
    n_rows = len(region_codes)
    n_cols = len(batt_order)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.2 * n_cols, 3.8 * n_rows),
        sharey=False,
        facecolor="white",
        squeeze=False,
    )

    secondary_axes = []

    # =====================================================
    # Draw panels
    # =====================================================
    for row_idx, region in enumerate(region_codes):
        for col_idx, batt in enumerate(batt_order):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            sub = df_delta_all[
                (df_delta_all["region"] == region)
                & (df_delta_all["batt_capex_num"] == batt)
            ].copy()

            if sub.empty:
                ax.axis("off")
                continue

            if region in rps_order_by_region:
                x_order = list(rps_order_by_region[region])
            else:
                x_order = (
                    sub["rps_plot_value"]
                    .dropna()
                    .sort_values()
                    .unique()
                    .tolist()
                )

            x_labels = _format_rps_labels(region, x_order)

            ax.axhline(
                0,
                color="black",
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )

            for group, participation in series_order:
                plot_data = sub[
                    (sub["group"] == group)
                    & (sub["participation"] == participation)
                ].copy()

                if plot_data.empty:
                    continue

                pivot = plot_data.pivot_table(
                    index="rps_plot_value",
                    columns="adoption",
                    values=metric_col,
                    aggfunc="first",
                )

                pivot = pivot.reindex(x_order)

                main_col = _find_matching_column(
                    pivot.columns,
                    main_adoption_level,
                )

                if main_col is None:
                    continue

                # -----------------------------------------
                # Slow-to-fast adoption band
                # Left axis = percent reduction
                # -----------------------------------------
                if show_adoption_band:
                    low_requested, high_requested = band_adoption_levels

                    low_col = _find_matching_column(
                        pivot.columns,
                        low_requested,
                    )

                    high_col = _find_matching_column(
                        pivot.columns,
                        high_requested,
                    )

                    if low_col is not None and high_col is not None:
                        band_low = pivot[[low_col, high_col]].min(axis=1)
                        band_high = pivot[[low_col, high_col]].max(axis=1)

                        ax.fill_between(
                            x_order,
                            band_low.values,
                            band_high.values,
                            color=GROUP_COLORS[group],
                            alpha=0.14,
                            linewidth=0,
                            zorder=2,
                        )

                        ax.plot(
                            x_order,
                            band_low.values,
                            color=GROUP_COLORS[group],
                            linewidth=0.8,
                            alpha=0.35,
                            zorder=2,
                        )

                        ax.plot(
                            x_order,
                            band_high.values,
                            color=GROUP_COLORS[group],
                            linewidth=0.8,
                            alpha=0.35,
                            zorder=2,
                        )

                # -----------------------------------------
                # Main adoption line
                # Left axis = percent reduction
                # -----------------------------------------
                ax.plot(
                    x_order,
                    pivot[main_col].values,
                    color=GROUP_COLORS[group],
                    linestyle=linestyle_map[participation],
                    marker=MARKER_MAP[participation],
                    linewidth=2.2,
                    markersize=6.5,
                    zorder=3,
                    label=f"{group}, {participation}",
                )

            panel_title = panel_title_by_region.get(
                region,
                REGION_DISPLAY_NAMES.get(region, region),
            )

            ax.set_title(
                f"{panel_title}, battery capex = ${batt}/kWh",
                fontsize=panel_title_fontsize,
                fontweight="bold",
                color="black",
                pad=8,
            )

            ax.set_xlabel(
                "RPS target reached",
                fontsize=axis_label_fontsize,
                color="black",
            )

            ax.set_ylabel(
                y_label,
                fontsize=axis_label_fontsize,
                color="black",
            )

            ax.set_xticks(x_order)
            ax.set_xticklabels(
                x_labels,
                fontsize=tick_label_fontsize,
                color="black",
            )

            ax.tick_params(
                axis="both",
                colors="black",
                labelsize=tick_label_fontsize,
            )

            ax.grid(
                True,
                which="both",
                linestyle="--",
                alpha=0.35,
                color="gray",
            )

            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(1.0)

            # Add right y-axis after the left axis is fully scaled.
            ax2 = _add_second_y_axis(ax, sub)
            if ax2 is not None:
                secondary_axes.append(ax2)

    # =====================================================
    # Legend
    # =====================================================
    legend_handles = [
        Line2D(
            [0], [0],
            color=GROUP_COLORS["V1G"],
            marker=MARKER_MAP["25%"],
            lw=2.2,
            markersize=6.5,
            label="V1G, 25%",
        ),
        Line2D(
            [0], [0],
            color=GROUP_COLORS["V1G"],
            marker=MARKER_MAP["50%"],
            lw=2.2,
            markersize=6.5,
            label="V1G, 50%",
        ),
        Line2D(
            [0], [0],
            color=GROUP_COLORS["V2G"],
            marker=MARKER_MAP["25%"],
            lw=2.2,
            markersize=6.5,
            label="V2G, 25%",
        ),
        Line2D(
            [0], [0],
            color=GROUP_COLORS["V2G"],
            marker=MARKER_MAP["50%"],
            lw=2.2,
            markersize=6.5,
            label="V2G, 50%",
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            lw=1.1,
            label="No cost reduction",
        ),
    ]

    if show_adoption_band:
        legend_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="none",
                alpha=0.14,
                label=(
                    f"{band_adoption_levels[0]} to "
                    f"{band_adoption_levels[1]} adoption range"
                ),
            )
        )

    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=min(6, len(legend_handles)),
        frameon=True,
        fontsize=legend_fontsize,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)

    for txt in legend.get_texts():
        txt.set_color("black")
        txt.set_fontsize(legend_fontsize)

    # =====================================================
    # Figure title
    # =====================================================
    if figure_title is None:
        figure_title = (
            "Total system cost reduction vs RPS target reached\n"
            f"Left axis = percent reduction, right axis = million-dollar reduction; "
            f"main line = {main_adoption_level} adoption, "
            f"shaded band = {band_adoption_levels[0]}–{band_adoption_levels[1]} adoption"
        )

    fig.suptitle(
        figure_title,
        fontsize=figure_title_fontsize,
        color="black",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.965])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return df_delta_all, fig, axes

def plot_total_cost_delta_vs_rps_all_states(
    states=("CA", "NY", "TX"),
    **kwargs,
):
    """
    Backward-compatible wrapper.
    Just routes the old state-based function name to the new region-based one.
    """
    return plot_total_cost_delta_vs_rps_regions(
        regions=states,
        **kwargs,
    )

def find_solution_json_file(folder):
    """
    Find the correct solution JSON inside one scenario folder.

    This avoids accidentally reading an old/copy/warm-start/checkpoint file.
    """

    folder = Path(folder)

    matches = sorted(folder.rglob("*_solution.json"))

    if not matches:
        return None

    bad_words = [
        "old",
        "backup",
        "copy",
        "warm",
        "checkpoint",
        "initial",
        "debug",
    ]

    clean = []

    for path in matches:
        name = path.name.lower()
        full = str(path).lower()

        if any(word in name or word in full for word in bad_words):
            continue

        clean.append(path)

    if not clean:
        clean = matches

    clean = sorted(
        clean,
        key=lambda p: (
            0 if p.parent == folder else 1,
            len(str(p)),
            str(p),
        ),
    )

    return clean[0]

def parse_scenario_metadata(folder_name, region=None):
    """
    Parse scenario metadata from one scenario folder.

    Supports:
        rps50, rps60, rps70
        rps-10, rps0, rps10
        rps_minus10, rps_base, rps_plus10
    """

    meta = {}

    m = re.search(r"^s0*(\d+)_", folder_name)
    meta["scenario_id"] = int(m.group(1)) if m else None

    m_v1g = re.search(r"v1g(\d+)", folder_name, re.IGNORECASE)
    m_v2g = re.search(r"v2g(\d+)", folder_name, re.IGNORECASE)

    v1g_val = int(m_v1g.group(1)) if m_v1g else 0
    v2g_val = int(m_v2g.group(1)) if m_v2g else 0

    meta["v1g_share"] = v1g_val / 100.0
    meta["v2g_share"] = v2g_val / 100.0

    # Remove combined case from the new analysis.
    if v1g_val > 0 and v2g_val > 0:
        meta["group"] = "V1G+V2G"
    elif v1g_val > 0:
        meta["group"] = "V1G"
    elif v2g_val > 0:
        meta["group"] = "V2G"
    else:
        meta["group"] = "Base only"

    if meta["group"] == "Base only":
        meta["participation"] = "0%"
    elif meta["group"] == "V1G":
        meta["participation"] = f"{v1g_val}%"
    elif meta["group"] == "V2G":
        meta["participation"] = f"{v2g_val}%"
    else:
        meta["participation"] = f"{v1g_val + v2g_val}%"

    # Use the existing robust RPS parser if you already added it.
    rps_info = parse_rps_info(folder_name, region=region)

    meta["rps"] = rps_info["rps_plot_value"]
    meta["rps_display_label"] = rps_info["rps_display_label"]
    meta["rps_case"] = rps_info["rps_case"]

    m = re.search(r"bcapex(\d+)", folder_name, re.IGNORECASE)
    meta["batt_capex"] = int(m.group(1)) if m else None

    return meta

def get_scenario_results_dir(
    output_root,
    region,
    year,
    adoption,
    scenario_label,
):
    """
    Build the new region-level GOOD output path.

    Example:
        Output/PJM/scenario_results_2030_mid_PJM_flex
    """

    region_code = normalize_region(region)

    return (
        Path(output_root)
        / region_code
        / f"scenario_results_{year}_{adoption}_{region_code}_{scenario_label}"
    )

def collect_generation_delta_for_region_group(
    output_root,
    region,
    year,
    adoption,
    graph_json_path,
    batt_capex_target=150,
    baseline_scenario_label="flex",
    program_scenario_label="flex",
    selected_group="V2G",
    selected_participation=("30%", "50%"),
    time_step_h=1.0,
    rps_order=None,
    strict_rps_check=True,
):
    """
    Collect generation delta for one GOOD model region and one selected program group.

    Strict version:
        - filters requested RPS before reading solution files
        - records the exact solution_json path
        - checks missing RPS cases
        - checks duplicated scenario folders
    """

    from good.graph import graph_from_json

    region_code = normalize_region(region)
    graph = graph_from_json(str(graph_json_path))

    rps_set = set(rps_order) if rps_order is not None else None

    baseline_dir = get_scenario_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=baseline_scenario_label,
    )

    program_dir = get_scenario_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=program_scenario_label,
    )

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline folder: {baseline_dir}")

    if not program_dir.exists():
        raise FileNotFoundError(f"Missing program folder: {program_dir}")

    all_rows = []
    missing_solution_files = []

    # =====================================================
    # Helper to read one folder
    # =====================================================
    def read_generation_folder(folder, meta, scenario_label, case_type):
        solution_json = find_solution_json_file(folder)

        if solution_json is None:
            missing_solution_files.append(str(folder))
            return None

        gen = generation_summary_from_solution_json(
            solution_json_path=solution_json,
            graph=graph,
            time_step_h=time_step_h,
        )

        gen["region"] = region_code
        gen["scenario_id"] = meta["scenario_id"]
        gen["group"] = meta["group"]
        gen["participation"] = meta["participation"]
        gen["rps"] = meta["rps"]
        gen["rps_display_label"] = meta["rps_display_label"]
        gen["batt_capex"] = meta["batt_capex"]
        gen["scenario_label"] = scenario_label.lower()
        gen["case_type"] = case_type
        gen["scenario_folder"] = str(folder)
        gen["solution_json_path"] = str(solution_json)

        return gen

    # =====================================================
    # Baseline rows
    # =====================================================
    for folder in sorted(baseline_dir.glob("s*")):
        if not folder.is_dir():
            continue

        meta = parse_scenario_metadata(folder.name, region=region_code)

        if meta["batt_capex"] != batt_capex_target:
            continue

        if meta["group"] != "Base only":
            continue

        if rps_set is not None and meta["rps"] not in rps_set:
            continue

        gen = read_generation_folder(
            folder=folder,
            meta=meta,
            scenario_label=baseline_scenario_label,
            case_type="baseline",
        )

        if gen is not None:
            all_rows.append(gen)

    # =====================================================
    # Program rows
    # =====================================================
    for folder in sorted(program_dir.glob("s*")):
        if not folder.is_dir():
            continue

        meta = parse_scenario_metadata(folder.name, region=region_code)

        if meta["batt_capex"] != batt_capex_target:
            continue

        if meta["group"] != selected_group:
            continue

        if meta["participation"] not in selected_participation:
            continue

        if rps_set is not None and meta["rps"] not in rps_set:
            continue

        gen = read_generation_folder(
            folder=folder,
            meta=meta,
            scenario_label=program_scenario_label,
            case_type="program",
        )

        if gen is not None:
            all_rows.append(gen)

    if not all_rows:
        raise ValueError(
            f"No valid generation data found.\n"
            f"region = {region_code}\n"
            f"baseline_dir = {baseline_dir}\n"
            f"program_dir = {program_dir}\n"
            f"group = {selected_group}\n"
            f"battery capex = {batt_capex_target}\n"
            f"participation = {selected_participation}\n"
            f"requested RPS = {rps_order}\n"
            f"missing solution folders examples = {missing_solution_files[:5]}"
        )

    df = pd.concat(all_rows, ignore_index=True)
    df = resolve_duplicate_generation_cases(
        df,
        strategy="min_scenario_id",
        verbose=True,
    )
    # =====================================================
    # Check duplicate folders
    # =====================================================
    folder_keys = (
        df[
            [
                "case_type",
                "region",
                "scenario_label",
                "group",
                "participation",
                "rps",
                "batt_capex",
                "scenario_folder",
                "solution_json_path",
            ]
        ]
        .drop_duplicates()
    )

    duplicate_key_cols = [
        "case_type",
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
    ]

    duplicate_summary = (
        folder_keys
        .groupby(duplicate_key_cols, as_index=False)
        .agg(
            n_folders=("scenario_folder", "nunique"),
            folders=("scenario_folder", lambda x: "\n".join(map(str, x))),
        )
    )

    duplicated_cases = duplicate_summary[duplicate_summary["n_folders"] > 1].copy()

    if not duplicated_cases.empty:
        raise ValueError(
            "Duplicate generation scenario folders were found.\n"
            "This means the same region/RPS/group/participation is being read more than once.\n\n"
            + duplicated_cases.to_string(index=False)
        )

    # =====================================================
    # Check requested RPS coverage
    # =====================================================
    if rps_order is not None and strict_rps_check:
        requested_rps = list(rps_order)

        base_available = sorted(
            df.loc[
                df["case_type"] == "baseline",
                "rps",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        missing_base = [
            rps for rps in requested_rps
            if rps not in base_available
        ]

        problems = []

        if missing_base:
            problems.append(
                f"{region_code} Base only is missing requested RPS {missing_base}. "
                f"Available baseline RPS = {base_available}"
            )

        for part in selected_participation:
            program_available = sorted(
                df.loc[
                    (df["case_type"] == "program")
                    & (df["group"] == selected_group)
                    & (df["participation"] == part),
                    "rps",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            missing_program = [
                rps for rps in requested_rps
                if rps not in program_available
            ]

            if missing_program:
                problems.append(
                    f"{region_code} {selected_group}, {part} is missing requested RPS "
                    f"{missing_program}. Available program RPS = {program_available}"
                )

        if problems:
            raise ValueError(
                "Requested RPS cases are missing before plotting:\n"
                + "\n".join(problems)
            )

    # =====================================================
    # Build Base table
    # =====================================================
    base_raw = df[df["case_type"] == "baseline"].copy()

    base = (
        base_raw
        .groupby(
            [
                "region",
                "rps",
                "rps_display_label",
                "batt_capex",
                "resource",
            ],
            as_index=False,
            dropna=False,
        )["energy_gwh"]
        .sum()
        .rename(columns={"energy_gwh": "base_energy_gwh"})
    )

    # =====================================================
    # Build Program table
    # =====================================================
    comp_raw = df[
        (df["case_type"] == "program")
        & (df["group"] == selected_group)
        & (df["participation"].isin(selected_participation))
    ].copy()

    comp_energy = (
        comp_raw
        .groupby(
            [
                "region",
                "scenario_id",
                "group",
                "participation",
                "rps",
                "rps_display_label",
                "batt_capex",
                "scenario_label",
                "scenario_folder",
                "solution_json_path",
                "resource",
            ],
            as_index=False,
            dropna=False,
        )["energy_gwh"]
        .sum()
    )

    # =====================================================
    # Complete resource grid
    # =====================================================
    program_keys = comp_raw[
        [
            "region",
            "scenario_id",
            "group",
            "participation",
            "rps",
            "rps_display_label",
            "batt_capex",
            "scenario_label",
            "scenario_folder",
            "solution_json_path",
        ]
    ].drop_duplicates()

    base_resources = base[
        [
            "region",
            "rps",
            "rps_display_label",
            "batt_capex",
            "resource",
        ]
    ].drop_duplicates()

    program_resources = comp_energy[
        [
            "region",
            "rps",
            "rps_display_label",
            "batt_capex",
            "resource",
        ]
    ].drop_duplicates()

    all_resources = (
        pd.concat(
            [base_resources, program_resources],
            ignore_index=True,
        )
        .drop_duplicates()
    )

    full_grid = program_keys.merge(
        all_resources,
        on=[
            "region",
            "rps",
            "rps_display_label",
            "batt_capex",
        ],
        how="left",
    )

    # =====================================================
    # Merge Program and Base
    # =====================================================
    merged = full_grid.merge(
        comp_energy,
        on=[
            "region",
            "scenario_id",
            "group",
            "participation",
            "rps",
            "rps_display_label",
            "batt_capex",
            "scenario_label",
            "scenario_folder",
            "solution_json_path",
            "resource",
        ],
        how="left",
    )

    merged = merged.merge(
        base,
        on=[
            "region",
            "rps",
            "rps_display_label",
            "batt_capex",
            "resource",
        ],
        how="left",
    )

    merged["energy_gwh"] = merged["energy_gwh"].fillna(0.0)
    merged["base_energy_gwh"] = merged["base_energy_gwh"].fillna(0.0)

    merged["delta_gwh"] = (
        merged["energy_gwh"] - merged["base_energy_gwh"]
    )

    merged["program_missing_resource"] = merged["energy_gwh"].abs() < 1e-12
    merged["base_missing_resource"] = merged["base_energy_gwh"].abs() < 1e-12

    merged["baseline_scenario_label"] = baseline_scenario_label.lower()
    merged["program_scenario_label"] = program_scenario_label.lower()

    return merged.reset_index(drop=True)

def plot_generation_delta_regions_by_rps_2x3(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption="mid",
    batt_capex_target=150,
    selected_group="V2G",
    selected_participation=("30%", "50%"),
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    graph_json_path_by_region=None,
    rps_order_by_region=None,
    panel_title_by_region=None,
    label_threshold_by_region=None,
    figure_title=None,
    save_path=None,
    max_labels_per_bar_by_region=None,
    strict_rps_check=True,
    print_rps_diagnostics=True,
    font_size=14,

    # New y-axis controls
    same_y_scale=True,
    symmetric_y_axis=True,
    y_axis_padding_fraction=0.10,
):
    """
    Plot generation mix change relative to Base.

    Layout:
        2 rows x 3 columns

    For five regions:
        CAISO | NYISO | ERCOT
        FRCC  | PJM   | legend

    font_size controls all plot fonts.

    same_y_scale:
        If True, all region panels use the same y-axis limits.

    symmetric_y_axis:
        If True, y-axis is symmetric around zero.
        This is recommended for positive/negative generation changes.

    y_axis_padding_fraction:
        Padding added to the common y-axis limits.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from pathlib import Path

    # =====================================================
    # Font sizes
    # =====================================================
    bar_label_fontsize = max(6, font_size - 1.5)
    x_tick_fontsize = max(6, font_size - 2)
    rps_label_fontsize = max(6, font_size - 2)
    scenario_label_fontsize = max(6, font_size - 1)
    axis_label_fontsize = max(6, font_size)
    axis_tick_fontsize = max(6, font_size)
    panel_title_fontsize = max(8, font_size + 4)
    legend_fontsize = max(6, font_size - 2)
    legend_title_fontsize = max(6, font_size)
    figure_title_fontsize = max(8, font_size + 5)

    # =====================================================
    # Normalize inputs
    # =====================================================
    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    if graph_json_path_by_region is None:
        raise ValueError("graph_json_path_by_region cannot be None.")

    graph_path_map = {
        normalize_region(k): v
        for k, v in graph_json_path_by_region.items()
    }

    if rps_order_by_region is None:
        raise ValueError("rps_order_by_region cannot be None.")

    rps_order_by_region = {
        normalize_region(k): list(v)
        for k, v in rps_order_by_region.items()
    }

    missing_rps_keys = [
        region
        for region in region_codes
        if region not in rps_order_by_region
    ]

    if missing_rps_keys:
        raise ValueError(
            "Missing rps_order_by_region entries for these regions:\n"
            + "\n".join(missing_rps_keys)
        )

    if panel_title_by_region is None:
        panel_title_by_region = {}

    panel_title_by_region = {
        normalize_region(k): v
        for k, v in panel_title_by_region.items()
    }

    if label_threshold_by_region is None:
        label_threshold_by_region = {}

    label_threshold_by_region = {
        normalize_region(k): v
        for k, v in label_threshold_by_region.items()
    }

    resource_order = [
        "Solar",
        "Wind",
        "Nuclear",
        "Coal",
        "CCNG",
        "Gas\nTurbine",
        "Hydro",
        "Geothermal",
        "Biomass",
        "Waste",
        "Import",
        "Steam",
        "Non-fossil",
        "Other",
        "Battery Discharge",
        "V2G Discharge",
    ]

    colors = {
        "Solar": "#f4b400",
        "Wind": "#4a90e2",
        "Nuclear": "#ff1f1f",
        "Coal": "#000000",
        "CCNG": "#8c8c8c",
        "Gas\nTurbine": "#5f5f5f",
        "Hydro": "#4fc3f7",
        "Geothermal": "#8e44ad",
        "Biomass": "#2e7d32",
        "Waste": "#66bb6a",
        "Import": "#795548",
        "Steam": "#b71c1c",
        "Non-fossil": "#00acc1",
        "Other": "#9e9e9e",
        "Battery Discharge": "#6d4c41",
        "V2G Discharge": "#c2185b",
    }

    # =====================================================
    # Load data
    # =====================================================
    df_list = []

    for region in region_codes:
        if region not in graph_path_map:
            raise ValueError(f"Missing graph path for {region}")

        temp = collect_generation_delta_for_region_group(
            output_root=output_root,
            region=region,
            year=year,
            adoption=adoption,
            graph_json_path=graph_path_map[region],
            batt_capex_target=batt_capex_target,
            baseline_scenario_label=baseline_map[region],
            program_scenario_label=program_map[region],
            selected_group=selected_group,
            selected_participation=selected_participation,
            rps_order=rps_order_by_region[region],
            strict_rps_check=strict_rps_check,
        )

        df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError("No generation data was loaded.")

    df = pd.concat(df_list, ignore_index=True)

    # Keep only region-specific requested RPS values
    df = pd.concat(
        [
            df[
                (df["region"] == region)
                & (df["rps"].isin(rps_order_by_region[region]))
            ]
            for region in region_codes
        ],
        ignore_index=True,
    )

    if df.empty:
        raise ValueError(
            "Generation dataframe is empty after applying region-specific RPS filters."
        )

    # =====================================================
    # Check requested RPS coverage
    # =====================================================
    rps_diagnostic_df = validate_generation_rps_coverage(
        df=df,
        region_codes=region_codes,
        rps_order_by_region=rps_order_by_region,
        selected_group=selected_group,
        selected_participation=selected_participation,
        strict=strict_rps_check,
    )

    if print_rps_diagnostics:
        print("\nGeneration RPS diagnostic:")
        print(rps_diagnostic_df.to_string(index=False))

    present_resources = [
        r for r in resource_order
        if r in df["resource"].unique()
    ]

    if not present_resources:
        raise ValueError("No generation resources were found in the dataframe.")

    # =====================================================
    # Figure: 2 x 3
    # =====================================================
    fig, axes_grid = plt.subplots(
        2,
        3,
        figsize=(18, 12),
        sharey=False,
        facecolor="white",
        squeeze=False,
    )

    axes = axes_grid.flatten()

    for ax in axes:
        ax.set_facecolor("white")

    bar_width = 1.15
    participation_gap = 0.28
    rps_gap = 0.85

    resources_in_legend = set()

    # Store label information and add labels after y-axis scaling
    panel_label_records = []

    # =====================================================
    # Plot panels
    # =====================================================
    for plot_idx, region in enumerate(region_codes):
        ax = axes[plot_idx]

        rps_values = list(rps_order_by_region[region])
        sub_region = df[df["region"] == region].copy()

        x_map = {}
        xticks = []
        xticklabels = []
        rps_centers = []
        separator_positions = []

        current_x = 0.0

        for rps_idx, rps in enumerate(rps_values):
            local_positions = []

            for p in selected_participation:
                x_map[(rps, p)] = current_x
                xticks.append(current_x)
                xticklabels.append(p)
                local_positions.append(current_x)

                current_x += bar_width + participation_gap

            rps_centers.append(np.mean(local_positions))

            if rps_idx < len(rps_values) - 1:
                sep = current_x + rps_gap / 2 - participation_gap
                separator_positions.append(sep)

            current_x += rps_gap

        ordered_keys = [
            (rps, p)
            for rps in rps_values
            for p in selected_participation
        ]

        pos_bottom = {key: 0.0 for key in ordered_keys}
        neg_bottom = {key: 0.0 for key in ordered_keys}

        max_labels_this_region = get_max_labels_for_region(
            max_labels_per_bar_by_region=max_labels_per_bar_by_region,
            region=region,
            default_value=4,
        )

        for key in ordered_keys:
            rps, participation = key
            x = x_map[key]

            segment_records = []

            for resource in present_resources:
                temp = sub_region[
                    (sub_region["rps"] == rps)
                    & (sub_region["participation"] == participation)
                    & (sub_region["resource"] == resource)
                ]

                val = temp["delta_gwh"].sum() if not temp.empty else 0.0

                if abs(val) <= 1e-10:
                    continue

                resources_in_legend.add(resource)

                facecolor = colors.get(resource, "#9e9e9e")

                if val >= 0:
                    bottom = pos_bottom[key]
                else:
                    bottom = neg_bottom[key]

                ax.bar(
                    x,
                    val,
                    width=bar_width,
                    bottom=bottom,
                    color=facecolor,
                    edgecolor="none",
                    zorder=3,
                )

                segment_records.append({
                    "resource": resource,
                    "value": val,
                    "bottom": bottom,
                    "x": x,
                    "facecolor": facecolor,
                })

                if val >= 0:
                    pos_bottom[key] += val
                else:
                    neg_bottom[key] += val

            # -------------------------------------------------
            # Labels inside bars
            # -------------------------------------------------
            label_candidates = []

            for seg in segment_records:
                resource = seg["resource"]
                val = seg["value"]

                resource_threshold = get_resource_label_threshold(
                    label_threshold_by_region=label_threshold_by_region,
                    region=region,
                    resource=resource,
                    default_threshold=20,
                )

                if abs(val) >= resource_threshold:
                    label_candidates.append(seg)

            label_candidates = sorted(
                label_candidates,
                key=lambda d: abs(d["value"]),
                reverse=True,
            )

            if max_labels_this_region is not None:
                label_candidates = label_candidates[:max_labels_this_region]

            for seg in label_candidates:
                resource = seg["resource"]
                val = seg["value"]
                bottom = seg["bottom"]
                x = seg["x"]
                facecolor = seg["facecolor"]

                ax.text(
                    x,
                    bottom + val / 2,
                    resource,
                    ha="center",
                    va="center",
                    fontsize=bar_label_fontsize,
                    color=get_contrast_text_color(facecolor),
                    fontweight=(
                        "bold"
                        if resource in ["Coal", "Gas\nTurbine", "Steam"]
                        else "normal"
                    ),
                    clip_on=True,
                    zorder=5,
                )

        ax.axhline(
            0,
            color="black",
            linewidth=1.1,
        )

        for xpos in separator_positions:
            ax.axvline(
                xpos,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.75,
            )

        ax.set_xticks(xticks)
        ax.set_xticklabels(
            xticklabels,
            fontsize=x_tick_fontsize,
            fontweight="bold",
            color="black",
        )

        panel_title = panel_title_by_region.get(
            region,
            REGION_DISPLAY_NAMES.get(region, region),
        )

        ax.set_title(
            panel_title,
            fontsize=panel_title_fontsize,
            fontweight="bold",
            color="black",
            pad=24,
        )

        ax.set_ylabel(
            "Change in generation (GWh)",
            fontsize=axis_label_fontsize,
            color="black",
        )

        ax.grid(
            axis="y",
            color="gray",
            alpha=0.25,
            linestyle="--",
        )

        ax.tick_params(
            axis="x",
            colors="black",
            labelsize=x_tick_fontsize,
        )

        ax.tick_params(
            axis="y",
            colors="black",
            labelsize=axis_tick_fontsize,
        )

        for spine in ax.spines.values():
            spine.set_color("black")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Store label info. Add these labels after common y-scale is fixed.
        panel_label_records.append({
            "ax": ax,
            "region": region,
            "xticks": xticks,
            "rps_values": rps_values,
            "rps_centers": rps_centers,
        })

    # =====================================================
    # Force same y-axis scale across all region panels
    # =====================================================
    plot_axes = [
        axes[i]
        for i in range(len(region_codes))
        if axes[i] is not None and axes[i].has_data()
    ]

    if same_y_scale and plot_axes:
        ymins = []
        ymaxs = []

        for ax in plot_axes:
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)

        common_ymin = min(ymins)
        common_ymax = max(ymaxs)

        if symmetric_y_axis:
            abs_limit = max(
                abs(common_ymin),
                abs(common_ymax),
            )

            abs_limit = abs_limit * (1 + y_axis_padding_fraction)

            common_ymin = -abs_limit
            common_ymax = abs_limit

        else:
            y_range = common_ymax - common_ymin

            if y_range > 0:
                common_ymin = common_ymin - y_axis_padding_fraction * y_range
                common_ymax = common_ymax + y_axis_padding_fraction * y_range

        for ax in plot_axes:
            ax.set_ylim(common_ymin, common_ymax)

    # =====================================================
    # Add RPS labels and scenario labels after y-scale is fixed
    # =====================================================
    for record in panel_label_records:
        ax = record["ax"]
        region = record["region"]
        xticks = record["xticks"]
        rps_values = record["rps_values"]
        rps_centers = record["rps_centers"]

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax != ymin else 1.0

        scenario_label_y = ymin - 0.10 * y_range

        ax.text(
            np.mean(xticks),
            scenario_label_y,
            selected_group,
            ha="center",
            va="top",
            fontsize=scenario_label_fontsize,
            fontweight="bold",
            color="black",
            clip_on=False,
        )

        rps_label_y = ymax + 0.015 * y_range

        for rps, center in zip(rps_values, rps_centers):
            if region == "PJM":
                if rps == -10:
                    rps_label = "Base - 10"
                elif rps == 0:
                    rps_label = "Base"
                elif rps == 10:
                    rps_label = "Base + 10"
                else:
                    rps_label = f"Base {rps:+d}"
            else:
                rps_label = f"RPS {rps}%"

            ax.text(
                center,
                rps_label_y,
                rps_label,
                ha="center",
                va="bottom",
                fontsize=rps_label_fontsize,
                color="black",
                clip_on=False,
            )

    # =====================================================
    # Legend in unused sixth panel
    # =====================================================
    for extra_ax in axes[len(region_codes):]:
        extra_ax.axis("off")

    legend_ax = axes[-1]
    legend_ax.axis("off")

    legend_resources = [
        r for r in present_resources
        if r in resources_in_legend
    ]

    legend_elements = [
        Patch(
            facecolor=colors.get(resource, "#9e9e9e"),
            edgecolor="black",
            label=resource,
        )
        for resource in legend_resources
    ]

    legend = legend_ax.legend(
        handles=legend_elements,
        loc="center",
        ncol=2,
        frameon=True,
        fontsize=legend_fontsize,
        title="Generation resources",
    )

    legend.get_title().set_fontsize(legend_title_fontsize)
    legend.get_title().set_weight("bold")
    legend.get_title().set_color("black")

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    if figure_title is None:
        figure_title = (
            f"Generation mix change for {selected_group} relative to Base\n"
            f"Battery capex = ${batt_capex_target}/kWh, adoption = {adoption}"
        )

    fig.suptitle(
        figure_title,
        fontsize=figure_title_fontsize,
        y=0.98,
        color="black",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return df, fig, axes

def audit_generation_resources_from_solution(solution_json_path, graph_json_path):
    """
    Detailed audit for generation-resource mapping.

    Use this to check whether Import, Hydro, small_hydro, large_hydro,
    and pumped hydro are being mapped correctly.
    """

    import json
    import numpy as np
    import pandas as pd
    from good.graph import graph_from_json

    graph = graph_from_json(str(graph_json_path))

    with open(solution_json_path) as f:
        data = json.load(f)

    rows = []

    for node in data["nodes"]:
        region = node["id"]

        if region not in graph._node:
            continue

        for handle, a_sol in node.get("assets", {}).items():

            if handle not in graph._node[region]["assets"]:
                continue

            meta = graph._node[region]["assets"][handle]

            cls = str(meta.get("_class", ""))
            typ = str(meta.get("type", ""))
            fuel = meta.get("fuel", None)

            values = a_sol.get("production", a_sol.get("net", []))
            arr = np.array(values, dtype=float)

            if arr.size == 0:
                energy_gwh = 0.0
            else:
                energy_gwh = np.clip(arr, 0, None).sum() / 1e9

            mapped = map_fuel_for_plot_extended(
                fuel=fuel,
                typ=typ,
                cls=cls,
                handle=handle,
            )

            rows.append({
                "region": region,
                "handle": handle,
                "_class": cls,
                "type": typ,
                "fuel": fuel,
                "mapped_resource": mapped,
                "energy_gwh": energy_gwh,
            })

    df = pd.DataFrame(rows)

    return df.sort_values(
        ["mapped_resource", "region", "fuel", "type", "handle"]
    ).reset_index(drop=True)

def map_capacity_resource_for_plot(row):
    """
    Map capacity expansion rows to the resources used in the plot.

    Keeps only:
        Solar
        Wind
        Battery
    """

    useful_cols = {
        "fuel",
        "type",
        "resource",
        "technology",
        "asset",
        "handle",
        "name",
        "asset_name",
        "generator",
        "gen_type",
        "category",
    }

    text = " ".join(
        str(row.get(col, ""))
        for col in row.index
        if str(col).lower() in useful_cols
    ).lower()

    text = (
        text
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("\\", " ")
    )

    if "solar" in text or "pv" in text or "photovoltaic" in text:
        return "Solar"

    if "wind" in text:
        return "Wind"

    if (
        "battery" in text
        or "storage" in text
        or "bess" in text
        or "li ion" in text
        or "lithium" in text
    ):
        return "Battery"

    return "Other"


def _capacity_template():
    """
    Always return these three resources so missing resources become zero,
    not missing rows.
    """

    return pd.DataFrame({
        "resource": ["Solar", "Wind", "Battery"],
        "capex_added_MW": [0.0, 0.0, 0.0],
    })


def _normalized_col_name(col):
    return re.sub(r"[^a-z0-9]", "", str(col).lower())


def _find_capacity_added_column(df):
    """
    Find the column that stores added / installed capacity.

    This is intentionally broader than the old version because different GOOD
    output files may use different names.
    """

    exact_candidates = [
        "capex_added_MW",
        "capex_added_mw",
        "added_capacity_mw",
        "capacity_added_mw",
        "new_capacity_mw",
        "installed_capacity_mw",
        "build_capacity_mw",
        "built_capacity_mw",
        "expansion_mw",
        "capacity_expansion_mw",
        "new_build_mw",
        "added_mw",
    ]

    # Exact match first.
    for col in exact_candidates:
        if col in df.columns:
            return col

    # Normalized exact match.
    normalized_lookup = {
        _normalized_col_name(col): col
        for col in df.columns
    }

    for col in exact_candidates:
        key = _normalized_col_name(col)
        if key in normalized_lookup:
            return normalized_lookup[key]

    # Fuzzy match.
    fuzzy_candidates = []

    for col in df.columns:
        col_lower = str(col).lower()

        if any(skip in col_lower for skip in ["cost", "price", "objective", "opex", "total_cost"]):
            continue

        has_capacity_word = (
            "mw" in col_lower
            or "capacity" in col_lower
            or "cap" in col_lower
        )

        has_added_word = (
            "add" in col_lower
            or "new" in col_lower
            or "build" in col_lower
            or "built" in col_lower
            or "install" in col_lower
            or "expansion" in col_lower
            or "capex" in col_lower
        )

        if has_capacity_word and has_added_word:
            numeric_test = pd.to_numeric(df[col], errors="coerce")
            if numeric_test.notna().any():
                fuzzy_candidates.append(col)

    if fuzzy_candidates:
        return fuzzy_candidates[0]

    return None


def find_capex_summary_files(folder):
    """
    Find candidate capacity/capex CSV files inside one scenario folder.

    The old code only looked for *_capex_summary.csv.
    This version searches more broadly but still avoids total-cost and hourly files.
    """

    folder = Path(folder)

    patterns = [
        "*_capex_summary.csv",
        "*capex_summary*.csv",
        "*capacity*capex*.csv",
        "*capex*capacity*.csv",
        "*capacity_expansion*.csv",
        "*installed*capacity*.csv",
        "*added*capacity*.csv",
        "*new*capacity*.csv",
        "*asset*capacity*.csv",
        "*capacity*.csv",
        "*capex*.csv",
        "*asset*.csv",
    ]

    matches = []
    seen = set()

    for pattern in patterns:
        for path in sorted(folder.rglob(pattern)):
            if path in seen:
                continue

            seen.add(path)

            name = path.name.lower()

            # These are not capacity expansion summaries.
            if any(skip in name for skip in [
                "total_cost",
                "cost_components",
                "hourly",
                "timeseries",
                "time_series",
                "dispatch",
                "emission",
                "emissions",
                "load",
                "demand",
            ]):
                continue

            matches.append(path)

    matches = sorted(
        matches,
        key=lambda p: (
            0 if "capex_summary" in p.name.lower() else
            1 if "capacity_expansion" in p.name.lower() else
            2 if "installed" in p.name.lower() else
            3 if "capacity" in p.name.lower() else
            4 if "capex" in p.name.lower() else
            5,
            len(str(p)),
            str(p),
        ),
    )

    return matches


def find_capex_summary_file(folder):
    """
    Backward-compatible helper.
    """

    files = find_capex_summary_files(folder)

    if not files:
        return None

    return files[0]


def _extract_capacity_value_from_asset_solution(asset_solution):
    """
    Fallback for cases where no capex/capacity CSV exists.

    It tries to read added/installed capacity directly from the solution JSON.
    """

    exact_keys = [
        "capex_added_MW",
        "capex_added_mw",
        "added_capacity_mw",
        "capacity_added_mw",
        "new_capacity_mw",
        "installed_capacity_mw",
        "build_capacity_mw",
        "built_capacity_mw",
        "expansion_mw",
        "capacity_expansion_mw",
        "new_build_mw",
        "added_mw",
        "capacity",
        "capacity_mw",
    ]

    found = []

    def walk(obj, parent_key=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                k_str = str(k)
                k_norm = _normalized_col_name(k_str)

                # Exact or normalized exact match.
                if any(k_norm == _normalized_col_name(x) for x in exact_keys):
                    if isinstance(v, (int, float)):
                        found.append((k_str, float(v)))

                # Fuzzy match.
                k_lower = k_str.lower()
                if (
                    isinstance(v, (int, float))
                    and not any(skip in k_lower for skip in ["cost", "price", "objective", "opex"])
                    and (
                        "mw" in k_lower
                        or "capacity" in k_lower
                        or "cap" in k_lower
                    )
                    and (
                        "add" in k_lower
                        or "new" in k_lower
                        or "build" in k_lower
                        or "built" in k_lower
                        or "install" in k_lower
                        or "expansion" in k_lower
                        or k_lower == "capacity"
                    )
                ):
                    found.append((k_str, float(v)))

                walk(v, k_str)

        elif isinstance(obj, list):
            for item in obj:
                walk(item, parent_key)

    walk(asset_solution)

    if not found:
        return 0.0

    key, value = found[0]
    key_lower = key.lower()

    # GOOD usually stores internal power in W if the key is not explicitly MW.
    if "gw" in key_lower:
        return value * 1000.0

    if "kw" in key_lower:
        return value / 1000.0

    if "mw" in key_lower:
        return value

    # Fallback unit guess.
    # If it is huge, it is probably W.
    if abs(value) > 1e6:
        return value / 1e6

    return value


def capacity_summary_from_solution_json(folder):
    """
    Fallback reader:
    If no capacity/capex CSV exists, read installed/added capacity from solution JSON.
    """

    solution_json = find_solution_json_file(folder)

    if solution_json is None:
        raise ValueError(f"No solution JSON found in {folder}")

    import json

    with open(solution_json) as f:
        data = json.load(f)

    rows = []

    for node in data.get("nodes", []):
        for handle, asset_solution in node.get("assets", {}).items():
            if not isinstance(asset_solution, dict):
                continue

            row_for_mapping = pd.Series({
                "handle": handle,
                "fuel": asset_solution.get("fuel", ""),
                "type": asset_solution.get("type", ""),
                "technology": asset_solution.get("technology", ""),
                "resource": asset_solution.get("resource", ""),
                "name": asset_solution.get("name", ""),
            })

            resource = map_capacity_resource_for_plot(row_for_mapping)

            if resource not in ["Solar", "Wind", "Battery"]:
                continue

            added_mw = _extract_capacity_value_from_asset_solution(asset_solution)

            rows.append({
                "resource": resource,
                "capex_added_MW": added_mw,
            })

    if not rows:
        raise ValueError(
            f"No Solar/Wind/Battery capacity rows found in solution JSON: {solution_json}"
        )

    out = (
        pd.DataFrame(rows)
        .groupby("resource", as_index=False)["capex_added_MW"]
        .sum()
    )

    out = (
        _capacity_template()[["resource"]]
        .merge(out, on="resource", how="left")
    )

    out["capex_added_MW"] = out["capex_added_MW"].fillna(0.0)

    return out

def resolve_duplicate_capacity_cases(
    df,
    strategy="min_scenario_id",
    verbose=True,
):
    """
    Keep only one capacity folder per:
        case_type, region, scenario_label, group, participation, rps, batt_capex

    This prevents the same RPS/program case from being counted more than once.
    """

    key_cols = [
        "case_type",
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
    ]

    folder_cols = key_cols + [
        "scenario_id",
        "scenario_folder",
        "capex_file",
    ]

    folder_keys = df[folder_cols].drop_duplicates().copy()

    duplicate_summary = (
        folder_keys
        .groupby(key_cols, as_index=False)
        .agg(
            n_folders=("scenario_folder", "nunique"),
            folders=("scenario_folder", lambda x: "\n".join(map(str, x))),
        )
    )

    duplicated_cases = duplicate_summary[
        duplicate_summary["n_folders"] > 1
    ].copy()

    if duplicated_cases.empty:
        return df.reset_index(drop=True)

    if verbose:
        print("\nDuplicate capacity scenario folders were found.")
        print("Keeping only one folder per case.")
        print(duplicated_cases.to_string(index=False))

    if strategy == "min_scenario_id":
        keep_keys = (
            folder_keys
            .sort_values(key_cols + ["scenario_id", "scenario_folder"])
            .drop_duplicates(subset=key_cols, keep="first")
        )
    elif strategy == "max_scenario_id":
        keep_keys = (
            folder_keys
            .sort_values(key_cols + ["scenario_id", "scenario_folder"])
            .drop_duplicates(subset=key_cols, keep="last")
        )
    else:
        raise ValueError("Use strategy='min_scenario_id' or 'max_scenario_id'.")

    keep_folders = set(keep_keys["scenario_folder"].tolist())

    if verbose:
        print("\nKeeping these capacity folders:")
        print(
            keep_keys[
                key_cols + ["scenario_id", "scenario_folder"]
            ]
            .sort_values(key_cols)
            .to_string(index=False)
        )

    return df[df["scenario_folder"].isin(keep_folders)].reset_index(drop=True)

def capacity_summary_from_scenario_folder(folder):
    """
    Read capacity from one scenario folder.

    Priority:
        1. Candidate capex/capacity CSV files
        2. solution JSON fallback
    """

    folder = Path(folder)

    errors = []

    for csv_path in find_capex_summary_files(folder):
        try:
            cap = capex_summary_from_capex_summary_csv(csv_path)
            cap["capacity_source"] = str(csv_path)
            return cap
        except Exception as exc:
            errors.append(f"{csv_path}: {exc}")

    try:
        cap = capacity_summary_from_solution_json(folder)
        cap["capacity_source"] = str(find_solution_json_file(folder))
        return cap
    except Exception as exc:
        errors.append(f"solution_json fallback: {exc}")

    raise ValueError(
        f"Could not read capacity expansion from folder:\n{folder}\n\n"
        f"Tried files / fallback:\n" + "\n".join(errors[:10])
    )

def find_capacity_result_file(folder):
    """
    Find the file that contains installed / added capacity.

    New output folders do not have *_capex_summary.csv.
    They have *_solution.csv, so we use that.
    """

    folder = Path(folder)

    preferred_patterns = [
        "*_capex_summary.csv",
        "*_solution.csv",
        "*_summary.csv",
    ]

    for pattern in preferred_patterns:
        files = sorted(folder.glob(pattern))

        # Do not use total_cost as a capacity file.
        files = [
            p for p in files
            if "total_cost" not in p.name.lower()
        ]

        if files:
            return files[0]

    return None

def get_capacity_column_from_solution(df, file_path):
    """
    Find the column that stores added / installed capacity.
    """

    candidates = [
        "capex_added_MW",
        "capex_added_mw",
        "added_capacity_mw",
        "capacity_added_mw",
        "new_capacity_mw",
        "installed_capacity_mw",
        "capacity_mw",
        "capacity",
        "power_capacity_mw",
        "p_nom",
        "p_nom_opt",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    # Fuzzy fallback
    for col in df.columns:
        name = str(col).lower()

        if any(x in name for x in ["cost", "price", "objective", "emission", "energy"]):
            continue

        if (
            "capacity" in name
            or "cap" in name
            or "mw" in name
            or "p_nom" in name
        ):
            test = pd.to_numeric(df[col], errors="coerce")
            if test.notna().any():
                return col

    raise ValueError(
        f"No capacity column found in {file_path}.\n"
        f"Available columns:\n{list(df.columns)}"
    )

def map_capacity_resource_from_row(row):
    """
    Map solution/capex rows to Solar, Wind, Battery.
    """

    text = " ".join(
        str(row.get(col, ""))
        for col in row.index
    ).lower()

    text = (
        text
        .replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
    )

    if "solar" in text or " pv" in text or "photovoltaic" in text:
        return "Solar"

    if "wind" in text:
        return "Wind"

    if "battery" in text or "storage" in text or "bess" in text:
        return "Battery"

    return "Other"

def capacity_summary_from_result_csv(result_path):
    """
    Read installed/added capacity from either:
        *_capex_summary.csv
        *_solution.csv
        *_summary.csv

    Returns:
        resource, capex_added_MW
    """

    df = pd.read_csv(result_path)

    resources = ["Solar", "Wind", "Battery"]

    if df.empty:
        return pd.DataFrame({
            "resource": resources,
            "capex_added_MW": [0.0, 0.0, 0.0],
        })

    df = df.copy()

    capacity_col = get_capacity_column_from_solution(df, result_path)

    df["resource"] = df.apply(map_capacity_resource_from_row, axis=1)

    df = df[df["resource"].isin(resources)].copy()

    if df.empty:
        raise ValueError(
            f"No Solar/Wind/Battery rows found in {result_path}.\n"
            f"Available columns:\n{list(pd.read_csv(result_path).columns)}"
        )

    df[capacity_col] = pd.to_numeric(df[capacity_col], errors="coerce").fillna(0.0)

    # If the file stores W instead of MW, convert to MW.
    # Large values usually mean W.
    if df[capacity_col].abs().max() > 1e6:
        df[capacity_col] = df[capacity_col] / 1e6

    out = (
        df
        .groupby("resource", as_index=False)[capacity_col]
        .sum()
        .rename(columns={capacity_col: "capex_added_MW"})
    )

    out = (
        pd.DataFrame({"resource": resources})
        .merge(out, on="resource", how="left")
    )

    out["capex_added_MW"] = out["capex_added_MW"].fillna(0.0)

    return out

def is_policy_scenario_folder(folder_name):
    """
    Policy folders have metadata in the folder name.

    Example:
        s01_Base_only_rps50_v1g0_v2g0_bcapex150_mid_m7_d7
        s07_V2G_rps60_v1g0_v2g30_bcapex150_mid_m7_d7

    Simple folders should be skipped for metadata:
        s01_s01_mid_m7_d7
    """

    name = str(folder_name).lower()

    return (
        name.startswith("s")
        and "rps" in name
        and "bcapex" in name
        and ("base_only" in name or "v1g" in name or "v2g" in name)
    )


def find_matching_capex_summary_file(policy_folder, scenario_id):
    """
    Find capex summary for a policy folder.

    The metadata is in:
        s01_Base_only_rps50_v1g0_v2g0_bcapex150_mid_m7_d7

    But the capex file may be in the sibling simple folder:
        s01_s01_mid_m7_d7/s01_s01_capex_summary.csv
    """

    policy_folder = Path(policy_folder)
    parent_dir = policy_folder.parent

    # -----------------------------------------------------
    # 1. First try the policy folder itself
    # -----------------------------------------------------
    files = sorted(policy_folder.glob("*_capex_summary.csv"))

    if files:
        return files[0]

    # -----------------------------------------------------
    # 2. Then try matching sibling simple folders
    # -----------------------------------------------------
    sid = f"s{int(scenario_id):02d}"

    sibling_dirs = []

    for child in sorted(parent_dir.glob(f"{sid}_*")):
        if not child.is_dir():
            continue

        # Skip the policy folder itself.
        if child == policy_folder:
            continue

        # Prefer simple folders like s01_s01_mid_m7_d7.
        # Do not use another policy folder.
        if is_policy_scenario_folder(child.name):
            continue

        sibling_dirs.append(child)

    for sibling in sibling_dirs:
        files = sorted(sibling.glob("*_capex_summary.csv"))

        if files:
            return files[0]

    # -----------------------------------------------------
    # 3. Last fallback: recursive search in matching siblings
    # -----------------------------------------------------
    for sibling in sibling_dirs:
        files = sorted(sibling.rglob("*_capex_summary.csv"))

        if files:
            return files[0]

    return None


def capex_summary_from_capex_summary_csv(capex_summary_path):
    """
    Read the capex summary CSV.

    Expected columns from your file:
        fuel
        capex_added_MW
    """

    df = pd.read_csv(capex_summary_path)

    if "fuel" not in df.columns:
        raise ValueError(
            f"'fuel' column not found in {capex_summary_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if "capex_added_MW" not in df.columns:
        raise ValueError(
            f"'capex_added_MW' column not found in {capex_summary_path}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df["fuel_clean"] = df["fuel"].astype(str).str.lower().str.strip()

    resource_map = {
        "solar": "Solar",
        "wind": "Wind",
        "battery": "Battery",
    }

    df = df[df["fuel_clean"].isin(resource_map.keys())].copy()
    df["resource"] = df["fuel_clean"].map(resource_map)

    out = (
        df
        .groupby("resource", as_index=False)["capex_added_MW"]
        .sum()
    )

    # Force all three resources to exist.
    out = (
        pd.DataFrame({"resource": ["Solar", "Wind", "Battery"]})
        .merge(out, on="resource", how="left")
    )

    out["capex_added_MW"] = out["capex_added_MW"].fillna(0.0)

    return out


def collect_capex_delta_for_region_group(
    output_root,
    region,
    year,
    adoption,
    batt_capex_target=150,
    baseline_scenario_label="arrive",
    program_scenario_label="flex",
    selected_group="V2G",
    selected_participation=("30%", "50%"),
    rps_order=None,
    strict_rps_check=True,
):
    """
    Collect capacity expansion delta for one model region.

    Correct logic for the current folder structure:

    1. Use policy folders for metadata:
        s07_V2G_rps50_v1g0_v2g30_bcapex150_mid_m7_d7

    2. Read capex summary from either:
        same policy folder
        OR sibling simple folder:
        s07_s07_mid_m7_d7/s07_s07_capex_summary.csv

    3. Filter requested RPS cases before reading/merging,
       same as the generation plot.
    """

    region_code = normalize_region(region)

    # Important: define requested RPS set here.
    rps_set = set(rps_order) if rps_order is not None else None

    baseline_dir = get_scenario_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=baseline_scenario_label,
    )

    program_dir = get_scenario_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=program_scenario_label,
    )

    if not baseline_dir.exists():
        raise FileNotFoundError(f"Missing baseline folder: {baseline_dir}")

    if not program_dir.exists():
        raise FileNotFoundError(f"Missing program folder: {program_dir}")

    all_rows = []

    checked_baseline = 0
    checked_program = 0
    missing_capex_files = []

    def read_policy_folder(folder, meta, scenario_label, case_type):
        capex_file = find_matching_capex_summary_file(
            policy_folder=folder,
            scenario_id=meta["scenario_id"],
        )

        if capex_file is None:
            missing_capex_files.append(str(folder))
            return None

        cap = capex_summary_from_capex_summary_csv(capex_file)

        for key, value in meta.items():
            cap[key] = value

        cap["region"] = region_code
        cap["scenario_label"] = scenario_label.lower()
        cap["case_type"] = case_type
        cap["scenario_folder"] = str(folder)
        cap["capex_file"] = str(capex_file)

        return cap

    # =====================================================
    # Baseline rows
    # =====================================================
    for folder in sorted(baseline_dir.glob("s*")):
        if not folder.is_dir():
            continue

        # Skip simple folders like s01_s01_mid_m7_d7.
        if not is_policy_scenario_folder(folder.name):
            continue

        meta = parse_scenario_metadata(folder.name, region=region_code)

        if meta["batt_capex"] != batt_capex_target:
            continue

        if meta["group"] != "Base only":
            continue

        # Same strict RPS filtering as generation plot.
        if rps_set is not None and meta["rps"] not in rps_set:
            continue

        checked_baseline += 1

        cap = read_policy_folder(
            folder=folder,
            meta=meta,
            scenario_label=baseline_scenario_label,
            case_type="baseline",
        )

        if cap is not None:
            all_rows.append(cap)

    # =====================================================
    # Program rows
    # =====================================================
    for folder in sorted(program_dir.glob("s*")):
        if not folder.is_dir():
            continue

        # Skip simple folders like s07_s07_mid_m7_d7.
        if not is_policy_scenario_folder(folder.name):
            continue

        meta = parse_scenario_metadata(folder.name, region=region_code)

        if meta["batt_capex"] != batt_capex_target:
            continue

        if meta["group"] != selected_group:
            continue

        if meta["participation"] not in selected_participation:
            continue

        # Same strict RPS filtering as generation plot.
        if rps_set is not None and meta["rps"] not in rps_set:
            continue

        checked_program += 1

        cap = read_policy_folder(
            folder=folder,
            meta=meta,
            scenario_label=program_scenario_label,
            case_type="program",
        )

        if cap is not None:
            all_rows.append(cap)

    if not all_rows:
        raise ValueError(
            f"No valid capex summary data found.\n"
            f"region = {region_code}\n"
            f"baseline_dir = {baseline_dir}\n"
            f"program_dir = {program_dir}\n"
            f"checked_baseline_folders = {checked_baseline}\n"
            f"checked_program_folders = {checked_program}\n"
            f"group = {selected_group}\n"
            f"battery capex = {batt_capex_target}\n"
            f"participation = {selected_participation}\n"
            f"requested RPS = {rps_order}\n"
            f"missing_capex_files examples = {missing_capex_files[:8]}"
        )

    df = pd.concat(all_rows, ignore_index=True)

    # =====================================================
    # Remove duplicate scenario folders
    # =====================================================
    duplicate_key_cols = [
        "case_type",
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
    ]

    folder_key_cols = duplicate_key_cols + [
        "scenario_id",
        "scenario_folder",
        "capex_file",
    ]

    folder_keys = df[folder_key_cols].drop_duplicates().copy()

    duplicate_summary = (
        folder_keys
        .groupby(duplicate_key_cols, as_index=False)
        .agg(
            n_folders=("scenario_folder", "nunique"),
            folders=("scenario_folder", lambda x: "\n".join(map(str, x))),
        )
    )

    duplicated_cases = duplicate_summary[
        duplicate_summary["n_folders"] > 1
    ].copy()

    if not duplicated_cases.empty:
        print("\nDuplicate capacity scenario folders were found.")
        print("Keeping only one folder per case.")
        print(duplicated_cases.to_string(index=False))

        keep_keys = (
            folder_keys
            .sort_values(
                duplicate_key_cols + ["scenario_id", "scenario_folder"]
            )
            .drop_duplicates(
                subset=duplicate_key_cols,
                keep="first",
            )
        )

        keep_folders = set(keep_keys["scenario_folder"].tolist())
        df = df[df["scenario_folder"].isin(keep_folders)].copy()

    df = df.reset_index(drop=True)

    # =====================================================
    # Strict RPS coverage check
    # =====================================================
    if rps_order is not None and strict_rps_check:
        requested_rps = list(rps_order)
        problems = []

        base_available = sorted(
            df.loc[
                df["case_type"] == "baseline",
                "rps",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        missing_base = [
            rps for rps in requested_rps
            if rps not in base_available
        ]

        if missing_base:
            problems.append(
                f"{region_code} Base only is missing requested RPS {missing_base}. "
                f"Available baseline RPS = {base_available}"
            )

        for part in selected_participation:
            program_available = sorted(
                df.loc[
                    (df["case_type"] == "program")
                    & (df["group"] == selected_group)
                    & (df["participation"] == part),
                    "rps",
                ]
                .dropna()
                .unique()
                .tolist()
            )

            missing_program = [
                rps for rps in requested_rps
                if rps not in program_available
            ]

            if missing_program:
                problems.append(
                    f"{region_code} {selected_group}, {part} is missing requested RPS "
                    f"{missing_program}. Available program RPS = {program_available}"
                )

        if problems:
            raise ValueError(
                "Requested RPS cases are missing before plotting:\n"
                + "\n".join(problems)
            )

    # =====================================================
    # Base table
    # =====================================================
    base = df[df["case_type"] == "baseline"].copy()

    if base.empty:
        raise ValueError(
            f"No Base-only capex rows found for {region_code}."
        )

    base = (
        base
        .groupby(
            [
                "region",
                "rps",
                "rps_display_label",
                "batt_capex",
                "resource",
            ],
            as_index=False,
            dropna=False,
        )["capex_added_MW"]
        .sum()
        .rename(columns={"capex_added_MW": "base_capex_mw"})
    )

    # =====================================================
    # Program table
    # =====================================================
    comp = df[
        (df["case_type"] == "program")
        & (df["group"] == selected_group)
        & (df["participation"].isin(selected_participation))
    ].copy()

    if comp.empty:
        raise ValueError(
            f"No program capex rows found for {region_code}."
        )

    comp_capacity = (
        comp
        .groupby(
            [
                "region",
                "scenario_id",
                "group",
                "participation",
                "rps",
                "rps_display_label",
                "batt_capex",
                "scenario_label",
                "scenario_folder",
                "capex_file",
                "resource",
            ],
            as_index=False,
            dropna=False,
        )["capex_added_MW"]
        .sum()
    )

    # Force Solar/Wind/Battery for every program case.
    program_keys = comp[
        [
            "region",
            "scenario_id",
            "group",
            "participation",
            "rps",
            "rps_display_label",
            "batt_capex",
            "scenario_label",
            "scenario_folder",
            "capex_file",
        ]
    ].drop_duplicates()

    resource_grid = pd.DataFrame({
        "resource": ["Solar", "Wind", "Battery"],
    })

    full_grid = program_keys.merge(resource_grid, how="cross")

    merged = full_grid.merge(
        comp_capacity,
        on=[
            "region",
            "scenario_id",
            "group",
            "participation",
            "rps",
            "rps_display_label",
            "batt_capex",
            "scenario_label",
            "scenario_folder",
            "capex_file",
            "resource",
        ],
        how="left",
    )

    merged = merged.merge(
        base,
        on=[
            "region",
            "rps",
            "rps_display_label",
            "batt_capex",
            "resource",
        ],
        how="left",
    )

    merged["capex_added_MW"] = merged["capex_added_MW"].fillna(0.0)
    merged["base_capex_mw"] = merged["base_capex_mw"].fillna(0.0)

    merged["delta_mw"] = (
        merged["capex_added_MW"] - merged["base_capex_mw"]
    )

    merged["baseline_scenario_label"] = baseline_scenario_label.lower()
    merged["program_scenario_label"] = program_scenario_label.lower()

    return merged.reset_index(drop=True)


def parse_participation_to_fraction(value):
    """
    Convert participation labels such as '30%' or '50%' to 0.30 or 0.50.
    """

    if pd.isna(value):
        return np.nan

    if isinstance(value, str):
        value = value.strip().replace("%", "")
        return float(value) / 100.0

    return float(value)


def normalize_adoption_label(value):
    """
    Match the adoption label style used by your loader:
        slow -> Slow
        mid  -> Mid
        fast -> Fast
    """

    return str(value).strip().capitalize()

def normalize_region(region):
    """
    Convert state-style names to GOOD model-region names.

    Examples:
        CA -> CAISO
        NY -> NYISO
        TX -> ERCOT
        FL -> FRCC
        PJM -> PJM
    """
    region = str(region).upper()

    if region not in REGION_ALIASES:
        raise ValueError(
            f"Unknown region/state code: {region}. "
            f"Allowed values are: {sorted(REGION_ALIASES)}"
        )

    return REGION_ALIASES[region]

def prepare_ev_adoption_df(ev_adoption_df):
    """
    Standardize the EV adoption table.

    Required information:
        region
        year
        adoption
        total_evs

    Also accepts adoption_level instead of adoption.
    """

    df = ev_adoption_df.copy()

    if "adoption" not in df.columns and "adoption_level" in df.columns:
        df = df.rename(columns={"adoption_level": "adoption"})

    required_cols = ["region", "year", "adoption", "total_evs"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"ev_adoption_df is missing required columns: {missing}\n"
            f"Required columns are: {required_cols}"
        )

    df["region"] = df["region"].apply(normalize_region)
    df["adoption"] = df["adoption"].apply(normalize_adoption_label)
    df["year"] = df["year"].astype(int)
    df["total_evs"] = pd.to_numeric(df["total_evs"], errors="coerce")

    if df["total_evs"].isna().any():
        bad = df[df["total_evs"].isna()]
        raise ValueError(f"Some total_evs values could not be converted to numbers:\n{bad}")

    return df


def get_total_evs(
    ev_adoption_df,
    region,
    year,
    adoption_level,
):
    """
    Return total EVs for a selected region-year-adoption case.
    """

    region_code = normalize_region(region)
    adoption = normalize_adoption_label(adoption_level)

    df = prepare_ev_adoption_df(ev_adoption_df)

    match = df[
        (df["region"] == region_code)
        & (df["year"] == int(year))
        & (df["adoption"] == adoption)
    ].copy()

    if match.empty:
        raise ValueError(
            f"No EV adoption value found for:\n"
            f"region={region_code}, year={year}, adoption={adoption}\n\n"
            f"Available combinations:\n"
            f"{df[['region', 'year', 'adoption', 'total_evs']].drop_duplicates()}"
        )

    if len(match) > 1:
        raise ValueError(
            f"More than one EV adoption value found for "
            f"{region_code}, {year}, {adoption}.\n"
            f"Please keep only one row."
        )

    return float(match.iloc[0]["total_evs"])


def compute_available_driver_kwh(
    total_evs,
    participation_fraction,
    usable_battery_kwh=60.0,
    availability_fraction=1,
):
    """
    Denominator for R1.

    available_driver_kwh =
        total EVs
        × participation fraction
        × usable battery size
        × availability fraction

    Example:
        1,000,000 EVs × 50% × 60 kWh × 30%
        = 9,000,000 available kWh
    """

    return (
        total_evs
        * participation_fraction
        * usable_battery_kwh
        * availability_fraction
    )

def add_v2g_charging_scenario_ladder_labels(
    fig,
    axes,
    r1_all_scenarios_df,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    rps_order_by_region=None,
    label_group="V2G",
    label_participation="50%",
    label_rps_by_region=None,
    scenario_name_map=None,
    show_min_max_tags=True,
    fontsize=9.5,
    x_text_offset=0.38,
    min_label_gap_frac=0.055,
):
    """
    Add direct ladder labels with arrows for charging scenarios.

    This does not remove the current legend.

    It labels the charging scenarios used inside the shaded band, usually:
        arrive
        midnight
        flex

    Recommended default:
        label_group = "V2G"
        label_participation = "50%"

    The label is placed at the last RPS value by default.
    """

    import numpy as np

    if scenario_name_map is None:
        scenario_name_map = {
            "arrive": "Arrive",
            "midnight": "Midnight",
            "flex": "Flex",
            "delay": "Delay",
        }

    region_codes = [normalize_region(r) for r in regions]

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    if label_rps_by_region is None:
        label_rps_by_region = {}
    else:
        label_rps_by_region = {
            normalize_region(k): v
            for k, v in label_rps_by_region.items()
        }

    df = r1_all_scenarios_df.copy()

    # Use only the selected group and participation for direct labels.
    df = df[
        (df["group"] == label_group)
        & (df["participation"] == label_participation)
    ].copy()

    if df.empty:
        print(
            f"No rows found for label_group={label_group}, "
            f"label_participation={label_participation}."
        )
        return fig, axes

    for plot_idx, region in enumerate(region_codes):
        if plot_idx >= len(axes):
            continue

        ax = axes[plot_idx]

        sub_region = df[df["region"] == region].copy()

        if sub_region.empty:
            continue

        # Get the same RPS order used in the plot.
        rps_order, _ = build_rps_axis_labels(
            sub_region=sub_region,
            region=region,
            rps_order=rps_order_by_region.get(region, None),
        )

        if len(rps_order) == 0:
            continue

        # By default, label the last RPS value.
        label_rps = label_rps_by_region.get(region, rps_order[-1])

        if label_rps not in rps_order:
            print(f"Skipping {region}: label_rps={label_rps} not in RPS order.")
            continue

        x_positions = np.arange(len(rps_order))
        x_point = x_positions[rps_order.index(label_rps)]

        label_rows = sub_region[
            sub_region["rps_plot_value"] == label_rps
        ].copy()

        if label_rows.empty:
            continue

        # One point per charging scenario.
        label_rows = (
            label_rows
            .sort_values("value_per_available_kwh_year")
            .drop_duplicates(
                subset=["band_program_scenario_label"],
                keep="last",
            )
        )

        if label_rows.empty:
            continue

        y_values = label_rows["value_per_available_kwh_year"].astype(float)

        y_min_value = y_values.min()
        y_max_value = y_values.max()

        records = []

        for _, row in label_rows.iterrows():
            raw_scenario = str(row["band_program_scenario_label"]).lower()
            clean_scenario = scenario_name_map.get(
                raw_scenario,
                raw_scenario.replace("_", " ").title(),
            )

            y = float(row["value_per_available_kwh_year"])

            # -------------------------------------------------
            # Keep only max and min scenarios.
            # Skip middle scenarios.
            # -------------------------------------------------
            is_max = np.isclose(y, y_max_value)
            is_min = np.isclose(y, y_min_value)

            if not (is_max or is_min):
                continue

            if is_max:
                tag = " (max)"
            elif is_min:
                tag = " (min)"
            else:
                tag = ""

            records.append(
                {
                    "scenario": f"{clean_scenario} {label_participation}" + tag,
                    "x": x_point,
                    "y": y,
                }
            )

        if len(records) == 0:
            continue

        # -------------------------------------------------
        # Make room on the right side of the panel.
        # -------------------------------------------------
        old_xlim = ax.get_xlim()
        new_right = max(old_xlim[1], x_point + x_text_offset + 0.65)
        ax.set_xlim(old_xlim[0], new_right)

        y_low, y_high = ax.get_ylim()
        y_range = y_high - y_low

        if y_range <= 0:
            y_range = 1.0

        min_gap = min_label_gap_frac * y_range

        # -------------------------------------------------
        # Ladder placement to avoid overlap.
        # -------------------------------------------------
        records = sorted(records, key=lambda d: d["y"])

        adjusted_y = []

        lower_bound = y_low + 0.04 * y_range
        upper_bound = y_high - 0.04 * y_range

        for i, rec in enumerate(records):
            y_target = rec["y"]

            if i == 0:
                y_adj = max(y_target, lower_bound)
            else:
                y_adj = max(y_target, adjusted_y[-1] + min_gap)

            adjusted_y.append(y_adj)

        # Shift down if labels go too high.
        overflow = adjusted_y[-1] - upper_bound

        if overflow > 0:
            adjusted_y = [y - overflow for y in adjusted_y]

        # Shift up if labels go too low.
        underflow = lower_bound - adjusted_y[0]

        if underflow > 0:
            adjusted_y = [y + underflow for y in adjusted_y]

        # -------------------------------------------------
        # Draw arrows and labels.
        # -------------------------------------------------
        line_color = GROUP_COLORS.get(label_group, "gray")

        for rec, y_text in zip(records, adjusted_y):
            ax.annotate(
                rec["scenario"],
                xy=(rec["x"], rec["y"]),
                xytext=(rec["x"] + x_text_offset, y_text),
                textcoords="data",
                ha="left",
                va="center",
                fontsize=fontsize,
                color=line_color,
                fontweight="bold",
                arrowprops=dict(
                    arrowstyle="->",
                    color=line_color,
                    linewidth=1.0,
                    alpha=0.9,
                    shrinkA=2,
                    shrinkB=3,
                ),
                bbox=dict(
                    facecolor="white",
                    edgecolor=line_color,
                    alpha=0.82,
                    boxstyle="round,pad=0.25",
                    linewidth=0.8,
                ),
                zorder=20,
                clip_on=False,
            )

    return fig, axes

def compute_r1_value_vs_rps_for_region(
    output_root,
    region,
    year,
    adoption_level,
    batt_capex,
    baseline_scenario_label,
    program_scenario_label,
    ev_adoption_df,
    rps_order=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
):
    """
    Compute R1 value for one region across RPS levels.

    Important:
        The GOOD model is run for one representative period, usually one week.
        Therefore, if annualize_model_period=True, system savings are converted
        to annual savings before dividing by available EV kWh.

    R1 metric:
        value_per_available_kwh_year =
            annualized system savings / available driver EV kWh

    Participation interpretation:
        30% participation means 30% of weighted regional EVs participate.

    Denominator:
        available_driver_kwh =
            weighted regional EVs
            × participation fraction
            × usable battery kWh
            × availability fraction
    """

    region_code = normalize_region(region)
    adoption_label = normalize_adoption_label(adoption_level)

    needed_labels = sorted({
        baseline_scenario_label.lower(),
        program_scenario_label.lower(),
    })

    # -----------------------------------------------------
    # Load GOOD cost results
    # -----------------------------------------------------
    df = load_region_total_costs(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption_level=adoption_level,
        batt_capex=batt_capex,
        scenario_labels=needed_labels,
        model_days_per_run=model_days_per_run,
    )

    df = df[df["group"].isin(["Base only", "V1G", "V2G"])].copy()

    if rps_order is not None:
        df = df[df["rps_plot_value"].isin(rps_order)].copy()

    df = df[
        (df["group"] == "Base only")
        | (df["participation"].isin(selected_participation))
    ].copy()

    # -----------------------------------------------------
    # Baseline rows
    # -----------------------------------------------------
    base = df[
        (df["group"] == "Base only")
        & (df["scenario_label"] == baseline_scenario_label.lower())
    ].copy()

    if base.empty:
        raise ValueError(
            f"No baseline rows found for {region_code}.\n"
            f"baseline_scenario_label={baseline_scenario_label}\n"
            f"batt_capex={batt_capex}\n"
            f"adoption_level={adoption_level}"
        )

    base = base[
        [
            "region",
            "year",
            "adoption",
            "batt_capex_num",
            "rps_plot_value",
            "rps_display_label",
            "objective_value",
            "objective_value_mil",
            "folder",
            "file_path",
        ]
    ].copy()

    base = base.rename(
        columns={
            "objective_value": "baseline_objective_value",
            "objective_value_mil": "baseline_objective_value_mil",
            "folder": "baseline_folder",
            "file_path": "baseline_file_path",
        }
    )

    base = (
        base.sort_values(["region", "rps_plot_value", "batt_capex_num"])
        .drop_duplicates(
            subset=["region", "batt_capex_num", "rps_plot_value"],
            keep="first",
        )
    )

    # -----------------------------------------------------
    # Program rows
    # -----------------------------------------------------
    prog = df[
        (df["group"].isin(selected_groups))
        & (df["scenario_label"] == program_scenario_label.lower())
        & (df["participation"].isin(selected_participation))
    ].copy()

    if prog.empty:
        raise ValueError(
            f"No program rows found for {region_code}.\n"
            f"program_scenario_label={program_scenario_label}\n"
            f"selected_groups={selected_groups}\n"
            f"selected_participation={selected_participation}\n"
            f"batt_capex={batt_capex}\n"
            f"adoption_level={adoption_level}"
        )

    # -----------------------------------------------------
    # Match program rows to baseline at same RPS and battery capex
    # -----------------------------------------------------
    out = prog.merge(
        base,
        on=["region", "batt_capex_num", "rps_plot_value"],
        how="left",
        suffixes=("", "_base"),
    )

    if out["baseline_objective_value"].isna().any():
        missing = out[out["baseline_objective_value"].isna()][
            [
                "region",
                "group",
                "participation",
                "batt_capex_num",
                "rps_plot_value",
                "rps_display_label",
            ]
        ].drop_duplicates()

        raise ValueError(
            f"Missing baseline cost for some program rows in {region_code}.\n"
            f"Missing combinations:\n{missing}"
        )

    # -----------------------------------------------------
    # Annualization
    # -----------------------------------------------------
    if annualize_model_period:
        annualization_factor = 365.0 / float(model_days_per_run)
    else:
        annualization_factor = 1.0

    out["model_days_per_run"] = model_days_per_run
    out["annualization_factor"] = annualization_factor

    # This is the model-period saving, e.g., one-week saving.
    out["system_savings_model_period"] = (
        out["baseline_objective_value"] - out["objective_value"]
    )

    out["system_savings_model_period_mil"] = (
        out["system_savings_model_period"] / 1e6
    )

    # This is the annualized saving used for $/kWh-year.
    out["system_savings_annualized"] = (
        out["system_savings_model_period"] * annualization_factor
    )

    out["system_savings_annualized_mil"] = (
        out["system_savings_annualized"] / 1e6
    )

    # Keep old names for compatibility with plotting/table code.
    out["system_savings"] = out["system_savings_annualized"]
    out["system_savings_mil"] = out["system_savings_annualized_mil"]

    # -----------------------------------------------------
    # Denominator: only participating vehicles
    # -----------------------------------------------------
    weighted_regional_evs = get_total_evs(
        ev_adoption_df=ev_adoption_df,
        region=region_code,
        year=year,
        adoption_level=adoption_label,
    )

    out["weighted_regional_evs"] = weighted_regional_evs
    out["total_evs"] = weighted_regional_evs

    out["participation_fraction"] = out["participation"].apply(
        parse_participation_to_fraction
    )

    out["participating_evs"] = (
        out["weighted_regional_evs"] * out["participation_fraction"]
    )

    out["usable_battery_kwh"] = usable_battery_kwh
    out["availability_fraction"] = availability_fraction

    out["available_kwh_per_participating_ev"] = (
        usable_battery_kwh * availability_fraction
    )

    out["available_driver_kwh"] = (
        out["participating_evs"]
        * out["available_kwh_per_participating_ev"]
    )

    # -----------------------------------------------------
    # R1 value metrics
    # -----------------------------------------------------
    out["value_per_available_kwh_year"] = (
        out["system_savings_annualized"] / out["available_driver_kwh"]
    )

    out["value_per_available_mwh_year"] = (
        out["value_per_available_kwh_year"] * 1000.0
    )

    out["value_per_participating_ev_year"] = (
        out["system_savings_annualized"] / out["participating_evs"]
    )

    out["r1_metric"] = "annualized_gross_system_value_before_charger_cost"

    out["series_label"] = out["group"] + ", " + out["participation"]
    out["baseline_scenario_label"] = baseline_scenario_label.lower()
    out["program_scenario_label"] = program_scenario_label.lower()

    keep_cols = [
        "region",
        "year",
        "adoption",
        "baseline_scenario_label",
        "program_scenario_label",
        "group",
        "participation",
        "participation_fraction",
        "series_label",
        "batt_capex_num",
        "batt_capex_label",
        "rps_plot_value",
        "rps_display_label",
        "objective_value",
        "baseline_objective_value",
        "model_days_per_run",
        "annualization_factor",
        "system_savings_model_period",
        "system_savings_model_period_mil",
        "system_savings_annualized",
        "system_savings_annualized_mil",
        "system_savings",
        "system_savings_mil",
        "weighted_regional_evs",
        "total_evs",
        "participating_evs",
        "usable_battery_kwh",
        "availability_fraction",
        "available_kwh_per_participating_ev",
        "available_driver_kwh",
        "value_per_available_kwh_year",
        "value_per_available_mwh_year",
        "value_per_participating_ev_year",
        "folder",
        "file_path",
        "baseline_folder",
        "baseline_file_path",
        "r1_metric",
    ]

    keep_cols = [col for col in keep_cols if col in out.columns]

    return out[keep_cols].reset_index(drop=True)
        
def build_rps_axis_labels(
    sub_region,
    region,
    rps_order=None,
):
    """
    Build region-specific RPS axis order and labels.

    Multi-state regions:
        -10 -> RPS −10 pp
          0 -> RPS baseline
         10 -> RPS +10 pp

    Single-state regions:
        50 -> 50%
        60 -> 60%
    """

    region_code = normalize_region(region)

    if rps_order is None:
        rps_order = (
            sub_region["rps_plot_value"]
            .dropna()
            .astype(float)
            .sort_values()
            .unique()
            .tolist()
        )

    labels = []

    for rps in rps_order:
        # Multi-state regions use relative
        # percentage-point RPS cases.
        if region_code in MULTI_STATE_REGIONS:
            if rps == -10:
                label = "RPS −10 pp"

            elif rps == 0:
                label = "RPS baseline"

            elif rps == 10:
                label = "RPS +10 pp"

            else:
                label = (
                    f"RPS {int(rps):+d} pp"
                )

        # Single-state regions use absolute
        # RPS percentages.
        else:
            local = sub_region[
                sub_region["rps_plot_value"] == rps
            ]

            if (
                not local.empty
                and "rps_display_label"
                in local.columns
                and local.iloc[0][
                    "rps_display_label"
                ] is not None
            ):
                label = str(
                    local.iloc[0][
                        "rps_display_label"
                    ]
                )

            else:
                label = f"{int(rps)}%"

        labels.append(label)

    return list(rps_order), labels


def plot_r1_value_vs_rps_regions(
    r1_df,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    rps_order_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    sharey=True,
    y_limits=None,
    figure_title=None,
    save_path=None,
):
    """
    Plot R1 value vs RPS target.

    x-axis:
        RPS target, region-specific.

    y-axis:
        value_per_available_kwh_year.

    Lines:
        V1G/V2G by participation.

    Panels:
        regions.
    """

    region_codes = [normalize_region(r) for r in regions]

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    df = r1_df.copy()
    df = df[df["region"].isin(region_codes)].copy()
    df = df[df["group"].isin(selected_groups)].copy()
    df = df[df["participation"].isin(selected_participation)].copy()

    if df.empty:
        raise ValueError("No data left after filtering. Check regions/groups/participation.")

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    n_regions = len(region_codes)

    if n_regions <= 3:
        nrows = 1
        ncols = n_regions
        figsize = (5.5 * ncols, 5.2)
    else:
        nrows = 2
        ncols = 3
        figsize = (17.0, 9.5)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
        sharey=sharey,
    )

    axes = axes_grid.flatten()

    series_order = []

    for group in selected_groups:
        for part in selected_participation:
            series_order.append((group, part))

    # -----------------------------------------------------
    # Plot each region
    # -----------------------------------------------------
    for plot_idx, region in enumerate(region_codes):
        ax = axes[plot_idx]
        ax.set_facecolor("white")
        sub_region = df[df["region"] == region].copy()

        if sub_region.empty:
            ax.axis("off")
            ax.set_title(region)
            continue

        rps_order, rps_labels = build_rps_axis_labels(
            sub_region=sub_region,
            region=region,
            rps_order=rps_order_by_region.get(region, None),
        )

        x_positions = np.arange(len(rps_order))
        x_lookup = dict(zip(rps_order, x_positions))

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            zorder=1,
        )

        for group, part in series_order:
            temp = sub_region[
                (sub_region["group"] == group)
                & (sub_region["participation"] == part)
            ].copy()

            if temp.empty:
                continue

            temp = (
                temp.sort_values("rps_plot_value")
                .drop_duplicates(subset=["rps_plot_value"], keep="first")
                .set_index("rps_plot_value")
                .reindex(rps_order)
            )

            y = temp["value_per_available_kwh_year"].values

            ax.plot(
                x_positions,
                y,
                color=GROUP_COLORS.get(group, "gray"),
                marker=MARKER_MAP.get(part, "o"),
                linewidth=2.2,
                markersize=7.0,
                label=f"{group}, {part}",
                zorder=3,
            )

        panel_title = REGION_DISPLAY_NAMES.get(region, region)

        ax.set_title(
            panel_title,
            fontsize=17,
            fontweight="bold",
            color="black",
            pad=8,
        )

        ax.set_xlabel(
            "RPS target",
            fontsize=13,
            color="black",
        )

        ax.set_ylabel(
            "Value of EV flexibility\n($/available kWh-year)",
            fontsize=13,
            color="black",
        )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            rps_labels,
            fontsize=11,
            color="black",
        )

        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.35,
            color="gray",
        )

        ax.tick_params(axis="both", colors="black", labelsize=11)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    # -----------------------------------------------------
    # Legend in unused panel, if available
    # -----------------------------------------------------
    legend_handles = []

    for group in selected_groups:
        legend_handles.append(
            Line2D(
                [0], [0],
                color=GROUP_COLORS.get(group, "gray"),
                lw=2.4,
                label=group,
            )
        )

    for part in selected_participation:
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker=MARKER_MAP.get(part, "o"),
                lw=0,
                markersize=7,
                label=f"{part} participation",
            )
        )

    legend_handles.append(
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            lw=1.1,
            label="Zero value",
        )
    )

    if len(axes) > n_regions:
        legend_ax = axes[n_regions]
        legend_ax.axis("off")

        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=13,
            ncol=1,
        )

        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(1.0)

        for txt in legend.get_texts():
            txt.set_color("black")

        for extra_ax in axes[n_regions + 1:]:
            extra_ax.axis("off")

    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(5, len(legend_handles)),
            frameon=True,
            fontsize=11,
        )

    # -----------------------------------------------------
    # Figure title
    # -----------------------------------------------------
    if figure_title is None:
        first = df.iloc[0]
        figure_title = (
            "Value of 1 kWh of driver-provided EV flexibility across RPS targets\n"
            f"Year = {first['year']}, adoption = {first['adoption']}, "
            f"battery capex = {first['batt_capex_label']}"
        )

    fig.suptitle(
        figure_title,
        fontsize=18,
        color="black",
        y=0.98,
    )

    if n_regions <= 3:
        fig.subplots_adjust(
            top=0.82,
            bottom=0.16,
            left=0.08,
            right=0.98,
            wspace=0.25,
        )
    else:
        fig.subplots_adjust(
            top=0.86,
            bottom=0.08,
            left=0.07,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, axes



def discover_available_program_scenario_labels(
    output_root,
    region,
    year,
    adoption_level,
):
    """
    Discover available scenario labels for one region-year-adoption case.

    Example output:
        ["arrive", "flex", "midnight"]
    """

    region_code = normalize_region(region)

    result_dirs = find_result_dirs(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption_levels=(adoption_level,),
        scenario_labels=None,
    )

    labels = sorted({
        item["scenario_label"].lower()
        for item in result_dirs
    })

    if len(labels) == 0:
        raise ValueError(
            f"No scenario labels found for {region_code}, "
            f"year={year}, adoption={adoption_level}."
        )

    return labels

def compute_r1_value_vs_rps_with_program_band_for_region(
    output_root,
    region,
    year,
    adoption_level,
    batt_capex,
    baseline_scenario_label,
    main_program_scenario_label,
    ev_adoption_df,
    rps_order=None,
    band_program_scenario_labels=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
    verbose=True,
):
    """
    Compute R1 for one region.

    Main line:
        R1 for main_program_scenario_label.

    Shaded band:
        min/max R1 across band_program_scenario_labels.

    The baseline is fixed:
        every program scenario is compared against the same baseline_scenario_label.
    """

    region_code = normalize_region(region)

    # -----------------------------------------------------
    # Decide which charging scenarios enter the band
    # -----------------------------------------------------
    if band_program_scenario_labels is None:
        band_program_scenario_labels = discover_available_program_scenario_labels(
            output_root=output_root,
            region=region_code,
            year=year,
            adoption_level=adoption_level,
        )
    else:
        band_program_scenario_labels = [
            str(x).lower()
            for x in band_program_scenario_labels
        ]

    main_program_scenario_label = str(main_program_scenario_label).lower()

    if main_program_scenario_label not in band_program_scenario_labels:
        band_program_scenario_labels = (
            list(band_program_scenario_labels)
            + [main_program_scenario_label]
        )

    all_program_rows = []
    skipped = []

    # -----------------------------------------------------
    # Compute R1 for each possible program charging scenario
    # -----------------------------------------------------
    for scenario_label in band_program_scenario_labels:
        try:
            temp = compute_r1_value_vs_rps_for_region(
                output_root=output_root,
                region=region_code,
                year=year,
                adoption_level=adoption_level,
                batt_capex=batt_capex,
                baseline_scenario_label=baseline_scenario_label,
                program_scenario_label=scenario_label,
                ev_adoption_df=ev_adoption_df,
                rps_order=rps_order,
                selected_groups=selected_groups,
                selected_participation=selected_participation,
                usable_battery_kwh=usable_battery_kwh,
                availability_fraction=availability_fraction,
                model_days_per_run=model_days_per_run,
                annualize_model_period=annualize_model_period,
            )

            temp["band_program_scenario_label"] = scenario_label
            temp["is_main_program_scenario"] = (
                scenario_label == main_program_scenario_label
            )

            all_program_rows.append(temp)

        except Exception as exc:
            skipped.append((scenario_label, str(exc)))

    if len(all_program_rows) == 0:
        msg = "\n".join([f"{x[0]}: {x[1]}" for x in skipped])
        raise ValueError(
            f"No valid R1 rows found for {region_code}.\n"
            f"Skipped scenarios:\n{msg}"
        )

    all_df = pd.concat(all_program_rows, ignore_index=True)

    if verbose and skipped:
        print(f"\nSkipped scenarios for {region_code}:")
        for scenario_label, reason in skipped:
            print(f"  - {scenario_label}: {reason.splitlines()[0]}")

    # -----------------------------------------------------
    # Main-line data
    # -----------------------------------------------------
    main_df = all_df[
        all_df["band_program_scenario_label"] == main_program_scenario_label
    ].copy()

    if main_df.empty:
        raise ValueError(
            f"Main program scenario '{main_program_scenario_label}' "
            f"was not found for {region_code}."
        )

    # -----------------------------------------------------
    # Band data: min/max across charging scenarios
    # -----------------------------------------------------
    group_cols = [
        "region",
        "year",
        "adoption",
        "group",
        "participation",
        "batt_capex_num",
        "batt_capex_label",
        "rps_plot_value",
        "rps_display_label",
    ]

    band_rows = []

    for keys, sub in all_df.groupby(group_cols, dropna=False):
        sub = sub.copy()

        value_col = "value_per_available_kwh_year"

        idx_min = sub[value_col].idxmin()
        idx_max = sub[value_col].idxmax()

        row_dict = dict(zip(group_cols, keys))

        row_dict.update(
            {
                "r1_min": sub.loc[idx_min, value_col],
                "r1_max": sub.loc[idx_max, value_col],
                "r1_mean": sub[value_col].mean(),
                "r1_main": (
                    sub.loc[
                        sub["band_program_scenario_label"]
                        == main_program_scenario_label,
                        value_col,
                    ].iloc[0]
                    if (
                        sub["band_program_scenario_label"]
                        == main_program_scenario_label
                    ).any()
                    else np.nan
                ),
                "scenario_at_min": sub.loc[
                    idx_min,
                    "band_program_scenario_label",
                ],
                "scenario_at_max": sub.loc[
                    idx_max,
                    "band_program_scenario_label",
                ],
                "n_scenarios_in_band": sub[
                    "band_program_scenario_label"
                ].nunique(),
                "main_program_scenario_label": main_program_scenario_label,
                "baseline_scenario_label": baseline_scenario_label.lower(),
            }
        )

        band_rows.append(row_dict)

    band_df = pd.DataFrame(band_rows)

    return main_df.reset_index(drop=True), band_df.reset_index(drop=True), all_df.reset_index(drop=True)

def compute_r1_value_vs_rps_with_program_band_regions(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption_level="mid",
    batt_capex=150,
    baseline_scenario_label_by_region=None,
    main_program_scenario_label_by_region=None,
    ev_adoption_df=None,
    rps_order_by_region=None,
    band_program_scenario_labels_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
    verbose=True,
):
    """
    Compute main R1 line and charging-scenario band for multiple regions.
    """

    if ev_adoption_df is None:
        raise ValueError("ev_adoption_df cannot be None.")

    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    main_program_map = normalize_label_map(
        main_program_scenario_label_by_region,
        regions,
        "main_program_scenario_label_by_region",
    )

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    if band_program_scenario_labels_by_region is None:
        band_map = {}
    else:
        band_map = {
            normalize_region(k): v
            for k, v in band_program_scenario_labels_by_region.items()
        }

    main_list = []
    band_list = []
    all_list = []

    for region in region_codes:
        main_df, band_df, all_df = compute_r1_value_vs_rps_with_program_band_for_region(
            output_root=output_root,
            region=region,
            year=year,
            adoption_level=adoption_level,
            batt_capex=batt_capex,
            baseline_scenario_label=baseline_map[region],
            main_program_scenario_label=main_program_map[region],
            ev_adoption_df=ev_adoption_df,
            rps_order=rps_order_by_region.get(region, None),
            band_program_scenario_labels=band_map.get(region, None),
            selected_groups=selected_groups,
            selected_participation=selected_participation,
            usable_battery_kwh=usable_battery_kwh,
            availability_fraction=availability_fraction,
            model_days_per_run=model_days_per_run,
            annualize_model_period=annualize_model_period,
            verbose=verbose,
        )

        main_list.append(main_df)
        band_list.append(band_df)
        all_list.append(all_df)

    r1_main_df = pd.concat(main_list, ignore_index=True)
    r1_band_df = pd.concat(band_list, ignore_index=True)
    r1_all_scenarios_df = pd.concat(all_list, ignore_index=True)

    return r1_main_df, r1_band_df, r1_all_scenarios_df

def plot_r1_value_vs_rps_regions_with_program_band(
    r1_main_df,
    r1_band_df,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    rps_order_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("25%", "50%"),
    sharey=True,
    y_limits=None,
    figure_title=None,
    save_path=None,
    show_participation_band=True,
    base_participation="25%",
    upper_participation="50%",
):
    """
    Plot the best charging scenario at the base participation level.

    Main line:
        Best charging scenario at base_participation, normally 25%.

    Shaded band:
        Range between the best results at base_participation and
        upper_participation, normally 25% and 50%.

    r1_band_df is retained only for backward compatibility.
    The charging-scenario range is no longer plotted.
    """

    import matplotlib as mpl
    from matplotlib.patches import Patch

    plt.style.use("default")

    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "grid.color": "gray",
        }
    )

    region_codes = [normalize_region(r) for r in regions]

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    # r1_band_df is intentionally not used anymore.
    # The new band comes from the 25% and 50% rows in r1_main_df.
    main_df = r1_main_df.copy()

    main_df = main_df[
        main_df["region"].isin(region_codes)
    ].copy()

    main_df = main_df[
        main_df["group"].isin(selected_groups)
    ].copy()

    main_df = main_df[
        main_df["participation"].isin(selected_participation)
    ].copy()

    if main_df.empty:
        raise ValueError(
            "No main-line data left after filtering."
        )

    required_participation = {
        base_participation,
        upper_participation,
    }

    available_participation = set(
        main_df["participation"].dropna().unique()
    )

    missing_participation = (
        required_participation - available_participation
    )

    if missing_participation:
        raise ValueError(
            "The participation band requires both participation levels.\n"
            f"Missing: {sorted(missing_participation)}\n"
            f"Available: {sorted(available_participation)}"
        )

    # =====================================================
    # Layout
    # =====================================================
    n_regions = len(region_codes)

    if n_regions <= 3:
        nrows = 1
        ncols = n_regions
        figsize = (5.5 * ncols, 5.2)
    else:
        nrows = 2
        ncols = 3
        figsize = (17.0, 9.5)

    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        facecolor="white",
        squeeze=False,
        sharey=sharey,
    )

    axes = axes_grid.flatten()

    # =====================================================
    # Helper to obtain one participation series
    # =====================================================
    def _get_participation_series(
        sub_region,
        group,
        participation,
        rps_order,
    ):
        temp = sub_region[
            (sub_region["group"] == group)
            & (
                sub_region["participation"]
                == participation
            )
        ].copy()

        if temp.empty:
            return None

        temp = (
            temp
            .sort_values("rps_plot_value")
            .drop_duplicates(
                subset=["rps_plot_value"],
                keep="first",
            )
            .set_index("rps_plot_value")
            .reindex(rps_order)
        )

        return pd.to_numeric(
            temp["value_per_available_kwh_year"],
            errors="coerce",
        ).to_numpy(dtype=float)

    # =====================================================
    # Plot each region
    # =====================================================
    for plot_idx, region in enumerate(region_codes):
        ax = axes[plot_idx]
        ax.set_facecolor("white")

        sub_main = main_df[
            main_df["region"] == region
        ].copy()

        if sub_main.empty:
            ax.axis("off")
            ax.set_title(region)
            continue

        rps_order, rps_labels = build_rps_axis_labels(
            sub_region=sub_main,
            region=region,
            rps_order=rps_order_by_region.get(
                region,
                None,
            ),
        )

        x_positions = np.arange(len(rps_order))

        ax.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.9,
            zorder=1,
        )

        for group in selected_groups:
            color = GROUP_COLORS.get(
                group,
                "gray",
            )

            # ---------------------------------------------
            # Best scenario at 25% participation
            # ---------------------------------------------
            y_base = _get_participation_series(
                sub_region=sub_main,
                group=group,
                participation=base_participation,
                rps_order=rps_order,
            )

            if y_base is None:
                continue

            # ---------------------------------------------
            # Best scenario at 50% participation
            # ---------------------------------------------
            y_upper = _get_participation_series(
                sub_region=sub_main,
                group=group,
                participation=upper_participation,
                rps_order=rps_order,
            )

            # ---------------------------------------------
            # Shade between actual 25% and 50% values
            # ---------------------------------------------
            if (
                show_participation_band
                and y_upper is not None
            ):
                valid = (
                    np.isfinite(y_base)
                    & np.isfinite(y_upper)
                )

                y_low = np.minimum(
                    y_base,
                    y_upper,
                )

                y_high = np.maximum(
                    y_base,
                    y_upper,
                )

                ax.fill_between(
                    x_positions,
                    y_low,
                    y_high,
                    where=valid,
                    interpolate=True,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                    zorder=2,
                )

            # ---------------------------------------------
            # Plot only the 25% line
            # ---------------------------------------------
            ax.plot(
                x_positions,
                y_base,
                color=color,
                marker="o",
                linewidth=2.4,
                markersize=7.0,
                label=f"{group}, {base_participation}",
                zorder=4,
            )

        panel_title = REGION_DISPLAY_NAMES.get(
            region,
            region,
        )

        ax.set_title(
            panel_title,
            fontsize=17,
            fontweight="bold",
            color="black",
            pad=8,
        )

        ax.set_xlabel(
            "RPS target",
            fontsize=13,
            color="black",
        )

        ax.set_ylabel(
            "EV flexibility value\n"
            "($/available kWh-year)",
            fontsize=13,
            color="black",
        )

        ax.set_xticks(x_positions)

        ax.set_xticklabels(
            rps_labels,
            fontsize=11,
            color="black",
        )

        if y_limits is not None:
            ax.set_ylim(y_limits)

        ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.35,
            color="gray",
        )

        ax.tick_params(
            axis="both",
            colors="black",
            labelsize=11,
        )

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    # =====================================================
    # Legend
    # =====================================================
    legend_handles = []

    for group in selected_groups:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=GROUP_COLORS.get(
                    group,
                    "gray",
                ),
                marker="o",
                lw=2.4,
                markersize=7,
                label=(
                    f"{group} at "
                    f"{base_participation} participation"
                ),
            )
        )

    if show_participation_band:
        legend_handles.append(
            Patch(
                facecolor="gray",
                edgecolor="none",
                alpha=0.18,
                label=(
                    f"{base_participation}–"
                    f"{upper_participation} "
                    f"participation range"
                ),
            )
        )

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            lw=1.1,
            label="Zero value",
        )
    )

    if len(axes) > n_regions:
        legend_ax = axes[n_regions]
        legend_ax.axis("off")

        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=12,
            ncol=1,
        )

        legend.get_frame().set_facecolor(
            "white"
        )

        legend.get_frame().set_edgecolor(
            "black"
        )

        legend.get_frame().set_linewidth(
            1.0
        )

        for txt in legend.get_texts():
            txt.set_color("black")

        for extra_ax in axes[n_regions + 1:]:
            extra_ax.axis("off")

    else:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=min(
                4,
                len(legend_handles),
            ),
            frameon=True,
            fontsize=11,
        )

    # =====================================================
    # Title
    # =====================================================
    if figure_title is None:
        first = main_df.iloc[0]

        figure_title = (
            "Value of 1 kWh of driver-provided EV "
            "flexibility across RPS targets\n"
            f"Line = best charging scenario at "
            f"{base_participation} participation, "
            f"shaded band = {base_participation}–"
            f"{upper_participation} participation range; "
            f"year = {first['year']}, "
            f"adoption = {first['adoption']}, "
            f"battery capex = "
            f"{first['batt_capex_label']}"
        )

    fig.suptitle(
        figure_title,
        fontsize=18,
        color="black",
        y=0.98,
    )

    if n_regions <= 3:
        fig.subplots_adjust(
            top=0.82,
            bottom=0.16,
            left=0.08,
            right=0.98,
            wspace=0.25,
        )
    else:
        fig.subplots_adjust(
            top=0.86,
            bottom=0.08,
            left=0.07,
            right=0.98,
            hspace=0.35,
            wspace=0.25,
        )

    if save_path is not None:
        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return fig, axes


def plot_r1_value_vs_rps_from_outputs_with_program_band(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption_level="mid",
    batt_capex=150,
    baseline_scenario_label_by_region=None,
    main_program_scenario_label_by_region=None,
    ev_adoption_df=None,
    rps_order_by_region=None,
    band_program_scenario_labels_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
    sharey=True,
    y_limits=None,
    figure_title=None,
    save_path=None,
    save_main_csv_path=None,
    save_band_csv_path=None,
    save_all_csv_path=None,
    verbose=True,
    font_size=13,
    figure_title_fontsize=None,
    panel_title_fontsize=None,
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    legend_fontsize=None,
    legend_title_fontsize=None,
    main_program_mode="selected",
    show_main_scenario_change_labels=True,
    scenario_change_label_group="V2G",
    scenario_change_label_participation="30%",
    scenario_change_label_regions=None,
    scenario_name_map=None,
    scenario_change_label_fontsize=8.8,
    scenario_change_label_x_offset=0.18,
    scenario_change_label_y_offset=0.00,
    scenario_change_label_show_tag=True,
    scenario_change_label_rps_by_region=None,
    show_participation_band=True,
    base_participation="25%",
    upper_participation="50%",
):
    """
    Full workflow with charging-scenario shaded band.

    main_program_mode:
        "selected":
            Old behavior. The main line uses one selected charging scenario
            per region from main_program_scenario_label_by_region.

        "upper_band":
            New behavior. The main line uses the best charging scenario at
            each region / RPS / group / participation point. This makes the
            main line equal to the top edge of the shaded band.

    If main_program_mode="upper_band" and the best scenario changes across RPS,
    the function adds direct labels on the graph showing which scenario is used.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
# =====================================================
    # Font controls
    # =====================================================
    if figure_title_fontsize is None:
        figure_title_fontsize = font_size + 5

    if panel_title_fontsize is None:
        panel_title_fontsize = font_size + 4

    if axis_label_fontsize is None:
        axis_label_fontsize = font_size

    if tick_label_fontsize is None:
        tick_label_fontsize = font_size - 1

    if legend_fontsize is None:
        legend_fontsize = font_size - 1

    if legend_title_fontsize is None:
        legend_title_fontsize = font_size
    # =====================================================
    # Helpers
    # =====================================================
    def _normalize_region_local(region):
        try:
            return normalize_region(region)
        except Exception:
            return str(region)

    def _get_group_color(group):
        if "GROUP_COLORS" in globals():
            return GROUP_COLORS.get(group, "black")

        fallback_colors = {
            "Base only": "black",
            "V1G": "#4E79A7",
            "V2G": "#E15759",
        }

        return fallback_colors.get(group, "black")

    def _get_marker_for_participation(participation):
        if "MARKER_MAP" in globals():
            return MARKER_MAP.get(participation, "o")

        fallback_markers = {
            "0%": "o",
            "10%": "D",
            "25%": "s",
            "50%": "^",
        }

        return fallback_markers.get(participation, "o")

    def _build_upper_band_main_df(r1_all_scenarios_df):
        """
        Build main line as the point-by-point maximum of the band.

        This fixes cases like NY where:
            60% -> delay
            70% -> delay
            80% -> arrive

        Important:
            Do not use idxmax + loc here, because if the dataframe index is
            duplicated or not clean, loc can return the wrong row or extra rows.
            Sorting + drop_duplicates is more stable.
        """

        value_col = "value_per_available_kwh_year"
        rps_col = "rps_plot_value"
        scenario_col = "program_scenario_label"

        required_cols = [
            "region",
            rps_col,
            "group",
            "participation",
            scenario_col,
            value_col,
        ]

        missing_cols = [
            col for col in required_cols
            if col not in r1_all_scenarios_df.columns
        ]

        if missing_cols:
            raise ValueError(
                "Cannot build upper-band main line. Missing columns:\n"
                + "\n".join(missing_cols)
            )

        key_cols = [
            "region",
            rps_col,
            "group",
            "participation",
        ]

        work = r1_all_scenarios_df.copy().reset_index(drop=True)

        work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
        work = work.dropna(subset=[value_col]).copy()

        # Keep only selected groups and participation levels.
        work = work[
            (work["group"].isin(selected_groups))
            & (work["participation"].isin(selected_participation))
        ].copy()

        # Normalize scenario names for stable tie-breaking.
        work["_scenario_lower"] = (
            work[scenario_col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        # Optional deterministic tie-break order.
        # This only matters if two scenarios have exactly the same value.
        scenario_tie_order = {
            "delay": 0,
            "arrive": 1,
            "flex": 2,
            "midnight": 3,
        }

        work["_scenario_tie_rank"] = (
            work["_scenario_lower"]
            .map(scenario_tie_order)
            .fillna(99)
        )

        # Sort so the highest value is first within each key.
        # If values tie exactly, use scenario_tie_rank.
        work = work.sort_values(
            key_cols + [value_col, "_scenario_tie_rank"],
            ascending=[True, True, True, True, False, True],
        )

        out = (
            work
            .drop_duplicates(
                subset=key_cols,
                keep="first",
            )
            .copy()
            .reset_index(drop=True)
        )

        out["main_program_mode"] = "upper_band"

        # Clean helper columns.
        out = out.drop(
            columns=[
                "_scenario_lower",
                "_scenario_tie_rank",
            ],
            errors="ignore",
        )

        # Safety check: no duplicate main-line points should remain.
        dup = out[out.duplicated(key_cols, keep=False)].copy()

        if not dup.empty:
            raise ValueError(
                "Duplicate upper-band main-line rows remain after cleaning:\n"
                + dup[
                    key_cols + [scenario_col, value_col]
                ].to_string(index=False)
            )

        return out

    def _print_main_scenario_switches(r1_main_df):
        """
        Print where the main scenario changes across RPS.
        """

        value_col = "value_per_available_kwh_year"
        rps_col = "rps_plot_value"

        switch_records = []

        for region in sorted(r1_main_df["region"].dropna().unique()):
            for group in selected_groups:
                for part in selected_participation:
                    sub = r1_main_df[
                        (r1_main_df["region"] == region)
                        & (r1_main_df["group"] == group)
                        & (r1_main_df["participation"] == part)
                    ].copy()

                    if sub.empty:
                        continue

                    sub = sub.sort_values(rps_col).reset_index(drop=True)
                    scenarios = (
                        sub["program_scenario_label"]
                        .astype(str)
                        .str.lower()
                        .unique()
                        .tolist()
                    )

                    if len(scenarios) <= 1:
                        continue

                    for _, row in sub.iterrows():
                        switch_records.append({
                            "region": region,
                            "group": group,
                            "participation": part,
                            "rps": row[rps_col],
                            "rps_display_label": row.get("rps_display_label", row[rps_col]),
                            "program_scenario_label": row["program_scenario_label"],
                            value_col: row[value_col],
                        })

        if switch_records and verbose:
            print("\nMain scenario changes across RPS:")
            print(pd.DataFrame(switch_records).to_string(index=False))

    def _add_main_scenario_change_labels(
        fig,
        axes,
        r1_main_df,
    ):
        """
        Add labels only when the upper-band main scenario changes within a line.

        Important:
            The main plotting function uses categorical x positions:
                first RPS  -> x = 0
                second RPS -> x = 1
                third RPS  -> x = 2

            Therefore this label function must NOT use raw RPS values
            such as 60, 70, 80 as x-coordinates.

        New feature:
            scenario_change_label_rps_by_region lets you specify the exact
            RPS points where labels should be added for a region.

            Example:
                {"NY": [70, 80]}
        """

        if not show_main_scenario_change_labels:
            return fig, axes

        if main_program_mode != "upper_band":
            return fig, axes

        value_col = "value_per_available_kwh_year"
        rps_col = "rps_plot_value"
        scenario_col = "program_scenario_label"

        if scenario_name_map is None:
            local_scenario_name_map = {
                "arrive": "Arrive",
                "midnight": "Midnight",
                "flex": "Flex",
                "delay": "Delay",
            }
        else:
            local_scenario_name_map = {
                str(k).lower(): v
                for k, v in scenario_name_map.items()
            }

        region_codes = [_normalize_region_local(r) for r in regions]

        if scenario_change_label_regions is None:
            label_regions = set(region_codes)
        else:
            label_regions = {
                _normalize_region_local(r)
                for r in scenario_change_label_regions
            }

        # Normalize RPS order keys.
        rps_order_map = {
            _normalize_region_local(k): list(v)
            for k, v in rps_order_by_region.items()
        }

        # Normalize region-specific label RPS map.
        if scenario_change_label_rps_by_region is None:
            label_rps_map = {}
        else:
            label_rps_map = {}
            for k, v in scenario_change_label_rps_by_region.items():
                region_key = _normalize_region_local(k)

                if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
                    label_rps_map[region_key] = list(v)
                else:
                    label_rps_map[region_key] = [v]

        if hasattr(axes, "flatten"):
            axes_flat = axes.flatten()
        else:
            axes_flat = list(axes)

        for plot_idx, region in enumerate(region_codes):
            if plot_idx >= len(axes_flat):
                continue

            if region not in label_regions:
                continue

            ax = axes_flat[plot_idx]

            # Save original x-limits before adding labels.
            original_xlim = ax.get_xlim()

            sub = r1_main_df[
                (r1_main_df["region"] == region)
                & (r1_main_df["group"] == scenario_change_label_group)
                & (r1_main_df["participation"] == scenario_change_label_participation)
            ].copy()

            if sub.empty:
                continue

            sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
            sub = sub.dropna(subset=[value_col]).copy()
            sub = sub.sort_values(rps_col).reset_index(drop=True)

            scenarios = (
                sub[scenario_col]
                .astype(str)
                .str.lower()
                .unique()
                .tolist()
            )

            # Only label if the best scenario changes inside the region,
            # unless the user explicitly requested label positions.
            user_requested_specific_rps = region in label_rps_map

            if len(scenarios) <= 1 and not user_requested_specific_rps:
                continue

            color = _get_group_color(scenario_change_label_group)
            marker = _get_marker_for_participation(scenario_change_label_participation)

            # -------------------------------------------------
            # Build categorical x-position map.
            # Example for NY:
            #   60 -> 0
            #   70 -> 1
            #   80 -> 2
            # -------------------------------------------------
            if region not in rps_order_map:
                raise ValueError(
                    f"Missing rps_order_by_region for {region}."
                )

            rps_values = list(rps_order_map[region])

            rps_to_x = {
                rps_value: idx
                for idx, rps_value in enumerate(rps_values)
            }

            sub["_x_plot"] = sub[rps_col].map(rps_to_x)

            if sub["_x_plot"].isna().any():
                bad = sub[sub["_x_plot"].isna()][
                    [rps_col, "rps_display_label", scenario_col]
                ]

                raise ValueError(
                    f"Some RPS values for {region} are not in rps_order_by_region.\n"
                    f"{bad}"
                )

            # =================================================
            # CASE A: user explicitly gives label RPS positions
            # =================================================
            if user_requested_specific_rps:
                requested_rps = label_rps_map[region]

                label_df = sub[sub[rps_col].isin(requested_rps)].copy()
                label_df = label_df.sort_values("_x_plot").reset_index(drop=True)

                if label_df.empty:
                    continue

                label_rows = []

                for _, row in label_df.iterrows():
                    scenario_lower = str(row[scenario_col]).lower()
                    scenario_label = local_scenario_name_map.get(
                        scenario_lower,
                        str(row[scenario_col]).capitalize(),
                    )

                    label_rows.append({
                        "x": float(row["_x_plot"]),
                        "y": float(row[value_col]),
                        "scenario_label": scenario_label,
                        "participation": row["participation"],
                        "rps": row[rps_col],
                        "rps_display_label": row.get("rps_display_label", row[rps_col]),
                    })

            # =================================================
            # CASE B: default behavior = label the end of each
            # scenario block
            # =================================================
            else:
                sub["_scenario_lower"] = sub[scenario_col].astype(str).str.lower()
                sub["_block_id"] = (
                    sub["_scenario_lower"] != sub["_scenario_lower"].shift()
                ).cumsum()

                label_rows = []

                for _, block in sub.groupby("_block_id"):
                    block = block.sort_values("_x_plot").copy()

                    scenario_lower = block["_scenario_lower"].iloc[-1]
                    scenario_label = local_scenario_name_map.get(
                        scenario_lower,
                        str(block[scenario_col].iloc[-1]).capitalize(),
                    )

                    row = block.iloc[-1]

                    label_rows.append({
                        "x": float(row["_x_plot"]),
                        "y": float(row[value_col]),
                        "scenario_label": scenario_label,
                        "participation": row["participation"],
                        "rps": row[rps_col],
                        "rps_display_label": row.get("rps_display_label", row[rps_col]),
                    })

            # -------------------------------------------------
            # Draw labels
            # -------------------------------------------------
            for i, item in enumerate(label_rows):
                x0 = item["x"]
                y0 = item["y"]

                x_text = x0 + scenario_change_label_x_offset

                # alternate small vertical shift to avoid overlap
                y_text = (
                    y0
                    + scenario_change_label_y_offset
                    + (0.35 * (i % 2))
                )

                tag = " (max)" if scenario_change_label_show_tag else ""

                label = (
                    f"{item['scenario_label']} "
                    f"{item['participation']}"
                    f"{tag}"
                )

                ax.plot(
                    [x0, x_text],
                    [y0, y_text],
                    color=color,
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=25,
                    clip_on=False,
                )

                ax.scatter(
                    [x0],
                    [y0],
                    color=color,
                    marker=marker,
                    s=42,
                    zorder=26,
                    clip_on=False,
                )

                ax.text(
                    x_text,
                    y_text,
                    label,
                    ha="left",
                    va="center",
                    fontsize=scenario_change_label_fontsize,
                    color=color,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        edgecolor=color,
                        linewidth=0.8,
                        alpha=0.88,
                        boxstyle="round,pad=0.20",
                    ),
                    zorder=30,
                    clip_on=False,
                )

            # Restore original x-limits so labels do not rescale the axis.
            ax.set_xlim(original_xlim)

        return fig, axes

    def _apply_font_sizes_to_r1_figure(fig, axes):
            """
            Apply font sizes after the base plot is created.

            This is needed because plot_r1_value_vs_rps_regions_with_program_band
            controls the main plot formatting internally.
            """

            if hasattr(axes, "flatten"):
                axes_flat = axes.flatten()
            else:
                axes_flat = list(axes)

            for ax in axes_flat:
                # Panel title
                ax.title.set_fontsize(panel_title_fontsize)
                ax.title.set_fontweight("bold")

                # Axis labels
                ax.xaxis.label.set_size(axis_label_fontsize)
                ax.yaxis.label.set_size(axis_label_fontsize)

                # Tick labels
                ax.tick_params(
                    axis="both",
                    labelsize=tick_label_fontsize,
                )

                for tick_label in ax.get_xticklabels():
                    tick_label.set_fontsize(tick_label_fontsize)

                for tick_label in ax.get_yticklabels():
                    tick_label.set_fontsize(tick_label_fontsize)

                # Legend, including the legend-only panel
                legend = ax.get_legend()

                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(legend_fontsize)

                    if legend.get_title() is not None:
                        legend.get_title().set_fontsize(legend_title_fontsize)
                        legend.get_title().set_fontweight("bold")

            # Figure title
            if getattr(fig, "_suptitle", None) is not None:
                fig._suptitle.set_fontsize(figure_title_fontsize)

            return fig, axes
    # =====================================================
    # Compute all data
    # =====================================================
    r1_main_df, r1_band_df, r1_all_scenarios_df = (
        compute_r1_value_vs_rps_with_program_band_regions(
            output_root=output_root,
            regions=regions,
            year=year,
            adoption_level=adoption_level,
            batt_capex=batt_capex,
            baseline_scenario_label_by_region=baseline_scenario_label_by_region,
            main_program_scenario_label_by_region=main_program_scenario_label_by_region,
            ev_adoption_df=ev_adoption_df,
            rps_order_by_region=rps_order_by_region,
            band_program_scenario_labels_by_region=band_program_scenario_labels_by_region,
            selected_groups=selected_groups,
            selected_participation=selected_participation,
            usable_battery_kwh=usable_battery_kwh,
            availability_fraction=availability_fraction,
            model_days_per_run=model_days_per_run,
            annualize_model_period=annualize_model_period,
            verbose=verbose,
        )
    )

    # =====================================================
    # Update main line if requested
    # =====================================================
    if main_program_mode == "selected":
        r1_main_df = r1_main_df.copy()
        r1_main_df["main_program_mode"] = "selected"

    elif main_program_mode == "upper_band":
        r1_main_df = _build_upper_band_main_df(
            r1_all_scenarios_df=r1_all_scenarios_df,
        )

        _print_main_scenario_switches(r1_main_df)

    else:
        raise ValueError(
            "main_program_mode must be either 'selected' or 'upper_band'."
        )

    # =====================================================
    # Update figure title if the user did not pass one
    # =====================================================
    if figure_title is None:
        if main_program_mode == "upper_band":
            main_line_text = "best charging scenario at each RPS"
        else:
            main_line_text = "selected charging scenario"

        figure_title = (
            "Value of 1 kWh of driver-provided EV flexibility across RPS targets\n"
            f"Main line = {main_line_text}, "
            f"shaded band = charging-scenario range; "
            f"year = {year}, "
            f"adoption = {str(adoption_level).capitalize()}, "
            f"battery capex = ${batt_capex}/kWh"
        )

    # =====================================================
    # Save CSVs after the main line has been updated
    # =====================================================
    if save_main_csv_path is not None:
        save_main_csv_path = Path(save_main_csv_path)
        save_main_csv_path.parent.mkdir(parents=True, exist_ok=True)
        r1_main_df.to_csv(save_main_csv_path, index=False)

    if save_band_csv_path is not None:
        save_band_csv_path = Path(save_band_csv_path)
        save_band_csv_path.parent.mkdir(parents=True, exist_ok=True)
        r1_band_df.to_csv(save_band_csv_path, index=False)

    if save_all_csv_path is not None:
        save_all_csv_path = Path(save_all_csv_path)
        save_all_csv_path.parent.mkdir(parents=True, exist_ok=True)
        r1_all_scenarios_df.to_csv(save_all_csv_path, index=False)

    # =====================================================
    # Plot
    # =====================================================
    fig, axes = plot_r1_value_vs_rps_regions_with_program_band(
        r1_main_df=r1_main_df,
        r1_band_df=r1_band_df,
        regions=regions,
        rps_order_by_region=rps_order_by_region,
        selected_groups=selected_groups,
        selected_participation=selected_participation,
        sharey=sharey,
        show_participation_band=show_participation_band,
        base_participation=base_participation,
        upper_participation=upper_participation,
        y_limits=y_limits,
        figure_title=figure_title,
        save_path=None,
    )
    # =====================================================
    # Apply font sizes to title, panel titles, labels, ticks, legend
    # =====================================================
    fig, axes = _apply_font_sizes_to_r1_figure(
        fig=fig,
        axes=axes,
    )
    # =====================================================
    # Add labels that show scenario changes
    # =====================================================
    fig, axes = _add_main_scenario_change_labels(
        fig=fig,
        axes=axes,
        r1_main_df=r1_main_df,
    )

    # =====================================================
    # Save final figure after labels are added
    # =====================================================
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return r1_main_df, r1_band_df, r1_all_scenarios_df, fig, axes

def compute_r1_value_vs_rps_regions(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption_level="mid",
    batt_capex=150,
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    ev_adoption_df=None,
    rps_order_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
):
    """
    Compute annualized R1 value for multiple regions.
    """

    if ev_adoption_df is None:
        raise ValueError("ev_adoption_df cannot be None.")

    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    if rps_order_by_region is None:
        rps_order_by_region = {}
    else:
        rps_order_by_region = {
            normalize_region(k): v
            for k, v in rps_order_by_region.items()
        }

    all_rows = []

    for region in region_codes:
        temp = compute_r1_value_vs_rps_for_region(
            output_root=output_root,
            region=region,
            year=year,
            adoption_level=adoption_level,
            batt_capex=batt_capex,
            baseline_scenario_label=baseline_map[region],
            program_scenario_label=program_map[region],
            ev_adoption_df=ev_adoption_df,
            rps_order=rps_order_by_region.get(region, None),
            selected_groups=selected_groups,
            selected_participation=selected_participation,
            usable_battery_kwh=usable_battery_kwh,
            availability_fraction=availability_fraction,
            model_days_per_run=model_days_per_run,
            annualize_model_period=annualize_model_period,
        )

        all_rows.append(temp)

    if len(all_rows) == 0:
        raise ValueError("No R1 results were created.")

    return pd.concat(all_rows, ignore_index=True)


plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "grid.color": "gray",
})


def plot_r1_value_vs_rps_from_outputs(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption_level="mid",
    batt_capex=150,
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    ev_adoption_df=None,
    rps_order_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("30%", "50%"),
    usable_battery_kwh=60.0,
    availability_fraction=1,
    model_days_per_run=7,
    annualize_model_period=True,
    sharey=True,
    y_limits=None,
    figure_title=None,
    save_path=None,
    save_csv_path=None,
):
    """
    Full R1 workflow:
        1. Load GOOD outputs
        2. Compute R1 value
        3. Plot R1 value vs RPS
    """

    r1_df = compute_r1_value_vs_rps_regions(
        output_root=output_root,
        regions=regions,
        year=year,
        adoption_level=adoption_level,
        batt_capex=batt_capex,
        baseline_scenario_label_by_region=baseline_scenario_label_by_region,
        program_scenario_label_by_region=program_scenario_label_by_region,
        ev_adoption_df=ev_adoption_df,
        rps_order_by_region=rps_order_by_region,
        selected_groups=selected_groups,
        selected_participation=selected_participation,
        usable_battery_kwh=usable_battery_kwh,
        availability_fraction=availability_fraction,
        model_days_per_run=model_days_per_run,
        annualize_model_period=annualize_model_period,
    )

    if save_csv_path is not None:
        save_csv_path = Path(save_csv_path)
        save_csv_path.parent.mkdir(parents=True, exist_ok=True)
        r1_df.to_csv(save_csv_path, index=False)

    fig, axes = plot_r1_value_vs_rps_regions(
        r1_df=r1_df,
        regions=regions,
        rps_order_by_region=rps_order_by_region,
        selected_groups=selected_groups,
        selected_participation=selected_participation,
        sharey=sharey,
        y_limits=y_limits,
        figure_title=figure_title,
        save_path=save_path,
    )

    return r1_df, fig, axes


def build_weighted_ev_adoption_df_from_state_dict(
    ev_adoption_by_state,
    state_ipm_weight_file,
    model_region_ipm_nodes=None,
    years=(2025, 2030, 2035),
    adoption_levels=("slow", "mid", "fast"),
    include_detail=True,
):
    """
    Build GOOD-region EV adoption using state-to-IPM vehicle weights.

    This avoids assigning the full EV adoption of a state to one region
    when only part of that state belongs to the region.

    Example:
        PJM EVs from NC =
            NC total EVs × NC share located in PJM IPM regions

        PJM EVs from VA =
            VA total EVs × VA share located in PJM IPM regions

    Inputs
    ------
    ev_adoption_by_state : dict
        Your EV_ADOPTION_BY_STATE dictionary.

    state_ipm_weight_file : str or Path
        CSV with at least:
            state
            ipm_region
            weight

    model_region_ipm_nodes : dict
        Mapping from model region to the IPM nodes included in that GOOD region.

    Returns
    -------
    ev_adoption_df : pd.DataFrame
        Columns:
            region
            year
            adoption
            total_evs

    ev_adoption_detail_df : pd.DataFrame
        Columns:
            region
            state
            year
            adoption
            state_evs
            region_weight
            allocated_evs
    """

    weights = pd.read_csv(state_ipm_weight_file)

    required_cols = {"state", "ipm_region", "weight"}
    missing = required_cols - set(weights.columns)

    if missing:
        raise ValueError(
            f"state_ipm_weight_file is missing columns: {missing}. "
            f"Available columns: {weights.columns.tolist()}"
        )

    weights = weights.copy()
    weights["state"] = weights["state"].astype(str).str.upper()
    weights["ipm_region"] = weights["ipm_region"].astype(str)
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce").fillna(0)

    adoption_rows = []
    detail_rows = []

    for region, ipm_nodes in model_region_ipm_nodes.items():
        ipm_nodes = set(ipm_nodes)

        region_weights = (
            weights[weights["ipm_region"].isin(ipm_nodes)]
            .groupby("state", as_index=False)["weight"]
            .sum()
            .rename(columns={"weight": "region_weight"})
        )

        for year in years:
            for adoption in adoption_levels:
                region_total_evs = 0.0

                for state, state_values in ev_adoption_by_state.items():
                    state = state.upper()
                    key = (year, adoption)

                    if key not in state_values:
                        continue

                    state_evs = float(state_values[key])

                    match = region_weights[
                        region_weights["state"] == state
                    ]

                    if match.empty:
                        region_weight = 0.0
                    else:
                        region_weight = float(match.iloc[0]["region_weight"])

                    allocated_evs = state_evs * region_weight
                    region_total_evs += allocated_evs

                    if include_detail and region_weight > 0:
                        detail_rows.append(
                            {
                                "region": region,
                                "state": state,
                                "year": year,
                                "adoption": adoption,
                                "state_evs": state_evs,
                                "region_weight": region_weight,
                                "allocated_evs": allocated_evs,
                            }
                        )

                adoption_rows.append(
                    {
                        "region": region,
                        "year": year,
                        "adoption": adoption,
                        "total_evs": region_total_evs,
                    }
                )

    ev_adoption_df = pd.DataFrame(adoption_rows)
    ev_adoption_detail_df = pd.DataFrame(detail_rows)

    return ev_adoption_df, ev_adoption_detail_df


# def load_total_costs_from_results(results_dir, scenarios=None, scenario_label=None):
#     records = []
#
#     total_files = glob.glob(os.path.join(results_dir, "**", "*_total_cost.csv"), recursive=True)
#     summary_files = glob.glob(os.path.join(results_dir, "**", "*_summary.csv"), recursive=True)
#     files = sorted(total_files + summary_files)
#
#     print(f"Found {len(files)} candidate files in {results_dir}")
#
#     seen = set()
#
#     for fp in files:
#         try:
#             temp = pd.read_csv(fp)
#             if temp.empty:
#                 continue
#
#             row = temp.iloc[0].to_dict()
#
#             fname = os.path.basename(fp)
#             folder = os.path.basename(os.path.dirname(fp))
#
#             scenario_id = int(fname.split("_")[0][1:])
#
#             # prefer total_cost over summary
#             if scenario_id in seen and fp.endswith("_summary.csv"):
#                 continue
#
#             cfg = {}
#             cfg["scenario_id"] = scenario_id
#             cfg["scenario_tag"] = row.get("scenario_tag", "")
#             cfg["objective_value"] = float(row["objective_value"])
#             cfg["file_path"] = fp
#
#             base_dir_name = os.path.basename(os.path.normpath(results_dir)).lower()
#
#             if "_slow_" in f"_{base_dir_name}_":
#                 cfg["adoption"] = "Slow"
#             elif "_fast_" in f"_{base_dir_name}_":
#                 cfg["adoption"] = "Fast"
#             elif "_mid_" in f"_{base_dir_name}_":
#                 cfg["adoption"] = "Mid"
#             else:
#                 cfg["adoption"] = "Unknown"
#
#             # NEW: Add scenario label from directory
#             if scenario_label is None:
#                 # Extract label from directory name (e.g., "flex", "midnight")
#                 parts = base_dir_name.split("_")
#                 cfg["scenario_label"] = parts[-1] if len(parts) > 0 else "default"
#             else:
#                 cfg["scenario_label"] = scenario_label
#
#             # Extract v1g_share and v2g_share from folder name
#             m_v1g = re.search(r"v1g(\d+)", folder, re.IGNORECASE)
#             m_v2g = re.search(r"v2g(\d+)", folder, re.IGNORECASE)
#
#             v1g_val = int(m_v1g.group(1)) if m_v1g else 0
#             v2g_val = int(m_v2g.group(1)) if m_v2g else 0
#
#             cfg["v1g_share"] = v1g_val / 100.0
#             cfg["v2g_share"] = v2g_val / 100.0
#
#             # Determine participation level (10% or 30%)
#             total_participation = v1g_val + v2g_val
#             if total_participation == 0:
#                 cfg["participation"] = "0%"
#                 cfg["group"] = "Base only"
#             elif total_participation == 10:
#                 cfg["participation"] = "10%"
#             elif total_participation == 30:
#                 cfg["participation"] = "30%"
#             elif total_participation == 50:
#                 cfg["participation"] = "50%"
#             else:
#                 cfg["participation"] = f"{total_participation}%"
#
#             # Determine group
#             if v1g_val > 0 and v2g_val > 0:
#                 cfg["group"] = "V1G+V2G"
#             elif v1g_val > 0:
#                 cfg["group"] = "V1G"
#             elif v2g_val > 0:
#                 cfg["group"] = "V2G"
#             else:
#                 cfg["group"] = "Base only"
#
#             # Handles both normal RPS folders like rps50/rps60/rps70
#             # and PJM folders like rps-10/rps0/rps10
#             m = re.search(r"rps(-?\d+)", folder, re.IGNORECASE)
#
#             if m:
#                 rps_value = int(m.group(1))
#                 cfg["rps_ratio"] = rps_value / 100
#             else:
#                 cfg["rps_ratio"] = None
#
#             m = re.search(r"bcapex(\d+)", folder)
#             if m:
#                 batt = int(m.group(1))
#                 cfg["batt_capex_label"] = f"${batt}/kWh"
#             else:
#                 cfg["batt_capex_label"] = None
#
#             records = [r for r in records if r["scenario_id"] != scenario_id]
#             records.append(cfg)
#             seen.add(scenario_id)
#
#         except Exception as e:
#             print(f"Could not read {fp}: {e}")
#
#     df = pd.DataFrame(records)
#
#     if df.empty:
#         raise ValueError(f"No valid total cost or summary files found in: {results_dir}")
#
#     df = df.sort_values("scenario_id").reset_index(drop=True)
#     print("Available scenario IDs:", sorted(df["scenario_id"].unique().tolist()))
#     print("Participation levels found:", df["participation"].dropna().unique())
#
#     return df

def load_cost_components_from_results(results_dir, scenario_label=None):
    records = []

    # Use total_cost/summary reader as metadata source
    meta_df = load_total_costs_from_results(results_dir, scenarios=None, scenario_label=scenario_label)
    meta_map = {
        int(row["scenario_id"]): row
        for _, row in meta_df.iterrows()
    }

    component_files = glob.glob(
        os.path.join(results_dir, "**", "*_cost_components.csv"),
        recursive=True
    )

    print(f"Found {len(component_files)} cost component files in {results_dir}")

    for fp in sorted(component_files):
        try:
            temp = pd.read_csv(fp)
            if temp.empty:
                continue

            row = temp.iloc[0].to_dict()
            fname = os.path.basename(fp)

            scenario_id = int(fname.split("_")[0][1:])

            if scenario_id not in meta_map:
                print(f"Could not find metadata match for cost component file: {fp}")
                continue

            meta = meta_map[scenario_id]

            cfg = {}
            cfg["scenario_id"] = scenario_id
            cfg["scenario_tag"] = meta.get("scenario_tag", "")
            cfg["objective_value"] = float(row.get("objective_value", meta.get("objective_value", 0.0)))

            cfg["capex_total"] = float(row.get("capex_total", 0.0))
            cfg["asset_opex_total"] = float(row.get("asset_opex_total", 0.0))
            cfg["line_opex_total"] = float(row.get("line_opex_total", 0.0))
            cfg["fixed_total"] = float(row.get("fixed_total", 0.0))
            cfg["penalty_total"] = float(row.get("penalty_total", 0.0))
            cfg["other_gap"] = float(row.get("other_gap", 0.0))
            cfg["file_path"] = fp

            # Bring all plotting metadata from total_cost/summary loader
            cfg["adoption"] = meta.get("adoption", "Unknown")
            cfg["scenario_label"] = meta.get("scenario_label", scenario_label if scenario_label is not None else "default")
            cfg["v1g_share"] = float(meta.get("v1g_share", 0.0))
            cfg["v2g_share"] = float(meta.get("v2g_share", 0.0))
            cfg["participation"] = meta.get("participation", "0%")
            cfg["group"] = meta.get("group", "Base only")
            cfg["rps_ratio"] = meta.get("rps_ratio", None)

            batt_label = meta.get("batt_capex_label", None)
            if batt_label is not None:
                m = re.search(r"(\d+)", str(batt_label))
                cfg["batt_capex_num"] = float(m.group(1)) if m else None
            else:
                cfg["batt_capex_num"] = None

            records.append(cfg)

        except Exception as e:
            print(f"Could not read {fp}: {e}")

    df = pd.DataFrame(records)

    if not df.empty:
        print("Cost component scenario IDs:", sorted(df["scenario_id"].unique().tolist()))
        if "batt_capex_num" in df.columns:
            print("Cost component battery capex values:", sorted(df["batt_capex_num"].dropna().unique().tolist()))

    return df
component_colors = {
    "capex": "#7B6D8D",
    "asset_opex": "#59A14F",
    "line_opex": "#E15759",
    "other": "#C7C7C7",
}

def plot_base_cost_by_state_rps_bcapex(
    results_dirs_by_state,
    batt_capex=150,
    states=("CA", "NY", "TX", "FL", "PJM"),
    rps_order_by_state=None,
    baseline_order=("midnight", "arrive", "delay", "flex"),
    save_path=None,
    figsize=(18, 11),
    xlim_by_state=None,
    ncols=3,
):
    """
    Plot only Base scenarios.

    Layout example for five regions:
        CA | NY  | TX
        FL | PJM | LEGEND
    """

    if rps_order_by_state is None:
        rps_order_by_state = {
            "CA": [50, 60, 70],
            "NY": [50, 60, 70],
            "TX": [0, 60, 70],
            "FL": [0, 25, 50],
            "PJM": [50, 60, 70],
        }

    nrows = int(np.ceil(len(states) / ncols))
    total_panels = nrows * ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        sharey=False
    )

    axes_grid = np.atleast_2d(axes)
    axes_flat = axes_grid.ravel()

    fig.patch.set_facecolor("white")

    # Reserve last panel for legend if there is an extra panel
    legend_ax_idx = None
    if total_panels > len(states):
        legend_ax_idx = total_panels - 1

    plot_ax_indices = [i for i in range(total_panels) if i != legend_ax_idx]

    # Only two colors now
    capex_color = "#7B6D8D"   # purple
    opex_color = "#59A14F"    # green

    bar_height = 0.55
    scenario_gap = 0.18
    rps_gap = 0.70

    all_plot_data = []

    def draw_two_part_bar(ax, y, row):
        capex = row["capex_total_mil"]
        opex = row["opex_total_mil"]

        ax.barh(
            y, capex,
            height=bar_height,
            color=capex_color,
            edgecolor="black",
            linewidth=0.5
        )

        ax.barh(
            y, opex,
            left=capex,
            height=bar_height,
            color=opex_color,
            edgecolor="black",
            linewidth=0.5
        )

    for state, ax_idx in zip(states, plot_ax_indices):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("white")

        if state not in results_dirs_by_state:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5, 0.5,
                f"{state} is missing from results_dirs_by_state",
                ha="center", va="center",
                fontsize=12, color="red",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        df_list = []

        for scenario_label, dirs in results_dirs_by_state[state].items():
            for rd in dirs:
                if not os.path.isdir(rd):
                    print(f"Skipping missing directory: {rd}")
                    continue

                try:
                    temp = load_cost_components_from_results(
                        rd,
                        scenario_label=scenario_label
                    )
                except ValueError:
                    print(f"Skipping directory with no valid results: {rd}")
                    continue

                if temp.empty:
                    continue

                df_list.append(temp)

        if len(df_list) == 0:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5, 0.5,
                f"No data found for {state}",
                ha="center", va="center",
                fontsize=13, color="red",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        df = pd.concat(df_list, ignore_index=True)

        # Keep only selected battery capex
        df = df[df["batt_capex_num"] == batt_capex].copy()

        # Keep only Base scenario
        df = df[df["group"] == "Base only"].copy()

        if df.empty:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5, 0.5,
                f"No Base-only rows for {state}\n"
                f"at ${batt_capex}/kWh",
                ha="center", va="center",
                fontsize=13, color="red",
                transform=ax.transAxes
            )
            ax.set_axis_off()
            continue

        df["rps_percent"] = (df["rps_ratio"] * 100).round().astype(int)

        # Convert to million $
        df["capex_total_mil"] = df["capex_total"] / 1e6
        df["asset_opex_total_mil"] = df["asset_opex_total"] / 1e6
        df["line_opex_total_mil"] = df["line_opex_total"] / 1e6
        df["fixed_total_mil"] = df["fixed_total"] / 1e6
        df["penalty_total_mil"] = df["penalty_total"] / 1e6
        df["other_gap_mil"] = df["other_gap"] / 1e6
        df["objective_value_mil"] = df["objective_value"] / 1e6

        # Combine everything except capex into OPEX
        df["opex_total_mil"] = (
            df["asset_opex_total_mil"]
            + df["line_opex_total_mil"]
            + df["fixed_total_mil"]
            + df["penalty_total_mil"]
            + df["other_gap_mil"]
        )

        all_plot_data.append(df.assign(state=state))

        y_positions = []
        y_labels = []
        rps_section_centers = []

        current_y = 0

        rps_order = rps_order_by_state.get(
            state,
            sorted(df["rps_percent"].dropna().unique())
        )

        for rps in rps_order:
            sub_rps = df[df["rps_percent"] == rps].copy()

            if sub_rps.empty:
                continue

            section_start_y = current_y

            for scenario_label in baseline_order:
                sub_scenario = sub_rps[
                    sub_rps["scenario_label"] == scenario_label
                ].copy()

                mid = sub_scenario[sub_scenario["adoption"] == "Mid"]
                slow = sub_scenario[sub_scenario["adoption"] == "Slow"]
                fast = sub_scenario[sub_scenario["adoption"] == "Fast"]

                if mid.empty:
                    continue

                mid_row = mid.iloc[0]
                mid_val = mid_row["objective_value_mil"]

                slow_val = slow["objective_value_mil"].iloc[0] if not slow.empty else mid_val
                fast_val = fast["objective_value_mil"].iloc[0] if not fast.empty else mid_val

                low_val = min(slow_val, mid_val, fast_val)
                high_val = max(slow_val, mid_val, fast_val)

                left_err = mid_val - low_val
                right_err = high_val - mid_val

                draw_two_part_bar(ax, current_y, mid_row)

                ax.errorbar(
                    x=mid_val,
                    y=current_y,
                    xerr=np.array([[left_err], [right_err]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=4
                )

                # No text inside bars
                y_positions.append(current_y)
                y_labels.append(scenario_label.capitalize())

                current_y += bar_height + scenario_gap

            section_end_y = current_y - scenario_gap
            section_center = (section_start_y + section_end_y) / 2
            rps_section_centers.append((rps, section_center))

            current_y += rps_gap

            ax.axhline(
                current_y - rps_gap / 2,
                color="black",
                linewidth=0.6,
                alpha=0.25
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=10, color="black")
        ax.invert_yaxis()

        ax.set_title(
            state,
            fontsize=22,
            fontweight="bold",
            color="black"
        )

        ax.set_xlabel(
            "Total system cost (Million $)",
            fontsize=19,
            color="black"
        )

        ax.tick_params(axis="x", labelsize=18, colors="black")
        ax.tick_params(axis="y", labelsize=20, colors="black")
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color("black")

        x_max = df["objective_value_mil"].max() * 1.20

        if xlim_by_state is not None and state in xlim_by_state:
            ax.set_xlim(xlim_by_state[state])
            x_text = xlim_by_state[state][1] * 0.98
        else:
            ax.set_xlim(0, x_max)
            x_text = x_max * 0.985

        for rps, y_center in rps_section_centers:

            if state == "PJM":
                pjm_label_map = {
                    -10: "Base -10",
                    0: "Base",
                    10: "Base +10",
                }
                label_text = pjm_label_map.get(rps, str(rps))
            else:
                label_text = f"RPS {rps}%"

            ax.text(
                x_text,
                y_center,
                label_text,
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.22",
                    linewidth=0.6,
                ),
            )

    # Add y-label to first column panels only
    for row_idx in range(nrows):
        first_col_idx = row_idx * ncols
        if first_col_idx < total_panels and first_col_idx != legend_ax_idx:
            ax = axes_flat[first_col_idx]
            if ax.axison:
                ax.set_ylabel(
                    "Base charging strategy",
                    fontsize=18,
                    color="black"
                )

    # Put legend inside the last panel
    if legend_ax_idx is not None:
        legend_ax = axes_flat[legend_ax_idx]
        legend_ax.axis("off")

        legend_handles = [
            Patch(facecolor=capex_color, edgecolor="black", label="CAPEX"),
            Patch(facecolor=opex_color, edgecolor="black", label="OPEX"),
        ]

        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            ncol=1,
            frameon=True,
            fontsize=20,
            title="Cost components",
            title_fontsize=20
        )

        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("black")

    fig.suptitle(
        f"Base EV charging strategies affect system cost across RPS targets\n"
        f"Battery capex = ${int(batt_capex)}/kWh",
        fontsize=20,
        color="black",
        y=0.98
    )

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.95])

    if save_path is not None:
        folder = os.path.dirname(save_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white"
        )

    if len(all_plot_data) == 0:
        return pd.DataFrame(), fig, axes_grid

    return pd.concat(all_plot_data, ignore_index=True), fig, axes_grid



def plot_base_cost_by_state_rps_bcapex(
    results_dirs_by_state,
    batt_capex=150,
    states=("CA", "NY", "TX", "FL", "PJM"),
    rps_order_by_state=None,
    baseline_order=("midnight", "arrive", "delay", "flex"),
    save_path=None,
    figsize=(18, 11),
    xlim_by_state=None,
    ncols=3,
):
    """
    Plot Base scenarios in a 2 x 3 layout.

    Layout:
        CA | NY  | TX
        FL | PJM | Legend
    """

    if rps_order_by_state is None:
        rps_order_by_state = {
            "CA": [50, 60, 70],
            "NY": [50, 60, 70],
            "TX": [0, 60, 70],
            "FL": [0, 25, 50],
            "PJM": [50, 60, 70],
        }

    # Fixed 2 x 3 layout
    nrows = 2
    ncols = 3

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=False,
        sharey=False
    )

    axes_grid = np.atleast_2d(axes)
    axes_flat = axes_grid.ravel()

    fig.patch.set_facecolor("white")

    # Explicit panel layout
    panel_position = {
        "CA": 0,
        "NY": 1,
        "TX": 2,
        "FL": 3,
        "PJM": 4,
    }

    legend_ax = axes_flat[5]

    bar_height = 0.55
    scenario_gap = 0.18
    rps_gap = 0.70

    all_plot_data = []

    def draw_stacked_bar(ax, y, row, label_components=True):
        """
        Keep all cost categories.
        Add labels inside the main bar components.
        """

        capex = row["capex_total_mil"]
        asset_opex = row["asset_opex_total_mil"]
        line_opex = row["line_opex_total_mil"]
        other = row["other_total_mil"]

        def add_segment_label(x_left, width, label, text_color="white"):
            # Avoid labels on very tiny segments
            if width < 70:
                return

            ax.text(
                x_left + width / 2,
                y,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=text_color,
                zorder=6,
            )

        left = 0.0

        # CAPEX
        ax.barh(
            y,
            capex,
            left=left,
            height=bar_height,
            color=component_colors["capex"],
            edgecolor="black",
            linewidth=0.5,
        )

        if label_components:
            add_segment_label(left, capex, "CAPEX", text_color="white")

        left += capex

        # Asset OPEX
        ax.barh(
            y,
            asset_opex,
            left=left,
            height=bar_height,
            color=component_colors["asset_opex"],
            edgecolor="black",
            linewidth=0.5,
        )

        if label_components:
            add_segment_label(left, asset_opex, "Asset OPEX", text_color="white")

        left += asset_opex

        # Line OPEX
        ax.barh(
            y,
            line_opex,
            left=left,
            height=bar_height,
            color=component_colors["line_opex"],
            edgecolor="black",
            linewidth=0.5,
        )

        # Usually this is too small, but this keeps it available
        if label_components:
            add_segment_label(left, line_opex, "Line OPEX", text_color="black")

        left += line_opex

        # Fixed / Penalty / Other
        ax.barh(
            y,
            other,
            left=left,
            height=bar_height,
            color=component_colors["other"],
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
        )

    for state in states:
        ax = axes_flat[panel_position[state]]
        ax.set_facecolor("white")

        if state not in results_dirs_by_state:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5,
                0.5,
                f"{state} missing from results_dirs_by_state",
                ha="center",
                va="center",
                fontsize=13,
                color="red",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df_list = []

        for scenario_label, dirs in results_dirs_by_state[state].items():
            for rd in dirs:

                if not os.path.isdir(rd):
                    print(f"Skipping missing directory: {rd}")
                    continue

                try:
                    temp = load_cost_components_from_results(
                        rd,
                        scenario_label=scenario_label,
                    )
                except ValueError:
                    print(f"Skipping directory with no valid results: {rd}")
                    continue

                if temp.empty:
                    continue

                df_list.append(temp)

        if len(df_list) == 0:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5,
                0.5,
                f"No valid data found for {state}",
                ha="center",
                va="center",
                fontsize=13,
                color="red",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df = pd.concat(df_list, ignore_index=True)

        # Keep only selected battery capex
        df = df[df["batt_capex_num"] == batt_capex].copy()

        # Keep only base scenario
        df = df[df["group"] == "Base only"].copy()

        if df.empty:
            ax.set_title(state, fontsize=18, fontweight="bold", color="black")
            ax.text(
                0.5,
                0.5,
                f"No Base-only rows for {state}\n"
                f"at ${batt_capex}/kWh",
                ha="center",
                va="center",
                fontsize=13,
                color="red",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        df["rps_percent"] = (df["rps_ratio"] * 100).round().astype(int)

        cost_cols = [
            "capex_total",
            "asset_opex_total",
            "line_opex_total",
            "fixed_total",
            "penalty_total",
            "other_gap",
            "objective_value",
        ]

        for col in cost_cols:
            df[col + "_mil"] = df[col] / 1e6

        df["other_total_mil"] = (
            df["fixed_total_mil"]
            + df["penalty_total_mil"]
            + df["other_gap_mil"]
        )

        df["objective_value_mil"] = df["objective_value"] / 1e6

        all_plot_data.append(df.assign(state=state))

        y_positions = []
        y_labels = []
        rps_section_centers = []

        current_y = 0

        rps_order = rps_order_by_state.get(
            state,
            sorted(df["rps_percent"].unique())
        )

        for rps in rps_order:
            sub_rps = df[df["rps_percent"] == rps].copy()

            if sub_rps.empty:
                continue

            section_start_y = current_y

            for scenario_label in baseline_order:
                sub_scenario = sub_rps[
                    sub_rps["scenario_label"] == scenario_label
                ].copy()

                mid = sub_scenario[sub_scenario["adoption"] == "Mid"]
                slow = sub_scenario[sub_scenario["adoption"] == "Slow"]
                fast = sub_scenario[sub_scenario["adoption"] == "Fast"]

                if mid.empty:
                    continue

                mid_row = mid.iloc[0]
                mid_val = mid_row["objective_value_mil"]

                slow_val = slow["objective_value_mil"].iloc[0] if not slow.empty else mid_val
                fast_val = fast["objective_value_mil"].iloc[0] if not fast.empty else mid_val

                low_val = min(slow_val, mid_val, fast_val)
                high_val = max(slow_val, mid_val, fast_val)

                left_err = mid_val - low_val
                right_err = high_val - mid_val

                draw_stacked_bar(ax, current_y, mid_row)

                ax.errorbar(
                    x=mid_val,
                    y=current_y,
                    xerr=np.array([[left_err], [right_err]]),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.5,
                    capsize=4,
                    capthick=1.5,
                    zorder=4,
                )

                # Keep scenario names only on y-axis, not inside bars
                y_positions.append(current_y)
                y_labels.append(scenario_label.capitalize())

                current_y += bar_height + scenario_gap

            section_end_y = current_y - scenario_gap
            section_center = (section_start_y + section_end_y) / 2
            rps_section_centers.append((rps, section_center))

            current_y += rps_gap

            ax.axhline(
                current_y - rps_gap / 2,
                color="black",
                linewidth=0.6,
                alpha=0.25,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=18, color="black")
        ax.invert_yaxis()

        ax.set_title(
            state,
            fontsize=22,
            fontweight="bold",
            color="black",
        )

        ax.set_xlabel(
            "Total system cost (Million $)",
            fontsize=18,
            color="black",
        )

        ax.tick_params(axis="x", labelsize=17, colors="black")
        ax.tick_params(axis="y", labelsize=18, colors="black")

        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        for spine in ax.spines.values():
            spine.set_color("black")

        x_max = df["objective_value_mil"].max() * 1.20

        if xlim_by_state is not None and state in xlim_by_state:
            ax.set_xlim(xlim_by_state[state])
            x_text = xlim_by_state[state][1] * 0.98
        else:
            ax.set_xlim(0, x_max)
            x_text = x_max * 0.985

        for rps, y_center in rps_section_centers:

            if state == "PJM":
                pjm_label_map = {
                    -10: "Base -10",
                    0: "Base",
                    10: "Base +10",
                }
                label_text = pjm_label_map.get(rps, f"Base {rps:+d}")
            else:
                label_text = f"RPS {rps}%"

            ax.text(
                x_text,
                y_center,
                label_text,
                va="center",
                ha="right",
                fontsize=10,
                fontweight="bold",
                color="black",
                bbox=dict(
                    facecolor="white",
                    edgecolor="black",
                    boxstyle="round,pad=0.22",
                    linewidth=0.6,
                ),
            )

    # Y-axis label only for first column
    axes_flat[0].set_ylabel(
        "Base charging strategy",
        fontsize=18,
        color="black",
    )

    axes_flat[3].set_ylabel(
        "Base charging strategy",
        fontsize=18,
        color="black",
    )

    # =====================================================
    # Legend in bottom-right panel
    # =====================================================
    legend_ax.axis("off")

    legend_handles = [
        Patch(
            facecolor=component_colors["capex"],
            edgecolor="black",
            label="CAPEX",
        ),
        Patch(
            facecolor=component_colors["asset_opex"],
            edgecolor="black",
            label="Asset OPEX",
        ),
        Patch(
            facecolor=component_colors["line_opex"],
            edgecolor="black",
            label="Line OPEX",
        ),
        Patch(
            facecolor=component_colors["other"],
            edgecolor="black",
            hatch="///",
            label="Fixed / Penalty / Other",
        ),
    ]

    legend = legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=1,
        frameon=True,
        fontsize=16,
        title="Cost components",
        title_fontsize=18,
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    fig.suptitle(
        f"Base EV charging strategies affect system cost across RPS targets\n"
        f"Battery capex = ${int(batt_capex)}/kWh",
        fontsize=20,
        color="black",
        y=0.98,
    )

    plt.tight_layout(rect=[0.02, 0.04, 1, 0.94])

    if save_path is not None:
        folder = os.path.dirname(save_path)

        if folder:
            os.makedirs(folder, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    if len(all_plot_data) == 0:
        return pd.DataFrame(), fig, axes_grid

    return pd.concat(all_plot_data, ignore_index=True), fig, axes_grid


def plot_peak_demand_bar_clean(
    demand,
    save_path=None,
    region_order=("NY", "CA", "FL", "TX", "PJM"),
    title="Projected Peak Electricity Demand",
    figsize=(12, 6.5),

    # Font controls
    title_fontsize=17,
    axis_label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=10,
    legend_title_fontsize=11,
    bar_label_fontsize=10,

    # Label control
    show_bar_labels=True,
    bar_label_offset_fraction=0.015,

    # Bar layout controls
    total_group_width=0.82,
    bar_width_fraction=0.92,
):
    """
    Plot projected peak demand by model region and year.

    Parameters
    ----------
    demand : dict
        Nested dictionary:
            demand[region][year] = peak demand in GW

    save_path : str or Path, optional
        Output file path.

    region_order : tuple
        Display order of regions in each year group.

    title : str
        Figure title.

    figsize : tuple
        Figure size.

    Returns
    -------
    pd.DataFrame
        Long-format demand table used for plotting.
    """

    # -----------------------------
    # Convert dictionary to dataframe
    # -----------------------------
    rows = []

    for region, years in demand.items():
        for year, value in years.items():
            rows.append(
                {
                    "Region": region,
                    "Year": int(year),
                    "Demand": float(value),
                }
            )

    df = pd.DataFrame(rows)

    year_order = sorted(df["Year"].unique())

    # -----------------------------
    # Operator display labels
    # -----------------------------
    operator_label_map = {
        "CA": "CAISO",
        "TX": "ERCOT",
        "NY": "NYISO",
        "FL": "FRCC",
        "PJM": "PJM",
    }

    # Keep only regions that exist in the demand dictionary
    region_order = [
        region for region in region_order
        if region in df["Region"].unique()
    ]

    df["Region"] = pd.Categorical(
        df["Region"],
        categories=region_order,
        ordered=True,
    )

    df["Year"] = pd.Categorical(
        df["Year"],
        categories=year_order,
        ordered=True,
    )

    df = df.sort_values(["Year", "Region"])

    # -----------------------------
    # Plot settings
    # -----------------------------
    sns.set_theme(style="whitegrid", font_scale=1.15)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    palette = {
        "CA": "#1f77b4",
        "TX": "#ff7f0e",
        "NY": "#2ca02c",
        "FL": "#9467bd",
        "PJM": "#d62728",
    }

    x = np.arange(len(year_order))

    n_regions = len(region_order)
    width = total_group_width / n_regions

    offsets = (
        np.arange(n_regions)
        - (n_regions - 1) / 2
    ) * width

    max_demand = max(df["Demand"].max(), 1)
    bar_label_offset = max_demand * bar_label_offset_fraction

    # -----------------------------
    # Draw bars
    # -----------------------------
    for i, region in enumerate(region_order):

        region_df = (
            df[df["Region"] == region]
            .set_index("Year")
            .reindex(year_order)
            .reset_index()
        )

        xpos = x + offsets[i]

        display_label = operator_label_map.get(region, region)

        bars = ax.bar(
            xpos,
            region_df["Demand"],
            width=width * bar_width_fraction,
            color=palette.get(region, "gray"),
            label=display_label,
        )

        # Add operator label above each nonzero bar
        if show_bar_labels:
            for bar, value in zip(bars, region_df["Demand"]):
                if pd.isna(value) or value <= 0:
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + bar_label_offset,
                    display_label,
                    ha="center",
                    va="bottom",
                    fontsize=bar_label_fontsize,
                    weight="bold",
                    color="black",
                )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_xticks(x)

    ax.set_xticklabels(
        year_order,
        fontsize=tick_fontsize,
        color="black",
    )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        weight="bold",
        pad=15,
        color="black",
    )

    ax.set_xlabel(
        "Year",
        fontsize=axis_label_fontsize,
        color="black",
    )

    ax.set_ylabel(
        "Peak demand (GW)",
        fontsize=axis_label_fontsize,
        color="black",
    )

    ax.tick_params(
        axis="both",
        colors="black",
        labelsize=tick_fontsize,
    )

    legend = ax.legend(
        title="Model region",
        frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        loc="upper left",
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    legend.get_title().set_color("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.ticker import AutoMinorLocator

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )

    ax.grid(
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return df

def prepare_ev_adoption_plot_df(
    ev_adoption_df,
    region_name_map=None,
):
    """
    Convert weighted regional EV adoption table into plotting format.

    Expected input columns:
        region
        year
        adoption
        total_evs

    Example input rows:
        CAISO | 2030 | slow | ...
        CAISO | 2030 | mid  | ...
        CAISO | 2030 | fast | ...
        PJM   | 2030 | fast | ...

    Output columns:
        Region
        Year
        Slow
        Mid
        Fast
        Lower Error
        Upper Error
    """

    if region_name_map is None:
        region_name_map = {
            "CAISO": "CA",
            "ERCOT": "TX",
            "NYISO": "NY",
            "FRCC": "FL",
            "PJM": "PJM",
        }

    df = ev_adoption_df.copy()

    required_cols = {"region", "year", "adoption", "total_evs"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"ev_adoption_df is missing required columns: {missing}"
        )

    df["region"] = df["region"].astype(str)
    df["Region"] = df["region"].replace(region_name_map)

    df["year"] = df["year"].astype(int)
    df["adoption"] = df["adoption"].astype(str).str.lower()

    df["total_evs"] = pd.to_numeric(df["total_evs"], errors="coerce")

    plot_df = (
        df.pivot_table(
            index=["Region", "year"],
            columns="adoption",
            values="total_evs",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={"year": "Year"})
    )

    required_adoptions = ["slow", "mid", "fast"]
    missing_adoptions = [
        col for col in required_adoptions
        if col not in plot_df.columns
    ]

    if missing_adoptions:
        raise ValueError(
            f"Missing adoption cases in ev_adoption_df: {missing_adoptions}"
        )

    plot_df = plot_df.rename(
        columns={
            "slow": "Slow",
            "mid": "Mid",
            "fast": "Fast",
        }
    )

    # Convert to millions
    plot_df["Slow"] = plot_df["Slow"] / 1e6
    plot_df["Mid"] = plot_df["Mid"] / 1e6
    plot_df["Fast"] = plot_df["Fast"] / 1e6

    plot_df["Lower Error"] = plot_df["Mid"] - plot_df["Slow"]
    plot_df["Upper Error"] = plot_df["Fast"] - plot_df["Mid"]

    return plot_df


def plot_ev_adoption_with_annotation(
    ev_adoption_df,
    save_path=None,
    region_order=("NY", "CA", "FL", "TX", "PJM"),
    annotate_region="CA",
    annotate_year=2035,
):
    """
    Plot EV adoption uncertainty by region.

    Mid adoption is the main line.
    Slow and fast adoption define the error range.

    This version is designed for model regions:
        CA, NY, TX, FL, PJM
    """

    # -----------------------------
    # Prepare dataframe
    # -----------------------------
    df = prepare_ev_adoption_plot_df(ev_adoption_df)

    available_regions = df["Region"].unique().tolist()

    region_order = [
        region for region in region_order
        if region in available_regions
    ]

    df["Region"] = pd.Categorical(
        df["Region"],
        categories=region_order,
        ordered=True,
    )

    df = df.sort_values(["Region", "Year"])

    # -----------------------------
    # Plot
    # -----------------------------
    sns.set_theme(style="whitegrid", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor="white")
    ax.set_facecolor("white")

    palette = {
        "CA": "#1f77b4",
        "TX": "#ff7f0e",
        "NY": "#2ca02c",
        "FL": "#9467bd",
        "PJM": "#d62728",
    }

    for region in region_order:
        region_df = df[df["Region"] == region].sort_values("Year")

        ax.errorbar(
            region_df["Year"],
            region_df["Mid"],
            yerr=[
                region_df["Lower Error"],
                region_df["Upper Error"],
            ],
            marker="o",
            linewidth=2.5,
            capsize=6,
            elinewidth=1.8,
            label=region,
            color=palette.get(region, "gray"),
        )

    # -----------------------------
    # Optional annotation
    # -----------------------------
    annotation_rows = df[
        (df["Region"] == annotate_region)
        & (df["Year"] == annotate_year)
    ]

    if not annotation_rows.empty:
        row = annotation_rows.iloc[0]

        x = row["Year"]
        y_slow = row["Slow"]
        y_fast = row["Fast"]

        ax.vlines(
            x=x,
            ymin=y_slow,
            ymax=y_fast,
            colors="gray",
            linestyles="dashed",
            linewidth=1.5,
            alpha=0.7,
        )

        ax.annotate(
            "Fast",
            xy=(x, y_fast),
            xytext=(x + 0.55, y_fast + 0.5),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color="black",
            ),
            fontsize=15,
            ha="left",
            color="black",
        )

        ax.annotate(
            "Slow",
            xy=(x, y_slow),
            xytext=(x + 0.55, max(y_slow - 0.8, 0)),
            arrowprops=dict(
                arrowstyle="->",
                lw=1.5,
                color="black",
            ),
            fontsize=15,
            ha="left",
            color="black",
        )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_title(
        "Projected EV Adoption with Scenario Uncertainty",
        fontsize=19,
        weight="bold",
        pad=15,
        color="black",
    )

    ax.set_xlabel("Year", fontsize=16, color="black")
    ax.set_ylabel("Number of EVs (million)", fontsize=16, color="black")

    ax.set_xticks(sorted(df["Year"].unique()))
    ax.tick_params(axis="both", colors="black", labelsize=12)

    legend = ax.legend(
        title="Region",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
        loc="upper left",
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    legend.get_title().set_color("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.minorticks_on()

    ax.grid(
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.7,
        alpha=0.5,
    )

    ax.grid(
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return df


def prepare_ev_adoption_plot_df(
    ev_adoption_df,
    region_name_map=None,
):
    """
    Convert weighted regional EV adoption table into plotting format.

    Expected input columns:
        region
        year
        adoption
        total_evs

    Output columns:
        Region
        Year
        Slow
        Mid
        Fast
        Lower Error
        Upper Error
    """

    if region_name_map is None:
        region_name_map = {
            "CAISO": "CA",
            "ERCOT": "TX",
            "NYISO": "NY",
            "FRCC": "FL",
            "PJM": "PJM",
        }

    df = ev_adoption_df.copy()

    required_cols = {"region", "year", "adoption", "total_evs"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"ev_adoption_df is missing required columns: {missing}"
        )

    df["region"] = df["region"].astype(str)
    df["Region"] = df["region"].replace(region_name_map)

    df["year"] = df["year"].astype(int)
    df["adoption"] = df["adoption"].astype(str).str.lower()

    df["total_evs"] = pd.to_numeric(df["total_evs"], errors="coerce")

    plot_df = (
        df.pivot_table(
            index=["Region", "year"],
            columns="adoption",
            values="total_evs",
            aggfunc="sum",
        )
        .reset_index()
        .rename(columns={"year": "Year"})
    )

    required_adoptions = ["slow", "mid", "fast"]
    missing_adoptions = [
        col for col in required_adoptions
        if col not in plot_df.columns
    ]

    if missing_adoptions:
        raise ValueError(
            f"Missing adoption cases in ev_adoption_df: {missing_adoptions}"
        )

    plot_df = plot_df.rename(
        columns={
            "slow": "Slow",
            "mid": "Mid",
            "fast": "Fast",
        }
    )

    # Convert to millions
    plot_df["Slow"] = plot_df["Slow"] / 1e6
    plot_df["Mid"] = plot_df["Mid"] / 1e6
    plot_df["Fast"] = plot_df["Fast"] / 1e6

    plot_df["Lower Error"] = plot_df["Mid"] - plot_df["Slow"]
    plot_df["Upper Error"] = plot_df["Fast"] - plot_df["Mid"]

    return plot_df


def plot_ev_adoption_bar_clean(
    ev_adoption_df,
    save_path=None,
    region_order=("NY", "CA", "FL", "TX", "PJM"),
    year_order=(2025, 2030, 2035),
    title="Projected EV Adoption with Scenario Uncertainty",
    figsize=(12, 6.5),

    # Font controls
    title_fontsize=17,
    axis_label_fontsize=13,
    tick_fontsize=11,
    legend_fontsize=10,
    legend_title_fontsize=11,
    bar_label_fontsize=9,

    # Label controls
    show_bar_labels=True,
    bar_label_offset_fraction=0.012,

    # Style controls
    total_group_width=0.82,
    bar_width_fraction=0.92,
    capsize=5,
):
    """
    Plot projected EV adoption by model region and year.

    For each year:
        - one bar per model region
        - bar height = Mid adoption
        - error bar spans from Slow to Fast

    Model-region display labels:
        NY  -> NYISO
        CA  -> CAISO
        FL  -> FRCC
        TX  -> ERCOT
        PJM -> PJM

    Parameters
    ----------
    ev_adoption_df : pd.DataFrame
        Input EV adoption dataframe with columns:
            region, year, adoption, total_evs

    save_path : str or Path, optional
        Output file path.

    region_order : tuple
        Display order of regions inside each year group.

    year_order : tuple
        Display order of years.

    title : str
        Figure title.

    figsize : tuple
        Figure size.

    title_fontsize : int or float
        Font size for figure title.

    axis_label_fontsize : int or float
        Font size for x and y axis labels.

    tick_fontsize : int or float
        Font size for x and y tick labels.

    legend_fontsize : int or float
        Font size for legend labels.

    legend_title_fontsize : int or float
        Font size for legend title.

    bar_label_fontsize : int or float
        Font size for labels above bars.

    show_bar_labels : bool
        If True, show model-region labels above bars.

    bar_label_offset_fraction : float
        Vertical offset for bar labels as a fraction of max Fast value.

    total_group_width : float
        Total width used by all bars within each year group.

    bar_width_fraction : float
        Fraction of available width used by each bar.

    capsize : int or float
        Error bar cap size.

    Returns
    -------
    pd.DataFrame
        Plot dataframe used for plotting.
    """

    # -----------------------------
    # Prepare dataframe
    # -----------------------------
    df = prepare_ev_adoption_plot_df(ev_adoption_df).copy()

    # Display names for the figure
    operator_label_map = {
        "CA": "CAISO",
        "TX": "ERCOT",
        "NY": "NYISO",
        "FL": "FRCC",
        "PJM": "PJM",
    }

    # Keep only requested years if they exist
    available_years = sorted(df["Year"].unique())
    year_order = [
        year for year in year_order
        if year in available_years
    ]

    if not year_order:
        raise ValueError("No requested years were found in ev_adoption_df.")

    # Keep only requested regions if they exist
    available_regions = df["Region"].unique().tolist()
    region_order = [
        region for region in region_order
        if region in available_regions
    ]

    if not region_order:
        raise ValueError("No requested regions were found in ev_adoption_df.")

    df = df[df["Year"].isin(year_order)].copy()

    df["Region"] = pd.Categorical(
        df["Region"],
        categories=region_order,
        ordered=True,
    )

    df["Year"] = pd.Categorical(
        df["Year"],
        categories=year_order,
        ordered=True,
    )

    df = df.sort_values(["Year", "Region"])

    # -----------------------------
    # Plot settings
    # -----------------------------
    sns.set_theme(style="whitegrid", font_scale=1.15)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    palette = {
        "CA": "#1f77b4",
        "TX": "#ff7f0e",
        "NY": "#2ca02c",
        "FL": "#9467bd",
        "PJM": "#d62728",
    }

    x = np.arange(len(year_order))

    n_regions = len(region_order)
    width = total_group_width / n_regions

    offsets = (
        np.arange(n_regions)
        - (n_regions - 1) / 2
    ) * width

    max_fast_value = max(df["Fast"].max(), 1)
    bar_label_offset = max_fast_value * bar_label_offset_fraction

    # -----------------------------
    # Draw bars
    # -----------------------------
    for i, region in enumerate(region_order):

        region_df = (
            df[df["Region"] == region]
            .set_index("Year")
            .reindex(year_order)
            .reset_index()
        )

        xpos = x + offsets[i]

        display_label = operator_label_map.get(region, region)

        lower_error = region_df["Lower Error"].to_numpy()
        upper_error = region_df["Upper Error"].to_numpy()

        # Prevent matplotlib issues if a scenario has accidental negative error.
        # This can happen when slow > mid or fast < mid.
        lower_error = np.maximum(lower_error, 0)
        upper_error = np.maximum(upper_error, 0)

        bars = ax.bar(
            xpos,
            region_df["Mid"],
            width=width * bar_width_fraction,
            color=palette.get(region, "gray"),
            label=display_label,
            yerr=[lower_error, upper_error],
            capsize=capsize,
            error_kw={
                "elinewidth": 1.6,
                "capthick": 1.6,
                "ecolor": "black",
            },
        )

        # Add model-region label above each bar
        if show_bar_labels:
            for bar, value, upper in zip(
                bars,
                region_df["Mid"],
                upper_error,
            ):
                if pd.isna(value) or value <= 0:
                    continue

                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + upper + bar_label_offset,
                    display_label,
                    ha="center",
                    va="bottom",
                    fontsize=bar_label_fontsize,
                    weight="bold",
                    color="black",
                )

    # -----------------------------
    # Formatting
    # -----------------------------
    ax.set_xticks(x)
    ax.set_xticklabels(
        year_order,
        fontsize=tick_fontsize,
        color="black",
    )

    ax.set_title(
        title,
        fontsize=title_fontsize,
        weight="bold",
        pad=15,
        color="black",
    )

    ax.set_xlabel(
        "Year",
        fontsize=axis_label_fontsize,
        color="black",
    )

    ax.set_ylabel(
        "Number of EVs (million)",
        fontsize=axis_label_fontsize,
        color="black",
    )

    ax.tick_params(
        axis="both",
        colors="black",
        labelsize=tick_fontsize,
    )

    legend = ax.legend(
        title="Model region",
        frameon=True,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
        loc="upper left",
    )

    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    legend.get_title().set_color("black")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.ticker import AutoMinorLocator

    ax.minorticks_on()
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.grid(
        which="major",
        axis="y",
        linestyle="--",
        linewidth=0.9,
        alpha=0.7,
    )

    ax.grid(
        which="minor",
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.4,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return df


def normalize_region(region):
    """
    Convert state-style names to GOOD model-region names.

    Examples:
        CA -> CAISO
        NY -> NYISO
        TX -> ERCOT
        FL -> FRCC
        PJM -> PJM
    """

    region = str(region).upper()

    if region not in REGION_ALIASES:
        raise ValueError(
            f"Unknown region code: {region}. "
            f"Allowed values are: {sorted(REGION_ALIASES)}"
        )

    return REGION_ALIASES[region]


def normalize_label_map(label_input, regions, map_name):
    """
    Accept either one label for all regions or a dictionary by region.

    Example:
        "flex"

    or:

        {
            "CA": "delay",
            "NY": "delay",
            "TX": "midnight",
            "FL": "delay",
            "PJM": "delay",
        }
    """

    region_codes = [normalize_region(r) for r in regions]

    if isinstance(label_input, str):
        return {
            region: label_input.lower()
            for region in region_codes
        }

    if not isinstance(label_input, dict):
        raise ValueError(
            f"{map_name} must be either a string or a dictionary."
        )

    out = {}

    for region_original in regions:
        region_code = normalize_region(region_original)

        if region_original in label_input:
            out[region_code] = str(label_input[region_original]).lower()
        elif region_code in label_input:
            out[region_code] = str(label_input[region_code]).lower()
        else:
            raise ValueError(
                f"Missing {map_name} for {region_original}. "
                f"Add either key '{region_original}' or '{region_code}'."
            )

    return out


def normalize_adoption(adoption):
    """
    Folder label should be lower-case.
    Plot label should use capitalized form.
    """

    return str(adoption).strip().lower()


def get_rps_axis_labels(
    region,
    rps_order,
):
    """
    Return region-specific RPS labels.

    Multi-state regions:
        -10 -> RPS −10 pp
          0 -> RPS baseline
         10 -> RPS +10 pp

    Single-state regions:
        50 -> 50%
    """

    region_code = normalize_region(region)

    labels = []

    for rps in rps_order:
        if region_code in MULTI_STATE_REGIONS:
            if rps == -10:
                labels.append(
                    "RPS −10 pp"
                )

            elif rps == 0:
                labels.append(
                    "RPS baseline"
                )

            elif rps == 10:
                labels.append(
                    "RPS +10 pp"
                )

            else:
                labels.append(
                    f"RPS {int(rps):+d} pp"
                )

        else:
            labels.append(
                f"{int(rps)}%"
            )

    return labels
    
def get_wastage_results_dir(
    output_root,
    region,
    year,
    adoption,
    scenario_label,
):
    """
    Find the correct GOOD result folder.

    Preferred new format:
        OUTPUT_ROOT / REGION / scenario_results_{year}_{adoption}_{REGION}_{scenario_label}

    Example:
        Output/PJM/scenario_results_2030_fast_PJM_delay
        Output/FRCC/scenario_results_2030_fast_FRCC_delay

    Backup old format:
        OUTPUT_ROOT / scenario_results_{year}_{adoption}_even_{state}_{scenario_label}
    """

    output_root = Path(output_root)
    region_code = normalize_region(region)
    adoption = normalize_adoption(adoption)
    scenario_label = str(scenario_label).lower()

    # Short state-style aliases for old folders
    short_name = {
        "CAISO": "CA",
        "NYISO": "NY",
        "ERCOT": "TX",
        "FRCC": "FL",
        "PJM": "PJM",
    }[region_code]

    candidates = [
        # New region-level folder structure
        output_root
        / region_code
        / f"scenario_results_{year}_{adoption}_{region_code}_{scenario_label}",

        # Sometimes output_root may already be the region folder
        output_root
        / f"scenario_results_{year}_{adoption}_{region_code}_{scenario_label}",

        # Old state-level folder structure
        output_root
        / f"scenario_results_{year}_{adoption}_even_{short_name}_{scenario_label}",

        # Old folder but with region code instead of state code
        output_root
        / f"scenario_results_{year}_{adoption}_even_{region_code}_{scenario_label}",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find result folder. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )

# ============================================================
# Wastage reader from solution JSON
# ============================================================

def get_scenario_id_from_folder_or_file(fp):
    folder = os.path.basename(os.path.dirname(fp))
    fname = os.path.basename(fp)

    m = re.search(r"^s(\d+)_", folder)
    if m:
        return int(m.group(1))

    m = re.search(r"^s(\d+)_", fname)
    if m:
        return int(m.group(1))

    return None


def batt_capex_to_kwh(value):
    if value < 1:
        return int(round(value * 3.6e9 / 1000))
    return int(round(value))


def get_scenario_cfg(SCENARIOS, scenario_id):
    for key in [
        str(scenario_id),
        f"s{scenario_id:02d}",
        f"s{scenario_id}",
        scenario_id,
    ]:
        if key in SCENARIOS:
            return SCENARIOS[key]

    raise KeyError(f"Scenario {scenario_id} not found in SCENARIOS")


def get_rps_from_folder_or_file(fp):
    folder = os.path.basename(os.path.dirname(fp))
    fname = os.path.basename(fp)
    text = folder + "_" + fname

    # Standard numeric RPS case
    m = re.search(r"rps(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # PJM relative RPS cases
    text_lower = text.lower()

    if "rps_minus10" in text_lower:
        return -10

    if "rps_base" in text_lower:
        return 0

    if "rps_plus10" in text_lower:
        return 10

    return None

def audit_available_wastage_policy_folders(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoptions=("slow", "mid", "fast"),
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    batt_capex=250,
):
    """
    Check which RPS values actually exist in the output folders.

    This reads folder names only. It does not use scenario JSON.
    """

    output_root = Path(output_root)

    region_codes = [normalize_region(r) for r in regions]

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    rows = []

    for region in region_codes:
        scenario_labels = sorted({
            baseline_map[region],
            program_map[region],
        })

        for adoption in adoptions:
            for scenario_label in scenario_labels:
                try:
                    results_dir = get_wastage_results_dir(
                        output_root=output_root,
                        region=region,
                        year=year,
                        adoption=adoption,
                        scenario_label=scenario_label,
                    )
                except FileNotFoundError:
                    rows.append(
                        {
                            "region": region,
                            "adoption": adoption,
                            "scenario_label": scenario_label,
                            "folder_status": "MISSING RESULT FOLDER",
                            "group": None,
                            "participation": None,
                            "batt_capex_num": None,
                            "rps_plot_value": None,
                            "folder": None,
                        }
                    )
                    continue

                solution_files = sorted(results_dir.glob("s*/*_solution.json"))

                for fp in solution_files:
                    folder = fp.parent.name

                    if not is_policy_result_folder(folder):
                        continue

                    try:
                        meta = parse_policy_folder_metadata_from_name(
                            folder_name=folder,
                            region=region,
                        )
                    except Exception:
                        continue

                    if meta["batt_capex_num"] != batt_capex:
                        continue

                    rows.append(
                        {
                            "region": region,
                            "adoption": adoption,
                            "scenario_label": scenario_label,
                            "folder_status": "FOUND",
                            "group": meta["group"],
                            "participation": meta["participation"],
                            "batt_capex_num": meta["batt_capex_num"],
                            "rps_plot_value": meta["rps_plot_value"],
                            "rps_display_label": meta["rps_display_label"],
                            "folder": folder,
                        }
                    )

    return pd.DataFrame(rows)
def metadata_from_json_or_folder(
    solution_json_path,
    SCENARIOS=None,
    region=None,
):
    """
    For wastage analysis, use the policy folder name as the source of truth.
    """

    fp = Path(solution_json_path)
    folder_name = fp.parent.name

    return parse_policy_folder_metadata_from_name(
        folder_name=folder_name,
        region=region,
    )

def audit_pjm_wastage_folders(
    output_root,
    year=2030,
    adoptions=("slow", "mid", "fast"),
    scenario_labels=("arrive", "flex"),
    batt_capex=250,
):
    """
    Audit PJM policy folders and parsed metadata.
    """

    output_root = Path(output_root)

    rows = []

    for adoption in adoptions:
        for scenario_label in scenario_labels:

            try:
                results_dir = get_wastage_results_dir(
                    output_root=output_root,
                    region="PJM",
                    year=year,
                    adoption=adoption,
                    scenario_label=scenario_label,
                )
            except FileNotFoundError:
                rows.append(
                    {
                        "adoption": adoption,
                        "scenario_label": scenario_label,
                        "status": "MISSING RESULT FOLDER",
                        "folder": None,
                    }
                )
                continue

            files = sorted(results_dir.glob("s*/*_solution.json"))

            for fp in files:
                folder = fp.parent.name

                if not is_policy_result_folder(folder):
                    continue

                try:
                    meta = parse_policy_folder_metadata_from_name(
                        folder_name=folder,
                        region="PJM",
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "adoption": adoption,
                            "scenario_label": scenario_label,
                            "status": f"PARSE FAILED: {exc}",
                            "folder": folder,
                        }
                    )
                    continue

                if meta["batt_capex_num"] != batt_capex:
                    continue

                rows.append(
                    {
                        "adoption": adoption.capitalize(),
                        "scenario_label": scenario_label,
                        "status": "FOUND",
                        "scenario_id": meta["scenario_id"],
                        "group": meta["group"],
                        "participation": meta["participation"],
                        "rps_plot_value": meta["rps_plot_value"],
                        "rps_display_label": meta["rps_display_label"],
                        "batt_capex_num": meta["batt_capex_num"],
                        "folder": folder,
                        "file_path": str(fp),
                    }
                )

    audit = pd.DataFrame(rows)

    return audit

def collapse_wastage_rows_for_plot(
    df,
    verbose=True,
):
    """
    Verify that every plotted case has exactly one result.

    Duplicate cases are treated as an error rather than averaged.
    """

    key_cols = [
        "region",
        "adoption",
        "scenario_label",
        "group",
        "participation",
        "batt_capex_num",
        "rps_plot_value",
        "rps_display_label",
    ]

    missing = [
        col
        for col in key_cols
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns needed for duplicate check: "
            f"{missing}"
        )

    duplicated = df[
        df.duplicated(
            key_cols,
            keep=False,
        )
    ].copy()

    if not duplicated.empty:
        display_cols = key_cols + [
            "model_days_in_folder",
            "wastage_gwh",
            "folder",
            "file_path",
        ]

        display_cols = [
            col
            for col in display_cols
            if col in duplicated.columns
        ]

        raise ValueError(
            "Duplicate curtailment cases remain after filtering.\n"
            "Do not average these rows. Check the folders below:\n\n"
            + duplicated[
                display_cols
            ]
            .sort_values(key_cols)
            .to_string(index=False)
        )

    return (
        df.sort_values(key_cols)
        .reset_index(drop=True)
    )

def wastage_gwh_from_solution_json(solution_json_path, time_step_h=1.0):
    """
    Sum node-level wastage from one GOOD solution JSON.

    Output unit:
        GWh over the modeled period.
    """

    with open(solution_json_path, "r") as f:
        data = json.load(f)

    total_wastage_gwh = 0.0

    for node in data.get("nodes", []):
        wastage = node.get("wastage", None)

        if wastage is None:
            continue

        if isinstance(wastage, list):
            total_wastage_gwh += sum(wastage) * time_step_h / 1e9
        else:
            total_wastage_gwh += float(wastage) * time_step_h / 1e9

    return total_wastage_gwh

def load_wastage_from_results_json(
    results_dir,
    region,
    SCENARIOS=None,  # retained for backward compatibility
    scenario_label=None,
    adoption_filter="mid",
    time_step_h=1.0,
    batt_capex_filter=None,
    rps_filter=None,
    group_filter=None,
    participation_filter=None,
    model_days_filter=None,
    verbose=True,
):
    """
    Read only the selected solution JSON files.

    Filtering is completed from the policy-folder name before opening
    the large solution JSON file.
    """

    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory does not exist:\n{results_dir}"
        )

    if rps_filter is not None:
        rps_filter = set(rps_filter)

    if group_filter is not None:
        group_filter = set(group_filter)

    if participation_filter is not None:
        participation_filter = set(participation_filter)

    # Search scenario directories, not every JSON file.
    scenario_folders = sorted(
        folder
        for folder in results_dir.glob("s*")
        if folder.is_dir()
    )

    records = []

    n_candidate_folders = len(scenario_folders)
    n_not_policy = 0
    n_wrong_duration = 0
    n_parse_failed = 0
    n_filtered_out = 0
    n_missing_solution = 0
    n_read = 0

    for scenario_folder in scenario_folders:
        folder_name = scenario_folder.name

        # Skip simple duplicate folders.
        if not is_policy_result_folder(folder_name):
            n_not_policy += 1
            continue

        # ---------------------------------------------
        # Filter model duration before reading JSON.
        # ---------------------------------------------
        folder_model_days = parse_model_days_from_folder(
            folder_name
        )

        if model_days_filter is not None:
            if folder_model_days != int(model_days_filter):
                n_wrong_duration += 1
                continue

        # Metadata comes from the policy folder name.
        try:
            meta = parse_policy_folder_metadata_from_name(
                folder_name=folder_name,
                region=region,
            )
        except Exception as exc:
            n_parse_failed += 1

            if verbose:
                print(
                    f"Could not parse metadata for "
                    f"{scenario_folder}: {exc}"
                )

            continue

        # ---------------------------------------------
        # Apply all cheap filters before JSON reading.
        # ---------------------------------------------
        if batt_capex_filter is not None:
            if (
                int(meta["batt_capex_num"])
                != int(batt_capex_filter)
            ):
                n_filtered_out += 1
                continue

        if rps_filter is not None:
            if meta["rps_plot_value"] not in rps_filter:
                n_filtered_out += 1
                continue

        if group_filter is not None:
            if meta["group"] not in group_filter:
                n_filtered_out += 1
                continue

        if participation_filter is not None:
            if meta["participation"] not in participation_filter:
                n_filtered_out += 1
                continue

        # Find the solution only after all filters pass.
        solution_json = find_solution_json_file(
            scenario_folder
        )

        if solution_json is None:
            n_missing_solution += 1

            if verbose:
                print(
                    "No solution JSON found in:\n"
                    f"{scenario_folder}"
                )

            continue

        try:
            wastage_gwh = wastage_gwh_from_solution_json(
                solution_json_path=solution_json,
                time_step_h=time_step_h,
            )

            row = meta.copy()

            row["region"] = normalize_region(region)
            row["adoption"] = str(
                adoption_filter
            ).capitalize()

            row["scenario_label"] = str(
                scenario_label
            ).lower()

            row["model_days_in_folder"] = (
                folder_model_days
            )

            row["wastage_gwh"] = wastage_gwh
            row["file_path"] = str(solution_json)
            row["folder"] = folder_name

            records.append(row)
            n_read += 1

        except Exception as exc:
            if verbose:
                print(
                    f"Could not read {solution_json}: {exc}"
                )

    if verbose:
        print(
            f"\nWastage loading summary for {results_dir.name}\n"
            f"  Candidate folders:       {n_candidate_folders}\n"
            f"  JSON files read:         {n_read}\n"
            f"  Wrong duration skipped:  {n_wrong_duration}\n"
            f"  Other filters skipped:   {n_filtered_out}\n"
            f"  Non-policy skipped:      {n_not_policy}\n"
            f"  Metadata parse failed:   {n_parse_failed}\n"
            f"  Missing solution JSON:   {n_missing_solution}"
        )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            "No valid selected solution JSON files found in:\n"
            f"{results_dir}\n\n"
            f"Requested duration: d{model_days_filter}\n"
            f"Battery capex: {batt_capex_filter}\n"
            f"RPS: {rps_filter}\n"
            f"Groups: {group_filter}\n"
            f"Participation: {participation_filter}"
        )

    return df

def load_one_region_wastage_json(
    output_root,
    region,
    year,
    SCENARIOS=None,
    batt_capex=150,
    program_scenario_label="flex",
    baseline_scenario_label="flex",
    adoption_filter="Mid",
    time_step_h=1.0,

    # New fast filters
    rps_filter=None,
    selected_participation=("30%", "50%"),
    selected_program_groups=("V1G", "V2G"),
    model_days_filter=None,
    verbose=True,
):
    """
    Load selected baseline and program wastage for one GOOD model region.

    This version avoids loading every JSON file.

    It only loads:
        - Base only from the baseline charging scenario folder
        - selected program groups from the program charging scenario folder
        - selected participation levels
        - selected RPS values
        - selected battery capex
    """

    region_code = normalize_region(region)

    df_list = []

    baseline_label = str(baseline_scenario_label).lower()
    program_label = str(program_scenario_label).lower()

    needed_labels = sorted(set([baseline_label, program_label]))

    for scenario_label in needed_labels:
        results_dir = get_wastage_results_dir(
            output_root=output_root,
            region=region_code,
            year=year,
            adoption=adoption_filter,
            scenario_label=scenario_label,
        )

        # -----------------------------
        # Decide what to load from this charging scenario folder
        # -----------------------------
        group_filter = set()
        participation_filter = set()

        if scenario_label == baseline_label:
            group_filter.add("Base only")
            participation_filter.add("0%")

        if scenario_label == program_label:
            group_filter.update(selected_program_groups)
            participation_filter.update(selected_participation)

        # If baseline and program labels are somehow neither, skip.
        if not group_filter:
            continue

        temp = load_wastage_from_results_json(
            results_dir=results_dir,
            region=region_code,
            SCENARIOS=SCENARIOS,
            scenario_label=scenario_label,
            adoption_filter=adoption_filter,
            time_step_h=time_step_h,
            batt_capex_filter=batt_capex,
            rps_filter=rps_filter,
            group_filter=group_filter,
            participation_filter=participation_filter,
            model_days_filter=model_days_filter,
            verbose=verbose,
        )

        df_list.append(temp)

    if len(df_list) == 0:
        raise ValueError(
            f"No wastage data loaded for {region_code}, "
            f"adoption={adoption_filter}."
        )

    df = pd.concat(df_list, ignore_index=True)

    # Final safety filter
    df = df[df["batt_capex_num"] == batt_capex].copy()
    df = df[df["adoption"] == str(adoption_filter).capitalize()].copy()

    base = df[
        (df["group"] == "Base only")
        & (df["scenario_label"] == baseline_label)
    ].copy()

    programs = df[
        (df["group"].isin(selected_program_groups))
        & (df["scenario_label"] == program_label)
        & (df["participation"].isin(selected_participation))
    ].copy()

    out = pd.concat([base, programs], ignore_index=True)

    return out



def parse_policy_folder_metadata_from_name(folder_name, region):
    """
    Parse GOOD policy result folder name.

    Example for normal regions:
        s30_V2G_rps70_v1g0_v2g50_bcapex250_fast_m7_d7

    Example for PJM:
        s30_V2G_rps_plus10_v1g0_v2g50_bcapex250_fast_m7_d7
        s30_V2G_rps_base_v1g0_v2g50_bcapex250_fast_m7_d7
        s30_V2G_rps_minus10_v1g0_v2g50_bcapex250_fast_m7_d7
    """

    region_code = normalize_region(region)
    name = str(folder_name)
    name_lower = name.lower()

    # -----------------------------
    # Scenario ID
    # -----------------------------
    m_id = re.search(r"^s(?P<scenario_id>\d+)_", name, re.IGNORECASE)

    if m_id is None:
        raise ValueError(f"Could not parse scenario ID from folder: {folder_name}")

    scenario_id = int(m_id.group("scenario_id"))

    # -----------------------------
    # Battery capex
    # -----------------------------
    m_capex = re.search(r"bcapex(?P<bcapex>\d+)", name_lower)

    if m_capex is None:
        raise ValueError(f"Could not parse battery capex from folder: {folder_name}")

    batt_capex_num = int(m_capex.group("bcapex"))

    # -----------------------------
    # V1G and V2G shares
    # -----------------------------
    m_v1g = re.search(r"v1g(?P<v1g>\d+)", name_lower)
    m_v2g = re.search(r"v2g(?P<v2g>\d+)", name_lower)

    if m_v1g is None or m_v2g is None:
        raise ValueError(f"Could not parse V1G/V2G shares from folder: {folder_name}")

    v1g_percent = int(m_v1g.group("v1g"))
    v2g_percent = int(m_v2g.group("v2g"))

    # -----------------------------
    # RPS
    # -----------------------------
    if region_code == "PJM":
        rps_plot_value, rps_display_label = parse_pjm_rps_value_from_text(name_lower)
    else:
        m_rps = re.search(r"rps(?P<rps>\d+)", name_lower)

        if m_rps is None:
            raise ValueError(f"Could not parse RPS from folder: {folder_name}")

        rps_plot_value = int(m_rps.group("rps"))
        rps_display_label = f"{rps_plot_value}%"

    # -----------------------------
    # Group and participation
    # -----------------------------
    if v1g_percent == 0 and v2g_percent == 0:
        group = "Base only"
        participation = "0%"
    elif v1g_percent > 0 and v2g_percent == 0:
        group = "V1G"
        participation = f"{v1g_percent}%"
    elif v1g_percent == 0 and v2g_percent > 0:
        group = "V2G"
        participation = f"{v2g_percent}%"
    else:
        group = "V1G+V2G"
        participation = f"{v1g_percent + v2g_percent}%"

    return {
        "scenario_id": scenario_id,
        "rps_plot_value": rps_plot_value,
        "rps_percent": rps_plot_value,
        "rps_display_label": rps_display_label,
        "batt_capex_num": batt_capex_num,
        "v1g_share": v1g_percent / 100.0,
        "v2g_share": v2g_percent / 100.0,
        "participation": participation,
        "group": group,
    }

def parse_pjm_rps_value_from_text(text):
    """
    Parse PJM relative RPS cases from folder names.

    PJM folder convention in your outputs:
        rps-10  -> Base - 10
        rps0    -> Base
        rps10   -> Base + 10

    Also supports:
        rps_minus10
        rps_base
        rps_plus10
    """

    text = str(text).lower()

    # -----------------------------
    # Base - 10
    # -----------------------------
    if (
        "rps-10" in text
        or "rps_minus10" in text
        or "rps_minus_10" in text
        or "rps_minus_10" in text
    ):
        return -10, "Base - 10"

    # -----------------------------
    # Base
    # -----------------------------
    if (
        "rps0" in text
        or "rps_0" in text
        or "rps_base" in text
    ):
        return 0, "Base"

    # -----------------------------
    # Base + 10
    # Important:
    # For PJM, rps10 means Base + 10, not 10% RPS.
    # -----------------------------
    if (
        "rps10" in text
        or "rps_10" in text
        or "rps+10" in text
        or "rps_plus10" in text
        or "rps_plus_10" in text
    ):
        return 10, "Base + 10"

    raise ValueError(f"Could not parse PJM RPS case from: {text}")

def get_scenarios_for_region(SCENARIOS, region):
    """
    Return the correct scenario dictionary for a region.

    This allows both old and new usage:

    Old:
        SCENARIOS = json.load(one_file)

    New:
        SCENARIOS_BY_REGION = {
            "CAISO": {...},
            "NYISO": {...},
            ...
        }
    """

    region_code = normalize_region(region)

    # New format: dictionary of dictionaries by region
    if isinstance(SCENARIOS, dict) and region_code in SCENARIOS:
        return SCENARIOS[region_code]

    # Old format: one scenario dictionary
    return SCENARIOS


def is_policy_result_folder(folder_name):
    """
    Return True only for GOOD policy result folders that contain
    RPS and battery capex metadata.

    Keep folders like:
        s30_V2G_rps70_v1g0_v2g50_bcapex250_fast_m7_d7

    Exclude simple duplicate folders like:
        s30_s30_fast_m7_d7
    """

    name = str(folder_name).lower()

    return (
        name.startswith("s")
        and "rps" in name
        and "bcapex" in name
        and "v1g" in name
        and "v2g" in name
    )


def load_scenarios_by_region(
    scenario_file_by_region=None,
    required_regions=("CAISO", "NYISO", "ERCOT", "FRCC", "PJM"),
):
    """
    Load one scenario JSON file for each GOOD model region.

    Output:
        SCENARIOS_BY_REGION["CAISO"]
        SCENARIOS_BY_REGION["NYISO"]
        SCENARIOS_BY_REGION["ERCOT"]
        SCENARIOS_BY_REGION["FRCC"]
        SCENARIOS_BY_REGION["PJM"]
    """

    scenarios_by_region = {}

    missing_files = []

    for region in required_regions:
        path = Path(scenario_file_by_region[region])

        if not path.exists():
            missing_files.append(str(path))
            continue

        with open(path, "r") as f:
            scenarios_by_region[region] = json.load(f)

        print(f"Loaded {region}: {path}")

    if missing_files:
        raise FileNotFoundError(
            "Missing scenario JSON files:\n"
            + "\n".join(missing_files)
            + "\n\nCreate these files or update SCENARIO_FILE_BY_REGION."
        )

    return scenarios_by_region

def find_shapefile(shp_folder):
    """
    Find the .shp file from either a shapefile path or a folder.
    """

    shp_folder = Path(shp_folder)

    if shp_folder.is_file() and shp_folder.suffix.lower() == ".shp":
        return shp_folder

    shp_files = list(shp_folder.glob("*.shp"))

    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in: {shp_folder}")

    return shp_files[0]

def lighten_color(color, amount=0.70):
    """
    Lighten a color by blending it with white.

    amount = 0 means original color.
    amount = 1 means white.
    """

    r, g, b = to_rgb(color)

    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount

    return to_hex((r, g, b))


def plot_ipm_regions_highlighted_focus_fast(
    shp_folder,
    region_col,
    highlight_groups=None,
    group_colors=None,
    focus_groups=None,
    title="Selected IPM Regions for Power System Study",
    figsize=(14, 8),
    default_color="#fde2e4",
    faded_default_color="#fff0f3",
    edgecolor="black",
    faded_edgecolor="#bbbbbb",
    linewidth=0.6,
    background_color="white",
    font_color="black",
    faded_font_color="#999999",
    title_fontsize=18,
    label_fontsize=7,
    label_regions=True,
    label_faded_regions=False,
    label_other_regions=False,
    legend=True,
    legend_fontsize=10,
    fade_amount=0.72,
    simplify_tolerance=None,
    print_columns=False,
    save_path=None,
):
    """
    Fast version of IPM region highlight map.

    Main speed improvement:
        - plots by color group instead of plotting one row at a time

    Optional speed improvement:
        - simplify_tolerance can simplify geometry after projection.
        - EPSG:5070 uses meters, so simplify_tolerance=3000 means about 3 km.
    """

    # -----------------------------
    # Read shapefile
    # -----------------------------
    shp_path = find_shapefile(shp_folder)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:5070")

    if simplify_tolerance is not None:
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.simplify(
            tolerance=simplify_tolerance,
            preserve_topology=True,
        )

    print("Loaded shapefile:")
    print(shp_path)

    if print_columns:
        print("\nAvailable columns:")
        print(list(gdf.columns))

    if region_col not in gdf.columns:
        raise ValueError(
            f"region_col='{region_col}' was not found. "
            f"Available columns are: {list(gdf.columns)}"
        )

    if highlight_groups is None:
        highlight_groups = {}

    if group_colors is None:
        group_colors = {
            "PJM": "#1f77b4",
            "CAISO": "#d62728",
            "ERCOT": "#ff7f0e",
            "NYISO": "#2ca02c",
            "FRCC": "#9467bd",
        }

    # -----------------------------
    # Focus mode
    # -----------------------------
    if focus_groups is None:
        focus_groups = set(highlight_groups.keys())
        use_focus_mode = False
    else:
        focus_groups = set(focus_groups)
        use_focus_mode = True

    # -----------------------------
    # Assign plotting groups/colors
    # -----------------------------
    gdf = gdf.copy()

    gdf["plot_group"] = "Other"
    gdf["is_focused"] = False
    gdf["is_highlighted"] = False

    if use_focus_mode:
        gdf["plot_color"] = faded_default_color
        gdf["plot_edgecolor"] = faded_edgecolor
    else:
        gdf["plot_color"] = default_color
        gdf["plot_edgecolor"] = edgecolor

    available_region_names = set(gdf[region_col].astype(str).unique())

    for group_name, region_list in highlight_groups.items():

        region_list = [str(r) for r in region_list]
        mask = gdf[region_col].astype(str).isin(region_list)

        sharp_color = group_colors.get(group_name, "#999999")
        faded_color = lighten_color(sharp_color, amount=fade_amount)

        is_focused_group = group_name in focus_groups

        if is_focused_group:
            plot_color = sharp_color
            plot_edgecolor = edgecolor
        else:
            plot_color = faded_color if use_focus_mode else sharp_color
            plot_edgecolor = faded_edgecolor if use_focus_mode else edgecolor

        gdf.loc[mask, "plot_group"] = group_name
        gdf.loc[mask, "plot_color"] = plot_color
        gdf.loc[mask, "plot_edgecolor"] = plot_edgecolor
        gdf.loc[mask, "is_focused"] = is_focused_group
        gdf.loc[mask, "is_highlighted"] = True

        missing_regions = sorted(set(region_list) - available_region_names)

        if missing_regions:
            print(f"\nWarning: These regions were not found for group '{group_name}':")
            print(missing_regions)

    # -----------------------------
    # Plot
    # -----------------------------
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)

    # FAST PART:
    # Plot by color/edgecolor group instead of one geometry at a time.
    plot_style_cols = ["plot_color", "plot_edgecolor"]

    for (plot_color, plot_edgecolor), sub in gdf.groupby(plot_style_cols, dropna=False):
        sub.plot(
            ax=ax,
            color=plot_color,
            edgecolor=plot_edgecolor,
            linewidth=linewidth,
        )

    # -----------------------------
    # Region labels
    # -----------------------------
    if label_regions:
        label_gdf = gdf.copy()

        if not label_other_regions:
            label_gdf = label_gdf[label_gdf["is_highlighted"]].copy()

        if use_focus_mode and not label_faded_regions:
            label_gdf = label_gdf[label_gdf["is_focused"]].copy()

        if not label_gdf.empty:
            label_gdf["label_point"] = label_gdf.geometry.representative_point()

            for _, row in label_gdf.iterrows():
                point = row["label_point"]
                label = str(row[region_col])

                if use_focus_mode:
                    text_color = font_color if row["is_focused"] else faded_font_color
                else:
                    text_color = font_color

                ax.text(
                    point.x,
                    point.y,
                    label,
                    fontsize=label_fontsize,
                    color=text_color,
                    ha="center",
                    va="center",
                )

    # -----------------------------
    # Title
    # -----------------------------
    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        color=font_color,
    )

    ax.axis("off")

    # -----------------------------
    # Legend
    # -----------------------------
    if legend:
        legend_handles = []

        for group_name in highlight_groups.keys():
            sharp_color = group_colors.get(group_name, "#999999")

            if use_focus_mode and group_name not in focus_groups:
                legend_color = lighten_color(sharp_color, amount=fade_amount)
                legend_label = f"{group_name} faded"
            else:
                legend_color = sharp_color
                legend_label = group_name

            legend_handles.append(
                Patch(
                    facecolor=legend_color,
                    edgecolor=edgecolor,
                    label=legend_label,
                )
            )

        other_color = faded_default_color if use_focus_mode else default_color

        legend_handles.append(
            Patch(
                facecolor=other_color,
                edgecolor=edgecolor,
                label="Other regions",
            )
        )

        ax.legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            fontsize=legend_fontsize,
        )

    # -----------------------------
    # Save
    # -----------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=background_color,
        )

        print(f"\nSaved figure to: {save_path}")

    plt.show()

    return fig, ax, gdf

def find_shapefile(shp_folder):
    """
    Find the .shp file from either a shapefile path or a folder.
    """

    shp_folder = Path(shp_folder)

    if shp_folder.is_file() and shp_folder.suffix.lower() == ".shp":
        return shp_folder

    shp_files = list(shp_folder.glob("*.shp"))

    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in: {shp_folder}")

    return shp_files[0]

def build_pjm_region_dc_loads(
    load_csv,
    weights_csv,
    scenario="mid",
):
    """
    Allocate state data-center loads to PJM IPM regions.

    Parameters
    ----------
    load_csv : str or Path
        CSV with state-level load ranges.
        Expected columns include: state, low_mw, mid_mw, high_mw

    weights_csv : str or Path
        CSV with state-to-region weights.
        Expected columns include:
        state, region, weight, state_in_pjm_fraction

    scenario : str
        One of: slow, mid, fast
        Also accepts: low, medium, high

    Returns
    -------
    region_loads : pd.DataFrame
        Columns:
        region, allocated_mw

    detail_df : pd.DataFrame
        State-by-region allocation details
    """

    load_df = pd.read_csv(load_csv)
    weights_df = pd.read_csv(weights_csv)

    scenario_key = str(scenario).strip().lower()

    scenario_to_col = {
        "slow": "low_mw",
        "low": "low_mw",
        "mid": "mid_mw",
        "medium": "mid_mw",
        "fast": "high_mw",
        "high": "high_mw",
    }

    if scenario_key not in scenario_to_col:
        raise ValueError(
            "scenario must be one of: 'slow', 'mid', 'fast' "
            "(or 'low', 'medium', 'high')"
        )

    load_col = scenario_to_col[scenario_key]

    required_load_cols = {"state", load_col}
    missing_load_cols = required_load_cols - set(load_df.columns)
    if missing_load_cols:
        raise ValueError(
            f"Missing columns in load CSV: {missing_load_cols}"
        )

    required_weight_cols = {"state", "region", "weight", "state_in_pjm_fraction"}
    missing_weight_cols = required_weight_cols - set(weights_df.columns)
    if missing_weight_cols:
        raise ValueError(
            f"Missing columns in weights CSV: {missing_weight_cols}"
        )

    df = weights_df.merge(
        load_df[["state", load_col]],
        on="state",
        how="left",
    )

    if df[load_col].isna().any():
        missing_states = sorted(df.loc[df[load_col].isna(), "state"].unique())
        print("Warning: some states did not find a load value:")
        print(missing_states)

    df["allocated_mw"] = (
        df[load_col].fillna(0.0)
        * df["state_in_pjm_fraction"].fillna(0.0)
        * df["weight"].fillna(0.0)
    )

    region_loads = (
        df.groupby("region", as_index=False)["allocated_mw"]
        .sum()
        .sort_values("allocated_mw", ascending=False)
    )

    return region_loads, df


def plot_ipm_group_zoomed(
    shp_folder,
    region_col,
    selected_regions,
    group_name="PJM",
    color="#1f77b4",
    region_color_map=None,
    region_label_map=None,
    title=None,
    figsize=(20, 12),
    edgecolor="black",
    linewidth=0.8,
    background_color="white",
    font_color="black",
    title_fontsize=16,
    label_fontsize=12,
    legend_fontsize=12.5,
    legend_title_fontsize=16,
    label_regions=True,
    label_with_full_name=False,
    legend=True,
    legend_on_left=True,
    legend_width_ratio=0.42,
    legend_ncol=1,
    match_case=False,
    save_path=None,
):
    """
    Plot one selected IPM group, such as PJM, as a zoomed map.

    This version uses a separate left-side legend panel, so the legend
    does not overlap the map.
    """
    PJM_REGION_LABELS = {
        "PJM_AP": "Allegheny Power Systems",
        "PJM_ATSI": "American Transmission Systems, Inc.",
        "PJM_COMD": "Commonwealth Edison Company",
        "PJM_Dom": "Dominion",
        "PJM_DOM": "Dominion",

        # IPM aggregate PJM regions
        "PJM_EMAC": "Eastern MAAC",
        "PJM_PENE": "Pennsylvania Electric Co.",
        "PJM_SMAC": "Southern MAAC",
        "PJM_WMAC": "Western MAAC",
        "PJM_West": "PJM West",
        "PJM_WEST": "PJM West",
    }

    PJM_REGION_COLORS_DARK = {
        "PJM_AP": "#1f4e79",  # dark blue
        "PJM_ATSI": "#38761d",  # dark green
        "PJM_COMD": "#b45f06",  # dark orange
        "PJM_Dom": "#674ea7",  # dark purple
        "PJM_DOM": "#674ea7",
        "PJM_EMAC": "#7f6000",  # dark gold/brown
        "PJM_PENE": "#a64d79",  # dark rose
        "PJM_SMAC": "#134f5c",  # dark teal
        "PJM_West": "#444444",  # dark gray
        "PJM_WEST": "#444444",
        "PJM_WMAC": "#783f04",  # dark brown
    }

    if region_color_map is None:
        if group_name.upper() == "PJM":
            region_color_map = PJM_REGION_COLORS_DARK.copy()
        else:
            region_color_map = {}

    if region_label_map is None:
        if group_name.upper() == "PJM":
            region_label_map = PJM_REGION_LABELS.copy()
        else:
            region_label_map = {}

    shp_path = find_shapefile(shp_folder)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:5070")

    if region_col not in gdf.columns:
        raise ValueError(
            f"region_col='{region_col}' was not found. "
            f"Available columns are: {list(gdf.columns)}"
        )

    gdf = gdf.copy()
    region_values = gdf[region_col].astype(str).str.strip()
    requested_regions = [str(r).strip() for r in selected_regions]

    if match_case:
        mask = region_values.isin(requested_regions)
        found_regions = set(region_values[mask])
        missing_regions = sorted(set(requested_regions) - found_regions)

    else:
        requested_lower = {r.lower() for r in requested_regions}
        mask = region_values.str.lower().isin(requested_lower)
        found_lower = set(region_values[mask].str.lower())
        missing_regions = sorted(
            [r for r in requested_regions if r.lower() not in found_lower]
        )

    group_gdf = gdf.loc[mask].copy()

    if group_gdf.empty:
        possible_matches = sorted(
            gdf.loc[
                region_values.str.upper().str.contains(group_name.upper(), na=False),
                region_col,
            ]
            .astype(str)
            .unique()
        )

        print(f"No regions matched for {group_name}.")
        print(f"\nPossible '{group_name}' regions in the shapefile:")
        print(possible_matches)

        raise ValueError(
            f"No selected regions were found for group_name='{group_name}'. "
            "Check region names or set match_case=False."
        )

    if missing_regions:
        print(f"\nWarning: These {group_name} regions were not found:")
        print(missing_regions)

    # -----------------------------------------------------
    # Assign color and display label
    # -----------------------------------------------------
    group_gdf["plot_region"] = group_gdf[region_col].astype(str)

    group_gdf["plot_color"] = (
        group_gdf["plot_region"]
        .map(region_color_map)
        .fillna(color)
    )

    group_gdf["display_label"] = (
        group_gdf["plot_region"]
        .map(region_label_map)
        .fillna(group_gdf["plot_region"])
    )

    # -----------------------------------------------------
    # Figure layout
    # -----------------------------------------------------
    if legend and legend_on_left:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(background_color)

        gs = fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[legend_width_ratio, 1.0],
            wspace=0.02,
        )

        legend_ax = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[0, 1])

        legend_ax.set_facecolor(background_color)
        legend_ax.axis("off")

    else:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(background_color)
        legend_ax = None

    ax.set_facecolor(background_color)

    # -----------------------------------------------------
    # Plot map
    # -----------------------------------------------------
    for plot_color, sub in group_gdf.groupby("plot_color", dropna=False):
        sub.plot(
            ax=ax,
            color=plot_color,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )

    # Zoom tightly around selected group
    minx, miny, maxx, maxy = group_gdf.total_bounds
    x_pad = (maxx - minx) * 0.06
    y_pad = (maxy - miny) * 0.06

    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)

    # -----------------------------------------------------
    # Labels
    # -----------------------------------------------------
    if label_regions:
        label_points = group_gdf.copy()
        label_points["label_point"] = label_points.geometry.representative_point()

        for _, row in label_points.iterrows():
            point = row["label_point"]

            if label_with_full_name:
                label = row["display_label"]
            else:
                label = row["plot_region"]

            ax.text(
                point.x,
                point.y,
                label,
                fontsize=label_fontsize,
                color=font_color,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                    pad=0.8,
                ),
            )

    if title is None:
        title = f"{group_name} IPM Regions"

    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontweight="bold",
        color=font_color,
        pad=10,
    )

    ax.axis("off")

    # -----------------------------------------------------
    # Legend
    # -----------------------------------------------------
    if legend:
        legend_df = (
            group_gdf[["plot_region", "display_label", "plot_color"]]
            .drop_duplicates()
            .sort_values("plot_region")
        )

        legend_handles = [
            Patch(
                facecolor=row["plot_color"],
                edgecolor=edgecolor,
                label=f"{row['plot_region']} — {row['display_label']}",
            )
            for _, row in legend_df.iterrows()
        ]

        if legend_on_left:
            leg = legend_ax.legend(
                handles=legend_handles,
                title=f"{group_name} IPM regions",
                loc="center left",
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                ncol=legend_ncol,
                borderaxespad=0.0,
            )
        else:
            leg = ax.legend(
                handles=legend_handles,
                title=f"{group_name} IPM regions",
                loc="lower left",
                frameon=True,
                fontsize=legend_fontsize,
                title_fontsize=legend_title_fontsize,
                ncol=legend_ncol,
            )

        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("black")

        for text in leg.get_texts():
            text.set_color("black")

        leg.get_title().set_color("black")
        leg.get_title().set_weight("bold")

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=background_color,
        )

        print(f"\nSaved figure to: {save_path}")

    plt.show()

    return fig, ax, group_gdf


def build_pjm_region_dc_loads(
    load_csv,
    weights_csv,
    scenario="mid",
):
    """
    Allocate state data-center loads to PJM IPM regions.

    Parameters
    ----------
    load_csv : str or Path
        CSV with state-level load ranges.
        Expected columns include: state, low_mw, mid_mw, high_mw

    weights_csv : str or Path
        CSV with state-to-region weights.
        Expected columns include:
        state, region, weight, state_in_pjm_fraction

    scenario : str
        One of: slow, mid, fast
        Also accepts: low, medium, high

    Returns
    -------
    region_loads : pd.DataFrame
        Columns:
        region, allocated_mw

    detail_df : pd.DataFrame
        State-by-region allocation details
    """

    load_df = pd.read_csv(load_csv)
    weights_df = pd.read_csv(weights_csv)

    scenario_key = str(scenario).strip().lower()

    scenario_to_col = {
        "slow": "low_mw",
        "low": "low_mw",
        "mid": "mid_mw",
        "medium": "mid_mw",
        "fast": "high_mw",
        "high": "high_mw",
    }

    if scenario_key not in scenario_to_col:
        raise ValueError(
            "scenario must be one of: 'slow', 'mid', 'fast' "
            "(or 'low', 'medium', 'high')"
        )

    load_col = scenario_to_col[scenario_key]

    required_load_cols = {"state", load_col}
    missing_load_cols = required_load_cols - set(load_df.columns)

    if missing_load_cols:
        raise ValueError(
            f"Missing columns in load CSV: {missing_load_cols}"
        )

    required_weight_cols = {
        "state",
        "region",
        "weight",
        "state_in_pjm_fraction",
    }

    missing_weight_cols = required_weight_cols - set(weights_df.columns)

    if missing_weight_cols:
        raise ValueError(
            f"Missing columns in weights CSV: {missing_weight_cols}"
        )

    df = weights_df.merge(
        load_df[["state", load_col]],
        on="state",
        how="left",
    )

    if df[load_col].isna().any():
        missing_states = sorted(df.loc[df[load_col].isna(), "state"].unique())
        print("Warning: some states did not find a load value:")
        print(missing_states)

    df["allocated_mw"] = (
        df[load_col].fillna(0.0)
        * df["state_in_pjm_fraction"].fillna(0.0)
        * df["weight"].fillna(0.0)
    )

    region_loads = (
        df.groupby("region", as_index=False)["allocated_mw"]
        .sum()
        .sort_values("allocated_mw", ascending=False)
    )

    return region_loads, df


def plot_pjm_dc_heatmap(
    shp_folder,
    region_col,
    load_csv,
    weights_csv,
    scenario=None,
    display_mode="single",   # "single" or "panel"
    panel_scenarios=("slow", "mid", "fast"),
    title=None,
    figsize=None,
    cmap="YlOrBr",
    edgecolor="black",
    linewidth=0.8,
    background_color="white",
    font_color="black",
    title_fontsize=16,
    label_fontsize=8,
    show_labels=True,
    show_load_in_labels=True,
    colorbar_label="Allocated data center load (MW)",
    save_path=None,
):
    """
    Plot a PJM heat map of allocated data-center load.

    This function stays as a heatmap:
        color = allocated_mw

    It does not assign categorical colors by IPM region.
    The categorical region colors belong to plot_ipm_group_zoomed().
    """

    # ---------------------------------------------------------
    # Scenario name standardization
    # ---------------------------------------------------------
    scenario_alias = {
        "slow": "slow",
        "low": "slow",
        "mid": "mid",
        "medium": "mid",
        "fast": "fast",
        "high": "fast",
    }

    scenario_label_map = {
        "slow": "Slow",
        "mid": "Mid",
        "fast": "Fast",
    }

    def normalize_scenario_name(x):
        x = str(x).strip().lower()

        if x not in scenario_alias:
            raise ValueError(
                "Scenario must be one of: 'slow', 'mid', 'fast' "
                "(or 'low', 'medium', 'high')."
            )

        return scenario_alias[x]

    display_mode = str(display_mode).strip().lower()

    if display_mode not in {"single", "panel"}:
        raise ValueError("display_mode must be either 'single' or 'panel'")

    # ---------------------------------------------------------
    # Load shapefile
    # ---------------------------------------------------------
    shp_path = find_shapefile(shp_folder)
    gdf = gpd.read_file(shp_path)

    if gdf.crs is not None:
        gdf = gdf.to_crs("EPSG:5070")

    if region_col not in gdf.columns:
        raise ValueError(
            f"region_col='{region_col}' was not found. "
            f"Available columns are: {list(gdf.columns)}"
        )

    # ---------------------------------------------------------
    # Helper to build one scenario gdf
    # ---------------------------------------------------------
    def build_map_for_one_scenario(one_scenario):
        one_scenario = normalize_scenario_name(one_scenario)

        region_loads, detail_df = build_pjm_region_dc_loads(
            load_csv=load_csv,
            weights_csv=weights_csv,
            scenario=one_scenario,
        )

        pjm_regions = sorted(region_loads["region"].astype(str).unique())

        pjm_gdf = gdf[
            gdf[region_col].astype(str).isin(pjm_regions)
        ].copy()

        if pjm_gdf.empty:
            raise ValueError(
                "No PJM regions from the weights CSV matched the shapefile region names."
            )

        pjm_gdf = pjm_gdf.merge(
            region_loads,
            left_on=region_col,
            right_on="region",
            how="left",
        )

        pjm_gdf["allocated_mw"] = pjm_gdf["allocated_mw"].fillna(0.0)

        return pjm_gdf, region_loads, detail_df

    # ---------------------------------------------------------
    # Helper to draw one heatmap panel
    # ---------------------------------------------------------
    def draw_heatmap_panel(
        ax,
        pjm_gdf,
        scenario_key,
        vmin,
        vmax,
    ):
        ax.set_facecolor(background_color)

        pjm_gdf.plot(
            column="allocated_mw",
            ax=ax,
            cmap=cmap,
            edgecolor=edgecolor,
            linewidth=linewidth,
            legend=False,
            vmin=vmin,
            vmax=vmax,
        )

        # Zoom tightly to PJM
        minx, miny, maxx, maxy = pjm_gdf.total_bounds
        x_pad = (maxx - minx) * 0.06
        y_pad = (maxy - miny) * 0.06

        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)

        if show_labels:
            label_points = pjm_gdf.copy()
            label_points["label_point"] = label_points.geometry.representative_point()

            for _, row in label_points.iterrows():
                point = row["label_point"]
                region_name = str(row[region_col])
                value = row["allocated_mw"]

                if show_load_in_labels:
                    label_text = f"{region_name}\n{value:,.0f} MW"
                else:
                    label_text = region_name

                ax.text(
                    point.x,
                    point.y,
                    label_text,
                    fontsize=label_fontsize,
                    color=font_color,
                    ha="center",
                    va="center",
                    bbox=dict(
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.65,
                        pad=0.8,
                    ),
                )

        ax.set_title(
            f"{scenario_label_map[scenario_key]} Scenario",
            fontsize=title_fontsize,
            fontweight="bold",
            color=font_color,
            pad=5,
        )

        ax.axis("off")

    # ---------------------------------------------------------
    # SINGLE MODE
    # ---------------------------------------------------------
    if display_mode == "single":
        if scenario is None:
            scenario = input("Choose data center scenario (slow / mid / fast): ").strip()

        scenario_key = normalize_scenario_name(scenario)
        nice_scenario = scenario_label_map[scenario_key]

        pjm_gdf, region_loads, detail_df = build_map_for_one_scenario(scenario_key)

        if figsize is None:
            figsize = (10, 8)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(background_color)

        vmin = 0
        vmax = pjm_gdf["allocated_mw"].max()

        draw_heatmap_panel(
            ax=ax,
            pjm_gdf=pjm_gdf,
            scenario_key=scenario_key,
            vmin=vmin,
            vmax=vmax,
        )

        if title is None:
            title = f"PJM Data Center Load Heat Map ({nice_scenario} Scenario)"

        fig.suptitle(
            title,
            fontsize=title_fontsize + 2,
            fontweight="bold",
            color=font_color,
            y=0.96,
        )

        # Colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            fraction=0.05,
            pad=0.05,
        )

        cbar.set_label(
            colorbar_label,
            color=font_color,
            fontsize=label_fontsize + 2,
        )

        cbar.ax.tick_params(labelsize=label_fontsize)

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
                facecolor=background_color,
            )

            print(f"Saved figure to: {save_path}")

        print(f"\nAllocated PJM regional data center loads ({nice_scenario} scenario, MW):")
        print(region_loads.to_string(index=False))

        plt.show()

        return fig, ax, pjm_gdf, region_loads, detail_df

    # ---------------------------------------------------------
    # PANEL MODE
    # ---------------------------------------------------------
    panel_scenarios = [
        normalize_scenario_name(s)
        for s in panel_scenarios
    ]

    maps = {}
    loads = {}
    details = {}

    for s in panel_scenarios:
        pjm_gdf_s, region_loads_s, detail_df_s = build_map_for_one_scenario(s)
        maps[s] = pjm_gdf_s
        loads[s] = region_loads_s
        details[s] = detail_df_s

    all_values = pd.concat(
        [maps[s]["allocated_mw"] for s in panel_scenarios],
        axis=0,
        ignore_index=True,
    )

    vmin = 0
    vmax = all_values.max()

    n_panels = len(panel_scenarios)

    if figsize is None:
        figsize = (15, 4.2)

    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=figsize,
        constrained_layout=False,
    )

    if n_panels == 1:
        axes = [axes]

    fig.patch.set_facecolor(background_color)

    # Same extent across panels
    reference_gdf = maps[panel_scenarios[0]]
    minx, miny, maxx, maxy = reference_gdf.total_bounds
    x_pad = (maxx - minx) * 0.04
    y_pad = (maxy - miny) * 0.04

    for ax, s in zip(axes, panel_scenarios):
        draw_heatmap_panel(
            ax=ax,
            pjm_gdf=maps[s],
            scenario_key=s,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_xlim(minx - x_pad, maxx + x_pad)
        ax.set_ylim(miny - y_pad, maxy + y_pad)

    if title is None:
        title = "PJM Data Center Load Heat Maps"

    fig.suptitle(
        title,
        fontsize=title_fontsize + 2,
        fontweight="bold",
        color=font_color,
        y=0.96,
    )

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.82,
        bottom=0.22,
        wspace=0.02,
    )

    # Shared bottom colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar_ax = fig.add_axes([0.28, 0.10, 0.44, 0.04])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
    )

    cbar.set_label(
        colorbar_label,
        color=font_color,
        fontsize=18,
    )

    cbar.ax.tick_params(labelsize=16)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=background_color,
        )

        print(f"Saved figure to: {save_path}")

    for s in panel_scenarios:
        print(f"\nAllocated PJM regional data center loads ({scenario_label_map[s]} scenario, MW):")
        print(loads[s].to_string(index=False))

    plt.show()

    return fig, axes, maps, loads, details

def normalize_region_or_state(x):
    """
    Accept CA, CAISO, NY, NYISO, TX, ERCOT, FL, FRCC, or PJM.
    Return GOOD model region name.
    """

    x = str(x).strip().upper()

    if x not in REGION_ALIASES:
        raise ValueError(
            f"Unknown region/state: {x}. "
            f"Allowed values are: {sorted(REGION_ALIASES)}"
        )

    return REGION_ALIASES[x]

def clean_strategy_text(x):
    """
    Normalize strategy text.
    """

    x = str(x).strip()
    x = re.sub(r"[^A-Za-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_").lower()

def normalize_strategy_label(strategy):
    """
    Normalize strategy names from JSON keys.
    """

    s = clean_strategy_text(strategy)

    strategy_map = {
        "arrive": "Arrive",
        "arrival": "Arrive",
        "arrive_charging": "Arrive",

        "midnight": "Midnight",
        "midnight_charging": "Midnight",

        "delay": "Delay",
        "delayed": "Delay",
        "delayed_charging": "Delay",

        "max_delay": "Max delay",
        "min_delay": "Min delay",

        "timed_charging": "Timed charging",
        "timed": "Timed charging",

        "load_leveling": "Load leveling",
        "loadleveling": "Load leveling",
        "even": "Load leveling",

        "flex": "Flex",
        "v1g": "V1G",
        "v2g": "V2G",
    }

    return strategy_map.get(s, str(strategy).strip())

def load_ev_charging_json(
    json_path,
    region=None,
    year=None,
    verbose=True,
    region_to_state=None,
):
    if region_to_state is None:
        raise ValueError(
            "region_to_state must be provided. "
            "Pass REGION_TO_STATE from the notebook."
        )
    """
    Load new GOOD EVDATA JSON.

    Expected structure:

        {
            "2030": {
                "fast": {
                    "CAISO": {
                        "WECC_SCE": {
                            "min_delay": {
                                "n_hours": 8760,
                                "hourly_total_kw": [...]
                            }
                        }
                    }
                }
            }
        }

    This function aggregates across all subregions inside the selected
    GOOD model region.

    Output columns:
        region
        state
        year
        adoption
        subregion
        strategy
        raw_strategy
        hour
        kw
        mw
    """

    json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("EVDATA JSON must be a dictionary.")

    selected_region = None

    if region is not None:
        selected_region = normalize_region_or_state(region)

    rows = []

    # --------------------------------------------------------
    # Loop over year
    # --------------------------------------------------------
    for year_key, year_block in data.items():

        try:
            year_int = int(year_key)
        except Exception:
            continue

        if year is not None and year_int != int(year):
            continue

        if not isinstance(year_block, dict):
            continue

        # ----------------------------------------------------
        # Loop over adoption
        # ----------------------------------------------------
        for adoption_key, adoption_block in year_block.items():

            adoption = str(adoption_key).strip().lower()

            if adoption not in {"slow", "mid", "fast"}:
                continue

            if not isinstance(adoption_block, dict):
                continue

            # ------------------------------------------------
            # Loop over model regions: CAISO, ERCOT, etc.
            # ------------------------------------------------
            for region_key, region_block in adoption_block.items():

                try:
                    model_region = normalize_region_or_state(region_key)
                except Exception:
                    continue

                if selected_region is not None and model_region != selected_region:
                    continue

                if not isinstance(region_block, dict):
                    continue

                state = region_to_state.get(model_region, model_region)
                # --------------------------------------------
                # Loop over subregions inside model region
                # Example: WECC_SCE, WEC_CALN, etc.
                # --------------------------------------------
                for subregion_key, subregion_block in region_block.items():

                    if not isinstance(subregion_block, dict):
                        continue

                    # ----------------------------------------
                    # Loop over charging strategies
                    # Example: min_delay, max_delay, etc.
                    # ----------------------------------------
                    for strategy_key, strategy_block in subregion_block.items():

                        if not isinstance(strategy_block, dict):
                            continue

                        if "hourly_total_kw" not in strategy_block:
                            continue

                        hourly_kw = strategy_block["hourly_total_kw"]

                        if hourly_kw is None:
                            continue

                        hourly_kw = np.asarray(hourly_kw, dtype=float)

                        strategy_label = normalize_strategy_label(strategy_key)

                        for hour, kw in enumerate(hourly_kw):
                            rows.append(
                                {
                                    "region": model_region,
                                    "state": state,
                                    "year": year_int,
                                    "adoption": adoption,
                                    "subregion": str(subregion_key),
                                    "strategy": strategy_label,
                                    "raw_strategy": str(strategy_key),
                                    "hour": hour,
                                    "kw": float(kw),
                                    "mw": float(kw) / 1000.0,
                                }
                            )

    df_detail = pd.DataFrame(rows)

    if df_detail.empty:
        raise ValueError(
            "No EV charging data was loaded. "
            "Check year, region, and whether the JSON contains hourly_total_kw."
        )

    # --------------------------------------------------------
    # Aggregate across all subregions
    # --------------------------------------------------------
    df_agg = (
        df_detail
        .groupby(
            [
                "region",
                "state",
                "year",
                "adoption",
                "strategy",
                "raw_strategy",
                "hour",
            ],
            as_index=False,
        )
        .agg(
            kw=("kw", "sum"),
            mw=("mw", "sum"),
            n_subregions=("subregion", "nunique"),
        )
    )

    if verbose:
        print("\nLoaded EV charging data.")
        print("Detail rows:", len(df_detail))
        print("Aggregated rows:", len(df_agg))

        print("\nAvailable region/year/adoption/strategy combinations:")
        display_cols = [
            "region",
            "state",
            "year",
            "adoption",
            "strategy",
            "raw_strategy",
            "n_subregions",
        ]

        print(
            df_agg[display_cols]
            .drop_duplicates()
            .sort_values(["region", "year", "adoption", "strategy"])
            .to_string(index=False)
        )

    return df_agg, df_detail

def inspect_ev_json_structure(
    json_path,
    region=None,
    year=None,
):
    """
    Inspect the new EVDATA JSON and show available strategies.
    """

    df_agg, df_detail = load_ev_charging_json(
        json_path=json_path,
        region=region,
        year=year,
        verbose=False,
    )

    summary = (
        df_agg[
            [
                "region",
                "state",
                "year",
                "adoption",
                "strategy",
                "raw_strategy",
                "n_subregions",
            ]
        ]
        .drop_duplicates()
        .sort_values(["region", "year", "adoption", "strategy"])
    )

    print(summary.to_string(index=False))

    return summary, df_agg, df_detail


def plot_2030_charging_strategies_with_adoption_shade(
    json_path,
    state="CAISO",
    year=2030,
    strategies=None,
    aggregate="daily_mean",
    plot_unit="kw",          # "kw" or "mw"
    save_path=None,
    figsize=(12, 6),
    fixed_ylim=None,
    verbose=True,

    # Background shaded windows
    show_time_windows=True,
    overnight_window=(0, 6),
    peak_window=(16, 21),

    # Direct label controls
    use_direct_labels=True,
    label_hour=23,               # anchor labels to this hour on the mid line
    label_x_offset=1.2,          # push text to the right
    direct_label_min_gap=1200,   # vertical spacing between labels in kW
    direct_label_fontsize=14,

    # Font controls
    title_size=20,
    label_size=18,
    tick_size=17,
    region_to_state=None,
):
    """
    Plot EV charging demand by strategy.

    - Reads hourly_total_kw
    - Aggregates across all subregions inside the selected model region
    - Uses mid adoption as main line
    - Uses slow-fast as uncertainty band
    - Uses direct ladder labels instead of a legend
    """

    plt.style.use("default")
    plt.rcdefaults()

    target_region = normalize_region_or_state(state)
    if region_to_state is None:
        raise ValueError(
            "region_to_state must be provided. "
            "Pass REGION_TO_STATE from the notebook."
        )

    target_state = region_to_state.get(target_region, target_region)

    df, df_detail = load_ev_charging_json(
        json_path=json_path,
        region=target_region,
        year=year,
        verbose=verbose,
    )

    df = df[
        (df["region"] == target_region)
        & (df["year"] == int(year))
    ].copy()

    if df.empty:
        raise ValueError(f"No data found for {target_region} in {year}.")

    plot_unit = str(plot_unit).strip().lower()

    if plot_unit not in {"kw", "mw"}:
        raise ValueError("plot_unit must be either 'kw' or 'mw'.")

    value_col = plot_unit

    if plot_unit == "kw":
        y_label = "EV charging demand (kW)"
        default_gap = 1200
    else:
        y_label = "EV charging demand (MW)"
        default_gap = 1.2

    if direct_label_min_gap is None:
        direct_label_min_gap = default_gap

    # --------------------------------------------------------
    # Strategy display names
    # --------------------------------------------------------
    strategy_display_names = {
        "Min delay": "Arrive charging",
        "Max delay": "Delayed charging",
        "Timed charging": "Timed charging",
        "Load leveling": "Load leveling",
        "Arrive": "Arrive charging",
        "Delay": "Delayed charging",
        "Midnight": "Midnight charging",
    }

    strategy_colors = {
        "Arrive charging": "#1f77b4",
        "Delayed charging": "#ff7f0e",
        "Timed charging": "#2ca02c",
        "Load leveling": "#d62728",
        "Midnight charging": "#17becf",
    }

    available_strategies = sorted(df["strategy"].dropna().unique().tolist())

    if verbose:
        print("\nAvailable strategies after region/year filter:")
        print(available_strategies)

    if strategies is None:
        preferred_order = [
            "Min delay",
            "Max delay",
            "Timed charging",
            "Load leveling",
        ]
        strategies = [s for s in preferred_order if s in available_strategies]
    else:
        strategies = [normalize_strategy_label(s) for s in strategies]

    missing_requested = [s for s in strategies if s not in available_strategies]

    if missing_requested:
        print("\nWarning: these requested strategies are not available:")
        print(missing_requested)
        print("\nAvailable strategies:")
        print(available_strategies)

    strategies = [s for s in strategies if s in available_strategies]

    if not strategies:
        raise ValueError("None of the selected strategies were found.")

    df = df[df["strategy"].isin(strategies)].copy()

    # --------------------------------------------------------
    # Aggregate time profile
    # --------------------------------------------------------
    if aggregate == "hourly":
        plot_df = (
            df.groupby(["strategy", "adoption", "hour"], as_index=False)[value_col]
            .sum()
            .rename(columns={"hour": "x", value_col: "value"})
        )
        x_label = "Hour of year"
        title_time = "Hourly"

    elif aggregate == "daily_mean":
        df["hour_of_day"] = df["hour"] % 24

        plot_df = (
            df.groupby(["strategy", "adoption", "hour_of_day"], as_index=False)[value_col]
            .mean()
            .rename(columns={"hour_of_day": "x", value_col: "value"})
        )
        x_label = "Hour of day"
        title_time = "Average daily"

    elif aggregate == "weekly_mean":
        df["hour_of_week"] = df["hour"] % 168

        plot_df = (
            df.groupby(["strategy", "adoption", "hour_of_week"], as_index=False)[value_col]
            .mean()
            .rename(columns={"hour_of_week": "x", value_col: "value"})
        )
        x_label = "Hour of week"
        title_time = "Average weekly"

    else:
        raise ValueError("aggregate must be 'hourly', 'daily_mean', or 'weekly_mean'.")

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Background shaded regions
    if show_time_windows and aggregate == "daily_mean":
        ax.axvspan(
            overnight_window[0],
            overnight_window[1],
            color="#2ca02c",
            alpha=0.10,
            zorder=0,
        )

        ax.axvspan(
            peak_window[0],
            peak_window[1],
            color="#d62728",
            alpha=0.10,
            zorder=0,
        )

        if fixed_ylim is not None:
            y_label_top = fixed_ylim[1] * 0.93
        else:
            y_label_top = plot_df["value"].max() * 1.08

        ax.text(
            np.mean(overnight_window),
            y_label_top,
            "Flexibility opportunity\n(overnight)",
            ha="center",
            va="top",
            fontsize=max(10, label_size - 4),
            color="#1b5e20",
            fontweight="bold",
            zorder=1,
        )

        ax.text(
            np.mean(peak_window),
            y_label_top,
            "Peak stress\n(4 PM–9 PM)",
            ha="center",
            va="top",
            fontsize=max(10, label_size - 4),
            color="#b71c1c",
            fontweight="bold",
            zorder=1,
        )

    line_endpoints = []

    for strategy in strategies:
        temp = plot_df[plot_df["strategy"] == strategy].copy()

        mid = temp[temp["adoption"] == "mid"].sort_values("x")
        slow = temp[temp["adoption"] == "slow"].sort_values("x")
        fast = temp[temp["adoption"] == "fast"].sort_values("x")

        if mid.empty:
            print(f"Skipped {strategy} because mid adoption is missing.")
            continue

        display_label = strategy_display_names.get(strategy, strategy)
        color = strategy_colors.get(display_label, "#333333")

        ax.plot(
            mid["x"].to_numpy(dtype=float),
            mid["value"].to_numpy(dtype=float),
            linewidth=2.8,
            color=color,
            zorder=3,
        )

        if not slow.empty and not fast.empty:
            shade = pd.merge(
                slow[["x", "value"]],
                fast[["x", "value"]],
                on="x",
                suffixes=("_slow", "_fast"),
            )

            shade["lower"] = shade[["value_slow", "value_fast"]].min(axis=1)
            shade["upper"] = shade[["value_slow", "value_fast"]].max(axis=1)
            shade = shade.sort_values("x")

            ax.fill_between(
                shade["x"].to_numpy(dtype=float),
                shade["lower"].to_numpy(dtype=float),
                shade["upper"].to_numpy(dtype=float),
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=2,
            )

        # store anchor point for direct label
        anchor_row = mid[mid["x"] == label_hour]

        if anchor_row.empty:
            anchor_row = mid.iloc[[-1]]

        x_anchor = float(anchor_row["x"].iloc[0])
        y_anchor = float(anchor_row["value"].iloc[0])

        line_endpoints.append(
            {
                "strategy": strategy,
                "display_label": display_label,
                "color": color,
                "x_anchor": x_anchor,
                "y_anchor": y_anchor,
            }
        )

    # --------------------------------------------------------
    # Direct ladder labels
    # --------------------------------------------------------
    if use_direct_labels and line_endpoints:
        label_df = pd.DataFrame(line_endpoints).sort_values("y_anchor").reset_index(drop=True)

        # stagger vertically
        adjusted_y = []
        prev_y = None

        for y in label_df["y_anchor"]:
            if prev_y is None:
                y_new = y
            else:
                y_new = max(y, prev_y + direct_label_min_gap)
            adjusted_y.append(y_new)
            prev_y = y_new

        label_df["y_text"] = adjusted_y
        label_df["x_text"] = label_df["x_anchor"] + label_x_offset

        # if fixed ylim exists, keep labels inside it
        if fixed_ylim is not None:
            y_min, y_max = fixed_ylim
            top_margin = 0.04 * (y_max - y_min)

            if label_df["y_text"].max() > y_max - top_margin:
                overflow = label_df["y_text"].max() - (y_max - top_margin)
                label_df["y_text"] = label_df["y_text"] - overflow

        for _, row in label_df.iterrows():
            ax.annotate(
                row["display_label"],
                xy=(row["x_anchor"], row["y_anchor"]),
                xytext=(row["x_text"], row["y_text"]),
                fontsize=direct_label_fontsize,
                color=row["color"],
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    color=row["color"],
                    lw=1.8,
                    shrinkA=0,
                    shrinkB=0,
                ),
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.8,
                    pad=0.2,
                ),
                zorder=4,
            )

        # extend x-limit so labels fit
        current_xlim = ax.get_xlim()
        ax.set_xlim(current_xlim[0], max(current_xlim[1], label_df["x_text"].max() + 1.0))

    display_name = target_state if target_state != "PJM" else "PJM"

    ax.set_title(
        f"{title_time} EV charging demand by strategy in {display_name}, {year}",
        fontsize=title_size,
        color="black",
        pad=10,
    )

    ax.set_xlabel(x_label, fontsize=label_size, color="black")
    ax.set_ylabel(y_label, fontsize=label_size, color="black")

    ax.tick_params(axis="both", labelsize=tick_size, colors="black")

    for spine in ax.spines.values():
        spine.set_color("black")

    ax.grid(True, color="gray", alpha=0.28, linestyle="--")
    ax.set_axisbelow(True)

    if aggregate == "daily_mean":
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlim(0, max(24, label_hour + label_x_offset + 2))

    if aggregate == "weekly_mean":
        ax.set_xticks(range(0, 169, 24))

    if fixed_ylim is not None:
        ax.set_ylim(fixed_ylim)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return plot_df, df, df_detail


def normalize_region(region):
    """
    Convert old state labels to GOOD model region labels.

    CA -> CAISO
    NY -> NYISO
    TX -> ERCOT
    FL -> FRCC
    PJM -> PJM
    """

    region = str(region).strip().upper()

    if region not in REGION_ALIASES:
        raise ValueError(
            f"Unknown region: {region}. "
            f"Allowed values are: {sorted(REGION_ALIASES)}"
        )

    return REGION_ALIASES[region]


def get_label_for_region(label_input, region):
    """
    Accept either one label for all regions or a dictionary by region.

    Examples
    --------
    One label for all regions:
        "delay"

    Dictionary with short region names:
        {"CA": "delay", "NY": "arrive", "TX": "midnight"}

    Dictionary with GOOD region names:
        {"CAISO": "delay", "NYISO": "arrive", "ERCOT": "midnight"}
    """

    region_code = normalize_region(region)

    if isinstance(label_input, str):
        return label_input

    if not isinstance(label_input, dict):
        raise ValueError(
            "label_input must be either a string or a dictionary."
        )

    # Try exact user key first: CA, NY, TX, FL, PJM
    if region in label_input:
        return label_input[region]

    # Try normalized GOOD region key: CAISO, NYISO, ERCOT, FRCC, PJM
    if region_code in label_input:
        return label_input[region_code]

    raise ValueError(
        f"Missing label for region={region}. "
        f"Add either key '{region}' or '{region_code}'."
    )


def normalize_label_map(label_input, regions, map_name):
    """
    Convert scenario label input into a dictionary keyed by GOOD region.

    Output example:
        {
            "CAISO": "delay",
            "NYISO": "arrive",
            "ERCOT": "midnight",
        }
    """

    if label_input is None:
        raise ValueError(f"{map_name} cannot be None.")

    out = {}

    for region in regions:
        region_code = normalize_region(region)
        out[region_code] = str(
            get_label_for_region(label_input, region)
        ).strip().lower()

    return out


def get_rps_display_label(region, rps):
    """
    Format RPS label for normal regions and PJM relative cases.
    """

    region_code = normalize_region(region)

    if region_code == "PJM":
        if rps == -10:
            return "Base - 10"
        if rps == 0:
            return "Base"
        if rps == 10:
            return "Base + 10"
        return f"Base {int(rps):+d}"

    return f"RPS {int(rps)}%"


# ============================================================
# Folder helpers
# ============================================================

def get_line_results_dir(
    output_root,
    region,
    year,
    adoption,
    scenario_label,
    region_to_state=None,
):
    """
    Find the GOOD result directory.

    Search order
    ------------
    1. Current canonical region folder:
       OUTPUT_ROOT / REGION /
       scenario_results_{year}_{adoption}_{REGION}_{scenario_label}

    2. Current flat region folder:
       OUTPUT_ROOT /
       scenario_results_{year}_{adoption}_{REGION}_{scenario_label}

    3. Old state-based folder, when region_to_state is provided:
       OUTPUT_ROOT /
       scenario_results_{year}_{adoption}_even_{STATE}_{scenario_label}

    4. Old region-based even folder:
       OUTPUT_ROOT /
       scenario_results_{year}_{adoption}_even_{REGION}_{scenario_label}
    """

    output_root = Path(output_root)
    region_code = normalize_region(region)

    adoption = str(adoption).strip().lower()
    scenario_label = str(
        scenario_label
    ).strip().lower()

    folder_name = (
        f"scenario_results_{year}_"
        f"{adoption}_"
        f"{region_code}_"
        f"{scenario_label}"
    )

    candidates = [
        # Current preferred structure:
        # Output/CAISO/scenario_results_2030_mid_CAISO_midnight
        output_root
        / region_code
        / folder_name,

        # Current flat structure:
        # Output/scenario_results_2030_mid_CAISO_midnight
        output_root
        / folder_name,
    ]

    # Add the old state-based path only when the mapping
    # was actually supplied.
    if region_to_state is not None:
        short_name = region_to_state.get(
            region_code,
            None,
        )

        if short_name is not None:
            candidates.append(
                output_root
                / (
                    f"scenario_results_{year}_"
                    f"{adoption}_even_"
                    f"{short_name}_"
                    f"{scenario_label}"
                )
            )

    # Old fallback using the model-region name.
    candidates.append(
        output_root
        / (
            f"scenario_results_{year}_"
            f"{adoption}_even_"
            f"{region_code}_"
            f"{scenario_label}"
        )
    )

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find the requested result folder.\n\n"
        f"Region: {region_code}\n"
        f"Year: {year}\n"
        f"Adoption: {adoption}\n"
        f"Scenario label: {scenario_label}\n\n"
        "Paths checked:\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )


def is_policy_result_folder(folder_name):
    """
    Keep only GOOD policy folders with RPS, battery capex, V1G, and V2G metadata.
    """

    name = str(folder_name).lower()

    return (
        name.startswith("s")
        and "rps" in name
        and "bcapex" in name
        and "v1g" in name
        and "v2g" in name
    )


def find_policy_folder_name_from_path(path):
    """
    Walk upward from a file/folder path and find the policy folder name.
    """

    path = Path(path)

    candidates = [path.name] + [p.name for p in path.parents]

    for name in candidates:
        if is_policy_result_folder(name):
            return name

    return None


def find_solution_json_in_folder(folder):
    """
    Find solution JSON inside one scenario folder.
    """

    folder = Path(folder)

    files = sorted(folder.glob("*_solution.json"))

    if files:
        return files[0]

    files = sorted(folder.glob("*solution*.json"))

    if files:
        return files[0]

    return None


def parse_pjm_rps_value_from_text(text):
    """
    Parse PJM relative RPS cases.

    Supports:
        rps-10
        rps_minus10
        rps_minus_10
        rps0
        rps_base
        rps10
        rps+10
        rps_plus10
        rps_plus_10
    """

    text = str(text).lower()

    if (
        "rps-10" in text
        or "rps_minus10" in text
        or "rps_minus_10" in text
    ):
        return -10, "Base - 10"

    if (
        "rps0" in text
        or "rps_0" in text
        or "rps_base" in text
    ):
        return 0, "Base"

    if (
        "rps10" in text
        or "rps_10" in text
        or "rps+10" in text
        or "rps_plus10" in text
        or "rps_plus_10" in text
    ):
        return 10, "Base + 10"

    m = re.search(r"rps(?P<rps>-?\d+)", text)

    if m:
        value = int(m.group("rps"))
        return value, get_rps_display_label("PJM", value)

    raise ValueError(f"Could not parse PJM RPS case from: {text}")


def normalize_participation_label(value):
    """
    Convert 0.5, 50, '50%', etc. to '50%'.
    """

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.endswith("%"):
        try:
            return f"{int(round(float(text.replace('%', ''))))}%"
        except Exception:
            return text

    val = pd.to_numeric(text, errors="coerce")

    if pd.isna(val):
        return text

    if val <= 1:
        return f"{int(round(val * 100))}%"

    return f"{int(round(val))}%"


def batt_capex_value_to_kwh(value):
    """
    Convert battery capex value to $/kWh style integer.

    Handles:
        150
        "$150/kWh"
        "bcapex150"
        small model unit values from cost components
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()

    m = re.search(r"bcapex(?P<capex>\d+)", text)

    if m:
        return int(m.group("capex"))

    text = (
        text.replace("$", "")
        .replace("/kwh", "")
        .replace("kwh", "")
        .replace(",", "")
        .strip()
    )

    val = pd.to_numeric(text, errors="coerce")

    if pd.isna(val):
        return None

    val = float(val)

    if val < 1:
        # Common GOOD internal unit conversion used elsewhere:
        # $/J to $/kWh.
        return int(round(val * 3.6e9 / 1000))

    return int(round(val))


def parse_policy_folder_metadata_from_name(folder_name, region):
    """
    Parse GOOD policy result folder name.

    Examples:
        s01_Base_only_rps50_v1g0_v2g0_bcapex150_mid_m7_d7
        s07_V2G_rps60_v1g0_v2g30_bcapex150_mid_m7_d7
        s30_V2G_rps_plus10_v1g0_v2g50_bcapex150_fast_m7_d7
    """

    region_code = normalize_region(region)

    name = str(folder_name)
    name_lower = name.lower()

    m_id = re.search(r"^s(?P<scenario_id>\d+)_", name, re.IGNORECASE)

    if m_id is None:
        raise ValueError(f"Could not parse scenario ID from folder: {folder_name}")

    scenario_id = int(m_id.group("scenario_id"))

    m_capex = re.search(r"bcapex(?P<bcapex>\d+)", name_lower)

    if m_capex is None:
        raise ValueError(f"Could not parse battery capex from folder: {folder_name}")

    batt_capex = int(m_capex.group("bcapex"))

    m_v1g = re.search(r"v1g(?P<v1g>\d+)", name_lower)
    m_v2g = re.search(r"v2g(?P<v2g>\d+)", name_lower)

    v1g_val = int(m_v1g.group("v1g")) if m_v1g else 0
    v2g_val = int(m_v2g.group("v2g")) if m_v2g else 0

    if "base_only" in name_lower or (v1g_val == 0 and v2g_val == 0):
        group = "Base only"
        participation = "0%"
    elif v1g_val > 0 and v2g_val > 0:
        group = "V1G+V2G"
        participation = f"{v1g_val + v2g_val}%"
    elif v1g_val > 0:
        group = "V1G"
        participation = f"{v1g_val}%"
    elif v2g_val > 0:
        group = "V2G"
        participation = f"{v2g_val}%"
    else:
        group = "Base only"
        participation = "0%"

    if region_code == "PJM":
        rps, rps_display_label = parse_pjm_rps_value_from_text(name_lower)
    else:
        m_rps = re.search(r"rps(?P<rps>\d+)", name_lower)

        if m_rps is None:
            raise ValueError(f"Could not parse RPS from folder: {folder_name}")

        rps = int(m_rps.group("rps"))
        rps_display_label = f"RPS {rps}%"

    return {
    "scenario_id": scenario_id,
    "region": region_code,

    # RPS names used by different plotting functions
    "rps": rps,
    "rps_plot_value": rps,
    "rps_percent": (
        None
        if region_code == "PJM"
        else rps
    ),
    "rps_ratio": (
        None
        if region_code == "PJM"
        else rps / 100.0
    ),
    "rps_display_label": rps_display_label,

    # Battery CAPEX names used by different plotting functions
    "batt_capex": batt_capex,
    "batt_capex_num": batt_capex,
    "batt_capex_label": f"${batt_capex}/kWh",

    # Participation metadata
    "v1g_percent": v1g_val,
    "v2g_percent": v2g_val,
    "v1g_share": v1g_val / 100.0,
    "v2g_share": v2g_val / 100.0,
    "participation": participation,
    "group": group,

    "folder_name": folder_name,
}


# ============================================================
# Line CAPEX readers
# ============================================================

def build_line_capex_cost_lookup(graph_json_path):
    """
    Read line capex cost from graph JSON.
    """

    graph_json_path = Path(graph_json_path)

    if not graph_json_path.exists():
        raise FileNotFoundError(f"Graph JSON not found: {graph_json_path}")

    with open(graph_json_path, "r") as f:
        graph = json.load(f)

    lookup = {}

    for edge in graph.get("edges", []):
        for line_id, line_data in edge.get("lines", {}).items():
            lookup[str(line_id)] = line_data.get("capex_cost", 0.0)

    return lookup


def line_expansion_cost_from_solution_json(
    solution_json_path,
    line_capex_lookup,
    n_model_hours=168,
    line_capex_cost_unit="per_w",          # "per_w" or "per_mw"
    line_capex_cost_basis="annualized",    # "annualized", "overnight", or "model_period"
    capital_recovery_factor=0.08,
):
    """
    Calculate line expansion MW and line expansion cost.

    Parameters
    ----------
    line_capex_cost_unit:
        "per_w"  means graph capex_cost is $/W.
        "per_mw" means graph capex_cost is $/MW.

    line_capex_cost_basis:
        "annualized":
            capex_cost is already annualized, so scale by modeled hours / 8760.

        "overnight":
            capex_cost is overnight capital cost, so apply CRF and modeled hours / 8760.

        "model_period":
            capex_cost is already for the modeled horizon, so do not time-scale.
    """

    solution_json_path = Path(solution_json_path)

    with open(solution_json_path, "r") as f:
        sol = json.load(f)

    if line_capex_cost_basis == "annualized":
        horizon_factor = float(n_model_hours) / 8760.0

    elif line_capex_cost_basis == "overnight":
        horizon_factor = float(capital_recovery_factor) * float(n_model_hours) / 8760.0

    elif line_capex_cost_basis == "model_period":
        horizon_factor = 1.0

    else:
        raise ValueError(
            "line_capex_cost_basis must be 'annualized', 'overnight', or 'model_period'."
        )

    total_expansion_mw = 0.0
    total_expansion_cost_mil = 0.0

    for edge in sol.get("edges", []):
        for line_id, line_data in edge.get("lines", {}).items():

            capex_raw = line_data.get("capex", [0.0])

            if isinstance(capex_raw, list):
                capex_w = capex_raw[0] if capex_raw else 0.0
            else:
                capex_w = capex_raw

            capex_w = float(capex_w)
            capex_mw = capex_w / 1e6

            capex_cost = float(line_capex_lookup.get(str(line_id), 0.0))

            total_expansion_mw += capex_mw

            if line_capex_cost_unit == "per_w":
                # capex_cost is $/W
                cost_mil = (
                    capex_w
                    * capex_cost
                    * horizon_factor
                ) / 1e6

            elif line_capex_cost_unit == "per_mw":
                # capex_cost is $/MW
                cost_mil = (
                    capex_mw
                    * capex_cost
                    * horizon_factor
                ) / 1e6

            else:
                raise ValueError("line_capex_cost_unit must be 'per_w' or 'per_mw'.")

            total_expansion_cost_mil += cost_mil

    return total_expansion_mw, total_expansion_cost_mil

# ============================================================
# Robust line OPEX reader
# ============================================================
def build_line_opex_table_from_results_dir(
    results_dir,
    region,
    scenario_label,
    scenario_ids=None,
    verbose=False,
):
    """
    Read line OPEX only for selected scenario IDs.

    Parameters
    ----------
    results_dir:
        One charging-scenario result directory.

    region:
        GOOD model region.

    scenario_label:
        Charging scenario name such as arrive, delay, or midnight.

    scenario_ids:
        Optional collection of selected scenario IDs.
        When provided, unrelated cost_components files are skipped
        before pandas opens them.

    verbose:
        Print selected files and loading diagnostics.
    """

    results_dir = Path(results_dir)
    region_code = normalize_region(region)
    scenario_label = str(scenario_label).strip().lower()

    if not results_dir.exists():
        raise FileNotFoundError(
            f"Line OPEX result directory does not exist:\n"
            f"{results_dir}"
        )

    if scenario_ids is not None:
        scenario_ids = {
            int(value)
            for value in scenario_ids
        }

    all_files = sorted(
        results_dir.rglob("*_cost_components.csv")
    )

    selected_files = []

    for file_path in all_files:
        file_scenario_id = parse_scenario_id(
            file_path.parent.name
        )

        if file_scenario_id is None:
            file_scenario_id = parse_scenario_id(
                file_path.name
            )

        # Skip unrelated scenario files before reading CSV.
        if (
            scenario_ids is not None
            and file_scenario_id is not None
            and file_scenario_id not in scenario_ids
        ):
            continue

        selected_files.append(
            (
                file_path,
                file_scenario_id,
            )
        )

    if verbose:
        print(
            f"\nSearching selected line OPEX in:\n"
            f"{results_dir}"
        )

        print(
            f"Selected {len(selected_files)} of "
            f"{len(all_files)} cost_components files"
        )

        for file_path, _ in selected_files:
            print(f"  {file_path}")

    records = []

    for file_path, file_scenario_id in selected_files:
        try:
            temp = pd.read_csv(file_path)

        except Exception as exc:
            if verbose:
                print(
                    f"Could not read {file_path}: {exc}"
                )
            continue

        if temp.empty:
            continue

        if "line_opex_total" not in temp.columns:
            if verbose:
                print(
                    f"Skipped {file_path}: "
                    "line_opex_total column not found."
                )
            continue

        temp = temp.copy()

        # A combined file may contain multiple scenario IDs.
        if (
            file_scenario_id is None
            and "scenario_id" in temp.columns
            and scenario_ids is not None
        ):
            temp["scenario_id"] = pd.to_numeric(
                temp["scenario_id"],
                errors="coerce",
            )

            temp = temp[
                temp["scenario_id"].isin(
                    scenario_ids
                )
            ].copy()

            if temp.empty:
                continue

        # -------------------------------------------------
        # Case A: scenario ID is determined by the folder
        # -------------------------------------------------
        if file_scenario_id is not None:
            scenario_groups = [
                (
                    int(file_scenario_id),
                    temp,
                )
            ]

        # -------------------------------------------------
        # Case B: combined CSV with scenario_id column
        # -------------------------------------------------
        elif "scenario_id" in temp.columns:
            scenario_groups = []

            for sid, group_df in temp.groupby(
                "scenario_id",
                dropna=True,
            ):
                try:
                    sid = int(sid)
                except Exception:
                    continue

                if (
                    scenario_ids is not None
                    and sid not in scenario_ids
                ):
                    continue

                scenario_groups.append(
                    (
                        sid,
                        group_df,
                    )
                )

        else:
            if verbose:
                print(
                    f"Skipped {file_path}: "
                    "scenario ID could not be determined."
                )
            continue

        for scenario_id, scenario_df in scenario_groups:
            line_opex = pd.to_numeric(
                scenario_df["line_opex_total"],
                errors="coerce",
            ).dropna()

            if line_opex.empty:
                continue

            # line_opex_total is stored in dollars.
            line_opex_mil = (
                float(line_opex.sum())
                / 1e6
            )

            records.append(
                {
                    "scenario_id": int(
                        scenario_id
                    ),
                    "region": region_code,

                    # Metadata is obtained later from
                    # the policy folder.
                    "rps": pd.NA,
                    "batt_capex": pd.NA,
                    "group": pd.NA,
                    "participation": pd.NA,

                    "scenario_label": (
                        scenario_label
                    ),
                    "line_opex_mil": (
                        line_opex_mil
                    ),
                    "n_opex_rows": int(
                        line_opex.shape[0]
                    ),
                    "source_cost_components": str(
                        file_path
                    ),
                }
            )

    out = pd.DataFrame(records)

    if out.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "region",
                "rps",
                "batt_capex",
                "group",
                "participation",
                "scenario_label",
                "line_opex_mil",
                "n_opex_rows",
                "source_cost_components",
            ]
        )

    # Avoid reading the same scenario twice.
    out = (
        out.sort_values(
            [
                "scenario_id",
                "source_cost_components",
            ]
        )
        .drop_duplicates(
            subset=[
                "scenario_id",
                "scenario_label",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    if verbose:
        print("\nSelected line OPEX table:")

        print(
            out[
                [
                    "scenario_id",
                    "region",
                    "scenario_label",
                    "line_opex_mil",
                    "n_opex_rows",
                ]
            ].to_string(index=False)
        )

    return out


def lookup_line_opex_mil(
    line_opex_table,
    meta,
    scenario_label,
    verbose=False,
):
    """
    Find line OPEX for one scenario using robust matching.

    Match order:
        1. scenario_id + batt_capex + rps + group + participation
        2. scenario_id + batt_capex + rps
        3. scenario_id + batt_capex
        4. scenario_id
        5. return 0.0
    """

    if line_opex_table is None or line_opex_table.empty:
        return 0.0

    df = line_opex_table.copy()

    df = df[
        df["scenario_label"].astype(str).str.lower()
        == str(scenario_label).strip().lower()
    ].copy()

    if df.empty:
        return 0.0

    scenario_id = int(meta["scenario_id"])
    batt_capex = int(meta["batt_capex"])
    rps = int(meta["rps"])
    group = str(meta["group"])
    participation = str(meta["participation"])

    match_specs = [
        {
            "scenario_id": scenario_id,
            "batt_capex": batt_capex,
            "rps": rps,
            "group": group,
            "participation": participation,
        },
        {
            "scenario_id": scenario_id,
            "batt_capex": batt_capex,
            "rps": rps,
        },
        {
            "scenario_id": scenario_id,
            "batt_capex": batt_capex,
        },
        {
            "scenario_id": scenario_id,
        },
    ]

    for spec in match_specs:
        temp = df.copy()

        for col, value in spec.items():
            if col not in temp.columns:
                continue

            if col in ["scenario_id", "batt_capex", "rps"]:
                temp = temp[
                    pd.to_numeric(temp[col], errors="coerce")
                    == int(value)
                ].copy()
            else:
                temp = temp[
                    temp[col].astype(str).str.lower()
                    == str(value).lower()
                ].copy()

        if not temp.empty:
            return float(temp["line_opex_mil"].mean())

    if verbose:
        print(
            "Line OPEX not matched for:",
            {
                "scenario_id": scenario_id,
                "batt_capex": batt_capex,
                "rps": rps,
                "group": group,
                "participation": participation,
                "scenario_label": scenario_label,
            },
        )

    return 0.0


def audit_line_opex_for_region(
    output_root,
    region,
    year=2030,
    adoption="mid",
    baseline_scenario_label="arrive",
    program_scenario_label="delay",
    verbose=True,
):
    """
    Diagnostic helper to check whether line OPEX exists for one region.
    """

    region_code = normalize_region(region)

    baseline_dir = get_line_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=baseline_scenario_label,
    )

    program_dir = get_line_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=program_scenario_label,
    )

    base_table = build_line_opex_table_from_results_dir(
        results_dir=baseline_dir,
        region=region_code,
        scenario_label=baseline_scenario_label,
        verbose=verbose,
    )

    program_table = build_line_opex_table_from_results_dir(
        results_dir=program_dir,
        region=region_code,
        scenario_label=program_scenario_label,
        verbose=verbose,
    )

    audit = pd.concat([base_table, program_table], ignore_index=True)

    print("\nLine OPEX audit summary:")
    if audit.empty:
        print("No line OPEX rows found.")
    else:
        print(
            audit[
                [
                    "region",
                    "scenario_label",
                    "scenario_id",
                    "rps",
                    "batt_capex",
                    "group",
                    "participation",
                    "line_opex_mil",
                    "n_opex_rows",
                ]
            ]
            .sort_values(["scenario_label", "scenario_id", "rps", "batt_capex"])
            .to_string(index=False)
        )

    return audit


# ============================================================
# Collect line cost for one GOOD region
# ============================================================
def collect_line_cost_for_region(
    output_root,
    region,
    year=2030,
    adoption="mid",
    graph_json_path=None,
    batt_capex_target=150,
    baseline_scenario_label="arrive",
    program_scenario_label="delay",
    selected_groups=("V1G", "V2G", "V1G+V2G"),
    selected_participation=("50%",),
    rps_order=None,
    n_model_hours=168,
    line_capex_cost_unit="per_w",
    line_capex_cost_basis="annualized",
    capital_recovery_factor=0.08,
    verbose=False,
):
    """
    Collect line CAPEX and OPEX only for selected cases.

    Baseline
    --------
    Read Base-only cases from the selected baseline
    charging-scenario directory.

    Program
    -------
    Read only selected groups and participation levels
    from the selected program charging-scenario directory.

    Important
    ---------
    Folder metadata is checked before opening the large
    annual solution JSON.
    """

    region_code = normalize_region(region)

    if graph_json_path is None:
        raise ValueError(
            f"graph_json_path cannot be None for "
            f"{region_code}."
        )

    selected_groups = {
        str(value)
        for value in selected_groups
    }

    selected_participation = {
        str(value)
        for value in selected_participation
    }

    rps_set = (
        None
        if rps_order is None
        else {
            int(value)
            for value in rps_order
        }
    )

    line_capex_lookup = (
        build_line_capex_cost_lookup(
            graph_json_path
        )
    )

    baseline_dir = get_line_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=(
            baseline_scenario_label
        ),
    )

    program_dir = get_line_results_dir(
        output_root=output_root,
        region=region_code,
        year=year,
        adoption=adoption,
        scenario_label=(
            program_scenario_label
        ),
    )

    if verbose:
        print(f"\n{region_code}")
        print(
            "Baseline charging scenario:",
            baseline_dir,
        )
        print(
            "Program charging scenario:",
            program_dir,
        )

    # =====================================================
    # Select policy folders before reading any large files
    # =====================================================
    def select_policy_folders(
        results_dir,
        case_type,
    ):
        selected = []

        for folder in sorted(
            Path(results_dir).glob("s*")
        ):
            if not folder.is_dir():
                continue

            if not is_policy_result_folder(
                folder.name
            ):
                continue

            try:
                meta = (
                    parse_policy_folder_metadata_from_name(
                        folder_name=folder.name,
                        region=region_code,
                    )
                )

            except Exception as exc:
                if verbose:
                    print(
                        "Skipped metadata parse:",
                        folder.name,
                        "->",
                        exc,
                    )
                continue

            meta_bcapex = meta.get(
                "batt_capex_num",
                meta.get("batt_capex"),
            )

            meta_rps = meta.get(
                "rps_plot_value",
                meta.get("rps"),
            )

            if meta_bcapex is None:
                continue

            if int(meta_bcapex) != int(
                batt_capex_target
            ):
                continue

            if (
                rps_set is not None
                and int(meta_rps) not in rps_set
            ):
                continue

            # ---------------------------------------------
            # Baseline selection
            # ---------------------------------------------
            if case_type == "baseline":
                if meta["group"] != "Base only":
                    continue

            # ---------------------------------------------
            # Program selection
            # ---------------------------------------------
            elif case_type == "program":
                if (
                    meta["group"]
                    not in selected_groups
                ):
                    continue

                if (
                    meta["participation"]
                    not in selected_participation
                ):
                    continue

            else:
                raise ValueError(
                    f"Unknown case_type: {case_type}"
                )

            selected.append(
                {
                    "folder": folder,
                    "meta": meta,
                    "case_type": case_type,
                }
            )

        return selected

    selected_baseline = select_policy_folders(
        results_dir=baseline_dir,
        case_type="baseline",
    )

    selected_program = select_policy_folders(
        results_dir=program_dir,
        case_type="program",
    )

    if not selected_baseline:
        raise ValueError(
            f"No selected Base-only line cases found "
            f"for {region_code}.\n"
            f"Baseline directory: {baseline_dir}\n"
            f"Battery capex: {batt_capex_target}\n"
            f"RPS order: {rps_order}"
        )

    if not selected_program:
        raise ValueError(
            f"No selected program line cases found "
            f"for {region_code}.\n"
            f"Program directory: {program_dir}\n"
            f"Groups: {sorted(selected_groups)}\n"
            f"Participation: "
            f"{sorted(selected_participation)}\n"
            f"Battery capex: {batt_capex_target}\n"
            f"RPS order: {rps_order}"
        )

    baseline_scenario_ids = {
        int(item["meta"]["scenario_id"])
        for item in selected_baseline
    }

    program_scenario_ids = {
        int(item["meta"]["scenario_id"])
        for item in selected_program
    }

    if verbose:
        print(
            "Selected baseline scenario IDs:",
            sorted(baseline_scenario_ids),
        )

        print(
            "Selected program scenario IDs:",
            sorted(program_scenario_ids),
        )

    # =====================================================
    # Read OPEX only for selected scenario IDs
    # =====================================================
    baseline_opex_table = (
        build_line_opex_table_from_results_dir(
            results_dir=baseline_dir,
            region=region_code,
            scenario_label=(
                baseline_scenario_label
            ),
            scenario_ids=(
                baseline_scenario_ids
            ),
            verbose=verbose,
        )
    )

    program_opex_table = (
        build_line_opex_table_from_results_dir(
            results_dir=program_dir,
            region=region_code,
            scenario_label=(
                program_scenario_label
            ),
            scenario_ids=(
                program_scenario_ids
            ),
            verbose=verbose,
        )
    )

    # Fast dictionary lookup.
    baseline_opex_lookup = {
        int(row["scenario_id"]): float(
            row["line_opex_mil"]
        )
        for _, row
        in baseline_opex_table.iterrows()
    }

    program_opex_lookup = {
        int(row["scenario_id"]): float(
            row["line_opex_mil"]
        )
        for _, row
        in program_opex_table.iterrows()
    }

    # =====================================================
    # Read only selected annual solution JSON files
    # =====================================================
    rows = []

    def read_selected_case(
        item,
        scenario_label,
        opex_lookup,
    ):
        folder = item["folder"]
        meta = item["meta"]
        case_type = item["case_type"]

        scenario_id = int(
            meta["scenario_id"]
        )

        solution_json = (
            find_solution_json_in_folder(
                folder
            )
        )

        if solution_json is None:
            raise FileNotFoundError(
                f"No solution JSON found in:\n"
                f"{folder}"
            )

        expansion_mw, expansion_cost_mil = (
            line_expansion_cost_from_solution_json(
                solution_json_path=(
                    solution_json
                ),
                line_capex_lookup=(
                    line_capex_lookup
                ),
                n_model_hours=(
                    n_model_hours
                ),
                line_capex_cost_unit=(
                    line_capex_cost_unit
                ),
                line_capex_cost_basis=(
                    line_capex_cost_basis
                ),
                capital_recovery_factor=(
                    capital_recovery_factor
                ),
            )
        )

        line_opex_mil = float(
            opex_lookup.get(
                scenario_id,
                0.0,
            )
        )

        if (
            scenario_id not in opex_lookup
            and verbose
        ):
            print(
                "WARNING: no line OPEX matched for",
                {
                    "region": region_code,
                    "scenario_id": scenario_id,
                    "scenario_label": (
                        scenario_label
                    ),
                    "folder": str(folder),
                },
            )

        return {
            **meta,
            "case_type": case_type,
            "scenario_label": str(
                scenario_label
            ).lower(),
            "adoption": str(
                adoption
            ).lower(),
            "line_expansion_mw": (
                expansion_mw
            ),
            "line_expansion_cost_mil": (
                expansion_cost_mil
            ),
            "line_opex_mil": (
                line_opex_mil
            ),
            "total_line_cost_mil": (
                expansion_cost_mil
                + line_opex_mil
            ),
            "folder": str(folder),
            "solution_json_path": str(
                solution_json
            ),
        }

    for item in selected_baseline:
        rows.append(
            read_selected_case(
                item=item,
                scenario_label=(
                    baseline_scenario_label
                ),
                opex_lookup=(
                    baseline_opex_lookup
                ),
            )
        )

    for item in selected_program:
        rows.append(
            read_selected_case(
                item=item,
                scenario_label=(
                    program_scenario_label
                ),
                opex_lookup=(
                    program_opex_lookup
                ),
            )
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            f"No selected line-cost rows were "
            f"collected for {region_code}."
        )

    # =====================================================
    # Remove accidental duplicates
    # =====================================================
    key_cols = [
        "case_type",
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
    ]

    available_key_cols = [
        column
        for column in key_cols
        if column in df.columns
    ]

    df = (
        df.sort_values(
            available_key_cols
            + [
                "scenario_id",
                "folder",
            ]
        )
        .drop_duplicates(
            subset=available_key_cols,
            keep="first",
        )
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"\nLoaded {len(df)} selected "
            f"line-cost cases for {region_code}."
        )

        print(
            df[
                [
                    column
                    for column in [
                        "case_type",
                        "scenario_id",
                        "rps",
                        "batt_capex",
                        "group",
                        "participation",
                        "line_expansion_mw",
                        "line_opex_mil",
                        "total_line_cost_mil",
                    ]
                    if column in df.columns
                ]
            ].to_string(index=False)
        )

        # =====================================================
    # Aggregate accidental duplicate rows
    # =====================================================
    key_cols = [
        "region",
        "scenario_label",
        "group",
        "participation",
        "rps",
        "batt_capex",
        "case_type",
    ]

    available_key_cols = [
        col for col in key_cols
        if col in df.columns
    ]

    df = (
        df.groupby(
            available_key_cols,
            as_index=False,
            dropna=False,
        )
        .agg(
            line_expansion_mw=(
                "line_expansion_mw",
                "mean",
            ),
            line_expansion_cost_mil=(
                "line_expansion_cost_mil",
                "mean",
            ),
            line_opex_mil=(
                "line_opex_mil",
                "mean",
            ),
            total_line_cost_mil=(
                "total_line_cost_mil",
                "mean",
            ),
            folder=(
                "folder",
                lambda x: " | ".join(
                    map(str, x)
                ),
            ),
            solution_json_path=(
                "solution_json_path",
                lambda x: " | ".join(
                    map(str, x)
                ),
            ),
        )
    )

    # =====================================================
    # Build the Base table
    # =====================================================
    base = df[
        df["case_type"] == "baseline"
    ].copy()

    if base.empty:
        raise ValueError(
            f"No Base line-cost rows found for "
            f"{region_code}."
        )

    base = base.rename(
        columns={
            "line_expansion_mw":
                "base_line_expansion_mw",
            "line_expansion_cost_mil":
                "base_line_expansion_cost_mil",
            "line_opex_mil":
                "base_line_opex_mil",
            "total_line_cost_mil":
                "base_total_line_cost_mil",
        }
    )

    base = base[
        [
            "region",
            "rps",
            "batt_capex",
            "base_line_expansion_mw",
            "base_line_expansion_cost_mil",
            "base_line_opex_mil",
            "base_total_line_cost_mil",
        ]
    ].copy()

    duplicate_base = base[
        base.duplicated(
            subset=[
                "region",
                "rps",
                "batt_capex",
            ],
            keep=False,
        )
    ]

    if not duplicate_base.empty:
        raise ValueError(
            "Duplicate Base line-cost rows were found.\n\n"
            + duplicate_base.to_string(index=False)
        )

    # =====================================================
    # Build the program table
    # =====================================================
    program = df[
        df["case_type"] == "program"
    ].copy()

    if program.empty:
        raise ValueError(
            f"No program line-cost rows found for "
            f"{region_code}."
        )

    # =====================================================
    # Match each program result to Base
    # =====================================================
    merged = program.merge(
        base,
        on=[
            "region",
            "rps",
            "batt_capex",
        ],
        how="left",
        validate="many_to_one",
    )

    missing_base = merged[
        merged[
            "base_total_line_cost_mil"
        ].isna()
    ].copy()

    if not missing_base.empty:
        raise ValueError(
            f"Some program rows have no matching Base "
            f"result for {region_code}.\n\n"
            + missing_base[
                [
                    "region",
                    "rps",
                    "group",
                    "participation",
                    "batt_capex",
                ]
            ]
            .drop_duplicates()
            .to_string(index=False)
        )

    # =====================================================
    # Calculate changes relative to Base
    #
    # Positive means program cost is higher than Base.
    # Negative means program cost is lower than Base.
    # =====================================================
    merged[
        "delta_line_expansion_mw"
    ] = (
        merged["line_expansion_mw"]
        - merged["base_line_expansion_mw"]
    )

    merged[
        "delta_line_expansion_cost_mil"
    ] = (
        merged["line_expansion_cost_mil"]
        - merged["base_line_expansion_cost_mil"]
    )

    merged[
        "delta_line_opex_mil"
    ] = (
        merged["line_opex_mil"]
        - merged["base_line_opex_mil"]
    )

    merged[
        "delta_total_line_cost_mil"
    ] = (
        merged["total_line_cost_mil"]
        - merged["base_total_line_cost_mil"]
    )

    if verbose:
        print(
            f"\nCollected line-cost deltas for "
            f"{region_code}:"
        )

        print(
            merged[
                [
                    "region",
                    "rps",
                    "group",
                    "participation",
                    "batt_capex",
                    "delta_line_expansion_cost_mil",
                    "delta_line_opex_mil",
                    "delta_total_line_cost_mil",
                ]
            ].to_string(index=False)
        )

    return merged.reset_index(drop=True)

# ============================================================
# Plot all regions in 2 x 3
# ============================================================

def plot_line_cost_delta_regions_2x3(
    output_root,
    regions=("CA", "NY", "TX", "FL", "PJM"),
    year=2030,
    adoption="mid",
    batt_capex_target=150,
    graph_json_path_by_region=None,
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    selected_groups=("V1G", "V2G", "V1G+V2G"),
    selected_participation=("50%",),
    rps_order_by_region=None,
    n_model_hours=168,
    line_capex_cost_unit="per_w",
    line_capex_cost_basis="annualized",
    capital_recovery_factor=0.08,
    save_path=None,

    # Plot controls
    figure_title=None,
    font_size=13,
    same_y_scale=True,
    symmetric_y_axis=True,
    y_axis_padding_fraction=0.10,
    label_threshold=0.5,
    verbose=False,
):
    """
    Plot transmission line cost change relative to Base for all five regions.

    Layout:
        2 rows x 3 columns

    Panels:
        CAISO | NYISO | ERCOT
        FRCC  | PJM   | legend

    Shows one battery capex only:
        batt_capex_target
    """

    output_root = Path(output_root)

    region_codes = [
        normalize_region(r)
        for r in regions
    ]

    if graph_json_path_by_region is None:
        raise ValueError("graph_json_path_by_region cannot be None.")

    graph_path_map = {
        normalize_region(k): v
        for k, v in graph_json_path_by_region.items()
    }

    missing_graph = [
        r for r in region_codes
        if r not in graph_path_map
    ]

    if missing_graph:
        raise ValueError(
            "Missing graph_json_path_by_region for: "
            + ", ".join(missing_graph)
        )

    if baseline_scenario_label_by_region is None:
        raise ValueError("baseline_scenario_label_by_region cannot be None.")

    if program_scenario_label_by_region is None:
        raise ValueError("program_scenario_label_by_region cannot be None.")

    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    if rps_order_by_region is None:
        rps_order_by_region = {
            "CAISO": [50, 60, 70],
            "NYISO": [60, 70, 80],
            "ERCOT": [0, 60, 70],
            "FRCC": [0, 25, 50],
            "PJM": [-10, 0, 10],
        }
    else:
        rps_order_by_region = {
            normalize_region(k): list(v)
            for k, v in rps_order_by_region.items()
        }

    missing_rps = [
        r for r in region_codes
        if r not in rps_order_by_region
    ]

    if missing_rps:
        raise ValueError(
            "Missing rps_order_by_region for: "
            + ", ".join(missing_rps)
        )

    # --------------------------------------------------------
    # Colors and font sizes
    # --------------------------------------------------------
    component_colors = {
        "expansion": "#7B6D8D",
        "opex": "#E15759",
    }

    axis_label_fontsize = max(7, font_size)
    axis_tick_fontsize = max(7, font_size - 1)
    panel_title_fontsize = max(9, font_size + 4)
    rps_label_fontsize = max(7, font_size - 1)
    bar_label_fontsize = max(6, font_size - 2)
    legend_fontsize = max(7, font_size - 1)
    legend_title_fontsize = max(8, font_size)
    figure_title_fontsize = max(10, font_size + 5)

    def stacked_bar_signed(ax, x, expansion_val, opex_val, width):
        """
        Draw signed stacked CAPEX/OPEX bar.
        """

        pos_bottom = 0.0
        neg_bottom = 0.0

        values = [
            (expansion_val, "expansion", "CAPEX"),
            (opex_val, "opex", "OPEX"),
        ]

        for val, key, label in values:
            if pd.isna(val) or abs(val) < 1e-12:
                continue

            if val >= 0:
                bottom = pos_bottom
                pos_bottom += val
            else:
                bottom = neg_bottom
                neg_bottom += val

            ax.bar(
                x,
                val,
                width=bar_width,
                bottom=bottom,
                color=component_colors[key],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )

            if abs(val) >= label_threshold:
                ax.text(
                    x,
                    bottom + val / 2,
                    label,
                    ha="center",
                    va="center",
                    fontsize=bar_label_fontsize,
                    rotation=90,
                    color="black",
                    zorder=5,
                )

    # --------------------------------------------------------
    # Collect data
    # --------------------------------------------------------
    df_list = []

    for region in region_codes:
        temp = collect_line_cost_for_region(
            output_root=output_root,
            region=region,
            year=year,
            adoption=adoption,
            graph_json_path=graph_path_map[region],
            batt_capex_target=batt_capex_target,
            baseline_scenario_label=baseline_map[region],
            program_scenario_label=program_map[region],
            selected_groups=selected_groups,
            selected_participation=selected_participation,
            rps_order=rps_order_by_region[region],
            n_model_hours=n_model_hours,
            line_capex_cost_unit=line_capex_cost_unit,
            line_capex_cost_basis=line_capex_cost_basis,
            capital_recovery_factor=capital_recovery_factor,
            verbose=verbose,
        )

        df_list.append(temp)

    df = pd.concat(df_list, ignore_index=True)

    # Keep only requested RPS values
    df = pd.concat(
        [
            df[
                (df["region"] == region)
                & (df["rps"].isin(rps_order_by_region[region]))
            ]
            for region in region_codes
        ],
        ignore_index=True,
    )

    if df.empty:
        raise ValueError("No line-cost data after RPS filtering.")

    if verbose:
        print("\nFinal plotted line-cost dataframe:")
        print(
            df[
                [
                    "region",
                    "rps",
                    "group",
                    "participation",
                    "batt_capex",
                    "delta_line_expansion_cost_mil",
                    "delta_line_opex_mil",
                    "delta_total_line_cost_mil",
                ]
            ].to_string(index=False)
        )

    if np.isclose(df["delta_line_opex_mil"].fillna(0).abs().sum(), 0):
        print(
            "\nWarning: all delta_line_opex_mil values are zero. "
            "This can be real, but it also means no line_opex_total values were matched. "
            "Run audit_line_opex_for_region(...) for one region to verify."
        )

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, axes_grid = plt.subplots(
        2,
        3,
        figsize=(18, 11.5),
        sharey=False,
        facecolor="white",
        squeeze=False,
    )

    axes = axes_grid.flatten()

    for ax in axes:
        ax.set_facecolor("white")

    bar_width = 0.82
    bar_gap = 0.20
    rps_gap = 0.65

    panel_label_records = []

    # Bar sequence
    bar_cases = []

    for group in selected_groups:
        for part in selected_participation:
            bar_cases.append((group, part))

    one_participation = len(selected_participation) == 1

    # --------------------------------------------------------
    # Plot panels
    # --------------------------------------------------------
    for plot_idx, region in enumerate(region_codes):
        ax = axes[plot_idx]

        region_df = df[df["region"] == region].copy()
        rps_values = list(rps_order_by_region[region])

        x_positions = []
        x_labels = []
        rps_centers = []
        separator_positions = []

        current_x = 0.0

        for rps_idx, rps in enumerate(rps_values):
            local_positions = []

            for group, part in bar_cases:
                x = current_x
                x_positions.append(x)

                if one_participation:
                    x_labels.append(group)
                else:
                    x_labels.append(f"{group}\n{part}")

                local_positions.append(x)

                temp = region_df[
                    (region_df["rps"] == rps)
                    & (region_df["group"] == group)
                    & (region_df["participation"] == part)
                ].copy()

                if temp.empty:
                    expansion_val = 0.0
                    opex_val = 0.0
                else:
                    expansion_val = temp["delta_line_expansion_cost_mil"].mean()
                    opex_val = temp["delta_line_opex_mil"].mean()

                stacked_bar_signed(
                    ax=ax,
                    x=x,
                    expansion_val=expansion_val,
                    opex_val=opex_val,
                    width=bar_width,
                )

                current_x += 1.0 + bar_gap

            rps_centers.append(np.mean(local_positions))

            if rps_idx < len(rps_values) - 1:
                separator_positions.append(current_x - 0.5 + rps_gap / 2)

            current_x += rps_gap

        ax.axhline(
            0,
            color="black",
            linewidth=1.1,
            zorder=4,
        )

        for xpos in separator_positions:
            ax.axvline(
                xpos,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.65,
                zorder=2,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            x_labels,
            fontsize=axis_tick_fontsize,
            rotation=25,
            ha="right",
            color="black",
        )

        ax.set_title(
            REGION_DISPLAY_NAMES.get(region, region),
            fontsize=panel_title_fontsize,
            fontweight="bold",
            pad=16,
            color="black",
        )

        ax.set_ylabel(
            "Change in line cost relative to Base ($M)",
            fontsize=axis_label_fontsize,
            color="black",
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.30,
            color="gray",
            zorder=0,
        )

        ax.tick_params(
            axis="both",
            labelsize=axis_tick_fontsize,
            colors="black",
        )

        for spine in ax.spines.values():
            spine.set_color("black")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        panel_label_records.append(
            {
                "ax": ax,
                "region": region,
                "rps_values": rps_values,
                "rps_centers": rps_centers,
            }
        )

    # --------------------------------------------------------
    # Force same y-axis scale across all region panels
    # --------------------------------------------------------
    plot_axes = [
        axes[i]
        for i in range(len(region_codes))
        if axes[i] is not None and axes[i].has_data()
    ]

    if same_y_scale and plot_axes:
        ymins = []
        ymaxs = []

        for ax in plot_axes:
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)

        common_ymin = min(ymins)
        common_ymax = max(ymaxs)

        if symmetric_y_axis:
            abs_limit = max(
                abs(common_ymin),
                abs(common_ymax),
            )

            abs_limit = abs_limit * (1 + y_axis_padding_fraction)

            common_ymin = -abs_limit
            common_ymax = abs_limit

        else:
            y_range = common_ymax - common_ymin

            if y_range > 0:
                common_ymin = common_ymin - y_axis_padding_fraction * y_range
                common_ymax = common_ymax + y_axis_padding_fraction * y_range

        for ax in plot_axes:
            ax.set_ylim(common_ymin, common_ymax)

    # --------------------------------------------------------
    # Add RPS labels after y-axis scale is fixed
    # --------------------------------------------------------
    for record in panel_label_records:
        ax = record["ax"]
        region = record["region"]
        rps_values = record["rps_values"]
        rps_centers = record["rps_centers"]

        ymin, ymax = ax.get_ylim()
        y_range = ymax - ymin if ymax != ymin else 1.0

        rps_y = ymin - 0.12 * y_range

        for rps, center in zip(rps_values, rps_centers):
            ax.text(
                center,
                rps_y,
                get_rps_display_label(region, rps),
                ha="center",
                va="top",
                fontsize=rps_label_fontsize,
                fontweight="bold",
                color="black",
                clip_on=False,
            )

    # --------------------------------------------------------
    # Legend in sixth panel
    # --------------------------------------------------------
    for extra_ax in axes[len(region_codes):]:
        extra_ax.axis("off")

    legend_ax = axes[-1]
    legend_ax.axis("off")

    legend_elements = [
        Patch(
            facecolor=component_colors["expansion"],
            edgecolor="black",
            label="Line expansion CAPEX",
        ),
        Patch(
            facecolor=component_colors["opex"],
            edgecolor="black",
            label="Line OPEX",
        ),
    ]

    legend = legend_ax.legend(
        handles=legend_elements,
        loc="center",
        ncol=1,
        frameon=True,
        fontsize=legend_fontsize,
        title="Line cost component",
    )

    legend.get_title().set_fontsize(legend_title_fontsize)
    legend.get_title().set_weight("bold")
    legend.get_title().set_color("black")
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    for text in legend.get_texts():
        text.set_color("black")

    # --------------------------------------------------------
    # Title and save
    # --------------------------------------------------------
    if figure_title is None:
        if len(selected_participation) == 1:
            part_text = selected_participation[0]
        else:
            part_text = ", ".join(selected_participation)

        figure_title = (
            "Transmission cost change relative to Base\n"
            f"Battery capex = ${batt_capex_target}/kWh, "
            f"adoption = {adoption}, participation = {part_text}"
        )

    fig.suptitle(
        figure_title,
        fontsize=figure_title_fontsize,
        y=0.98,
        color="black",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return df, fig, axes


def plot_total_cost_delta_vs_rps_regions_2x3_dual_axis(
    output_root,
    regions=(
    "CAISO",
    "ERCOT",
    "FRCC",
    "ISO-NE",
    "MAPP",
    "MISO",
    "NWPP",
    "NYISO",
    "PJM",
    "RMRG",
    "SERC-E",
    "SERC-N",
    "SERC-SE",
    "SPP",
    "SRSG",
    ),
    year=2030,
    adoption_levels=("slow", "mid", "fast"),
    main_adoption_level="Mid",
    band_adoption_levels=("Slow", "Fast"),
    batt_order=(315,),
    baseline_scenario_label_by_region=None,
    program_scenario_label_by_region=None,
    rps_order_by_region=None,
    selected_groups=("V1G", "V2G"),
    selected_participation=("25%", "50%"),
    panel_title_by_region=None,
    figure_title=None,

    # Right axis only changes the labels.
    # It does not add another plotted line.
    show_second_y_axis=True,
    second_y_axis_label=(
        "Total system cost\n"
        "reduction (Million $)"
    ),
    show_adoption_band=True,
    show_endpoint_labels=False,
    # Axis controls
    same_y_scale=False,
    symmetric_y_axis=False,
    y_axis_padding_fraction=0.08,
    save_path=None,
    # Font controls
    font_size=12,
    figure_title_fontsize=None,
    panel_title_fontsize=None,
    axis_label_fontsize=None,
    tick_label_fontsize=None,
    legend_fontsize=None,
    common_ymin_value=None,
    common_ymax_value=None,
):
    
    # =====================================================
    # Validate battery CAPEX
    # =====================================================
    batt_order = tuple(batt_order)

    if len(batt_order) != 1:
        raise ValueError(
            "This 2 x 3 function requires exactly one "
            "battery CAPEX value.\n"
            "Example: batt_order=(315,)"
        )

    batt_capex = batt_order[0]
    # =====================================================
    # Font controls
    # =====================================================
    if figure_title_fontsize is None:
        figure_title_fontsize = font_size + 5

    if panel_title_fontsize is None:
        panel_title_fontsize = font_size + 4

    if axis_label_fontsize is None:
        axis_label_fontsize = font_size + 1

    if tick_label_fontsize is None:
        tick_label_fontsize = max(
            7,
            font_size - 1,
        )

    if legend_fontsize is None:
        legend_fontsize = max(
            8,
            font_size - 1,
        )
    # =====================================================
    # Internal helpers
    # =====================================================
    def _find_matching_column(
        columns,
        requested_name,
    ):
        requested_clean = str(
            requested_name
        ).strip().lower()

        for column in columns:
            if (
                str(column).strip().lower()
                == requested_clean
            ):
                return column

        return None

    def _format_rps_labels(
        region,
        rps_order,
    ):
        labels = []
    
        for x in rps_order:
            if region in MULTI_STATE_REGIONS:
                if x == -10:
                    labels.append(
                        "RPS −10 pp"
                    )
    
                elif x == 0:
                    labels.append(
                        "RPS baseline"
                    )
    
                elif x == 10:
                    labels.append(
                        "RPS +10 pp"
                    )
    
                else:
                    labels.append(
                        f"RPS {int(x):+d} pp"
                    )
    
            else:
                labels.append(
                    f"{int(x)}%"
                )
    
        return labels

    def _add_second_y_axis(
        ax,
        panel_df,
    ):
        """
        Add a right axis without plotting another line.

        The conversion factor is the panel-specific median:

            million dollars / percentage point

        This matches the behavior of the original plotting
        function.
        """

        if not show_second_y_axis:
            return None

        required_columns = [
            "cost_reduction_percent",
            "cost_reduction_mil",
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in panel_df.columns
        ]

        if missing_columns:
            return None

        valid = panel_df[
            required_columns
        ].copy()

        valid[
            "cost_reduction_percent"
        ] = pd.to_numeric(
            valid[
                "cost_reduction_percent"
            ],
            errors="coerce",
        )

        valid[
            "cost_reduction_mil"
        ] = pd.to_numeric(
            valid[
                "cost_reduction_mil"
            ],
            errors="coerce",
        )

        valid = valid.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna()

        valid = valid[
            valid[
                "cost_reduction_percent"
            ].abs() > 1e-9
        ].copy()

        if valid.empty:
            return None

        valid[
            "mil_per_percent"
        ] = (
            valid[
                "cost_reduction_mil"
            ]
            / valid[
                "cost_reduction_percent"
            ]
        )

        valid = valid.replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        ).dropna(
            subset=[
                "mil_per_percent"
            ]
        )

        if valid.empty:
            return None

        mil_per_percent = (
            valid[
                "mil_per_percent"
            ].median()
        )

        if (
            not np.isfinite(
                mil_per_percent
            )
            or abs(
                mil_per_percent
            ) < 1e-12
        ):
            return None

        ymin, ymax = ax.get_ylim()

        ax2 = ax.twinx()

        ax2.set_ylim(
            ymin * mil_per_percent,
            ymax * mil_per_percent,
        )

        ax2.set_ylabel(
            second_y_axis_label,
            fontsize=axis_label_fontsize,
            color="black",
        )

        ax2.tick_params(
            axis="y",
            colors="black",
            labelcolor="black",
            labelsize=tick_label_fontsize,
        )

        ax2.grid(
            False
        )

        ax2.set_facecolor(
            "none"
        )

        for spine in ax2.spines.values():
            spine.set_color(
                "black"
            )

            spine.set_linewidth(
                1.0
            )

        ax2.spines[
            "top"
        ].set_visible(False)

        ax2.spines[
            "left"
        ].set_visible(False)

        return ax2

    # =====================================================
    # Normalize regions
    # =====================================================
    region_codes = [
        normalize_region(region)
        for region in regions
    ]

    if len(region_codes) == 0:
        raise ValueError(
            "At least one region must be provided."
        )

    # =====================================================
    # Normalize scenario dictionaries
    # =====================================================
    baseline_map = normalize_label_map(
        baseline_scenario_label_by_region,
        regions,
        "baseline_scenario_label_by_region",
    )

    program_map = normalize_label_map(
        program_scenario_label_by_region,
        regions,
        "program_scenario_label_by_region",
    )

    # =====================================================
    # Normalize RPS dictionary
    # =====================================================
    if rps_order_by_region is None:
        rps_order_map = {}

    else:
        rps_order_map = {
            normalize_region(region):
                list(rps_values)
            for region, rps_values
            in rps_order_by_region.items()
        }

    # =====================================================
    # Normalize panel titles
    # =====================================================
    if panel_title_by_region is None:
        panel_title_map = {}

    else:
        panel_title_map = {
            normalize_region(region):
                title
            for region, title
            in panel_title_by_region.items()
        }

    # =====================================================
    # Normalize adoption labels
    # =====================================================
    main_adoption_label = str(
        main_adoption_level
    ).strip().capitalize()

    band_adoption_labels = [
        str(adoption).strip().capitalize()
        for adoption
        in band_adoption_levels
    ]

    if len(
        band_adoption_labels
    ) != 2:
        raise ValueError(
            "band_adoption_levels must contain "
            "exactly two values.\n"
            "Example: ('Slow', 'Fast')"
        )

    low_adoption_label = (
        band_adoption_labels[0]
    )

    high_adoption_label = (
        band_adoption_labels[1]
    )

    # =====================================================
    # Load total-system-cost differences
    # =====================================================
    df_list = []

    for region in region_codes:
        region_rps_order = (
            rps_order_map.get(
                region,
                None,
            )
        )

        temp = (
            compute_total_cost_delta_for_region_battery_adoption_range(
                output_root=output_root,
                region=region,
                year=year,
                adoption_levels=(
                    adoption_levels
                ),
                batt_capex=(
                    batt_capex
                ),
                baseline_scenario_label=(
                    baseline_map[region]
                ),
                program_scenario_label=(
                    program_map[region]
                ),
                rps_order=(
                    region_rps_order
                ),
                selected_participation=(
                    selected_participation
                ),
            )
        )

        if (
            temp is None
            or temp.empty
        ):
            raise ValueError(
                "No total-system-cost data were "
                f"returned for {region}."
            )

        df_list.append(
            temp
        )

    if not df_list:
        raise ValueError(
            "No total-system-cost data were loaded."
        )

    cleaned_df_list = [
        regional_df.dropna(
            axis=1,
            how="all",
        )
        for regional_df in df_list
        if (
            regional_df is not None
            and not regional_df.empty
        )
    ]

    df_delta_all = pd.concat(
        cleaned_df_list,
        ignore_index=True,
        sort=False,
    )

    # =====================================================
    # Validate required columns
    # =====================================================
    required_columns = [
        "region",
        "adoption",
        "group",
        "participation",
        "rps_plot_value",
        "delta_cost_mil",
        "delta_cost_percent",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df_delta_all.columns
    ]

    if missing_columns:
        raise ValueError(
            "The cost-difference dataframe is missing "
            "these columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # =====================================================
    # Keep requested cases
    # =====================================================
    df_delta_all = df_delta_all[
        df_delta_all[
            "region"
        ].isin(
            region_codes
        )
    ].copy()

    df_delta_all = df_delta_all[
        df_delta_all[
            "group"
        ].isin(
            selected_groups
        )
    ].copy()

    df_delta_all = df_delta_all[
        df_delta_all[
            "participation"
        ].isin(
            selected_participation
        )
    ].copy()

    if df_delta_all.empty:
        raise ValueError(
            "No data remained after filtering "
            "groups and participation levels."
        )

    # =====================================================
    # Calculate reduction metrics
    #
    # Existing delta:
    #     program cost - Base cost
    #
    # Reduction:
    #     Base cost - program cost
    # =====================================================
    df_delta_all[
        "delta_cost_mil"
    ] = pd.to_numeric(
        df_delta_all[
            "delta_cost_mil"
        ],
        errors="coerce",
    )

    df_delta_all[
        "delta_cost_percent"
    ] = pd.to_numeric(
        df_delta_all[
            "delta_cost_percent"
        ],
        errors="coerce",
    )

    df_delta_all[
        "cost_reduction_mil"
    ] = (
        -df_delta_all[
            "delta_cost_mil"
        ]
    )

    df_delta_all[
        "cost_reduction_percent"
    ] = (
        -df_delta_all[
            "delta_cost_percent"
        ]
    )

    # =====================================================
    # Plot styles
    # =====================================================
    global_group_colors = globals().get(
        "GROUP_COLORS",
        {},
    )

    group_colors = {
        "V1G": global_group_colors.get(
            "V1G",
            "#4E79A7",
        ),

        "V2G": global_group_colors.get(
            "V2G",
            "#E15759",
        ),

        "V1G+V2G":
            global_group_colors.get(
                "V1G+V2G",
                "#59A14F",
            ),
    }

    marker_map = {
        "0%": "o",
        "10%": "D",
        "25%": "s",
        "30%": "o",
        "50%": "^",
        "75%": "D",
        "100%": "P",
    }

    fallback_markers = [
        "o",
        "s",
        "^",
        "D",
        "P",
        "X",
    ]

    for participation_index, participation in enumerate(
        selected_participation
    ):
        if participation not in marker_map:
            marker_map[
                participation
            ] = fallback_markers[
                participation_index
                % len(fallback_markers)
            ]

    series_order = [
        (
            group,
            participation,
        )
        for group in selected_groups
        for participation
        in selected_participation
    ]

    # =====================================================
    # Create dynamic three-column figure
    #
    # One additional cell is always reserved for the
    # legend in the bottom-right corner.
    # =====================================================
    
    n_regions = len(region_codes)
    ncols = 3
    
    n_required_cells = (
        n_regions + 1
    )
    
    nrows = (
        n_required_cells
        + ncols
        - 1
    ) // ncols
    
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            19.0,
            5.0 * nrows,
        ),
        facecolor="white",
        squeeze=False,
        sharey=False,
    )
    
    axes = axes_grid.flatten()
    plotted_axes = []
    panel_data_by_index = {}
    endpoint_data_by_index = {}
    missing_combinations = []
    # =====================================================
    # Plot regional panels
    # =====================================================
    for plot_index, region in enumerate(
        region_codes
    ):
        ax = axes[
            plot_index
        ]

        plotted_axes.append(
            ax
        )

        ax.set_facecolor(
            "white"
        )

        sub_region = df_delta_all[
            df_delta_all[
                "region"
            ] == region
        ].copy()

        panel_data_by_index[
            plot_index
        ] = sub_region
        endpoint_data_by_index[
            plot_index
        ] = []
        if sub_region.empty:
            ax.axis(
                "off"
            )

            continue

        # ---------------------------------------------
        # RPS order
        # ---------------------------------------------
        if region in rps_order_map:
            rps_order = list(
                rps_order_map[
                    region
                ]
            )

        else:
            rps_order = (
                sub_region[
                    "rps_plot_value"
                ]
                .dropna()
                .astype(float)
                .sort_values()
                .unique()
                .tolist()
            )

        if not rps_order:
            ax.axis(
                "off"
            )

            continue

        rps_labels = (
            _format_rps_labels(
                region=region,
                rps_order=rps_order,
            )
        )

        x_positions = np.arange(
            len(rps_order)
        )

        # ---------------------------------------------
        # Plot each program and participation
        # ---------------------------------------------
        for group, participation in series_order:
            sub_series = sub_region[
                (
                    sub_region[
                        "group"
                    ] == group
                )
                & (
                    sub_region[
                        "participation"
                    ] == participation
                )
            ].copy()

            if sub_series.empty:
                missing_combinations.append(
                    f"{region}, {group}, "
                    f"{participation}"
                )

                continue

            # =========================================
            # Percentage reduction pivot
            #
            # This is the only metric plotted.
            # =========================================
            pivot_percent = (
                sub_series.pivot_table(
                    index=(
                        "rps_plot_value"
                    ),
                    columns=(
                        "adoption"
                    ),
                    values=(
                        "cost_reduction_percent"
                    ),
                    aggfunc="first",
                )
            )

            pivot_percent = (
                pivot_percent.reindex(
                    rps_order
                )
            )

            main_percent_col = (
                _find_matching_column(
                    pivot_percent.columns,
                    main_adoption_label,
                )
            )

            if main_percent_col is None:
                missing_combinations.append(
                    f"{region}, {group}, "
                    f"{participation}, missing "
                    f"{main_adoption_label}"
                )

                continue

            group_color = (
                group_colors.get(
                    group,
                    "gray",
                )
            )

            marker = (
                marker_map.get(
                    participation,
                    "o",
                )
            )

            # =========================================
            # Slow-to-Fast adoption band
            # Percentage reduction only
            # =========================================
            if show_adoption_band:
                low_percent_col = (
                    _find_matching_column(
                        pivot_percent.columns,
                        low_adoption_label,
                    )
                )

                high_percent_col = (
                    _find_matching_column(
                        pivot_percent.columns,
                        high_adoption_label,
                    )
                )

                if (
                    low_percent_col
                    is not None
                    and high_percent_col
                    is not None
                ):
                    percentage_band = (
                        pivot_percent[
                            [
                                low_percent_col,
                                high_percent_col,
                            ]
                        ]
                    )

                    band_low = (
                        percentage_band.min(
                            axis=1
                        )
                    )

                    band_high = (
                        percentage_band.max(
                            axis=1
                        )
                    )
                    ax.fill_between(
                        x_positions,
                        band_low.to_numpy(
                            dtype=float
                        ),
                        band_high.to_numpy(
                            dtype=float
                        ),
                        color=group_color,
                        alpha=0.15,
                        linewidth=0,
                        zorder=2,
                    )
                    ax.plot(
                        x_positions,
                        band_low.to_numpy(
                            dtype=float
                        ),
                        color=group_color,
                        linewidth=0.8,
                        alpha=0.30,
                        zorder=2,
                    )
                    ax.plot(
                        x_positions,
                        band_high.to_numpy(
                            dtype=float
                        ),
                        color=group_color,
                        linewidth=0.8,
                        alpha=0.30,
                        zorder=2,
                    )
            # =========================================
            # Main percentage-reduction line
            #
            # Only one line is plotted.
            # =========================================
            line_values = pivot_percent[
                main_percent_col
            ].to_numpy(
                dtype=float
            )
            ax.plot(
                x_positions,
                line_values,
                color=group_color,
                linestyle="-",
                marker=marker,
                markerfacecolor=group_color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linewidth=2.3,
                markersize=7,
                zorder=4,
            )
            valid_endpoint_indices = np.where(
                np.isfinite(line_values)
            )[0]
            
            if len(valid_endpoint_indices) > 0:
                endpoint_index = valid_endpoint_indices[-1]
                endpoint_data_by_index[
                    plot_index
                ].append({
                    "group": group,
                    "participation": participation,
                    "color": group_color,
                    "x": float(
                        x_positions[endpoint_index]
                    ),
                    "y": float(
                        line_values[endpoint_index]
                    ),
                })
        # ---------------------------------------------
        # Panel title
        # ---------------------------------------------
        panel_title = (
            panel_title_map.get(
                region,
                REGION_DISPLAY_NAMES.get(
                    region,
                    region,
                ),
            )
        )
        ax.set_title(
            panel_title,
            fontsize=(
                panel_title_fontsize
            ),
            fontweight="bold",
            color="black",
            pad=8,
        )
        # ---------------------------------------------
        # X-axis
        # ---------------------------------------------
        ax.set_xlabel(
            "RPS target",
            fontsize=(
                axis_label_fontsize
            ),
            color="black",
        )
        ax.set_xticks(
            x_positions
        )
        ax.set_xticklabels(
            rps_labels,
            fontsize=(
                tick_label_fontsize
            ),
            color="black",
        )
        ax.set_xlim(
            -0.05,
            (len(rps_order) - 1) + 0.08,
        )
        # ---------------------------------------------
        # Left y-axis
        # ---------------------------------------------
        ax.set_ylabel(
            "Total system cost reduction\n"
            "relative to Base (%)",
            fontsize=(
                axis_label_fontsize
            ),
            color="black",
        )
        ax.tick_params(
            axis="both",
            colors="black",
            labelcolor="black",
            labelsize=(
                tick_label_fontsize
            ),
        )
        ax.grid(
            True,
            axis="both",
            linestyle="--",
            alpha=0.30,
            color="gray",
        )
        for spine in ax.spines.values():
            spine.set_color(
                "black"
            )
            spine.set_linewidth(
                1.0
            )
        ax.spines[
            "top"
        ].set_visible(False)

        ax.spines[
            "right"
        ].set_visible(False)

    # =====================================================
    # Optional common percentage scale
    # =====================================================
    # valid_plot_axes = [
    #     axis
    #     for axis in plotted_axes
    #     if axis.has_data()
    # ]

    # if (
    #     same_y_scale
    #     and valid_plot_axes
    # ):
    #     y_min_values = []
    #     y_max_values = []

    #     for axis in valid_plot_axes:
    #         ymin, ymax = (
    #             axis.get_ylim()
    #         )

    #         y_min_values.append(
    #             ymin
    #         )

    #         y_max_values.append(
    #             ymax
    #         )

    #     common_ymin = min(
    #         y_min_values
    #     )

    #     common_ymax = max(
    #         y_max_values
    #     )

    #     if symmetric_y_axis:
    #         common_abs_limit = max(
    #             abs(common_ymin),
    #             abs(common_ymax),
    #         )

    #         common_abs_limit *= (
    #             1.0
    #             + y_axis_padding_fraction
    #         )

    #         common_ymin = (
    #             -common_abs_limit
    #         )

    #         common_ymax = (
    #             common_abs_limit
    #         )

    #     else:
    #         common_range = (
    #             common_ymax
    #             - common_ymin
    #         )

    #         if common_range > 0:
    #             common_ymin -= (
    #                 y_axis_padding_fraction
    #                 * common_range
    #             )

    #             common_ymax += (
    #                 y_axis_padding_fraction
    #                 * common_range
    #             )

    #     for axis in valid_plot_axes:
    #         axis.set_ylim(
    #             common_ymin,
    #             common_ymax,
    #         )
    # =====================================================
    # Fixed common percentage scale for all left y-axes
    # =====================================================
    common_ymin = common_ymin_value
    common_ymax = common_ymax_value
    
    for axis in plotted_axes:
        if axis.has_data():
            axis.set_ylim(
                common_ymin,
                common_ymax,
            )
    # =====================================================
    # Automatic endpoint labels
    # =====================================================
    
    if show_endpoint_labels:
        for plot_index, endpoint_items in (
            endpoint_data_by_index.items()
        ):
            if not endpoint_items:
                continue
            ax = axes[plot_index]
            # These are now the final fixed limits,
            # for example 0% to 10%.
            ymin, ymax = ax.get_ylim()
            endpoint_items = sorted(
                endpoint_items,
                key=lambda item: item["y"],
            )
            chosen_label_positions = []
            for item in endpoint_items:
                endpoint_y = item["y"]
                candidate_positions = []
                # Try positions above and below
                # the endpoint.
                for distance in (
                    0.60,
                    1.20,
                    1.80,
                    2.40,
                ):
                    above = endpoint_y + distance
                    below = endpoint_y - distance
    
                    if (
                        ymin + 0.40
                        <= above
                        <= ymax - 0.40
                    ):
                        candidate_positions.append(
                            above
                        )
                    if (
                        ymin + 0.40
                        <= below
                        <= ymax - 0.40
                    ):
                        candidate_positions.append(
                            below
                        )
                if not candidate_positions:
                    candidate_positions = [
                        max(
                            ymin + 0.40,
                            min(
                                endpoint_y,
                                ymax - 0.40,
                            ),
                        )
                    ]
                other_endpoint_positions = [
                    other["y"]
                    for other in endpoint_items
                    if other is not item
                ]
                occupied_positions = (
                    other_endpoint_positions
                    + chosen_label_positions
                )
                def position_score(candidate):
                    if not occupied_positions:
                        return 999.0
                    return min(
                        abs(
                            candidate - occupied
                        )
                        for occupied
                        in occupied_positions
                    )
                label_y = max(
                    candidate_positions,
                    key=position_score,
                )
                chosen_label_positions.append(
                    label_y
                )
                label = (
                    f"{item['group']}, "
                    f"{item['participation']}"
                )
                # Keep the label inside the panel,
                # to the left of the endpoint.
                label_x = item["x"] - 0.12
                ax.annotate(
                    label,
                    xy=(
                        item["x"],
                        item["y"],
                    ),
                    xytext=(
                        label_x,
                        label_y,
                    ),
                    fontsize=tick_label_fontsize,
                    color=item["color"],
                    fontweight="bold",
                    ha="right",
                    va="center",
                    arrowprops={
                        "arrowstyle": "->",
                        "color": item["color"],
                        "linewidth": 1.2,
                        "shrinkA": 0,
                        "shrinkB": 4,
                    },
                    zorder=10,
                )
    # =====================================================
    # Add right y-axes after left limits are finalized
    #
    # No line is plotted on these axes.
    # =====================================================
    right_axes = []

    for plot_index, region in enumerate(
        region_codes
    ):
        ax = axes[
            plot_index
        ]
        if not ax.has_data():
            right_axes.append(
                None
            )
            continue
        panel_df = (
            panel_data_by_index[
                plot_index
            ]
        )
        ax2 = _add_second_y_axis(
            ax=ax,
            panel_df=panel_df,
        )
        right_axes.append(
            ax2
        )
    # =====================================================
    # Turn off unused panels between the final regional
    # panel and the bottom-right legend panel
    # =====================================================
    legend_axis_index = (
        len(axes) - 1
    )
    for extra_axis_index in range(
        n_regions,
        legend_axis_index,
    ):
        axes[
            extra_axis_index
        ].axis(
            "off"
        )
    # =====================================================
    # Legend in row 2, column 3
    # =====================================================
    legend_ax = axes[
        legend_axis_index
    ]

    legend_ax.set_facecolor(
        "white"
    )

    legend_ax.axis(
        "off"
    )

    if show_endpoint_labels:

        # No box legend when endpoint labels are enabled.
        legend_ax.axis("off")

    else:
    
        legend_handles = []
    
        for group, participation in series_order:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=group_colors.get(
                        group,
                        "gray",
                    ),
                    linestyle="-",
                    marker=marker_map.get(
                        participation,
                        "o",
                    ),
                    markerfacecolor=group_colors.get(
                        group,
                        "gray",
                    ),
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    linewidth=2.3,
                    markersize=7,
                    label=(
                        f"{group}, "
                        f"{participation}"
                    ),
                )
            )
    
        legend = legend_ax.legend(
            handles=legend_handles,
            loc="center",
            frameon=True,
            fontsize=legend_fontsize,
            ncol=1,
            title="Charging programs",
            borderpad=1.0,
            labelspacing=1.0,
            handlelength=2.4,
        )
    
        legend.get_frame().set_facecolor(
            "white"
        )
    
        legend.get_frame().set_edgecolor(
            "black"
        )
    
        legend.get_frame().set_linewidth(
            1.0
        )
    
        legend.get_title().set_fontsize(
            legend_fontsize + 1
        )
    
        legend.get_title().set_fontweight(
            "bold"
        )
    
        legend.get_title().set_color(
            "black"
        )
    
        for legend_text in legend.get_texts():
            legend_text.set_color(
                "black"
            )

    # =====================================================
    # Figure title
    # =====================================================
    if figure_title is None:
        figure_title = (
            "Total system cost reduction across RPS targets\n"
            "Left axis = percentage reduction, "
            "right axis = million-dollar reduction; "
            f"battery capex = ${batt_capex}/kWh; "
            f"main line = {main_adoption_label} adoption"
        )

    fig.suptitle(
        figure_title,
        fontsize=(
            figure_title_fontsize
        ),
        color="black",
        y=0.985,
    )

    fig.subplots_adjust(
        top=0.94,
        bottom=0.05,
        left=0.06,
        right=0.97,
        hspace=0.42,
        wspace=0.63,
    )

    # =====================================================
    # Save
    # =====================================================
    if save_path is not None:
        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
    # =====================================================
    # Missing-series warning
    # =====================================================
    if missing_combinations:
        print(
            "\nWarning: these requested series "
            "were not available:"
        )

        for item in sorted(
            set(missing_combinations)
        ):
            print(f"  {item}")

    return (df_delta_all,fig,axes,)

