import os
import sys
import time
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_lmps(ax, graph, solution):

    for source, node in solution._node.items():

        ax.plot(
            np.array(node['clearing_price']) * 3.6e9,
            # np.array(node['clearing_price']),
            label = source,
        )

    kw = {
        'facecolor': 'whitesmoke',
        'ylabel': 'Region Marginal Price [$ / MWh]',
    }

    ax.set(**kw)

    kw = {
        'ls': '--',
    }

    ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    ax.legend(**kw)

    return ax

def plot_base_loads(ax, graph, solution):

    for source, node in solution._node.items():

        ax.plot(
            -np.array(node['assets'][f"base_load_{source}"]['net']) / 1e9,
            label = source,
        )
        
        # ax.set_title('Regional Base Load')

    kw = {
        'facecolor': 'whitesmoke',
        'ylabel': 'Regional Base Load [GW]',
        # 'ylabel': 'Power [GW]',
        'xlabel': 'Time [h]'
    }

    _ = ax.set(**kw)

    kw = {
        'ls': '--',
    }

    _ = ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    _ = ax.legend(**kw)

    return ax

def plot_total_generation(ax, graph, solution):

    for source, node in solution._node.items():

        values = np.vstack(
            [v["net"] for k, v in node['assets'].items() if "base" not in k]
        )

        ax.plot(
            values.sum(axis = 0) / 1e9,
            label = source,
        )

    kw = {
        'facecolor': 'whitesmoke',
        'ylabel': 'Regional Total Generation [GW]',
        # 'ylabel': 'Power [GW]',
        'xlabel': 'Time [h]'
    }

    _ = ax.set(**kw)

    kw = {
        'ls': '--',
    }

    _ = ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    _ = ax.legend(**kw)

    return ax

def plot_net_generation(ax, graph, solution):

    for source, node in solution._node.items():

        values = np.vstack(
            [v["net"] for k, v in node['assets'].items()]
        )

        ax.plot(
            values.sum(axis = 0) / 1e9,
            label = source,
        )

    kw = {
        'facecolor': 'whitesmoke',
        'ylabel': 'Regional Net Generation [GW]',
        # 'ylabel': 'Power [GW]',
        'xlabel': 'Time [h]'
    }

    _ = ax.set(**kw)

    kw = {
        'ls': '--',
    }

    _ = ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    _ = ax.legend(**kw)

    return ax

def plot_generation_by_type(ax, graph, solution):

    gen_amounts = {'wastage': [], 'shortfall': []}

    for source, node in solution._node.items():

        gen_amounts['wastage'].append(node['wastage'])
        gen_amounts['shortfall'].append(node['shortfall'])

    for source, node in solution._node.items():

        for handle, asset in node['assets'].items():

            asset_type = graph._node[source]['assets'][handle]['type']

            if asset_type not in gen_amounts:

                gen_amounts[asset_type] = (
                    [node['assets'][handle]['net']]
                )

            else:

                gen_amounts[asset_type].append(
                    [node['assets'][handle]['net']]
                )
                

    for key, val in gen_amounts.items():

        gen_amounts[key] = np.vstack(val).sum(axis = 0) / 1e9

    gen_amounts['load'] *= -1

    for key, val in gen_amounts.items():

        ax.plot(
            val,
            label = key,
            ls = '--' if key in ['wastage', 'shortfall'] else None,
        )

    kw = {
        'facecolor': 'whitesmoke',
        'ylabel': 'Power [GW]',
    }

    _ = ax.set(**kw)

    kw = {
        'ls': '--',
    }

    _ = ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    _ = ax.legend(**kw)

    return ax

def plot_generation_by_fuel(ax, graph, solution):
    # -------------------------
    # Collect timeseries by fuel
    # -------------------------
    gen_amounts = {"wastage": [], "shortfall": []}

    # node-level slack terms
    for source, node in solution._node.items():
        gen_amounts["wastage"].append(node["wastage"])
        gen_amounts["shortfall"].append(node["shortfall"])

    # asset-level generation by fuel
    for source, node in solution._node.items():
        for handle, asset in node["assets"].items():
            asset_fuel = graph._node[source]["assets"][handle].get("fuel", None)
            if asset_fuel is None:
                continue

            gen_amounts.setdefault(asset_fuel, [])
            gen_amounts[asset_fuel].append(asset["net"])

    # Convert lists -> summed vector [GW]
    for k, v in list(gen_amounts.items()):
        # ensure each entry is a 1D array
        arrs = [np.asarray(x).ravel() for x in v]
        gen_amounts[k] = np.vstack(arrs).sum(axis=0) / 1e9

    # -------------------------
    # Stable order + unique colors
    # -------------------------
    special = ["wastage", "shortfall"]
    fuels = sorted([k for k in gen_amounts.keys() if k not in special])
    keys = fuels + special  # plot fuels first, then slack terms

    # One unique color per key (deterministic)
    cmap = plt.get_cmap("tab20")  # good for categorical lines
    colors = {k: cmap(i % cmap.N) for i, k in enumerate(keys)}

    # -------------------------
    # Plot
    # -------------------------
    for k in keys:
        ax.plot(
            gen_amounts[k],
            label=k,
            color=colors[k],
            linestyle="--" if k in special else "-",
            linewidth=1.8 if k in special else 2.2,
            alpha=0.95,
        )

    ax.set(facecolor="whitesmoke", ylabel="Power [GW]")
    ax.grid(ls="--", alpha=0.35)
    ax.legend(fontsize="x-small", ncol=2, frameon=True)

    return ax

def plot_capex_by_type(ax, graph, solution):

    capex_amounts = {}

    for source, node in solution._node.items():

        for handle, asset in node['assets'].items():

            asset_type = graph._node[source]['assets'][handle]['type']

            if asset_type not in capex_amounts:

                capex_amounts[asset_type] = (
                    [node['assets'][handle]['capex']]
                )

            else:

                capex_amounts[asset_type].append(
                    [node['assets'][handle]['capex']]
                )

    for key, val in capex_amounts.items():

        capex_amounts[key] = np.vstack(val).sum(axis = 0)[0] / 1e9

    capex_amounts['battery'] /= (4 * 3600)

    kw = {
        'color': 'xkcd:seafoam',
        'ec': 'k',
    }

    ax.barh(list(capex_amounts.keys()), list(capex_amounts.values()), **kw)

    kw = {
        'facecolor': 'whitesmoke',
        'xlabel': 'Capacity Expansion [GW]',
    }

    _ = ax.set(**kw)

    kw = {
        'ls': '--',
    }

    _ = ax.grid(**kw)

    kw = {
        'fontsize': 'x-small',
    }

    _ = ax.legend(**kw)

    return ax

def plot_generation_by_fuel_plotly(graph, solution):

    gen_amounts = {"wastage": [], "shortfall": []}

    # -------------------------
    # Collect node-level slack
    # -------------------------
    for source, node in solution._node.items():
        gen_amounts["wastage"].append(node["wastage"])
        gen_amounts["shortfall"].append(node["shortfall"])

    # -------------------------
    # Collect asset generation
    # -------------------------
    for source, node in solution._node.items():
        for handle, asset in node["assets"].items():
            asset_fuel = graph._node[source]["assets"][handle].get("fuel", None)
            if asset_fuel is None:
                continue

            gen_amounts.setdefault(asset_fuel, [])
            gen_amounts[asset_fuel].append(asset["net"])

    # -------------------------
    # Sum and convert to GW
    # -------------------------
    for k, v in list(gen_amounts.items()):
        arrs = [np.asarray(x).ravel() for x in v]
        gen_amounts[k] = np.vstack(arrs).sum(axis=0) / 1e9

    # Stable order
    special = ["wastage", "shortfall"]
    fuels = sorted([k for k in gen_amounts.keys() if k not in special])
    keys = fuels + special

    # Color palette (categorical)
    colors = px.colors.qualitative.Dark24
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(keys)}

    # -------------------------
    # Build figure
    # -------------------------
    fig = go.Figure()

    for k in keys:
        fig.add_trace(
            go.Scatter(
                y=gen_amounts[k],
                mode="lines",
                name=k,
                line=dict(
                    color=color_map[k],
                    dash="dash" if k in special else "solid",
                    width=2,
                ),
                hovertemplate=
                    "<b>%{fullData.name}</b><br>" +
                    "Hour: %{x}<br>" +
                    "Generation: %{y:.2f} GW<extra></extra>"
            )
        )

    fig.update_layout(
        template="plotly_white",
        yaxis_title="Power [GW]",
        xaxis_title="Hour",
        legend_title="Fuel Type",
        hovermode="x unified",
        height=600,
        width=1000,
    )

    return fig