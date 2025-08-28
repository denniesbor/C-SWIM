"""Visualization module for economic analysis results"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from configs import USE_ALPHA_BETA_SCENARIO, FIGURES_DIR, setup_logger

logger = setup_logger("viz_spwio")


def plot_socio_economic_impact(results_data, confidence_df, model_type="io"):
    """Plot socio-economic impact analysis showing establishments, population, and economic impacts"""

    if USE_ALPHA_BETA_SCENARIO:
        scenario_map = {
            "gic_75yr": 75,
            "gic_100yr": 100,
            "gic_125yr": 125,
            "gic_150yr": 150,
            "gic_175yr": 175,
            "gic_200yr": 200,
            "gic_225yr": 225,
            "gic_250yr": 250,
        }
    else:
        scenario_map = {
            "50-year": 50,
            "75-year": 75,
            "100-year": 100,
            "125-year": 125,
            "150-year": 150,
            "175-year": 175,
            "200-year": 200,
            "225-year": 225,
            "250-year": 250,
        }

    econ_filtered = results_data[results_data["scenario"] != "gannon-year"]
    conf_filtered = confidence_df[~confidence_df["scenario"].str.contains("gannon")]

    est_data = conf_filtered[conf_filtered["variable"] == "EST_TOTAL"].set_index(
        "scenario"
    )
    est_x = [
        scenario_map[s.replace("e_", "").replace("-hazard A/ph", "")]
        for s in est_data.index
    ]
    est_mean = est_data["mean"].astype(int) / 1000
    est_err_low = est_mean - est_data["p5"].astype(int) / 1000
    est_err_high = est_data["p95"].astype(int) / 1000 - est_mean
    est_sorted = sorted(zip(est_x, est_mean, est_err_low, est_err_high))
    est_x_sorted, est_mean_sorted, est_err_low_sorted, est_err_high_sorted = zip(
        *est_sorted
    )

    pop_data = conf_filtered[conf_filtered["variable"] == "POP_AFFECTED"].set_index(
        "scenario"
    )
    pop_x = [
        scenario_map[s.replace("e_", "").replace("-hazard A/ph", "")]
        for s in pop_data.index
    ]
    pop_mean = pop_data["mean"] / 1e6
    pop_err_low = pop_mean - pop_data["p5"] / 1e6
    pop_err_high = pop_data["p95"] / 1e6 - pop_mean
    pop_sorted = sorted(zip(pop_x, pop_mean, pop_err_low, pop_err_high))
    pop_x_sorted, pop_mean_sorted, pop_err_low_sorted, pop_err_high_sorted = zip(
        *pop_sorted
    )

    direct_summary = (
        econ_filtered.groupby(["scenario", "confidence"])["direct_shock"]
        .sum()
        .abs()
        .unstack()
    )
    direct_x = [scenario_map[s] for s in direct_summary.index]
    direct_mean = direct_summary["mean"] / 1000
    direct_err_low = direct_mean - direct_summary["p5"] / 1000
    direct_err_high = direct_summary["p95"] / 1000 - direct_mean
    direct_sorted = sorted(zip(direct_x, direct_mean, direct_err_low, direct_err_high))
    (
        direct_x_sorted,
        direct_mean_sorted,
        direct_err_low_sorted,
        direct_err_high_sorted,
    ) = zip(*direct_sorted)

    total_summary = (
        econ_filtered.groupby(["scenario", "confidence"])["total_impact"]
        .sum()
        .abs()
        .unstack()
    )
    total_x = [scenario_map[s] for s in total_summary.index]
    total_mean = total_summary["mean"] / 1000
    total_err_low = total_mean - total_summary["p5"] / 1000
    total_err_high = total_summary["p95"] / 1000 - total_mean
    total_sorted = sorted(zip(total_x, total_mean, total_err_low, total_err_high))
    total_x_sorted, total_mean_sorted, total_err_low_sorted, total_err_high_sorted = (
        zip(*total_sorted)
    )

    max_direct = max(np.array(direct_mean_sorted) + np.array(direct_err_high_sorted))
    max_total = max(np.array(total_mean_sorted) + np.array(total_err_high_sorted))
    econ_ylim = max(max_direct, max_total) * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].errorbar(
        est_x_sorted,
        est_mean_sorted,
        yerr=[est_err_low_sorted, est_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="green",
    )
    axes[0, 0].plot(est_x_sorted, est_mean_sorted, "-", color="green", alpha=0.7)
    axes[0, 0].set_ylabel(r"Establishments (Thousands)")
    axes[0, 0].set_title("(a) Business Impact")

    axes[0, 1].errorbar(
        pop_x_sorted,
        pop_mean_sorted,
        yerr=[pop_err_low_sorted, pop_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="red",
    )
    axes[0, 1].plot(pop_x_sorted, pop_mean_sorted, "-", color="red", alpha=0.7)
    axes[0, 1].set_ylabel(r"Population (Millions)")
    axes[0, 1].set_title("(b) Population Impact")

    axes[1, 0].errorbar(
        direct_x_sorted,
        direct_mean_sorted,
        yerr=[direct_err_low_sorted, direct_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="purple",
    )
    axes[1, 0].plot(direct_x_sorted, direct_mean_sorted, "-", color="purple", alpha=0.7)
    axes[1, 0].set_ylabel(r"Direct Impact (\$Bn/day)")
    axes[1, 0].set_xlabel("Storm Return Period (years)")
    axes[1, 0].set_title("(c) Direct Economic Loss")
    axes[1, 0].set_ylim(0, econ_ylim)

    axes[1, 1].errorbar(
        total_x_sorted,
        total_mean_sorted,
        yerr=[total_err_low_sorted, total_err_high_sorted],
        fmt="o",
        capsize=4,
        capthick=2,
        color="blue",
    )
    axes[1, 1].plot(total_x_sorted, total_mean_sorted, "-", color="blue", alpha=0.7)
    axes[1, 1].set_ylabel(r"Total Impact (\$Bn/day)")
    axes[1, 1].set_xlabel("Storm Return Period (years)")
    axes[1, 1].set_title("(d) Total Economic Loss")
    axes[1, 1].set_ylim(0, econ_ylim)

    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.patch.set_facecolor("#F0F0F0")
    fig.savefig(
        FIGURES_DIR / f"economic_impact_{model_type}.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES_DIR / f"economic_impact_{model_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def plot_vuln_trafos(vuln_data, df_lines):
    """Plot vulnerable transformers/substations on a map"""

    def setup_map(ax, spatial_extent=[-120, -75, 25, 50]):
        ax.set_extent(spatial_extent, ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="#F0F0F0")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="grey")
        ax.add_feature(cfeature.STATES, linewidth=0.2, edgecolor="darkgrey")
        ax.add_feature(cfeature.LAKES, alpha=0.5, linewidth=0.5, edgecolor="grey")
        gl = ax.gridlines(
            draw_labels=False, linewidth=0.2, color="grey", alpha=0.5, linestyle="--"
        )
        return ax

    projection = ccrs.LambertConformal(central_longitude=-98, central_latitude=39.5)

    if USE_ALPHA_BETA_SCENARIO:
        scenarios_to_plot = [
            "gic_100yr_mean_prediction",
            "gic_150yr_mean_prediction",
            "gic_200yr_mean_prediction",
            "gic_250yr_mean_prediction",
        ]
    else:
        scenarios_to_plot = [
            "e_100-year-hazard A/ph",
            "e_150-year-hazard A/ph",
            "e_200-year-hazard A/ph",
            "e_250-year-hazard A/ph",
        ]

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 6), subplot_kw={"projection": projection}
    )
    axes = axes.flatten()

    voltage_levels = [161, 230, 345, 500, 765]
    voltage_colors = ["purple", "blue", "green", "orange", "maroon"]
    voltage_widths = [0.2, 0.3, 0.4, 0.5, 0.6]
    voltage_color_map = dict(zip(voltage_levels, voltage_colors))
    voltage_width_map = dict(zip(voltage_levels, voltage_widths))

    all_coords = vuln_data.groupby("sub_id")[["latitude", "longitude"]].first()

    for i, scenario in enumerate(scenarios_to_plot):
        ax = axes[i]

        if i == 0:
            ax.text(
                0,
                1.2,
                "Substations Exceeding 50% Failure Probability",
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="center",
            )

        scenario_vulnerable = vuln_data[
            (vuln_data["scenario"] == scenario) & (vuln_data["mean_failure_prob"] > 0.4)
        ]["sub_id"].unique()

        if len(scenario_vulnerable) == 0:
            vulnerable_coords = all_coords[all_coords.index.isin(scenario_vulnerable)]
            padding = 3
            min_lon = vulnerable_coords["longitude"].min() - padding
            max_lon = vulnerable_coords["longitude"].max() + padding
            min_lat = vulnerable_coords["latitude"].min() - padding
            max_lat = vulnerable_coords["latitude"].max() + padding
            extent = [min_lon, max_lon, min_lat, max_lat]
        else:
            extent = [-120, -75, 25, 50]

        ax = setup_map(ax, extent)

        non_vulnerable = all_coords[~all_coords.index.isin(scenario_vulnerable)]
        ax.scatter(
            non_vulnerable["longitude"],
            non_vulnerable["latitude"],
            c="black",
            s=0.5,
            alpha=0.4,
            transform=ccrs.PlateCarree(),
            zorder=3,
        )

        vulnerable_coords = all_coords[all_coords.index.isin(scenario_vulnerable)]
        ax.scatter(
            vulnerable_coords["longitude"],
            vulnerable_coords["latitude"],
            s=100,
            facecolors="red",
            edgecolors="darkred",
            alpha=0.6,
            linewidths=1,
            transform=ccrs.PlateCarree(),
            zorder=4,
        )

        line_coords = [list(geom.coords) for geom in df_lines["geometry"]]
        line_colors = [voltage_color_map[v] for v in df_lines["V"]]
        line_widths = [voltage_width_map[v] for v in df_lines["V"]]
        lc = LineCollection(
            line_coords,
            linewidths=line_widths,
            alpha=0.8,
            colors=line_colors,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
        ax.add_collection(lc)

        scenario_name = scenario.replace("e_", "").replace("-hazard A/ph", "")
        ax.set_title(f"({chr(97 + i)}) {scenario_name} Storm", fontsize=10, loc="left")

    legend_elements = []
    for voltage in voltage_levels:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                color=voltage_color_map[voltage],
                linewidth=voltage_width_map[voltage],
                label=f"{voltage} kV",
                alpha=0.6,
            )
        )
    legend_elements.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markersize=4,
            alpha=0.4,
            label="Other Substations",
        )
    )

    legend = fig.legend(
        handles=legend_elements,
        loc="lower center",
        frameon=False,
        fontsize=9,
        ncol=6,
        bbox_to_anchor=(0.5, -0.04),
    )

    for ax in axes.flat:
        ax.spines["geo"].set_visible(False)
        ax.set_facecolor("#F0F0F0")

    fig.patch.set_facecolor("#F0F0F0")

    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "vulnerable_trafos.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "vulnerable_trafos.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_econo_naics(econ_results, model_type="io"):
    """Plot economic impacts by NAICS sector"""

    SCALE = 1_000
    LABEL = "Impact (Bn $/day)"
    scenarios = ["100-year", "150-year", "200-year", "250-year"]

    scenario_250_mean = econ_results[
        (econ_results.scenario == "250-year") & (econ_results.confidence == "mean")
    ].set_index("sector")
    sector_order = (
        scenario_250_mean.total_impact.abs().sort_values(ascending=False).index
    )

    sector_labels_full = {
        "AGR": "Agriculture &\nForestry",
        "MINING": "Mining &\nOil/Gas Extraction",
        "UTIL_CONST": "Utilities &\nConstruction",
        "MANUF": "Manufacturing",
        "TRADE_TRANSP": "Trade &\nTransportation",
        "INFO": "Information",
        "FIRE": "Finance &\nReal Estate",
        "PROF_OTHER": "Professional &\nOther Services",
        "EDUC_ENT": "Education &\nEntertainment",
        "G": "Government",
    }

    short_labels = [sector_labels_full[s] for s in sector_order]

    scenario_250_p95 = econ_results[
        (econ_results.scenario == "250-year") & (econ_results.confidence == "p95")
    ].set_index("sector")
    max_x = scenario_250_p95.total_impact.abs().max() / SCALE * 1.1
    baseline_gap = 0.02 * max_x

    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True)

    for i, scen in enumerate(scenarios):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        df = econ_results[econ_results.scenario == scen]
        mean = df[df.confidence == "mean"].set_index("sector").reindex(sector_order)
        p5 = df[df.confidence == "p5"].set_index("sector").reindex(sector_order)
        p95 = df[df.confidence == "p95"].set_index("sector").reindex(sector_order)

        direct = mean.direct_shock.abs() / SCALE
        multiplier = mean.multiplier_effect.abs() / SCALE
        mean_tot = mean.total_impact.abs() / SCALE

        p5_abs = p5.total_impact.abs() / SCALE
        p95_abs = p95.total_impact.abs() / SCALE

        lower_err = np.abs(mean_tot - np.minimum(p5_abs, p95_abs))
        upper_err = np.abs(np.maximum(p5_abs, p95_abs) - mean_tot)

        y = np.arange(len(sector_order))[::-1]

        ax.barh(
            y,
            direct,
            left=baseline_gap,
            color="darkred",
            alpha=0.8,
            height=0.8,
            zorder=3,
        )
        ax.barh(
            y,
            multiplier,
            left=direct + baseline_gap,
            color="lightcoral",
            alpha=0.8,
            height=0.8,
            zorder=3,
        )

        ax.errorbar(
            mean_tot + baseline_gap,
            y,
            xerr=[lower_err, upper_err],
            fmt="none",
            color="black",
            capsize=2,
            capthick=1,
            elinewidth=1,
            zorder=4,
        )

        ax.set_xlim(0, max_x)
        ax.set_ylim(-0.5, len(sector_order) - 0.5)
        ax.grid(True, axis="x", linewidth=0.25, color="0.9", zorder=0)
        ax.spines["left"].set_color("0.3")
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["left"].set_zorder(5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        direct_mean = direct.sum()
        direct_low = p5.direct_shock.abs().sum() / SCALE
        direct_high = p95.direct_shock.abs().sum() / SCALE
        direct_err = (direct_high - direct_low) / 2

        indirect_mean = multiplier.sum()
        indirect_low = p5.multiplier_effect.abs().sum() / SCALE
        indirect_high = p95.multiplier_effect.abs().sum() / SCALE
        indirect_err = (indirect_high - indirect_low) / 2

        total_mean = mean_tot.sum()
        total_low = p5.total_impact.abs().sum() / SCALE
        total_high = p95.total_impact.abs().sum() / SCALE
        total_err = (total_high - total_low) / 2

        sx, sy, dy = 0.6, 0.5, 0.07
        ax.text(
            sx,
            sy,
            scen,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            sx,
            sy - dy,
            f"Direct: ${direct_mean:.1f} ± {direct_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 2 * dy,
            f"Indirect: ${indirect_mean:.1f} ± {indirect_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.text(
            sx,
            sy - 3 * dy,
            f"Total: ${total_mean:.1f} ± {total_err:.1f} Bn",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

        ax.add_patch(
            Rectangle(
                (sx, sy - 4.7 * dy),
                0.04,
                0.03,
                facecolor="darkred",
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            sy - 4.7 * dy + 0.015,
            "Direct Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )
        ax.add_patch(
            Rectangle(
                (sx, sy - 5.7 * dy),
                0.04,
                0.03,
                facecolor="lightcoral",
                alpha=0.8,
                transform=ax.transAxes,
            )
        )
        ax.text(
            sx + 0.05,
            sy - 5.7 * dy + 0.015,
            "Indirect Impact",
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
        )

        ax.set_yticks(y)

        if col == 0:
            ax.set_yticklabels(short_labels, fontsize=10)
        else:
            ax.set_yticklabels([])

        if row == 1:
            ax.set_xlabel(LABEL, fontsize=11)
        else:
            ax.tick_params(labelbottom=False)

        ax.tick_params(labelsize=8)
        ax.set_facecolor("#F0F0F0")

    fig.patch.set_facecolor("#F0F0F0")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}.pdf", dpi=300, bbox_inches="tight"
    )
    fig.savefig(
        FIGURES_DIR / f"indirect_impact_{model_type}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
