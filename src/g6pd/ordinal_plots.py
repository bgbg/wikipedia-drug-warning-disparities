"""Plots for the ordinal-aware analysis pipeline.

Uses the Wong colorblind-safe palette (the project's CODE_COLORS convention).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

CODE_COLORS = {
    0: "#CC79A7",  # muted pink  (Wong palette)
    1: "#E69F00",  # orange
    2: "#0072B2",  # blue
}

CODE_LABELS = {
    0: "0 — no mention",
    1: "1 — mentioned, no clinical action",
    2: "2 — explicit contraindication",
}

LANG_LABELS = {"en": "EN", "he": "HE", "ar": "AR", "ru": "RU"}
LANG_ORDER = ("en", "he", "ar", "ru")
TOPIC_LABELS = {"g6pd": "G6PD", "teratogenicity": "Teratogenicity"}


def stacked_composition_plot(
    observed: pd.DataFrame,
    *,
    title: str = "Observed warning-code composition by language and topic",
    figsize: tuple[float, float] = (10, 4.5),
) -> Figure:
    """Stacked bar of observed {0, 1, 2} proportions per (language, topic).

    `observed` is the output of `observed_composition(df)`: rows with
    columns (language, topic, code, count, proportion, n_obs).
    """
    topics = list(TOPIC_LABELS.keys())
    fig, axes = plt.subplots(1, len(topics), figsize=figsize, sharey=True)

    for ax, tp in zip(axes, topics):
        sub = observed[observed["topic"] == tp]
        if sub.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(LANG_ORDER))
        bottom = np.zeros(len(LANG_ORDER))
        for code in (0, 1, 2):
            vals = []
            for lg in LANG_ORDER:
                cell = sub[(sub["language"] == lg) & (sub["code"] == code)]
                vals.append(float(cell["proportion"].iloc[0]) if len(cell) else 0.0)
            vals_arr = np.asarray(vals)
            ax.bar(
                x,
                vals_arr,
                bottom=bottom,
                color=CODE_COLORS[code],
                label=CODE_LABELS[code] if tp == topics[0] else None,
                edgecolor="white",
                linewidth=0.6,
            )
            bottom += vals_arr
        labels = []
        for lg in LANG_ORDER:
            n_obs = sub[sub["language"] == lg]["n_obs"]
            n_obs_val = int(n_obs.iloc[0]) if len(n_obs) else 0
            labels.append(f"{LANG_LABELS[lg]}\n(n={n_obs_val})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(TOPIC_LABELS[tp])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion of articles" if tp == topics[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def posterior_predictive_overlay_plot(
    observed: pd.DataFrame,
    posterior_predictive: pd.DataFrame,
    *,
    title: str = "Posterior predictive composition vs. observed",
    figsize: tuple[float, float] = (10, 4.5),
) -> Figure:
    """Stacked bar of observed proportions with posterior-predictive 95% bands.

    `posterior_predictive` is from `posterior_predictive_composition(idata, df)`:
    columns (language, topic, code, mean, hdi_low, hdi_high, n_obs).
    Bands are drawn as vertical lines on the running stack.
    """
    topics = list(TOPIC_LABELS.keys())
    fig, axes = plt.subplots(1, len(topics), figsize=figsize, sharey=True)

    for ax, tp in zip(axes, topics):
        obs = observed[observed["topic"] == tp]
        post = posterior_predictive[posterior_predictive["topic"] == tp]
        if obs.empty or post.empty:
            ax.set_visible(False)
            continue
        x = np.arange(len(LANG_ORDER))
        bottom = np.zeros(len(LANG_ORDER))
        for code in (0, 1, 2):
            obs_vals = []
            for lg in LANG_ORDER:
                obs_cell = obs[(obs["language"] == lg) & (obs["code"] == code)]
                obs_vals.append(float(obs_cell["proportion"].iloc[0]) if len(obs_cell) else 0.0)
            obs_arr = np.asarray(obs_vals)
            ax.bar(
                x,
                obs_arr,
                bottom=bottom,
                color=CODE_COLORS[code],
                label=CODE_LABELS[code] if tp == topics[0] else None,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
            )
            bottom_after = bottom + obs_arr

            for i, lg in enumerate(LANG_ORDER):
                cell_post = post[(post["language"] == lg) & (post["code"] == code)]
                if cell_post.empty:
                    continue
                low = float(cell_post["hdi_low"].iloc[0])
                high = float(cell_post["hdi_high"].iloc[0])
                low_pos = bottom[i] + low
                high_pos = bottom[i] + high
                ax.vlines(
                    x[i] + 0.18,
                    low_pos,
                    high_pos,
                    color="black",
                    linewidth=1.2,
                    alpha=0.7,
                )
                ax.hlines(
                    [low_pos, high_pos],
                    x[i] + 0.13,
                    x[i] + 0.23,
                    color="black",
                    linewidth=1.2,
                    alpha=0.7,
                )

            bottom = bottom_after

        labels = []
        for lg in LANG_ORDER:
            cell_obs = obs[obs["language"] == lg]["n_obs"]
            n_obs_val = int(cell_obs.iloc[0]) if len(cell_obs) else 0
            labels.append(f"{LANG_LABELS[lg]}\n(n={n_obs_val})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(TOPIC_LABELS[tp])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Proportion of articles" if tp == topics[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.02, fontsize=12)
    fig.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    return fig


def or_forest_plot(
    or_table: pd.DataFrame,
    *,
    title: str = "Cumulative odds ratios vs. EN (Bayesian posterior, 95% CrI)",
    figsize: tuple[float, float] = (8, 4.5),
) -> Figure:
    """Forest plot of cumulative ORs vs. EN per (language, topic).

    `or_table` is from `cumulative_or_vs_reference(idata)`: columns
    (language, topic, mean, sd, hdi_low, hdi_high, prob_or_gt_1).

    OR < 1 means the language warns less strongly than EN. The plot
    uses log-scale on the OR axis.
    """
    topics = list(TOPIC_LABELS.keys())
    fig, ax = plt.subplots(figsize=figsize)

    rows = []
    for tp in topics:
        for lg in LANG_ORDER:
            if lg == "en":
                continue
            cell = or_table[(or_table["language"] == lg) & (or_table["topic"] == tp)]
            if cell.empty:
                continue
            rows.append(
                {
                    "label": f"{LANG_LABELS[lg]} ({TOPIC_LABELS[tp]})",
                    "topic": tp,
                    "mean": float(cell["mean"].iloc[0]),
                    "low": float(cell["hdi_low"].iloc[0]),
                    "high": float(cell["hdi_high"].iloc[0]),
                }
            )
    if not rows:
        ax.text(0.5, 0.5, "no OR data", ha="center", va="center")
        return fig

    rows_df = pd.DataFrame(rows)
    rows_df["topic_order"] = rows_df["topic"].apply(topics.index)
    rows_df = rows_df.sort_values(["topic_order", "label"])
    y_pos = np.arange(len(rows_df))[::-1]

    topic_colors = {"g6pd": "#0072B2", "teratogenicity": "#D55E00"}
    for y, (_, r) in zip(y_pos, rows_df.iterrows()):
        ax.errorbar(
            r["mean"],
            y,
            xerr=[[r["mean"] - r["low"]], [r["high"] - r["mean"]]],
            fmt="o",
            color=topic_colors[r["topic"]],
            ecolor=topic_colors[r["topic"]],
            elinewidth=1.5,
            capsize=4,
            markersize=6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(rows_df["label"])
    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative odds ratio vs. EN (log scale)")
    ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)

    handles = [
        plt.Line2D([0], [0], marker="o", color=color, linestyle="", markersize=6, label=TOPIC_LABELS[tp])
        for tp, color in topic_colors.items()
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def prior_predictive_summary_plot(
    prior_or_samples: Mapping[str, np.ndarray],
    *,
    title: str = "Prior-predictive distribution of cumulative ORs",
    figsize: tuple[float, float] = (8, 4),
) -> Figure:
    """Histogram of prior-predictive cumulative ORs by prior width.

    `prior_or_samples` maps a label (e.g., 'sd=1.5') to a 1D array of
    sampled ORs from the prior alone. Used to confirm priors imply a
    plausible OR range before fitting.
    """
    fig, ax = plt.subplots(figsize=figsize)
    bins = np.logspace(-2, 2, 60)
    for label, samples in prior_or_samples.items():
        ax.hist(
            samples,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative OR (log scale)")
    ax.set_ylabel("Frequency (prior samples)")
    ax.axvline(1.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
