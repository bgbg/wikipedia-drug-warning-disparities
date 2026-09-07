"""Three framings of study-article Wikipedia traffic, one conclusion each.

Each figure's title states its finding as a declarative sentence; the plot is
built to substantiate that sentence and nothing else.

  A_exposure    the audited articles still drew 10.9M views last year
  B_not_medical the decline is general to reference content, not drug-specific
  C_disparity   exposure is grossly unequal across the four language editions

Usage:  python scripts/plot_pageviews.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from g6pd.config import DRUGS_CSV, FIGURES_DIR, RAW_DIR

WINDOW = ("202507", "202606")  # most recent 12 complete months, for the count
LAST_COMPLETE_MONTH = "202607"

LANG_ORDER = ("en", "he", "ar", "ru")
LANG_LABELS = {"en": "English", "he": "Hebrew", "ar": "Arabic", "ru": "Russian"}
TOPIC_ORDER = ("G6PD", "Teratogenicity", "QT prolongation", "Negative control")

# Wong colorblind-safe palette, matching src/g6pd/ordinal_plots.py so this
# figure set is consistent with the rest of the manuscript's figures.
LANG_COLORS = {"en": "#0072B2", "he": "#E69F00", "ar": "#009E73", "ru": "#CC79A7"}
TOPIC_COLORS = {
    "G6PD": "#0072B2",
    "Teratogenicity": "#E69F00",
    "QT prolongation": "#009E73",
    "Negative control": "#767676",
}
CONTROL_GREY = "#767676"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
    }
)


def despine(ax, left: bool = False) -> None:
    """Drop top/right spines (and optionally left). Replaces seaborn.despine."""
    for side in ("top", "right") + (("left",) if left else ()):
        ax.spines[side].set_visible(False)


def compact(v: float) -> str:
    """9_100_000 -> '9.1M'; 153_000 -> '153k'. Readable direct labels."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}k"
    return f"{v:.0f}"


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def title(fig, text: str, y: float = 1.0) -> None:
    """Declarative finding, not a description of the chart.

    va="bottom" at y=1.0 seats the title entirely ABOVE the figure box, so it
    cannot collide with a panel title or a horizontal y-label; bbox_inches
    ="tight" pulls it back into the saved image.
    """
    fig.suptitle(
        text, fontsize=13, fontweight="bold", x=0.0, ha="left", y=y, va="bottom"
    )


def ax_title(ax, text: str, pad: float = 34) -> None:
    """Title on a single-panel figure; pad clears the horizontal y-label."""
    ax.set_title(text, fontsize=13, fontweight="bold", loc="left", pad=pad)


def ylabel(ax, text: str, y: float = 1.0) -> None:
    """Horizontal, never rotated. y=0.5 centres it clear of a suptitle."""
    ax.set_ylabel(text, rotation=0, ha="right", ma="left", y=y, labelpad=6)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audited-drug pageviews and the matched non-medical control cohort.

    Drops the 7 PGx-only drugs: PGx-SCAR was dropped as an analysed topic on
    2026-07-01, so those articles are not part of what this study audits.
    """
    drugs = pd.read_csv(DRUGS_CSV)
    analysed = drugs[
        drugs[["cpic_risk", "pregnancy_risk", "qt_risk"]].notna().any(axis=1)
        | drugs[["cpic_risk", "pregnancy_risk", "qt_risk", "pgx_risk"]]
        .isna()
        .all(axis=1)
    ]
    # 40 drugs minus the 7 PGx-only; guards the risk-column filter above.
    assert len(analysed) == 33, f"expected 33 analysed drugs, got {len(analysed)}"

    def topic(row) -> str:
        if pd.notna(row.cpic_risk):
            return "G6PD"
        if pd.notna(row.pregnancy_risk):
            return "Teratogenicity"
        if pd.notna(row.qt_risk):
            return "QT prolongation"
        return "Negative control"

    pv = pd.read_csv(RAW_DIR / "pageviews.csv", dtype={"month": str})
    pv = pv[pv.drug.isin(analysed.drug) & (pv.month <= LAST_COMPLETE_MONTH)]
    pv = pv.merge(
        analysed.assign(topic=analysed.apply(topic, axis=1))[["drug", "topic"]],
        on="drug",
    )

    controls = pd.read_csv(RAW_DIR / "pageviews_controls.csv", dtype={"month": str})
    controls = controls[controls.month <= LAST_COMPLETE_MONTH]
    return pv, controls


def _months(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%Y%m")


def fig_a_exposure(pv: pd.DataFrame) -> None:
    """Conclusion: the decline is real, and the residual audience is still large."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    series = pv.groupby(["lang", "month"], as_index=False).views.sum()
    window_total = pv[pv.month.between(*WINDOW)].views.sum()

    for lang in LANG_ORDER:
        s = series[series.lang == lang].sort_values("month")
        x, y = _months(s.month), s.views
        ax.plot(x, y, color=LANG_COLORS[lang], lw=1.6)
        # Direct label at the line end, coloured to match — no legend.
        ax.text(
            x.iloc[-1] + pd.Timedelta(days=25),
            y.iloc[-1],
            f"{LANG_LABELS[lang]}  {compact(y.iloc[-1])}/mo",
            color=LANG_COLORS[lang],
            va="center",
            fontsize=9,
            clip_on=False,
        )

    ax.set_yscale("log")
    ax.set_ylim(2_000, 3_000_000)
    ax.yaxis.set_major_locator(FixedLocator([10_000, 100_000, 1_000_000]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    ax.set_xlim(right=pd.Timestamp("2026-11-01"))
    ylabel(ax, "Monthly\npageviews")

    total = series.groupby("month").views.sum().sort_index()
    drop = 100 - 100 * total.iloc[-12:].mean() / total.iloc[:12].mean()
    ax_title(
        ax,
        f"Traffic is down {drop:.0f}% since 2021, yet the audited articles\n"
        f"still drew {window_total / 1e6:.1f} million views in the last 12 months",
    )
    despine(ax)
    save(fig, "pageviews_A_exposure")


def _median_index(df: pd.DataFrame, entity: str, lang: str) -> pd.Series:
    """Median across articles of each article's own index (first 12 months = 100).

    Median-of-indices, not index-of-sums: a sum is dominated by the largest
    article (paracetamol alone is 12.7% of English drug traffic) and by
    news-driven spikes in any single article. The median tracks what the
    TYPICAL article did, which is the comparison the conclusion rests on.
    """
    wide = (
        df[df.lang == lang]
        .pivot_table(index="month", columns=entity, values="views", aggfunc="sum")
        .sort_index()
    )
    base = wide.iloc[:12].mean()
    return (100 * wide.div(base[base > 0], axis=1)).median(axis=1)


def fig_b_not_medical(pv: pd.DataFrame, controls: pd.DataFrame) -> None:
    """Conclusion: the decline is general to reference content, not drug-specific.

    Baseline is a matched cohort of stable non-medical reference articles, NOT
    the whole-wiki total. The whole-wiki total is invalid here: its composition
    churns as new and trending pages enter, so it holds flat while every
    established article declines, manufacturing a spurious "medical content is
    dying faster" gap.
    """
    fig, axes = plt.subplots(
        2, 2, figsize=(8.5, 5.5), sharex=True, constrained_layout=True
    )

    for ax, lang in zip(axes.flat, LANG_ORDER):
        a = _median_index(pv, "drug", lang)
        b = _median_index(controls, "article", lang)
        x = _months(pd.Series(a.index))

        ax.plot(x, b.to_numpy(), color=CONTROL_GREY, lw=1.4)
        ax.plot(x, a.to_numpy(), color=LANG_COLORS[lang], lw=1.6)

        ax.axhline(100, color="#CCCCCC", lw=0.6, zorder=0)
        ax.set_title(LANG_LABELS[lang], loc="left", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 175)
        ax.yaxis.set_major_locator(FixedLocator([0, 100]))
        despine(ax)

    # Label the two series once, on the first panel — the encoding repeats.
    lead = axes.flat[0]
    lead.text(
        pd.Timestamp("2021-09-01"), 163, "audited drug articles",
        color=LANG_COLORS["en"], fontsize=8.5, fontweight="bold",
    )
    lead.text(
        pd.Timestamp("2021-09-01"), 143, "non-medical reference articles",
        color=CONTROL_GREY, fontsize=8.5,
    )
    for ax in axes[:, 0]:
        ylabel(ax, "Median article,\nindexed to\n2021–22 = 100", y=0.5)

    title(
        fig,
        "Drug articles fell no faster than non-medical reference articles —\n"
        "the decline is not specific to drug safety",
    )
    save(fig, "pageviews_B_not_medical")


def fig_c_disparity(pv: pd.DataFrame) -> None:
    """Descriptive reference figure: how much traffic each drug drew, by edition.

    Deliberately NOT conclusion-titled — this one is a reference exhibit rather
    than an argument, so the title just names what is plotted.

    One dot per drug, not bars and not a summary marker. Bars would encode
    magnitude by LENGTH, which a log axis (unavoidable across two orders of
    magnitude) misrepresents; dot POSITION survives the transform. Showing all
    33 drugs rather than a total also shows the spread within each edition.
    """
    window = pv[pv.month.between(*WINDOW)]
    per_drug = window.groupby(["lang", "drug"]).views.sum()
    totals = per_drug.groupby("lang").sum().sort_values(ascending=False)
    medians = per_drug.groupby("lang").median()
    order = list(totals.index)

    fig, ax = plt.subplots(figsize=(8, 4.0), constrained_layout=True)
    # Fixed seed: jitter must be identical on every rebuild of the manuscript.
    rng = np.random.default_rng(0)

    for y, lang in enumerate(reversed(order)):
        vals = per_drug.loc[lang].to_numpy()
        ax.scatter(
            vals,
            y + rng.uniform(-0.16, 0.16, len(vals)),
            s=26,
            color=LANG_COLORS[lang],
            alpha=0.55,
            linewidths=0,
            zorder=2,
        )
        # Median marker: the summary the conclusion's ratio is computed from.
        ax.plot(
            [medians[lang]] * 2, [y - 0.34, y + 0.34],
            color=LANG_COLORS[lang], lw=2.2, zorder=3,
        )
        ax.text(
            medians[lang], y + 0.40, f"median {compact(medians[lang])}",
            color=LANG_COLORS[lang], fontsize=8, ha="center", va="bottom",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [LANG_LABELS[l] for l in reversed(order)], rotation=0, ha="right", ma="left"
    )
    ax.set_ylim(-0.6, len(order) - 0.25)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([100, 1_000, 10_000, 100_000, 1_000_000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: compact(v)))
    ax.set_xlabel("Pageviews (log scale)", fontsize=9)
    title(
        fig,
        f"Pageviews per drug article, July 2025 – June 2026\n"
        f"({pv.drug.nunique()} drugs, four language editions)",
    )
    despine(ax, left=True)
    ax.tick_params(axis="y", length=0)
    save(fig, "pageviews_C_disparity")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pv, controls = load()

    fig_a_exposure(pv)
    fig_b_not_medical(pv, controls)
    fig_c_disparity(pv)

    # Numbers the manuscript text will need to quote.
    window = pv[pv.month.between(*WINDOW)]
    print(f"study window total: {window.views.sum():,}")
    print(window.groupby("lang").views.sum().reindex(LANG_ORDER).to_string())
    for lang in LANG_ORDER:
        s = pv[pv.lang == lang].groupby("month").views.sum().sort_index()
        b = controls[controls.lang == lang].groupby("month").views.sum().sort_index()
        print(
            f"{lang}: last 12mo vs first 12mo — "
            f"drugs {100 * s.iloc[-12:].mean() / s.iloc[:12].mean():.0f}%, "
            f"non-medical controls "
            f"{100 * b.iloc[-12:].mean() / b.iloc[:12].mean():.0f}%"
        )
    print(f"figures -> {FIGURES_DIR}")


if __name__ == "__main__":
    main()
