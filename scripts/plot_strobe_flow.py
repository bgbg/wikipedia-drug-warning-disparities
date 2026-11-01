"""STROBE study flow diagram (Supplementary Figure S5 of the Drug Safety submission).

Every number is computed from committed data (drug list, article file, coded cells, inter-model
agreement table) and asserted against the numbers the manuscript states, so a data change that
would make the diagram disagree with the text fails loudly.

    conda run -n pharmaco python scripts/plot_strobe_flow.py [out_stem]

Called by submission/drugsafety_submission_20261015/build.py, which writes the PNG and TIFF.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from g6pd.config import ARTICLES_JSONL, CODED_DIR, CODED_FOUR_TOPIC_CSV, DRUGS_CSV  # noqa: E402

RISK = {"g6pd": "cpic_risk", "teratogenicity": "pregnancy_risk", "qt": "qt_risk"}
LANGS = [("en", "English"), ("ru", "Russian"), ("ar", "Arabic"), ("he", "Hebrew")]


def counts() -> dict:
    cells = pd.read_csv(CODED_FOUR_TOPIC_CSV)
    drugs = pd.read_csv(DRUGS_CSV).set_index("drug")
    arts = pd.read_json(ARTICLES_JSONL, lines=True)
    arts = arts[arts.drug.isin(cells.drug)][["drug", "lang", "exists"]]
    cells = cells.merge(arts, on=["drug", "lang"], how="left", validate="many_to_one")
    cells["positive"] = [pd.notna(drugs.at[d, RISK[t]]) for d, t in zip(cells.drug, cells.topic)]
    pos, ctl = cells[cells.positive], cells[~cells.positive]
    agree = pd.read_csv(CODED_DIR / "model_agreement_three_domains.csv").set_index(["language", "domain"])
    disagree = {t: int(agree.n_cells["all", t] - agree.n_unanimous["all", t]) for t in RISK}
    n = {
        "drugs": cells.drug.nunique(), "pos_drugs": pos.drug.nunique(), "ctl_drugs": ctl.drug.nunique(),
        "pos_per_domain": {t: pos[pos.topic == t].drug.nunique() for t in RISK},
        "sought": len(arts), "exist": int(arts.exists.sum()),
        "exist_by_lang": arts.groupby("lang").exists.sum().astype(int).to_dict(),
        "absent_by_lang": (~arts.exists).groupby(arts.lang).sum().astype(int).to_dict(),
        "cells": len(cells), "pos_cells": len(pos), "ctl_cells": len(ctl),
        "pos_cells_per_domain": {t: int((pos.topic == t).sum()) for t in RISK},
        "ctl_cells_per_domain": {t: int((ctl.topic == t).sum()) for t in RISK},
        "with_article": int(cells.exists.sum()), "pos_with_article": int(pos.exists.sum()),
        "ctl_with_article": int(ctl.exists.sum()),
        "pos_with_article_by_lang": pos.groupby("lang").exists.sum().astype(int).to_dict(),
        "absent_cells_codes": sorted(cells[~cells.exists].final_code.unique().tolist()),
        "unanimous": int(agree.n_unanimous["all", "all"]), "coded_by_models": int(agree.n_cells["all", "all"]),
        "disagree": disagree,
    }
    # Numbers stated in submission/manuscript.md (2.2, 2.6, 2.7, 2.8, 3.1, 3.8, 3.10).
    assert (n["drugs"], n["pos_drugs"], n["ctl_drugs"]) == (33, 24, 9), n
    assert set(n["pos_per_domain"].values()) == {8}, n
    assert (n["sought"], n["exist"]) == (132, 118), n
    assert n["exist_by_lang"] == {"en": 33, "ru": 26, "ar": 32, "he": 27}, n
    assert (n["cells"], n["pos_cells"], n["ctl_cells"]) == (204, 96, 108), n
    assert set(n["ctl_cells_per_domain"].values()) == {36}, n
    assert n["pos_with_article_by_lang"] == {"en": 24, "ru": 17, "ar": 23, "he": 21}, n
    assert n["with_article"] == n["coded_by_models"] == 184 and n["unanimous"] == 160, n
    assert n["disagree"]["g6pd"] + n["disagree"]["teratogenicity"] == 12 and n["disagree"]["qt"] == 12, n
    assert n["absent_cells_codes"] == [0], n
    return n


# ---------------------------------------------------------------- drawing (millimetres, y down)
W, H = 174, 158
PT = 8.5          # body lettering, Arial
LINE = PT * 1.3 / 72 * 25.4   # line height in mm
PAD = 2.2


def box(ax, x, y, w, lines, bold_first=True):
    """Box of width w at top-left (x, y); first line bold. Returns (x, y, w, h)."""
    h = 2 * PAD + LINE * len(lines)
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=0.8, ec="black"))
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + PAD + LINE * (i + 0.5), line, ha="center", va="center", fontsize=PT,
                fontweight="bold" if (i == 0 and bold_first) else "normal")
    return x, y, w, h


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="black",
                                                        shrinkA=0, shrinkB=0, mutation_scale=8))


def line(ax, xs, ys):
    ax.plot(xs, ys, color="black", lw=0.8, solid_capstyle="butt")


def draw(n: dict):
    lang = lambda d, sub=LANGS: ", ".join(f"{name} {d[k]}" for k, name in sub)
    absent = n["absent_by_lang"]
    dis = n["disagree"]
    pd_ = n["pos_cells_per_domain"]

    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    mx, mw = 22, 116   # main column
    cx = mx + mw / 2
    b1 = box(ax, mx, 2, mw, [f"{n['drugs']} drugs in the panel",
                             f"{n['pos_drugs']} risk positive: {n['pos_per_domain']['g6pd']} G6PD, "
                             f"{n['pos_per_domain']['teratogenicity']} teratogenicity, {n['pos_per_domain']['qt']} QT",
                             f"{n['ctl_drugs']} negative controls (no warning expected in any domain)"])
    b2 = box(ax, mx, b1[1] + b1[3] + 6, mw, [f"{n['sought']} drug articles sought",
                                             f"{n['drugs']} drugs × 4 Wikipedia editions (English, Russian, Arabic, Hebrew)",
                                             "MediaWiki API"])
    arrow(ax, cx, b1[1] + b1[3], cx, b2[1])
    gap_top = b2[1] + b2[3]
    side = [f"{n['sought'] - n['exist']} articles absent",
            lang(absent, LANGS[1:]) + f", English {absent['en']}",
            "Retained; their cells scored 0"]
    sx, sw = cx + 14, W - (cx + 14) - 2
    s1 = box(ax, sx, gap_top + 6, sw, side)
    b3 = box(ax, mx, s1[1] + s1[3] + 6, mw, [f"{n['exist']} articles retrieved", lang(n["exist_by_lang"])])
    arrow(ax, cx, gap_top, cx, b3[1])
    arrow(ax, cx, s1[1] + s1[3] / 2, sx, s1[1] + s1[3] / 2)
    b4 = box(ax, mx, b3[1] + b3[3] + 6, mw, [
        f"{n['cells']} drug-by-domain cells (drug × edition × domain)",
        f"{n['pos_cells']} risk positive: {n['pos_drugs']} drugs × 4 editions, own domain only",
        f"{n['ctl_cells']} control: {n['ctl_drugs']} drugs × 4 editions × 3 domains"])
    arrow(ax, cx, b3[1] + b3[3], cx, b4[1])

    # coding row: cells with an article (models) and without (scored 0)
    lx, lw_, rx = 2, 100, 108
    rw = W - 2 - rx
    y5 = b4[1] + b4[3] + 10
    l5 = box(ax, lx, y5, lw_, [f"{n['with_article']} cells with an article",
                               "coded 0, 1 or 2 by three large language models",
                               f"{n['unanimous']} unanimous; {n['with_article'] - n['unanimous']} disagreements:",
                               f"{dis['g6pd'] + dis['teratogenicity']} G6PD and teratogenicity: pharmacist's code",
                               f"{dis['qt']} QT: majority code (extreme-code sensitivity)"])
    n_abs = n["cells"] - n["with_article"]
    r5 = box(ax, rx, y5, rw, [f"{n_abs} cells without an article",
                              f"risk positive {n['pos_cells'] - n['pos_with_article']}, "
                              f"control {n['ctl_cells'] - n['ctl_with_article']}",
                              "scored 0 (no warning available)"])
    split = b4[1] + b4[3] + 5
    line(ax, [cx, cx], [b4[1] + b4[3], split])
    line(ax, [lx + lw_ / 2, rx + rw / 2], [split, split])
    arrow(ax, lx + lw_ / 2, split, lx + lw_ / 2, y5)
    arrow(ax, rx + rw / 2, split, rx + rw / 2, y5)

    # analysis row: both coding boxes feed both analysis sets
    y6 = y5 + max(l5[3], r5[3]) + 12
    merge = y5 + max(l5[3], r5[3]) + 6
    line(ax, [lx + lw_ / 2, lx + lw_ / 2], [y5 + l5[3], merge])
    line(ax, [rx + rw / 2, rx + rw / 2], [y5 + r5[3], merge])
    line(ax, [lx + lw_ / 2, rx + rw / 2], [merge, merge])
    l6 = box(ax, lx, y6, lw_, ["Primary analysis",
                               f"{n['pos_cells']} risk-positive cells, {n['pos_drugs']} drugs",
                               f"(G6PD {pd_['g6pd']}, teratogenicity {pd_['teratogenicity']}, QT {pd_['qt']})",
                               f"{n['pos_with_article']} with an article: "
                               + lang(n["pos_with_article_by_lang"], LANGS[:2]) + ",",
                               lang(n["pos_with_article_by_lang"], LANGS[2:]),
                               "Hierarchical Bayesian cumulative-logit model"])
    r6 = box(ax, rx, y6, rw, ["Negative controls",
                              f"{n['ctl_cells']} cells, {n['ctl_drugs']} drugs",
                              f"{n['ctl_cells_per_domain']['g6pd']} per domain; {n['ctl_with_article']} with an article",
                              "Descriptive analysis"])
    arrow(ax, lx + lw_ / 2, merge, lx + lw_ / 2, y6)
    arrow(ax, rx + rw / 2, merge, rx + rw / 2, y6)
    bottom = max(l6[1] + l6[3], r6[1] + r6[3]) + 2
    ax.set_ylim(bottom, 0)
    fig.set_size_inches(W / 25.4, bottom / 25.4)
    return fig


def save(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    fig.savefig(stem.with_suffix(".tif"), dpi=1200, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("STROBE_flowchart")
    save(draw(counts()), out)
    print(f"wrote {out}.png / .tif")
