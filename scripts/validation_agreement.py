"""
validation_agreement.py: compare the collaborators' blind marks with the reference codes (#77).

Reads the pulled `annotations_<name>.csv` files (scripts/pull_annotations.sh) and the
final codes in data/coded/coded_four_topic.csv, and writes aggregate tables
(committed, no article text, no per-cell marks). The measures and thresholds are the rule
pre-specified on 3/10/2026 (methodological_decisions.md; constants in src/g6pd/agreement.py).

Primary, per language (cells pooled across domains; two raters on one cell are collapsed to one
human code when they agree, and the cell is excluded and counted as disputed when they do not):

    agreement_by_language.csv       n cells, "can't tell" marks, disputed cells, exact agreement,
                                    quadratic-weighted kappa with a seeded bootstrap 95% CI,
                                    signed bias (human minus LLM), LLM under/over-coding shares,
                                    exact sign test, asymmetry flag
    sens_spec_by_language.csv       LLM final code vs the human code for two binary decisions,
                                    mention (0 vs >=1) and directive (2 vs 1, among cells where both
                                    codes are >=1): sensitivity and specificity with Clopper-Pearson CIs
    impact_substitution.csv         human codes substituted on validated cells: Table 4 shares
                                    (P(directive | mentioned)) and the pooled OR direction versus English

The pre-specified outcome (A, B, C, or no data) is printed. Per annotator:

    agreement_vs_consensus.csv      per annotator x language x domain: n marked, n "can't tell",
                                    exact agreement, kappa vs the final code; strata with n < 8 are
                                    flagged `interpretable` = False
    agreement_between_annotators.csv  per annotator pair x language: n shared, agreement, kappa
    disagreement_direction.csv      per annotator (and pooled) x language: disagreements split into
                                    annotator HIGHER than the final code (the LLMs under-detected)
                                    and annotator LOWER (the LLMs over-detected), with the transitions

"Can't tell" marks are counted but excluded from agreement. The item-level merge and the
substituted coded dataset are written to the gitignored annotations directory only.

The pooled OR in the impact table is a fast approximation (frequentist ordered logit on language,
cluster-robust SE by drug), run on the original and the substituted codes alike; it reads the
direction, not the manuscript's hierarchical estimate. `--refit` reruns
scripts/run_hierarchical_analysis.py (MCMC, slow) on the substituted codes.

Usage:
    python scripts/validation_agreement.py
    python scripts/validation_agreement.py --refit
    python scripts/validation_agreement.py --annotations-dir DIR --reference CSV --out-dir DIR
"""

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from g6pd import ordinal_analysis as oa  # noqa: E402
from g6pd.agreement import (  # noqa: E402
    LANGS,
    MIN_STRATUM_N,
    asymmetry,
    bootstrap_kappa_ci,
    classify_outcome,
    kappa_band,
    sens_spec,
    sign_test_p,
    weighted_kappa,
)
from g6pd.config import (  # noqa: E402
    CODED_FOUR_TOPIC_CSV,
    DRUGS_CSV,
    VALIDATION_ANNOTATIONS_DIR,
    VALIDATION_DIR,
)

DOMAINS = ["g6pd", "teratogenicity", "qt"]
TRANSITIONS = [(0, 1), (0, 2), (1, 2), (1, 0), (2, 0), (2, 1)]  # (final code, annotator code)


def load_marks(ann_dir: Path) -> pd.DataFrame:
    """One row per marked cell: annotator, cell_id, drug, lang, domain, code (int or 'cant_tell')."""
    frames = []
    for path in sorted(ann_dir.glob("annotations_*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
        df = df[df["code"] != ""].copy()
        if df.empty:  # only notes so far
            continue
        name = next((a for a in df["annotator"] if a), path.stem.removeprefix("annotations_"))
        df["annotator"] = name
        frames.append(df[["annotator", "cell_id", "code"]])
    if not frames:
        raise SystemExit(f"no annotations_*.csv with marks in {ann_dir}")
    marks = pd.concat(frames, ignore_index=True)
    parts = marks["cell_id"].str.split("__", expand=True)
    marks["drug"], marks["lang"], marks["domain"] = parts[0], parts[1], parts[2]
    return marks


def attach_reference(marks: pd.DataFrame, ref_path: Path) -> pd.DataFrame:
    ref = pd.read_csv(ref_path)
    if "human_final_code" in ref:  # same coalescing as oa.load_coded, so this IS the analysed code
        ref["final_code"] = ref["human_final_code"].where(ref["human_final_code"].notna(), ref["final_code"])
    ref = ref.rename(columns={"topic": "domain", "final_code": "ref_code"})
    out = marks.merge(ref[["drug", "lang", "domain", "ref_code"]], on=["drug", "lang", "domain"], how="left")
    missing = out[out["ref_code"].isna()]["cell_id"].unique()
    if len(missing):
        raise SystemExit(f"cells without a reference code: {list(missing)[:5]}")
    out["ref_code"] = out["ref_code"].astype(int)
    out["cant_tell"] = out["code"] == "cant_tell"
    out["ann_code"] = pd.to_numeric(out["code"], errors="coerce")
    return out


def strata(df: pd.DataFrame):
    """Yield (language, domain, subframe) for each language x domain with 'all' margins."""
    for lang in LANGS + ["all"]:
        for dom in DOMAINS + ["all"]:
            sub = df if lang == "all" else df[df["lang"] == lang]
            sub = sub if dom == "all" else sub[sub["domain"] == dom]
            if len(sub):
                yield lang, dom, sub


def vs_consensus(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for annotator, a in df.groupby("annotator"):
        for lang, dom, sub in strata(a):
            m = sub[~sub["cant_tell"]]
            ann, ref = m["ann_code"].astype(int), m["ref_code"]
            rows.append({
                "annotator": annotator, "language": lang, "domain": dom,
                "n_marked": len(m), "n_cant_tell": int(sub["cant_tell"].sum()),
                "n_exact_agree": int((ann == ref).sum()),
                "pct_exact_agree": round(100 * (ann == ref).mean(), 1) if len(m) else float("nan"),
                "weighted_kappa": round(weighted_kappa(ref, ann), 3),
                "interpretable": len(m) >= MIN_STRATUM_N,  # smaller strata: read the counts, not the kappa
            })
    return pd.DataFrame(rows)


def cell_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per marked cell with a single human code.

    Raters who gave a code agree -> that code; "can't tell" marks are ignored when another rater
    coded the cell; raters who gave different codes -> disputed (human_code NaN, excluded from the
    pooled measures and from the substitution, but counted).
    """
    cells = df.drop_duplicates("cell_id").set_index("cell_id")[["drug", "lang", "domain", "ref_code"]]
    used = df[~df["cant_tell"]].groupby("cell_id")["ann_code"].agg(["nunique", "first"])
    cells["human_code"] = used["first"].where(used["nunique"] == 1)
    cells["disputed"] = (used["nunique"] > 1).reindex(cells.index, fill_value=False)
    cells["n_cant_tell"] = df.groupby("cell_id")["cant_tell"].sum()
    return cells


def _by_lang(cells: pd.DataFrame):
    for lang in LANGS + ["all"]:
        yield lang, (cells if lang == "all" else cells[cells["lang"] == lang])


def by_language(cells: pd.DataFrame) -> pd.DataFrame:
    """Primary measures per language, cells pooled across domains (a row per language, none missing)."""
    rows = []
    for lang, sub in _by_lang(cells):
        m = sub[sub["human_code"].notna()]
        human, llm = m["human_code"].astype(int), m["ref_code"]
        n_under, n_over = int((llm < human).sum()), int((llm > human).sum())  # LLM below / above the human
        # kappa stays at full precision (the benchmark applies to the unrounded estimate, so a kappa of
        # 0.6996 must not become "adequate" by rounding); only the CI is rounded for display
        kappa = weighted_kappa(llm, human)
        lo, hi = (round(x, 3) for x in bootstrap_kappa_ci(llm, human))
        rows.append({
            "language": lang, "n_cells": len(m),
            "n_cant_tell": int(sub["n_cant_tell"].sum()), "n_disputed": int(sub["disputed"].sum()),
            "n_exact_agree": int((human == llm).sum()),
            "pct_exact_agree": round(100 * (human == llm).mean(), 1) if len(m) else float("nan"),
            "weighted_kappa": kappa, "kappa_ci_low": lo, "kappa_ci_high": hi,
            "kappa_band": kappa_band(kappa, len(m)),
            "mean_bias_human_minus_llm": round((human - llm).mean(), 3) if len(m) else float("nan"),
            "n_llm_under": n_under, "n_llm_over": n_over,
            "share_llm_under": round(n_under / len(m), 3) if len(m) else float("nan"),
            "share_llm_over": round(n_over / len(m), 3) if len(m) else float("nan"),
            "sign_test_p": round(sign_test_p(n_under, n_over), 4),
            "asymmetry": asymmetry(n_under, n_over, len(m)),
        })
    return pd.DataFrame(rows)


def sens_spec_table(cells: pd.DataFrame) -> pd.DataFrame:
    """LLM final code vs the human code: mention (0 vs >=1) and directive (2 vs 1, among both >=1)."""
    rows = []
    for lang, sub in _by_lang(cells):
        m = sub[sub["human_code"].notna()]
        human, llm = m["human_code"].astype(int), m["ref_code"]
        both = (human >= 1) & (llm >= 1)
        for decision, truth, pred in (("mention", human >= 1, llm >= 1),
                                      ("directive", human[both] == 2, llm[both] == 2)):
            r = sens_spec(truth, pred)
            r = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
            rows.append({"language": lang, "decision": decision, "n": r["n_pos"] + r["n_neg"], **r})
    return pd.DataFrame(rows)


def pooled_ors_fast(df: pd.DataFrame) -> dict:
    """{language: (OR, ci_low, ci_high)} of a stronger warning versus English, non-English languages only.

    Ordered logit on language alone with SEs clustered by drug (the project's frequentist check, minus
    the topic interaction). A quick read of the direction, not the manuscript's hierarchical model;
    (NaN, NaN, NaN) where the fit is impossible.
    """
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    nan3 = (float("nan"),) * 3
    out = {lang: nan3 for lang in LANGS if lang != "en"}
    langs = [lang for lang in LANGS if lang in set(df["lang"])]
    if "en" not in langs or len(langs) < 2:
        return out
    work = df.assign(lang=pd.Categorical(df["lang"], categories=langs))
    try:
        fit = OrderedModel.from_formula("code ~ C(lang, Treatment(reference='en'))", data=work,
                                        distr="logit").fit(method="bfgs", disp=False)
        cov = oa._cluster_robust_cov(fit, work["drug"].astype("category").cat.codes.values)
    except (ValueError, np.linalg.LinAlgError) as exc:
        # non-estimable data (no variation, separation): recorded as NaN / "undefined" in the impact table
        print(f"WARNING: fast pooled OR not estimable: {exc}")
        return out
    for lang in langs:
        if lang == "en":
            continue
        i = fit.params.index.get_loc(f"C(lang, Treatment(reference='en'))[T.{lang}]")
        b, se = float(fit.params.iloc[i]), float(np.sqrt(cov[i, i]))
        out[lang] = (float(np.exp(b)), float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se)))
    return out


def substitute(coded: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Copy of `coded` (drug, lang, topic, code) with the human code on validated cells.

    Disputed cells and cells with only "can't tell" marks keep the LLM final code. Adds `llm_code`
    (the original) and `substituted`.
    """
    h = (cells[cells["human_code"].notna()].reset_index()
         [["drug", "lang", "domain", "human_code"]].rename(columns={"domain": "topic"}))
    out = coded.merge(h, on=["drug", "lang", "topic"], how="left")
    out["llm_code"] = out["code"]
    out["substituted"] = out["human_code"].notna()
    out["code"] = out["human_code"].fillna(out["code"]).astype(int)
    return out.drop(columns="human_code")


def impact_table(coded: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Table 4 shares and the pooled OR direction, on the LLM final codes and after the substitution."""
    sub = substitute(coded, cells)
    n_sub = int(sub["substituted"].sum())
    n_human = int(cells["human_code"].notna().sum())
    if n_sub != n_human:
        print(f"WARNING: {n_human - n_sub} validated cell(s) are not in the analysed dataset and were not substituted")
    versions = {"llm_final": sub.assign(code=sub["llm_code"]), "human_substituted": sub}
    rows = []
    for version, d in versions.items():
        ors = pooled_ors_fast(d)
        for lang in LANGS:
            x = d[d["lang"] == lang]
            ment = x[x["code"] >= 1]
            n_dir = int((ment["code"] == 2).sum())
            lc = cells[cells["lang"] == lang]
            row = {"version": version, "language": lang, "n_cells": len(x), "n_mentioned": len(ment),
                   "n_directive": n_dir,
                   "p_directive_given_mention": round(n_dir / len(ment), 3) if len(ment) else float("nan")}
            if version == "human_substituted":
                row.update(n_substituted=int(x["substituted"].sum()),
                           n_changed=int((x["code"] != x["llm_code"]).sum()),
                           n_disputed_excluded=int(lc["disputed"].sum()),
                           n_cant_tell_only=int((lc["human_code"].isna() & ~lc["disputed"]).sum()))
            else:
                row.update(n_substituted=pd.NA, n_changed=pd.NA, n_disputed_excluded=pd.NA, n_cant_tell_only=pd.NA)
            if lang == "en":
                row.update(pooled_or=float("nan"), or_ci_low=float("nan"), or_ci_high=float("nan"),
                           or_direction="reference")
            else:
                o, lo, hi = ors[lang]
                row.update(pooled_or=round(o, 3), or_ci_low=round(lo, 3), or_ci_high=round(hi, 3),
                           or_direction="undefined" if np.isnan(o) else "lower" if o < 1 else "higher")
            rows.append(row)
    t = pd.DataFrame(rows)
    ints = ["n_substituted", "n_changed", "n_disputed_excluded", "n_cant_tell_only"]
    t[ints] = t[ints].astype("Int64")
    return t


def direction_changes(impact: pd.DataFrame) -> list[str]:
    """Languages whose pooled OR direction differs between the LLM final codes and the substitution."""
    d = impact.pivot(index="language", columns="version", values="or_direction")
    return [lang for lang in d.index if lang != "en" and d.loc[lang, "llm_final"] != d.loc[lang, "human_substituted"]]


def write_substituted_csv(ref_path: Path, cells: pd.DataFrame, out_path: Path) -> Path:
    """The reference file with the human codes in `human_final_code` (what load_coded prefers)."""
    ref = pd.read_csv(ref_path)
    h = (cells[cells["human_code"].notna()].reset_index()
         [["drug", "lang", "domain", "human_code"]].rename(columns={"domain": "topic"}))
    ref = ref.merge(h, on=["drug", "lang", "topic"], how="left")
    base = ref["human_final_code"] if "human_final_code" in ref else pd.Series(np.nan, index=ref.index)
    ref["human_final_code"] = ref["human_code"].where(ref["human_code"].notna(), base)
    ref.drop(columns="human_code").to_csv(out_path, index=False)
    return out_path


BETWEEN_COLUMNS = ["annotator_a", "annotator_b", "language", "n_shared", "n_exact_agree",
                   "pct_exact_agree", "weighted_kappa"]
DIRECTION_COLUMNS = (["annotator", "language", "n_marked", "n_disagree", "n_llm_under_detect", "n_llm_over_detect"]
                     + [f"final{r}_annotator{c}" for r, c in TRANSITIONS])


def between_annotators(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    wide = df[~df["cant_tell"]].pivot_table(index=["cell_id", "lang"], columns="annotator",
                                             values="ann_code", aggfunc="first")
    for a, b in itertools.combinations(sorted(wide.columns), 2):
        both = wide[[a, b]].dropna().reset_index()
        for lang in LANGS + ["all"]:
            sub = both if lang == "all" else both[both["lang"] == lang]
            if len(sub):
                x, y = sub[a].astype(int), sub[b].astype(int)
                rows.append({
                    "annotator_a": a, "annotator_b": b, "language": lang, "n_shared": len(sub),
                    "n_exact_agree": int((x == y).sum()), "pct_exact_agree": round(100 * (x == y).mean(), 1),
                    "weighted_kappa": round(weighted_kappa(x, y), 3),
                })
    return pd.DataFrame(rows, columns=BETWEEN_COLUMNS)


def direction(df: pd.DataFrame) -> pd.DataFrame:
    m = df[~df["cant_tell"]].copy()
    m["ann_code"] = m["ann_code"].astype(int)
    rows = []
    for who, a in [(n, g) for n, g in m.groupby("annotator")] + [("all_pooled", m)]:
        for lang in LANGS + ["all"]:
            sub = a if lang == "all" else a[a["lang"] == lang]
            if not len(sub):
                continue
            dis = sub[sub["ann_code"] != sub["ref_code"]]
            row = {
                "annotator": who, "language": lang, "n_marked": len(sub), "n_disagree": len(dis),
                "n_llm_under_detect": int((dis["ann_code"] > dis["ref_code"]).sum()),
                "n_llm_over_detect": int((dis["ann_code"] < dis["ref_code"]).sum()),
            }
            for r, c in TRANSITIONS:
                row[f"final{r}_annotator{c}"] = int(((dis["ref_code"] == r) & (dis["ann_code"] == c)).sum())
            rows.append(row)
    return pd.DataFrame(rows, columns=DIRECTION_COLUMNS)


def progress(marks: pd.DataFrame) -> pd.DataFrame:
    """Marks per annotator and language (including "can't tell")."""
    t = marks.pivot_table(index="annotator", columns="lang", values="cell_id", aggfunc="count", fill_value=0)
    t = t.reindex(columns=LANGS, fill_value=0)
    t["total"] = t.sum(axis=1)
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--progress", action="store_true", help="only print marks per annotator and language")
    ap.add_argument("--refit", action="store_true",
                    help="also refit the hierarchical model on the substituted codes (MCMC, slow)")
    ap.add_argument("--annotations-dir", type=Path, default=VALIDATION_ANNOTATIONS_DIR)
    ap.add_argument("--reference", type=Path, default=CODED_FOUR_TOPIC_CSV)
    ap.add_argument("--drugs", type=Path, default=DRUGS_CSV)
    ap.add_argument("--out-dir", type=Path, default=VALIDATION_DIR)
    args = ap.parse_args()

    marks = load_marks(args.annotations_dir)
    if args.progress:
        print("Marks per annotator and language\n" + progress(marks).to_string())
        return
    df = attach_reference(marks, args.reference)
    cells = cell_level(df)
    coded = oa.filter_main_fit(oa.load_coded(args.reference, args.drugs))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.annotations_dir / "merged_item_level.csv", index=False)  # gitignored dir
    primary = by_language(cells)
    impact = impact_table(coded, cells)
    tables = {
        "agreement_by_language.csv": primary,
        "sens_spec_by_language.csv": sens_spec_table(cells),
        "impact_substitution.csv": impact,
        "agreement_vs_consensus.csv": vs_consensus(df),
        "agreement_between_annotators.csv": between_annotators(df),
        "disagreement_direction.csv": direction(df),
    }
    for name, t in tables.items():
        t.to_csv(args.out_dir / name, index=False)
        print(f"\n{name}\n{t.to_string(index=False)}")

    outcome, note = classify_outcome(primary)
    print(f"\nPre-specified outcome: {outcome} ({note})")
    changed = direction_changes(impact)
    print("Pooled OR direction vs English after substitution (fast approximation): "
          + (f"changed for {', '.join(changed)}" if changed else "unchanged in every language"))

    if args.refit:
        sub_csv = write_substituted_csv(args.reference, cells, args.annotations_dir / "coded_human_substituted.csv")
        subprocess.run([sys.executable, str(REPO / "scripts" / "run_hierarchical_analysis.py"),
                        "--input", str(sub_csv), "--drugs", str(args.drugs),
                        "--output-dir", str(args.out_dir / "impact_refit_hierarchical")], check=True)


if __name__ == "__main__":
    main()
