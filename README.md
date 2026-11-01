# Wikipedia Drug-Safety Warning Disparities

Code, data, and analysis outputs for a cross-lingual audit of drug-safety warnings on Wikipedia.

> Berlin M, Berkovitch M, Benassayag Kaduri N, Gorelik E, Gorelik B. *Drug Safety Warnings in English, Russian, Arabic and Hebrew Wikipedia: A Cross-Sectional Audit of 33 Drugs Across Three Risk Domains.* Manuscript submitted to the European Journal of Clinical Pharmacology, 2026.

## What this is

We audited the English, Russian, Arabic, and Hebrew editions of Wikipedia across 33 drugs in three mechanistically unrelated drug-safety domains, glucose-6-phosphate dehydrogenase (G6PD) deficiency, teratogenicity, and QT-interval prolongation, each benchmarked against an authoritative drug-by-drug reference standard. Each article/domain pair was classified on a 0/1/2 actionability scale (no mention / mentioned / explicit directive) by a three-model LLM consensus with pharmacist adjudication of disagreements, then analysed with a hierarchical Bayesian cumulative-logit regression pooling the language effect across domains.

Non-English editions were consistently less likely to carry an actionable warning than English, and the mechanism differed by edition: in Russian the gap tracked article length and coverage, while in Arabic a warning-mention-without-directive gap persisted independent of length. None of the 24 positive-control drugs carried a structured Wikidata contraindication link.

## Repository contents

```
src/g6pd/                    Core package
  config.py                    path constants
  fetch.py                     MediaWiki API article collection
  coding.py                    warning coding logic (LLM-based; legacy keyword
                                dictionaries kept for the baseline comparison)
  wikidata.py                  SPARQL queries against Wikidata
  ordinal_analysis.py          Bayesian cumulative-logit model + robustness checks
  ordinal_plots.py             figure generation for the ordinal analysis

scripts/                     Runnable pipeline stages (see "Reproducing" below)

tests/                       pytest suite

data/
  drugs.csv                    master drug list: CPIC/pregnancy/QT/PGx risk
                                columns, Wikidata QIDs, per-language article titles
  wikidata_gaps.csv            Wikidata structured-data audit output
  prompts/classification_prompts.json
                                the domain-specific few-shot classification prompts
  coded/                       classification and analysis outputs, including:
    llm_response_log.jsonl       full LLM response log (three-model consensus)
    llm_classifications.csv,
    llm_disagreements.csv        per-model codes and pharmacist-adjudicated disagreements
    coded_four_topic.csv         combined long-format coded dataset (drug, lang,
                                  topic, code) feeding the hierarchical model
                                  (name is historical; the study reports three topics)
    *_warnings.csv                per-topic coded outputs
    hierarchical_four_topic/     primary Bayesian model output (pooled and
                                  per-domain odds ratios, convergence diagnostics)
    ordinal_analysis_four_topic/,
    ordinal_analysis_four_topic_2dom/
                                  per-cell ordinal analysis and robustness checks
                                  (frequentist cluster-robust fit, prior-sensitivity
                                  sweep, Wilcoxon test, Cliff's delta)
    length_sensitivity/          length-adjusted model and mention-to-directive
                                  conditional analysis
    hierarchical_reporting/      model-reporting additions (finite-population
                                  summary, tau prior table, joint posterior
                                  probabilities, post hoc common-effect model)
    model_agreement_three_domains.csv
                                  inter-model agreement by domain
  snapshot_revisions.csv       archived Wikipedia revision IDs of the audited articles
  snapshot_drift.csv           drift between the audited and later revisions
  wikidata_base_rates.csv      Wikidata denominator and base rates
  wikidata_properties.json     verified Wikidata property labels
  validation/                  blind-validation precheck agreement tables
```

Raw fetched Wikipedia article text is not included, for size and licensing reasons. It is reproducible from `scripts/collect_articles.py` and the snapshot dates reported in the manuscript.

## Reproducing the analysis

```bash
conda create -n pharmaco python=3.11
conda activate pharmaco
pip install -r requirements.txt
```

Pipeline stages, in order (each script's own docstring/`--help` has exact usage):

1. `scripts/build_drug_list.py` — build `data/drugs.csv`
2. `scripts/collect_articles.py` — fetch Wikipedia article text via the MediaWiki API (requires network access; not idempotent-free of Wikipedia's current content, so results will drift from the paper's snapshot dates)
3. `scripts/classify_warnings.py --topic <g6pd|teratogenicity|qt>` — three-model LLM consensus classification (requires an OpenRouter API key)
4. `scripts/merge_classifications.py` — merge classifications with adjudication into per-topic coded outputs
5. `scripts/assemble_four_topic_coded.py` — combine topics into the long-format dataset the models read
6. `scripts/run_hierarchical_analysis.py` — fit the pooled hierarchical Bayesian model (Table 2, Figure 3)
7. `scripts/run_ordinal_analysis.py` — per-cell ordinal analysis and robustness checks
8. `scripts/length_sensitivity.py` — length-adjusted and mention-to-directive models (Table 3, Table 4)
9. `scripts/audit_wikidata.py` — Wikidata structured-data audit (`data/wikidata_gaps.csv`)
10. `scripts/fetch_pageviews.py` / `scripts/plot_pageviews.py` — pageview reach figures

```bash
pytest
```

## License

Code is released under the [MIT License](LICENSE). Data (the `data/` directory) is released under [CC BY 4.0](LICENSE-DATA).

## Contact

Boris Gorelik, Azrieli College of Engineering, Jerusalem, Israel — boris@gorelik.net
