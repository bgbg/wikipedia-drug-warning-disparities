# Ordinal-aware analysis — validation notes

Pipeline run on `data/coded/coded_four_topic.csv` with seed = 23, 4 chains × 2000 draws.

## Bayesian convergence
- Max R-hat: **1.0000** (PASS — threshold 1.01)
- Min bulk ESS: **2351** (PASS — threshold 400)

## Posterior predictive coverage
- 24 cells checked; 4 outside the 95% HDI.
- Cells outside 95% HDI:
  - en / teratogenicity / code=0: observed 0.000, HDI [0.008, 0.201]
  - ar / teratogenicity / code=0: observed 0.000, HDI [0.033, 0.300]
  - ar / teratogenicity / code=1: observed 0.500, HDI [0.132, 0.410]
  - ru / g6pd / code=1: observed 0.000, HDI [0.117, 0.414]

## Frequentist robustness (statsmodels OrderedModel + cluster-robust SE)
| language | topic | OR (Bayes mean) | OR (freq) | sign agree |
|---|---|---|---|---|
| he | g6pd | 0.511 | 0.326 | yes |
| he | teratogenicity | 0.493 | 0.345 | yes |
| ar | g6pd | 0.256 | 0.099 | yes |
| ar | teratogenicity | 0.671 | 0.415 | yes |
| ru | g6pd | 0.0783 | 0.039 | yes |
| ru | teratogenicity | 0.409 | 0.277 | yes |

## Prior sensitivity
Posterior-mean OR by prior_sd:

|                          |   0.5 |   1.5 |   5.0 |
|:-------------------------|------:|------:|------:|
| ('ar', 'g6pd')           | 0.658 | 0.256 | 0.111 |
| ('ar', 'teratogenicity') | 0.877 | 0.671 | 0.547 |
| ('he', 'g6pd')           | 0.798 | 0.511 | 0.377 |
| ('he', 'teratogenicity') | 0.787 | 0.493 | 0.373 |
| ('ru', 'g6pd')           | 0.422 | 0.078 | 0.029 |
| ('ru', 'teratogenicity') | 0.725 | 0.409 | 0.295 |

Worst max/min ratio across the prior_sd grid: **14.31** (sensitive — heuristic threshold 5x).

## Paired Wilcoxon (distributional cross-check)

| language   | topic          |   n_pairs |   n_nonzero_diffs |   statistic |   pvalue |
|:-----------|:---------------|----------:|------------------:|------------:|---------:|
| he         | g6pd           |         8 |                 4 |           2 |   0.5    |
| he         | teratogenicity |         8 |                 3 |           0 |   0.25   |
| ar         | g6pd           |         8 |                 4 |           0 |   0.125  |
| ar         | teratogenicity |         8 |                 2 |           0 |   0.5    |
| ru         | g6pd           |         8 |                 5 |           0 |   0.0625 |
| ru         | teratogenicity |         8 |                 3 |           0 |   0.25   |

## Cliff's delta (clinician-facing effect size)

| lang_a   | lang_b   | topic          |   n_drugs |   delta |    ci_low |   ci_high |   n_boot |
|:---------|:---------|:---------------|----------:|--------:|----------:|----------:|---------:|
| en       | he       | g6pd           |         8 | 0.25    | -0.187891 |    0.6875 |     2000 |
| en       | he       | teratogenicity |         8 | 0.28125 |  0.015625 |    0.625  |     2000 |
| en       | ar       | g6pd           |         8 | 0.53125 |  0.15625  |    0.875  |     2000 |
| en       | ar       | teratogenicity |         8 | 0.25    |  0        |    0.625  |     2000 |
| en       | ru       | g6pd           |         8 | 0.59375 |  0.25     |    0.875  |     2000 |
| en       | ru       | teratogenicity |         8 | 0.3125  |  0.015625 |    0.6875 |     2000 |

## Negative-control validation
- 9 non-zero control rows: 3 pre-documented (see `methodological_decisions.md`), 6 unexplained (INVESTIGATE).

| drug         | lang   | topic          |   code |   expected | deviates   | documented_exception   | exception_note                                                    |
|:-------------|:-------|:---------------|-------:|-----------:|:-----------|:-----------------------|:------------------------------------------------------------------|
| atorvastatin | ar     | teratogenicity |      2 |          0 | True       | True                   | FDA pregnancy category rescinded 2021; Wikipedia regulatory lag   |
| atorvastatin | en     | teratogenicity |      2 |          0 | True       | True                   | FDA pregnancy category rescinded 2021; Wikipedia regulatory lag   |
| famotidine   | ru     | teratogenicity |      2 |          0 | True       | False                  |                                                                   |
| loperamide   | ar     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| loperamide   | en     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| omeprazole   | ar     | teratogenicity |      2 |          0 | True       | True                   | Arabic Wikipedia lists pregnancy under contraindications (Rule 4) |
| paracetamol  | ar     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| paracetamol  | he     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| paracetamol  | ru     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
