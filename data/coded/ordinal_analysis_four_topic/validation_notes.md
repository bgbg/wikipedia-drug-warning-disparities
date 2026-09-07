# Ordinal-aware analysis — validation notes

Pipeline run on `data/coded/coded_four_topic.csv` with seed = 23, 4 chains × 2000 draws.

## Bayesian convergence
- Max R-hat: **1.0000** (PASS — threshold 1.01)
- Min bulk ESS: **2332** (PASS — threshold 400)

## Posterior predictive coverage
- 36 cells checked; 6 outside the 95% HDI.
- Cells outside 95% HDI:
  - en / teratogenicity / code=0: observed 0.000, HDI [0.008, 0.204]
  - en / qt / code=0: observed 0.000, HDI [0.004, 0.171]
  - ar / g6pd / code=0: observed 0.500, HDI [0.185, 0.494]
  - ar / teratogenicity / code=0: observed 0.000, HDI [0.038, 0.318]
  - ar / teratogenicity / code=1: observed 0.500, HDI [0.132, 0.377]
  - ru / g6pd / code=1: observed 0.000, HDI [0.111, 0.380]

## Frequentist robustness (statsmodels OrderedModel + cluster-robust SE)
| language | topic | OR (Bayes mean) | OR (freq) | sign agree |
|---|---|---|---|---|
| he | g6pd | 0.533 | 0.328 | yes |
| he | teratogenicity | 0.512 | 0.347 | yes |
| he | qt | 0.0807 | 0.328 | yes |
| ar | g6pd | 0.284 | 0.101 | yes |
| ar | teratogenicity | 0.677 | 0.417 | yes |
| ar | qt | 0.045 | 0.101 | yes |
| ru | g6pd | 0.0843 | 0.04 | yes |
| ru | teratogenicity | 0.426 | 0.28 | yes |
| ru | qt | 0.878 | 0.04 | yes |

## Prior sensitivity
Posterior-mean OR by prior_sd:

|                          |   0.5 |   1.5 |   5.0 |
|:-------------------------|------:|------:|------:|
| ('ar', 'g6pd')           | 0.687 | 0.284 | 0.12  |
| ('ar', 'qt')             | 0.349 | 0.045 | 0.011 |
| ('ar', 'teratogenicity') | 0.903 | 0.677 | 0.56  |
| ('he', 'g6pd')           | 0.816 | 0.533 | 0.378 |
| ('he', 'qt')             | 0.432 | 0.081 | 0.024 |
| ('he', 'teratogenicity') | 0.823 | 0.512 | 0.365 |
| ('ru', 'g6pd')           | 0.44  | 0.084 | 0.03  |
| ('ru', 'qt')             | 0.967 | 0.878 | 0.776 |
| ('ru', 'teratogenicity') | 0.759 | 0.426 | 0.292 |

Worst max/min ratio across the prior_sd grid: **32.06** (sensitive — heuristic threshold 5x).

## Paired Wilcoxon (distributional cross-check)

| language   | topic          |   n_pairs |   n_nonzero_diffs |   statistic |   pvalue |
|:-----------|:---------------|----------:|------------------:|------------:|---------:|
| he         | g6pd           |         8 |                 4 |         2   |  0.5     |
| he         | teratogenicity |         8 |                 3 |         0   |  0.25    |
| he         | qt             |         8 |                 5 |         0   |  0.0625  |
| ar         | g6pd           |         8 |                 4 |         0   |  0.125   |
| ar         | teratogenicity |         8 |                 2 |         0   |  0.5     |
| ar         | qt             |         8 |                 6 |         0   |  0.03125 |
| ru         | g6pd           |         8 |                 5 |         0   |  0.0625  |
| ru         | teratogenicity |         8 |                 3 |         0   |  0.25    |
| ru         | qt             |         8 |                 3 |         1.5 |  0.75    |

## Cliff's delta (clinician-facing effect size)

| lang_a   | lang_b   | topic          |   n_drugs |    delta |    ci_low |   ci_high |   n_boot |
|:---------|:---------|:---------------|----------:|---------:|----------:|----------:|---------:|
| en       | he       | g6pd           |         8 | 0.25     | -0.187891 |  0.6875   |     2000 |
| en       | he       | teratogenicity |         8 | 0.28125  |  0.015625 |  0.625    |     2000 |
| en       | he       | qt             |         8 | 0.6875   |  0.34375  |  0.984375 |     2000 |
| en       | ar       | g6pd           |         8 | 0.53125  |  0.15625  |  0.875    |     2000 |
| en       | ar       | teratogenicity |         8 | 0.25     |  0        |  0.625    |     2000 |
| en       | ar       | qt             |         8 | 0.828125 |  0.5      |  1        |     2000 |
| en       | ru       | g6pd           |         8 | 0.59375  |  0.25     |  0.875    |     2000 |
| en       | ru       | teratogenicity |         8 | 0.3125   |  0.015625 |  0.6875   |     2000 |
| en       | ru       | qt             |         8 | 0.140625 | -0.25     |  0.546875 |     2000 |

## Negative-control validation
- 12 non-zero control rows: 3 pre-documented (see `methodological_decisions.md`), 9 unexplained (INVESTIGATE).

| drug         | lang   | topic          |   code |   expected | deviates   | documented_exception   | exception_note                                                    |
|:-------------|:-------|:---------------|-------:|-----------:|:-----------|:-----------------------|:------------------------------------------------------------------|
| atorvastatin | ar     | teratogenicity |      2 |          0 | True       | True                   | FDA pregnancy category rescinded 2021; Wikipedia regulatory lag   |
| atorvastatin | en     | teratogenicity |      2 |          0 | True       | True                   | FDA pregnancy category rescinded 2021; Wikipedia regulatory lag   |
| famotidine   | en     | qt             |      1 |          0 | True       | False                  |                                                                   |
| famotidine   | ru     | teratogenicity |      2 |          0 | True       | False                  |                                                                   |
| loperamide   | en     | qt             |      1 |          0 | True       | False                  |                                                                   |
| loperamide   | ru     | qt             |      1 |          0 | True       | False                  |                                                                   |
| loperamide   | ar     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| loperamide   | en     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| omeprazole   | ar     | teratogenicity |      2 |          0 | True       | True                   | Arabic Wikipedia lists pregnancy under contraindications (Rule 4) |
| paracetamol  | ar     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| paracetamol  | he     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
| paracetamol  | ru     | teratogenicity |      1 |          0 | True       | False                  |                                                                   |
