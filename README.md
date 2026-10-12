# Bee-product surveillance: Bayesian profiling, exposure screening and monitoring optimization

Data and code for the article

> **Veterinary drug residues and non-compliance patterns in Chinese bee products: Bayesian profiling, exposure screening and monitoring optimization**
> Zhaolong Liu, Hongxia Li, Zeyu Zhang, Ying Fan, Xinlei Yu, Jun Zhang

The repository contains the analysis tables, the code that produces every modelled
quantity, the result files and the figures of the article. One command rebuilds the
results and figures from the tables in `data/`.

## Layout

```
run_all.py            run every step in order
requirements.txt      package versions used for the article
code/                 analysis scripts (see "Pipeline")
data/                 analysis tables (inputs of the pipeline)
results/              CSV/JSON outputs of the pipeline
figures/              Figures 1-8, Supplementary Figures S1-S4, graphical abstract (PNG, PDF, SVG)
```

## Running the analysis

Python 3.11.5 was used, with the package versions in `requirements.txt`.

```
pip install -r requirements.txt
python run_all.py
```

The run takes about one minute on a laptop and rewrites `results/` and `figures/`.
All random draws use the fixed seed 20260920; a repeat run with the listed package
versions reproduced the distributed result files and figures byte for byte. Set the
environment variable `BEE_TIFF=1` to write 600-dpi TIFF copies of the figures as well.

## Pipeline

| Step | Script | Purpose | Article |
|---|---|---|---|
| 0 | `bee_preprocess.py` | Builds `data/` from the original platform exports (runs only when they are present; see "Data") | Section 2.1 |
| 1 | `bee_descriptive.py` | Reconciles aggregate and detailed counts; annual, regional, product, category, item and concentration summaries | Sections 2.1-2.2; Fig. 2; Tables S1-S6 |
| 2 | `bee_period_audit.py` | Hazard categories, items and lead reports by notification period | Table S11 |
| 3 | `bee_association.py` | Product-specific odds ratios of non-compliance (binomial regression adjusted for notification year, standard errors clustered by month) | Section 2.2; Table S1 |
| 4 | `bee_bn.py` | Six-node tree-augmented naive Bayes network (BDeu, equivalent sample size 5), grouped hold-out, five-fold and out-of-time validation at regional and provincial resolution, posterior-weighted production-sampling footprint | Sections 2.3-2.4; Figs. 3-4; Tables S7-S10 |
| 5 | `bee_risk.py` | Estimated daily intake, hazard quotient, lead margin of exposure, intake scenarios; regulatory and HQ-augmented priority scores with 473 weight combinations | Sections 2.5 and 2.7; Figs. 5-6; Tables S11-S12 |
| 6 | `bee_optim.py` | Integer allocation of batch-item assessments under scenarios S1-S7 at 101 budgets (5,000-30,000), stratum floors, dual objectives and reserves | Section 2.6; Fig. 7; Supplementary Figs. S2-S4; Table S13 |
| 7 | `bee_federated.py` | Federated aggregation of count tables over 31 clients, split-budget controls, central differential privacy (Laplace, ε = 0.1-10, 20 repetitions), communication volume | Section 2.8; Fig. 8; Table S14 |
| 8 | `bee_figures.py` | All figures from the result files | Figs. 1-8; Supplementary Figs. S1-S4 |
| 9 | `validate_project.py` | Numerical and artifact checks (normalization of the network, exact integer optimum against exhaustive enumeration, differential-privacy sensitivity, counts and file inventory) | - |

`bee_common.py` holds the paths, constants and loaders shared by all scripts.

## Data

The dataset compiles the surveillance-sampling notices on bee products published by
national, provincial and municipal market-regulation authorities in China between
January 2015 and 20 September 2026: 87,313 monitored batches, of which 1,798 were
non-compliant. Detailed records exist for the non-compliant batches; compliant
batches are available as counts by month, production province, notifying authority
and product.

| File | Rows | Content |
|---|---|---|
| `records.csv` | 1,798 | One row per non-compliant report: notification date, product, production province and region, sampling province and region, sampling level and stage, notifying authority, link to the published notice |
| `items.csv` | 1,922 | One row per non-compliant item entry: standardized item, hazard category, reported value and limit, concentration in µg/kg where a quantitative chemical result could be parsed |
| `monthly.csv`, `province.csv`, `authority.csv` | 201 / 35 / 339 | Monitored, compliant and non-compliant batches by month, production province and notifying authority |
| `product_denominators.csv`, `product_monthly.csv`, `product_province.csv`, `product_authority.csv` | 4 / 804 / 128 / 724 | The same counts for honey, royal jelly, bee pollen and other bee products |
| `reported_reasons.csv` | 46 | Item entries per standardized item as reported by the source |
| `item_dictionary.csv` | 46 | Chinese and English names of the standardized items |
| `source_manifest.json` | - | SHA-256 hashes of the original exports from which the tables were built |

Notes on the fields:

- Province, item, category, sampling-level and sampling-stage values are kept in
  Chinese as published; `*_en` columns and `item_dictionary.csv` give the English
  names used in the article. Sampling levels are 国抽 (national), 省抽 (provincial)
  and 市抽 (municipal); stages are 流通环节 (distribution), 生产环节 (production) and
  餐饮环节 (catering).
- `production_province` is the province stated for the producer. `sampling_province`
  is taken from the sampling address when it begins with a province name and is
  otherwise 不详 (unknown). Imported, non-mainland and unspecified origins form the
  region "Other / unknown".
- `rate`, `ci_low` and `ci_high` are the non-compliance proportion and its Wilson
  95% confidence interval.
- `group_id` is a truncated SHA-256 hash that identifies reports of the same product
  batch; it is the grouping unit of the validation splits.
- `possible_exact_duplicate` marks records that are identical to another record in
  every published field, and `exact_duplicate_of_earlier_row` marks the second and
  later copies. All records are retained; `results/duplicate_sensitivity.csv` gives
  the counts after collapsing identical records.
- `concentration_ugkg` is filled only for chemical items with a single numerical
  result and an explicit mass unit (`parse_status = parsed`).

The original platform exports contain the names and addresses of producers and
sampled businesses and are not redistributed; the tables above carry no such
fields, and each record keeps the link to its published notice. With the exports
in a folder `raw/` (or in the folder named by the environment variable
`BEE_RAW_DIR`), `run_all.py` rebuilds `data/` first.

### Scope of the quantities

- Proportions are non-compliant batches divided by monitored batches in the
  compiled notices. Item-specific testing denominators are not published, so
  item-level quantities are reported as counts and as shares of non-compliant
  reports.
- The year 2026 covers 1 January to 20 September.
- Exposure estimates are conditional on the quantified non-compliant records and
  use the intake scenarios in `results/intake_scenarios.csv`. Hazard quotients are
  calculated for the nine chemicals with an applicable acceptable daily intake
  (`results/toxicology_parameters.csv`); lead is assessed by margin of exposure.
- Footprint coverage in the allocation analysis is the modelled share of the
  historical report-item footprint reached by an allocation of batch-item
  assessments.

## Figures and tables

| Item | File in `figures/` | Result files |
|---|---|---|
| Fig. 1 | `Figure_1` | schematic |
| Fig. 2 | `Figure_2` | `annual.csv`; `data/records.csv`, `data/items.csv`, `data/province.csv` |
| Fig. 3 | `Figure_3` | `bn_protocol.json`, `bn_counts.npz`, `bn_item_by_sampling_region.csv`, `bn_validation.csv` |
| Fig. 4 | `Figure_4` | `spatial_category_footprints.csv`, `spatial_footprint.csv` |
| Fig. 5 | `Figure_5` | `exposure.csv`, `critical_intake.csv`, `hypothetical_coexposure.csv` |
| Fig. 6 | `Figure_6` | `risk_ranking.csv`, `product_risk_ranking.csv`, `risk_weight_sensitivity.csv` |
| Fig. 7 | `Figure_7` | `optimization_curves.csv`, `optimization_knees.csv` |
| Fig. 8 | `Figure_8` | `federated_comparison.csv`, `federated_dp_summary.csv`, `federated_clients.csv`, `federated_communication.csv`, `federated_communication_scaling.csv` |
| Supplementary Fig. S1 | `Supplementary_Figure_S1` | `exposure.csv`, `lead_moe.csv`, `federated_minimum_policy_sensitivity.csv` |
| Supplementary Fig. S2 | `Supplementary_Figure_S2` | `optimization_allocations.csv`, `optimization_stratum_contributions.csv`, `optimization_curves.csv` |
| Supplementary Fig. S3 | `Supplementary_Figure_S3` | `optimization_floors.csv`, `optimization_floor_allocations.csv` |
| Supplementary Fig. S4 | `Supplementary_Figure_S4` | `optimization_pareto.csv`, `optimization_pareto_allocations.csv`, `optimization_health_reserves.csv` |

| Table | Result files |
|---|---|
| S1 Products | `data/product_denominators.csv`, `product_associations.csv`, `product_risk_ranking.csv` |
| S2 Annual | `annual.csv` |
| S3 Geography | `data/province.csv` |
| S4 Hazard categories | `category_summary.csv`, `product_category.csv` |
| S5 Items and concentrations | `item_summary.csv`, `concentration_summary.csv` |
| S6 Data audit | `summary.json`, `duplicate_sensitivity.csv`, `validation_checks.csv` |
| S7 Network specification | `bn_protocol.json` |
| S8 Validation | `bn_validation.csv` |
| S9 Posterior probabilities | `bn_item_by_sampling_region.csv`, `spatial_category_footprints.csv` |
| S10 Spatial footprint | `spatial_footprint.csv`, `bn_sensitivity.csv` |
| S11 Exposure | `exposure.csv`, `toxicology_parameters.csv`, `intake_scenarios.csv`, `critical_intake.csv`, `exposure_grid.csv`, `hypothetical_coexposure.csv`, `lead_moe.csv`, `lead_chronology_audit.csv` |
| S12 Priority ranking | `risk_ranking.csv`, `risk_weight_sensitivity.csv` |
| S13 Allocation | `optimization_knees.csv`, `optimization_curves.csv`, `optimization_allocations.csv`, `optimization_stratum_contributions.csv`, `optimization_floors.csv`, `optimization_pareto.csv`, `optimization_pareto_allocations.csv`, `optimization_health_reserves.csv` |
| S14 Federated estimation | `federated_comparison.csv`, `federated_dp_summary.csv`, `federated_clients.csv`, `federated_minimum_policy_sensitivity.csv`, `federated_communication.csv`, `federated_communication_scaling.csv`, `federated_protocol.json` |

Each figure is written as 600-dpi PNG and as editable PDF and SVG;
`figures/figure_manifest.json` lists the size and source files of every figure.
The typography of the published figures may differ slightly from these files.

## Main parameters

| Parameter | Value | Set in |
|---|---|---|
| Random seed | 20260920 | `bee_common.py` |
| BDeu equivalent sample size | 5 (sensitivity 1 and 20) | `bee_common.py`, `bee_bn.py` |
| Intake, g/day (baseline; uniform range) | honey and other bee products 10 (5-30); royal jelly 3 (1-5); bee pollen 5 (2-10) | `bee_risk.py` |
| Body weight | 60 kg (scenarios 45-90 kg) | `bee_risk.py` |
| Acceptable daily intake, mg/kg bw/day | sorbates 25; benzoates 20; acesulfame K 15; cyclamate 11; tartrazine 10; saccharin 9; sunset yellow 4; amitraz 0.01; tau-fluvalinate 0.005 | `bee_risk.py` |
| Lead benchmark dose lower limits | 0.63 and 1.50 µg/kg bw/day | `bee_risk.py` |
| Scenario draws | 20,000 per item and product | `bee_risk.py` |
| Budgets | 5,000-30,000 assessments in steps of 250; minimum 5 per cell; reference 15,000 | `bee_optim.py` |
| Differential privacy | Laplace noise on aggregated counts; at most 5 items per report; L1 sensitivity 30 | `bee_federated.py` |

## Citation

If you use the data or code, please cite the article above.
