# Responsible AI

This document covers the four Responsible AI deliverables for the M5 demand
forecast project: fairness analysis, model explainability, data privacy, and
ethical implications.

## 1. Fairness Analysis and Bias Detection

The training pipeline now generates a validation-time fairness report under
`models/responsible_ai/` whenever `pipeline.run_pipeline` or
`scripts.train_baseline` is executed.

Generated artifacts:

- `fairness_group_metrics.csv`
- `fairness_summary.json`

The fairness analysis compares validation error across the main retail groups
available in the M5 data:

- `state_id`
- `store_id`
- `cat_id`
- `dept_id`
- `item_id`

For each group, the report computes:

- row count
- RMSE
- MAE
- WAPE

The summary JSON also records the best and worst subgroup by MAE and the MAE
gap between them. This is a practical bias-detection check for this project:
if one geography, store, or product family has consistently worse forecast
error, that signals uneven model performance and possible operational bias.

## 2. Model Explainability

The project uses permutation importance as the explainability method. This
satisfies the rubric requirement of "SHAP, LIME, or equivalent."

Generated artifacts:

- `explainability_permutation_importance.csv`
- `explainability_summary.json`
- `explainability_top_features.png`

Why this method was chosen:

- it works directly with the deployed scikit-learn pipeline
- it reflects validation-time feature impact on forecast quality
- it avoids adding a heavier dependency surface just for explanation

The explainability output ranks features by their average impact on validation
MAE when shuffled. This gives a global explanation of which inputs the model
depends on most strongly.

## 3. Data Privacy Considerations

This project has relatively low privacy risk because it uses aggregate retail
data instead of customer-level records.

What the model uses:

- product identifiers (`item_id`, `dept_id`, `cat_id`)
- store and state identifiers (`store_id`, `state_id`)
- calendar features
- SNAP flags
- price
- historical unit sales

What it does not use:

- names
- addresses
- phone numbers
- payment data
- customer identifiers
- transaction-level personal behavior

Current privacy safeguards and design choices:

- raw M5 CSV files are kept outside git
- model artifacts contain model metadata and Responsible AI summaries, not
  personal records
- the API request schema only accepts item/store metadata, demand history, and
  optional price

Residual privacy considerations:

- even non-PII operational data should be access-controlled in a real system
- request logging should avoid storing unnecessary payload history
- model artifacts and experiment outputs should be retained only as long as
  needed

## 4. Ethical Implications

Although the dataset is not personally sensitive, forecast errors still have
real operational consequences.

Main ethical risks:

- under-forecasting can increase stockouts and reduce product availability
- over-forecasting can increase waste, storage cost, and markdown pressure
- uneven accuracy across stores or states can shift service quality between
  regions
- a single global model may work better for high-volume groups than sparse or
  niche products

Project limitations that should be stated clearly:

- this is a baseline model optimized for reproducibility, not fairness-aware
  optimization
- subgroup analysis is observational and does not remove disparities by itself
- aggregate retail forecasting should support human planning, not replace it
  without review

Recommended mitigation in future iterations:

- set alert thresholds for subgroup MAE/WAPE gaps
- retrain or tune with subgroup-aware validation gates
- review inventory decisions with human oversight for low-volume groups
- compare performance before deploying to new stores or categories

## 5. How To Generate The Responsible AI Deliverables

Run the normal training pipeline:

```bash
python -m pipeline.run_pipeline --max-series 300 --recent-days 365 --validation-days 28
```

After training, inspect:

- `models/responsible_ai/fairness_group_metrics.csv`
- `models/responsible_ai/fairness_summary.json`
- `models/responsible_ai/explainability_permutation_importance.csv`
- `models/responsible_ai/explainability_summary.json`
- `models/responsible_ai/explainability_top_features.png`

The model bundle and `GET /model/info` also include a Responsible AI summary so
the analysis is attached to the trained artifact, not only written as a
separate report.
