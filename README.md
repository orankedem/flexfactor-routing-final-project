# FlexFactor — AI-Based Constrained Payment Routing Optimization

## Overview

This project develops an AI-based payment-routing system for declined/card-payment traffic. For each transaction attempt, the system estimates route-specific approval probabilities and chooses a route while respecting operational capacity constraints.

The project is designed as a **dynamic financial decision system**, not only a classifier:

\[
\text{transaction context}
\rightarrow
\text{temporal state}
\rightarrow
\text{route-specific probability}
\rightarrow
\text{capacity-aware optimization}
\rightarrow
\text{route decision}
\]

The main notebook is:

`FlexFactor_Final_Project.ipynb`

It presents the full methodological progression, empirical results, architecture, and a transaction-level inference walkthrough.

---

## Main technical ideas

### 1. Non-stationary payment environment

Payment-route quality changes over time. Static merchant/issuer/route identifiers cannot express whether an entity has recently improved or deteriorated.

The feature layer therefore adds historical state with exponential memory:

\[
w(\Delta t)=0.5^{\Delta t/h}
\]

using multiple half-lives:

\[
h\in\{3,14,60,180\}\text{ days}
\]

and contextual histories such as merchant, issuer, route, merchant×route and issuer×route.

All historical features are emitted using information available before the current decision time.

### 2. Stage-specific predictive models

Transactions can contain multiple attempts:

\[
A1 \rightarrow A2 \rightarrow A3
\]

The information set changes after a failure, so separate XGBoost probability models are used:

\[
P(S_1\mid X,r_1,H_t)
\]

\[
P(S_2\mid X,r_2,A1_{observed},H_t)
\]

\[
P(S_3\mid X,r_3,A1_{observed},A2_{observed},H_t)
\]

Current-attempt provider response fields are excluded from the decision-time feature set.

### 3. Constrained optimization

If capacity were unlimited, routing could use:

\[
r_i^*=\arg\max_r \hat p_{ir}
\]

In practice, route capacity is scarce. The project therefore evaluates:

- an offline full-hindsight linear-programming oracle;
- an online greedy policy;
- an online pressure/shadow-price policy.

The online success score is:

\[
Score_{ir,t}
=
\hat p_{ir,t}
-
\lambda Pressure_{r,t}
\]

with:

\[
Pressure_{r,t}
=
\frac{A_{r,t}-B_{r,t}}
{\max(0.3B_{r,t},1)}
\]

where \(A_{r,t}\) is cumulative policy use and \(B_{r,t}\) is cumulative baseline use.

---

## Dataset

The standardized attempt-level dataset contains approximately:

| Stage | Rows | Observed success rate |
|---|---:|---:|
| A1 | 595,434 | 8.52% |
| A2 | 317,819 | 4.91% |
| A3 | 122,900 | 1.80% |
| **Total attempts** | **1,036,153** | — |

There are approximately **595,434 logical transactions**.

The company transaction dataset is not redistributed in this public repository. The submitted notebook contains executed outputs; full artifact-backed reproduction requires authorized access to the project data/checkpoints.

---

## Final predictive results

Untouched June evaluation:

| Stage | ROC-AUC | Average Precision | Log-loss skill | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| A1 | 0.797 | 0.324 | 0.171 | 0.0788 | 0.0055 |
| A2 | 0.794 | 0.212 | 0.151 | 0.0440 | 0.0019 |
| A3 | 0.871 | 0.137 | 0.207 | 0.0180 | 0.0026 |

The notebook also includes reliability/calibration plots because the optimizer uses probability magnitudes, not only rank ordering.

---

## Optimization results

For the June reference evaluation:

- **Success-optimal LP oracle:** +188.8 model-implied approvals and +1.299M expected approved transaction value.
- **Value-optimal LP oracle:** +173.1 model-implied approvals and +1.424M expected approved transaction value.
- **Greedy online policy:** captures about 46% of the success LP opportunity.
- **Capacity-aware pressure policy:** captures about 86% of the success LP opportunity.
- **Value pressure policy:** captures about 85% of the value LP opportunity.

“100% LP opportunity” means the maximum incremental objective under the frozen probability model and stated capacity constraints. It does **not** mean 100% real-world improvement.

---

## Repository structure

```text
FlexFactor_Final_Project.ipynb
README.md
requirements.txt

src/
    data.py
    features.py
    models.py
    optimization.py
    showcase.py

configs/
    policy_final.json
    a1_final.json   # exported from frozen metadata when available
    a2_final.json
    a3_final.json

artifacts/
    reference_predictive_metrics.csv
    reference_feature_family_importance.csv
    reference_policy_results.csv
```

`src/features.py` contains the temporal/cross-feature implementation.

`src/models.py` contains predictive-model and probability-evaluation utilities.

`src/optimization.py` contains LP, capacity, greedy and pressure-policy logic.

`src/showcase.py` connects the final notebook to the frozen project artifacts and implements the detailed inference replay.

---

## Detailed inference demonstration

The final notebook selects a real A1 event for which the raw highest-probability route differs from the capacity-aware policy choice.

For that event it shows:

1. transaction context;
2. candidate routes;
3. historical/temporal feature values;
4. frozen XGBoost route probabilities;
5. cumulative capacity state;
6. pressure/shadow price;
7. adjusted policy score;
8. greedy route versus final route.

The notebook then links the backtest event to the exact original candidate feature rows. The saved candidate probability is the canonical value actually used by the chronological policy backtest. As an additional reproducibility check, the notebook reloads the frozen XGBoost model and attempts a fresh prediction using the model's stored categorical vocabulary; when the current runtime reproduces the historical native-categorical pipeline, the saved and fresh probabilities are compared directly.

---

## Reproducibility

The notebook uses a public GitHub code repository and mounts the authorized project artifacts from Google Drive.

For a full internal rerun:

```bash
pip install -r requirements.txt
```

The notebook then discovers the existing frozen data/model/backtest artifacts and validates their presence before running the showcase.

Large/private transaction data and heavy model artifacts are intentionally not committed to GitHub.

---

## Interpretation and limitations

Observed-route predictive metrics are factual historical evaluations.

Alternative-route optimization gains are **model-implied counterfactual estimates** because only the outcome of the historically selected route is observed.

Therefore the project does not claim causal production lift from the offline replay.

A production validation path is:

\[
\text{offline analysis}
\rightarrow
\text{shadow deployment}
\rightarrow
\text{controlled live/A-B test}
\rightarrow
\text{production rollout}
\]
