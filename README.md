# FlexFactor — AI-Based Constrained Payment Routing Optimization

## Project objective

This project develops an AI-based decision system for routing declined/card-payment attempts through alternative payment routes.

For each transaction attempt, the system estimates:

\[
P(\text{success}\mid \text{transaction},\text{candidate route},\text{historical state})
\]

and then selects a route while respecting route-volume constraints.

The final system is intentionally separated into:

\[
\text{prediction}
\rightarrow
\text{online constrained decisioning}
\]

rather than treating routing as a static classification problem.

The main graded artifact is:

`FlexFactor_Final_Project.ipynb`

It contains the full methodology, empirical results, architecture, model calibration, optimization benchmarks, a worked online-routing example, and the scientific limitations of the offline evaluation.

---

## Why the problem is dynamic

Raw approval rates change through time, but the notebook goes beyond the raw trend.

Concept drift is examined using:

- monthly A1 approval rates;
- a traffic-mix-adjusted residual using merchant × route × card type × amount-decile contexts;
- a large recurring near-comparable cohort selected by volume/stability rather than by observed drift.

The result motivates explicit temporal state rather than a purely static feature representation.

Historical state is represented using exponentially decayed features:

\[
w(\Delta t)=0.5^{\Delta t/h}
\]

with multiple half-lives:

\[
h\in\{3,14,60,180\}\text{ days}
\]

across route, merchant, issuer and interaction contexts.

All historical features use information available before the current decision time.

---

## Stage-specific probability models

A logical payment can contain multiple attempts:

\[
A1 \rightarrow A2 \rightarrow A3
\]

Later attempts are selected populations and expose information that did not exist at A1.

Separate XGBoost models therefore estimate:

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

The company transaction dataset is not redistributed publicly.

---

## Predictive results

Untouched June evaluation:

| Stage | ROC-AUC | Average Precision | Log-loss skill | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| A1 | 0.797 | 0.324 | 0.171 | 0.0788 | 0.0055 |
| A2 | 0.794 | 0.212 | 0.151 | 0.0440 | 0.0019 |
| A3 | 0.871 | 0.137 | 0.207 | 0.0180 | 0.0026 |

The notebook also compares mean predicted probability with the success actually observed on the logged route and shows quantile-binned reliability curves.

Sparse extreme A2/A3 predictions are reported separately with their sample counts so that very small tail bins are not visually over-weighted.

---

## Offline LP benchmark

The linear program is a **month-end full-hindsight oracle**.

It sees the entire period's candidate transactions and frozen model scores and solves the globally optimal assignment under route constraints.

For success:

\[
\max_x\sum_{i,r}x_{ir}\hat p_{ir}
\]

For expected approved transaction value:

\[
\max_x\sum_{i,r}x_{ir}Amount_i\hat p_{ir}
\]

The LP is not deployable online. It is used to measure the maximum model-implied opportunity under the stated assumptions.

Reference June results:

- success-optimal LP: **+188.8 model-implied approvals** and approximately **+1.299M expected approved transaction value**;
- value-optimal LP: approximately **+1.424M expected approved transaction value**.

---

## Online capacity-aware policy

A live router cannot see future transactions.

A greedy rule that always chooses the largest current probability can consume scarce route capacity too early.

The online policy therefore defines route pressure:

\[
Pressure_{r,t}
=
\frac{A_{r,t}-B_{r,t}}
{\max(0.3B_{r,t},1)}
\]

and success score:

\[
Score_{ir,t}
=
\hat p_{ir,t}
-
\lambda Pressure_{r,t}
\]

where \(\lambda\) controls the strength of the scarcity adjustment.

The notebook includes a synthetic four-transaction worked example showing how the route state changes after every decision and why the capacity-aware choice can differ from the raw highest-probability route.

---

## Optimization results

For the June reference evaluation:

- greedy captures about **46%** of the model-implied LP success opportunity;
- success pressure with \(\lambda=0.10\) captures about **86%**;
- the value-oriented policy captures about **85%** of the value LP opportunity.

Capacity behavior also improves materially.

Absolute route deviation is defined as:

\[
d_r
=
\left|
\frac{V_r^{policy}}{V_r^{baseline}}-1
\right|
\]

Greedy leaves 9 of 12 routes at or above 27% absolute deviation, i.e. close to the ±30% capacity boundary. The main pressure policies leave only 1 of 12 routes near that boundary.

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
    a1_final.json
    a2_final.json
    a3_final.json

artifacts/
    reference_predictive_metrics.csv
    reference_feature_family_importance.csv
    reference_policy_results.csv
```

`src/features.py` contains leakage-safe temporal/cross-feature utilities.

`src/models.py` contains probability-model and evaluation utilities.

`src/optimization.py` contains the LP, capacity and online pressure-policy implementation.

`src/showcase.py` discovers the frozen artifacts and exports compact exact model configurations for reproducibility.

---

## Reproducibility

The public repository contains code, configuration and compact result artifacts.

Large/private transaction data and heavy frozen model artifacts remain in the authorized project Google Drive.

The notebook mounts the authorized Drive, discovers the saved artifacts, and uses those frozen checkpoints rather than retraining models during the presentation run.

---

## Scientific limitation

Observed-route predictive metrics are factual historical evaluations.

Alternative-route policy gains are **model-implied counterfactual estimates** because only the outcome of the historically selected route is observed.

The project therefore does not claim causal production uplift from offline replay.

A production validation path is:

\[
\text{offline validation}
\rightarrow
\text{shadow deployment}
\rightarrow
\text{controlled A/B test}
\rightarrow
\text{production rollout}
\]
