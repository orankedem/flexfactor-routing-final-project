# FlexFactor — AI-Based Constrained Payment Routing Optimization

## Project objective

FlexFactor is an AI-based decision system for routing declined/card-payment attempts through alternative payment routes.

For every candidate route, the predictive layer estimates:

```math
P(\mathrm{success}\mid \mathrm{transaction\ context},\ \mathrm{candidate\ route},\ \mathrm{historical\ state})
```

The decision layer then chooses a route while respecting route-volume constraints.

The project is therefore not only a binary classifier. Its end-to-end flow is:

```math
\mathrm{transaction}
\rightarrow
\mathrm{dynamic\ state}
\rightarrow
\mathrm{route\ probabilities}
\rightarrow
\mathrm{capacity\text{-}aware\ decision}
\rightarrow
\mathrm{outcome/state\ update}
```

The main graded artifact is **`FlexFactor_Final_Project.ipynb`**. It contains the methodology, feature engineering, model comparison, calibration analysis, optimization benchmarks, worked online-routing example, robustness analysis, and scientific limitations of the offline evaluation.

---

## 1. Why the problem is dynamic

Raw approval rates change through time, but aggregate drift can also reflect a changing mix of merchants, routes, cards, or transaction sizes.

The notebook therefore examines non-stationarity using three views:

- monthly first-attempt approval rates;
- a traffic-mix-adjusted residual using merchant × route × card type × amount-decile contexts;
- a large recurring near-comparable cohort selected by volume and recurrence rather than by the amount of observed drift.

The result motivates a model whose representation can adapt to the current environment instead of relying only on static identifiers.

---

## 2. Feature engineering

### 2.1 Continuous temporal memory

A static model sees the transaction at time $t$, but not whether a route, merchant, or issuer has recently improved or deteriorated.

Instead of relying only on hard rolling windows, historical information is allowed to fade continuously:

```math
w(\Delta t)=0.5^{\Delta t/h}
```

The final representation exposes several memory speeds:

```math
h\in\{3,14,60,180\}\ \mathrm{days}
```

Historical state is maintained across contexts such as route, merchant, issuer, and their interactions.

All historical features obey the decision-time information set: the current outcome is added only after the feature row for that decision has been emitted.

### 2.2 Cross-feature engineering

A route is not universally good or bad. Its performance can depend on the transaction population being routed through it.

The feature set therefore includes explicit compatibility features such as:

- `bankcat_route`
- `cardnetwork_route`
- `cardtype_route`
- `cardlevel_route`
- `mcc_route`

as well as historical interaction state such as merchant × route, issuer × route, card type × route, and MCC × route.

Conceptually, the model can represent quantities such as:

```math
P_t(Y=1\mid \mathrm{Merchant},\ \mathrm{Route})
```

and:

```math
P_t(Y=1\mid \mathrm{Issuer},\ \mathrm{Route})
```

These features encode **route compatibility**, rather than only the separate average quality of a transaction context and a route.

The notebook links this engineering step to the frozen XGBoost feature-importance artifact. It reports the normalized total gain assigned to explicit cross features and lists the strongest cross features together with their rank among all features in each stage model.

This gain analysis is model-internal predictive evidence, not a causal feature effect.

---

## 3. Model-family comparison

Model comparison is performed **after the feature representation is defined**.

This ordering is important because comparing algorithms that receive different information would confound feature engineering with model choice.

The original A1 experiment compares:

- CatBoost
- LightGBM
- XGBoost
- linear SGD/logistic baseline

on the same frozen feature architecture and chronological development folds.

XGBoost is the final selected model family.

---

## 4. Stage-specific probability models

A logical payment can contain multiple attempts:

```math
A1 \rightarrow A2 \rightarrow A3
```

Later attempts are selected populations and contain information that did not exist at the first attempt.

The final system therefore uses separate models:

```math
P(S_1\mid X,r_1,H_t)
```

```math
P(S_2\mid X,r_2,A1_{\mathrm{observed}},H_t)
```

```math
P(S_3\mid X,r_3,A1_{\mathrm{observed}},A2_{\mathrm{observed}},H_t)
```

Current-attempt provider response fields are excluded because they are only known after the attempt occurs.

---

## 5. Dataset

The standardized attempt-level dataset contains approximately:

| Stage | Rows | Observed success rate |
|---|---:|---:|
| A1 | 595,434 | 8.52% |
| A2 | 317,819 | 4.91% |
| A3 | 122,900 | 1.80% |
| **Total attempt rows** | **1,036,153** | — |

There are approximately **595,434 logical transactions**.

The company transaction dataset is not redistributed publicly.

---

## 6. Predictive results

Untouched June evaluation:

| Stage | ROC-AUC | Average Precision | Log-loss skill | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| A1 | 0.797 | 0.324 | 0.171 | 0.0788 | 0.0055 |
| A2 | 0.794 | 0.212 | 0.151 | 0.0440 | 0.0019 |
| A3 | 0.871 | 0.137 | 0.207 | 0.0180 | 0.0026 |

Average-precision lift relative to stage prevalence is approximately:

- A1: **3.22×**
- A2: **4.17×**
- A3: **7.00×**

The notebook also compares the mean predicted probability with the success rate actually observed on the logged route and presents fixed-probability-bin reliability diagrams.

The sparse high-probability A2/A3 bins are retained rather than hidden, but their transaction counts are shown explicitly so visually unusual points are interpreted in proportion to the amount of evidence behind them.

---

## 7. Offline LP benchmark

The linear program is a **full-hindsight benchmark**, not the live routing algorithm.

At the end of the evaluation period, it sees the complete set of transactions and their frozen model scores simultaneously and finds the best assignment under the stated route-volume constraints.

For the success objective:

```math
\max_x \sum_{i,r} x_{ir}\hat p_{ir}
```

For expected approved transaction value:

```math
\max_x \sum_{i,r} x_{ir}\,Amount_i\,\hat p_{ir}
```

subject to one route per event and the stated route-volume constraints.

Because the LP sees the full period, it answers:

> If the entire period had been known in advance, what is the best model-implied assignment under the constraints?

It establishes the **100% model-implied LP opportunity benchmark** against which chronological policies can be compared.

Reference June results:

- success-optimal LP: **+188.8 model-implied approvals** and about **+1.299M expected approved transaction value**;
- value-optimal LP: about **+1.424M expected approved transaction value**.

These are counterfactual model estimates, not causal production uplift.

---

## 8. Online capacity-aware policy

The production-style problem is different from the LP because future transactions are unknown.

When a transaction arrives, the system must choose a route immediately using only the current transaction, currently eligible routes, current historical state, and capacity already consumed.

A greedy policy chooses:

```math
r_i^{\mathrm{greedy}}=\arg\max_r \hat p_{ir}
```

This can consume a strong route too aggressively early in the period.

The online policy therefore defines route pressure:

```math
Pressure_{r,t}
=
\frac{A_{r,t}-B_{r,t}}
{\max(0.3B_{r,t},1)}
```

and uses the success-oriented decision score:

```math
Score_{ir,t}
=
\hat p_{ir,t}
-
\lambda\,Pressure_{r,t}
```

Here, $\lambda$ controls how strongly route scarcity affects the decision:

| Lambda behavior | Interpretation |
|---|---|
| $\lambda=0$ | greedy probability maximization |
| small positive $\lambda$ | scarcity matters mainly when candidate probabilities are close |
| larger $\lambda$ | stronger willingness to preserve an over-used route |

The reference success policy uses **$\lambda=0.10$**.

The notebook includes a synthetic four-transaction example showing the full chronological process and how earlier route choices change the state used for later decisions.

---

## 9. Capacity and optimization results

For the June reference evaluation:

- greedy captures about **46%** of the model-implied LP success opportunity;
- success pressure with **$\lambda=0.10$** captures about **86%**;
- the value-oriented pressure policy captures about **85%** of the value LP opportunity.

Absolute route deviation is:

```math
d_r
=
\left|
\frac{V_r^{\mathrm{policy}}}
{V_r^{\mathrm{baseline}}}
-
1
\right|
```

For example, $d_r=0.10$ means the final policy route volume is 10% away from its baseline volume.

The main capacity constraint is approximately ±30% around the reference route volume.

A route is called **near the boundary** when:

```math
d_r \ge 0.27
```

That is, it lies within three percentage points of the ±30% limit.

Observed capacity behavior:

- greedy: 9 of 12 routes near the boundary; mean absolute route deviation about 26%;
- success pressure, $\lambda=0.10$: 1 of 12 routes near the boundary; mean absolute route deviation about 10%;
- nearby pressure policies show similarly controlled aggregate behavior.

---

## 10. Robustness

The later-period robustness analysis checks whether the result depends on one finely tuned $\lambda$ value.

Moderate success-pressure settings remain stronger than greedy on model-implied approvals, while the value-oriented setting trades some modeled approvals for greater expected approved transaction value.

Movement is defined as:

```math
Movement
=
\frac{1}{N}
\sum_i
\mathbf{1}
\left(
R_i^{\mathrm{policy}}
\neq
R_i^{\mathrm{logged}}
\right)
```

Movement measures how much operational behavior changes. It is not itself a measure of benefit.

---

## 11. Repository structure

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

`src/features.py` contains leakage-safe temporal and interaction-feature utilities.

`src/models.py` contains predictive-model and probability-evaluation utilities.

`src/optimization.py` contains LP, capacity, greedy, and pressure-policy logic.

`src/showcase.py` discovers frozen artifacts and exports compact model configurations for reproducibility.

---

## 12. Reproducibility

The public repository contains code, configuration, the final notebook, and compact result artifacts.

Large/private transaction data and heavy frozen model artifacts remain in the authorized project Google Drive.

The notebook mounts the authorized Drive, discovers the existing frozen checkpoints, and uses them rather than retraining the full project during the showcase run.

---

## 13. Scientific limitation

Observed-route predictive metrics are factual historical evaluations.

Alternative-route policy gains are **model-implied counterfactual estimates** because only the outcome of the historically selected route is observed.

The project therefore does not claim causal production uplift from offline replay.

The intended validation path is:

```math
\mathrm{offline\ validation}
\rightarrow
\mathrm{shadow\ deployment}
\rightarrow
\mathrm{controlled\ A/B\ test}
\rightarrow
\mathrm{production\ rollout}
```
