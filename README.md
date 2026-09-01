# FlexFactor — AI-Based Constrained Payment Routing Optimization

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/orankedem/flexfactor-routing-final-project.git
cd flexfactor-routing-final-project
```

### 2. Install the dependencies

```bash
pip install -r requirements.txt
```

The project is designed to be run in **Google Colab**. The public repository contains the notebook, source code, configuration files, compact reference artifacts, and `requirements.txt`.

### 3. Open the main notebook

Open:

```text
FlexFactor_Final_Project.ipynb
```

in Google Colab and run the notebook **from top to bottom**.

The setup cells mount Google Drive and locate the frozen project artifacts used by the final showcase. When prompted, authorize access to the project Drive.

### 4. Private data and frozen artifacts

The transaction dataset and large trained-model/checkpoint artifacts are **not included in the public repository**. A full artifact-backed reproduction therefore requires access to the authorized project Google Drive.

Without that private Drive access, the public repository still provides the complete source-code structure, methodology, configurations, compact reference results, and executed notebook outputs needed to inspect the project.

### 5. Security

The public repository must not contain API keys, passwords, access tokens, credentials, or other secrets. No private credentials are required by the source code itself; Google Drive authorization is handled interactively by Colab.

---

## Project objective

FlexFactor is an AI-based decision system for routing declined/card-payment attempts through alternative payment routes.

For every candidate route, the predictive layer estimates:

`P(success | transaction context, candidate route, historical state)`

The decision layer then chooses a route while respecting route-volume constraints.

The project is therefore not only a binary classifier. Its full flow is:

`transaction → dynamic state → route probabilities → capacity-aware decision → outcome/state update`

The main graded artifact is **`FlexFactor_Final_Project.ipynb`**. It contains the methodology, empirical results, calibration analysis, feature engineering, model comparison, optimization benchmarks, worked online-routing example, and scientific limitations of the offline evaluation.

---

## 1. Why the problem is dynamic

Raw approval rates change through time, but a changing aggregate rate can also reflect a changing mix of merchants, routes, cards, or amounts.

The notebook therefore examines drift using three views:

- monthly first-attempt approval rates;
- a traffic-mix-adjusted residual using merchant × route × card type × amount-decile contexts;
- a large recurring near-comparable cohort selected by volume and recurrence, not by the amount of drift observed.

The result motivates a model whose representation can change with the environment rather than relying only on static identifiers.

---

## 2. Feature engineering

### 2.1 Continuous temporal memory

Historical performance is represented with exponentially decayed state rather than one hard rolling-window cutoff.

Conceptually:

```text
historical weight = 0.5 ^ (age / half-life)
```

The final representation exposes several memory speeds:

```text
half-lives = 3, 14, 60, and 180 days
```

Historical state is maintained across contexts such as route, merchant, issuer, and their interactions.

All historical features obey the decision-time information set: the current outcome is not added until after the feature row for that decision has been emitted.

### 2.2 Cross-feature engineering

A route is not universally good or bad. Its performance can depend on the transaction population being routed through it.

The feature set therefore includes explicit compatibility features such as:

- `bankcat_route`
- `cardnetwork_route`
- `cardtype_route`
- `cardlevel_route`
- `mcc_route`

and historical interaction state such as:

- merchant × route
- issuer × route
- card type × route
- MCC × route

The practical distinction is:

```text
route quality alone
```

versus:

```text
route quality for this merchant / issuer / card context
```

The notebook links this engineering step to the frozen XGBoost feature-importance artifact. It reports the normalized total gain assigned to explicit cross features and lists the strongest cross features together with their rank among all features in each stage model.

This is model-internal predictive evidence, not a causal feature effect.

---

## 3. Model-family comparison

Model comparison is performed **after the feature representation is defined**.

This is important because comparing two algorithms that receive different information would confound feature engineering with model choice.

The original A1 experiment compares CatBoost, LightGBM, XGBoost, and a linear SGD/logistic baseline on the same frozen feature architecture and chronological development folds. XGBoost is the final selected model family.

---

## 4. Stage-specific probability models

A logical payment can contain multiple attempts:

```text
A1 → A2 → A3
```

Later attempts are selected populations and contain information that did not exist at the first attempt.

The final system therefore uses separate models:

```text
A1: P(success | current transaction, candidate route, historical state)

A2: P(success | current transaction, candidate route,
                observed A1 state, historical state)

A3: P(success | current transaction, candidate route,
                observed A1 + A2 state, historical state)
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

The notebook also compares the mean predicted probability with the success rate actually observed on the logged route and presents fixed-probability-bin reliability diagrams.

The sparse high-probability A2/A3 bins are retained rather than hidden, but the number of transactions behind those visually unusual points is shown explicitly.

---

## 7. Offline LP benchmark

The linear program is a **full-hindsight benchmark**, not the live routing algorithm.

At the end of the evaluation period, it sees the entire set of transactions and their frozen model scores simultaneously and finds the best assignment under the stated route-volume constraints.

Success objective:

```text
maximize the sum of predicted success probabilities
across all transaction-route assignments
```

Approved-value objective:

```text
maximize the sum of:
transaction amount × predicted success probability
```

Because the LP sees the full period, it answers:

> If the entire period had been known in advance, what is the best model-implied assignment under the constraints?

It establishes the **100% model-implied opportunity benchmark** against which chronological policies can be compared.

Reference June results:

- success-optimal LP: **+188.8 model-implied approvals** and about **+1.299M expected approved transaction value**;
- value-optimal LP: about **+1.424M expected approved transaction value**.

These are counterfactual model estimates, not causal production uplift.

---

## 8. Online capacity-aware policy

The production-style problem is different from the LP because future transactions are unknown.

When a transaction arrives, the system must choose a route immediately using only:

- the current transaction;
- currently eligible routes;
- current historical state;
- capacity already consumed.

A greedy policy always chooses the largest current predicted success probability. This can consume a strong route too aggressively early in the period.

The online policy therefore adds a route-pressure term.

In plain notation:

```text
pressure(route, time)
    =
    (policy cumulative use - baseline cumulative use)
    / max(0.30 × baseline cumulative use, 1)
```

The success-oriented decision score is:

```text
adjusted score
    =
    predicted success probability
    - lambda × route pressure
```

`lambda` controls how strongly route scarcity affects the decision:

| Lambda behavior | Interpretation |
|---|---|
| `lambda = 0` | greedy probability maximization |
| small positive lambda | scarcity matters when route probabilities are close |
| larger lambda | stronger willingness to preserve an over-used route |

The reference success policy uses **lambda = 0.10**.

The notebook includes a synthetic four-transaction example showing the full chronological process and how earlier route choices change the state used for later decisions.

---

## 9. Capacity and optimization results

For the June reference evaluation:

- greedy captures about **46%** of the model-implied LP success opportunity;
- success pressure with **lambda = 0.10** captures about **86%**;
- the value-oriented pressure policy captures about **85%** of the value LP opportunity.

Route deviation is defined as:

```text
absolute route deviation
    =
    absolute value of:
    (policy route volume / baseline route volume) - 1
```

Example: a deviation of `0.10` means the final route volume is 10% away from the baseline volume.

The main capacity constraint is approximately ±30% around the reference route volume. A route is called **near the boundary** when its absolute deviation is at least 27%, i.e. within three percentage points of that limit.

Observed capacity behavior:

- greedy: 9 of 12 routes near the boundary; mean absolute route deviation about 26%;
- success pressure, lambda = 0.10: 1 of 12 near the boundary; mean absolute deviation about 10%;
- nearby pressure policies show similarly controlled aggregate behavior.

---

## 10. Robustness

The later-period robustness analysis checks whether the result depends on one finely tuned lambda value.

Moderate success-pressure settings remain stronger than greedy on model-implied approvals, while the value-oriented setting trades some modeled approvals for greater expected approved transaction value.

Movement is defined as:

```text
movement rate
    =
    share of replayed events where:
    policy route != historically logged route
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

```text
offline validation
    ↓
shadow deployment
    ↓
controlled A/B test
    ↓
production rollout
```
