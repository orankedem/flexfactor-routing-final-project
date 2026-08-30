# FlexFactor — AI-Based Constrained Payment Routing Optimization

This repository contains the final technical showcase for a dynamic payment-routing decision system.

## Main artifact

Open **`FlexFactor_Final_Project.ipynb`** in Google Colab and run it top-to-bottom.

The notebook connects public GitHub code to the existing private Google Drive data/model checkpoints.

## What the showcase covers

The project is presented as a deliberate engineering progression:

1. payment-routing financial problem and multi-attempt structure;
2. adjusted route-signal feasibility analysis;
3. chronological concept drift;
4. fixed-window historical memory and its cutoff problem;
5. continuous 3/14/60/180-day exponential state;
6. exact A1 model-family comparison and final XGBoost selection;
7. stage-specific A1/A2/A3 models and leakage rules;
8. factual AUC/AP/log-loss/Brier/ECE and reliability plots;
9. family-level and individual feature importance;
10. exact frozen-model configuration audit;
11. final system architecture;
12. offline LP oracle;
13. greedy vs capacity-aware pressure routing;
14. capacity behavior and chronological robustness;
15. a real A1 transaction-level decision walkthrough;
16. re-running the exact frozen XGBoost model on the exact candidate feature rows to verify saved policy scores;
17. counterfactual claim boundaries and live A/B validation path.

## Repository architecture

```text
FlexFactor_Final_Project.ipynb   # final narrative + results + demo
src/
  data.py                        # data/schema utilities
  features.py                    # temporal/cross feature implementation
  models.py                      # model/metric utilities
  optimization.py                # LP + pressure-policy utilities
  showcase.py                    # artifact discovery, calibration, exact inference
configs/
  policy_final.json              # final reference policy configuration
  README.md                      # exact model-config export instructions
artifacts/
  reference_*.csv                # small frozen headline-result tables
```

## Private/large artifacts

Raw/standardized company data, full candidate matrices and frozen model binaries remain in Google Drive rather than public GitHub.

The notebook expects Drive at:

```text
/content/gdrive/MyDrive
```

and discovers the existing FlexFactor checkpoint directories automatically.

## Exact model configs

When the notebook runs, it reads the **actual frozen model metadata** and exports compact exact configs to:

```text
configs/a1_final.json
configs/a2_final.json
configs/a3_final.json
```

Upload those generated files to GitHub once the final run is verified.

## Scientific claim boundary

Observed-route predictive metrics are factual. Capacity/movement statistics are operational. Alternative-route approval/value gains are model-implied counterfactual estimates under the frozen probability models. The full-hindsight LP is an offline model-based oracle, not a causal upper bound.

True production lift requires controlled live validation / A-B testing.
