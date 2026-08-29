# FlexFactor Routing Final Project

Main artifact: `FlexFactor_Final_Project.ipynb`.

The notebook connects to the real FlexFactor project data/checkpoints in Google Drive, while reusable implementation is kept in `src/`.

## Drive
The notebook mounts Drive at `/content/gdrive` and auto-discovers the existing project artifacts, including:

- `flexfactor_route_model_checkpoints/standardized_attempts.parquet`
- `flexfactor_a1_model_selection/a1_model_selection_v1`
- `flexfactor_a1_optuna/a1_xgboost_optuna_v1`
- `flexfactor_retry_models/a2_a3_objective_xgb_v2`
- `flexfactor_backtest/chronological_shadow_v1`
- `flexfactor_policy_v2/...`

A copy of the standardized dataset may also live at:
`MyDrive/FlexFactor_Final_Project/data/standardized_attempts.parquet`.

## GitHub / Colab
Open the notebook from GitHub in Colab. Its first cell clones/pulls this public repository so imports from `src/` work, then it mounts Google Drive.

## Notebook coverage
The showcase includes real-data EDA, concept drift, original model-family comparison when the saved table is present, temporal-memory motivation, A1/A2/A3 predictive metrics, feature-family and individual feature importance, LP oracle interpretation, greedy-vs-pressure results, chronological robustness, and a detailed real June inference/decision trace from saved candidate scores.

## Scientific claim discipline
LP and online-policy gains are calibrated-model-implied counterfactual estimates, not causal production lift. A controlled online/A-B test is required for the latter.
