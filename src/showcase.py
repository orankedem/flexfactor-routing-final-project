from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd


def first_existing(paths, label=None):
    for p in map(Path, paths):
        if p.exists():
            return p
    if label:
        raise FileNotFoundError(
            f"Could not locate {label}. Tried:\n" + "\n".join(map(str, paths))
        )
    return None


def discover_artifacts(drive_root: str | Path):
    """Discover frozen artifacts produced during FlexFactor development."""
    root = Path(drive_root)

    found = {
        "data": first_existing([
            root / "FlexFactor_Final_Project/data/standardized_attempts.parquet",
            root / "flexfactor_route_model_checkpoints/standardized_attempts.parquet",
        ]),
        "a1_model_selection": first_existing([
            root / "flexfactor_a1_model_selection/a1_model_selection_v1",
        ]),
        "a1_model": first_existing([
            root / "flexfactor_a1_optuna/a1_xgboost_optuna_v1",
        ]),
        "retry_models": first_existing([
            root / "flexfactor_retry_models/a2_a3_objective_xgb_v2",
        ]),
        "a1_policy": first_existing([
            root / "flexfactor_a1_policy/a1_xgb_policy_stress_v1",
        ]),
        "backtest": first_existing([
            root / "flexfactor_backtest/chronological_shadow_v1",
        ]),
        "feature_policy": first_existing([
            root / "flexfactor_policy_v2/success_vs_value_online_v2",
            root / "flexfactor_policy_v2/feature_importance_lp_online_v1",
        ]),
        "final_policy": first_existing([
            root / "flexfactor_policy_v2/final_online_policy_selection_v3",
        ]),
        "robustness": first_existing([
            root / "flexfactor_policy_v2/chronological_policy_robustness_v4",
        ]),
    }

    if found["a1_model"]:
        a1 = found["a1_model"]
        found.update({
            "a1_model_json": a1 / "final/a1_xgboost_optuna.json",
            "a1_metadata": a1 / "final/final_model_metadata.json",
            "a1_predictions": a1 / "final/june_predictions.parquet",
            "a1_metrics": a1 / "final/june_metrics.json",
            "a1_best_config": a1 / "summary/best_optuna_config.json",
        })

    if found["retry_models"]:
        rr = found["retry_models"]
        found.update({
            "a2_model_json": rr / "final/a2_xgboost.json",
            "a2_metadata": rr / "final/a2_model_metadata.json",
            "a2_predictions": rr / "final/a2_june_predictions.parquet",
            "a2_metrics": rr / "final/a2_june_metrics.json",
            "a3_model_json": rr / "final/a3_xgboost.json",
            "a3_metadata": rr / "final/a3_model_metadata.json",
            "a3_predictions": rr / "final/a3_june_predictions.parquet",
            "a3_metrics": rr / "final/a3_june_metrics.json",
        })

    return found


def artifact_audit(artifacts: dict) -> pd.DataFrame:
    rows = []
    for name, value in artifacts.items():
        p = None if value is None else Path(value)
        rows.append({
            "artifact": name,
            "found": bool(p is not None and p.exists()),
            "path": None if p is None else str(p),
        })
    return pd.DataFrame(rows)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_model_metadata(metadata: dict, stage: str) -> dict:
    """Compact reproducibility metadata suitable for the public repository."""
    feature_cols = metadata.get("feature_columns")
    cats = metadata.get("categorical_features", []) or []
    nums = metadata.get("numeric_features", []) or []

    if not feature_cols:
        feature_cols = list(cats) + list(nums)

    return {
        "stage": stage,
        "family": metadata.get("family", "xgboost"),
        "architecture": metadata.get("architecture"),
        "params": metadata.get("best_params", metadata.get("params")),
        "iterations": metadata.get(
            "final_iterations",
            metadata.get("fixed_iterations"),
        ),
        "development_months": metadata.get("development_months"),
        "tuning_months": metadata.get("tuning_months"),
        "final_test_month": metadata.get(
            "final_test_month",
            metadata.get("test_month", "2026-06"),
        ),
        "feature_columns": feature_cols,
        "categorical_features": cats,
        "numeric_features": nums,
        "feature_count": len(feature_cols or []),
        "categorical_feature_count": len(cats),
        "numeric_feature_count": len(nums),
        "history_half_lives_days": metadata.get(
            "history_half_lives_days",
            [3.0, 14.0, 60.0, 180.0],
        ),
        "strict_history_rule": metadata.get(
            "strict_history_rule",
            "historical state uses only information available before decision time",
        ),
    }


def export_exact_configs(artifacts: dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stage_meta = {
        "A1": artifacts.get("a1_metadata"),
        "A2": artifacts.get("a2_metadata"),
        "A3": artifacts.get("a3_metadata"),
    }

    exported = {}

    for stage, path in stage_meta.items():
        if path is None or not Path(path).exists():
            continue

        clean = sanitize_model_metadata(load_json(path), stage)
        out = output_dir / f"{stage.lower()}_final.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2, default=str)
        exported[stage] = out

    policy = {
        "capacity_scope_primary": "shared_all_attempts",
        "capacity_shift": 0.30,
        "pressure_definition": (
            "(policy cumulative - baseline cumulative) / "
            "max(0.30 * baseline cumulative, 1)"
        ),
        "success_score": "p_success - lambda * pressure",
        "approved_value_score": (
            "amount * p_success - lambda * median_amount * pressure"
        ),
        "reference_lambdas": {
            "success": 0.10,
            "balanced_success": 0.15,
            "approved_value": 0.20,
        },
        "evaluation_note": (
            "alternative-route gains are model-implied counterfactual estimates; "
            "causal lift requires controlled live validation"
        ),
    }

    policy_out = output_dir / "policy_final.json"
    with open(policy_out, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    exported["POLICY"] = policy_out

    return exported
