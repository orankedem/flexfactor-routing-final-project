from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable
import json
import math

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
    """Discover the exact saved artifacts produced during project development."""
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
            "retry_best_params": rr / "summary/best_params.json",
            "retry_architecture": rr / "summary/architecture_aggregate.csv",
        })

    if found["a1_policy"]:
        p = found["a1_policy"]
        found.update({
            "a1_candidate_matrix": p / "candidate_features/candidate_model_matrix.parquet",
            "a1_candidate_scores_original": p / "scores/candidate_scores_xgb.parquet",
        })

    return found


def artifact_audit(artifacts: dict) -> pd.DataFrame:
    rows = []
    for name, value in artifacts.items():
        if value is None:
            rows.append({"artifact": name, "found": False, "path": None})
        else:
            p = Path(value)
            rows.append({"artifact": name, "found": p.exists(), "path": str(p)})
    return pd.DataFrame(rows)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_model_metadata(metadata: dict, stage: str) -> dict:
    out = {
        "stage": stage,
        "family": metadata.get("family", "xgboost"),
        "architecture": metadata.get("architecture"),
        "params": metadata.get("best_params", metadata.get("params")),
        "iterations": metadata.get("final_iterations", metadata.get("fixed_iterations")),
        "development_months": metadata.get("development_months"),
        "tuning_months": metadata.get("tuning_months"),
        "final_test_month": metadata.get("final_test_month", metadata.get("test_month", "2026-06")),
        "feature_columns": metadata.get("feature_columns"),
        "categorical_features": metadata.get("categorical_features", []),
        "numeric_features": metadata.get("numeric_features", []),
        "history_features": metadata.get("history_features"),
        "history_half_lives_days": metadata.get("history_half_lives_days", [3.0, 14.0, 60.0, 180.0]),
        "strict_history_rule": metadata.get(
            "strict_history_rule",
            "historical state uses information available strictly before decision time",
        ),
        "current_attempt_response_used": metadata.get("current_attempt_response_used"),
    }
    if not out["feature_columns"]:
        out["feature_columns"] = list(out["categorical_features"] or []) + list(out["numeric_features"] or [])
    out["feature_count"] = len(out["feature_columns"] or [])
    out["categorical_feature_count"] = len(out["categorical_features"] or [])
    out["numeric_feature_count"] = len(out["numeric_features"] or [])
    return out


def export_exact_configs(artifacts: dict, output_dir: str | Path):
    """Export compact, GitHub-friendly exact model configs from Drive metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = {}

    for stage, key in [("A1", "a1_metadata"), ("A2", "a2_metadata"), ("A3", "a3_metadata")]:
        path = artifacts.get(key)
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
        "pressure_definition": "(A_rt - B_rt) / max(0.30 * B_rt, 1)",
        "success_score": "p_success - lambda * pressure",
        "approved_value_score": "amount * p_success - lambda * median_amount * pressure",
        "reference_lambdas": {"success": 0.10, "balanced_success": 0.15, "approved_value": 0.20},
        "claim_guardrail": "alternative-route gains are model-implied; causal lift requires controlled live validation",
    }
    p = output_dir / "policy_final.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)
    exported["POLICY"] = p
    return exported


def reliability_table(y_true, y_prob, n_bins=10):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(p, edges[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        m = ids == b
        if not np.any(m):
            continue
        rows.append({
            "bin": b + 1,
            "n": int(m.sum()),
            "predicted_mean": float(p[m].mean()),
            "observed_rate": float(y[m].mean()),
            "abs_gap": float(abs(p[m].mean() - y[m].mean())),
        })
    return pd.DataFrame(rows)


def load_candidate_universe(backtest_root, month="2026-06", attempts=(1, 2, 3)):
    backtest_root = Path(backtest_root)
    parts = []
    for attempt in attempts:
        p = backtest_root / f"candidate_scores/a{attempt}_candidates.parquet"
        c = pd.read_parquet(p)
        c["timestamp"] = pd.to_datetime(c["timestamp"], utc=True)
        c = c[c["timestamp"].dt.strftime("%Y-%m").eq(month)].copy()
        c["attempt"] = attempt
        if "candidate_allowed" in c:
            c = c[c["candidate_allowed"].astype(bool)].copy()
        parts.append(c)
    cand = pd.concat(parts, ignore_index=True)
    cand["candidate_route"] = cand["candidate_route"].astype(str)
    cand["logged_route"] = cand["logged_route"].astype(str)
    cand["amount_numeric"] = pd.to_numeric(cand["amount_numeric"], errors="coerce").fillna(0.0)
    return cand


def load_event_baseline(candidates):
    logged = candidates[candidates["is_logged_route"].astype(bool)].copy()
    chk = logged.groupby("event_id").size()
    if not chk.eq(1).all():
        raise ValueError("Every event must have exactly one logged-route candidate")
    cols = ["event_id", "attempt", "timestamp", "logged_route", "success", "amount_numeric", "model_score"]
    for c in ["TransactionId", "_policy_tx_id", "_row_id"]:
        if c in logged.columns:
            cols.append(c)
    return (
        logged[cols]
        .rename(columns={"model_score": "logged_model_score"})
        .sort_values(["timestamp", "attempt", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def route_pressure(route, baseline_counts, policy_counts, shift=.30):
    b = baseline_counts[route]
    a = policy_counts[route]
    return (a - b) / max(shift * b, 1.0)


def strict_bounds(v, shift=.30):
    return int(math.ceil((1-shift)*v-1e-12)), int(math.floor((1+shift)*v+1e-12))


def feasible_after_choice(route, baseline_counts, policy_counts, shift=.30):
    all_routes = set(baseline_counts) | set(policy_counts) | {route}
    for r in all_routes:
        b = baseline_counts[r]
        a = policy_counts[r] + (1 if r == route else 0)
        lo, hi = strict_bounds(b, shift)
        if a < lo or a > hi:
            return False
    return True


def detailed_pressure_trace(
    candidates, *, lambda_=0.10, target_event_id=None,
    choose_interesting=True, shift=.30, attempt_filter=None,
):
    """Replay chronology until a real event where pressure changes the greedy decision."""
    baseline = load_event_baseline(candidates)
    lookup = {e: g.copy() for e, g in candidates.groupby("event_id", sort=False)}
    baseline_counts = defaultdict(int)
    policy_counts = defaultdict(int)

    for _, ev in baseline.iterrows():
        eid = ev["event_id"]
        logged = str(ev["logged_route"])
        baseline_counts[logged] += 1
        g = lookup[eid].copy()
        rows = []

        for _, r in g.iterrows():
            route = str(r["candidate_route"])
            p = float(r["model_score"])
            pr = route_pressure(route, baseline_counts, policy_counts, shift)
            score = p - lambda_ * pr
            feasible = feasible_after_choice(route, baseline_counts, policy_counts, shift)
            row = {
                "event_id": eid,
                "attempt": int(ev["attempt"]),
                "route": route,
                "raw_p_success": p,
                "baseline_cumulative": baseline_counts[route],
                "policy_cumulative_before": policy_counts[route],
                "pressure": pr,
                "adjusted_score": score,
                "feasible": feasible,
                "is_logged_route": bool(r["is_logged_route"]),
            }

            # A1 chronological-backtest event IDs were saved as:
            #     A1__<policy_tx_id>
            # The backtest parquet intentionally omitted _candidate_id, but
            # this lets us recover the original candidate row exactly.
            if int(ev["attempt"]) == 1 and str(eid).startswith("A1__"):
                try:
                    row["_policy_tx_id"] = int(str(eid).split("A1__", 1)[1])
                except Exception:
                    row["_policy_tx_id"] = str(eid).split("A1__", 1)[1]
            for c in [
                "_candidate_id", "_policy_tx_id", "_row_id", "TransactionId",
                "decay__route__rate__hl14p0d", "hist__merchant_route__logn__exp",
            ]:
                if c in r.index:
                    row[c] = r[c]
            rows.append(row)

        table = pd.DataFrame(rows).sort_values(
            ["adjusted_score", "raw_p_success"], ascending=False, kind="stable"
        )
        feasible_table = table[table["feasible"]]
        chosen = logged if feasible_table.empty else str(feasible_table.iloc[0]["route"])
        raw_best = str(table.sort_values(["raw_p_success", "route"], ascending=[False, True], kind="stable").iloc[0]["route"])

        stage_ok = attempt_filter is None or int(ev["attempt"]) == int(attempt_filter)
        interesting = chosen != raw_best
        hit = (
            (target_event_id is not None and eid == target_event_id)
            or (target_event_id is None and choose_interesting and interesting and stage_ok)
        )
        if hit:
            meta = {
                "event_id": eid,
                "attempt": int(ev["attempt"]),
                "timestamp": ev["timestamp"],
                "amount": float(ev["amount_numeric"]),
                "logged_route": logged,
                "raw_probability_best_route": raw_best,
                "pressure_policy_route": chosen,
                "lambda": lambda_,
            }
            for c in ["TransactionId", "_policy_tx_id", "_row_id"]:
                if c in ev.index:
                    meta[c] = ev[c]
            table["selected"] = table["route"].eq(chosen)
            table["raw_greedy_selected"] = table["route"].eq(raw_best)
            return meta, table.reset_index(drop=True)

        policy_counts[chosen] += 1

    raise ValueError("No matching/interesting event found")


def _read_candidate_rows(matrix_path, candidate_ids, columns):
    ids = list(candidate_ids)
    cols = list(dict.fromkeys(["_candidate_id"] + list(columns)))
    try:
        frame = pd.read_parquet(
            matrix_path, columns=cols, filters=[("_candidate_id", "in", ids)]
        )
    except Exception:
        # Avoid loading the full large candidate matrix into RAM.
        import pyarrow.dataset as ds
        dataset = ds.dataset(str(matrix_path), format="parquet")
        table = dataset.to_table(
            columns=cols,
            filter=ds.field("_candidate_id").isin(ids),
        )
        frame = table.to_pandas()
    return frame


def _decode_xgboost_string_categories(enc: dict):
    """
    Decode string categories from XGBoost's serialized category container.

    JSON/UBJSON serialization may expose byte values as signed integers in
    some environments. Values in [-128, 255] are therefore normalized back
    to uint8 with two's-complement semantics.
    """
    offsets = [int(v) for v in enc.get("offsets", [])]
    values = [int(v) for v in enc.get("values", [])]

    if not offsets:
        return None

    # Normal XGBoost string storage is UTF-8 bytes.
    if all(-128 <= v <= 255 for v in values):
        raw = bytes((v & 0xFF) for v in values)
        return [
            raw[offsets[i]:offsets[i + 1]].decode("utf-8", errors="strict")
            for i in range(len(offsets) - 1)
        ]

    # Defensive fallback for a JSON representation containing Unicode
    # codepoints rather than bytes.
    if all(0 <= v <= 0x10FFFF for v in values):
        return [
            "".join(chr(v) for v in values[offsets[i]:offsets[i + 1]])
            for i in range(len(offsets) - 1)
        ]

    raise ValueError("Unsupported serialized XGBoost string-category encoding")


def _categories_from_booster_api(model_path: str | Path) -> dict:
    """
    Preferred category extraction path for XGBoost >= 3.1.

    Uses Booster.get_categories(export_to_arrow=True), which delegates the
    model's category representation to XGBoost itself instead of reverse-
    engineering its serialization.
    """
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    categories = booster.get_categories(export_to_arrow=True)
    arrow_items = categories.to_arrow()

    levels = {}
    for item in arrow_items:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue

        feature_name, array = item
        if array is None:
            continue

        # PyArrow arrays expose to_pylist(). Keep the stored scalar type.
        if hasattr(array, "to_pylist"):
            levels[str(feature_name)] = array.to_pylist()
        else:
            levels[str(feature_name)] = list(array)

    return levels


def _categories_from_model_json(model_path: str | Path) -> dict:
    """
    Fallback parser for the documented XGBoost JSON category format.
    """
    model_path = Path(model_path)
    with open(model_path, "r", encoding="utf-8") as f:
        model_json = json.load(f)

    learner = model_json["learner"]
    feature_names = learner.get("feature_names", [])
    feature_types = learner.get("feature_types", [])

    cats_block = (
        learner
        .get("gradient_booster", {})
        .get("model", {})
        .get("cats", {})
    )
    encodings = cats_block.get("enc", [])

    levels = {}

    for i, (name, ftype) in enumerate(zip(feature_names, feature_types)):
        if ftype != "c" or i >= len(encodings):
            continue

        enc = encodings[i]

        if "offsets" in enc and "values" in enc:
            decoded = _decode_xgboost_string_categories(enc)
            if decoded is not None:
                levels[name] = decoded
                continue

        if "values" in enc:
            levels[name] = enc["values"]

    return levels


def load_model_category_levels(model_path: str | Path) -> dict:
    """
    Load exact training-time category vocabularies from the frozen booster.

    The official XGBoost category-export API is used first. A parser for the
    documented serialized JSON format is retained only as a compatibility
    fallback.
    """
    api_error = None

    try:
        levels = _categories_from_booster_api(model_path)
        if levels:
            return levels
    except Exception as exc:
        api_error = exc

    try:
        levels = _categories_from_model_json(model_path)
        if levels:
            return levels
    except Exception as json_exc:
        if api_error is not None:
            raise RuntimeError(
                "Could not extract frozen-model categorical vocabularies "
                f"through either XGBoost's API ({api_error}) or JSON fallback "
                f"({json_exc})."
            ) from json_exc
        raise

    return {}


def align_categories_to_frozen_model(
    X: pd.DataFrame,
    categorical_columns,
    model_category_levels: dict,
):
    """
    Align inference categories to the exact vocabulary stored in the model.

    For an inference value unseen during training:
      1. use an explicit __OTHER__ category if the model was trained with one;
      2. otherwise represent it as missing.

    The function returns both the aligned frame and a per-row count of fields
    that required unseen-category handling.
    """
    out = X.copy()
    unseen_per_row = pd.Series(0, index=out.index, dtype="int64")

    for c in categorical_columns:
        if c not in out.columns:
            continue

        categories = model_category_levels.get(c)

        # If the serialized booster does not expose a vocabulary for this
        # metadata-declared categorical column, keep a stable pandas category.
        if categories is None:
            out[c] = (
                out[c]
                .astype("string")
                .fillna("__MISSING__")
                .astype("category")
            )
            continue

        # Most project categorical features are strings, but preserve numeric
        # category types when the model stored numeric categories.
        string_categories = all(isinstance(v, str) for v in categories)

        if string_categories:
            values = out[c].astype("string")
            valid = values.isin(categories)
            actual_missing = values.isna()

            unseen = (~valid) & (~actual_missing)
            unseen_per_row = unseen_per_row + unseen.astype("int64")

            if "__OTHER__" in categories:
                mapped = values.where(valid | actual_missing, "__OTHER__")
            else:
                mapped = values.where(valid, pd.NA)

            if "__MISSING__" in categories:
                mapped = mapped.fillna("__MISSING__")

            out[c] = pd.Categorical(
                mapped,
                categories=categories,
            )

        else:
            numeric_values = pd.to_numeric(out[c], errors="coerce")
            valid = numeric_values.isin(categories)
            actual_missing = numeric_values.isna()

            unseen = (~valid) & (~actual_missing)
            unseen_per_row = unseen_per_row + unseen.astype("int64")

            mapped = numeric_values.where(valid)
            out[c] = pd.Categorical(
                mapped,
                categories=categories,
            )

    return out, unseen_per_row


def recompute_a1_candidate_probabilities(
    artifacts: dict,
    trace_table: pd.DataFrame,
    *,
    top_features: Iterable[str] | None = None,
    event_meta: dict | None = None,
):
    """
    Re-run the exact frozen A1 XGBoost model for the candidates in one traced
    event and verify the saved candidate score.

    Important checkpoint detail:
    `chronological_shadow_v1/candidate_scores/a1_candidates.parquet` was
    intentionally saved WITHOUT `_candidate_id`. The original exact candidate
    model matrix still contains `_candidate_id`, `_policy_tx_id`, and
    `route_core`.

    Therefore, if `_candidate_id` is absent, this function reconstructs it
    deterministically from:

        event_id = "A1__<policy_tx_id>"
        candidate route = route_core

    This is an exact key-based recovery, not an approximate feature match.
    """
    import xgboost as xgb

    model_path = Path(artifacts["a1_model_json"])
    meta_path = Path(artifacts["a1_metadata"])
    matrix_path = Path(artifacts["a1_candidate_matrix"])

    for p in [model_path, meta_path, matrix_path]:
        if not p.exists():
            raise FileNotFoundError(p)

    trace = trace_table.copy()

    # ------------------------------------------------------------
    # Recover exact candidate IDs when the chronological-backtest
    # checkpoint does not contain them.
    # ------------------------------------------------------------
    if "_candidate_id" not in trace.columns:
        policy_tx_id = None

        if "_policy_tx_id" in trace.columns:
            vals = trace["_policy_tx_id"].dropna().unique()
            if len(vals) == 1:
                policy_tx_id = vals[0]

        if policy_tx_id is None and "event_id" in trace.columns:
            vals = trace["event_id"].dropna().astype(str).unique()
            if len(vals) == 1 and vals[0].startswith("A1__"):
                policy_tx_id = vals[0].split("A1__", 1)[1]

        if policy_tx_id is None and event_meta is not None:
            if "_policy_tx_id" in event_meta:
                policy_tx_id = event_meta["_policy_tx_id"]
            else:
                eid = str(event_meta.get("event_id", ""))
                if eid.startswith("A1__"):
                    policy_tx_id = eid.split("A1__", 1)[1]

        if policy_tx_id is None:
            raise KeyError(
                "Trace table does not contain _candidate_id and the A1 "
                "_policy_tx_id could not be recovered. Re-run the decision "
                "walkthrough cell after updating src/showcase.py, or pass "
                "event_meta=meta."
            )

        # Match against the exact original A1 candidate matrix.
        key_cols = ["_candidate_id", "_policy_tx_id", "route_core"]
        key_frame = pd.read_parquet(matrix_path, columns=key_cols)

        # Preserve numeric IDs when possible; fall back to string comparison.
        numeric_pid = pd.to_numeric(pd.Series([policy_tx_id]), errors="coerce").iloc[0]
        if pd.notna(numeric_pid):
            matches = key_frame[
                key_frame["_policy_tx_id"].eq(int(numeric_pid))
            ].copy()
        else:
            matches = key_frame[
                key_frame["_policy_tx_id"].astype(str).eq(str(policy_tx_id))
            ].copy()

        if matches.empty:
            raise KeyError(
                f"No original A1 candidate rows found for _policy_tx_id={policy_tx_id}"
            )

        route_key = matches[["_candidate_id", "route_core"]].copy()
        route_key["route"] = route_key["route_core"].astype(str)
        route_key = route_key.drop(columns="route_core")

        trace["route"] = trace["route"].astype(str)
        trace = trace.merge(
            route_key,
            on="route",
            how="left",
            validate="one_to_one",
        )

        if trace["_candidate_id"].isna().any():
            missing_routes = trace.loc[
                trace["_candidate_id"].isna(), "route"
            ].tolist()
            raise KeyError(
                "Could not recover exact candidate IDs for routes: "
                + ", ".join(map(str, missing_routes))
            )

        print(
            "✓ Recovered exact _candidate_id values from "
            "_policy_tx_id + route_core"
        )

    # ------------------------------------------------------------
    # Re-run exact frozen model.
    # ------------------------------------------------------------
    metadata = load_json(meta_path)
    feature_cols = list(metadata["feature_columns"])
    cat_cols = list(metadata["categorical_features"])
    num_cols = list(metadata["numeric_features"])

    ids = trace["_candidate_id"].astype("int64").tolist()
    Xraw = _read_candidate_rows(matrix_path, ids, feature_cols)

    # Preserve trace order.
    order = {int(v): i for i, v in enumerate(ids)}
    Xraw["__order"] = Xraw["_candidate_id"].astype("int64").map(order)
    Xraw = (
        Xraw.sort_values("__order")
        .drop(columns="__order")
        .reset_index(drop=True)
    )

    key = trace[["_candidate_id", "route", "raw_p_success"]].copy()
    key["_candidate_id"] = key["_candidate_id"].astype("int64")

    # Feature values can always be shown once the exact candidate row has
    # been recovered, even if a future XGBoost runtime cannot reproduce
    # native-categorical inference byte-for-byte.
    requested = [
        f for f in (top_features or [])
        if f in Xraw.columns
    ]

    feature_values = None
    if requested:
        feature_values = Xraw[
            ["_candidate_id"] + requested
        ].merge(
            key[["_candidate_id", "route"]],
            on="_candidate_id",
            how="left",
            validate="one_to_one",
        )
        feature_values = feature_values[
            ["_candidate_id", "route"] + requested
        ]

    verification = key.copy()
    verification["recomputed_p_success"] = np.nan
    verification["abs_difference"] = np.nan
    verification["unseen_category_fields_handled"] = 0
    verification["replay_status"] = "not_attempted"

    try:
        X = Xraw[feature_cols].copy()

        # Use the exact training vocabulary stored in the frozen booster.
        model_levels = load_model_category_levels(model_path)
        X, unseen_per_row = align_categories_to_frozen_model(
            X,
            cat_cols,
            model_levels,
        )

        for c in num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        p = model.predict_proba(X)[:, 1]

        pred_frame = Xraw[["_candidate_id"]].copy()
        pred_frame["_candidate_id"] = pred_frame["_candidate_id"].astype("int64")
        pred_frame["recomputed_p_success"] = p
        pred_frame["unseen_category_fields_handled"] = unseen_per_row.to_numpy()

        verification = key.merge(
            pred_frame,
            on="_candidate_id",
            how="left",
            validate="one_to_one",
        )
        verification["abs_difference"] = (
            verification["recomputed_p_success"]
            - verification["raw_p_success"]
        ).abs()
        verification["replay_status"] = "recomputed_from_frozen_model"

    except Exception as exc:
        # The saved model_score is the exact value used by the historical
        # policy backtest. A fresh replay is a reproducibility check, not a
        # prerequisite for the policy result itself.
        verification["replay_status"] = (
            "saved_backtest_score_used; fresh categorical replay unavailable: "
            + str(exc).splitlines()[0][:180]
        )

    verification = verification[
        [
            "_candidate_id",
            "route",
            "raw_p_success",
            "recomputed_p_success",
            "abs_difference",
            "unseen_category_fields_handled",
            "replay_status",
        ]
    ]

    return verification, feature_values, metadata

