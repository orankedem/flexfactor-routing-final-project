from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import pandas as pd


def first_existing(paths, label=None):
    for p in map(Path, paths):
        if p.exists():
            return p
    if label:
        raise FileNotFoundError(f"Could not locate {label}. Tried:\n" + "\n".join(map(str, paths)))
    return None


def discover_artifacts(drive_root: str | Path):
    root = Path(drive_root)
    candidates = {
        "data": [
            root / "FlexFactor_Final_Project/data/standardized_attempts.parquet",
            root / "flexfactor_route_model_checkpoints/standardized_attempts.parquet",
        ],
        "a1_model_selection": [
            root / "flexfactor_a1_model_selection/a1_model_selection_v1",
        ],
        "a1_model": [
            root / "flexfactor_a1_optuna/a1_xgboost_optuna_v1",
        ],
        "retry_models": [
            root / "flexfactor_retry_models/a2_a3_objective_xgb_v2",
        ],
        "backtest": [
            root / "flexfactor_backtest/chronological_shadow_v1",
        ],
        "feature_policy": [
            root / "flexfactor_policy_v2/success_vs_value_online_v2",
            root / "flexfactor_policy_v2/feature_importance_lp_online_v1",
        ],
        "final_policy": [
            root / "flexfactor_policy_v2/final_online_policy_selection_v3",
        ],
        "robustness": [
            root / "flexfactor_policy_v2/chronological_policy_robustness_v4",
        ],
    }
    found = {}
    for key, paths in candidates.items():
        found[key] = first_existing(paths)
    return found


def load_candidate_universe(backtest_root, month="2026-06"):
    backtest_root = Path(backtest_root)
    parts=[]
    for attempt in [1,2,3]:
        p=backtest_root/f"candidate_scores/a{attempt}_candidates.parquet"
        c=pd.read_parquet(p)
        c["timestamp"]=pd.to_datetime(c["timestamp"],utc=True)
        c=c[c["timestamp"].dt.strftime("%Y-%m").eq(month)].copy()
        c["attempt"]=attempt
        if "candidate_allowed" in c:
            c=c[c["candidate_allowed"].astype(bool)].copy()
        parts.append(c)
    cand=pd.concat(parts,ignore_index=True)
    cand["candidate_route"]=cand["candidate_route"].astype(str)
    cand["logged_route"]=cand["logged_route"].astype(str)
    cand["amount_numeric"]=pd.to_numeric(cand["amount_numeric"],errors="coerce").fillna(0.0)
    return cand


def load_event_baseline(candidates):
    logged=candidates[candidates["is_logged_route"].astype(bool)].copy()
    chk=logged.groupby("event_id").size()
    if not chk.eq(1).all():
        raise ValueError("Every event must have exactly one logged-route candidate")
    return (logged[["event_id","attempt","timestamp","logged_route","success","amount_numeric","model_score"]]
            .rename(columns={"model_score":"logged_model_score"})
            .sort_values(["timestamp","attempt","event_id"],kind="stable")
            .reset_index(drop=True))


def route_pressure(route, baseline_counts, policy_counts, shift=.30):
    b=baseline_counts[route]
    a=policy_counts[route]
    return (a-b)/max(shift*b,1.0)


def strict_bounds(v, shift=.30):
    return int(math.ceil((1-shift)*v-1e-12)), int(math.floor((1+shift)*v+1e-12))


def feasible_after_choice(route, baseline_counts, policy_counts, shift=.30):
    all_routes=set(baseline_counts)|set(policy_counts)|{route}
    for r in all_routes:
        b=baseline_counts[r]
        a=policy_counts[r]+(1 if r==route else 0)
        lo,hi=strict_bounds(b,shift)
        if a<lo or a>hi:
            return False
    return True


def detailed_pressure_trace(candidates, *, lambda_=0.10, target_event_id=None,
                            choose_interesting=True, shift=.30):
    """Replay the exact development-style chronological pressure decision.

    If target_event_id is omitted, return the first event where the pressure
    policy chooses a different route from the raw-probability top route.
    """
    baseline=load_event_baseline(candidates)
    lookup={e:g.copy() for e,g in candidates.groupby("event_id",sort=False)}
    baseline_counts=defaultdict(int)
    policy_counts=defaultdict(int)

    for _,ev in baseline.iterrows():
        eid=ev["event_id"]
        logged=str(ev["logged_route"])
        baseline_counts[logged]+=1
        g=lookup[eid].copy()
        rows=[]
        for _,r in g.iterrows():
            route=str(r["candidate_route"])
            p=float(r["model_score"])
            pr=route_pressure(route,baseline_counts,policy_counts,shift)
            score=p-lambda_*pr
            feasible=feasible_after_choice(route,baseline_counts,policy_counts,shift)
            rows.append({
                "route":route,
                "raw_p_success":p,
                "baseline_cumulative":baseline_counts[route],
                "policy_cumulative_before":policy_counts[route],
                "pressure":pr,
                "adjusted_score":score,
                "feasible":feasible,
                "is_logged_route":bool(r["is_logged_route"]),
            })
        table=pd.DataFrame(rows).sort_values(["adjusted_score","raw_p_success"],ascending=False)
        feasible_table=table[table["feasible"]]
        if feasible_table.empty:
            chosen=logged
        else:
            chosen=str(feasible_table.iloc[0]["route"])
        raw_best=str(table.sort_values("raw_p_success",ascending=False).iloc[0]["route"])

        interesting=(chosen!=raw_best)
        hit=(target_event_id is not None and eid==target_event_id) or (target_event_id is None and choose_interesting and interesting)
        if hit:
            meta={
                "event_id":eid,
                "attempt":int(ev["attempt"]),
                "timestamp":ev["timestamp"],
                "amount":float(ev["amount_numeric"]),
                "logged_route":logged,
                "raw_probability_best_route":raw_best,
                "pressure_policy_route":chosen,
                "lambda":lambda_,
            }
            table["selected"] = table["route"].eq(chosen)
            return meta, table.reset_index(drop=True)

        policy_counts[chosen]+=1

    raise ValueError("No matching/interesting event found")
