from __future__ import annotations

from typing import Dict, Iterable, Sequence
import math
import numpy as np
import pandas as pd

def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'amount' in out.columns:
        amt = pd.to_numeric(out['amount'], errors='coerce')
        out['amount_numeric'] = amt
        out['amount_log1p'] = np.log1p(amt.clip(lower=0))
        out['amount_exact'] = amt.round(2)
        for d in [10,50,100,500,1000]:
            out[f'amount_divisible_by_{d}'] = (np.isfinite(amt) & np.isclose(np.mod(amt,d),0.0)).astype(int)
    ts = pd.to_datetime(out['timestamp'], utc=True)
    hour = ts.dt.hour + ts.dt.minute/60.0 + ts.dt.second/3600.0
    dow = ts.dt.dayofweek
    out['hour_sin'] = np.sin(2*np.pi*hour/24.0)
    out['hour_cos'] = np.cos(2*np.pi*hour/24.0)
    out['dow_sin'] = np.sin(2*np.pi*dow/7.0)
    out['dow_cos'] = np.cos(2*np.pi*dow/7.0)
    out['is_weekend'] = (dow>=5).astype(int)
    return out

def add_cross_features(df: pd.DataFrame, cross_specs: Dict[str,Sequence[str]], sep='__') -> pd.DataFrame:
    out = df.copy()
    for name, cols in cross_specs.items():
        if all(c in out.columns for c in cols):
            out[name] = out[list(cols)].fillna('__MISSING__').astype(str).agg(sep.join, axis=1)
    return out

def smoothed_rate(success_weight, count_weight, prior_rate, prior_strength):
    return (success_weight + prior_strength*prior_rate)/(count_weight + prior_strength)

def _keys(frame: pd.DataFrame, group_cols: Sequence[str]):
    if not group_cols:
        return [('__GLOBAL__',)] * len(frame)
    vals = frame[list(group_cols)].fillna('__MISSING__').astype(str)
    return list(map(tuple, vals.itertuples(index=False, name=None)))

def add_expanding_binary_rate(df, group_cols, *, timestamp_col='timestamp', target_col='success', prefix, prior_rate, prior_strength=50.0):
    out = df.copy()
    rate = np.empty(len(out), dtype=float)
    logn = np.empty(len(out), dtype=float)
    temp = out.copy().reset_index(drop=True)
    temp['__pos'] = np.arange(len(temp))
    temp['__key'] = _keys(temp, group_cols)
    temp = temp.sort_values(['__key',timestamp_col,'__pos'])
    for key, g in temp.groupby('__key', sort=False):
        s = 0.0; n = 0.0
        for _, bucket in g.groupby(timestamp_col, sort=True):
            idx = bucket['__pos'].to_numpy()
            rate[idx] = smoothed_rate(s,n,prior_rate,prior_strength)
            logn[idx] = math.log1p(n)
            y = bucket[target_col].to_numpy(dtype=float)
            s += float(np.nansum(y)); n += float(np.isfinite(y).sum())
    out[f'{prefix}__rate__exp'] = rate
    out[f'{prefix}__logn__exp'] = logn
    return out

def add_decayed_binary_rate(df, group_cols, *, half_life_days, timestamp_col='timestamp', target_col='success', prefix, prior_rate, prior_strength=50.0):
    """
    Strictly lagged exponential-memory feature.

    Conceptually w(delta_t)=0.5**(delta_t/half_life), but implementation stores
    only weighted successes S, weighted count N and last timestamp per group.
    """
    out = df.copy()
    rate = np.empty(len(out), dtype=float)
    logn = np.empty(len(out), dtype=float)
    temp = out.copy().reset_index(drop=True)
    temp['__pos'] = np.arange(len(temp))
    temp['__key'] = _keys(temp, group_cols)
    temp = temp.sort_values(['__key',timestamp_col,'__pos'])
    one_day = 86400.0
    for key, g in temp.groupby('__key', sort=False):
        s = 0.0; n = 0.0; last_ts = None
        for ts, bucket in g.groupby(timestamp_col, sort=True):
            if last_ts is not None:
                delta_days = max((ts-last_ts).total_seconds()/one_day,0.0)
                d = 0.5 ** (delta_days/float(half_life_days))
                s *= d; n *= d
            idx = bucket['__pos'].to_numpy()
            rate[idx] = smoothed_rate(s,n,prior_rate,prior_strength)
            logn[idx] = math.log1p(n)
            y = bucket[target_col].to_numpy(dtype=float)
            s += float(np.nansum(y)); n += float(np.isfinite(y).sum())
            last_ts = ts
    tag = str(half_life_days).replace('.','p')
    out[f'{prefix}__rate__hl{tag}d'] = rate
    out[f'{prefix}__logn__hl{tag}d'] = logn
    return out

def build_multiscale_history(df, history_groups: Dict[str,Sequence[str]], *, half_lives_days=(3,14,60,180), timestamp_col='timestamp', target_col='success', prior_rate=0.08, prior_strength=50.0, include_expanding=True, verbose=True):
    out = df.copy()
    for name, cols in history_groups.items():
        if cols and not all(c in out.columns for c in cols):
            if verbose:
                print('Skipping', name, 'missing', [c for c in cols if c not in out.columns])
            continue
        if verbose:
            print('History group:', name, '->', list(cols))
        if include_expanding:
            out = add_expanding_binary_rate(out, cols, timestamp_col=timestamp_col, target_col=target_col, prefix=f'hist__{name}', prior_rate=prior_rate, prior_strength=prior_strength)
        for h in half_lives_days:
            out = add_decayed_binary_rate(out, cols, half_life_days=h, timestamp_col=timestamp_col, target_col=target_col, prefix=f'decay__{name}', prior_rate=prior_rate, prior_strength=prior_strength)
    return out

def leakage_audit(stage_feature_cols, stage: int):
    current_hits = [c for c in stage_feature_cols if c in {'response_code','response_description'}]
    illegal_prior = []
    if stage == 1:
        illegal_prior = [c for c in stage_feature_cols if c.startswith('a1_') or c.startswith('a2_')]
    elif stage == 2:
        illegal_prior = [c for c in stage_feature_cols if c.startswith('a2_')]
    return {
        'stage': stage,
        'current_response_leakage_hits': current_hits,
        'illegal_prior_state_hits': illegal_prior,
        'passed': not current_hits and not illegal_prior,
    }
