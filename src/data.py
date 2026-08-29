from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

RAW_COLUMN_CANDIDATES = {
    'timestamp': ['timestamp','Timestamp','CreatedAt','created_at','TransactionTime','TransactionTimestamp','Order_CreatedAt','PaymentProvider_CreatedAt'],
    'success': ['success','Success','is_success','approved','Approved','IsSuccess','PaymentSuccess'],
    'attempt': ['attempt','Attempt','attempt_number','AttemptNumber','attempt_index','RetryAttempt'],
    'transaction_id': ['transaction_id','TransactionId','transactionId','OrderId','Order_Id','Order_OrderId','TransactionID'],
    'route': ['route','route_core','route_clean','PaymentProvider_GroupId','Route','route_id'],
    'amount': ['amount','Amount','Order_Amount','Order_AmountValue','TransactionAmount'],
    'merchant_id': ['merchant_id','MerchantId','Order_MerchantId','merchant'],
    'issuer_name': ['issuer_name','IssuerName','BinCheck_IssuerName'],
    'card_network': ['card_network','CardNetwork','BinCheck_CardNetwork'],
    'card_type': ['card_type','CardType','BinCheck_CardType'],
    'card_level': ['card_level','CardLevel','BinCheck_CardLevel'],
    'bank_category': ['bank_category','BankCategory'],
    'mcc': ['mcc','MCC','PaymentProvider_MCC'],
    'processor': ['processor','Processor','PaymentProvider_Processor'],
    'provider': ['provider','Provider','PaymentProvider_Provider'],
    'sponsor_bank': ['sponsor_bank','SponsorBank','PaymentProvider_SponsorBank'],
    'response_code': ['response_code','ResponseCode','PaymentProvider_ResponseCode','ProviderResponseCode'],
    'response_description': ['response_description','ResponseDescription','PaymentProvider_ResponseDescription','ProviderResponseDescription'],
}

def mount_google_drive(mount_point='/content/drive'):
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError('Google Drive mounting is available only in Google Colab.') from exc
    drive.mount(mount_point)

def load_table(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {path}')
    suffix = path.suffix.lower()
    if suffix in {'.parquet','.pq'}:
        return pd.read_parquet(path)
    if suffix in {'.csv','.txt'}:
        return pd.read_csv(path)
    if suffix == '.feather':
        return pd.read_feather(path)
    raise ValueError(f'Unsupported data format: {suffix}')

def _find_first_existing(columns, candidates):
    lookup = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        hit = lookup.get(str(c).lower())
        if hit is not None:
            return hit
    return None

def infer_column_mapping(df: pd.DataFrame, overrides: Optional[Dict[str,str]]=None):
    overrides = overrides or {}
    mapping = {}
    for canonical, candidates in RAW_COLUMN_CANDIDATES.items():
        raw = overrides.get(canonical) or _find_first_existing(df.columns, candidates)
        if raw is not None:
            mapping[raw] = canonical
    return mapping

def standardize_schema(df: pd.DataFrame, overrides: Optional[Dict[str,str]]=None, verbose=True):
    out = df.copy()
    mapping = infer_column_mapping(out, overrides)
    out = out.rename(columns=mapping)
    if verbose:
        print('Detected schema mapping:')
        for raw, canonical in mapping.items():
            print(f'  {raw} -> {canonical}')

    required = ['timestamp','success','attempt','transaction_id']
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f'Missing canonical columns {missing}. Add COLUMN_OVERRIDES in the notebook.')

    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True, errors='coerce')
    if out['timestamp'].isna().any():
        raise ValueError('Some timestamps could not be parsed.')

    if out['success'].dtype == bool:
        out['success'] = out['success'].astype(int)
    elif out['success'].dtype == object:
        s = out['success'].astype(str).str.strip().str.lower()
        yes = {'1','true','yes','success','approved','approve'}
        no = {'0','false','no','failure','failed','declined','decline'}
        mapped = s.map(lambda x: 1 if x in yes else (0 if x in no else np.nan))
        if mapped.notna().all():
            out['success'] = mapped
    out['success'] = pd.to_numeric(out['success'], errors='coerce')
    if out['success'].isna().any():
        raise ValueError('Could not map all success values to 0/1.')
    out['success'] = out['success'].astype(int)

    out['attempt'] = pd.to_numeric(out['attempt'], errors='coerce').astype('Int64')
    if 'amount' in out.columns:
        out['amount'] = pd.to_numeric(out['amount'], errors='coerce')

    if 'route' not in out.columns:
        comps = [c for c in ['processor','provider','sponsor_bank'] if c in out.columns]
        if len(comps) < 2:
            raise ValueError('No route column detected and not enough route components to construct one.')
        out['route'] = out[comps].fillna('__MISSING__').astype(str).agg('|'.join, axis=1)

    return out.sort_values(['timestamp','transaction_id','attempt']).reset_index(drop=True)

def basic_audit(df: pd.DataFrame):
    rows = []
    for attempt, g in df.groupby('attempt', dropna=False):
        rows.append({
            'attempt': int(attempt) if pd.notna(attempt) else attempt,
            'rows': len(g),
            'success_rate': float(g['success'].mean()),
            'min_timestamp': g['timestamp'].min(),
            'max_timestamp': g['timestamp'].max(),
            'routes': int(g['route'].nunique()) if 'route' in g else np.nan,
        })
    return pd.DataFrame(rows).sort_values('attempt')

def split_attempts(df: pd.DataFrame):
    return {int(k): g.copy().reset_index(drop=True) for k,g in df.groupby('attempt') if pd.notna(k)}

def attach_prior_attempt_state(df: pd.DataFrame):
    out = df.sort_values(['transaction_id','attempt','timestamp']).copy()
    keep = ['transaction_id','attempt','timestamp','route']
    keep += [c for c in ['response_code','response_description','success'] if c in out.columns]
    hist = out[keep].copy()

    for prior_attempt in [1,2]:
        prior = hist[hist['attempt']==prior_attempt].copy()
        rename = {
            'timestamp': f'a{prior_attempt}_timestamp',
            'route': f'a{prior_attempt}_route',
            'success': f'a{prior_attempt}_success_prior',
            'response_code': f'a{prior_attempt}_response_code',
            'response_description': f'a{prior_attempt}_response_description',
        }
        prior = prior.drop(columns=['attempt']).rename(columns=rename).drop_duplicates('transaction_id', keep='last')
        out = out.merge(prior, on='transaction_id', how='left')
        tcol = f'a{prior_attempt}_timestamp'
        if tcol in out.columns:
            gap = (out['timestamp'] - out[tcol]).dt.total_seconds()
            out[f'gap_a{prior_attempt}_to_current_seconds'] = gap.where(gap >= 0)

    a1_cols = [c for c in out.columns if c.startswith('a1_') or c.startswith('gap_a1_')]
    out.loc[out['attempt'] <= 1, a1_cols] = np.nan
    a2_cols = [c for c in out.columns if c.startswith('a2_') or c.startswith('gap_a2_')]
    out.loc[out['attempt'] <= 2, a2_cols] = np.nan
    return out.sort_values(['timestamp','transaction_id','attempt']).reset_index(drop=True)
