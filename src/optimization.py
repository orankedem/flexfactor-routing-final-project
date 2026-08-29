from __future__ import annotations

from collections import Counter
from typing import Dict
import math
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, csr_matrix

def capacity_bounds_from_baseline(baseline_counts: Dict[str,int], shift=.30):
    lower={r:int(math.ceil((1-shift)*int(v))) for r,v in baseline_counts.items()}
    upper={r:int(math.floor((1+shift)*int(v))) for r,v in baseline_counts.items()}
    return lower,upper

def solve_assignment_lp(candidates: pd.DataFrame, baseline_counts: Dict[str,int], *, event_col='event_id', route_col='route', probability_col='p_success', amount_col='amount', objective='success', capacity_shift=.30):
    """Full-hindsight LP oracle; not an online-deployable policy."""
    z=candidates.reset_index(drop=True).copy()
    events=list(pd.unique(z[event_col])); routes=list(pd.unique(z[route_col])); n=len(z)
    if objective=='success': value=z[probability_col].astype(float).to_numpy()
    elif objective in {'approved_value','value'}: value=(z[probability_col].astype(float)*z[amount_col].astype(float)).to_numpy()
    else: raise ValueError("objective must be 'success' or 'approved_value'")
    c=-value
    eidx={e:i for i,e in enumerate(events)}; ridx={r:i for i,r in enumerate(routes)}
    Aeq=lil_matrix((len(events),n),dtype=float)
    for j,e in enumerate(z[event_col]): Aeq[eidx[e],j]=1.0
    beq=np.ones(len(events))
    lower,upper=capacity_bounds_from_baseline(baseline_counts,capacity_shift)
    Aub=lil_matrix((2*len(routes),n),dtype=float); bub=np.zeros(2*len(routes))
    for j,r in enumerate(z[route_col]):
        k=ridx[r]; Aub[k,j]=1.0; Aub[len(routes)+k,j]=-1.0
    for r,k in ridx.items():
        bub[k]=upper.get(r,0); bub[len(routes)+k]=-lower.get(r,0)
    res=linprog(c,A_ub=csr_matrix(Aub),b_ub=bub,A_eq=csr_matrix(Aeq),b_eq=beq,bounds=(0,1),method='highs')
    if not res.success: raise RuntimeError('LP failed: '+res.message)
    z['x']=res.x
    chosen=z.sort_values([event_col,'x'],ascending=[True,False]).drop_duplicates(event_col).copy()
    chosen['objective_value']=chosen[probability_col].astype(float) if objective=='success' else chosen[probability_col].astype(float)*chosen[amount_col].astype(float)
    return {'assignments':chosen,'objective':objective,'objective_sum':float(chosen['objective_value'].sum()),'solver_objective':float(-res.fun),'lower_bounds':lower,'upper_bounds':upper,'raw_result':res}

def pressure(policy_count:int, baseline_count:int, shift=.30):
    return (float(policy_count)-float(baseline_count))/max(shift*float(baseline_count),1.0)

def _score(prob,amount,pr,lam,objective):
    if objective=='success': return prob-lam*pr
    if objective in {'approved_value','value'}: return amount*prob-lam*amount*pr
    raise ValueError("objective must be 'success' or 'approved_value'")

def simulate_pressure_policy(candidates: pd.DataFrame, *, lambda_:float, objective='success', capacity_shift=.30, event_col='event_id', timestamp_col='timestamp', route_col='route', probability_col='p_success', amount_col='amount', logged_route_col='logged_route'):
    """Chronological online replay with an immediate upper-capacity guard."""
    z=candidates.copy(); z[timestamp_col]=pd.to_datetime(z[timestamp_col],utc=True)
    order=z[[event_col,timestamp_col,logged_route_col]].drop_duplicates(event_col).sort_values([timestamp_col,event_col])
    A=Counter(); B=Counter(); chosen=[]
    for _,erow in order.iterrows():
        event=erow[event_col]; logged=erow[logged_route_col]; B[logged]+=1
        ec=z[z[event_col]==event]
        feasible=[]
        for _,row in ec.iterrows():
            r=row[route_col]; b=B[r]; a=A[r]
            upper=math.floor((1+capacity_shift)*b) if b>0 else 0
            if b==0 and r!=logged: continue
            if a+1>upper: continue
            pr=pressure(a,b,capacity_shift); prob=float(row[probability_col]); amount=float(row.get(amount_col,1.0))
            rr=row.copy(); rr['pressure']=pr; rr['policy_score']=_score(prob,amount,pr,lambda_,objective); rr['fallback']=False
            feasible.append(rr)
        if feasible:
            pick=max(feasible,key=lambda r:r['policy_score'])
        else:
            fb=ec[ec[route_col]==logged]
            if fb.empty: fb=ec.iloc[[0]]
            pick=fb.iloc[0].copy(); r=pick[route_col]; pr=pressure(A[r],B[r],capacity_shift); amount=float(pick.get(amount_col,1.0))
            pick['pressure']=pr; pick['policy_score']=_score(float(pick[probability_col]),amount,pr,lambda_,objective); pick['fallback']=True
        A[pick[route_col]]+=1; chosen.append(pick)
    out=pd.DataFrame(chosen)
    out['expected_approval']=out[probability_col].astype(float)
    out['expected_approved_value']=out[probability_col].astype(float)*out[amount_col].astype(float) if amount_col in out else np.nan
    return {'assignments':out,'policy_counts':dict(A),'baseline_counts':dict(B),'expected_approvals':float(out['expected_approval'].sum()),'expected_approved_value':float(out['expected_approved_value'].sum())}

def simulate_greedy_policy(candidates: pd.DataFrame, **kwargs):
    return simulate_pressure_policy(candidates,lambda_=0.0,objective='success',**kwargs)

def capacity_audit(policy_counts:Dict[str,int], baseline_counts:Dict[str,int], shift=.30):
    lower,upper=capacity_bounds_from_baseline(baseline_counts,shift); routes=sorted(set(policy_counts)|set(baseline_counts)); rows=[]
    for r in routes:
        b=int(baseline_counts.get(r,0)); a=int(policy_counts.get(r,0)); dev=(a/b-1) if b else np.nan
        rows.append({'route':r,'baseline':b,'policy':a,'lower':lower.get(r,0),'upper':upper.get(r,0),'deviation':dev,'abs_deviation':abs(dev) if pd.notna(dev) else np.nan,'near_27pct_boundary':bool(pd.notna(dev) and abs(dev)>=.27),'violation':bool(a<lower.get(r,0) or a>upper.get(r,0))})
    return pd.DataFrame(rows)
