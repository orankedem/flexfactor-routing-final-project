from __future__ import annotations

from typing import Optional, Sequence
import math
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from xgboost import XGBClassifier

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true,dtype=float); y_prob = np.asarray(y_prob,dtype=float)
    edges = np.linspace(0,1,n_bins+1); bins = np.digitize(y_prob, edges[1:-1])
    ece = 0.0; n = len(y_true)
    for b in range(n_bins):
        m = bins==b
        if m.any():
            ece += (m.sum()/n)*abs(y_true[m].mean()-y_prob[m].mean())
    return float(ece)

def binary_entropy_logloss(y_true):
    p = float(np.mean(y_true)); eps=1e-15; p=min(max(p,eps),1-eps)
    return float(-(p*math.log(p)+(1-p)*math.log(1-p)))

def evaluate_probabilities(y_true,y_prob,n_bins=10):
    y_true=np.asarray(y_true,dtype=int); y_prob=np.clip(np.asarray(y_prob,dtype=float),1e-12,1-1e-12)
    ll=float(log_loss(y_true,y_prob)); ent=binary_entropy_logloss(y_true)
    return {
        'rows':len(y_true),'success_rate':float(y_true.mean()),
        'roc_auc':float(roc_auc_score(y_true,y_prob)),
        'average_precision':float(average_precision_score(y_true,y_prob)),
        'logloss':ll,'entropy_logloss':ent,'normalized_logloss':ll/ent,
        'logloss_skill':1-ll/ent,'brier':float(brier_score_loss(y_true,y_prob)),
        'ece_10':expected_calibration_error(y_true,y_prob,n_bins),
        'predicted_mean':float(y_prob.mean()),
    }

def _infer_types(df, feature_cols):
    cat=[]; num=[]
    for c in feature_cols:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype,pd.CategoricalDtype): cat.append(c)
        else: num.append(c)
    return cat,num

def make_tabular_preprocessor(df, feature_cols):
    cat,num=_infer_types(df,feature_cols)
    pre=ColumnTransformer([
        ('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1,encoded_missing_value=-1),cat),
        ('num','passthrough',num)
    ], remainder='drop', verbose_feature_names_out=False)
    return pre,cat,num

def _bundle(pipe, feature_cols, cat, num, family):
    return {'pipeline':pipe,'feature_cols':list(feature_cols),'categorical_cols':cat,'numeric_cols':num,'family':family}

def train_xgboost(train_df, feature_cols: Sequence[str], *, target_col='success', random_state=42, params: Optional[dict]=None):
    pre,cat,num=make_tabular_preprocessor(train_df,feature_cols)
    cfg=dict(n_estimators=500,learning_rate=.05,max_depth=6,min_child_weight=5,subsample=.85,colsample_bytree=.85,reg_lambda=2.0,objective='binary:logistic',eval_metric='logloss',random_state=random_state,tree_method='hist',n_jobs=-1)
    if params: cfg.update(params)
    pipe=Pipeline([('preprocess',pre),('model',XGBClassifier(**cfg))])
    pipe.fit(train_df[list(feature_cols)],train_df[target_col].astype(int))
    return _bundle(pipe,feature_cols,cat,num,'xgboost')

def train_lightgbm(train_df, feature_cols, *, target_col='success', random_state=42, params=None):
    from lightgbm import LGBMClassifier
    pre,cat,num=make_tabular_preprocessor(train_df,feature_cols)
    cfg=dict(n_estimators=500,learning_rate=.05,num_leaves=31,subsample=.85,colsample_bytree=.85,random_state=random_state,n_jobs=-1)
    if params: cfg.update(params)
    pipe=Pipeline([('preprocess',pre),('model',LGBMClassifier(**cfg))])
    pipe.fit(train_df[list(feature_cols)],train_df[target_col].astype(int))
    return _bundle(pipe,feature_cols,cat,num,'lightgbm')

def train_catboost(train_df, feature_cols, *, target_col='success', random_state=42, params=None):
    from catboost import CatBoostClassifier
    pre,cat,num=make_tabular_preprocessor(train_df,feature_cols)
    cfg=dict(iterations=500,learning_rate=.05,depth=7,loss_function='Logloss',verbose=False,random_seed=random_state)
    if params: cfg.update(params)
    pipe=Pipeline([('preprocess',pre),('model',CatBoostClassifier(**cfg))])
    pipe.fit(train_df[list(feature_cols)],train_df[target_col].astype(int))
    return _bundle(pipe,feature_cols,cat,num,'catboost')

def predict_proba(bundle,df):
    return bundle['pipeline'].predict_proba(df[bundle['feature_cols']])[:,1]

def compare_model_families(train_df,valid_df,feature_cols,*,target_col='success',families=('lightgbm','catboost','xgboost'),random_state=42):
    trainers={'lightgbm':train_lightgbm,'catboost':train_catboost,'xgboost':train_xgboost}
    rows=[]; bundles={}
    for family in families:
        b=trainers[family](train_df,feature_cols,target_col=target_col,random_state=random_state)
        p=predict_proba(b,valid_df); m=evaluate_probabilities(valid_df[target_col],p); m['family']=family
        rows.append(m); bundles[family]=b
    return pd.DataFrame(rows).sort_values('logloss'),bundles
