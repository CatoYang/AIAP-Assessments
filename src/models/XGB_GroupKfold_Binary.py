import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    fbeta_score, confusion_matrix, average_precision_score,
    brier_score_loss, log_loss, make_scorer, matthews_corrcoef
)
from category_encoders import TargetEncoder
from typing import Dict, Any, Optional, Tuple, List
from xgboost import XGBClassifier

# Global configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2 

class TunedThresholdClassifier:
    """Wrapper to ensure the model uses the optimized threshold for .predict()"""
    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = threshold
        self.classes_ = getattr(estimator, 'classes_', [0, 1])

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

def optimize_threshold(y_true: np.ndarray, y_proba: np.ndarray, beta: float = 2.0) -> float:
    """Finds threshold that maximizes F-beta score."""
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = [fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, zero_division=0) for t in thresholds]
    return thresholds[np.argmax(scores)]

def train_evaluate(
    df: pd.DataFrame, 
    target_col: str = 'is_legitimate',                          #<----------- Inputs
    group_col: str = 'user_id',                                 #<----------- Inputs
    high_card_cols: List[str] = []
) -> Tuple[Dict[str, Any], Any]:
    
    # 1. Feature Selection
    numeric_bool = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    features = list(set(numeric_bool + high_card_cols))
    if target_col in features: features.remove(target_col)
    if group_col in features: features.remove(group_col)

    X, y, groups = df[features], df[target_col], df[group_col]

    # 2. Stratified Group Split (Final Test Set)
    sgkf = StratifiedGroupKFold(n_splits=int(1/TEST_SIZE), shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, y, groups))
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train_full = groups.iloc[train_idx]

    # 3. Create Internal Validation Split for Threshold Tuning
    # We split the training data again to find the threshold without touching X_test
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=RANDOM_STATE
    )
    # We need the groups for the subset training if we were doing further CV
    groups_train = groups_train_full.loc[X_train.index]

    # 4. Pipeline & Hyperparameter Search
    pipeline = Pipeline([
        ('encoder', TargetEncoder(cols=high_card_cols, smoothing=10)),
        ('xgb', XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1
        ))
    ])

    param_dist = {
        'xgb__n_estimators': [100, 300, 500],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__scale_pos_weight': [1, (y==0).sum()/(y==1).sum()]
    }

    print("-> Tuning Hyperparameters (ROC-AUC)...")
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=15, 
        scoring='roc_auc', cv=StratifiedGroupKFold(n_splits=3), 
        random_state=RANDOM_STATE, n_jobs=-1
    )

    search.fit(X_train, y_train, groups=groups_train)
    best_pipeline = search.best_estimator_

    # 5. Threshold Tuning (Generalization Step)
    print("-> Optimizing Threshold for F2 Score...")
    y_val_proba = best_pipeline.predict_proba(X_val)[:, 1]
    best_thresh = optimize_threshold(y_val, y_val_proba, beta=2.0)

    # 6. Final "Tuned" Model
    best_model = TunedThresholdClassifier(best_pipeline, threshold=best_thresh)

    # 7. Final Evaluation on Hold-out Test Set
    print("-> Final Evaluation on Test Set...")
    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'optimized_threshold': round(best_thresh, 4),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'avg_precision': average_precision_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'f2_score': fbeta_score(y_test, y_pred, beta=2.0),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'TN, FP, FN, TP': [int(tn), int(fp), int(fn), int(tp)]
    }

    return metrics, best_model