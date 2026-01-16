# src/models/XGB_GroupKfold_Binary.py

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

    # 2. Group Split (Note: Using GroupKFold for continuous targets)
    gkf = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]

    # 3. Regression Pipeline
    pipeline = Pipeline([
        ('encoder', TargetEncoder(cols=high_card_cols, smoothing=10)),
        ('xgb', XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            random_state=RANDOM_STATE,
            tree_method='hist'
        ))
    ])

    # 4. Search (Optimizing for R-Squared or Negative MSE)
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=20,
        scoring='r2', cv=gkf, random_state=RANDOM_STATE, n_jobs=-1
    )

    search.fit(X_train, y_train, groups=groups_train)
    best_model = search.best_estimator_

    # 5. Evaluate
    y_pred = best_model.predict(X_test)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2_score': r2_score(y_test, y_pred)
    }
    return metrics, best_model