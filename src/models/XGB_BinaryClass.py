# src/models/XGB_BinaryClass.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    fbeta_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score, brier_score_loss, log_loss,
    make_scorer, matthews_corrcoef
)
from typing import Dict, Any, Optional, Tuple
from xgboost import XGBClassifier

# Global configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2 

def train_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[Any]]:
    # 1. Define Target and Features
    TARGET_COLUMN = 'is_legitimate'                             #<------------Input
    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Target column '{TARGET_COLUMN}' not found.")
        return {}, None

    features = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    if TARGET_COLUMN in features:
        features.remove(TARGET_COLUMN)

    X = df[features]
    y = df[TARGET_COLUMN]

    # 2. Standardized Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Custom Scorer (F2 prioritized for recall-heavy tasks)
    f2_scorer = make_scorer(fbeta_score, beta=2.0)

    # 3. BASELINE CROSS-VALIDATION (Generalization Check)
    # We test the default model first to see how it generalizes naturally.
    print("-> Assessing baseline generalization (Default Params, 5-Fold CV)...")
    baseline_xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        tree_method='hist',
        n_jobs=-1
    )
    
    cv_results = cross_validate(
        baseline_xgb, X_train, y_train, 
        cv=5, 
        scoring={'f2': f2_scorer, 'roc_auc': 'roc_auc'},
        return_train_score=False
    )
    
    baseline_f2_mean = np.mean(cv_results['test_f2'])
    baseline_f2_std = np.std(cv_results['test_f2'])
    print(f"   - Baseline F2: {baseline_f2_mean:.4f} (+/- {baseline_f2_std:.4f})")

    # 4. TUNING AFTER GENERALIZATION CHECK
    print("-> Starting hyperparameter tuning (RandomizedSearchCV)...")
    pipeline = Pipeline([
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            tree_method='hist',
            n_jobs=-1
        ))
    ])

    param_dist = {
        'xgb__n_estimators': [100, 300, 500],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__gamma': [0, 1, 5],
        'xgb__min_child_weight': [1, 5, 10],
    }

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring=f2_scorer,                                      #<---- Tuning Target
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # 5. Final Evaluation on Hold-Out Test Set
    print("-> Evaluating tuned model on hold-out test set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Feature Importance extraction
    xgb_model = best_model.named_steps['xgb']
    sorted_importances = pd.Series(
        xgb_model.feature_importances_, index=X.columns
    ).sort_values(ascending=False).round(4).to_dict()

    metrics = {
        'baseline_cv_f2': baseline_f2_mean, # Added baseline for tracking
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f2_score': fbeta_score(y_test, y_pred, beta=2.0),
        'MCC': matthews_corrcoef(y_test, y_pred),
        'Native_Feature_Importance': sorted_importances,
        'TN, FP, FN, TP': [int(tn), int(fp), int(fn), int(tp)]
    }
    
    return metrics, best_model
