# src/models/SVC.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    fbeta_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score, brier_score_loss, log_loss, make_scorer, matthews_corrcoef
)
from sklearn.inspection import permutation_importance
from typing import Dict, Any, Optional, Tuple

# Global configuration for reproducibility and consistency
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% train, 20% test split

def train_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[SVC]]:
    """
    Trains a Support Vector Classifier (SVC) using a standardized pipeline
    that includes scaling and hyperparameter tuning via RandomizedSearchCV.

    The model is tuned on the training set (optimizing for F2 score)
    and evaluated on a final hold-out test set. Scaling is crucial for SVC.
    """

    # 1. Define Target and Features
    TARGET_COLUMN = 'is_legitimate'
    if TARGET_COLUMN not in df.columns:
        print(f"ERROR: Target column '{TARGET_COLUMN}' not found in DataFrame.")
        return {}, None

    # Select only numeric/boolean features
    features = df.select_dtypes(include=['number', 'bool']).columns.tolist()
    if TARGET_COLUMN in features:
        features.remove(TARGET_COLUMN)

    if not features:
        print("ERROR: No numeric/bool features found for modelling.")
        return {}, None

    X = df[features]
    y = df[TARGET_COLUMN]

    # 2. Standardized Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"-> Data split into Training ({len(X_train)}) and Testing ({len(X_test)}) sets.")

    # (Optional but recommended) Subsample training data for CV to keep SVC practical
    # SVC with non-linear kernels does NOT scale well with n_samples.
    MAX_TRAIN_FOR_CV = 3000  # tweak if you like
    if len(X_train) > MAX_TRAIN_FOR_CV:
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(
            n_splits=1,
            train_size=MAX_TRAIN_FOR_CV,
            random_state=RANDOM_STATE
        )
        idx_small, _ = next(sss.split(X_train, y_train))
        X_cv = X_train.iloc[idx_small]
        y_cv = y_train.iloc[idx_small]
        print(f"-> Using a subsample of {len(X_cv)} rows for SVC hyperparameter search.")
    else:
        X_cv, y_cv = X_train, y_train
        print("-> Using full training set for SVC hyperparameter search.")

    # 3. Define Pipeline and Parameter Grid for CV
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        # linear SVM; probability=True will still be fitted later
        ('svc', SVC(
            kernel='linear',
            probability=True,       # needed for ROC AUC / log_loss / Brier
            random_state=RANDOM_STATE
        ))
    ])

    # Much smaller search space – only C for linear kernel
    param_dist = {
        'svc__C': [0.1, 1, 10, 100],
        # if you want to experiment further without going crazy:
        # 'svc__class_weight': [None, 'balanced']
    }

    # Custom F2 scorer
    f2_scorer = make_scorer(fbeta_score, beta=2.0)

    # 4. Hyperparameter Tuning using RandomizedSearchCV (CV on Training Data)
    print("-> Fast Randomized Search (3-fold CV) on the SVC...")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=4,              # small but sufficient
        scoring=f2_scorer,
        cv=3,                  # reduce CV folds
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_cv, y_cv)

    print(f"-> Best parameters found (on CV subset): {search.best_params_}")

    # 5. Refit best model on FULL training data
    print("-> Re-fitting best SVC on the full training data...")
    best_model: Pipeline = search.best_estimator_
    best_model.fit(X_train, y_train)

    # 6. Predict and Evaluate on the Hold-Out Test Set
    print("-> Evaluating best model on the hold-out test set...")
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    f1 = fbeta_score(y_test, y_pred, beta=1.0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)

    # 7. Calculate Permutation Feature Importance (PFI) on Test Set <<< NEW STEP
    print("-> Calculating Permutation Feature Importance (PFI) on test set...")
    
    # PFI measures the drop in ROC AUC when a feature is randomly shuffled.
    r = permutation_importance(
        best_model, 
        X_test, 
        y_test, 
        scoring='roc_auc', 
        n_repeats=10, 
        random_state=RANDOM_STATE, 
        n_jobs=-1
    )

    pfi_scores = {}

    # Iterate through features sorted by mean importance score (highest first)
    for i in r.importances_mean.argsort()[::-1]:
        feature_name = X.columns[i]
        mean_score = r.importances_mean[i]
        
        # Only include features with a positive mean importance score
        if mean_score > 0:
            pfi_scores[feature_name] = round(mean_score, 4)

    metrics: Dict[str, Any] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),

        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'specificity_tnr': specificity,
        'pr_auc_avg_precision': average_precision_score(y_test, y_proba),
        'f1_score': f1,
        'f2_score': f2,
        'brier_loss': brier_score_loss(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred),

        # Diagnostic Metrics (Feature Importance)
        'PFI_Top_Features': pfi_scores,
        
        'TN, FP, FN, TP': [int(tn), int(fp), int(fn), int(tp)]
    }

    return metrics, best_model