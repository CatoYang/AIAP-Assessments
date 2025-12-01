# src/models/Hist_boost.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    fbeta_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score, brier_score_loss, log_loss,
    matthews_corrcoef, make_scorer, 
)
from sklearn.inspection import permutation_importance
from typing import Dict, Any, Optional, Tuple

# Global configuration for reproducibility and consistency
RANDOM_STATE = 42
TEST_SIZE = 0.2 # 80% train, 20% test split


def train_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, Any], Optional[HistGradientBoostingClassifier]]:
    """
    Trains a HistGradientBoostingClassifier (Boosted Trees) using a standardized
    approach with hyperparameter tuning via RandomizedSearchCV and calculates 
    Permutation Feature Importance (PFI) on the test set.

    Parameters
    ----------
    df : pd.DataFrame
        The engineered DataFrame containing features and the target variable.

    Returns
    -------
    (metrics, trained_model)
        metrics (Dict): A dictionary of performance metrics and PFI results from the test set.
        trained_model (Optional[HistGradientBoostingClassifier]): The best fitted model object.
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

    X = df[features]
    y = df[TARGET_COLUMN]

    # 2. Standardized Split Data (seeded for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"-> Data split into Training ({len(X_train)}) and Testing ({len(X_test)}) sets.")

    # 3. Define Pipeline and Parameter Grid for CV
    pipeline = Pipeline([
        ('hgbc', HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    ])

    # Hyperparameter search space for HistGradientBoostingClassifier
    param_dist = {
        'hgbc__learning_rate': [0.01, 0.1, 0.2],
        'hgbc__max_depth': [3, 5, 7, 10],
        'hgbc__l2_regularization': [0.1, 1.0, 10.0],
        'hgbc__max_leaf_nodes': [31, 50, 100]
    }

    # Custom F2 scorer for optimization
    f2_scorer = make_scorer(fbeta_score, beta=2.0) 

    # 4. Hyperparameter Tuning using RandomizedSearchCV (CV on Training Data)
    print("-> Starting Randomized Search for best parameters (5-fold CV)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring=f2_scorer,
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"-> Best parameters found: {search.best_params_}")

    # 5. Predict and Evaluate on the Hold-Out Test Set
    print("-> Evaluating best model on the hold-out test set...")
    y_pred = best_model.predict(X_test)

    # HistGradientBoostingClassifier supports predict_proba for classification
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix components (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate F1 and F2 scores
    f1 = fbeta_score(y_test, y_pred, beta=1.0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)

    # 6. Calculate Permutation Feature Importance (PFI) on Test Set
    print("-> Calculating Permutation Feature Importance (PFI) on test set...")
    
    # We use 'roc_auc' as the stable probability metric for PFI, 
    # and set n_repeats to 10 for a robust estimate.
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

    # 7. Compile Metrics
    metrics = {
        # Standard Classification Metrics
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),

        # Other Metrics
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'specificity_tnr': specificity,
        'pr_auc_avg_precision': average_precision_score(y_test, y_proba),
        'f1_score': f1,
        'f2_score': f2,
        'brier_loss': brier_score_loss(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred),

        # Diagnostic Metrics (Feature Importance)
        'PFI_Top_Features': pfi_scores, # <<< NEW METRIC ADDED
        
        # Confusion Matrix (Flattened for easier JSON serialization)
        'TN, FP, FN, TP': [int(tn), int(fp), int(fn), int(tp)]
    }

    return metrics, best_model