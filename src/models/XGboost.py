# src/models/xgboost.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    fbeta_score, confusion_matrix, average_precision_score,
    balanced_accuracy_score, brier_score_loss, log_loss,
    make_scorer,  matthews_corrcoef
)
from typing import Dict, Any, Optional, Tuple

from xgboost import XGBClassifier

# Global configuration for reproducibility and consistency
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% train, 20% test split


def train_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, float], Optional[Any]]:
    """
    Trains an XGBClassifier (XGBoost Gradient Boosted Trees) using a standardized
    approach with hyperparameter tuning via RandomizedSearchCV.

    The model is tuned on the training set (optimizing for F2 score)
    and evaluated on a final hold-out test set. Scaling is not required for this model type.

    Parameters
    ----------
    df : pd.DataFrame
        The engineered DataFrame containing features and the target variable.

    Returns
    -------
    (metrics, trained_model)
        metrics (Dict): A dictionary of performance metrics from the test set.
        trained_model (Optional[Any]): The best fitted model object (a Pipeline with XGBClassifier).
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
        ('xgb', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',   # avoid deprecated label encoder behaviour
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method='hist'       # fast histogram-based algorithm (if supported)
        ))
    ])

    # Hyperparameter search space for XGBClassifier
    param_dist = {
        'xgb__n_estimators': [100, 300, 500],
        'xgb__max_depth': [3, 5, 7, 10],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.6, 0.8, 1.0],
        'xgb__gamma': [0, 1, 5],
        'xgb__min_child_weight': [1, 5, 10],
        'xgb__reg_lambda': [1, 5, 10],
        'xgb__reg_alpha': [0, 1, 5],
    }

    # Custom F2 scorer for optimization
    f2_scorer = make_scorer(fbeta_score, beta=2.0)

    # 4. Hyperparameter Tuning using RandomizedSearchCV (CV on Training Data)
    print("-> Starting Randomized Search for best parameters (5-fold CV)...")
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,         # a bit larger since space is bigger than HGB
        scoring=f2_scorer, # optimize for F2
        cv=5,              # 5-fold cross-validation on the training set
        random_state=RANDOM_STATE,
        n_jobs=-1,         # Use all available cores
        verbose=1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    print(f"-> Best parameters found: {search.best_params_}")

    # 5. Predict and Evaluate on the Hold-Out Test Set
    print("-> Evaluating best model on the hold-out test set...")
    y_pred = best_model.predict(X_test)

    # XGBClassifier supports predict_proba for classification
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # Calculate confusion matrix components (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Calculate Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Calculate F1 and F2 scores
    f1 = fbeta_score(y_test, y_pred, beta=1.0)
    f2 = fbeta_score(y_test, y_pred, beta=2.0)

    # 6. Extract Native Feature Importance (XGBoost) <<< NEW STEP
    print("-> Extracting native feature importance...")
    
    # The XGBClassifier model is the first step in the pipeline (named 'xgb')
    xgb_model = best_model.named_steps['xgb']

    # Get the native feature importances
    native_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)

    # Sort by magnitude (importance) and format
    # XGBoost's feature importances are based on 'gain' by default
    sorted_importances = native_importances.sort_values(ascending=False).round(4).to_dict()

    metrics = {
        # Standard Classification Metrics
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),  # TPR
        'roc_auc': roc_auc_score(y_test, y_proba),

        # Other Metrics
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
        'specificity_tnr': specificity,
        'pr_auc_avg_precision': average_precision_score(y_test, y_proba),
        'f1_score': f1,
        'f2_score': f2,  # Optimization target
        'brier_loss': brier_score_loss(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        'MCC': matthews_corrcoef(y_test, y_pred),

        # Diagnostic Metrics (Feature Importance)
        'Native_Feature_Importance': sorted_importances,

        # Confusion Matrix (Flattened for easier JSON serialization)
        'TN, FP, FN, TP': [int(tn), int(fp), int(fn), int(tp)]
    }
    return metrics, best_model
