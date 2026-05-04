from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ABD_prediction.preprocessing import make_preprocessor
from sampling import undersample_per_id
from ABD_prediction.splitting import group_stratified_train_test_split


param_grid = {
    "model__n_estimators": [200, 400, 800, 1000],
    "model__max_depth": [3, 5, 7, 10, 12],
    "model__learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.5, 0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "model__min_child_weight": [1, 3, 5, 7],
    "model__gamma": [0, 0.1, 0.5, 1.0],
    "model__reg_alpha": [0, 0.01, 0.1, 1.0],
    "model__reg_lambda": [0.5, 1.0, 2.0, 5.0],
    "model__scale_pos_weight": [1, 10, 20],
}


def make_xgb_pipeline(
    num_cols: list[str],
    cat_cols: list[str],
    random_state: int = 37,
    n_jobs: int = 1,
) -> Pipeline:
    """Create preprocessing + XGBoost classification pipeline."""
    preprocessor = make_preprocessor(num_cols=num_cols, cat_cols=cat_cols)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=max(1, n_jobs),
        verbosity=0,
    )

    return Pipeline(
        steps=[
            ("preproc", preprocessor),
            ("model", model),
        ]
    )


def choose_threshold_by_balanced_accuracy(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Choose threshold that maximises balanced accuracy.

    Returns
    -------
    chosen_threshold, best_balanced_accuracy
    """
    candidate_thresholds = np.unique(np.r_[0.0, y_proba, 1.0])

    chosen_threshold = 0.5
    best_balanced_accuracy = -np.inf

    for threshold in candidate_thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = balanced_accuracy_score(y_true, y_pred)

        if score > best_balanced_accuracy:
            best_balanced_accuracy = score
            chosen_threshold = float(threshold)

    return chosen_threshold, float(best_balanced_accuracy)


def binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Compute binary classification metrics safely."""
    y_true_arr = np.asarray(y_true)
    pos_count = int((y_true_arr == 1).sum())

    metrics = {
        f"{prefix}_balanced_accuracy_at_threshold": float(
            balanced_accuracy_score(y_true_arr, y_pred)
        ),
        f"{prefix}_recall_at_threshold": (
            float(((y_true_arr == 1) & (y_pred == 1)).sum() / pos_count)
            if pos_count > 0
            else 0.0
        ),
    }

    if len(np.unique(y_true_arr)) == 2:
        metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true_arr, y_proba))
        metrics[f"{prefix}_avg_precision"] = float(
            average_precision_score(y_true_arr, y_proba)
        )
    else:
        metrics[f"{prefix}_roc_auc"] = np.nan
        metrics[f"{prefix}_avg_precision"] = np.nan

    return metrics


def get_feature_importances(
    fitted_pipeline: Pipeline,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.Series:
    """Extract feature importances from a fitted preprocessing + XGB pipeline."""
    try:
        fitted_model = fitted_pipeline.named_steps["model"]
        importances = getattr(fitted_model, "feature_importances_", None)

        preprocessor = fitted_pipeline.named_steps["preproc"]

        try:
            feature_names = preprocessor.get_feature_names_out()
            feature_names = [name.split("__", 1)[-1] for name in feature_names]
        except Exception:
            feature_names = list(num_cols) + list(cat_cols)

        if importances is not None and len(importances) == len(feature_names):
            return pd.Series(
                importances,
                index=feature_names,
            ).sort_values(ascending=False)

    except Exception:
        pass

    return pd.Series(dtype=float)


def train_test_xgb(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    class_labels: list[str],
    outdir: str | Path,
    id_col = "ID",
    group_id_col = "group_ID",
    response_col: str = "response",
    test_size: float = 0.2,
    random_state: int = 37,
    n_iter: int = 500,
    cv_splits: int = 3,
    n_jobs: int = 1,
    verbose: int = 1,
    fig_width: float = 90 / 25.4,
    n_subsamples: int = 10,
    max_per_id: int = 200,
    replace: bool = True,
    param_dist: dict[str, list[Any]] | None = None,
) -> dict[str, Any]:
    """
    Train XGBoost using:
    - group-aware train/test split
    - group-aware CV hyperparameter search
    - out-of-fold threshold selection
    - optional undersampled ensemble on the test set
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feature_cols = list(num_cols) + list(cat_cols)

    train_df, test_df = group_stratified_train_test_split(
        df=df,
        test_size=test_size,
        random_state=random_state,
        response_col=response_col,
        group_id_col = group_id_col,
    )

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[response_col].astype(int).reset_index(drop=True)
    groups_train = (train_df[group_id_col]).astype(int).to_numpy()

    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[response_col].astype(int).reset_index(drop=True)

    pipeline = make_xgb_pipeline(
        num_cols=num_cols,
        cat_cols=cat_cols,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    gkf = GroupKFold(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist or param_grid,
        n_iter=n_iter,
        scoring="balanced_accuracy",
        cv=gkf,
        random_state=random_state,
        verbose=verbose,
        refit=True,
        n_jobs=n_jobs,
    )

    search.fit(X_train, y_train, groups=groups_train)

    best_pipeline = search.best_estimator_
    best_params = search.best_params_

    # Out-of-fold probabilities for threshold selection.
    oof_proba = cross_val_predict(
        best_pipeline,
        X_train,
        y_train,
        cv=gkf,
        groups=groups_train,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]

    chosen_threshold, _ = choose_threshold_by_balanced_accuracy(
        y_true=y_train,
        y_proba=oof_proba,
    )

    oof_pred = (oof_proba >= chosen_threshold).astype(int)
    cm_oof = confusion_matrix(y_train, oof_pred, labels=[0, 1])

    oof_metrics = binary_classification_metrics(
        y_true=y_train,
        y_proba=oof_proba,
        y_pred=oof_pred,
        prefix="oof",
    )

    # Test-set evaluation.
    test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= chosen_threshold).astype(int)
    cm_test = confusion_matrix(y_test, test_pred, labels=[0, 1])

    test_metrics = binary_classification_metrics(
        y_true=y_test,
        y_proba=test_proba,
        y_pred=test_pred,
        prefix="test",
    )

    # Ensemble using undersampled replicates.
    replicates = undersample_per_id(
        df=train_df,
        id_col=id_col,
        max_per_id=max_per_id,
        n_boot=n_subsamples,
        replace=replace,
        random_state=random_state,
    )

    boot_models = []
    boot_test_probas = []

    for rep in replicates:
        model = clone(best_pipeline)
        model.fit(rep[feature_cols], rep[response_col].astype(int))

        boot_models.append(model)
        boot_test_probas.append(model.predict_proba(X_test)[:, 1])

    if boot_test_probas:
        boot_test_probas_arr = np.vstack(boot_test_probas)
        boot_mean_proba = boot_test_probas_arr.mean(axis=0)
    else:
        boot_test_probas_arr = np.zeros((0, X_test.shape[0]))
        boot_mean_proba = np.zeros(X_test.shape[0])

    ensemble_pred = (boot_mean_proba >= chosen_threshold).astype(int)
    cm_ensemble = confusion_matrix(y_test, ensemble_pred, labels=[0, 1])

    ensemble_metrics = binary_classification_metrics(
        y_true=y_test,
        y_proba=boot_mean_proba,
        y_pred=ensemble_pred,
        prefix="ensemble_test",
    )

    feature_importances = get_feature_importances(
        fitted_pipeline=best_pipeline,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    bootstrap_info = {
        "n_boot": len(boot_models),
        "boot_models": boot_models,
        "boot_test_probas": boot_test_probas_arr,
        "boot_mean_proba": boot_mean_proba,
        "ensemble_pred": ensemble_pred,
    }

    return {
        "pipeline": best_pipeline,
        "best_params": best_params,
        "chosen_threshold": chosen_threshold,
        "train_df": train_df,
        "test_df": test_df,
        "oof_metrics": oof_metrics,
        "oof_confusion_matrix": cm_oof,
        "oof_proba": oof_proba,
        "oof_pred": oof_pred,
        "test_metrics": test_metrics,
        "test_confusion_matrix": cm_test,
        "test_proba": test_proba,
        "test_pred": test_pred,
        "feature_importances": feature_importances,
        "bootstrap_info": bootstrap_info,
        "ensemble_metrics": ensemble_metrics,
        "ensemble_test_proba": boot_mean_proba,
        "ensemble_test_pred": ensemble_pred,
    }