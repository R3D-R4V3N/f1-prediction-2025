from typing import Iterable
import logging
import os
import pickle

from numpy import mean, ndarray
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker, plot_importance
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def _fit_xgb_ranker(
    model,
    X,
    y,
    *,
    group=None,
    eval_set=None,
    eval_group=None,
    early_stopping_rounds=20,
    verbose=False,
):
    """Fit ``XGBRanker`` handling early stopping across versions."""

    try:
        model.fit(
            X,
            y,
            group=group,
            eval_set=eval_set,
            eval_group=eval_group,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        return model
    except TypeError:
        pass

    try:
        import xgboost as xgb

        model.fit(
            X,
            y,
            group=group,
            eval_set=eval_set,
            eval_group=eval_group,
            callbacks=[xgb.callback.EarlyStopping(rounds=early_stopping_rounds)],
            verbose=verbose,
        )
        return model
    except TypeError:
        pass

    model.fit(
        X,
        y,
        group=group,
        eval_set=eval_set,
        eval_group=eval_group,
        verbose=verbose,
    )
    return model


class SeasonSplit:
    def __init__(self, seasons: Iterable[int]):
        self.seasons = seasons

    def split(self, X, y=None, groups=None):
        for i in range(1, len(self.seasons)):
            train_mask = X['Season'].isin(self.seasons[:i])
            val_mask = X['Season'] == self.seasons[i]
            yield (train_mask[train_mask].index.values, val_mask[val_mask].index.values)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.seasons) - 1


def build_group_list(df: pd.DataFrame) -> list:
    return df.groupby(["Season", "RaceNumber"], sort=False).size().to_list()


def _rank_metrics(actual: pd.Series, preds: ndarray) -> dict:
    """Compute ranking-focused metrics."""
    actual_series = pd.Series(actual).reset_index(drop=True)
    pred_series = pd.Series(preds)
    rho = spearmanr(actual_series, pred_series).correlation
    pred_order = pred_series.rank(method="first").sort_values().index
    actual_order = actual_series.rank(method="first").sort_values().index
    top1 = float(pred_order[0] == actual_order[0]) if len(pred_order) else 0.0
    if len(pred_order) >= 3 and len(actual_order) >= 3:
        top3 = len(set(pred_order[:3]) & set(actual_order[:3])) / 3.0
    else:
        top3 = 0.0
    return {"spearman": rho, "top1": top1, "top3": top3}


def _train_model(features, target, cv, debug=False):
    """Train an XGBoost ranker using Bayesian optimisation."""
    def objective(trial):
        params = {
            # use a large number of estimators and rely on early stopping to find the best value
            'n_estimators': 1000,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 0, 10),
        }
        model = XGBRanker(objective="rank:pairwise", random_state=42, **params)
        scores = []
        splits = cv.split(features) if hasattr(cv, "split") else cv
        for train_idx, val_idx in splits:
            train_feat = features.iloc[train_idx].reset_index(drop=True)
            val_feat = features.iloc[val_idx].reset_index(drop=True)
            train_groups = (
                train_feat.groupby(["Season", "RaceNumber"], sort=False).size().to_list()
            )
            val_groups = (
                val_feat.groupby(["Season", "RaceNumber"], sort=False).size().to_list()
            )
            _fit_xgb_ranker(
                model,
                train_feat,
                target.iloc[train_idx].reset_index(drop=True),
                group=train_groups,
                eval_set=[(val_feat, target.iloc[val_idx].reset_index(drop=True))],
                eval_group=[val_groups],
                verbose=False,
                early_stopping_rounds=20,
            )
            preds = model.predict(val_feat)
            rho = spearmanr(
                target.iloc[val_idx].reset_index(drop=True), preds
            ).correlation
            if pd.isna(rho):
                rho = -1.0
            scores.append(rho)

        score = mean(scores)
        if pd.isna(score):
            score = -1.0
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    best_params = study.best_params
    best_score = study.best_value

    X_full_train, X_dev, y_full_train, y_dev = train_test_split(
        features, target, test_size=0.05, shuffle=False
    )
    group_full_train = build_group_list(X_full_train)
    group_dev = build_group_list(X_dev)

    model = XGBRanker(objective="rank:pairwise", random_state=42, **best_params)
    _fit_xgb_ranker(
        model,
        X_full_train,
        y_full_train,
        group=group_full_train,
        eval_set=[(X_dev, y_dev)],
        eval_group=[group_dev],
        early_stopping_rounds=20,
        verbose=False,
    )
    best_iter = None
    for attr in ("best_iteration", "best_iteration_", "best_ntree_limit"):
        if hasattr(model, attr):
            best_iter = getattr(model, attr)
            break
    if best_iter is None:
        best_iter = model.n_estimators

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=42,
        **best_params,
        n_estimators=best_iter,
    )
    model.fit(
        features,
        target,
        group=build_group_list(features),
        verbose=False,
    )

    if debug:
        plt.figure(figsize=(10, 8))
        plot_importance(model, max_num_features=20)
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        os.makedirs("model_info", exist_ok=True)
        plt.savefig("model_info/feature_importance.png")
        plt.close()

    return model, best_score


__all__ = ['_rank_metrics', '_train_model', 'SeasonSplit', 'build_group_list']
