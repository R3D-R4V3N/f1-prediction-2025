import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker, plot_importance
import matplotlib.pyplot as plt
import optuna


def _rank_metrics(actual: pd.Series, preds: np.ndarray) -> dict:
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
            model.fit(
                train_feat,
                target.iloc[train_idx].reset_index(drop=True),
                group=train_groups,
                eval_set=[(val_feat, target.iloc[val_idx].reset_index(drop=True))],
                eval_group=[val_groups],
                verbose=False,
                early_stopping_rounds=20,
            )
            preds = model.predict(val_feat)
            rho = spearmanr(target.iloc[val_idx].reset_index(drop=True), preds).correlation
            scores.append(rho)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60, show_progress_bar=False)
    best_params = study.best_params
    best_score = study.best_value
    model = XGBRanker(objective="rank:pairwise", random_state=42, **best_params)
    full_group = features.groupby(["Season", "RaceNumber"], sort=False).size().to_list()
    model.fit(
        features,
        target,
        group=full_group,
        eval_set=[(features, target)],
        eval_group=[full_group],
        verbose=False,
        early_stopping_rounds=20,
    )
    if debug:
        plot_importance(model, max_num_features=10)
        plt.show()
    return model, best_score


__all__ = ['_rank_metrics', '_train_model']
