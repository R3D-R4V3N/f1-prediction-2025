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
            'n_estimators': trial.suggest_int('n_estimators', 300, 800),
            'max_depth': trial.suggest_categorical('max_depth', [3, 5, 7, 9]),
            'learning_rate': trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample': trial.suggest_categorical('subsample', [0.6, 0.8, 1.0]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.8, 1.0]),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5, 7, 10]),
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
    model.fit(features, target, group=full_group)
    if debug:
        plot_importance(model, max_num_features=10)
        plt.show()
    return model, best_score


__all__ = ['_rank_metrics', '_train_model']
