from typing import Iterable
import logging
import os
import pickle

from numpy import mean, ndarray
import pandas as pd
from scipy.stats import spearmanr
from xgboost import XGBRanker, XGBRegressor, plot_importance
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

try:
    from lightgbm import LGBMRanker
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover - optional dependency
    HAS_LIGHTGBM = False

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


class CircuitSplit:
    def __init__(self, circuits: Iterable[str]):
        self.circuits = circuits

    def split(self, X, y=None, groups=None):
        for circ in self.circuits:
            mask_train = X['Circuit'] != circ
            mask_val = X['Circuit'] == circ
            yield (X[mask_train].index.values, X[mask_val].index.values)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.circuits)


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


def _train_model(features, target, cv, debug=False, use_regression=False):
    """Train a ranking/regression model using Bayesian optimisation.

    When ``use_regression`` is ``True`` the optimisation considers both
    ``XGBRanker`` and ``XGBRegressor`` objectives (and ``LGBMRanker`` if
    available). Trials are scored using a weighted combination of Spearman
    correlation and mean absolute error. Otherwise only ``XGBRanker`` is used
    and the score is the rank correlation alone.
    """
    max_mae = target.max() - target.min()
    if max_mae <= 0:
        max_mae = 1.0

    def objective(trial):
        params = {
            'n_estimators': 1000,
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.003, 0.3),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'max_leaves': trial.suggest_int('max_leaves', 0, 256),
            'grow_policy': trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide']),
        }

        model_choices = ['xgb_ranker']
        if use_regression:
            model_choices.append('xgb_regressor')
        if HAS_LIGHTGBM:
            model_choices.append('lgb_ranker')

        model_type = trial.suggest_categorical('model_type', model_choices)

        if model_type == 'lgb_ranker':
            params_lgb = {
                'num_leaves': trial.suggest_int('num_leaves', 15, 63),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 20),
            }
            model = LGBMRanker(objective='lambdarank', random_state=42,
                               **params_lgb, n_estimators=params['n_estimators'],
                               learning_rate=params['learning_rate'])
        elif model_type == 'xgb_regressor':
            model = XGBRegressor(objective='reg:squarederror', random_state=42, **params)
        else:
            model = XGBRanker(objective='rank:pairwise', random_state=42, **params)

        scores = []
        splits = cv.split(features) if hasattr(cv, 'split') else cv
        for train_idx, val_idx in splits:
            train_feat = features.iloc[train_idx].reset_index(drop=True)
            val_feat = features.iloc[val_idx].reset_index(drop=True)
            y_train = target.iloc[train_idx].reset_index(drop=True)
            y_val = target.iloc[val_idx].reset_index(drop=True)

            if model_type == 'xgb_ranker':
                train_groups = train_feat.groupby(['Season', 'RaceNumber'], sort=False).size().to_list()
                val_groups = val_feat.groupby(['Season', 'RaceNumber'], sort=False).size().to_list()
                _fit_xgb_ranker(
                    model,
                    train_feat,
                    y_train,
                    group=train_groups,
                    eval_set=[(val_feat, y_val)],
                    eval_group=[val_groups],
                    verbose=False,
                    early_stopping_rounds=20,
                )
            elif model_type == 'xgb_regressor':
                model.fit(
                    train_feat,
                    y_train,
                    eval_set=[(val_feat, y_val)],
                    early_stopping_rounds=20,
                    verbose=False,
                )
            else:  # lgb_ranker
                train_groups = train_feat.groupby(['Season', 'RaceNumber'], sort=False).size().to_list()
                val_groups = val_feat.groupby(['Season', 'RaceNumber'], sort=False).size().to_list()
                model.fit(
                    train_feat,
                    y_train,
                    group=train_groups,
                    eval_set=[(val_feat, y_val)],
                    eval_group=[val_groups],
                    callbacks=[lgb.early_stopping(20)],
                )

            preds = model.predict(val_feat)
            rho = spearmanr(y_val, preds).correlation
            mae = mean_absolute_error(y_val, preds)
            if pd.isna(rho):
                rho = -1.0
            if use_regression:
                score = 0.7 * rho + 0.3 * (1 - mae / max_mae)
            else:
                score = rho
            scores.append(score)

        score = mean(scores)
        if pd.isna(score):
            score = -1.0
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=120, show_progress_bar=False)
    os.makedirs("model_info", exist_ok=True)
    try:
        study.trials_dataframe().to_csv(
            "model_info/hyperparameter_trials.csv", index=False
        )
    except Exception as err:  # pragma: no cover - optional dependency issues
        logger.warning("Could not save trials dataframe: %s", err)
    try:
        importance = get_param_importances(study)
        pd.Series(importance).to_csv("model_info/param_importances.csv")
    except Exception as err:  # pragma: no cover - optional dependency issues
        logger.warning("Could not compute param importances: %s", err)

    best_params = study.best_params
    best_score = study.best_value
    best_model_type = best_params.pop('model_type', 'xgb_ranker')

    X_full_train, X_dev, y_full_train, y_dev = train_test_split(
        features, target, test_size=0.05, shuffle=False
    )
    group_full_train = build_group_list(X_full_train)
    group_dev = build_group_list(X_dev)

    if best_model_type == 'xgb_regressor':
        model = XGBRegressor(objective='reg:squarederror', random_state=42, **best_params)
        model.fit(
            X_full_train,
            y_full_train,
            eval_set=[(X_dev, y_dev)],
            early_stopping_rounds=20,
            verbose=False,
        )
        best_iter = getattr(model, 'best_iteration', getattr(model, 'best_ntree_limit', model.n_estimators))
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **best_params,
            n_estimators=best_iter,
        )
        model.fit(features, target, verbose=False)
    elif best_model_type == 'lgb_ranker':
        model = LGBMRanker(objective='lambdarank', random_state=42, **best_params)
        model.fit(
            X_full_train,
            y_full_train,
            group=group_full_train,
            eval_set=[(X_dev, y_dev)],
            eval_group=[group_dev],
            callbacks=[lgb.early_stopping(20)],
        )
        best_iter = getattr(model, 'best_iteration_', getattr(model, 'best_iteration', model.n_estimators))
        model = LGBMRanker(
            objective='lambdarank',
            random_state=42,
            **best_params,
            n_estimators=best_iter,
        )
        model.fit(features, target, group=build_group_list(features))
    else:
        model = XGBRanker(objective='rank:pairwise', random_state=42, **best_params)
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
        for attr in ('best_iteration', 'best_iteration_', 'best_ntree_limit'):
            if hasattr(model, attr):
                best_iter = getattr(model, attr)
                break
        if best_iter is None:
            best_iter = model.n_estimators

        model = XGBRanker(
            objective='rank:pairwise',
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


__all__ = ['_rank_metrics', '_train_model', 'SeasonSplit', 'CircuitSplit', 'build_group_list']
