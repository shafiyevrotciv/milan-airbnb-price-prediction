from pathlib import Path
import joblib
import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping
from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from preprocess import (
    clean_target,
    engineer_features,
    fit_and_save_preprocessor,
    transform_data,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "listings.csv"
MODELS_DIR = BASE_DIR / "models"


def evaluate_regression(y_true, y_pred, model_name="Model"):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    print(f"\n--- {model_name} ---")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE: {mse:.4f}")

    if not np.any(y_true == 0):
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f"MAPE: {mape * 100:.2f}%")


def train_pipeline(n_trials: int = 50):
    print("Loading and preprocessing data")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(how="all", axis=1)
    df = clean_target(df, filter_outliers=True)
    df = engineer_features(df, is_train=True)

    X = df.drop(columns="price")
    y = df["price"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    X_train = fit_and_save_preprocessor(X_train_raw)
    X_test = transform_data(X_test_raw)

    # Split for optuna
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=0
    )

    # ---------------------------------------------------------
    # 2. Basic CatBoost
    # ---------------------------------------------------------
    print("\n2. Basic CatBoost fitting")
    cb = CatBoostRegressor(random_state=0, verbose=0)
    cb.fit(X_train, y_train)
    preds_cat = cb.predict(X_test)
    evaluate_regression(y_test, preds_cat, "CatBoost (Out of the box)")

    # ---------------------------------------------------------
    # 3. LightGBM Tuning using Optuna
    # ---------------------------------------------------------
    print(f"\n3. LGBM Parameters search")

    def objective_lgbm(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": 0,
            "n_estimators": 1000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_X = X_val,
            eval_y = y_val,
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study_lgbm = optuna.create_study(
        direction="minimize", study_name="lgbm_regression_study"
    )
    study_lgbm.optimize(objective_lgbm, n_trials=n_trials, show_progress_bar=True)

    best_params_lgbm = study_lgbm.best_params
    best_params_lgbm["random_state"] = 0
    best_params_lgbm["verbosity"] = -1
    best_params_lgbm["n_estimators"] = 2500

    lgbm_tuned = LGBMRegressor(**best_params_lgbm)
    lgbm_tuned.fit(X_train, y_train)
    preds_lgb = lgbm_tuned.predict(X_test)
    evaluate_regression(y_test, preds_lgb, "LightGBM (Tuned)")

    # ---------------------------------------------------------
    # 4. XGBoost Tuning using Optuna
    # ---------------------------------------------------------
    print(f"\n4. XGBoost Parameters search")

    def objective_xgboost(trial):
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": 0,
            "n_jobs": 4,
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        model = XGBRegressor(**params, early_stopping_rounds=50)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=0)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study_xgboost = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=10, seed=0),
        study_name="xgboost_regression_study",
    )
    study_xgboost.optimize(
        objective_xgboost, n_trials=n_trials, show_progress_bar=True, n_jobs=4
    )

    best_params_xgboost = study_xgboost.best_params
    best_params_xgboost["random_state"] = 0
    best_params_xgboost["verbosity"] = 0
    best_params_xgboost["n_estimators"] = 2500

    xgboost_tuned = XGBRegressor(**best_params_xgboost)
    xgboost_tuned.fit(X_train, y_train)
    preds_xgb = xgboost_tuned.predict(X_test)
    evaluate_regression(y_test, preds_xgb, "XGBoost (Tuned)")

    # ---------------------------------------------------------
    # 5. Final Ensemble (weights: 0.15 XGB + 0.15 LGB + 0.70 CB)
    # ---------------------------------------------------------
    weights = [0.15, 0.15, 0.7]
    final_weighted = (
        weights[0] * preds_xgb + weights[1] * preds_lgb + weights[2] * preds_cat
    )
    evaluate_regression(y_test, final_weighted, "Final Weighted Ensemble")

    # ---------------------------------------------------------
    # 6. Saving artifacts in  models/
    # ---------------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgboost_tuned, MODELS_DIR / "xgboost_tuned.joblib")
    joblib.dump(lgbm_tuned, MODELS_DIR / "lgbm_tuned.joblib")
    joblib.dump(cb, MODELS_DIR / "catboost_base.joblib")
    joblib.dump(weights, MODELS_DIR / "weights.joblib")

    print(f"\n All models are successfully saved into: {MODELS_DIR}")


if __name__ == "__main__":
    train_pipeline(n_trials=50)