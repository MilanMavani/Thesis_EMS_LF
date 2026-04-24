from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def get_profile_dense_models(random_state: int = 42) -> dict[str, object]:
    models = {
        "ridge": MultiOutputRegressor(
            Ridge(alpha=5.0)
        ),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=500,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.9,
                reg_alpha=0.2,
                reg_lambda=2.0,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
        )

    if CatBoostRegressor is not None:
        models["catboost"] = MultiOutputRegressor(
            CatBoostRegressor(
                iterations=1200,
                learning_rate=0.03,
                depth=8,
                l2_leaf_reg=5,
                loss_function="RMSE",
                random_seed=random_state,
                verbose=0,
            )
        )

    return models


def get_profile_nan_friendly_models(random_state: int = 42) -> dict[str, object]:
    models = {
        "hist_gbr": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=600,
                learning_rate=0.05,
                max_depth=6,
                random_state=random_state,
            )
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=1200,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.9,
                reg_alpha=0.2,
                reg_lambda=2.0,
                colsample_bytree=0.9,
                min_child_weight=3,
                gamma=0.1,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
        )

    return models