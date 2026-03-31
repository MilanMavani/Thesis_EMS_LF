from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


def get_dense_models(random_state: int = 42):
    models = {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=600,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="reg:squarederror",
            booster="gbtree",
        )

    if CatBoostRegressor is not None:
        models["catboost"] = CatBoostRegressor(
            iterations=600,
            learning_rate=0.01,
            depth=6,
            random_state=random_state,
            verbose=0,
        )

    return models


def get_nan_friendly_models(random_state: int = 42):
    models = {
        "hist_gbr": HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.01,
            max_depth=6,
            random_state=random_state,
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="reg:squarederror",
            booster="gbtree",
            missing=float("nan"),
        )

    if CatBoostRegressor is not None:
        models["catboost"] = CatBoostRegressor(
            iterations=600,
            learning_rate=0.01,
            depth=6,
            random_state=random_state,
            verbose=0,
        )

    return models