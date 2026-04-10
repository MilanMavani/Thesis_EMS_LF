from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def get_profile_models(random_state: int = 42) -> dict[str, object]:
  
    models = {
        "ridge": MultiOutputRegressor(
            Ridge(alpha=1.0)
        ),
        "random_forest": MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=1000,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }

    if XGBRegressor is not None:
        models["xgboost"] = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=random_state,
                n_jobs=-1,
            )
        )

    return models