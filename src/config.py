from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
TRAINING_DATA_DIR = DATA_DIR / "03_Training"

IMPUTED_DATA_DIR = TRAINING_DATA_DIR / "Imputed_data"
NAN_DATA_DIR = TRAINING_DATA_DIR / "Data_with_NaNs"

IMPUTED_DATA_15MIN_DIR = TRAINING_DATA_DIR / "Imputed_data_15min"
NAN_DATA_15MIN_DIR = TRAINING_DATA_DIR / "Data_with_NaNs_15min"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
FORECAST_FIGURES_DIR = FIGURES_DIR / "forecast_diagnostics"
TABLES_DIR = OUTPUTS_DIR / "tables"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FEATURE_IMPORTANCE_DIR = OUTPUTS_DIR / "feature_importance"
LOGS_DIR = OUTPUTS_DIR / "logs"

for path in [
    OUTPUTS_DIR,
    FIGURES_DIR,
    FORECAST_FIGURES_DIR,
    TABLES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    FEATURE_IMPORTANCE_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# Global settings
RANDOM_STATE = 42

# Default targets
TARGET_COLUMNS = [
    "BU_TotActPwr_Academy",
    "BU_TotActPwr_Tech_Room",
    "BA_TotActPwr_BESS_AC_Panel1",
    "BA_TotActPwr_BESS_AC_Panel2",
]

# Default experiment log file
EXPERIMENT_LOG_FILE = LOGS_DIR / "experiment_log.csv"

# Profile forecasting output directories
PROFILE_OUTPUTS_DIR = OUTPUTS_DIR / "profile_forecasting"
PROFILE_PREDICTIONS_DIR = PROFILE_OUTPUTS_DIR / "predictions"
PROFILE_HORIZON_METRICS_DIR = PROFILE_OUTPUTS_DIR / "horizon_metrics"
PROFILE_MODELS_DIR = PROFILE_OUTPUTS_DIR / "models"
PROFILE_LOGS_DIR = PROFILE_OUTPUTS_DIR / "logs"
PROFILE_FIGURES_DIR = PROFILE_OUTPUTS_DIR / "figures"

for path in [
    PROFILE_OUTPUTS_DIR,
    PROFILE_PREDICTIONS_DIR,
    PROFILE_HORIZON_METRICS_DIR,
    PROFILE_MODELS_DIR,
    PROFILE_LOGS_DIR,
    PROFILE_FIGURES_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

PROFILE_EXPERIMENT_LOG_FILE = PROFILE_LOGS_DIR / "profile_experiment_log.csv"