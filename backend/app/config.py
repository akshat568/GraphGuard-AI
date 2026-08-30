from pathlib import Path
from typing import List

class Settings:
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    # Data Directory Paths
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PHASE4_DIR: Path = PROJECT_ROOT / "outputs" / "phase4"
    PHASE5_DIR: Path = PROJECT_ROOT / "outputs" / "phase5"

    # Specific Data File Paths
    FEATURES_FILE: Path = DATA_RAW_DIR / "elliptic_txs_features.csv"
    CLASSES_FILE: Path = DATA_RAW_DIR / "elliptic_txs_classes.csv"
    EDGELIST_FILE: Path = DATA_RAW_DIR / "elliptic_txs_edgelist.csv"

    GRAPH_FEATURES_FILE: Path = PHASE4_DIR / "graph_features.csv"
    LEAKAGE_SAFE_NEIGHBORHOOD_FILE: Path = PHASE5_DIR / "leakage_safe_neighborhood_risk.csv"
    FEATURE_IMPORTANCE_FILE: Path = PHASE5_DIR / "neighborhood_xgboost_feature_importance.csv"
    METRICS_FILE: Path = PHASE5_DIR / "neighborhood_xgboost_metrics.json"
    THRESHOLD_SUMMARY_FILE: Path = PHASE5_DIR / "threshold_analysis_summary.json"
    PRIMARY_MODEL_FILE: Path = PHASE5_DIR / "neighborhood_xgboost.joblib"

    # Risk Boundaries & Production Thresholds
    PRODUCTION_ALERT_THRESHOLD: float = 0.90
    HIGH_RISK_THRESHOLD: float = 0.50
    MEDIUM_RISK_THRESHOLD: float = 0.25

    # CORS Origins
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"
    ]

settings = Settings()
