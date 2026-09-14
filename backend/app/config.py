import os
from pathlib import Path
from typing import List

class Settings:
    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    # Data Directory Paths
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PHASE1_DIR: Path = PROJECT_ROOT / "outputs" / "phase1"
    PHASE4_DIR: Path = PROJECT_ROOT / "outputs" / "phase4"
    PHASE5_DIR: Path = PROJECT_ROOT / "outputs" / "phase5"
    PHASE6_DIR: Path = PROJECT_ROOT / "outputs" / "phase6"

    # Specific Data File Paths
    FEATURES_FILE: Path = DATA_RAW_DIR / "elliptic_txs_features.csv"
    CLASSES_FILE: Path = DATA_RAW_DIR / "elliptic_txs_classes.csv"
    EDGELIST_FILE: Path = DATA_RAW_DIR / "elliptic_txs_edgelist.csv"

    # Fallback / Tracked Output Artifact Paths
    EDGE_TIME_AUDIT_FILE: Path = PHASE1_DIR / "edge_time_audit.csv"
    DATASET_SUMMARY_FILE: Path = PHASE1_DIR / "dataset_summary.json"
    TEMPORAL_LABEL_FILE: Path = PHASE1_DIR / "temporal_label_distribution.csv"
    INVESTIGATION_ANALYSIS_FILE: Path = PHASE6_DIR / "investigation_priority_analysis.csv"

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

    # Server Host & Port Settings
    @property
    def HOST(self) -> str:
        return os.getenv("HOST", "0.0.0.0")

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "8001"))

    # CORS Origins (Parameterized via environment variables with local fallbacks)
    DEFAULT_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
        "*"
    ]

    @property
    def CORS_ORIGINS(self) -> List[str]:
        env_origins = os.getenv("CORS_ORIGINS") or os.getenv("ALLOWED_ORIGINS") or os.getenv("FRONTEND_URL")
        if env_origins:
            parsed = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
            return list(dict.fromkeys(self.DEFAULT_CORS_ORIGINS + parsed))
        return self.DEFAULT_CORS_ORIGINS

    # Raw CSV availability helpers
    @property
    def has_raw_dataset(self) -> bool:
        return self.FEATURES_FILE.exists() and self.CLASSES_FILE.exists() and self.EDGELIST_FILE.exists()

    @property
    def has_raw_features(self) -> bool:
        return self.FEATURES_FILE.exists()

    @property
    def has_raw_classes(self) -> bool:
        return self.CLASSES_FILE.exists()

    @property
    def has_raw_edgelist(self) -> bool:
        return self.EDGELIST_FILE.exists()

settings = Settings()
