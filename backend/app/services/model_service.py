import logging
from typing import Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from fastapi import HTTPException, status

from backend.app.config import settings
from backend.app.services.data_service import DataService, get_risk_level_and_prediction

logger = logging.getLogger("graphguard.model_service")

# Feature definition constants matching phase 5 training
ORIGINAL_FEATURES = [f"feature_{i}" for i in range(1, 166)]
NEIGHBORHOOD_FEATURES = [
    "model_risk",
    "incoming_mean_risk",
    "incoming_max_risk",
    "incoming_median_risk",
    "incoming_std_risk",
    "incoming_risk_neighbor_count",
    "outgoing_mean_risk",
    "outgoing_max_risk",
    "outgoing_median_risk",
    "outgoing_std_risk",
    "outgoing_risk_neighbor_count",
    "neighborhood_mean_risk",
    "neighborhood_max_risk",
    "neighborhood_median_risk",
    "neighborhood_neighbor_count",
    "incoming_high_risk_neighbors",
    "outgoing_high_risk_neighbors",
    "high_risk_neighbor_count",
    "high_risk_neighbor_fraction",
    "neighborhood_vs_self_risk",
]
MODEL_FEATURES = ORIGINAL_FEATURES + NEIGHBORHOOD_FEATURES

class ModelService:
    _instance = None

    def __init__(self):
        self.model = None
        self.is_loaded: bool = False

    @classmethod
    def get_instance(cls) -> "ModelService":
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def load_model(self):
        if self.is_loaded:
            return

        logger.info(f"Loading primary XGBoost model from {settings.PRIMARY_MODEL_FILE}")
        self.model = joblib.load(settings.PRIMARY_MODEL_FILE)
        self.is_loaded = True
        logger.info("Primary XGBoost model loaded successfully.")

    def predict_transaction(self, tx_id: int) -> Dict[str, Any]:
        ds = DataService.get_instance()
        if not ds.is_loaded:
            ds.load_data()

        # Check if transaction exists
        if tx_id not in ds.features_df.index:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction ID {tx_id} not found in dataset."
            )

        # Check if leakage-safe neighborhood features are present
        if tx_id not in ds.neighborhood_df.index:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Prediction unavailable because required leakage-safe neighborhood features are unavailable for transaction ID {tx_id}."
            )

        n_row = ds.neighborhood_df.loc[tx_id]
        if pd.isna(n_row["model_risk"]):
            time_step = int(ds.features_df.loc[tx_id, "time_step"])
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Prediction unavailable because required leakage-safe neighborhood features are unavailable for transaction ID {tx_id} (timestep {time_step})."
            )

        # Construct feature vector
        feat_row = ds.features_df.loc[tx_id, ORIGINAL_FEATURES]
        neigh_row = n_row[NEIGHBORHOOD_FEATURES]

        # Combine into single DataFrame for model prediction
        combined_df = pd.DataFrame([pd.concat([feat_row, neigh_row])], columns=MODEL_FEATURES)

        # Predict probability using loaded XGBoost model
        probs = self.model.predict_proba(combined_df)
        risk_score = float(probs[0][1])

        risk_level, prediction = get_risk_level_and_prediction(risk_score)

        return {
            "tx_id": tx_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "prediction": prediction
        }
