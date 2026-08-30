import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from fastapi import HTTPException, status

from backend.app.services.data_service import DataService

logger = logging.getLogger("graphguard.investigation_service")

def compute_priority_and_explanation(
    tx_id: int,
    time_step: int,
    model_risk: Optional[float],
    high_risk_fraction: Optional[float],
    neighborhood_mean_risk: Optional[float],
    neighborhood_vs_self_risk: Optional[float],
    total_degree: int,
    degree_imbalance: int
) -> Tuple[float, str, str, Dict[str, Any]]:
    if model_risk is None or pd.isna(model_risk):
        explanation = (
            f"Transaction {tx_id} (timestep {time_step}) is classified as UNASSESSED priority. "
            f"Precomputed leakage-safe neighborhood risk features are unavailable for timesteps 1-10."
        )
        return 0.0, "UNASSESSED", explanation, {}

    m_risk = float(model_risk)
    n_risk = float(high_risk_fraction) if high_risk_fraction is not None and not pd.isna(high_risk_fraction) \
        else float(neighborhood_mean_risk) if neighborhood_mean_risk is not None and not pd.isna(neighborhood_mean_risk) else 0.0

    raw_contrast = float(neighborhood_vs_self_risk) if neighborhood_vs_self_risk is not None and not pd.isna(neighborhood_vs_self_risk) else 0.0
    contrast_signal = min(1.0, max(0.0, (raw_contrast + 1.0) / 2.0))
    graph_signal = min(1.0, (float(total_degree) + abs(float(degree_imbalance))) / 20.0)

    # Composite Investigation Score
    score = (0.50 * m_risk) + (0.25 * n_risk) + (0.15 * contrast_signal) + (0.10 * graph_signal)
    score = round(min(1.0, max(0.0, score)), 4)

    # Priority Rules
    if m_risk >= 0.90 or (score >= 0.80 and m_risk >= 0.50):
        priority = "IMMEDIATE"
    elif (0.50 <= m_risk < 0.90) or (score >= 0.60 and m_risk >= 0.25):
        priority = "HIGH"
    elif (0.25 <= m_risk < 0.50) or (score >= 0.45):
        priority = "REVIEW"
    else:
        priority = "LOW"

    factors = {
        "model_risk_weight": 0.50,
        "model_risk_contribution": round(0.50 * m_risk, 4),
        "neighborhood_risk_weight": 0.25,
        "neighborhood_risk_contribution": round(0.25 * n_risk, 4),
        "contrast_weight": 0.15,
        "contrast_contribution": round(0.15 * contrast_signal, 4),
        "graph_activity_weight": 0.10,
        "graph_activity_contribution": round(0.10 * graph_signal, 4)
    }

    # Investigator explanation
    explanation_parts = [
        f"Transaction {tx_id} (timestep {time_step}) has been assigned an Investigation Priority of {priority} with a composite score of {score:.4f}."
    ]
    if priority == "IMMEDIATE":
        explanation_parts.append("High model risk score (≥ 0.90) or critical composite signals indicate urgent investigator triage.")
    elif priority == "HIGH":
        explanation_parts.append("Elevated model risk (0.50–0.89) or significant neighborhood propagation risk warrants prompt manual review.")
    elif priority == "REVIEW":
        explanation_parts.append("Moderate risk score or graph activity contrast suggests routine analyst inspection.")
    else:
        explanation_parts.append("Low composite risk score and baseline graph activity; routine processing recommended.")

    explanation_parts.append("Note: Investigation priority is a secondary ranking score designed for analyst workflow triage and does not imply confirmed illicit status.")
    explanation_text = " ".join(explanation_parts)

    return score, priority, explanation_text, factors

class InvestigationService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "InvestigationService":
        if cls._instance is None:
            cls._instance = InvestigationService()
        return cls._instance

    def get_investigation_details(self, tx_id: int) -> Dict[str, Any]:
        ds = DataService.get_instance()
        detail = ds.get_transaction_detail(tx_id)
        if not detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction ID {tx_id} not found."
            )

        gf = detail["graph_features"]
        nf = detail["neighborhood_risk_features"]

        model_risk = detail["risk_score"]
        high_risk_fraction = nf.get("high_risk_neighbor_fraction")
        neighborhood_mean = nf.get("neighborhood_mean_risk")
        neighborhood_vs_self = nf.get("neighborhood_vs_self_risk")
        total_degree = gf.get("total_degree", 0)
        degree_imbalance = gf.get("degree_imbalance", 0)

        score, priority, explanation_text, factors = compute_priority_and_explanation(
            tx_id,
            detail["time_step"],
            model_risk,
            high_risk_fraction,
            neighborhood_mean,
            neighborhood_vs_self,
            total_degree,
            degree_imbalance
        )

        model_signals = {
            "model_risk": model_risk,
            "neighborhood_mean_risk": nf.get("neighborhood_mean_risk"),
            "neighborhood_max_risk": nf.get("neighborhood_max_risk"),
            "neighborhood_median_risk": nf.get("neighborhood_median_risk"),
            "high_risk_neighbor_fraction": nf.get("high_risk_neighbor_fraction"),
            "neighborhood_vs_self_risk": nf.get("neighborhood_vs_self_risk")
        }

        graph_signals = {
            "in_degree": gf.get("in_degree", 0),
            "out_degree": gf.get("out_degree", 0),
            "total_degree": total_degree,
            "degree_imbalance": degree_imbalance,
            "absolute_degree_imbalance": gf.get("absolute_degree_imbalance", abs(degree_imbalance)),
            "unique_neighbors": gf.get("unique_neighbors", gf.get("total_degree", 0))
        }

        return {
            "tx_id": tx_id,
            "time_step": detail["time_step"],
            "model_risk": model_risk,
            "neighborhood_evidence": model_signals,
            "graph_evidence": graph_signals,
            "investigation_score": score,
            "investigation_priority": priority,
            "contributing_factors": factors,
            "explanation": explanation_text
        }
