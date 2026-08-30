import logging
from typing import Dict, Any, List
from fastapi import HTTPException, status

from backend.app.services.data_service import DataService

logger = logging.getLogger("graphguard.explanation_service")

class ExplanationService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "ExplanationService":
        if cls._instance is None:
            cls._instance = ExplanationService()
        return cls._instance

    def get_explanation(self, tx_id: int, top_k_features: int = 10) -> Dict[str, Any]:
        ds = DataService.get_instance()
        detail = ds.get_transaction_detail(tx_id)
        if not detail:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction ID {tx_id} not found."
            )

        # Top global model features from feature importance file
        top_feats = []
        for _, row in ds.feature_importance_df.head(top_k_features).iterrows():
            top_feats.append({
                "rank": int(row["rank"]),
                "feature": str(row["feature"]),
                "importance": float(row["importance"]),
                "group": str(row["group"])
            })

        # Important neighborhood signals for this transaction
        nf_dict = detail["neighborhood_risk_features"]
        key_n_features = [
            "model_risk",
            "neighborhood_vs_self_risk",
            "incoming_median_risk",
            "incoming_mean_risk",
            "neighborhood_mean_risk",
            "incoming_high_risk_neighbors",
            "outgoing_high_risk_neighbors",
            "high_risk_neighbor_fraction"
        ]
        important_n_signals = {k: nf_dict[k] for k in key_n_features if k in nf_dict}

        # Key graph structural signals for this transaction
        gf_dict = detail["graph_features"]
        key_g_features = [
            "in_degree",
            "out_degree",
            "total_degree",
            "degree_imbalance",
            "has_incoming",
            "has_outgoing",
            "is_source",
            "is_sink",
            "is_isolated"
        ]
        important_g_signals = {k: gf_dict[k] for k in key_g_features if k in gf_dict}

        # Synthesize clear, empirical explanation text
        risk_level = detail["risk_level"]
        risk_score = detail["risk_score"]
        time_step = detail["time_step"]
        class_label = detail["class"]

        explanation_parts = []
        explanation_parts.append(
            f"Transaction {tx_id} (timestep {time_step}) has an evaluated risk level of {risk_level}"
            + (f" with risk score {risk_score:.4f}." if risk_score is not None else " (risk score unassessed).")
        )
        if class_label != "unknown":
            explanation_parts.append(f"Ground truth label is known as '{class_label}'.")

        # Evaluate transaction-level empirical signals
        m_risk = float(risk_score) if risk_score is not None else (
            float(important_n_signals.get("model_risk")) if important_n_signals.get("model_risk") is not None else 0.0
        )

        # Neighborhood risk signals
        n_vs_self = important_n_signals.get("neighborhood_vs_self_risk")
        high_risk_neighbors = important_n_signals.get("high_risk_neighbor_fraction")
        inc_mean = important_n_signals.get("incoming_mean_risk")
        n_mean = important_n_signals.get("neighborhood_mean_risk")
        inc_high = important_n_signals.get("incoming_high_risk_neighbors")
        out_high = important_n_signals.get("outgoing_high_risk_neighbors")

        n_signals_desc = []
        if inc_mean is not None and inc_mean > 0.25:
            n_signals_desc.append(f"incoming neighborhood mean risk of {inc_mean:.2f}")
        elif n_mean is not None and n_mean > 0.25:
            n_signals_desc.append(f"neighborhood mean risk of {n_mean:.2f}")

        if high_risk_neighbors is not None and high_risk_neighbors > 0:
            n_signals_desc.append(f"fraction of high-risk neighbors is {high_risk_neighbors * 100:.1f}%")
        elif inc_high is not None and inc_high > 0:
            n_signals_desc.append(f"{inc_high} incoming high-risk neighbor(s)")
        elif out_high is not None and out_high > 0:
            n_signals_desc.append(f"{out_high} outgoing high-risk neighbor(s)")

        if n_vs_self is not None and n_vs_self > 0.1:
            n_signals_desc.append(f"neighborhood contrast score of {n_vs_self:.2f}")

        is_neighborhood_elevated = len(n_signals_desc) > 0
        is_neighborhood_low = not is_neighborhood_elevated

        # Graph structure signals
        total_degree = important_g_signals.get("total_degree", 0)
        degree_imbalance = important_g_signals.get("degree_imbalance", 0)

        g_signals_desc = []
        if total_degree >= 3:
            g_signals_desc.append(f"total degree of {total_degree}")
        if abs(degree_imbalance) >= 2:
            g_signals_desc.append(f"degree imbalance of {degree_imbalance}")

        is_graph_elevated = len(g_signals_desc) > 0

        # Synthesize explanation according to actual transaction-level evidence
        if m_risk >= 0.90:
            if is_neighborhood_low:
                explanation_parts.append("The primary model risk signal is very high while neighborhood signals are low.")
            else:
                explanation_parts.append("The primary model risk signal is very high.")
                explanation_parts.append(f"Neighborhood signals provide supporting evidence ({', '.join(n_signals_desc)}).")
        else:
            if is_neighborhood_elevated:
                explanation_parts.append(f"Neighborhood signals provide supporting evidence ({', '.join(n_signals_desc)}).")

        if is_graph_elevated:
            explanation_parts.append(f"Graph structure provides a supporting signal ({', '.join(g_signals_desc)}).")

        if not (m_risk >= 0.90) and not is_neighborhood_elevated and not is_graph_elevated:
            explanation_parts.append("Transaction exhibits low neighborhood risk signals and normal graph structural behavior.")

        explanation_parts.append(
            "Note: Global feature importance reflects trained model weights across all 185 features. Transaction evidence is derived from real graph and neighborhood statistics and does not establish illicit status."
        )

        explanation_text = " ".join(explanation_parts)

        return {
            "tx_id": tx_id,
            "risk_score": detail["risk_score"],
            "risk_level": detail["risk_level"],
            "prediction": detail["prediction"],
            "top_contributing_model_features": top_feats,
            "important_neighborhood_signals": important_n_signals,
            "graph_signals": important_g_signals,
            "explanation_text": explanation_text
        }
