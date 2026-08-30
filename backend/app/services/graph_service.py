import logging
from typing import Dict, List, Any
from fastapi import HTTPException, status

from backend.app.services.data_service import DataService

logger = logging.getLogger("graphguard.graph_service")

class GraphService:
    _instance = None

    @classmethod
    def get_instance(cls) -> "GraphService":
        if cls._instance is None:
            cls._instance = GraphService()
        return cls._instance

    def get_neighbors(self, tx_id: int) -> Dict[str, Any]:
        ds = DataService.get_instance()
        if tx_id not in ds.summary_df.index:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transaction ID {tx_id} not found."
            )

        inc_nodes = ds.incoming_adj.get(tx_id, [])
        out_nodes = ds.outgoing_adj.get(tx_id, [])

        neighbors_list: List[Dict[str, Any]] = []

        # Incoming neighbors (sources sending to tx_id)
        for n_id in inc_nodes:
            summary = ds.get_transaction_summary(n_id)
            if summary:
                cls_mapped = ds.classes_df.loc[n_id, "class_mapped"] if n_id in ds.classes_df.index else "unknown"
                neighbors_list.append({
                    "tx_id": n_id,
                    "relationship": "incoming",
                    "time_step": summary["time_step"],
                    "risk_score": summary["risk_score"],
                    "risk_level": summary["risk_level"],
                    "class": cls_mapped
                })

        # Outgoing neighbors (targets receiving from tx_id)
        for n_id in out_nodes:
            summary = ds.get_transaction_summary(n_id)
            if summary:
                cls_mapped = ds.classes_df.loc[n_id, "class_mapped"] if n_id in ds.classes_df.index else "unknown"
                neighbors_list.append({
                    "tx_id": n_id,
                    "relationship": "outgoing",
                    "time_step": summary["time_step"],
                    "risk_score": summary["risk_score"],
                    "risk_level": summary["risk_level"],
                    "class": cls_mapped
                })

        return {
            "tx_id": tx_id,
            "total_neighbors": len(neighbors_list),
            "incoming_count": len(inc_nodes),
            "outgoing_count": len(out_nodes),
            "neighbors": neighbors_list
        }
