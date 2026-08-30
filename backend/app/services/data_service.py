import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from backend.app.config import settings

logger = logging.getLogger("graphguard.data_service")

def get_risk_level_and_prediction(risk_score: Optional[float]) -> Tuple[str, str]:
    if risk_score is None or pd.isna(risk_score):
        return "UNKNOWN", "unassessed"
    score = float(risk_score)
    if score >= settings.PRODUCTION_ALERT_THRESHOLD:  # 0.90
        return "CRITICAL", "illicit / high-confidence alert"
    elif score >= settings.HIGH_RISK_THRESHOLD:  # 0.50
        return "HIGH", "review recommended"
    elif score >= settings.MEDIUM_RISK_THRESHOLD:  # 0.25
        return "MEDIUM", "medium risk"
    else:
        return "LOW", "low risk"

def map_class_name(raw_class: str) -> str:
    raw_str = str(raw_class).strip()
    if raw_str == "1":
        return "illicit"
    elif raw_str == "2":
        return "licit"
    else:
        return "unknown"

class DataService:
    _instance = None

    def __init__(self):
        self.is_loaded: bool = False

        # Raw DataFrames / Dictionaries
        self.classes_df: pd.DataFrame = pd.DataFrame()
        self.features_df: pd.DataFrame = pd.DataFrame()
        self.graph_features_df: pd.DataFrame = pd.DataFrame()
        self.neighborhood_df: pd.DataFrame = pd.DataFrame()

        # Graph Adjacencies
        self.incoming_adj: Dict[int, List[int]] = {}
        self.outgoing_adj: Dict[int, List[int]] = {}
        self.total_edges: int = 0

        # Metadata / JSONs
        self.feature_importance_df: pd.DataFrame = pd.DataFrame()
        self.metrics_json: Dict[str, Any] = {}
        self.threshold_summary_json: Dict[str, Any] = {}

        # Fast Summary DataFrame indexed by txId
        self.summary_df: pd.DataFrame = pd.DataFrame()

        # Timestep Aggregates Cache
        self.timesteps_summary: List[Dict[str, Any]] = []

    @classmethod
    def get_instance(cls) -> "DataService":
        if cls._instance is None:
            cls._instance = DataService()
        return cls._instance

    def load_data(self):
        if self.is_loaded:
            return

        logger.info("Starting DataService initialization...")

        # 1. Load classes CSV
        logger.info(f"Loading classes from {settings.CLASSES_FILE}")
        self.classes_df = pd.read_csv(settings.CLASSES_FILE)
        self.classes_df["txId"] = self.classes_df["txId"].astype(int)
        self.classes_df.set_index("txId", inplace=True)
        self.classes_df["class_mapped"] = self.classes_df["class"].astype(str).map(map_class_name)

        # 2. Load graph features CSV
        logger.info(f"Loading graph features from {settings.GRAPH_FEATURES_FILE}")
        self.graph_features_df = pd.read_csv(settings.GRAPH_FEATURES_FILE)
        self.graph_features_df["txId"] = self.graph_features_df["txId"].astype(int)
        self.graph_features_df.set_index("txId", inplace=True)

        # 3. Load leakage-safe neighborhood risk CSV
        logger.info(f"Loading neighborhood features from {settings.LEAKAGE_SAFE_NEIGHBORHOOD_FILE}")
        self.neighborhood_df = pd.read_csv(settings.LEAKAGE_SAFE_NEIGHBORHOOD_FILE)
        self.neighborhood_df["txId"] = self.neighborhood_df["txId"].astype(int)
        self.neighborhood_df.set_index("txId", inplace=True)

        # 4. Load original features CSV (657 MB) - no header in CSV
        logger.info(f"Loading features from {settings.FEATURES_FILE} (this may take a few seconds)...")
        feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, 166)]
        self.features_df = pd.read_csv(settings.FEATURES_FILE, header=None, names=feature_cols)
        self.features_df["txId"] = self.features_df["txId"].astype(int)
        self.features_df.set_index("txId", inplace=True)

        # 5. Load edgelist CSV & build adjacency dicts
        logger.info(f"Loading edgelist from {settings.EDGELIST_FILE}")
        edgelist_df = pd.read_csv(settings.EDGELIST_FILE)
        self.total_edges = len(edgelist_df)

        inc_adj: Dict[int, List[int]] = {}
        out_adj: Dict[int, List[int]] = {}
        for _, row in edgelist_df.iterrows():
            u = int(row["txId1"])
            v = int(row["txId2"])
            out_adj.setdefault(u, []).append(v)
            inc_adj.setdefault(v, []).append(u)
        self.incoming_adj = inc_adj
        self.outgoing_adj = out_adj

        # 6. Load Feature Importance & JSON Metrics
        logger.info(f"Loading metrics and feature importances...")
        self.feature_importance_df = pd.read_csv(settings.FEATURE_IMPORTANCE_FILE)

        with open(settings.METRICS_FILE, "r") as f:
            self.metrics_json = json.load(f)

        with open(settings.THRESHOLD_SUMMARY_FILE, "r") as f:
            self.threshold_summary_json = json.load(f)

        # 7. Construct Fast Transaction Summary DataFrame
        logger.info("Building fast transaction summary index...")
        summary = pd.DataFrame(index=self.features_df.index)
        summary["time_step"] = self.features_df["time_step"].astype(int)
        summary["class_mapped"] = self.classes_df["class_mapped"]
        summary["risk_score"] = self.neighborhood_df["model_risk"]
        summary["in_degree"] = self.graph_features_df["in_degree"].fillna(0).astype(int)
        summary["out_degree"] = self.graph_features_df["out_degree"].fillna(0).astype(int)

        # Assign risk level & prediction
        risk_scores = summary["risk_score"].to_numpy()
        risk_levels = []
        predictions = []
        for s in risk_scores:
            rl, p = get_risk_level_and_prediction(s)
            risk_levels.append(rl)
            predictions.append(p)

        summary["risk_level"] = risk_levels
        summary["prediction"] = predictions

        self.summary_df = summary

        # 8. Compute Timestep Aggregates
        logger.info("Computing timestep aggregate statistics...")
        ts_groups = self.summary_df.groupby("time_step")
        ts_list = []
        for ts, group in ts_groups:
            total_tx = len(group)
            illicit_cnt = int((group["class_mapped"] == "illicit").sum())
            licit_cnt = int((group["class_mapped"] == "licit").sum())
            unknown_cnt = int((group["class_mapped"] == "unknown").sum())
            labeled_cnt = illicit_cnt + licit_cnt
            ts_list.append({
                "timestep": int(ts),
                "total_transactions": total_tx,
                "labeled_transactions": labeled_cnt,
                "illicit": illicit_cnt,
                "licit": licit_cnt,
                "unknown": unknown_cnt
            })
        self.timesteps_summary = sorted(ts_list, key=lambda x: x["timestep"])

        self.is_loaded = True
        logger.info("DataService initialized successfully.")

    def get_transaction_summary(self, tx_id: int) -> Optional[Dict[str, Any]]:
        if tx_id not in self.summary_df.index:
            return None
        row = self.summary_df.loc[tx_id]
        risk_score = None if pd.isna(row["risk_score"]) else float(row["risk_score"])
        return {
            "tx_id": tx_id,
            "time_step": int(row["time_step"]),
            "risk_score": risk_score,
            "risk_level": str(row["risk_level"]),
            "prediction": str(row["prediction"]),
            "in_degree": int(row["in_degree"]),
            "out_degree": int(row["out_degree"])
        }

    def get_transaction_detail(self, tx_id: int) -> Optional[Dict[str, Any]]:
        summary = self.get_transaction_summary(tx_id)
        if summary is None:
            return None

        # Class label
        raw_cls = self.classes_df.loc[tx_id, "class"] if tx_id in self.classes_df.index else "unknown"
        class_mapped = map_class_name(raw_cls)

        # Graph features dictionary
        gf_dict = {}
        if tx_id in self.graph_features_df.index:
            gf_row = self.graph_features_df.loc[tx_id]
            for col in self.graph_features_df.columns:
                val = gf_row[col]
                gf_dict[col] = float(val) if isinstance(val, (np.floating, float)) else int(val)

        # Neighborhood risk features dictionary
        nf_dict = {}
        if tx_id in self.neighborhood_df.index:
            nf_row = self.neighborhood_df.loc[tx_id]
            for col in self.neighborhood_df.columns:
                if col in ["txId", "time_step"]:
                    continue
                val = nf_row[col]
                if pd.isna(val):
                    nf_dict[col] = None
                elif isinstance(val, (np.floating, float)):
                    nf_dict[col] = float(val)
                else:
                    nf_dict[col] = int(val)

        degree_info = {
            "in_degree": gf_dict.get("in_degree", summary["in_degree"]),
            "out_degree": gf_dict.get("out_degree", summary["out_degree"]),
            "total_degree": gf_dict.get("total_degree", summary["in_degree"] + summary["out_degree"])
        }

        return {
            "tx_id": tx_id,
            "time_step": summary["time_step"],
            "risk_score": summary["risk_score"],
            "risk_level": summary["risk_level"],
            "prediction": summary["prediction"],
            "class": class_mapped,
            "graph_features": gf_dict,
            "neighborhood_risk_features": nf_dict,
            "degree_information": degree_info
        }

    def get_paginated_transactions(
        self,
        page: int = 1,
        page_size: int = 20,
        risk_level: Optional[str] = None,
        timestep: Optional[int] = None,
        search: Optional[str] = None
    ) -> Dict[str, Any]:
        df = self.summary_df

        if risk_level:
            df = df[df["risk_level"].str.upper() == risk_level.upper()]
        if timestep is not None:
            df = df[df["time_step"] == timestep]
        if search:
            search_str = search.strip()
            # If search is pure numeric, try exact or prefix match on index
            try:
                search_val = int(search_str)
                df = df[df.index.astype(str).str.contains(search_str)]
            except ValueError:
                pass

        total_records = len(df)
        total_pages = max(1, (total_records + page_size - 1) // page_size)

        # Clamp page number
        current_page = max(1, min(page, total_pages))
        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size

        page_df = df.iloc[start_idx:end_idx]

        tx_list = []
        for tx_id, row in page_df.iterrows():
            risk_score = None if pd.isna(row["risk_score"]) else float(row["risk_score"])
            tx_list.append({
                "tx_id": int(tx_id),
                "time_step": int(row["time_step"]),
                "risk_score": risk_score,
                "risk_level": str(row["risk_level"]),
                "prediction": str(row["prediction"]),
                "in_degree": int(row["in_degree"]),
                "out_degree": int(row["out_degree"])
            })

        return {
            "total": total_records,
            "page": current_page,
            "page_size": page_size,
            "total_pages": total_pages,
            "transactions": tx_list
        }
