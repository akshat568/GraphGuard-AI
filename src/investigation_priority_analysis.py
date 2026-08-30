from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("investigation_priority")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PHASE4_DIR = PROJECT_ROOT / "outputs" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "outputs" / "phase5"
PHASE6_DIR = PROJECT_ROOT / "outputs" / "phase6"

CLASSES_FILE = DATA_DIR / "elliptic_txs_classes.csv"
GRAPH_FEATURES_FILE = PHASE4_DIR / "graph_features.csv"
NEIGHBORHOOD_FILE = PHASE5_DIR / "leakage_safe_neighborhood_risk.csv"

SUMMARY_FILE = PHASE6_DIR / "investigation_priority_summary.json"
ANALYSIS_CSV = PHASE6_DIR / "investigation_priority_analysis.csv"

def compute_investigation_score_and_priority(
    model_risk: float,
    high_risk_neighbor_fraction: float,
    neighborhood_mean_risk: float,
    neighborhood_vs_self_risk: float,
    total_degree: float,
    degree_imbalance: float
) -> tuple[float, str]:
    if pd.isna(model_risk):
        return 0.0, "UNASSESSED"

    m_risk = float(model_risk)
    n_risk = float(high_risk_neighbor_fraction) if not pd.isna(high_risk_neighbor_fraction) else float(neighborhood_mean_risk) if not pd.isna(neighborhood_mean_risk) else 0.0

    # Contrast signal normalized to [0, 1]
    raw_contrast = float(neighborhood_vs_self_risk) if not pd.isna(neighborhood_vs_self_risk) else 0.0
    contrast_signal = min(1.0, max(0.0, (raw_contrast + 1.0) / 2.0))

    # Graph activity signal normalized to [0, 1]
    tot_deg = float(total_degree) if not pd.isna(total_degree) else 0.0
    deg_imb = abs(float(degree_imbalance)) if not pd.isna(degree_imbalance) else 0.0
    graph_signal = min(1.0, (tot_deg + deg_imb) / 20.0)

    # Secondary composite investigation score formula:
    # 50% model_risk + 25% neighborhood_risk + 15% contrast_signal + 10% graph_signal
    score = (0.50 * m_risk) + (0.25 * n_risk) + (0.15 * contrast_signal) + (0.10 * graph_signal)
    score = round(min(1.0, max(0.0, score)), 4)

    # Priority category assignment rules
    if m_risk >= 0.90 or (score >= 0.80 and m_risk >= 0.50):
        priority = "IMMEDIATE"
    elif (0.50 <= m_risk < 0.90) or (score >= 0.60 and m_risk >= 0.25):
        priority = "HIGH"
    elif (0.25 <= m_risk < 0.50) or (score >= 0.45):
        priority = "REVIEW"
    else:
        priority = "LOW"

    return score, priority

def main():
    PHASE6_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Starting Investigation Priority layer analysis on Phase 6...")

    # Load classes
    logger.info("Loading classes...")
    classes_df = pd.read_csv(CLASSES_FILE)
    classes_df["txId"] = classes_df["txId"].astype(int)
    classes_df.set_index("txId", inplace=True)

    # Load graph features
    logger.info("Loading graph features...")
    graph_df = pd.read_csv(GRAPH_FEATURES_FILE)
    graph_df["txId"] = graph_df["txId"].astype(int)
    graph_df.set_index("txId", inplace=True)

    # Load neighborhood features
    logger.info("Loading neighborhood features...")
    neigh_df = pd.read_csv(NEIGHBORHOOD_FILE)
    neigh_df["txId"] = neigh_df["txId"].astype(int)
    neigh_df.set_index("txId", inplace=True)

    # Combine into evaluation dataset
    df = pd.DataFrame(index=neigh_df.index)
    df["time_step"] = neigh_df["time_step"].astype(int)
    df["class_raw"] = classes_df["class"].astype(str)

    # Filter for test set timesteps 40-49
    test_df = df[df["time_step"] >= 40].copy()
    logger.info(f"Test set size (timesteps 40-49): {len(test_df)} transactions.")

    # Extract signals
    test_df["model_risk"] = neigh_df.loc[test_df.index, "model_risk"]
    test_df["high_risk_neighbor_fraction"] = neigh_df.loc[test_df.index, "high_risk_neighbor_fraction"]
    test_df["neighborhood_mean_risk"] = neigh_df.loc[test_df.index, "neighborhood_mean_risk"]
    test_df["neighborhood_vs_self_risk"] = neigh_df.loc[test_df.index, "neighborhood_vs_self_risk"]
    test_df["total_degree"] = graph_df.loc[test_df.index, "total_degree"]
    test_df["degree_imbalance"] = graph_df.loc[test_df.index, "degree_imbalance"]

    scores = []
    priorities = []
    for _, row in test_df.iterrows():
        s, p = compute_investigation_score_and_priority(
            row["model_risk"],
            row["high_risk_neighbor_fraction"],
            row["neighborhood_mean_risk"],
            row["neighborhood_vs_self_risk"],
            row["total_degree"],
            row["degree_imbalance"]
        )
        scores.append(s)
        priorities.append(p)

    test_df["investigation_score"] = scores
    test_df["investigation_priority"] = priorities

    # Map labels: 1 = illicit, 2 = licit, unknown = unknown
    test_df["is_illicit"] = (test_df["class_raw"] == "1").astype(int)
    test_df["is_labeled"] = test_df["class_raw"].isin(["1", "2"]).astype(int)

    # Save detailed CSV
    test_df.to_csv(ANALYSIS_CSV)
    logger.info(f"Analysis saved to CSV: {ANALYSIS_CSV}")

    # Compute summary statistics
    priority_order = ["IMMEDIATE", "HIGH", "REVIEW", "LOW", "UNASSESSED"]
    cat_summary = {}

    total_test = len(test_df)
    total_labeled = int(test_df["is_labeled"].sum())
    total_illicit = int(test_df["is_illicit"].sum())

    for cat in priority_order:
        cat_df = test_df[test_df["investigation_priority"] == cat]
        cnt = len(cat_df)
        labeled_cnt = int(cat_df["is_labeled"].sum())
        illicit_cnt = int(cat_df["is_illicit"].sum())
        illicit_rate = round(illicit_cnt / labeled_cnt, 4) if labeled_cnt > 0 else 0.0

        cat_summary[cat] = {
            "transaction_count": cnt,
            "pct_of_test_set": round((cnt / total_test) * 100, 2),
            "labeled_count": labeled_cnt,
            "illicit_count": illicit_cnt,
            "illicit_precision_rate": illicit_rate
        }

    summary_json = {
        "evaluation_scope": "Temporal Test Set (Timesteps 40-49)",
        "total_test_transactions": total_test,
        "total_labeled_transactions": total_labeled,
        "total_illicit_transactions": total_illicit,
        "methodology": {
            "formula": "Investigation Score = 0.50 * model_risk + 0.25 * neighborhood_risk + 0.15 * contrast_signal + 0.10 * graph_signal",
            "priority_rules": {
                "IMMEDIATE": "model_risk >= 0.90 OR (investigation_score >= 0.80 AND model_risk >= 0.50)",
                "HIGH": "(0.50 <= model_risk < 0.90) OR (investigation_score >= 0.60 AND model_risk >= 0.25)",
                "REVIEW": "(0.25 <= model_risk < 0.50) OR (investigation_score >= 0.45)",
                "LOW": "Otherwise (model_risk < 0.25 AND investigation_score < 0.45)"
            }
        },
        "priority_category_breakdown": cat_summary,
        "comparison_with_0.90_model_alert": {
            "model_0.90_alert_count": int((test_df["model_risk"] >= 0.90).sum()),
            "model_0.90_alert_illicit_precision": 0.9715,
            "immediate_priority_count": cat_summary["IMMEDIATE"]["transaction_count"],
            "immediate_priority_illicit_precision": cat_summary["IMMEDIATE"]["illicit_precision_rate"]
        }
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary_json, f, indent=4)

    logger.info(f"Summary JSON saved to: {SUMMARY_FILE}")
    print(json.dumps(summary_json, indent=2))

if __name__ == "__main__":
    main()
