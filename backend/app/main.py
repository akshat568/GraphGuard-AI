import logging
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.schemas import (
    HealthResponse,
    DashboardResponse,
    PaginatedTransactionsResponse,
    TransactionDetailResponse,
    NeighborsResponse,
    ExplanationResponse,
    PredictRequest,
    PredictResponse,
    ModelMetricsResponse,
    FeatureImportanceResponse,
    TimestepItem,
    RiskDistributionResponse,
    InvestigationResponse,
)
from backend.app.services.data_service import DataService
from backend.app.services.model_service import ModelService
from backend.app.services.graph_service import GraphService
from backend.app.services.explanation_service import ExplanationService
from backend.app.services.dashboard_service import DashboardService
from backend.app.services.investigation_service import InvestigationService

# Set up server-side logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("graphguard.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up GraphGuard AI backend server...")
    # Load dataset index & model at startup
    DataService.get_instance().load_data()
    ModelService.get_instance().load_model()
    yield
    logger.info("Shutting down GraphGuard AI backend server...")

app = FastAPI(
    title="GraphGuard AI Backend API",
    description="Production API for Bitcoin transaction fraud/illicit-transaction detection system.",
    version="1.1.0",
    lifespan=lifespan
)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for 500 errors to prevent exposing stack traces
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    logger.error(f"Unhandled server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred on the GraphGuard AI server."}
    )

# ------------------------------------------------------------
# 1. GET /api/health
# ------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    ds = DataService.get_instance()
    ms = ModelService.get_instance()
    return HealthResponse(
        status="healthy",
        model_loaded=ms.is_loaded,
        transactions_loaded=ds.is_loaded,
        graph_loaded=ds.is_loaded and (ds.total_edges > 0)
    )

# ------------------------------------------------------------
# 2. GET /api/dashboard
# ------------------------------------------------------------
@app.get("/api/dashboard", response_model=DashboardResponse, tags=["Dashboard"])
def get_dashboard():
    return DashboardService.get_instance().get_dashboard_metrics()

# ------------------------------------------------------------
# 3. GET /api/transactions
# ------------------------------------------------------------
@app.get("/api/transactions", response_model=PaginatedTransactionsResponse, tags=["Transactions"])
def get_transactions(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Page size (max 100)"),
    risk_level: Optional[str] = Query(None, description="Filter by risk level (LOW, MEDIUM, HIGH, CRITICAL)"),
    timestep: Optional[int] = Query(None, description="Filter by integer time step (1-49)"),
    search: Optional[str] = Query(None, description="Search by transaction ID")
):
    return DataService.get_instance().get_paginated_transactions(
        page=page,
        page_size=page_size,
        risk_level=risk_level,
        timestep=timestep,
        search=search
    )

# ------------------------------------------------------------
# 4. GET /api/transactions/{tx_id}
# ------------------------------------------------------------
@app.get("/api/transactions/{tx_id}", response_model=TransactionDetailResponse, tags=["Transactions"])
def get_transaction_by_id(tx_id: int):
    detail = DataService.get_instance().get_transaction_detail(tx_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction ID {tx_id} not found."
        )
    inv = InvestigationService.get_instance().get_investigation_details(tx_id)
    detail["investigation_priority"] = inv["investigation_priority"]
    detail["investigation_score"] = inv["investigation_score"]
    return detail

# ------------------------------------------------------------
# 5. GET /api/transactions/{tx_id}/neighbors
# ------------------------------------------------------------
@app.get("/api/transactions/{tx_id}/neighbors", response_model=NeighborsResponse, tags=["Transactions"])
def get_transaction_neighbors(tx_id: int):
    return GraphService.get_instance().get_neighbors(tx_id)

# ------------------------------------------------------------
# 6. GET /api/transactions/{tx_id}/explanation
# ------------------------------------------------------------
@app.get("/api/transactions/{tx_id}/explanation", response_model=ExplanationResponse, tags=["Transactions"])
def get_transaction_explanation(tx_id: int):
    exp = ExplanationService.get_instance().get_explanation(tx_id)
    inv = InvestigationService.get_instance().get_investigation_details(tx_id)
    exp["investigation_priority"] = inv["investigation_priority"]
    exp["investigation_score"] = inv["investigation_score"]
    return exp

# ------------------------------------------------------------
# 7. POST /api/predict
# ------------------------------------------------------------
@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_transaction(req: PredictRequest):
    res = ModelService.get_instance().predict_transaction(req.tx_id)
    inv = InvestigationService.get_instance().get_investigation_details(req.tx_id)
    res["investigation_priority"] = inv["investigation_priority"]
    res["investigation_score"] = inv["investigation_score"]
    return res

# ------------------------------------------------------------
# 8. GET /api/model/metrics
# ------------------------------------------------------------
@app.get("/api/model/metrics", response_model=ModelMetricsResponse, tags=["Model"])
def get_model_metrics():
    ds = DataService.get_instance()
    metrics = ds.metrics_json.get("metrics", {}).get("test", {})
    thresh = ds.threshold_summary_json.get("best_f1", {})

    return ModelMetricsResponse(
        model_name="leakage_safe_neighborhood_xgboost",
        test_pr_auc=float(metrics.get("pr_auc", 0.7441)),
        test_roc_auc=float(metrics.get("roc_auc", 0.9334)),
        test_precision=float(metrics.get("precision", 0.8787)),
        test_recall=float(metrics.get("recall", 0.6148)),
        test_f1=float(metrics.get("f1", 0.7234)),
        production_threshold=settings.PRODUCTION_ALERT_THRESHOLD,
        threshold_precision=float(thresh.get("precision", 0.9715)),
        threshold_recall=float(thresh.get("recall", 0.5896)),
        threshold_f1=float(thresh.get("f1", 0.7339))
    )

# ------------------------------------------------------------
# 9. GET /api/model/feature-importance
# ------------------------------------------------------------
@app.get("/api/model/feature-importance", response_model=FeatureImportanceResponse, tags=["Model"])
def get_feature_importance(
    limit: int = Query(20, ge=1, le=185, description="Number of top features to return")
):
    ds = DataService.get_instance()
    df = ds.feature_importance_df.head(limit)
    feats = []
    for _, row in df.iterrows():
        feats.append({
            "rank": int(row["rank"]),
            "feature": str(row["feature"]),
            "importance": float(row["importance"]),
            "group": str(row["group"])
        })
    return FeatureImportanceResponse(
        total_features=len(ds.feature_importance_df),
        limit=limit,
        features=feats
    )

# ------------------------------------------------------------
# 10. GET /api/timesteps
# ------------------------------------------------------------
@app.get("/api/timesteps", response_model=List[TimestepItem], tags=["Timesteps"])
def get_timesteps():
    return DataService.get_instance().timesteps_summary

# ------------------------------------------------------------
# 11. GET /api/risk-distribution
# ------------------------------------------------------------
@app.get("/api/risk-distribution", response_model=RiskDistributionResponse, tags=["Dashboard"])
def get_risk_distribution():
    return DashboardService.get_instance().get_risk_distribution()

# ------------------------------------------------------------
# 12. GET /api/transactions/{tx_id}/investigation (NEW PHASE 6)
# ------------------------------------------------------------
@app.get("/api/transactions/{tx_id}/investigation", response_model=InvestigationResponse, tags=["Investigation"])
def get_transaction_investigation(tx_id: int):
    return InvestigationService.get_instance().get_investigation_details(tx_id)
