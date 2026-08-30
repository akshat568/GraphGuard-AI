from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    transactions_loaded: bool
    graph_loaded: bool

class DashboardResponse(BaseModel):
    total_transactions: int
    total_edges: int
    labeled_transactions: int
    illicit_transactions: int
    licit_transactions: int
    unknown_transactions: int
    high_risk_transactions: int
    critical_risk_transactions: int
    current_threshold: float
    model_pr_auc: float
    model_roc_auc: float
    model_f1: float

class TransactionSummary(BaseModel):
    tx_id: int
    time_step: int
    risk_score: Optional[float] = None
    risk_level: str
    prediction: str
    in_degree: int
    out_degree: int
    investigation_priority: Optional[str] = None
    investigation_score: Optional[float] = None

class PaginatedTransactionsResponse(BaseModel):
    total: int
    page: int
    page_size: int
    total_pages: int
    transactions: List[TransactionSummary]

class DegreeInformation(BaseModel):
    in_degree: int
    out_degree: int
    total_degree: int

class TransactionDetailResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tx_id: int
    time_step: int
    risk_score: Optional[float] = None
    risk_level: str
    prediction: str
    class_label: str = Field(..., alias="class")
    graph_features: Dict[str, Any]
    neighborhood_risk_features: Dict[str, Any]
    degree_information: DegreeInformation
    investigation_priority: Optional[str] = None
    investigation_score: Optional[float] = None

class NeighborItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    tx_id: int
    relationship: str  # "incoming" or "outgoing"
    time_step: int
    risk_score: Optional[float] = None
    risk_level: str
    class_label: str = Field(..., alias="class")

class NeighborsResponse(BaseModel):
    tx_id: int
    total_neighbors: int
    incoming_count: int
    outgoing_count: int
    neighbors: List[NeighborItem]

class FeatureImportanceItem(BaseModel):
    rank: int
    feature: str
    importance: float
    group: str

class ExplanationResponse(BaseModel):
    tx_id: int
    risk_score: Optional[float] = None
    risk_level: str
    prediction: str
    top_contributing_model_features: List[FeatureImportanceItem]
    important_neighborhood_signals: Dict[str, Any]
    graph_signals: Dict[str, Any]
    explanation_text: str
    investigation_priority: Optional[str] = None
    investigation_score: Optional[float] = None

class PredictRequest(BaseModel):
    tx_id: int

class PredictResponse(BaseModel):
    tx_id: int
    risk_score: float
    risk_level: str
    prediction: str
    investigation_priority: Optional[str] = None
    investigation_score: Optional[float] = None

class ModelMetricsResponse(BaseModel):
    model_name: str
    test_pr_auc: float
    test_roc_auc: float
    test_precision: float
    test_recall: float
    test_f1: float
    production_threshold: float
    threshold_precision: float
    threshold_recall: float
    threshold_f1: float

class FeatureImportanceResponse(BaseModel):
    total_features: int
    limit: int
    features: List[FeatureImportanceItem]

class TimestepItem(BaseModel):
    timestep: int
    total_transactions: int
    labeled_transactions: int
    illicit: int
    licit: int
    unknown: int

class RiskDistributionResponse(BaseModel):
    counts: Dict[str, int]
    ranges: Dict[str, str]
    total_assessed: int

class InvestigationResponse(BaseModel):
    tx_id: int
    time_step: int
    model_risk: Optional[float] = None
    neighborhood_evidence: Dict[str, Any]
    graph_evidence: Dict[str, Any]
    investigation_score: float
    investigation_priority: str
    contributing_factors: Dict[str, Any]
    explanation: str
