export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  transactions_loaded: boolean;
  graph_loaded: boolean;
}

export interface DashboardResponse {
  total_transactions: number;
  total_edges: number;
  labeled_transactions: number;
  illicit_transactions: number;
  licit_transactions: number;
  unknown_transactions: number;
  high_risk_transactions: number;
  critical_risk_transactions: number;
  current_threshold: number;
  model_pr_auc: number;
  model_roc_auc: number;
  model_f1: number;
}

export interface TransactionSummary {
  tx_id: number;
  time_step: number;
  risk_score: number | null;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL' | 'UNKNOWN';
  prediction: string;
  in_degree: number;
  out_degree: number;
  investigation_priority?: string;
  investigation_score?: number;
}

export interface PaginatedTransactionsResponse {
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  transactions: TransactionSummary[];
}

export interface DegreeInformation {
  in_degree: number;
  out_degree: number;
  total_degree: number;
}

export interface TransactionDetailResponse {
  tx_id: number;
  time_step: number;
  risk_score: number | null;
  risk_level: string;
  prediction: string;
  class: string;
  graph_features: Record<string, number>;
  neighborhood_risk_features: Record<string, number | null>;
  degree_information: DegreeInformation;
  investigation_priority?: string;
  investigation_score?: number;
}

export interface NeighborItem {
  tx_id: number;
  relationship: 'incoming' | 'outgoing';
  time_step: number;
  risk_score: number | null;
  risk_level: string;
  class: string;
}

export interface NeighborsResponse {
  tx_id: number;
  total_neighbors: number;
  incoming_count: number;
  outgoing_count: number;
  neighbors: NeighborItem[];
}

export interface FeatureImportanceItem {
  rank: number;
  feature: string;
  importance: number;
  group: string;
}

export interface ExplanationResponse {
  tx_id: number;
  risk_score: number | null;
  risk_level: string;
  prediction: string;
  top_contributing_model_features: FeatureImportanceItem[];
  important_neighborhood_signals: Record<string, number | null>;
  graph_signals: Record<string, number>;
  explanation_text: string;
  investigation_priority?: string;
  investigation_score?: number;
}

export interface PredictResponse {
  tx_id: number;
  risk_score: number;
  risk_level: string;
  prediction: string;
  investigation_priority?: string;
  investigation_score?: number;
}

export interface ModelMetricsResponse {
  model_name: string;
  test_pr_auc: number;
  test_roc_auc: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  production_threshold: number;
  threshold_precision: number;
  threshold_recall: number;
  threshold_f1: number;
}

export interface FeatureImportanceResponse {
  total_features: number;
  limit: number;
  features: FeatureImportanceItem[];
}

export interface TimestepItem {
  timestep: number;
  total_transactions: number;
  labeled_transactions: number;
  illicit: number;
  licit: number;
  unknown: number;
}

export interface RiskDistributionResponse {
  counts: {
    LOW: number;
    MEDIUM: number;
    HIGH: number;
    CRITICAL: number;
    UNASSESSED: number;
  };
  ranges: {
    LOW: string;
    MEDIUM: string;
    HIGH: string;
    CRITICAL: string;
  };
  total_assessed: number;
}

export interface InvestigationResponse {
  tx_id: number;
  time_step: number;
  model_risk: number | null;
  neighborhood_evidence: Record<string, number | null>;
  graph_evidence: Record<string, number>;
  investigation_score: number;
  investigation_priority: 'IMMEDIATE' | 'HIGH' | 'REVIEW' | 'LOW' | 'UNASSESSED';
  contributing_factors: Record<string, number>;
  explanation: string;
}
