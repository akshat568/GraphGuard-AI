import type {
  HealthResponse,
  DashboardResponse,
  PaginatedTransactionsResponse,
  TransactionDetailResponse,
  NeighborsResponse,
  ExplanationResponse,
  PredictResponse,
  ModelMetricsResponse,
  FeatureImportanceResponse,
  TimestepItem,
  RiskDistributionResponse,
  InvestigationResponse,
} from '../types/api';

const API_BASE_URL = 'http://127.0.0.1:8001';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const errorJson = await res.json();
      if (errorJson.detail) {
        errorDetail = typeof errorJson.detail === 'string' ? errorJson.detail : JSON.stringify(errorJson.detail);
      }
    } catch (_) {
      // ignore fallback
    }
    const error = new Error(errorDetail) as Error & { status?: number };
    error.status = res.status;
    throw error;
  }
  return (await res.json()) as T;
}

export const api = {
  getHealth: () => request<HealthResponse>('/api/health'),
  getDashboard: () => request<DashboardResponse>('/api/dashboard'),
  getTransactions: (
    page = 1,
    pageSize = 20,
    riskLevel?: string,
    timestep?: number,
    search?: string
  ) => {
    const params = new URLSearchParams();
    params.set('page', page.toString());
    params.set('page_size', pageSize.toString());
    if (riskLevel && riskLevel !== 'ALL') params.set('risk_level', riskLevel);
    if (timestep !== undefined && timestep !== null) params.set('timestep', timestep.toString());
    if (search && search.trim()) params.set('search', search.trim());
    return request<PaginatedTransactionsResponse>(`/api/transactions?${params.toString()}`);
  },
  getTransactionDetail: (txId: number) => request<TransactionDetailResponse>(`/api/transactions/${txId}`),
  getTransactionNeighbors: (txId: number) => request<NeighborsResponse>(`/api/transactions/${txId}/neighbors`),
  getTransactionExplanation: (txId: number) => request<ExplanationResponse>(`/api/transactions/${txId}/explanation`),
  predictTransaction: (txId: number) =>
    request<PredictResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ tx_id: txId }),
    }),
  getModelMetrics: () => request<ModelMetricsResponse>('/api/model/metrics'),
  getFeatureImportance: (limit = 20) => request<FeatureImportanceResponse>(`/api/model/feature-importance?limit=${limit}`),
  getTimesteps: () => request<TimestepItem[]>('/api/timesteps'),
  getRiskDistribution: () => request<RiskDistributionResponse>('/api/risk-distribution'),
  getInvestigation: (txId: number) => request<InvestigationResponse>(`/api/transactions/${txId}/investigation`),
};
