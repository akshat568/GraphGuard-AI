import { useEffect, useState, useCallback } from 'react';
import { api } from './api/client';
import type {
  HealthResponse,
  DashboardResponse,
  PaginatedTransactionsResponse,
  ModelMetricsResponse,
  FeatureImportanceItem,
  TimestepItem,
  RiskDistributionResponse,
} from './types/api';
import { Header } from './components/Header';
import { DashboardOverview } from './components/DashboardOverview';
import { RiskDistributionCard } from './components/RiskDistributionCard';
import { ModelPerformanceCard } from './components/ModelPerformanceCard';
import { TimestepTrendCard } from './components/TimestepTrendCard';
import { TransactionTable } from './components/TransactionTable';
import { PredictForm } from './components/PredictForm';
import { TransactionDetailModal } from './components/TransactionDetailModal';
import { RecentHighRiskCard } from './components/RecentHighRiskCard';

export default function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'transactions' | 'model' | 'predict'>('dashboard');

  const handleSetActiveTab = (tab: 'dashboard' | 'transactions' | 'model' | 'predict') => {
    if (tab === 'dashboard') {
      setSearchQuery('');
    }
    setActiveTab(tab);
  };

  // Backend State
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardResponse | null>(null);
  const [riskDist, setRiskDist] = useState<RiskDistributionResponse | null>(null);
  const [timesteps, setTimesteps] = useState<TimestepItem[]>([]);
  const [modelMetrics, setModelMetrics] = useState<ModelMetricsResponse | null>(null);
  const [featureImportances, setFeatureImportances] = useState<FeatureImportanceItem[]>([]);

  // Transactions Table State
  const [txData, setTxData] = useState<PaginatedTransactionsResponse | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 20;
  const [riskLevel, setRiskLevel] = useState<string>('ALL');
  const [timestepFilter, setTimestepFilter] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [txLoading, setTxLoading] = useState(false);

  // Selected Transaction Modal
  const [selectedTxId, setSelectedTxId] = useState<number | null>(null);

  // Initial Data Loader
  const loadInitialData = useCallback(async () => {
    try {
      const hData = await api.getHealth();
      setHealth(hData);
    } catch (_) {
      setHealth(null);
    }

    try {
      const [dData, rData, tData, mData, fData] = await Promise.all([
        api.getDashboard(),
        api.getRiskDistribution(),
        api.getTimesteps(),
        api.getModelMetrics(),
        api.getFeatureImportance(20),
      ]);
      setDashboard(dData);
      setRiskDist(rData);
      setTimesteps(tData);
      setModelMetrics(mData);
      setFeatureImportances(fData.features);
    } catch (err) {
      console.error('Failed to load dashboard metrics:', err);
    }
  }, []);

  // Transactions Explorer Loader
  const loadTransactions = useCallback(async () => {
    setTxLoading(true);
    try {
      const data = await api.getTransactions(page, pageSize, riskLevel, timestepFilter ?? undefined, searchQuery);
      setTxData(data);
    } catch (err) {
      console.error('Failed to load transactions:', err);
    } finally {
      setTxLoading(false);
    }
  }, [page, pageSize, riskLevel, timestepFilter, searchQuery]);

  useEffect(() => {
    loadInitialData();
  }, [loadInitialData]);

  useEffect(() => {
    loadTransactions();
  }, [loadTransactions]);

  // Predict Form Target State
  const [predictTxId, setPredictTxId] = useState<number | null>(null);

  // Handlers
  const handleSelectTx = (txId: number) => {
    setSelectedTxId(txId);
  };

  const handlePredictTx = (txId: number) => {
    setPredictTxId(txId);
    setActiveTab('predict');
  };

  const handleSearchFromHeader = (txId: number) => {
    setSelectedTxId(txId);
  };

  const handleSelectTimestepFromTrend = (ts: number) => {
    setTimestepFilter(ts);
    setActiveTab('transactions');
  };

  return (
    <div className="app-container">
      {/* Header Bar */}
      <Header
        health={health}
        onSearch={handleSearchFromHeader}
        activeTab={activeTab}
        setActiveTab={handleSetActiveTab}
      />

      {/* Main Content Area */}
      <main className="main-content">

        {/* TAB 1: DASHBOARD OVERVIEW */}
        {activeTab === 'dashboard' && (
          <>
            <DashboardOverview metrics={dashboard} />

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))', gap: '1.25rem' }}>
              <RiskDistributionCard distribution={riskDist} />
              <PredictForm initialTxId={predictTxId} onSelectTxDetails={handleSelectTx} />
            </div>

            <TimestepTrendCard timesteps={timesteps} onSelectTimestep={handleSelectTimestepFromTrend} />

            <RecentHighRiskCard
              onSelectTx={handleSelectTx}
              onViewAll={() => handleSetActiveTab('transactions')}
            />
          </>
        )}

        {/* TAB 2: TRANSACTIONS EXPLORER */}
        {activeTab === 'transactions' && (
          <TransactionTable
            data={txData}
            page={page}
            pageSize={pageSize}
            riskLevel={riskLevel}
            timestep={timestepFilter}
            search={searchQuery}
            loading={txLoading}
            onPageChange={(p) => setPage(p)}
            onRiskLevelChange={(l) => { setRiskLevel(l); setPage(1); }}
            onTimestepChange={(t) => { setTimestepFilter(t); setPage(1); }}
            onSearchChange={(s) => { setSearchQuery(s); setPage(1); }}
            onSelectTx={handleSelectTx}
            onPredictTx={handlePredictTx}
            onRefresh={loadTransactions}
          />
        )}

        {/* TAB 3: MODEL PERFORMANCE */}
        {activeTab === 'model' && (
          <ModelPerformanceCard metrics={modelMetrics} featureImportances={featureImportances} />
        )}

        {/* TAB 4: LIVE PREDICT TOOL */}
        {activeTab === 'predict' && (
          <PredictForm initialTxId={predictTxId} onSelectTxDetails={handleSelectTx} />
        )}

      </main>

      {/* Transaction Detail & Graph Inspection Modal */}
      <TransactionDetailModal
        txId={selectedTxId}
        onClose={() => setSelectedTxId(null)}
        onSelectTx={handleSelectTx}
      />
    </div>
  );
}
