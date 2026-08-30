import React from 'react';
import { Table, ChevronLeft, ChevronRight, Eye, ShieldAlert, RefreshCw, Search, X, Info } from 'lucide-react';
import type { PaginatedTransactionsResponse, TransactionSummary } from '../types/api';

interface TransactionTableProps {
  data: PaginatedTransactionsResponse | null;
  page: number;
  pageSize: number;
  riskLevel: string;
  timestep: number | null;
  search: string;
  loading: boolean;
  onPageChange: (newPage: number) => void;
  onRiskLevelChange: (level: string) => void;
  onTimestepChange: (ts: number | null) => void;
  onSearchChange: (val: string) => void;
  onSelectTx: (txId: number) => void;
  onPredictTx: (txId: number) => void;
  onRefresh: () => void;
}

export const TransactionTable: React.FC<TransactionTableProps> = ({
  data,
  page,
  pageSize,
  riskLevel,
  timestep,
  search,
  loading,
  onPageChange,
  onRiskLevelChange,
  onTimestepChange,
  onSearchChange,
  onSelectTx,
  onPredictTx,
  onRefresh
}) => {
  const getBadgeClass = (level: string) => {
    switch (level.toUpperCase()) {
      case 'CRITICAL': return 'badge-critical';
      case 'HIGH': return 'badge-high';
      case 'MEDIUM': return 'badge-medium';
      case 'LOW': return 'badge-low';
      default: return 'badge-unknown';
    }
  };

  const getScoreColor = (score: number | null, level: string) => {
    if (score === null) return '#72767a';
    switch (level.toUpperCase()) {
      case 'CRITICAL': return '#881337';
      case 'HIGH': return '#c2410c';
      case 'MEDIUM': return '#b45309';
      case 'LOW': return '#1e5631';
      default: return '#52565a';
    }
  };

  const hasActiveFilters = riskLevel !== 'ALL' || timestep !== null || search.trim() !== '';

  const handleResetFilters = () => {
    onRiskLevelChange('ALL');
    onTimestepChange(null);
    onSearchChange('');
  };

  const startItem = data && data.transactions.length > 0 ? (data.page - 1) * pageSize + 1 : 0;
  const endItem = data ? Math.min(data.page * pageSize, data.total) : 0;
  const totalCount = data ? data.total : 0;

  return (
    <div className="card">
      {/* Header */}
      <div className="card-header" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 className="card-title">
          <Table size={18} color="#1e5631" />
          Transactions Explorer
        </h3>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button className="btn btn-secondary btn-sm" onClick={onRefresh} disabled={loading}>
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
            Refresh Data
          </button>
        </div>
      </div>

      {/* Filter & Search Toolbar */}
      <div
        style={{
          backgroundColor: '#faf9f6',
          border: '1px solid #e3e0d7',
          borderRadius: '0.5rem',
          padding: '0.75rem 1rem',
          marginBottom: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.75rem'
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem'
          }}
        >
          {/* Controls Group */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', flex: 1 }}>
            {/* Search Input */}
            <div style={{ position: 'relative', minWidth: '220px', flex: 1, maxWidth: '320px' }}>
              <Search size={15} color="#72767a" style={{ position: 'absolute', left: '0.625rem', top: '50%', transform: 'translateY(-50%)' }} />
              <input
                type="text"
                className="input"
                style={{ paddingLeft: '2rem' }}
                placeholder="Search by Tx ID..."
                value={search}
                onChange={(e) => onSearchChange(e.target.value)}
              />
            </div>

            {/* Risk Level Select */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#45484c', whiteSpace: 'nowrap' }}>Risk Level:</span>
              <select
                className="select"
                value={riskLevel}
                onChange={(e) => onRiskLevelChange(e.target.value)}
              >
                <option value="ALL">All Risk Bands</option>
                <option value="CRITICAL">CRITICAL (≥ 0.90 Alert)</option>
                <option value="HIGH">HIGH (0.50 – 0.89 Review)</option>
                <option value="MEDIUM">MEDIUM (0.25 – 0.49)</option>
                <option value="LOW">LOW (0.00 – 0.24)</option>
                <option value="UNKNOWN">UNASSESSED (T1–10)</option>
              </select>
            </div>

            {/* Timestep Select */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ fontSize: '0.8125rem', fontWeight: 600, color: '#45484c', whiteSpace: 'nowrap' }}>Timestep:</span>
              <select
                className="select"
                value={timestep !== null ? timestep : 'ALL'}
                onChange={(e) => {
                  const val = e.target.value;
                  onTimestepChange(val === 'ALL' ? null : parseInt(val, 10));
                }}
              >
                <option value="ALL">All 49 Timesteps</option>
                {Array.from({ length: 49 }, (_, i) => i + 1).map((ts) => (
                  <option key={ts} value={ts}>Timestep #{ts}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Reset Filters Action */}
          {hasActiveFilters && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={handleResetFilters}
              style={{ color: '#881337', borderColor: '#fecdd3', backgroundColor: '#fff1f2' }}
            >
              <X size={14} color="#881337" /> Clear Filters
            </button>
          )}
        </div>

        {/* Real API Pagination Summary Bar */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', color: '#52565a', borderTop: '1px solid #e3e0d7', paddingTop: '0.5rem' }}>
          <div>
            Showing {startItem.toLocaleString()}–{endItem.toLocaleString()} of {totalCount.toLocaleString()} transactions
          </div>
          {hasActiveFilters && (
            <div style={{ fontWeight: 600, color: '#1e5631' }}>
              Filter Applied
            </div>
          )}
        </div>
      </div>

      {/* Table Container (Responsive Horizontal Scroll) */}
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Timestep</th>
              <th>Risk Score</th>
              <th>Risk Level</th>
              <th>Graph Activity (In/Out)</th>
              <th style={{ textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '2.5rem', color: '#72767a' }}>
                  Loading transactions from dataset index...
                </td>
              </tr>
            ) : !data || data.transactions.length === 0 ? (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', padding: '2.5rem', color: '#72767a' }}>
                  No transactions match the current filters.
                </td>
              </tr>
            ) : (
              data.transactions.map((tx: TransactionSummary) => {
                const isCritical = tx.risk_level.toUpperCase() === 'CRITICAL';

                return (
                  <tr
                    key={tx.tx_id}
                    style={{
                      backgroundColor: isCritical ? '#fff1f2' : 'transparent',
                      borderLeft: isCritical ? '3px solid #881337' : 'none'
                    }}
                  >
                    {/* Transaction ID */}
                    <td style={{ fontWeight: 700, fontFamily: 'monospace', color: '#1a1c1e', letterSpacing: '-0.01em' }}>
                      {tx.tx_id}
                    </td>

                    {/* Timestep */}
                    <td>
                      <span style={{ fontWeight: 500, color: '#45484c' }}>Step #{tx.time_step}</span>
                    </td>

                    {/* Risk Score */}
                    <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                      {tx.risk_score !== null ? (
                        <span style={{ color: getScoreColor(tx.risk_score, tx.risk_level) }}>
                          {tx.risk_score.toFixed(4)}
                        </span>
                      ) : (
                        <span style={{ color: '#72767a', fontWeight: 500 }}>Unassessed</span>
                      )}
                    </td>

                    {/* Risk Level */}
                    <td>
                      <span className={`badge ${getBadgeClass(tx.risk_level)}`}>
                        {tx.risk_level}
                      </span>
                    </td>

                    {/* Graph Activity */}
                    <td style={{ fontSize: '0.8125rem', color: '#45484c' }}>
                      In: <strong>{tx.in_degree}</strong> | Out: <strong>{tx.out_degree}</strong>
                    </td>

                    {/* Actions */}
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', gap: '0.375rem' }}>
                        <button
                          className="btn btn-secondary btn-sm"
                          onClick={() => onSelectTx(tx.tx_id)}
                          title="Inspect existing details and graph neighborhood"
                        >
                          <Eye size={14} /> Details
                        </button>
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => onPredictTx(tx.tx_id)}
                          title="Run model inference on this transaction"
                        >
                          <ShieldAlert size={14} /> Predict
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination Footer & Contextual Note */}
      <div style={{ marginTop: '0.875rem' }}>
        {data && (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingTop: '0.75rem',
              borderTop: '1px solid #e3e0d7',
              flexWrap: 'wrap',
              gap: '0.75rem',
              fontSize: '0.8125rem',
              color: '#52565a'
            }}
          >
            <div>
              Showing {startItem.toLocaleString()}–{endItem.toLocaleString()} of {totalCount.toLocaleString()} transactions
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <button
                className="btn btn-secondary btn-sm"
                disabled={page <= 1 || loading}
                onClick={() => onPageChange(page - 1)}
              >
                <ChevronLeft size={14} /> Prev
              </button>
              <span style={{ fontWeight: 600, color: '#1a1c1e' }}>
                Page {data.page} of {data.total_pages}
              </span>
              <button
                className="btn btn-secondary btn-sm"
                disabled={page >= data.total_pages || loading}
                onClick={() => onPageChange(page + 1)}
              >
                Next <ChevronRight size={14} />
              </button>
            </div>
          </div>
        )}

        {/* Subtle Contextual Note (Req 20) */}
        <div
          style={{
            fontSize: '0.75rem',
            color: '#72767a',
            marginTop: '0.625rem',
            display: 'flex',
            alignItems: 'center',
            gap: '0.375rem'
          }}
        >
          <Info size={14} color="#72767a" style={{ flexShrink: 0 }} />
          <span>Risk scores are available only for transactions with leakage-safe neighborhood features.</span>
        </div>
      </div>
    </div>
  );
};
