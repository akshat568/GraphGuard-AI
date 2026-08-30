import React, { useEffect, useState } from 'react';
import { ShieldAlert, ArrowRight, Eye, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import type { TransactionSummary } from '../types/api';

interface RecentHighRiskCardProps {
  onSelectTx: (txId: number) => void;
  onViewAll: () => void;
}

export const RecentHighRiskCard: React.FC<RecentHighRiskCardProps> = ({ onSelectTx, onViewAll }) => {
  const [transactions, setTransactions] = useState<TransactionSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchRecentHighRisk = async () => {
    setLoading(true);
    try {
      const res = await api.getTransactions(1, 5, 'CRITICAL');
      let txs = res.transactions;
      if (txs.length < 5) {
        const highRes = await api.getTransactions(1, 5 - txs.length, 'HIGH');
        txs = [...txs, ...highRes.transactions];
      }
      setTransactions(txs.slice(0, 5));
    } catch (err) {
      console.error('Failed to fetch recent high-risk transactions:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecentHighRisk();
  }, []);

  const getBadgeClass = (level: string) => {
    switch (level.toUpperCase()) {
      case 'CRITICAL': return 'badge-critical';
      case 'HIGH': return 'badge-high';
      case 'MEDIUM': return 'badge-medium';
      case 'LOW': return 'badge-low';
      default: return 'badge-unknown';
    }
  };

  return (
    <div className="card">
      <div className="card-header" style={{ marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <ShieldAlert size={18} color="#881337" />
          <h3 className="card-title" style={{ margin: 0 }}>
            Recent High-Risk Transactions
          </h3>
          <span style={{ fontSize: '0.75rem', color: '#72767a', fontWeight: 500, marginLeft: '0.5rem' }}>
            Top priority for investigator triage
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button
            className="btn btn-secondary btn-sm"
            onClick={fetchRecentHighRisk}
            disabled={loading}
            title="Refresh recent high-risk transactions"
          >
            <RefreshCw size={14} className={loading ? 'spin' : ''} />
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={onViewAll}
            style={{ color: '#1e5631', fontWeight: 600 }}
          >
            View All <ArrowRight size={14} style={{ marginLeft: '2px' }} />
          </button>
        </div>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Timestep</th>
              <th>Risk Score</th>
              <th>Risk Level</th>
              <th style={{ textAlign: 'right' }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '1.5rem', color: '#72767a' }}>
                  Loading recent high-risk transactions...
                </td>
              </tr>
            ) : transactions.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ textAlign: 'center', padding: '1.5rem', color: '#72767a' }}>
                  No high-risk transactions found.
                </td>
              </tr>
            ) : (
              transactions.map((tx) => (
                <tr key={tx.tx_id}>
                  <td style={{ fontWeight: 600, fontFamily: 'monospace', color: '#1a1c1e' }}>
                    {tx.tx_id}
                  </td>
                  <td>
                    <span style={{ fontWeight: 500, color: '#45484c' }}>Step #{tx.time_step}</span>
                  </td>
                  <td style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                    {tx.risk_score !== null ? (
                      <span style={{ color: tx.risk_score >= 0.90 ? '#881337' : tx.risk_score >= 0.50 ? '#c2410c' : '#1e5631' }}>
                        {tx.risk_score.toFixed(4)}
                      </span>
                    ) : (
                      <span style={{ color: '#72767a' }}>Unassessed</span>
                    )}
                  </td>
                  <td>
                    <span className={`badge ${getBadgeClass(tx.risk_level)}`}>
                      {tx.risk_level}
                    </span>
                  </td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => onSelectTx(tx.tx_id)}
                      title="View transaction details and graph evidence"
                    >
                      <Eye size={14} /> Details
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
