import React from 'react';
import { Network, Database, ShieldCheck, ShieldAlert, AlertTriangle, Cpu, Target, Award } from 'lucide-react';
import type { DashboardResponse } from '../types/api';

interface DashboardOverviewProps {
  metrics: DashboardResponse | null;
}

export const DashboardOverview: React.FC<DashboardOverviewProps> = ({ metrics }) => {
  if (!metrics) {
    return <div className="card" style={{ padding: '2rem', textAlign: 'center', color: '#72767a' }}>Loading system metrics...</div>;
  }

  const labeledPct = ((metrics.labeled_transactions / metrics.total_transactions) * 100).toFixed(1);
  const edgeRatio = (metrics.total_edges / metrics.total_transactions).toFixed(2);
  const illicitPct = ((metrics.illicit_transactions / metrics.total_transactions) * 100).toFixed(2);
  const criticalPct = ((metrics.critical_risk_transactions / metrics.total_transactions) * 100).toFixed(2);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Metrics Row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>

        {/* Total Transactions */}
        <div className="card" style={{ borderLeft: '4px solid #1e5631', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Transactions</span>
              <h2 style={{ fontSize: '1.875rem', fontWeight: 700, color: '#1a1c1e', marginTop: '0.25rem', marginBottom: '0.375rem', lineHeight: 1 }}>
                {metrics.total_transactions.toLocaleString()}
              </h2>
            </div>
            <div style={{ backgroundColor: '#eaf3ed', color: '#1e5631', padding: '0.625rem', borderRadius: '0.5rem' }}>
              <Database size={20} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #f0eee9', fontSize: '0.75rem' }}>
            <span style={{ color: '#45484c', fontWeight: 500 }}>49 temporal timesteps</span>
            <span style={{ backgroundColor: '#eaf3ed', color: '#1e5631', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', fontWeight: 600, fontSize: '0.6875rem' }}>
              {metrics.labeled_transactions.toLocaleString()} labeled ({labeledPct}%)
            </span>
          </div>
        </div>

        {/* Graph Edges */}
        <div className="card" style={{ borderLeft: '4px solid #45484c', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Graph Edges</span>
              <h2 style={{ fontSize: '1.875rem', fontWeight: 700, color: '#1a1c1e', marginTop: '0.25rem', marginBottom: '0.375rem', lineHeight: 1 }}>
                {metrics.total_edges.toLocaleString()}
              </h2>
            </div>
            <div style={{ backgroundColor: '#f0eee9', color: '#45484c', padding: '0.625rem', borderRadius: '0.5rem' }}>
              <Network size={20} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #f0eee9', fontSize: '0.75rem' }}>
            <span style={{ color: '#45484c', fontWeight: 500 }}>Directed transfer relationships</span>
            <span style={{ backgroundColor: '#f0eee9', color: '#45484c', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', fontWeight: 600, fontSize: '0.6875rem' }}>
              {edgeRatio} edges / tx
            </span>
          </div>
        </div>

        {/* Illicit Transactions */}
        <div className="card" style={{ borderLeft: '4px solid #881337', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Illicit Transactions</span>
              <h2 style={{ fontSize: '1.875rem', fontWeight: 700, color: '#881337', marginTop: '0.25rem', marginBottom: '0.375rem', lineHeight: 1 }}>
                {metrics.illicit_transactions.toLocaleString()}
              </h2>
            </div>
            <div style={{ backgroundColor: '#fff1f2', color: '#881337', padding: '0.625rem', borderRadius: '0.5rem' }}>
              <ShieldAlert size={20} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #f0eee9', fontSize: '0.75rem' }}>
            <span style={{ color: '#45484c', fontWeight: 500 }}>Ground-truth Class 1</span>
            <span style={{ backgroundColor: '#fff1f2', color: '#881337', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', fontWeight: 600, fontSize: '0.6875rem' }}>
              {illicitPct}% of total
            </span>
          </div>
        </div>

        {/* Critical Risk Alerts */}
        <div className="card" style={{ borderLeft: '4px solid #c2410c', padding: '1.25rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Critical Risk Alerts</span>
              <h2 style={{ fontSize: '1.875rem', fontWeight: 700, color: '#c2410c', marginTop: '0.25rem', marginBottom: '0.375rem', lineHeight: 1 }}>
                {metrics.critical_risk_transactions.toLocaleString()}
              </h2>
            </div>
            <div style={{ backgroundColor: '#fff5ed', color: '#c2410c', padding: '0.625rem', borderRadius: '0.5rem' }}>
              <AlertTriangle size={20} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px solid #f0eee9', fontSize: '0.75rem' }}>
            <span style={{ color: '#45484c', fontWeight: 500 }}>Model score ≥ {metrics.current_threshold.toFixed(2)}</span>
            <span style={{ backgroundColor: '#fff5ed', color: '#c2410c', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', fontWeight: 600, fontSize: '0.6875rem' }}>
              {criticalPct}% alert rate
            </span>
          </div>
        </div>

      </div>

      {/* Model Key Indicators Banner */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '1rem'
        }}>
          {/* Header Row: Title & Feature Composition */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: '0.75rem',
            paddingBottom: '0.875rem',
            borderBottom: '1px solid #f0eee9'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ backgroundColor: '#eaf3ed', color: '#1e5631', padding: '0.5rem', borderRadius: '0.375rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Cpu size={20} />
              </div>
              <div>
                <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#1a1c1e', margin: 0, lineHeight: 1.2 }}>
                  Primary Leakage-Safe XGBoost Model
                </h3>
                <div style={{ fontSize: '0.75rem', color: '#72767a', fontWeight: 500, marginTop: '0.25rem' }}>
                  Feature Composition: <span style={{ fontWeight: 600, color: '#45484c' }}>185 total = 165 original + 20 neighborhood</span>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Grid */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: '1rem'
          }}>
            {/* PR-AUC Metric Card */}
            <div style={{
              backgroundColor: '#faf9f6',
              border: '1px solid #e3e0d7',
              borderRadius: '0.5rem',
              padding: '0.875rem 1rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '0.375rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  PR-AUC
                </span>
                <Award size={16} color="#1e5631" />
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1e5631', fontFamily: 'monospace', lineHeight: 1 }}>
                {metrics.model_pr_auc.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.71875rem', color: '#52565a', fontWeight: 500 }}>
                Precision-recall performance
              </div>
            </div>

            {/* ROC-AUC Metric Card */}
            <div style={{
              backgroundColor: '#faf9f6',
              border: '1px solid #e3e0d7',
              borderRadius: '0.5rem',
              padding: '0.875rem 1rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '0.375rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  ROC-AUC
                </span>
                <Target size={16} color="#1e5631" />
              </div>
              <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#1e5631', fontFamily: 'monospace', lineHeight: 1 }}>
                {metrics.model_roc_auc.toFixed(4)}
              </div>
              <div style={{ fontSize: '0.71875rem', color: '#52565a', fontWeight: 500 }}>
                Ranking discrimination
              </div>
            </div>

            {/* Production F1 Metric Card */}
            <div style={{
              backgroundColor: '#faf9f6',
              border: '1px solid #e3e0d7',
              borderRadius: '0.5rem',
              padding: '0.875rem 1rem',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '0.375rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Production F1
                </span>
                <span style={{ backgroundColor: '#fef3c7', color: '#b45309', padding: '0.125rem 0.375rem', borderRadius: '0.25rem', fontWeight: 600, fontSize: '0.6875rem' }}>
                  Threshold {metrics.current_threshold.toFixed(2)}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#b45309', fontFamily: 'monospace', lineHeight: 1 }}>
                  {metrics.model_f1.toFixed(4)}
                </div>
                <ShieldCheck size={16} color="#b45309" />
              </div>
              <div style={{ fontSize: '0.71875rem', color: '#52565a', fontWeight: 500 }}>
                Balance at alert threshold
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
