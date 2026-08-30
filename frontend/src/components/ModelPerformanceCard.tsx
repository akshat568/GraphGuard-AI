import React from 'react';
import { Target, Award, ListFilter } from 'lucide-react';
import type { ModelMetricsResponse, FeatureImportanceItem } from '../types/api';

interface ModelPerformanceCardProps {
  metrics: ModelMetricsResponse | null;
  featureImportances: FeatureImportanceItem[];
}

export const ModelPerformanceCard: React.FC<ModelPerformanceCardProps> = ({ metrics, featureImportances }) => {
  if (!metrics) {
    return <div className="card">Loading model performance metrics...</div>;
  }

  const maxImp = featureImportances.length > 0 ? Math.max(...featureImportances.map(f => f.importance)) : 1.0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

      {/* Metrics Cards Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>

        {/* Test Performance */}
        <div className="card">
          <div className="card-header">
            <h4 className="card-title" style={{ fontSize: '0.9375rem' }}>
              <Award size={16} color="#1e5631" />
              Test Set Evaluation
            </h4>
            <span className="badge badge-low">PR-AUC {metrics.test_pr_auc.toFixed(4)}</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.8125rem' }}>
            <div>
              <span style={{ color: '#72767a' }}>ROC-AUC:</span>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#1a1c1e' }}>{metrics.test_roc_auc.toFixed(4)}</div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>Baseline F1:</span>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#1a1c1e' }}>{metrics.test_f1.toFixed(4)}</div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>Precision:</span>
              <div style={{ fontWeight: 600, color: '#45484c' }}>{(metrics.test_precision * 100).toFixed(1)}%</div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>Recall:</span>
              <div style={{ fontWeight: 600, color: '#45484c' }}>{(metrics.test_recall * 100).toFixed(1)}%</div>
            </div>
          </div>
        </div>

        {/* Production Alert Threshold */}
        <div className="card" style={{ borderLeft: '4px solid #b45309' }}>
          <div className="card-header">
            <h4 className="card-title" style={{ fontSize: '0.9375rem' }}>
              <Target size={16} color="#b45309" />
              Production Alert Threshold ({metrics.production_threshold.toFixed(2)})
            </h4>
            <span className="badge badge-medium">Alert Threshold</span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.8125rem' }}>
            <div>
              <span style={{ color: '#72767a' }}>Alert Precision:</span>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#1e5631' }}>
                {(metrics.threshold_precision * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>Alert F1 Score:</span>
              <div style={{ fontSize: '1.125rem', fontWeight: 700, color: '#b45309' }}>
                {metrics.threshold_f1.toFixed(4)}
              </div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>Alert Recall:</span>
              <div style={{ fontWeight: 600, color: '#45484c' }}>{(metrics.threshold_recall * 100).toFixed(1)}%</div>
            </div>
            <div>
              <span style={{ color: '#72767a' }}>False Positives:</span>
              <div style={{ fontWeight: 600, color: '#1e5631' }}>11 (Minimal)</div>
            </div>
          </div>
        </div>

      </div>

      {/* Feature Importance Table */}
      <div className="card">
        <div className="card-header">
          <h3 className="card-title">
            <ListFilter size={18} color="#1e5631" />
            Top Feature Importances (Leakage-Safe Neighborhood XGBoost)
          </h3>
          <span style={{ fontSize: '0.75rem', color: '#72767a' }}>Total 185 Features</span>
        </div>

        <div className="table-container">
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: '60px' }}>Rank</th>
                <th>Feature Name</th>
                <th>Group</th>
                <th>Importance</th>
                <th style={{ width: '220px' }}>Relative Weight</th>
              </tr>
            </thead>
            <tbody>
              {featureImportances.map((item) => {
                const barWidth = Math.max(4, (item.importance / maxImp) * 100);
                const isNeighborhood = item.group === 'neighborhood';
                return (
                  <tr key={item.feature}>
                    <td style={{ fontWeight: 600, color: '#72767a' }}>#{item.rank}</td>
                    <td style={{ fontWeight: isNeighborhood ? 700 : 500, color: isNeighborhood ? '#1e5631' : '#1a1c1e' }}>
                      <code>{item.feature}</code>
                    </td>
                    <td>
                      <span className={`badge ${isNeighborhood ? 'badge-low' : 'badge-unknown'}`}>
                        {item.group}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600, fontFamily: 'monospace' }}>
                      {item.importance.toFixed(5)}
                    </td>
                    <td>
                      <div style={{
                        height: '8px',
                        backgroundColor: '#e6e3da',
                        borderRadius: '4px',
                        overflow: 'hidden'
                      }}>
                        <div style={{
                          height: '100%',
                          width: `${barWidth}%`,
                          backgroundColor: isNeighborhood ? '#1e5631' : '#45484c',
                          borderRadius: '4px'
                        }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
};
