import React, { useState, useEffect } from 'react';
import { Cpu, CheckCircle2, AlertTriangle, Play, HelpCircle, ShieldCheck } from 'lucide-react';
import { api } from '../api/client';
import type { PredictResponse } from '../types/api';

interface PredictFormProps {
  initialTxId?: number | null;
  onSelectTxDetails?: (txId: number) => void;
}

export const PredictForm: React.FC<PredictFormProps> = ({ initialTxId, onSelectTxDetails }) => {
  const [txIdInput, setTxIdInput] = useState(initialTxId ? initialTxId.toString() : '');
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [errorState, setErrorState] = useState<{ isUnassessed: boolean; message: string } | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (initialTxId) {
      setTxIdInput(initialTxId.toString());
      runPrediction(initialTxId);
    }
  }, [initialTxId]);

  const runPrediction = async (parsedId: number) => {
    setLoading(true);
    setErrorState(null);

    try {
      const res = await api.predictTransaction(parsedId);
      setResult(res);
    } catch (err: any) {
      setResult(null);
      const is422 = err.status === 422 || (err.message && err.message.toLowerCase().includes('422'));

      if (is422) {
        setErrorState({
          isUnassessed: true,
          message: 'Prediction unavailable for this transaction because leakage-safe neighborhood risk features are not available for timesteps 1–10.'
        });
      } else if (err.status === 404 || (err.message && err.message.includes('404'))) {
        setErrorState({
          isUnassessed: false,
          message: `Transaction ID #${parsedId} not found in the dataset. Please enter a valid transaction ID.`
        });
      } else {
        setErrorState({
          isUnassessed: false,
          message: 'Unable to complete prediction request. Please verify the backend connection.'
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseInt(txIdInput.trim(), 10);
    if (isNaN(parsed) || parsed <= 0) {
      setErrorState({
        isUnassessed: false,
        message: 'Please enter a valid numeric transaction ID.'
      });
      return;
    }
    await runPrediction(parsed);
  };

  const getRiskBadgeClass = (level: string) => {
    switch (level.toUpperCase()) {
      case 'CRITICAL': return 'badge-critical';
      case 'HIGH': return 'badge-high';
      case 'MEDIUM': return 'badge-medium';
      case 'LOW': return 'badge-low';
      default: return 'badge-unknown';
    }
  };

  const getRiskColor = (level: string) => {
    switch (level.toUpperCase()) {
      case 'CRITICAL': return '#881337';
      case 'HIGH': return '#c2410c';
      case 'MEDIUM': return '#b45309';
      case 'LOW': return '#1e5631';
      default: return '#52565a';
    }
  };

  const isCritical = result?.risk_level.toUpperCase() === 'CRITICAL';

  return (
    <div className="card">
      {/* Header with Title and Subtitle */}
      <div className="card-header" style={{ flexDirection: 'column', alignItems: 'flex-start', gap: '0.25rem' }}>
        <h3 className="card-title">
          <Cpu size={18} color="#1e5631" />
          Live Transaction Predictor
        </h3>
        <span style={{ fontSize: '0.75rem', color: '#72767a', fontWeight: 500 }}>
          Run the production Leakage-Safe Neighborhood XGBoost model on an assessed transaction.
        </span>
      </div>

      {/* Input Form */}
      <form onSubmit={handlePredict} style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
        <div style={{ flex: 1, minWidth: '220px' }}>
          <input
            type="number"
            className="input"
            placeholder="Enter Transaction ID..."
            value={txIdInput}
            onChange={(e) => setTxIdInput(e.target.value)}
          />
        </div>
        <button
          type="submit"
          className="btn btn-primary"
          disabled={loading || !txIdInput.trim()}
          style={{ whiteSpace: 'nowrap' }}
        >
          {loading ? (
            <span>Predicting...</span>
          ) : (
            <>
              <Play size={15} /> Predict Risk
            </>
          )}
        </button>
      </form>

      {/* HTTP 422 Unassessed / Error Info Banners */}
      {errorState && (
        <div
          style={{
            backgroundColor: errorState.isUnassessed ? '#f0eee9' : '#fff1f2',
            border: `1px solid ${errorState.isUnassessed ? '#d6d3cb' : '#fecdd3'}`,
            color: errorState.isUnassessed ? '#45484c' : '#881337',
            padding: '0.875rem 1rem',
            borderRadius: '0.5rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.75rem',
            fontSize: '0.8125rem',
            marginBottom: '1rem',
            lineHeight: 1.4
          }}
        >
          {errorState.isUnassessed ? (
            <HelpCircle size={18} color="#52565a" style={{ flexShrink: 0, marginTop: '2px' }} />
          ) : (
            <AlertTriangle size={18} color="#881337" style={{ flexShrink: 0, marginTop: '2px' }} />
          )}
          <div>
            <div style={{ fontWeight: 700, marginBottom: '0.2rem', color: errorState.isUnassessed ? '#1a1c1e' : '#881337' }}>
              {errorState.isUnassessed ? 'Intentionally Unassessed Transaction' : 'Prediction Error'}
            </div>
            <div>{errorState.message}</div>
          </div>
        </div>
      )}

      {/* Prediction Result Display */}
      {result && (
        <div
          style={{
            backgroundColor: isCritical ? '#fff1f2' : '#faf9f6',
            border: `1px solid ${isCritical ? '#fecdd3' : '#e3e0d7'}`,
            borderLeft: isCritical ? '4px solid #881337' : `1px solid ${isCritical ? '#fecdd3' : '#e3e0d7'}`,
            borderRadius: '0.5rem',
            padding: '1.125rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem'
          }}
        >
          {/* Result Header */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <CheckCircle2 size={18} color={getRiskColor(result.risk_level)} />
              <span style={{ fontWeight: 700, fontSize: '1rem', color: '#1a1c1e' }}>
                Transaction #{result.tx_id}
              </span>
            </div>
            <span className={`badge ${getRiskBadgeClass(result.risk_level)}`} style={{ fontSize: '0.8125rem', padding: '0.25rem 0.625rem' }}>
              {result.risk_level}
            </span>
          </div>

          {/* Result Metrics Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '1rem', borderTop: `1px solid ${isCritical ? '#fecdd3' : '#e3e0d7'}`, paddingTop: '0.875rem' }}>
            <div>
              <span style={{ fontSize: '0.75rem', color: '#72767a', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.025em' }}>
                Risk Score
              </span>
              <div
                style={{
                  fontSize: '1.75rem',
                  fontWeight: 800,
                  fontFamily: 'monospace',
                  color: getRiskColor(result.risk_level),
                  lineHeight: 1.2
                }}
              >
                {result.risk_score.toFixed(4)}
              </div>
            </div>

            <div>
              <span style={{ fontSize: '0.75rem', color: '#72767a', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.025em' }}>
                Alert Status
              </span>
              <div style={{ fontSize: '0.9375rem', fontWeight: 700, color: '#1a1c1e', marginTop: '0.25rem' }}>
                {result.prediction}
              </div>
            </div>
          </div>

          {/* Critical Threshold Notice */}
          {isCritical && (
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #fecdd3',
                borderRadius: '0.375rem',
                padding: '0.5rem 0.75rem',
                fontSize: '0.75rem',
                color: '#881337',
                display: 'flex',
                alignItems: 'center',
                gap: '0.375rem',
                fontWeight: 600
              }}
            >
              <AlertTriangle size={14} color="#881337" />
              <span>Production alert threshold: 0.90</span>
            </div>
          )}

          {/* Inspect Action Link */}
          {onSelectTxDetails && (
            <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '0.25rem' }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => onSelectTxDetails(result.tx_id)}
              >
                Inspect Graph Neighbors & Details →
              </button>
            </div>
          )}
        </div>
      )}

      {/* Trust / Context System Behavior Note (Req 16) */}
      <div
        style={{
          fontSize: '0.75rem',
          color: '#72767a',
          marginTop: '0.875rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.375rem',
          borderTop: '1px solid #e3e0d7',
          paddingTop: '0.625rem'
        }}
      >
        <ShieldCheck size={14} color="#72767a" style={{ flexShrink: 0 }} />
        <span>Prediction uses the trained production model; no online retraining is performed.</span>
      </div>
    </div>
  );
};
