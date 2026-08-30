import React, { useEffect, useState } from 'react';
import {
  X,
  FileText,
  Activity,
  ChevronRight,
  ShieldAlert,
  Zap,
  Network,
  Scale,
  Info,
  ArrowDownLeft,
  ArrowUpRight,
  Layers
} from 'lucide-react';
import { api } from '../api/client';
import type {
  TransactionDetailResponse,
  NeighborsResponse,
  ExplanationResponse,
  InvestigationResponse
} from '../types/api';
import { InvestigationPriorityBadge } from './InvestigationPriorityBadge';

interface TransactionDetailModalProps {
  txId: number | null;
  onClose: () => void;
  onSelectTx: (txId: number) => void;
}

export const TransactionDetailModal: React.FC<TransactionDetailModalProps> = ({
  txId,
  onClose,
  onSelectTx
}) => {
  const [detail, setDetail] = useState<TransactionDetailResponse | null>(null);
  const [neighbors, setNeighbors] = useState<NeighborsResponse | null>(null);
  const [explanation, setExplanation] = useState<ExplanationResponse | null>(null);
  const [investigation, setInvestigation] = useState<InvestigationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'investigation' | 'overview' | 'neighbors' | 'explanation'>('investigation');

  useEffect(() => {
    if (!txId) {
      setDetail(null);
      setNeighbors(null);
      setExplanation(null);
      setInvestigation(null);
      return;
    }

    const loadTxData = async () => {
      setLoading(true);
      setErrorMsg(null);
      try {
        const [detailData, neighborsData, explanationData, invData] = await Promise.all([
          api.getTransactionDetail(txId),
          api.getTransactionNeighbors(txId),
          api.getTransactionExplanation(txId),
          api.getInvestigation(txId),
        ]);
        setDetail(detailData);
        setNeighbors(neighborsData);
        setExplanation(explanationData);
        setInvestigation(invData);
      } catch (err: any) {
        setErrorMsg(err.message || `Failed to load data for transaction #${txId}`);
      } finally {
        setLoading(false);
      }
    };

    loadTxData();
  }, [txId]);

  // Handle ESC key to close modal
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!txId) return null;

  const getBadgeClass = (level: string) => {
    switch ((level || '').toUpperCase()) {
      case 'CRITICAL': return 'badge-critical';
      case 'HIGH': return 'badge-high';
      case 'MEDIUM': return 'badge-medium';
      case 'LOW': return 'badge-low';
      default: return 'badge-unknown';
    }
  };

  const getScoreColor = (score: number | null | undefined, level: string | undefined) => {
    if (score === null || score === undefined) return '#72767a';
    switch ((level || '').toUpperCase()) {
      case 'CRITICAL': return '#881337';
      case 'HIGH': return '#c2410c';
      case 'MEDIUM': return '#b45309';
      case 'LOW': return '#1e5631';
      default: return '#52565a';
    }
  };

  const incomingNeighbors = neighbors?.neighbors.filter(n => n.relationship === 'incoming') || [];
  const outgoingNeighbors = neighbors?.neighbors.filter(n => n.relationship === 'outgoing') || [];

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(26, 28, 30, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
        padding: '1rem'
      }}
      onClick={onClose}
    >
      <div
        style={{
          backgroundColor: '#ffffff',
          borderRadius: '0.75rem',
          width: '100%',
          maxWidth: '960px',
          maxHeight: '92vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 20px 25px -5px rgba(30, 32, 34, 0.15)',
          overflow: 'hidden'
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div
          style={{
            padding: '1.125rem 1.5rem',
            borderBottom: '1px solid #e3e0d7',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            backgroundColor: '#ffffff'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ backgroundColor: '#1e5631', color: '#ffffff', padding: '0.5rem', borderRadius: '0.5rem', display: 'flex' }}>
              <Activity size={20} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 800, color: '#1a1c1e', fontFamily: 'monospace' }}>
                  Transaction #{txId}
                </h2>
                {detail && (
                  <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#45484c', backgroundColor: '#f0eee9', padding: '0.15rem 0.5rem', borderRadius: '0.25rem' }}>
                    Step #{detail.time_step}
                  </span>
                )}
              </div>
              <div style={{ fontSize: '0.75rem', color: '#72767a', marginTop: '0.15rem' }}>
                GraphGuard AI Investigation Workspace
              </div>
            </div>
          </div>

          <button
            onClick={onClose}
            title="Close modal (Esc)"
            style={{
              background: 'none',
              border: '1px solid #e3e0d7',
              cursor: 'pointer',
              color: '#52565a',
              padding: '0.375rem',
              borderRadius: '0.375rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.15s ease'
            }}
          >
            <X size={18} />
          </button>
        </div>

        {/* Header Risk & Priority Summary Bar (Req 2, 3, 11) */}
        <div
          style={{
            padding: '0.875rem 1.5rem',
            backgroundColor: '#faf9f6',
            borderBottom: '1px solid #e3e0d7'
          }}
        >
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', alignItems: 'center' }}>

            {/* Model Risk Summary */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', marginBottom: '0.15rem' }}>
                <span style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', color: '#72767a', letterSpacing: '0.04em' }}>
                  Model Risk (XGBoost)
                </span>
                <span title="Probability/risk score produced by the trained XGBoost model (0.00–1.00)" style={{ cursor: 'help' }}>
                  <Info size={12} color="#72767a" />
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.375rem', fontWeight: 800, fontFamily: 'monospace', color: getScoreColor(detail?.risk_score, detail?.risk_level) }}>
                  {detail?.risk_score !== null && detail?.risk_score !== undefined ? detail.risk_score.toFixed(4) : 'Unassessed'}
                </span>
                {detail?.risk_level && (
                  <span className={`badge ${getBadgeClass(detail.risk_level)}`}>
                    {detail.risk_level}
                  </span>
                )}
              </div>
            </div>

            {/* Investigation Priority Summary */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', marginBottom: '0.15rem' }}>
                <span style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', color: '#72767a', letterSpacing: '0.04em' }}>
                  Investigation Priority
                </span>
                <span title="Secondary operational triage score used to rank transactions for review (not a model prediction)" style={{ cursor: 'help' }}>
                  <Info size={12} color="#72767a" />
                </span>
              </div>
              {investigation ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <InvestigationPriorityBadge
                    priority={investigation.investigation_priority}
                    score={investigation.investigation_score}
                  />
                </div>
              ) : (
                <span style={{ fontSize: '0.8125rem', color: '#72767a' }}>Loading...</span>
              )}
            </div>

            {/* Target Ground-Truth Class Label */}
            {detail && (
              <div>
                <span style={{ fontSize: '0.6875rem', fontWeight: 700, textTransform: 'uppercase', color: '#72767a', letterSpacing: '0.04em' }}>
                  Ground-Truth Label
                </span>
                <div style={{ fontSize: '0.875rem', fontWeight: 700, marginTop: '0.15rem' }}>
                  <strong style={{ color: detail.class === 'illicit' ? '#881337' : detail.class === 'licit' ? '#1e5631' : '#52565a' }}>
                    {detail.class.toUpperCase()}
                  </strong>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Modal Navigation Tabs (Req 1) */}
        <div
          style={{
            display: 'flex',
            borderBottom: '1px solid #e3e0d7',
            backgroundColor: '#ffffff',
            padding: '0 1.5rem',
            gap: '0.5rem',
            overflowX: 'auto'
          }}
          role="tablist"
        >
          {[
            { id: 'investigation', label: 'Investigation Priority Layer', icon: Zap },
            { id: 'overview', label: 'Overview & Features', icon: Layers },
            { id: 'neighbors', label: `Graph Neighbors (${neighbors ? neighbors.total_neighbors : 0})`, icon: Network },
            { id: 'explanation', label: 'Model Risk Explanation', icon: FileText }
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                role="tab"
                aria-selected={isActive}
                onClick={() => setActiveTab(tab.id as any)}
                style={{
                  padding: '0.75rem 0.875rem',
                  fontWeight: 600,
                  fontSize: '0.8125rem',
                  border: 'none',
                  background: 'none',
                  borderBottom: isActive ? '2px solid #1e5631' : '2px solid transparent',
                  color: isActive ? '#1e5631' : '#52565a',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.375rem',
                  whiteSpace: 'nowrap'
                }}
              >
                <Icon size={14} color={isActive ? '#1e5631' : '#72767a'} />
                {tab.label}
              </button>
            );
          })}
        </div>

        {/* Modal Body */}
        <div style={{ padding: '1.25rem 1.5rem', overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div style={{ textAlign: 'center', padding: '3rem', color: '#72767a', fontSize: '0.875rem' }}>
              Loading transaction investigation workspace...
            </div>
          ) : errorMsg ? (
            <div style={{ color: '#881337', backgroundColor: '#fff1f2', border: '1px solid #fecdd3', padding: '1rem', borderRadius: '0.5rem', fontSize: '0.875rem' }}>
              {errorMsg}
            </div>
          ) : detail && (
            <>
              {/* TAB 1: INVESTIGATION PRIORITY LAYER */}
              {activeTab === 'investigation' && investigation && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

                  {/* Triage Narrative Banner */}
                  <div
                    style={{
                      backgroundColor: '#fff5ed',
                      border: '1px solid #ffedd5',
                      borderLeft: '4px solid #c2410c',
                      color: '#45484c',
                      padding: '0.875rem 1.125rem',
                      borderRadius: '0.5rem',
                      fontSize: '0.8125rem',
                      lineHeight: 1.5
                    }}
                  >
                    <div style={{ fontWeight: 700, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.375rem', color: '#c2410c' }}>
                      <ShieldAlert size={16} color="#c2410c" /> Fraud Analyst Triage Summary
                    </div>
                    {investigation.explanation}
                  </div>

                  {/* Categorized Supporting Evidence Cards (Req 5, 6, 7) */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#45484c' }}>
                      Categorized Multi-Signal Evidence
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '0.75rem' }}>

                      {/* Signal 1: Primary Model Output (50%) */}
                      <div style={{ backgroundColor: '#faf9f6', border: '1px solid #e3e0d7', borderRadius: '0.5rem', padding: '0.875rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                          <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#1e5631', textTransform: 'uppercase' }}>
                            Model Output Signal
                          </span>
                          <span style={{ fontSize: '0.625rem', fontWeight: 700, backgroundColor: '#eaf3ed', color: '#1e5631', padding: '0.1rem 0.4rem', borderRadius: '0.25rem' }}>
                            50% Weight
                          </span>
                        </div>
                        <div style={{ fontSize: '1.375rem', fontWeight: 800, fontFamily: 'monospace', color: '#1a1c1e', margin: '0.15rem 0' }}>
                          {investigation.model_risk !== null ? investigation.model_risk.toFixed(4) : 'Unassessed'}
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#52565a' }}>
                          XGBoost Leakage-Safe Probability
                        </div>
                      </div>

                      {/* Signal 2: Neighborhood Evidence (25%) */}
                      <div style={{ backgroundColor: '#faf9f6', border: '1px solid #e3e0d7', borderRadius: '0.5rem', padding: '0.875rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                          <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#b45309', textTransform: 'uppercase' }}>
                            Neighborhood Evidence
                          </span>
                          <span style={{ fontSize: '0.625rem', fontWeight: 700, backgroundColor: '#fef8ec', color: '#b45309', padding: '0.1rem 0.4rem', borderRadius: '0.25rem' }}>
                            25% Weight
                          </span>
                        </div>
                        <div style={{ fontSize: '1.375rem', fontWeight: 800, fontFamily: 'monospace', color: '#b45309', margin: '0.15rem 0' }}>
                          {investigation.neighborhood_evidence.high_risk_neighbor_fraction !== null && investigation.neighborhood_evidence.high_risk_neighbor_fraction !== undefined
                            ? `${(Number(investigation.neighborhood_evidence.high_risk_neighbor_fraction) * 100).toFixed(1)}%`
                            : investigation.neighborhood_evidence.neighborhood_mean_risk !== null && investigation.neighborhood_evidence.neighborhood_mean_risk !== undefined
                            ? Number(investigation.neighborhood_evidence.neighborhood_mean_risk).toFixed(4)
                            : 'N/A'}
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#52565a' }}>
                          High-Risk Neighbor Fraction / Mean
                        </div>
                      </div>

                      {/* Signal 3: Neighborhood Contrast Signal (15%) */}
                      <div style={{ backgroundColor: '#faf9f6', border: '1px solid #e3e0d7', borderRadius: '0.5rem', padding: '0.875rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                          <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#c2410c', textTransform: 'uppercase' }}>
                            Neighborhood Contrast
                          </span>
                          <span style={{ fontSize: '0.625rem', fontWeight: 700, backgroundColor: '#fff5ed', color: '#c2410c', padding: '0.1rem 0.4rem', borderRadius: '0.25rem' }}>
                            15% Weight
                          </span>
                        </div>
                        <div style={{ fontSize: '1.375rem', fontWeight: 800, fontFamily: 'monospace', color: '#1a1c1e', margin: '0.15rem 0' }}>
                          {investigation.neighborhood_evidence.risk_contrast !== null && investigation.neighborhood_evidence.risk_contrast !== undefined
                            ? Number(investigation.neighborhood_evidence.risk_contrast).toFixed(4)
                            : '0.0000'}
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#52565a' }}>
                          Target vs Local Mean Deviation
                        </div>
                      </div>

                      {/* Signal 4: Graph Structural Evidence (10%) */}
                      <div style={{ backgroundColor: '#faf9f6', border: '1px solid #e3e0d7', borderRadius: '0.5rem', padding: '0.875rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.25rem' }}>
                          <span style={{ fontSize: '0.6875rem', fontWeight: 700, color: '#45484c', textTransform: 'uppercase' }}>
                            Graph Structural Evidence
                          </span>
                          <span style={{ fontSize: '0.625rem', fontWeight: 700, backgroundColor: '#f0eee9', color: '#45484c', padding: '0.1rem 0.4rem', borderRadius: '0.25rem' }}>
                            10% Weight
                          </span>
                        </div>
                        <div style={{ fontSize: '1.125rem', fontWeight: 800, fontFamily: 'monospace', color: '#1a1c1e', margin: '0.25rem 0' }}>
                          Deg: {investigation.graph_evidence.total_degree} | Imb: {investigation.graph_evidence.degree_imbalance}
                        </div>
                        <div style={{ fontSize: '0.6875rem', color: '#52565a' }}>
                          In-Degree: {investigation.graph_evidence.in_degree} | Out-Degree: {investigation.graph_evidence.out_degree}
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Formula Methodology Note (Req 7) */}
                  <div
                    style={{
                      backgroundColor: '#f0eee9',
                      border: '1px solid #d6d3cb',
                      borderRadius: '0.375rem',
                      padding: '0.625rem 0.875rem',
                      fontSize: '0.75rem',
                      color: '#45484c',
                      lineHeight: 1.4
                    }}
                  >
                    <strong>Triage Formula:</strong> <code>Score = 0.50 * model_risk + 0.25 * neighborhood_risk + 0.15 * contrast_signal + 0.10 * graph_signal</code>. Investigation Priority is a secondary operational triage metric for ranking analyst workload and is separate from model risk prediction.
                  </div>

                </div>
              )}

              {/* TAB 2: OVERVIEW & FEATURES (Req 8) */}
              {activeTab === 'overview' && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

                  {/* Topology & Risk Summary Panel */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                      gap: '0.875rem',
                      backgroundColor: '#faf9f6',
                      padding: '1rem',
                      borderRadius: '0.5rem',
                      border: '1px solid #e3e0d7'
                    }}
                  >
                    <div>
                      <span style={{ fontSize: '0.6875rem', color: '#72767a', fontWeight: 700, textTransform: 'uppercase' }}>XGBoost Risk Score</span>
                      <div
                        style={{
                          fontSize: '1.375rem',
                          fontWeight: 800,
                          fontFamily: 'monospace',
                          color: getScoreColor(detail.risk_score, detail.risk_level),
                          marginTop: '0.15rem'
                        }}
                      >
                        {detail.risk_score !== null ? detail.risk_score.toFixed(4) : 'Unassessed'}
                      </div>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.6875rem', color: '#72767a', fontWeight: 700, textTransform: 'uppercase' }}>Risk Level Band</span>
                      <div style={{ marginTop: '0.25rem' }}>
                        <span className={`badge ${getBadgeClass(detail.risk_level)}`}>
                          {detail.risk_level}
                        </span>
                      </div>
                    </div>

                    <div>
                      <span style={{ fontSize: '0.6875rem', color: '#72767a', fontWeight: 700, textTransform: 'uppercase' }}>Degree Topology</span>
                      <div style={{ fontSize: '0.875rem', fontWeight: 700, color: '#1a1c1e', marginTop: '0.25rem' }}>
                        In: {detail.degree_information.in_degree} | Out: {detail.degree_information.out_degree} (Total: {detail.degree_information.total_degree})
                      </div>
                    </div>
                  </div>

                  {/* Group A: Leakage-Safe Neighborhood Features */}
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.5rem', color: '#1a1c1e', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <Network size={15} color="#1e5631" />
                      Leakage-Safe Neighborhood Risk Indicators
                    </h4>
                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Signal Name</th>
                            <th>Value</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(detail.neighborhood_risk_features).map(([k, v]) => (
                            <tr key={k}>
                              <td><code>{k}</code></td>
                              <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                                {v === null ? 'N/A' : typeof v === 'number' ? v.toFixed(5) : v}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Group B: Original Graph Structural Features */}
                  {detail.graph_features && Object.keys(detail.graph_features).length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.5rem', color: '#1a1c1e', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                        <Scale size={15} color="#45484c" />
                        Original Graph Topology Features
                      </h4>
                      <div className="table-container">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Feature Name</th>
                              <th>Value</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(detail.graph_features).map(([k, v]) => (
                              <tr key={k}>
                                <td><code>{k}</code></td>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                                  {v === null ? 'N/A' : typeof v === 'number' ? v.toFixed(5) : v}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                </div>
              )}

              {/* TAB 3: GRAPH NEIGHBORS (Req 9) */}
              {activeTab === 'neighbors' && neighbors && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

                  {/* Summary Ribbon */}
                  <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8125rem', color: '#45484c', backgroundColor: '#faf9f6', padding: '0.625rem 0.875rem', borderRadius: '0.375rem', border: '1px solid #e3e0d7' }}>
                    <span>Total Direct Neighbors: <strong>{neighbors.total_neighbors}</strong></span>
                    <span>• Incoming Sources: <strong>{neighbors.incoming_count}</strong></span>
                    <span>• Outgoing Targets: <strong>{neighbors.outgoing_count}</strong></span>
                  </div>

                  {/* Sub-Section 1: Incoming Neighbors */}
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.5rem', color: '#1a1c1e', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <ArrowDownLeft size={16} color="#1e5631" />
                      Incoming Transaction Sources ({incomingNeighbors.length})
                    </h4>

                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Neighbor Tx ID</th>
                            <th>Relationship</th>
                            <th>Timestep</th>
                            <th>Risk Score</th>
                            <th>Risk Level</th>
                            <th>Class</th>
                            <th style={{ textAlign: 'right' }}>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {incomingNeighbors.length === 0 ? (
                            <tr>
                              <td colSpan={7} style={{ textAlign: 'center', padding: '1.25rem', color: '#72767a', fontSize: '0.8125rem' }}>
                                No incoming transaction sources in graph edgelist.
                              </td>
                            </tr>
                          ) : (
                            incomingNeighbors.map((n) => (
                              <tr key={`in-${n.tx_id}`}>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700, color: '#1a1c1e' }}>#{n.tx_id}</td>
                                <td>
                                  <span className="badge badge-low">
                                    incoming
                                  </span>
                                </td>
                                <td>Step #{n.time_step}</td>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                                  {n.risk_score !== null ? (
                                    <span style={{ color: getScoreColor(n.risk_score, n.risk_level) }}>
                                      {n.risk_score.toFixed(4)}
                                    </span>
                                  ) : (
                                    <span style={{ color: '#72767a' }}>N/A</span>
                                  )}
                                </td>
                                <td>
                                  <span className={`badge ${getBadgeClass(n.risk_level)}`}>
                                    {n.risk_level}
                                  </span>
                                </td>
                                <td>
                                  <strong style={{ color: n.class === 'illicit' ? '#881337' : n.class === 'licit' ? '#1e5631' : '#52565a' }}>
                                    {n.class}
                                  </strong>
                                </td>
                                <td style={{ textAlign: 'right' }}>
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => onSelectTx(n.tx_id)}
                                    title="Inspect this neighbor transaction"
                                  >
                                    Inspect <ChevronRight size={12} />
                                  </button>
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Sub-Section 2: Outgoing Neighbors */}
                  <div>
                    <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.5rem', color: '#1a1c1e', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <ArrowUpRight size={16} color="#b45309" />
                      Outgoing Transaction Targets ({outgoingNeighbors.length})
                    </h4>

                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Neighbor Tx ID</th>
                            <th>Relationship</th>
                            <th>Timestep</th>
                            <th>Risk Score</th>
                            <th>Risk Level</th>
                            <th>Class</th>
                            <th style={{ textAlign: 'right' }}>Action</th>
                          </tr>
                        </thead>
                        <tbody>
                          {outgoingNeighbors.length === 0 ? (
                            <tr>
                              <td colSpan={7} style={{ textAlign: 'center', padding: '1.25rem', color: '#72767a', fontSize: '0.8125rem' }}>
                                No outgoing transaction targets in graph edgelist.
                              </td>
                            </tr>
                          ) : (
                            outgoingNeighbors.map((n) => (
                              <tr key={`out-${n.tx_id}`}>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700, color: '#1a1c1e' }}>#{n.tx_id}</td>
                                <td>
                                  <span className="badge badge-medium">
                                    outgoing
                                  </span>
                                </td>
                                <td>Step #{n.time_step}</td>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                                  {n.risk_score !== null ? (
                                    <span style={{ color: getScoreColor(n.risk_score, n.risk_level) }}>
                                      {n.risk_score.toFixed(4)}
                                    </span>
                                  ) : (
                                    <span style={{ color: '#72767a' }}>N/A</span>
                                  )}
                                </td>
                                <td>
                                  <span className={`badge ${getBadgeClass(n.risk_level)}`}>
                                    {n.risk_level}
                                  </span>
                                </td>
                                <td>
                                  <strong style={{ color: n.class === 'illicit' ? '#881337' : n.class === 'licit' ? '#1e5631' : '#52565a' }}>
                                    {n.class}
                                  </strong>
                                </td>
                                <td style={{ textAlign: 'right' }}>
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => onSelectTx(n.tx_id)}
                                    title="Inspect this neighbor transaction"
                                  >
                                    Inspect <ChevronRight size={12} />
                                  </button>
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                </div>
              )}

              {/* TAB 4: MODEL RISK EXPLANATION (Req 10) */}
              {activeTab === 'explanation' && explanation && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

                  {/* Empirical Narrative Explanation */}
                  <div
                    style={{
                      backgroundColor: '#eaf3ed',
                      border: '1px solid #b8dbc0',
                      borderLeft: '4px solid #1e5631',
                      color: '#1e5631',
                      padding: '0.875rem 1.125rem',
                      borderRadius: '0.5rem',
                      fontSize: '0.8125rem',
                      lineHeight: 1.5
                    }}
                  >
                    <div style={{ fontWeight: 700, marginBottom: '0.25rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
                      <FileText size={16} color="#1e5631" /> Empirical Risk Synthesis Narrative
                    </div>
                    {explanation.explanation_text}
                  </div>

                  {/* Section A: Empirical Neighborhood Signals Observed for this Transaction */}
                  {explanation.important_neighborhood_signals && Object.keys(explanation.important_neighborhood_signals).length > 0 && (
                    <div>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: 700, marginBottom: '0.5rem', color: '#1a1c1e' }}>
                        Transaction-Level Empirical Neighborhood Signals
                      </h4>
                      <div className="table-container">
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Signal Name</th>
                              <th>Observed Metric</th>
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(explanation.important_neighborhood_signals).map(([k, v]) => (
                              <tr key={k}>
                                <td><code>{k}</code></td>
                                <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>
                                  {v === null ? 'N/A' : typeof v === 'number' ? v.toFixed(5) : v}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {/* Section B: Global Model Feature Importance (XGBoost Weight) */}
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '0.5rem' }}>
                      <h4 style={{ fontSize: '0.875rem', fontWeight: 700, color: '#1a1c1e' }}>
                        Global Model Feature Importance (XGBoost Weight)
                      </h4>
                      <span style={{ fontSize: '0.6875rem', color: '#72767a' }}>
                        Model-wide importance weights (not individual transaction attribution)
                      </span>
                    </div>

                    <div className="table-container">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Rank</th>
                            <th>Feature Name</th>
                            <th>Category Group</th>
                            <th>Global Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {explanation.top_contributing_model_features.map((f) => (
                            <tr key={f.feature}>
                              <td style={{ fontWeight: 700 }}>#{f.rank}</td>
                              <td><code>{f.feature}</code></td>
                              <td><span className="badge badge-low">{f.group}</span></td>
                              <td style={{ fontFamily: 'monospace', fontWeight: 700 }}>{f.importance.toFixed(5)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Legal Disclaimer */}
                  <div
                    style={{
                      fontSize: '0.75rem',
                      color: '#72767a',
                      backgroundColor: '#faf9f6',
                      padding: '0.625rem 0.875rem',
                      borderRadius: '0.375rem',
                      border: '1px solid #e3e0d7',
                      lineHeight: 1.4
                    }}
                  >
                    <strong>Disclaimer:</strong> This risk explanation is generated from empirical model features and graph signals for investigation triage. It does not constitute a formal legal declaration of illicit activity.
                  </div>

                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};
