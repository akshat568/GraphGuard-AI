import React from 'react';
import { PieChart, Shield, AlertTriangle, CheckCircle, HelpCircle, Info } from 'lucide-react';
import type { RiskDistributionResponse } from '../types/api';

interface RiskDistributionCardProps {
  distribution: RiskDistributionResponse | null;
}

export const RiskDistributionCard: React.FC<RiskDistributionCardProps> = ({ distribution }) => {
  if (!distribution) {
    return <div className="card">Loading risk distribution...</div>;
  }

  const { counts, ranges, total_assessed } = distribution;
  const unassessedCount = counts.UNASSESSED || 0;
  const totalTxs = total_assessed + unassessedCount;

  const assessedBands = [
    {
      key: 'LOW',
      label: 'Low Risk',
      range: ranges.LOW || '0.00–0.24',
      count: counts.LOW || 0,
      color: '#1e5631',
      bg: '#eaf3ed',
      border: '#b8dbc0',
      icon: CheckCircle
    },
    {
      key: 'MEDIUM',
      label: 'Medium Risk',
      range: ranges.MEDIUM || '0.25–0.49',
      count: counts.MEDIUM || 0,
      color: '#b45309',
      bg: '#fef8ec',
      border: '#fde68a',
      icon: Shield
    },
    {
      key: 'HIGH',
      label: 'High Risk (Review)',
      range: ranges.HIGH || '0.50–0.89',
      count: counts.HIGH || 0,
      color: '#c2410c',
      bg: '#fff5ed',
      border: '#ffedd5',
      icon: AlertTriangle
    },
    {
      key: 'CRITICAL',
      label: 'Critical Alert',
      range: ranges.CRITICAL || '0.90–1.00',
      count: counts.CRITICAL || 0,
      color: '#881337',
      bg: '#fff1f2',
      border: '#fecdd3',
      icon: AlertTriangle,
      isCritical: true
    }
  ];

  const unassessedBand = {
    key: 'UNASSESSED',
    label: 'Unassessed (T1–10)',
    range: 'N/A',
    count: unassessedCount,
    color: '#52565a',
    bg: '#f0eee9',
    border: '#d6d3cb',
    icon: HelpCircle
  };

  const assessedPctOfTotal = totalTxs > 0 ? ((total_assessed / totalTxs) * 100).toFixed(1) : '0';
  const unassessedPctOfTotal = totalTxs > 0 ? ((unassessedCount / totalTxs) * 100).toFixed(1) : '0';

  return (
    <div className="card">
      {/* Card Header with Title and Summary */}
      <div className="card-header" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 className="card-title">
          <PieChart size={18} color="#1e5631" />
          Risk Score Distribution Breakdown
        </h3>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span
            style={{
              fontSize: '0.75rem',
              color: '#45484c',
              fontWeight: 600,
              backgroundColor: '#f0eee9',
              padding: '0.25rem 0.625rem',
              borderRadius: '1rem',
              border: '1px solid #d6d3cb'
            }}
          >
            {total_assessed.toLocaleString()} assessed • {unassessedCount.toLocaleString()} unassessed
          </span>
        </div>
      </div>

      {/* Distribution Visualizations */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1.25rem' }}>

        {/* Track 1: Assessed Risk Spectrum */}
        <div style={{ backgroundColor: '#faf9f6', padding: '0.875rem', borderRadius: '0.5rem', border: '1px solid #e3e0d7' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem', fontSize: '0.75rem', fontWeight: 600, color: '#1a1c1e' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#1e5631' }} />
              Assessed Risk Spectrum ({total_assessed.toLocaleString()} Txs)
            </span>
            <span style={{ color: '#72767a', fontWeight: 500 }}>
              100% of assessed ({assessedPctOfTotal}% of total)
            </span>
          </div>

          <div
            style={{
              display: 'flex',
              height: '16px',
              borderRadius: '8px',
              overflow: 'hidden',
              backgroundColor: '#e6e3da',
              boxShadow: 'inset 0 1px 2px rgba(0, 0, 0, 0.06)'
            }}
          >
            {assessedBands.map((band) => {
              const pctAssessed = total_assessed > 0 ? (band.count / total_assessed) * 100 : 0;
              const pctTotal = totalTxs > 0 ? (band.count / totalTxs) * 100 : 0;
              if (pctAssessed <= 0) return null;
              return (
                <div
                  key={band.key}
                  style={{
                    width: `${pctAssessed}%`,
                    backgroundColor: band.color,
                    transition: 'width 0.4s ease',
                    borderRight: '1px solid rgba(255, 255, 255, 0.3)'
                  }}
                  title={`${band.label}: ${band.count.toLocaleString()} (${pctAssessed.toFixed(1)}% of assessed, ${pctTotal.toFixed(1)}% of total)`}
                />
              );
            })}
          </div>

          {/* Mini Legend for Assessed Track */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', marginTop: '0.5rem', fontSize: '0.6875rem', color: '#45484c' }}>
            {assessedBands.map((band) => {
              const pctAssessed = total_assessed > 0 ? ((band.count / total_assessed) * 100).toFixed(1) : '0';
              return (
                <div key={band.key} style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: band.color }} />
                  <span style={{ fontWeight: 500 }}>{band.label}:</span>
                  <span style={{ color: '#72767a' }}>{pctAssessed}%</span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Track 2: Neutral Unassessed Baseline */}
        <div style={{ backgroundColor: '#f0eee9', padding: '0.75rem 0.875rem', borderRadius: '0.5rem', border: '1px solid #d6d3cb' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.375rem', fontSize: '0.75rem', fontWeight: 600, color: '#52565a' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
              <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#52565a' }} />
              Unassessed Baseline ({unassessedCount.toLocaleString()} Txs)
            </span>
            <span style={{ color: '#72767a', fontWeight: 500 }}>
              {unassessedPctOfTotal}% of total transactions (Timesteps 1–10)
            </span>
          </div>

          <div
            style={{
              display: 'flex',
              height: '8px',
              borderRadius: '4px',
              overflow: 'hidden',
              backgroundColor: '#d6d3cb'
            }}
          >
            <div
              style={{
                width: '100%',
                backgroundColor: '#52565a'
              }}
              title={`Unassessed (T1-10): ${unassessedCount.toLocaleString()} (${unassessedPctOfTotal}% of total)`}
            />
          </div>
        </div>
      </div>

      {/* Cards Layout - Divided into Assessed & Unassessed Groups */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>

        {/* Assessed Cards Group (4 Cards) */}
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#45484c', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <Shield size={14} color="#1e5631" />
            Assessed Risk Categories
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.75rem' }}>
            {assessedBands.map((band) => {
              const Icon = band.icon;
              const pctAssessed = total_assessed > 0 ? ((band.count / total_assessed) * 100).toFixed(1) : '0';
              const pctTotal = totalTxs > 0 ? ((band.count / totalTxs) * 100).toFixed(1) : '0';
              const isCritical = band.isCritical;

              return (
                <div
                  key={band.key}
                  style={{
                    backgroundColor: band.bg,
                    border: `1px solid ${band.border}`,
                    borderLeft: isCritical ? `4px solid ${band.color}` : `1px solid ${band.border}`,
                    borderRadius: '0.5rem',
                    padding: '0.75rem',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between',
                    gap: '0.375rem',
                    boxShadow: isCritical ? '0 2px 4px rgba(136, 19, 55, 0.08)' : 'none',
                    transition: 'transform 0.15s ease, box-shadow 0.15s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: band.color }}>
                      {band.label}
                    </span>
                    <Icon size={16} color={band.color} />
                  </div>

                  <div>
                    <div style={{ fontSize: isCritical ? '1.375rem' : '1.25rem', fontWeight: 800, color: band.color, lineHeight: 1.2 }}>
                      {band.count.toLocaleString()}
                    </div>
                    <div style={{ fontSize: '0.6875rem', fontWeight: 600, color: band.color, opacity: 0.9 }}>
                      {pctAssessed}% of assessed
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.6875rem', color: '#52565a', paddingTop: '0.25rem', borderTop: `1px dashed ${band.border}` }}>
                    <span>Score: {band.range}</span>
                    <span>{pctTotal}% total</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Unassessed Card & Explanatory Note Group */}
        <div>
          <div style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: '#52565a', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <HelpCircle size={14} color="#72767a" />
            Unassessed Category
          </div>

          <div
            style={{
              backgroundColor: unassessedBand.bg,
              border: `1px solid ${unassessedBand.border}`,
              borderRadius: '0.5rem',
              padding: '0.875rem 1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.625rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <HelpCircle size={18} color={unassessedBand.color} />
                <div>
                  <span style={{ fontSize: '0.875rem', fontWeight: 700, color: unassessedBand.color }}>
                    {unassessedBand.label}
                  </span>
                  <span style={{ fontSize: '0.75rem', color: '#72767a', marginLeft: '0.5rem' }}>
                    Score: {unassessedBand.range}
                  </span>
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'baseline', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.25rem', fontWeight: 800, color: unassessedBand.color }}>
                  {unassessedBand.count.toLocaleString()}
                </span>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: '#72767a' }}>
                  ({unassessedPctOfTotal}% of total)
                </span>
              </div>
            </div>

            {/* Explanatory Note for UNASSESSED (Req 6) */}
            <div
              style={{
                backgroundColor: '#ffffff',
                border: '1px solid #d6d3cb',
                borderRadius: '0.375rem',
                padding: '0.625rem 0.75rem',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.5rem',
                fontSize: '0.75rem',
                color: '#45484c',
                lineHeight: 1.4
              }}
            >
              <Info size={16} color="#52565a" style={{ flexShrink: 0, marginTop: '1px' }} />
              <span>
                Timesteps 1–10 are intentionally unassessed because leakage-safe historical neighborhood features are unavailable.
              </span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};
