import React, { useState } from 'react';
import { Calendar, Filter } from 'lucide-react';
import type { TimestepItem } from '../types/api';

interface TimestepTrendCardProps {
  timesteps: TimestepItem[];
  onSelectTimestep?: (timestep: number) => void;
}

export const TimestepTrendCard: React.FC<TimestepTrendCardProps> = ({ timesteps, onSelectTimestep }) => {
  const [hoveredTs, setHoveredTs] = useState<TimestepItem | null>(null);

  if (!timesteps || timesteps.length === 0) {
    return <div className="card">Loading temporal timestep trends...</div>;
  }

  const maxTotal = Math.max(...timesteps.map(t => t.total_transactions));

  const getPartitionInfo = (ts: number) => {
    if (ts <= 34) {
      return {
        partition: 'TRAIN',
        label: 'Train (1–34)',
        shortLabel: 'Train',
        color: '#52565a',
        barColor: '#72767a',
        bg: '#faf9f6',
        borderColor: '#d2cebf'
      };
    }
    if (ts <= 39) {
      return {
        partition: 'VALIDATION',
        label: 'Validation (35–39)',
        shortLabel: 'Val',
        color: '#b45309',
        barColor: '#d97706',
        bg: '#fef8ec',
        borderColor: '#fde68a'
      };
    }
    return {
      partition: 'TEST',
      label: 'Test (40–49)',
      shortLabel: 'Test',
      color: '#1e5631',
      barColor: '#1e5631',
      bg: '#eaf3ed',
      borderColor: '#b8dbc0'
    };
  };

  const trainSteps = timesteps.filter(t => t.timestep >= 1 && t.timestep <= 34);
  const valSteps = timesteps.filter(t => t.timestep >= 35 && t.timestep <= 39);
  const testSteps = timesteps.filter(t => t.timestep >= 40 && t.timestep <= 49);

  const renderBarColumn = (tsItem: TimestepItem) => {
    const partitionInfo = getPartitionInfo(tsItem.timestep);
    const isHovered = hoveredTs?.timestep === tsItem.timestep;
    const barHeight = Math.max(10, (tsItem.total_transactions / maxTotal) * 125);
    const illicitHeight = tsItem.illicit > 0 ? Math.max(3, (tsItem.illicit / maxTotal) * 125) : 0;

    const isKeyLabel = tsItem.timestep === 1 ||
                       tsItem.timestep === 5 ||
                       tsItem.timestep === 10 ||
                       tsItem.timestep === 15 ||
                       tsItem.timestep === 20 ||
                       tsItem.timestep === 25 ||
                       tsItem.timestep === 30 ||
                       tsItem.timestep === 34 ||
                       tsItem.timestep === 35 ||
                       tsItem.timestep === 39 ||
                       tsItem.timestep === 40 ||
                       tsItem.timestep === 45 ||
                       tsItem.timestep === 49;

    return (
      <div
        key={tsItem.timestep}
        onMouseEnter={() => setHoveredTs(tsItem)}
        onMouseLeave={() => setHoveredTs(null)}
        onClick={() => onSelectTimestep && onSelectTimestep(tsItem.timestep)}
        style={{
          flex: 1,
          minWidth: '10px',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'flex-end',
          cursor: 'pointer',
          opacity: hoveredTs && !isHovered ? 0.5 : 1.0,
          transform: isHovered ? 'scaleY(1.02)' : 'none',
          transition: 'opacity 0.15s ease, transform 0.15s ease'
        }}
        title={`Timestep #${tsItem.timestep} (${partitionInfo.label})\nTotal: ${tsItem.total_transactions.toLocaleString()} | Illicit: ${tsItem.illicit.toLocaleString()}`}
      >
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: `${barHeight}px`,
            backgroundColor: isHovered ? partitionInfo.color : partitionInfo.barColor,
            borderRadius: '3px 3px 0 0',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'flex-end',
            overflow: 'hidden',
            boxShadow: isHovered ? '0 0 0 1px #1a1c1e' : 'none'
          }}
        >
          {illicitHeight > 0 && (
            <div
              style={{
                width: '100%',
                height: `${illicitHeight}px`,
                backgroundColor: '#881337'
              }}
            />
          )}
        </div>

        <span
          style={{
            fontSize: '0.5625rem',
            color: isHovered ? '#1a1c1e' : isKeyLabel ? partitionInfo.color : '#72767a',
            fontWeight: isHovered || isKeyLabel ? 700 : 400,
            marginTop: '0.2rem',
            lineHeight: 1
          }}
        >
          {isKeyLabel || isHovered ? tsItem.timestep : '·'}
        </span>
      </div>
    );
  };

  return (
    <div className="card">
      {/* Header with Title and Legend */}
      <div className="card-header" style={{ flexWrap: 'wrap', gap: '0.5rem' }}>
        <h3 className="card-title">
          <Calendar size={18} color="#1e5631" />
          Temporal Distribution Across 49 Timesteps
        </h3>

        {/* Partition Legend */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#52565a', fontWeight: 600 }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#72767a' }} />
            Train (1–34)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#b45309', fontWeight: 600 }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#b45309' }} />
            Validation (35–39)
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#1e5631', fontWeight: 600 }}>
            <span style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: '#1e5631' }} />
            Test (40–49)
          </span>
        </div>
      </div>

      {/* Partition Labels Above Chart */}
      <div style={{ display: 'flex', gap: '2px', marginBottom: '0.375rem', fontSize: '0.6875rem', fontWeight: 700, letterSpacing: '0.04em' }}>
        <div
          style={{
            flex: trainSteps.length || 34,
            backgroundColor: '#faf9f6',
            border: '1px solid #e3e0d7',
            borderBottom: 'none',
            borderRadius: '0.375rem 0.375rem 0 0',
            padding: '0.2rem 0.5rem',
            color: '#52565a',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <span>TRAIN</span>
          <span style={{ fontSize: '0.625rem', color: '#72767a', fontWeight: 500 }}>Timesteps 1–34</span>
        </div>

        <div
          style={{
            flex: valSteps.length || 5,
            backgroundColor: '#fef8ec',
            border: '1px solid #fde68a',
            borderBottom: 'none',
            borderRadius: '0.375rem 0.375rem 0 0',
            padding: '0.2rem 0.5rem',
            color: '#b45309',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <span>VALIDATION</span>
          <span style={{ fontSize: '0.625rem', color: '#b45309', opacity: 0.8, fontWeight: 500 }}>35–39</span>
        </div>

        <div
          style={{
            flex: testSteps.length || 10,
            backgroundColor: '#eaf3ed',
            border: '1px solid #b8dbc0',
            borderBottom: 'none',
            borderRadius: '0.375rem 0.375rem 0 0',
            padding: '0.2rem 0.5rem',
            color: '#1e5631',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}
        >
          <span>TEST</span>
          <span style={{ fontSize: '0.625rem', color: '#1e5631', opacity: 0.8, fontWeight: 500 }}>40–49</span>
        </div>
      </div>

      {/* Main Chart Container with Subtle Partition Boundaries */}
      <div
        style={{
          display: 'flex',
          gap: '2px',
          height: '190px',
          padding: '0.75rem 0.25rem 0.5rem 0.25rem',
          backgroundColor: '#ffffff',
          border: '1px solid #e3e0d7',
          borderRadius: '0 0 0.375rem 0.375rem',
          overflowX: 'auto'
        }}
      >
        {/* Partition 1: TRAIN (T1-34) */}
        <div style={{ flex: trainSteps.length || 34, display: 'flex', gap: '2px', backgroundColor: '#faf9f6', padding: '0.25rem', borderRadius: '0.25rem', borderRight: '1px dashed #d2cebf' }}>
          {trainSteps.map((tsItem) => renderBarColumn(tsItem))}
        </div>

        {/* Partition 2: VALIDATION (T35-39) */}
        <div style={{ flex: valSteps.length || 5, display: 'flex', gap: '2px', backgroundColor: '#fffdfa', padding: '0.25rem', borderRadius: '0.25rem', borderRight: '1px dashed #fde68a' }}>
          {valSteps.map((tsItem) => renderBarColumn(tsItem))}
        </div>

        {/* Partition 3: TEST (T40-49) */}
        <div style={{ flex: testSteps.length || 10, display: 'flex', gap: '2px', backgroundColor: '#f6faf7', padding: '0.25rem', borderRadius: '0.25rem' }}>
          {testSteps.map((tsItem) => renderBarColumn(tsItem))}
        </div>
      </div>

      {/* Hover Info Banner */}
      <div
        style={{
          marginTop: '0.75rem',
          padding: '0.625rem 0.875rem',
          backgroundColor: '#faf9f6',
          borderRadius: '0.375rem',
          border: '1px solid #e3e0d7',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.5rem',
          fontSize: '0.8125rem'
        }}
      >
        {hoveredTs ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span
                style={{
                  fontSize: '0.6875rem',
                  fontWeight: 700,
                  padding: '0.15rem 0.4rem',
                  borderRadius: '0.25rem',
                  backgroundColor: getPartitionInfo(hoveredTs.timestep).bg,
                  color: getPartitionInfo(hoveredTs.timestep).color,
                  border: `1px solid ${getPartitionInfo(hoveredTs.timestep).borderColor}`
                }}
              >
                {getPartitionInfo(hoveredTs.timestep).partition} (T{hoveredTs.timestep})
              </span>
              <span style={{ fontWeight: 700, color: '#1a1c1e' }}>
                Timestep #{hoveredTs.timestep}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
              <span>Total Txs: <strong>{hoveredTs.total_transactions.toLocaleString()}</strong></span>
              <span style={{ color: '#881337' }}>Illicit: <strong>{hoveredTs.illicit.toLocaleString()}</strong></span>
              <span style={{ color: '#1e5631' }}>Licit: <strong>{hoveredTs.licit.toLocaleString()}</strong></span>
              <span style={{ color: '#52565a' }}>Unknown: <strong>{hoveredTs.unknown.toLocaleString()}</strong></span>
            </div>
          </>
        ) : (
          <div style={{ color: '#72767a', fontSize: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
            <Filter size={13} color="#72767a" />
            <span>Hover column to view exact transaction breakdown. Click column to filter Transactions Explorer.</span>
          </div>
        )}
      </div>

      {/* Explanatory Caption (Req 10) */}
      <div style={{ marginTop: '0.5rem', textAlign: 'center', fontSize: '0.75rem', color: '#72767a' }}>
        Temporal split used for leakage-safe evaluation: Train 1–34 • Validation 35–39 • Test 40–49
      </div>
    </div>
  );
};
