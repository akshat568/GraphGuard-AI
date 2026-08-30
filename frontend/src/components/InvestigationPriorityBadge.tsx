import React from 'react';
import { AlertCircle, AlertTriangle, ShieldCheck, HelpCircle } from 'lucide-react';

interface InvestigationPriorityBadgeProps {
  priority: string | undefined;
  score?: number;
  showScore?: boolean;
}

export const InvestigationPriorityBadge: React.FC<InvestigationPriorityBadgeProps> = ({ priority, score, showScore = true }) => {
  const p = (priority || 'UNASSESSED').toUpperCase();

  const getStyle = () => {
    switch (p) {
      case 'IMMEDIATE':
        return {
          bg: '#fff1f2',
          text: '#881337',
          border: '#fecdd3',
          icon: AlertCircle,
          label: 'IMMEDIATE PRIORITY'
        };
      case 'HIGH':
        return {
          bg: '#fff5ed',
          text: '#c2410c',
          border: '#ffedd5',
          icon: AlertTriangle,
          label: 'HIGH PRIORITY'
        };
      case 'REVIEW':
        return {
          bg: '#fef8ec',
          text: '#b45309',
          border: '#fde68a',
          icon: AlertTriangle,
          label: 'ANALYST REVIEW'
        };
      case 'LOW':
        return {
          bg: '#eaf3ed',
          text: '#1e5631',
          border: '#b8dbc0',
          icon: ShieldCheck,
          label: 'LOW PRIORITY'
        };
      default:
        return {
          bg: '#f0eee9',
          text: '#52565a',
          border: '#d6d3cb',
          icon: HelpCircle,
          label: 'UNASSESSED'
        };
    }
  };

  const style = getStyle();
  const Icon = style.icon;

  return (
    <span
      style={{
        backgroundColor: style.bg,
        color: style.text,
        border: `1px solid ${style.border}`,
        padding: '0.25rem 0.625rem',
        borderRadius: '0.375rem',
        fontSize: '0.75rem',
        fontWeight: 700,
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.375rem',
        letterSpacing: '0.025em',
        textTransform: 'uppercase'
      }}
    >
      <Icon size={14} />
      <span>{style.label}</span>
      {showScore && score !== undefined && (
        <span style={{ fontFamily: 'monospace', marginLeft: '0.25rem', opacity: 0.9 }}>
          ({score.toFixed(4)})
        </span>
      )}
    </span>
  );
};
