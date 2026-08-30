import React, { useState } from 'react';
import { ShieldAlert, Search, Activity } from 'lucide-react';
import type { HealthResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
  onSearch: (txId: number) => void;
  activeTab: 'dashboard' | 'transactions' | 'model' | 'predict';
  setActiveTab: (tab: 'dashboard' | 'transactions' | 'model' | 'predict') => void;
}

export const Header: React.FC<HeaderProps> = ({ onSearch, activeTab, setActiveTab }) => {
  const [searchInput, setSearchInput] = useState('');

  React.useEffect(() => {
    if (activeTab === 'dashboard') {
      setSearchInput('');
    }
  }, [activeTab]);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsed = parseInt(searchInput.trim(), 10);
    if (!isNaN(parsed) && parsed > 0) {
      onSearch(parsed);
      setSearchInput('');
    }
  };

  return (
    <header style={{
      backgroundColor: '#ffffff',
      borderBottom: '1px solid #e3e0d7',
      position: 'sticky',
      top: 0,
      zIndex: 100,
      boxShadow: '0 1px 3px rgba(30, 32, 34, 0.05)'
    }}>
      <div style={{
        maxWidth: '1600px',
        margin: '0 auto',
        padding: '0.75rem 1.5rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '1.25rem'
      }}>
        {/* Left Group: Brand Title + Navigation Tabs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', flexWrap: 'wrap' }}>
          {/* Brand Title */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{
              backgroundColor: '#1e5631',
              color: '#ffffff',
              padding: '0.5rem',
              borderRadius: '0.5rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}>
              <ShieldAlert size={22} />
            </div>
            <div>
              <h1 style={{ fontSize: '1.25rem', fontWeight: 700, color: '#1a1c1e', lineHeight: 1.1 }}>
                GraphGuard AI
              </h1>
              <span style={{ fontSize: '0.75rem', color: '#72767a', fontWeight: 500 }}>
                Bitcoin Fraud & Illicit Detection System
              </span>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              className={`btn ${activeTab === 'dashboard' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <Activity size={16} /> Dashboard
            </button>
            <button
              className={`btn ${activeTab === 'transactions' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveTab('transactions')}
            >
              Transactions Explorer
            </button>
            <button
              className={`btn ${activeTab === 'model' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveTab('model')}
            >
              Model Performance
            </button>
            <button
              className={`btn ${activeTab === 'predict' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setActiveTab('predict')}
            >
              Live Predict
            </button>
          </nav>
        </div>

        {/* Right Group: Search Form */}
        <form onSubmit={handleSearchSubmit} style={{ display: 'flex', alignItems: 'center', gap: '0.375rem' }}>
          <div style={{ position: 'relative' }}>
            <input
              type="text"
              className="input"
              placeholder="Search Tx ID..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              style={{ paddingLeft: '2rem', width: '180px' }}
            />
            <Search size={14} style={{ position: 'absolute', left: '0.625rem', top: '50%', transform: 'translateY(-50%)', color: '#72767a' }} />
          </div>
          <button type="submit" className="btn btn-secondary btn-sm">Find</button>
        </form>
      </div>
    </header>
  );
};
