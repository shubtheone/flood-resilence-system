import React from 'react';
import { TriangleAlert, CloudRain, Info } from 'lucide-react';
import { alerts } from '../../data/mockData';

const severityIcons = {
    critical: <TriangleAlert className="text-danger" size={20} />,
    warning: <CloudRain className="text-warning" size={20} />,
    info: <Info className="text-info" size={20} />
};

const AlertsPanel = () => {
    return (
        <div className="glass-panel alerts-panel">
            <div className="panel-header">
                <h3>Active Alerts</h3>
                <span className="live-pill">Live</span>
            </div>

            <div className="alerts-list">
                {alerts.map((alert) => (
                    <div key={alert.id} className={`alert-card ${alert.severity}`}>
                        <div className="alert-icon">
                            {severityIcons[alert.severity]}
                        </div>
                        <div className="alert-content">
                            <h4 className="alert-title">{alert.location}</h4>
                            <p className="alert-msg">{alert.message}</p>
                            <span className="alert-time">{alert.timestamp}</span>
                        </div>
                    </div>
                ))}
            </div>

            <style>{`
        .alerts-panel {
          grid-area: alerts;
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
          height: 100%;
          overflow: hidden;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.5rem;
        }

        .live-pill {
          background-color: rgba(239, 68, 68, 0.2);
          color: #EF4444;
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-size: 0.75rem;
          font-weight: 700;
          display: flex;
          align-items: center;
        }
        
        .live-pill::before {
          content: '';
          display: inline-block;
          width: 6px;
          height: 6px;
          background-color: currentColor;
          border-radius: 50%;
          margin-right: 6px;
          animation: pulse 2s infinite;
        }

        .alerts-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
          overflow-y: auto;
          scrollbar-width: thin;
        }

        .alert-card {
          display: flex;
          gap: 1rem;
          padding: 1rem;
          border-radius: var(--radius-button);
          background: rgba(255, 255, 255, 0.03);
          border-left: 4px solid transparent;
          transition: transform 0.2s;
        }

        .alert-card:hover {
          transform: translateX(4px);
          background: rgba(255, 255, 255, 0.06);
        }

        .alert-card.critical { border-left-color: var(--color-danger); }
        .alert-card.warning { border-left-color: var(--color-warning); }
        .alert-card.info { border-left-color: var(--color-info); }

        .alert-title {
          font-weight: 600;
          margin-bottom: 0.25rem;
          color: var(--color-text-main);
        }
        
        .alert-msg {
          font-size: 0.875rem;
          color: var(--color-text-muted);
          margin-bottom: 0.5rem;
        }

        .alert-time {
          font-size: 0.75rem;
          color: var(--color-text-dim);
          display: block;
          text-align: right;
        }

        .text-danger { color: var(--color-danger); }
        .text-warning { color: var(--color-warning); }
        .text-info { color: var(--color-info); }
      `}</style>
        </div>
    );
};

export default AlertsPanel;
