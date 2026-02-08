import React from 'react';
import FloodMap from '../components/dashboard/FloodMap';
import AlertsPanel from '../components/dashboard/AlertsPanel';
import WaterLevelChart from '../components/dashboard/WaterLevelChart';
import ResourceList from '../components/dashboard/ResourceList';
import PredictionPanel from '../components/dashboard/PredictionPanel';
import { CloudRain, Droplets, Wind, Thermometer } from 'lucide-react';

const StatCard = ({ title, value, unit, icon: Icon, trend }) => (
  <div className="glass-panel stat-card">
    <div className="stat-header">
      <div className="stat-icon-bg">
        <Icon size={20} className="stat-icon" />
      </div>
      <span className="stat-title">{title}</span>
    </div>
    <div className="stat-value">
      {value}<span className="stat-unit">{unit}</span>
    </div>
    <div className={`stat-trend ${trend > 0 ? 'up' : 'down'}`}>
      {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from yesterday
    </div>
  </div>
);

const Dashboard = () => {
  return (
    <div className="dashboard-grid">
      {/* Stats Row */}
      <div className="stats-row">
        <StatCard title="Avg Rainfall" value="45" unit="mm" icon={CloudRain} trend={12} />
        <StatCard title="Water Level" value="4.2" unit="m" icon={Droplets} trend={5} />
        <StatCard title="Wind Speed" value="18" unit="km/h" icon={Wind} trend={-2} />
        <StatCard title="Temperature" value="24" unit="°C" icon={Thermometer} trend={1} />
      </div>

      {/* AI Prediction Panel - Featured */}
      <PredictionPanel />

      {/* Main Content: Map & Alerts */}
      <div className="main-grid">
        <div className="map-section">
          <FloodMap height="100%" />
        </div>
        <div className="side-panel">
          <AlertsPanel />
        </div>
      </div>

      {/* Bottom Content: Charts & Resources */}
      <div className="bottom-grid">
        <div className="chart-section">
          <WaterLevelChart />
        </div>
        <div className="resource-section">
          <ResourceList />
        </div>
      </div>

      <style>{`
        .dashboard-grid {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
          height: 100%;
          padding-bottom: 2rem;
        }

        .stats-row {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 1.5rem;
        }

        .stat-card {
          padding: 1.25rem;
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }

        .stat-header {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          color: var(--color-text-muted);
          font-size: 0.875rem;
        }

        .stat-icon-bg {
          width: 36px;
          height: 36px;
          background: rgba(59, 130, 246, 0.1);
          border-radius: 8px;
          display: flex;
          align-items: center;
          justify-content: center;
          color: var(--color-brand-primary);
        }

        .stat-value {
          font-size: 2rem;
          font-weight: 700;
          color: var(--color-text-main);
        }

        .stat-unit {
          font-size: 1rem;
          font-weight: 500;
          color: var(--color-text-dim);
          margin-left: 0.25rem;
        }

        .stat-trend {
          font-size: 0.75rem;
          margin-top: auto;
        }
        .stat-trend.up { color: var(--color-danger); }
        .stat-trend.down { color: var(--color-success); }

        .main-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 1.5rem;
          min-height: 400px;
        }

        .bottom-grid {
          display: grid;
          grid-template-columns: 2fr 1fr;
          gap: 1.5rem;
          min-height: 300px;
        }

        @media (max-width: 1024px) {
          .main-grid, .bottom-grid {
            grid-template-columns: 1fr;
          }
        }
      `}</style>
    </div>
  );
};

export default Dashboard;
