import React, { useState, useEffect, useCallback } from 'react';
import { Play, Square, AlertTriangle, Truck, Activity, Droplets, Wind, Thermometer, Gauge, MapPin, Key, Check } from 'lucide-react';
import { api } from '../../services/api';

const CITIES = [
  { value: 'Mumbai', label: '🇮🇳 Mumbai' },
  { value: 'Delhi', label: '🇮🇳 Delhi' },
  { value: 'Chennai', label: '🇮🇳 Chennai' },
  { value: 'Kolkata', label: '🇮🇳 Kolkata' },
  { value: 'Bangalore', label: '🇮🇳 Bangalore' },
  { value: 'London', label: '🇬🇧 London' },
  { value: 'New York', label: '🇺🇸 New York' },
  { value: 'Tokyo', label: '🇯🇵 Tokyo' },
];

const PredictionPanel = () => {
  const [isSimulating, setIsSimulating] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [deployMessage, setDeployMessage] = useState('');
  const [backendOnline, setBackendOnline] = useState(false);
  const [selectedCity, setSelectedCity] = useState('Mumbai');
  const [apiKey, setApiKey] = useState('');
  const [apiKeyConfigured, setApiKeyConfigured] = useState(false);
  const [showApiKeyInput, setShowApiKeyInput] = useState(false);
  const [apiKeyMessage, setApiKeyMessage] = useState('');

  // Check backend health and config
  useEffect(() => {
    const checkHealth = async () => {
      const health = await api.healthCheck();
      setBackendOnline(health.status === 'ok');

      const config = await api.getConfig();
      if (config) {
        setApiKeyConfigured(config.openweather_configured);
      }
    };
    checkHealth();
  }, []);

  // Fetch prediction data
  const fetchPrediction = useCallback(async () => {
    const data = await api.getPrediction(selectedCity);
    if (data) {
      setPrediction(data);
    }
  }, [selectedCity]);

  // Real-time polling every 10 seconds during simulation
  useEffect(() => {
    fetchPrediction(); // Initial fetch

    if (isSimulating) {
      const interval = setInterval(fetchPrediction, 10000); // 10 seconds
      return () => clearInterval(interval);
    }
  }, [isSimulating, fetchPrediction]);

  const handleStartSimulation = async () => {
    setLoading(true);
    const result = await api.startSimulation();
    if (result) {
      setIsSimulating(true);
      fetchPrediction();
    }
    setLoading(false);
  };

  const handleStopSimulation = async () => {
    setLoading(true);
    const result = await api.stopSimulation();
    if (result) {
      setIsSimulating(false);
      fetchPrediction();
    }
    setLoading(false);
  };

  const handleDeployResources = async () => {
    setLoading(true);
    const result = await api.deployResources();
    if (result) {
      setDeployMessage(result.message);
      setTimeout(() => setDeployMessage(''), 5000);
    }
    setLoading(false);
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'CRITICAL': return 'var(--color-danger)';
      case 'HIGH': return '#F97316';
      case 'MODERATE': return 'var(--color-warning)';
      default: return 'var(--color-success)';
    }
  };

  const handleSetApiKey = async () => {
    if (!apiKey.trim()) return;
    setLoading(true);
    const result = await api.setApiKey(apiKey);
    if (result?.status === 'success') {
      setApiKeyConfigured(true);
      setApiKeyMessage('✅ ' + result.message);
      setShowApiKeyInput(false);
      fetchPrediction();
    } else {
      setApiKeyMessage('❌ ' + (result?.message || 'Failed to set API key'));
    }
    setTimeout(() => setApiKeyMessage(''), 5000);
    setLoading(false);
  };

  const weather = prediction?.weather || {};
  const pred = prediction?.prediction || {};

  return (
    <div className="glass-panel prediction-panel">
      {/* Header with status */}
      <div className="panel-header">
        <div className="header-left">
          <Activity size={20} className="pulse-icon" />
          <h3>Flood Prediction AI</h3>
        </div>
        <span className={`status-badge ${backendOnline ? 'online' : 'offline'}`}>
          {backendOnline ? 'Model Online' : 'Offline - Start Backend'}
        </span>
      </div>

      {/* City Selector & API Key Config */}
      <div className="config-row">
        <div className="city-selector">
          <MapPin size={16} />
          <select
            value={selectedCity}
            onChange={(e) => setSelectedCity(e.target.value)}
            disabled={isSimulating}
          >
            {CITIES.map(city => (
              <option key={city.value} value={city.value}>{city.label}</option>
            ))}
          </select>
        </div>

        <button
          className={`btn-config ${apiKeyConfigured ? 'configured' : ''}`}
          onClick={() => setShowApiKeyInput(!showApiKeyInput)}
        >
          {apiKeyConfigured ? <Check size={16} /> : <Key size={16} />}
          {apiKeyConfigured ? 'Live Weather' : 'Add API Key'}
        </button>
      </div>

      {/* API Key Input */}
      {showApiKeyInput && (
        <div className="api-key-input animate-fade-in">
          <input
            type="text"
            placeholder="Enter OpenWeatherMap API Key"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <button onClick={handleSetApiKey} disabled={loading || !apiKey.trim()}>
            Save
          </button>
        </div>
      )}

      {apiKeyMessage && (
        <div className="api-message">{apiKeyMessage}</div>
      )}

      {/* Control Buttons */}
      <div className="control-buttons">
        {!isSimulating ? (
          <button
            className="btn btn-primary"
            onClick={handleStartSimulation}
            disabled={loading || !backendOnline}
          >
            <Play size={18} />
            Simulate Flood
          </button>
        ) : (
          <button
            className="btn btn-danger"
            onClick={handleStopSimulation}
            disabled={loading}
          >
            <Square size={18} />
            Stop Simulation
          </button>
        )}

        <button
          className="btn btn-warning"
          onClick={handleDeployResources}
          disabled={loading || !backendOnline}
        >
          <Truck size={18} />
          Deploy Resources
        </button>
      </div>

      {/* Deploy Message */}
      {deployMessage && (
        <div className="deploy-alert animate-fade-in">
          <AlertTriangle size={20} />
          {deployMessage}
        </div>
      )}

      {/* Simulation Status */}
      {isSimulating && (
        <div className="simulation-active">
          <div className="pulse-ring"></div>
          <span>🌊 SIMULATION ACTIVE - Updating every 10 seconds</span>
        </div>
      )}

      {/* Weather Data */}
      {prediction && (
        <>
          <div className="weather-grid">
            <div className="weather-item">
              <Droplets size={16} />
              <span className="label">Rainfall</span>
              <span className="value">{weather.rainfall?.toFixed(1) || '0'} mm</span>
            </div>
            <div className="weather-item">
              <Thermometer size={16} />
              <span className="label">Temp</span>
              <span className="value">{weather.temperature?.toFixed(1) || '25'}°C</span>
            </div>
            <div className="weather-item">
              <Gauge size={16} />
              <span className="label">Humidity</span>
              <span className="value">{weather.humidity?.toFixed(0) || '50'}%</span>
            </div>
            <div className="weather-item">
              <Wind size={16} />
              <span className="label">Wind</span>
              <span className="value">{weather.wind_speed?.toFixed(1) || '10'} km/h</span>
            </div>
          </div>

          {/* Prediction Result */}
          <div className="prediction-result" style={{ borderColor: getRiskColor(pred.risk_level) }}>
            <div className="risk-header">
              <span className="risk-label">Flood Risk Level</span>
              <span className="risk-level" style={{ color: getRiskColor(pred.risk_level) }}>
                {pred.risk_level || 'CALCULATING...'}
              </span>
            </div>
            <div className="probability-bar">
              <div
                className="probability-fill"
                style={{
                  width: `${(pred.probability || 0) * 100}%`,
                  backgroundColor: getRiskColor(pred.risk_level)
                }}
              ></div>
            </div>
            <span className="probability-text">
              Probability: {((pred.probability || 0) * 100).toFixed(1)}%
            </span>
          </div>

          <div className="data-source">
            Source: {weather.source === 'live' ? '🌐 Live Weather API' : '🔬 Simulation Engine'}
            {weather.city && ` • ${weather.city}`}
          </div>
        </>
      )}

      <style>{`
        .prediction-panel {
          padding: 1.5rem;
          display: flex;
          flex-direction: column;
          gap: 1rem;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .header-left {
          display: flex;
          align-items: center;
          gap: 0.5rem;
        }

        .pulse-icon {
          color: var(--color-brand-primary);
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .status-badge {
          font-size: 0.75rem;
          padding: 0.25rem 0.75rem;
          border-radius: 1rem;
          font-weight: 600;
        }

        .status-badge.online {
          background: rgba(16, 185, 129, 0.2);
          color: var(--color-success);
        }

        .status-badge.offline {
          background: rgba(239, 68, 68, 0.2);
          color: var(--color-danger);
        }

        .config-row {
          display: flex;
          gap: 0.75rem;
          align-items: center;
        }

        .city-selector {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          flex: 1;
          background: rgba(255, 255, 255, 0.05);
          padding: 0.5rem 0.75rem;
          border-radius: var(--radius-button);
          color: var(--color-text-muted);
        }

        .city-selector select {
          background: transparent;
          border: none;
          color: var(--color-text-main);
          font-size: 0.875rem;
          cursor: pointer;
          flex: 1;
        }

        .city-selector select option {
          background: var(--color-bg-card);
          color: var(--color-text-main);
        }

        .btn-config {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.5rem 1rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: var(--radius-button);
          background: transparent;
          color: var(--color-text-muted);
          cursor: pointer;
          font-size: 0.875rem;
          transition: all 0.2s;
        }

        .btn-config:hover {
          border-color: var(--color-brand-primary);
          color: var(--color-brand-primary);
        }

        .btn-config.configured {
          border-color: var(--color-success);
          color: var(--color-success);
        }

        .api-key-input {
          display: flex;
          gap: 0.5rem;
        }

        .api-key-input input {
          flex: 1;
          padding: 0.625rem 0.875rem;
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: var(--radius-button);
          background: rgba(255, 255, 255, 0.05);
          color: var(--color-text-main);
          font-size: 0.875rem;
        }

        .api-key-input input:focus {
          outline: none;
          border-color: var(--color-brand-primary);
        }

        .api-key-input button {
          padding: 0.625rem 1rem;
          border: none;
          border-radius: var(--radius-button);
          background: var(--color-brand-primary);
          color: white;
          font-weight: 600;
          cursor: pointer;
        }

        .api-key-input button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .api-message {
          padding: 0.5rem 0.75rem;
          border-radius: var(--radius-button);
          font-size: 0.875rem;
          background: rgba(255, 255, 255, 0.03);
        }

        .control-buttons {
          display: flex;
          gap: 0.75rem;
        }

        .btn {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.75rem 1.25rem;
          border: none;
          border-radius: var(--radius-button);
          font-weight: 600;
          cursor: pointer;
          transition: all 0.2s;
          flex: 1;
          justify-content: center;
        }

        .btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .btn-primary {
          background: linear-gradient(135deg, var(--color-brand-primary), #60A5FA);
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          box-shadow: 0 0 20px var(--color-brand-glow);
          transform: translateY(-1px);
        }

        .btn-danger {
          background: linear-gradient(135deg, var(--color-danger), #F87171);
          color: white;
        }

        .btn-warning {
          background: linear-gradient(135deg, var(--color-warning), #FBBF24);
          color: #1F2937;
        }

        .deploy-alert {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 1rem;
          background: rgba(239, 68, 68, 0.15);
          border: 1px solid var(--color-danger);
          border-radius: var(--radius-button);
          color: var(--color-danger);
          font-weight: 600;
        }

        .simulation-active {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.75rem 1rem;
          background: rgba(59, 130, 246, 0.1);
          border-radius: var(--radius-button);
          font-size: 0.875rem;
          color: var(--color-brand-primary);
        }

        .pulse-ring {
          width: 10px;
          height: 10px;
          background: var(--color-brand-primary);
          border-radius: 50%;
          animation: pulse-ring 1.5s infinite;
        }

        @keyframes pulse-ring {
          0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.7); }
          70% { box-shadow: 0 0 0 10px rgba(59, 130, 246, 0); }
          100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
        }

        .weather-grid {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 0.75rem;
        }

        .weather-item {
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 0.25rem;
          padding: 0.75rem;
          background: rgba(255, 255, 255, 0.03);
          border-radius: var(--radius-button);
        }

        .weather-item .label {
          font-size: 0.7rem;
          color: var(--color-text-dim);
          text-transform: uppercase;
        }

        .weather-item .value {
          font-weight: 600;
          color: var(--color-text-main);
        }

        .prediction-result {
          padding: 1.25rem;
          background: rgba(0, 0, 0, 0.2);
          border-radius: var(--radius-button);
          border-left: 4px solid;
        }

        .risk-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 0.75rem;
        }

        .risk-label {
          color: var(--color-text-muted);
          font-size: 0.875rem;
        }

        .risk-level {
          font-size: 1.25rem;
          font-weight: 700;
        }

        .probability-bar {
          height: 8px;
          background: rgba(255, 255, 255, 0.1);
          border-radius: 4px;
          overflow: hidden;
          margin-bottom: 0.5rem;
        }

        .probability-fill {
          height: 100%;
          transition: width 0.5s ease;
        }

        .probability-text {
          font-size: 0.875rem;
          color: var(--color-text-muted);
        }

        .data-source {
          font-size: 0.75rem;
          color: var(--color-text-dim);
          text-align: center;
          padding-top: 0.5rem;
          border-top: 1px solid rgba(255, 255, 255, 0.05);
        }
      `}</style>
    </div>
  );
};

export default PredictionPanel;
