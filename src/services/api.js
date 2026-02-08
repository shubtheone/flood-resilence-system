const API_BASE_URL = 'http://localhost:5000/api';

export const api = {
    // Get flood prediction with weather data
    getPrediction: async (city = 'Mumbai') => {
        try {
            const response = await fetch(`${API_BASE_URL}/predict?city=${city}`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Start flood simulation
    startSimulation: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/simulate/start`, {
                method: 'POST'
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Stop flood simulation
    stopSimulation: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/simulate/stop`, {
                method: 'POST'
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Deploy emergency resources
    deployResources: async (type = 'all', location = 'Affected Area') => {
        try {
            const response = await fetch(`${API_BASE_URL}/deploy-resources`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type, location })
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Get active alerts
    getAlerts: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/alerts`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Health check
    healthCheck: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/health`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return { status: 'offline' };
        }
    },

    // Get API config status
    getConfig: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/config`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Set OpenWeatherMap API key
    setApiKey: async (apiKey) => {
        try {
            const response = await fetch(`${API_BASE_URL}/config/apikey`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_key: apiKey })
            });
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    },

    // Get model statistics and accuracy
    getModelStats: async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/model/stats`);
            return await response.json();
        } catch (error) {
            console.error('API Error:', error);
            return null;
        }
    }
};
