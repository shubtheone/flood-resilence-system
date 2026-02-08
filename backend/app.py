"""
Flood Prediction API Server
Provides endpoints for real-time flood prediction and resource deployment
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
import os
import requests
import random
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration - Load from environment or use defaults
OPENWEATHER_API_KEY = os.environ.get('OPENWEATHER_API_KEY', '')
DEFAULT_CITY = os.environ.get('DEFAULT_CITY', 'Mumbai')

# Load model on startup
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
model = None
scaler = None
feature_cols = None

def load_model():
    global model, scaler, feature_cols
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'flood_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
        print("✅ Model loaded successfully")
        return True
    except FileNotFoundError:
        print("⚠️ Model not found. Please run train_model.py first")
        return False

# Simulation state
simulation_active = False
simulation_data = {
    'rainfall': 50,
    'temperature': 28,
    'humidity': 65,
    'wind_speed': 15,
    'river_level': 3.5,
    'soil_moisture': 40
}

def get_live_weather(city=DEFAULT_CITY):
    """Fetch live weather data from OpenWeatherMap"""
    if not OPENWEATHER_API_KEY:
        # Return simulated weather if no API key
        return generate_simulated_weather()
    
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if response.status_code == 200:
            return {
                'rainfall': data.get('rain', {}).get('1h', random.uniform(0, 50)),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'wind_speed': data['wind']['speed'] * 3.6,  # Convert m/s to km/h
                'river_level': random.uniform(2, 5),  # Simulated
                'soil_moisture': random.uniform(30, 70),  # Simulated
                'city': city,
                'description': data['weather'][0]['description'],
                'source': 'live'
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    
    return generate_simulated_weather()

def generate_simulated_weather():
    """Generate simulated weather for demo purposes"""
    global simulation_data
    
    if simulation_active:
        # Gradually increase danger levels during simulation
        simulation_data['rainfall'] = min(300, simulation_data['rainfall'] + random.uniform(10, 30))
        simulation_data['humidity'] = min(100, simulation_data['humidity'] + random.uniform(2, 8))
        simulation_data['river_level'] = min(10, simulation_data['river_level'] + random.uniform(0.3, 0.8))
        simulation_data['soil_moisture'] = min(100, simulation_data['soil_moisture'] + random.uniform(3, 7))
    else:
        # Normal fluctuation
        simulation_data = {
            'rainfall': random.uniform(20, 80),
            'temperature': random.uniform(24, 32),
            'humidity': random.uniform(50, 75),
            'wind_speed': random.uniform(10, 25),
            'river_level': random.uniform(2.5, 4.5),
            'soil_moisture': random.uniform(30, 55)
        }
    
    return {
        **simulation_data,
        'city': 'Simulation',
        'description': 'Heavy rain' if simulation_data['rainfall'] > 100 else 'Moderate rain',
        'source': 'simulation'
    }

def predict_flood(weather_data):
    """Make flood prediction using the trained model
    
    Maps real-time weather data to model features:
    - MonsoonIntensity: Based on rainfall (0-10 scale, 10 = extreme)
    - TopographyDrainage: Inverse of soil moisture (saturated = poor drainage)
    - RiverManagement: Based on river level (lower = better management)
    - Deforestation: Simulated regional factor
    - Urbanization: Simulated regional factor
    - ClimateChange: Based on temperature anomaly
    """
    if model is None:
        # Fallback rule-based prediction
        risk_score = 0
        if weather_data['rainfall'] > 100:
            risk_score += 30
        if weather_data['humidity'] > 80:
            risk_score += 20
        if weather_data['river_level'] > 5:
            risk_score += 30
        if weather_data['soil_moisture'] > 70:
            risk_score += 20
        
        return {
            'probability': min(risk_score, 100) / 100,
            'risk_level': 'CRITICAL' if risk_score > 70 else 'HIGH' if risk_score > 50 else 'MODERATE' if risk_score > 30 else 'LOW',
            'method': 'rule-based'
        }
    
    # Map weather data to model features
    try:
        rainfall = weather_data.get('rainfall', 0)
        humidity = weather_data.get('humidity', 50)
        river_level = weather_data.get('river_level', 3)
        soil_moisture = weather_data.get('soil_moisture', 40)
        temperature = weather_data.get('temperature', 25)
        
        # Feature mapping (scale to 0-10 range matching Kaggle dataset)
        monsoon_intensity = min(10, (rainfall / 30))  # 300mm = intensity 10
        topography_drainage = max(0, 10 - (soil_moisture / 10))  # Low moisture = good drainage
        river_management = max(0, 10 - river_level)  # Low river = good management
        deforestation = 5 + (humidity - 50) / 20  # Humidity correlates with deforested areas
        urbanization = 5  # Base urbanization factor
        climate_change = min(10, max(0, (temperature - 20) / 2))  # Higher temp = more impact
        
        features = np.array([[
            monsoon_intensity,
            topography_drainage,
            river_management,
            deforestation,
            urbanization,
            climate_change
        ]])[:, :len(feature_cols)]
        
        features_scaled = scaler.transform(features)
        probability = model.predict_proba(features_scaled)[0][1]
        
        if probability > 0.75:
            risk_level = 'CRITICAL'
        elif probability > 0.5:
            risk_level = 'HIGH'
        elif probability > 0.25:
            risk_level = 'MODERATE'
        else:
            risk_level = 'LOW'
        
        return {
            'probability': float(probability),
            'risk_level': risk_level,
            'method': 'ml-model',
            'features_used': {
                'monsoon_intensity': round(monsoon_intensity, 2),
                'topography_drainage': round(topography_drainage, 2),
                'river_management': round(river_management, 2),
                'deforestation': round(deforestation, 2),
                'urbanization': round(urbanization, 2),
                'climate_change': round(climate_change, 2)
            }
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return {'probability': 0, 'risk_level': 'UNKNOWN', 'method': 'error'}

# ===================== API ENDPOINTS =====================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    """Get current flood prediction with weather data"""
    city = request.args.get('city', DEFAULT_CITY)
    weather = get_live_weather(city)
    prediction = predict_flood(weather)
    
    return jsonify({
        'timestamp': datetime.now().isoformat(),
        'weather': weather,
        'prediction': prediction,
        'simulation_active': simulation_active
    })

@app.route('/api/simulate/start', methods=['POST'])
def start_simulation():
    """Start flood simulation mode"""
    global simulation_active, simulation_data
    simulation_active = True
    simulation_data = {
        'rainfall': 50,
        'temperature': 28,
        'humidity': 65,
        'wind_speed': 15,
        'river_level': 3.5,
        'soil_moisture': 40
    }
    return jsonify({'status': 'simulation_started', 'message': 'Flood simulation activated. Weather conditions will worsen over time.'})

@app.route('/api/simulate/stop', methods=['POST'])
def stop_simulation():
    """Stop flood simulation mode"""
    global simulation_active
    simulation_active = False
    return jsonify({'status': 'simulation_stopped', 'message': 'Simulation stopped. Normal conditions restored.'})

@app.route('/api/deploy-resources', methods=['POST'])
def deploy_resources():
    """Deploy emergency resources"""
    data = request.json or {}
    resource_type = data.get('type', 'all')
    location = data.get('location', 'Affected Area')
    
    return jsonify({
        'status': 'deployed',
        'message': f'🚨 EMERGENCY ALERT: Forces deployed to {location}!',
        'details': {
            'rescue_boats': 5,
            'medical_teams': 3,
            'helicopters': 2,
            'evacuation_buses': 10,
            'timestamp': datetime.now().isoformat()
        }
    })

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current active alerts based on prediction"""
    weather = get_live_weather()
    prediction = predict_flood(weather)
    
    alerts = []
    
    if prediction['risk_level'] == 'CRITICAL':
        alerts.append({
            'id': 1,
            'severity': 'critical',
            'title': 'FLASH FLOOD WARNING',
            'message': f"Immediate evacuation required. Flood probability: {prediction['probability']*100:.1f}%",
            'location': 'All low-lying areas'
        })
    
    if weather['rainfall'] > 100:
        alerts.append({
            'id': 2,
            'severity': 'warning',
            'title': 'Heavy Rainfall Alert',
            'message': f"Rainfall: {weather['rainfall']:.1f}mm. Expect waterlogging.",
            'location': 'Urban areas'
        })
    
    if weather['river_level'] > 5:
        alerts.append({
            'id': 3,
            'severity': 'warning',
            'title': 'River Level Rising',
            'message': f"River level at {weather['river_level']:.1f}m (Danger: 6m)",
            'location': 'Riverside communities'
        })
    
    if not alerts:
        alerts.append({
            'id': 0,
            'severity': 'info',
            'title': 'All Clear',
            'message': 'No immediate flood threats detected.',
            'location': 'Region-wide'
        })
    
    return jsonify({'alerts': alerts, 'count': len(alerts)})

@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current API configuration status"""
    return jsonify({
        'openweather_configured': bool(OPENWEATHER_API_KEY),
        'default_city': DEFAULT_CITY,
        'model_loaded': model is not None,
        'simulation_active': simulation_active
    })

@app.route('/api/config/apikey', methods=['POST'])
def set_api_key():
    """Temporarily set OpenWeatherMap API key (session only)"""
    global OPENWEATHER_API_KEY
    data = request.json or {}
    new_key = data.get('api_key', '')
    
    if new_key:
        OPENWEATHER_API_KEY = new_key
        # Test the key
        test_url = f"http://api.openweathermap.org/data/2.5/weather?q=London&appid={new_key}&units=metric"
        try:
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                return jsonify({'status': 'success', 'message': 'API key configured and verified!'})
            else:
                return jsonify({'status': 'error', 'message': 'Invalid API key'}), 400
        except:
            return jsonify({'status': 'error', 'message': 'Could not verify API key'}), 400
    
    return jsonify({'status': 'error', 'message': 'No API key provided'}), 400

if __name__ == '__main__':
    load_model()
    print("\n🌊 Flood Prediction API Server")
    print("=" * 50)
    print(f"📍 Default City: {DEFAULT_CITY}")
    print(f"🔑 OpenWeatherMap API: {'✅ Configured' if OPENWEATHER_API_KEY else '❌ Not set (using simulation)'}")
    print(f"🤖 ML Model: {'✅ Loaded' if model else '❌ Not found'}")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /api/predict          - Get flood prediction")
    print("  GET  /api/config           - Get API configuration")
    print("  POST /api/config/apikey    - Set OpenWeatherMap API key")
    print("  POST /api/simulate/start   - Start simulation")
    print("  POST /api/simulate/stop    - Stop simulation")
    print("  POST /api/deploy-resources - Deploy resources")
    print("  GET  /api/alerts           - Get active alerts")
    print("=" * 50)
    app.run(debug=True, port=5000)
