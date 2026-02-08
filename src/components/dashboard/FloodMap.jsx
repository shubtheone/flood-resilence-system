import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle, useMap } from 'react-leaflet';
import L from 'leaflet';

// Fix for default marker icon
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

let DefaultIcon = L.icon({
    iconUrl: icon,
    shadowUrl: iconShadow,
    iconSize: [25, 41],
    iconAnchor: [12, 41]
});

L.Marker.prototype.options.icon = DefaultIcon;

// City coordinates mapping
const CITY_COORDS = {
    'Mumbai': [19.0760, 72.8777],
    'Delhi': [28.6139, 77.2090],
    'Chennai': [13.0827, 80.2707],
    'Kolkata': [22.5726, 88.3639],
    'Bangalore': [12.9716, 77.5946],
    'London': [51.5074, -0.1278],
    'New York': [40.7128, -74.0060],
    'Tokyo': [35.6762, 139.6503],
};

// Component to update map center when city changes
const MapUpdater = ({ center }) => {
    const map = useMap();
    useEffect(() => {
        map.setView(center, 12);
    }, [center, map]);
    return null;
};

const FloodMap = ({ height = "400px", city = "Mumbai", riskLevel = "LOW" }) => {
    const position = CITY_COORDS[city] || CITY_COORDS['Mumbai'];

    // Dynamic color based on risk level
    const getRiskColor = () => {
        switch (riskLevel) {
            case 'CRITICAL': return '#EF4444';
            case 'HIGH': return '#F97316';
            case 'MODERATE': return '#F59E0B';
            default: return '#10B981';
        }
    };

    const riskColor = getRiskColor();
    const circleRadius = riskLevel === 'CRITICAL' ? 2000 : riskLevel === 'HIGH' ? 1500 : riskLevel === 'MODERATE' ? 1000 : 500;

    return (
        <div className="glass-panel" style={{ height: '100%', minHeight: height, overflow: 'hidden', position: 'relative' }}>
            {/* City Label */}
            <div style={{
                position: 'absolute',
                top: '1rem',
                left: '1rem',
                zIndex: 1000,
                background: 'rgba(0,0,0,0.7)',
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                color: 'white',
                fontSize: '0.875rem',
                fontWeight: 600
            }}>
                📍 {city}
            </div>

            {/* Risk Indicator */}
            <div style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                zIndex: 1000,
                background: riskColor,
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                color: 'white',
                fontSize: '0.75rem',
                fontWeight: 700
            }}>
                {riskLevel} RISK
            </div>

            <MapContainer center={position} zoom={12} scrollWheelZoom={true} style={{ height: '100%', width: '100%' }}>
                <MapUpdater center={position} />
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {/* Flood Risk Zone */}
                <Circle
                    center={position}
                    pathOptions={{
                        fillColor: riskColor,
                        color: riskColor,
                        fillOpacity: 0.3
                    }}
                    radius={circleRadius}
                />
                <Marker position={position}>
                    <Popup>
                        <strong>{city}</strong><br />
                        Risk Level: {riskLevel}<br />
                        Zone Radius: {circleRadius}m
                    </Popup>
                </Marker>
            </MapContainer>
        </div>
    );
};

export default FloodMap;
