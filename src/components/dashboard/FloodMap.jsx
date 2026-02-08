import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
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

const FloodMap = ({ height = "400px" }) => {
    const position = [51.505, -0.09]; // London coordinates as default example

    return (
        <div className="glass-panel" style={{ height: '100%', minHeight: height, overflow: 'hidden' }}>
            <MapContainer center={position} zoom={13} scrollWheelZoom={false} style={{ height: '100%', width: '100%' }}>
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {/* Example Flood Zone */}
                <Circle
                    center={position}
                    pathOptions={{ fillColor: 'red', color: 'red' }}
                    radius={500}
                />
                <Marker position={position}>
                    <Popup>
                        Flood Warning Zone <br /> Water Level: Critical.
                    </Popup>
                </Marker>
            </MapContainer>
        </div>
    );
};

export default FloodMap;
