import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { riverLevels } from '../../data/mockData';

const WaterLevelChart = () => {
    return (
        <div className="glass-panel" style={{ padding: '1.5rem', height: '350px' }}>
            <h3 style={{ marginBottom: '1rem', color: '#CBD5E1' }}>River Water Levels (Simulated)</h3>
            <ResponsiveContainer width="100%" height="85%">
                <AreaChart
                    data={riverLevels}
                    margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                >
                    <defs>
                        <linearGradient id="colorLevel" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8} />
                            <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                        </linearGradient>
                    </defs>
                    <XAxis dataKey="time" stroke="#94A3B8" />
                    <YAxis stroke="#94A3B8" />
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <Tooltip
                        contentStyle={{ backgroundColor: '#1E293B', border: 'none', borderRadius: '8px' }}
                        itemStyle={{ color: '#E2E8F0' }}
                    />
                    <Area
                        type="monotone"
                        dataKey="level"
                        stroke="#3B82F6"
                        fillOpacity={1}
                        fill="url(#colorLevel)"
                    />
                </AreaChart>
            </ResponsiveContainer>
        </div>
    );
};

export default WaterLevelChart;
