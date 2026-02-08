export const riverLevels = [
    { time: '00:00', level: 2.1, threshold: 4.5 },
    { time: '04:00', level: 2.3, threshold: 4.5 },
    { time: '08:00', level: 2.8, threshold: 4.5 },
    { time: '12:00', level: 3.5, threshold: 4.5 },
    { time: '16:00', level: 4.2, threshold: 4.5 }, // Approaching danger
    { time: '20:00', level: 4.8, threshold: 4.5 }, // Flood level
];

export const alerts = [
    {
        id: 1,
        severity: 'critical',
        location: 'North District',
        message: 'Flash flood warning issued. Evacuate immediately.',
        timestamp: '10 mins ago'
    },
    {
        id: 2,
        severity: 'warning',
        location: 'East River Valley',
        message: 'Water levels rising rapidly above normal.',
        timestamp: '45 mins ago'
    },
    {
        id: 3,
        severity: 'info',
        location: 'City Center',
        message: 'Emergency shelters opened at Community Hall.',
        timestamp: '2 hours ago'
    }
];

export const resources = [
    { id: 1, type: 'Rescue Boat', status: 'Deployed', location: [51.505, -0.09] },
    { id: 2, type: 'Medical Team', status: 'Available', location: [51.51, -0.1] },
    { id: 3, type: 'Helicopter', status: 'Maintenance', location: [51.52, -0.12] },
];
