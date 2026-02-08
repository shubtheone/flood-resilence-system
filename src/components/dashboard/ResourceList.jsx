import React from 'react';
import { Truck, Navigation, LifeBuoy } from 'lucide-react';
import { resources } from '../../data/mockData';
import { clsx } from 'clsx'; // Utility for conditional classes

const ResourceList = () => {
    return (
        <div className="glass-panel resources-panel">
            <h3 className="panel-header">Deployed Resources</h3>
            <ul className="resources-list">
                {resources.map((resource) => (
                    <li key={resource.id} className="resource-item">
                        <div className={clsx('status-indicator', resource.status.toLowerCase())}></div>
                        <div className="icon-wrapper">
                            {resource.type === 'Rescue Boat' && <LifeBuoy size={18} />}
                            {resource.type === 'Medical Team' && <Truck size={18} />}
                            {resource.type === 'Helicopter' && <Navigation size={18} />}
                        </div>
                        <div className="resource-info">
                            <span className="resource-name">{resource.type}</span>
                            <span className={clsx('resource-status', resource.status.toLowerCase())}>{resource.status}</span>
                        </div>
                    </li>
                ))}
            </ul>
            <style>{`
                .resources-panel {
                    padding: 1.5rem;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    height: 100%;
                }
                .resources-list {
                    list-style: none;
                    padding: 0;
                    margin: 0;
                    flex: 1;
                    overflow-y: auto;
                }
                .resource-item {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 0.75rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }
                .resource-item:last-child {
                    border-bottom: none;
                }
                .status-indicator {
                  width: 8px;
                  height: 8px;
                  border-radius: 50%;
                }
                .status-indicator.deployed { background-color: var(--color-warning); box-shadow: 0 0 5px var(--color-warning); }
                .status-indicator.available { background-color: var(--color-success); box-shadow: 0 0 5px var(--color-success); }
                .status-indicator.maintenance { background-color: var(--color-text-dim); }

                .resource-info {
                    display: flex;
                    flex-direction: column;
                }
                .resource-name {
                    font-weight: 500;
                    color: var(--color-text-main);
                }
                .resource-status {
                    font-size: 0.75rem;
                    color: var(--color-text-muted);
                }
                .resource-status.deployed { color: var(--color-warning); }
                .resource-status.available { color: var(--color-success); }
            `}</style>
        </div>
    );
};

export default ResourceList;
