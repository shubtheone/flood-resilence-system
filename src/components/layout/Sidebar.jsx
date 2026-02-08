import React from 'react';
import { LayoutDashboard, Map, Bell, Package, Radio, Settings, LogOut } from 'lucide-react';

const Sidebar = ({ activeTab, setActiveTab }) => {
    const menuItems = [
        { id: 'dashboard', label: 'Overview', icon: LayoutDashboard },
        { id: 'map', label: 'Flood Map', icon: Map },
        { id: 'alerts', label: 'Alerts', icon: Bell },
        { id: 'resources', label: 'Resources', icon: Package },
        { id: 'coordinate', label: 'Coordinate', icon: Radio },
    ];

    return (
        <aside className="sidebar glass-panel">
            <div className="sidebar-header">
                <div className="logo-container">
                    <div className="logo-icon"></div>
                    <span className="brand-name">FloodGuard</span>
                </div>
            </div>

            <nav className="sidebar-nav">
                {menuItems.map((item) => {
                    const Icon = item.icon;
                    return (
                        <button
                            key={item.id}
                            className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
                            onClick={() => setActiveTab(item.id)}
                        >
                            <Icon size={20} />
                            <span>{item.label}</span>
                            {item.id === 'alerts' && <span className="badge-dot"></span>}
                        </button>
                    );
                })}
            </nav>

            <div className="sidebar-footer">
                <button className="nav-item">
                    <Settings size={20} />
                    <span>Settings</span>
                </button>
                <button className="nav-item danger">
                    <LogOut size={20} />
                    <span>Logout</span>
                </button>
            </div>

            <style>{`
        .sidebar {
          width: var(--sidebar-width);
          height: 95vh;
          margin: 2.5vh 0 2.5vh 1rem;
          display: flex;
          flex-direction: column;
          padding: 1.5rem;
          position: fixed;
          background: rgba(15, 23, 42, 0.6); /* Slightly darker/opaque for readability */
        }
        
        .sidebar-header {
          margin-bottom: 2rem;
          display: flex;
          align-items: center;
        }

        .logo-container {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .logo-icon {
          width: 32px;
          height: 32px;
          background: linear-gradient(135deg, var(--color-brand-primary), #60A5FA);
          border-radius: 8px;
          box-shadow: 0 0 15px var(--color-brand-glow);
        }

        .brand-name {
          font-weight: 700;
          font-size: 1.25rem;
          background: linear-gradient(to right, #fff, #cbd5e1);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .sidebar-nav {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
          flex: 1;
        }

        .nav-item {
          display: flex;
          align-items: center;
          gap: 1rem;
          padding: 0.875rem 1rem;
          border-radius: var(--radius-button);
          background: transparent;
          border: none;
          color: var(--color-text-muted);
          cursor: pointer;
          transition: all 0.2s ease;
          font-weight: 500;
          text-align: left;
          position: relative;
        }

        .nav-item:hover {
          background: rgba(255, 255, 255, 0.05);
          color: var(--color-text-main);
        }

        .nav-item.active {
          background: rgba(59, 130, 246, 0.15);
          color: var(--color-brand-primary);
          border-left: 3px solid var(--color-brand-primary);
        }

        .nav-item.danger:hover {
          color: var(--color-danger);
          background: rgba(239, 68, 68, 0.1);
        }

        .badge-dot {
          width: 8px;
          height: 8px;
          background-color: var(--color-danger);
          border-radius: 50%;
          position: absolute;
          right: 1rem;
          box-shadow: 0 0 8px var(--color-danger);
        }
      `}</style>
        </aside>
    );
};

export default Sidebar;
