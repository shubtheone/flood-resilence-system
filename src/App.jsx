import { useState } from 'react'
import Sidebar from './components/layout/Sidebar'
import Dashboard from './pages/Dashboard'
import { Monitor, Bell, Map as MapIcon, Package } from 'lucide-react'

// Placeholder components for other tabs
const Placeholder = ({ title, icon: Icon }) => (
  <div className="placeholder-container">
    <div className="glass-panel placeholder-content">
      <Icon size={48} className="placeholder-icon" />
      <h2>{title} Module</h2>
      <p>This feature is currently under development.</p>
    </div>
    <style>{`
      .placeholder-container {
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100%;
        color: var(--color-text-dim);
      }
      .placeholder-content {
        padding: 3rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
      }
      .placeholder-icon {
        color: var(--color-brand-primary);
        opacity: 0.5;
      }
    `}</style>
  </div>
)

function App() {
  const [activeTab, setActiveTab] = useState('dashboard')

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />
      case 'map':
        return <Placeholder title="Advanced Mapping" icon={MapIcon} />
      case 'alerts':
        return <Placeholder title="Alert Management" icon={Bell} />
      case 'resources':
        return <Placeholder title="Resource Coordination" icon={Package} />
      default:
        return <Dashboard />
    }
  }

  return (
    <div className="app-container">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="main-content">
        <header className="top-bar">
          <h1 className="page-title">
            {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
          </h1>
          <div className="user-profile">
            <div className="avatar">A</div>
            <span className="username">Admin User</span>
          </div>
        </header>
        {renderContent()}
      </main>

      <style>{`
        .top-bar {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 2rem;
          padding: 0.5rem 0;
        }

        .page-title {
          font-size: 1.5rem;
          font-weight: 600;
          color: var(--color-text-main);
        }

        .user-profile {
          display: flex;
          align-items: center;
          gap: 0.75rem;
        }

        .avatar {
          width: 36px;
          height: 36px;
          background: var(--gradient-accent); /* from global css? No, defined in first version but let's use brand */
          background: linear-gradient(135deg, var(--color-brand-primary), #60A5FA);
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          font-weight: 600;
          color: white;
          font-size: 0.875rem;
        }

        .username {
          font-weight: 500;
          color: var(--color-text-main);
        }
      `}</style>
    </div>
  )
}

export default App
