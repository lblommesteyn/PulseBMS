import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import DashboardHome from './pages/DashboardHome';
import FleetView from './pages/FleetView';
import DeviceDetail from './pages/DeviceDetail';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import Alerts from './pages/Alerts';

function App() {
    return (
        <Router>
            <div className="flex bg-background min-h-screen text-text">
                <Sidebar />
                <main className="flex-1 ml-64">
                    <Routes>
                        <Route path="/" element={<DashboardHome />} />
                        <Route path="/fleet" element={<FleetView />} />
                        <Route path="/device/:id" element={<DeviceDetail />} />
                        <Route path="/analytics" element={<Analytics />} />
                        <Route path="/settings" element={<Settings />} />
                        <Route path="/alerts" element={<Alerts />} />
                        <Route path="*" element={<div className="p-10">404: Not Found</div>} />
                    </Routes>
                </main>
            </div>
        </Router>
    );
}

export default App;
