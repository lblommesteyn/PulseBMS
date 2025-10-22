import React from 'react';
import { AlertTriangle, CheckCircle, Info, XCircle } from 'lucide-react';

const alerts = [
    { id: 1, type: 'critical', message: 'Device BAT-004: Thermal Runaway Risk Detected', time: '10 mins ago' },
    { id: 2, type: 'warning', message: 'Grid frequency deviation detected > 0.5Hz', time: '1 hour ago' },
    { id: 3, type: 'info', message: 'System optimization scheduled for 02:00 AM', time: '2 hours ago' },
    { id: 4, type: 'success', message: 'Firmware update completed for 12 devices', time: '5 hours ago' },
    { id: 5, type: 'warning', message: 'Device BAT-001: SoH dropped below 85%', time: '1 day ago' },
];

const Alerts = () => {
    return (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-white">System Alerts</h1>
                <span className="bg-red-500/20 text-red-400 px-3 py-1 rounded-full text-sm border border-red-500/50">
                    2 Active Critical Alerts
                </span>
            </div>

            <div className="space-y-3">
                {alerts.map((alert) => (
                    <div key={alert.id} className="bg-glass-panel border border-glass-border p-4 rounded-xl flex items-start gap-4 hover:bg-white/5 transition-colors">
                        <div className="mt-1">
                            {alert.type === 'critical' && <XCircle className="text-red-500" size={24} />}
                            {alert.type === 'warning' && <AlertTriangle className="text-yellow-500" size={24} />}
                            {alert.type === 'info' && <Info className="text-blue-500" size={24} />}
                            {alert.type === 'success' && <CheckCircle className="text-green-500" size={24} />}
                        </div>
                        <div className="flex-1">
                            <p className="text-lg font-medium text-white">{alert.message}</p>
                            <p className="text-sm text-gray-500">{alert.time}</p>
                        </div>
                        <button className="text-sm text-gray-400 hover:text-white underline">
                            Dismiss
                        </button>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default Alerts;
