import React, { useEffect, useState } from 'react';
import { Battery, AlertTriangle, CheckCircle2 } from 'lucide-react';
import api from '../api/client';
import { clsx } from 'clsx';
import { Link } from 'react-router-dom';

const FleetView = () => {
    const [devices, setDevices] = useState([]);

    useEffect(() => {
        const fetchDevices = async () => {
            try {
                // Mock devices if backend empty
                const res = await api.get('/fleet/status');
                if (res.data.devices && res.data.devices.length > 0) {
                    setDevices(res.data.devices);
                } else {
                    // Fallback visual mock
                    setDevices([
                        { device_id: 'sim_battery_001', measurements: { soc: 78, soh: 94, temperature: 28, power: 4500 } },
                        { device_id: 'sim_battery_002', measurements: { soc: 45, soh: 91, temperature: 32, power: -3200 } },
                        { device_id: 'sim_battery_003', measurements: { soc: 12, soh: 88, temperature: 35, power: 0 } },
                    ]);
                }
            } catch (e) {
                console.error(e);
            }
        };

        fetchDevices();
        const interval = setInterval(fetchDevices, 2000);
        return () => clearInterval(interval);
    }, []);

    const getStatusColor = (d) => {
        if (d.measurements?.soc < 20) return 'text-danger';
        if (d.measurements?.temperature > 40) return 'text-accent';
        return 'text-secondary';
    };

    return (
        <div className="p-8">
            <h1 className="text-3xl font-bold mb-8">Battery Fleet</h1>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {devices.map((device) => {
                    const m = device.measurements || {};
                    const isCharging = m.power < 0;

                    return (
                        <Link to={`/device/${device.device_id}`} key={device.device_id}>
                            <div className="glass-card hover:bg-white/5 cursor-pointer group">
                                <div className="flex justify-between items-start mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-lg bg-surfaceHighlight flex items-center justify-center">
                                            <Battery className={getStatusColor(device)} size={20} />
                                        </div>
                                        <div>
                                            <h3 className="font-semibold text-white group-hover:text-primary transition-colors">{device.device_id}</h3>
                                            <p className="text-xs text-textMuted">LFP Pack • Rack 2B</p>
                                        </div>
                                    </div>
                                    <div className={clsx("px-2 py-1 rounded text-xs font-medium", isCharging ? "bg-secondary/10 text-secondary" : "bg-primary/10 text-primary")}>
                                        {isCharging ? 'CHARGING' : 'DISCHARGING'}
                                    </div>
                                </div>

                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <p className="text-textMuted text-xs mb-1">SoC</p>
                                        <div className="text-2xl font-bold">{m.soc?.toFixed(1)}%</div>
                                    </div>
                                    <div>
                                        <p className="text-textMuted text-xs mb-1">Temperature</p>
                                        <div className="text-2xl font-bold">{m.temperature?.toFixed(1)}°C</div>
                                    </div>
                                    <div>
                                        <p className="text-textMuted text-xs mb-1">Power</p>
                                        <div className="text-lg font-mono text-white/80">{(m.power / 1000)?.toFixed(2)} kW</div>
                                    </div>
                                    <div>
                                        <p className="text-textMuted text-xs mb-1">Health</p>
                                        <div className="text-lg font-mono text-secondary">{m.soh?.toFixed(1)}%</div>
                                    </div>
                                </div>

                                {/* SoC Bar */}
                                <div className="mt-4 h-1.5 w-full bg-surfaceHighlight rounded-full overflow-hidden">
                                    <div
                                        className={clsx("h-full rounded-full transition-all duration-500", m.soc < 20 ? "bg-danger" : "bg-primary")}
                                        style={{ width: `${m.soc}%` }}
                                    />
                                </div>
                            </div>
                        </Link>
                    )
                })}
            </div>
        </div>
    );
};

export default FleetView;
