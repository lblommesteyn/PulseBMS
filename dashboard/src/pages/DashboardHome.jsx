import React, { useEffect, useState } from 'react';
import { Zap, Activity, Battery, Thermometer } from 'lucide-react';
import StatCard from '../components/StatCard';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../api/client';

const DashboardHome = () => {
    const [data, setData] = useState([]);
    const [metrics, setMetrics] = useState({
        totalPower: 0,
        avgSoH: 0,
        activeDevices: 0,
        dailyEnergy: 0
    });

    // Simulate real-time data for the chart
    useEffect(() => {
        const interval = setInterval(() => {
            const now = new Date();
            const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0');
            const power = 400 + Math.random() * 100;

            setData(prev => {
                const newData = [...prev, { time: timeStr, power: power, demand: power * 0.8 }];
                if (newData.length > 20) newData.shift();
                return newData;
            });

            // Polling actual backend (simulated result for now)
            api.get('/fleet/status').then(res => {
                const devices = res.data.devices || [];
                if (devices.length > 0) {
                    const totalP = devices.reduce((acc, d) => acc + (d.measurements?.power || 0), 0) / 1000;
                    const avgH = devices.reduce((acc, d) => acc + (d.measurements?.soh || 0), 0) / devices.length;
                    setMetrics({
                        totalPower: Math.abs(totalP).toFixed(1),
                        avgSoH: avgH.toFixed(1),
                        activeDevices: devices.length,
                        dailyEnergy: (Math.random() * 500 + 4000).toFixed(0) // Mock
                    });
                }
            }).catch(err => console.log("Backend offline"));

        }, 2000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="p-8 space-y-8">
            <div>
                <h1 className="text-3xl font-bold mb-2">Sites Overview</h1>
                <p className="text-textMuted">Real-time fleet performance monitoring</p>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard title="Total Power Load" value={metrics.totalPower} unit="kW" icon={Zap} trend="up" trendValue="12%" />
                <StatCard title="Daily Energy" value={metrics.dailyEnergy} unit="kWh" icon={Activity} trend="up" trendValue="5%" />
                <StatCard title="Avg Fleet Health" value={metrics.avgSoH} unit="%" icon={Battery} trend="down" trendValue="0.1%" />
                <StatCard title="Active Systems" value={metrics.activeDevices} unit="Units" icon={Thermometer} />
            </div>

            {/* Main Chart */}
            <div className="glass-card h-[400px]">
                <h3 className="text-lg font-semibold mb-6">Power Demand vs Supply</h3>
                <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={data}>
                        <defs>
                            <linearGradient id="colorPower" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                            </linearGradient>
                            <linearGradient id="colorDemand" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid vertical={false} stroke="#ffffff10" />
                        <XAxis dataKey="time" stroke="#9ca3af" tickLine={false} axisLine={false} />
                        <YAxis stroke="#9ca3af" tickLine={false} axisLine={false} />
                        <Tooltip
                            contentStyle={{ backgroundColor: '#1d1d20', border: '1px solid #333', borderRadius: '8px' }}
                            itemStyle={{ color: '#fff' }}
                        />
                        <Area type="monotone" dataKey="power" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorPower)" name="Supply (kW)" />
                        <Area type="monotone" dataKey="demand" stroke="#10b981" strokeWidth={3} fillOpacity={1} fill="url(#colorDemand)" name="Demand (kW)" />
                    </AreaChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default DashboardHome;
