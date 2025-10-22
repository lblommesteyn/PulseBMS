import React, { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Cpu, Activity, Zap } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import api from '../api/client';

const DeviceDetail = () => {
    const { id } = useParams();
    const [telemetry, setTelemetry] = useState(null);
    const [history, setHistory] = useState([]);

    useEffect(() => {
        // Mocking rapid refresh for visual effect
        const interval = setInterval(() => {
            // In a real app we would append real data
            const now = new Date();
            const timeStr = now.getHours() + ':' + String(now.getMinutes()).padStart(2, '0') + ':' + String(now.getSeconds()).padStart(2, '0');

            // Simulating "Noise" if no backend
            const voltage = 400 + Math.random() * 5;
            const temp = 25 + Math.random();

            setHistory(prev => {
                const nav = [...prev, { time: timeStr, voltage, temp }];
                if (nav.length > 30) nav.shift();
                return nav;
            });

            setTelemetry({
                voltage: voltage,
                current: Math.random() * 50,
                temperature: temp,
                soc: 65,
                soh: 92
            });

        }, 1000);
        return () => clearInterval(interval);
    }, [id]);

    return (
        <div className="p-8 space-y-8">
            <Link to="/fleet" className="flex items-center gap-2 text-textMuted hover:text-white transition-colors">
                <ArrowLeft size={20} /> Back to Fleet
            </Link>

            <div className="flex justify-between items-end">
                <div>
                    <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-white/50">{id}</h1>
                    <div className="flex gap-4 mt-2">
                        <span className="px-2 py-1 rounded bg-secondary/10 text-secondary text-xs border border-secondary/20">ONLINE</span>
                        <span className="px-2 py-1 rounded bg-surface border border-white/10 text-textMuted text-xs">Firmware v2.4.1</span>
                    </div>
                </div>
                <div className="flex gap-4">
                    <button className="px-4 py-2 bg-danger/10 text-danger rounded-lg border border-danger/20 hover:bg-danger/20 transition-colors">Emergency Stop</button>
                    <button className="px-4 py-2 bg-primary text-white rounded-lg hover:bg-primary/90 transition-colors shadow-lg shadow-primary/20">Run Diagnostics</button>
                </div>
            </div>

            <div className="grid grid-cols-3 gap-6">
                <div className="glass-card flex items-center gap-4">
                    <div className="p-3 bg-blue-500/20 rounded-full text-blue-400">
                        <Zap size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-textMuted">Voltage</p>
                        <p className="text-2xl font-mono font-bold">{telemetry?.voltage.toFixed(1)} V</p>
                    </div>
                </div>
                <div className="glass-card flex items-center gap-4">
                    <div className="p-3 bg-red-500/20 rounded-full text-red-400">
                        <Activity size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-textMuted">Temperature</p>
                        <p className="text-2xl font-mono font-bold">{telemetry?.temperature.toFixed(1)} °C</p>
                    </div>
                </div>
                <div className="glass-card flex items-center gap-4">
                    <div className="p-3 bg-emerald-500/20 rounded-full text-emerald-400">
                        <Cpu size={24} />
                    </div>
                    <div>
                        <p className="text-sm text-textMuted">Digital Twin Error</p>
                        <p className="text-2xl font-mono font-bold">0.4%</p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6 h-[400px]">
                <div className="glass-card">
                    <h3 className="mb-4 font-semibold">Real-time Voltage</h3>
                    <ResponsiveContainer width="100%" height="90%">
                        <LineChart data={history}>
                            <CartesianGrid stroke="#ffffff05" />
                            <YAxis domain={['auto', 'auto']} stroke="#ffffff50" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#1d1d20', border: 'none' }} />
                            <Line type="monotone" dataKey="voltage" stroke="#3b82f6" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
                <div className="glass-card">
                    <h3 className="mb-4 font-semibold">Thermal Performance</h3>
                    <ResponsiveContainer width="100%" height="90%">
                        <LineChart data={history}>
                            <CartesianGrid stroke="#ffffff05" />
                            <YAxis domain={['auto', 'auto']} stroke="#ffffff50" fontSize={12} tickLine={false} axisLine={false} />
                            <Tooltip contentStyle={{ backgroundColor: '#1d1d20', border: 'none' }} />
                            <Line type="monotone" dataKey="temp" stroke="#ef4444" strokeWidth={2} dot={false} />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    )
}

export default DeviceDetail;
