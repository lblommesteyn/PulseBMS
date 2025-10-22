import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Calendar } from 'lucide-react';

const data = [
    { name: 'Jan', degradation: 98, savings: 4000 },
    { name: 'Feb', degradation: 97.5, savings: 4500 },
    { name: 'Mar', degradation: 97.2, savings: 4200 },
    { name: 'Apr', degradation: 96.8, savings: 5100 },
    { name: 'May', degradation: 96.5, savings: 4800 },
    { name: 'Jun', degradation: 96.1, savings: 6000 },
    { name: 'Jul', degradation: 95.8, savings: 7200 },
];

const Analytics = () => {
    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-neon-blue to-neon-purple bg-clip-text text-transparent">
                System Analytics
            </h1>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* ROI Analysis */}
                <div className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                    <h2 className="text-xl font-semibold mb-4 text-white">Projected ROI & Savings</h2>
                    <div className="h-80 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={data}>
                                <defs>
                                    <linearGradient id="colorSavings" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                                        <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#888" />
                                <YAxis stroke="#888" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #333' }}
                                />
                                <Area type="monotone" dataKey="savings" stroke="#10b981" fillOpacity={1} fill="url(#colorSavings)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* SOH Trends */}
                <div className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                    <h2 className="text-xl font-semibold mb-4 text-white">Fleet SOH Degradation</h2>
                    <div className="h-80 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={data}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                                <XAxis dataKey="name" stroke="#888" />
                                <YAxis domain={[90, 100]} stroke="#888" />
                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid #333' }}
                                />
                                <Line type="monotone" dataKey="degradation" stroke="#f472b6" strokeWidth={2} dot={{ r: 4 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            <div className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-semibold text-white">Detailed Reports</h2>
                    <button className="flex items-center gap-2 px-4 py-2 bg-neon-blue/20 hover:bg-neon-blue/40 text-neon-blue rounded-lg transition-colors">
                        <Calendar size={18} />
                        Download CSV
                    </button>
                </div>
                <table className="w-full text-left border-collapse">
                    <thead>
                        <tr className="border-b border-glass-border text-gray-400">
                            <th className="p-4">Period</th>
                            <th className="p-4">Total Energy (MWh)</th>
                            <th className="p-4">Efficiency</th>
                            <th className="p-4">Revenue</th>
                        </tr>
                    </thead>
                    <tbody className="text-gray-300">
                        <tr className="border-b border-glass-border/30 hover:bg-white/5">
                            <td className="p-4">July 2026</td>
                            <td className="p-4">45.2</td>
                            <td className="p-4">97.8%</td>
                            <td className="p-4 text-green-400">+$7,200</td>
                        </tr>
                        <tr className="border-b border-glass-border/30 hover:bg-white/5">
                            <td className="p-4">June 2026</td>
                            <td className="p-4">42.1</td>
                            <td className="p-4">98.1%</td>
                            <td className="p-4 text-green-400">+$6,000</td>
                        </tr>
                        <tr className="hover:bg-white/5">
                            <td className="p-4">May 2026</td>
                            <td className="p-4">38.5</td>
                            <td className="p-4">98.2%</td>
                            <td className="p-4 text-green-400">+$4,800</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Analytics;
