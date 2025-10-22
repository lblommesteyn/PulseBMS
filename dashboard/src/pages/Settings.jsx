import React from 'react';
import { Save, RefreshCw, Shield, Zap } from 'lucide-react';

const Settings = () => {
    return (
        <div className="max-w-4xl mx-auto space-y-8">
            <h1 className="text-3xl font-bold text-white mb-8">System Configuration</h1>

            {/* General */}
            <section className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                <div className="flex items-center gap-3 mb-6">
                    <Zap className="text-neon-blue" />
                    <h2 className="text-xl font-semibold text-white">Power Policy</h2>
                </div>
                <div className="space-y-4">
                    <div className="flex flex-col gap-2">
                        <label className="text-gray-400">Grid Import Limit (kW)</label>
                        <input type="number" defaultValue={500} className="bg-black/40 border border-gray-700 rounded-lg p-2 text-white focus:outline-none focus:border-neon-blue" />
                    </div>
                    <div className="flex flex-col gap-2">
                        <label className="text-gray-400">Optimization Strategy</label>
                        <select className="bg-black/40 border border-gray-700 rounded-lg p-2 text-white focus:outline-none focus:border-neon-blue">
                            <option>Maximize Revenue</option>
                            <option>Maximize Battery Life</option>
                            <option>Grid Stability Only</option>
                        </select>
                    </div>
                </div>
            </section>

            {/* Safety */}
            <section className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                <div className="flex items-center gap-3 mb-6">
                    <Shield className="text-red-400" />
                    <h2 className="text-xl font-semibold text-white">Safety Thresholds</h2>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="text-gray-400 block mb-2">Max Cell Temp (°C)</label>
                        <input type="number" defaultValue={45} className="w-full bg-black/40 border border-gray-700 rounded-lg p-2 text-white" />
                    </div>
                    <div>
                        <label className="text-gray-400 block mb-2">Min Voltage Cutoff (V)</label>
                        <input type="number" defaultValue={2.8} className="w-full bg-black/40 border border-gray-700 rounded-lg p-2 text-white" />
                    </div>
                </div>
            </section>

            {/* Firmware */}
            <section className="bg-glass-panel border border-glass-border rounded-xl p-6 backdrop-blur-md">
                <div className="flex items-center gap-3 mb-6">
                    <RefreshCw className="text-purple-400" />
                    <h2 className="text-xl font-semibold text-white">Firmware Updates</h2>
                </div>
                <div className="flex items-center justify-between bg-white/5 p-4 rounded-lg">
                    <div>
                        <p className="text-white font-medium">BMS_Edge_v2.4.1</p>
                        <p className="text-sm text-gray-400">Current Version</p>
                    </div>
                    <button className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg text-white font-medium hover:opacity-90">
                        Check for Updates
                    </button>
                </div>
            </section>

            <div className="flex justify-end pt-4">
                <button className="flex items-center gap-2 px-8 py-3 bg-neon-blue text-black font-bold rounded-lg hover:bg-cyan-400 transition-colors">
                    <Save size={20} />
                    Save Changes
                </button>
            </div>
        </div>
    );
};

export default Settings;
