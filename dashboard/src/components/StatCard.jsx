import React from 'react';
import { ArrowUp, ArrowDown } from 'lucide-react';

const StatCard = ({ title, value, unit, trend, icon: Icon, trendValue }) => {
    const isPositive = trend === 'up';

    return (
        <div className="glass-card flex flex-col relative overflow-hidden group">
            <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                {Icon && <Icon size={64} />}
            </div>

            <h3 className="text-textMuted text-sm font-medium mb-1">{title}</h3>
            <div className="flex items-baseline gap-1">
                <span className="text-3xl font-bold tracking-tight text-white">{value}</span>
                {unit && <span className="text-textMuted text-sm">{unit}</span>}
            </div>

            {trendValue && (
                <div className={`flex items-center gap-1 mt-3 text-sm ${isPositive ? 'text-secondary' : 'text-danger'}`}>
                    {isPositive ? <ArrowUp size={16} /> : <ArrowDown size={16} />}
                    <span className="font-medium">{trendValue}</span>
                    <span className="text-textMuted/60 ml-1">vs last hr</span>
                </div>
            )}
        </div>
    );
};

export default StatCard;
