import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Battery, Zap, Settings as SettingsIcon, Activity, Bell } from 'lucide-react';
import { clsx } from 'clsx';

const NavItem = ({ to, icon: Icon, label }) => (
    <NavLink
        to={to}
        className={({ isActive }) =>
            clsx(
                "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-200",
                isActive
                    ? "bg-primary/20 text-primary border border-primary/20"
                    : "text-textMuted hover:bg-white/5 hover:text-text"
            )
        }
    >
        <Icon size={20} />
        <span className="font-medium">{label}</span>
    </NavLink>
);

const Sidebar = () => {
    return (
        <div className="w-64 h-screen bg-surface border-r border-white/5 flex flex-col p-6 fixed left-0 top-0">
            <div className="flex items-center gap-3 mb-10">
                <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-primary to-accent flex items-center justify-center shadow-lg shadow-primary/20">
                    <Zap className="text-white" size={24} fill="currentColor" />
                </div>
                <div>
                    <h1 className="text-xl font-bold font-sans tracking-tight">PulseBMS</h1>
                    <p className="text-xs text-primary font-medium tracking-wide">ENHANCED</p>
                </div>
            </div>

            <nav className="flex flex-col gap-2 flex-1">
                <NavItem to="/" icon={LayoutDashboard} label="Dashboard" />
                <NavItem to="/fleet" icon={Battery} label="Fleet View" />
                <NavItem to="/analytics" icon={Activity} label="Analytics" />
                <NavItem to="/alerts" icon={Bell} label="Alerts" />
            </nav>

            <div className="mt-auto pt-6 border-t border-white/5">
                <NavItem to="/settings" icon={SettingsIcon} label="Settings" />
            </div>
        </div>
    );
};

export default Sidebar;
