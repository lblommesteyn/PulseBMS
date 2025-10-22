/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0a0b',
        surface: '#121214',
        surfaceHighlight: '#1d1d20',
        primary: '#3b82f6', // Blue
        primaryGlow: '#60a5fa',
        secondary: '#10b981', // Emerald
        accent: '#f59e0b', // Amber
        danger: '#ef4444', // Red
        text: '#f3f4f6',
        textMuted: '#9ca3af',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      }
    },
  },
  plugins: [],
}
