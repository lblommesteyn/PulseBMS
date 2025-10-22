import axios from 'axios';

const api = axios.create({
    baseURL: 'http://localhost:8000/api/v1',
    timeout: 5000,
});

export const fetchFleetStatus = async () => {
    // Mock response if backend is offline or empty for dev
    try {
        const { data } = await api.get('/fleet/status');
        return data;
    } catch (error) {
        console.warn("Backend unavailable, returning empty data");
        return { devices: [] };
    }
};

export const fetchDeviceTelemetry = async (deviceId) => {
    try {
        const { data } = await api.get(`/telemetry/${deviceId}/latest`);
        return data;
    } catch (error) {
        return null;
    }
};

export default api;
