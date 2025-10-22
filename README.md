# PulseBMS Enhanced - Adaptive Battery Management System

## Overview

PulseBMS Enhanced is an advanced battery management system designed specifically for second-life EV batteries (SLBs). It combines reinforcement learning, physics-based digital twins, and real-time telemetry to optimize charge/discharge operations across heterogeneous battery fleets while ensuring safety and extending battery life by up to 25%.

## Key Features

- **Adaptive RL-Powered BMS**: Reinforcement learning algorithms for intelligent charge/discharge optimization
- **Digital Twin Integration**: PyBaMM-based physics models for accurate battery forecasting
- **Real-Time Telemetry**: MQTT-based streaming for low-latency, reliable communication
- **Edge Computing**: Local SoC/SoH estimation and safety interlocks on edge devices
- **Fleet Coordination**: Site-level power allocation across mixed SLB fleets
- **Safety-First Design**: Degradation-aware control with built-in safety constraints
- **Live Dashboard**: Real-time monitoring, health predictions, and actionable insights

## Architecture

The system consists of five main components:

1. **SLB Battery Packs**: Varying chemistries and health states from retired EV batteries
2. **Edge Devices**: RTOS-powered local monitoring, SoC/SoH estimation, and RL policy execution
3. **Digital Twin Service**: Cloud-hosted PyBaMM models for short-horizon forecasting
4. **Coordinator Service**: Site-level power demand allocation with safety constraints
5. **Dashboard & Analytics**: Live telemetry visualization and predictive insights

## Quick Start

### Prerequisites
- Python 3.8+
- MongoDB (for telemetry storage)
- Redis (for caching and coordination)
- MQTT broker (optional, for telemetry streaming)

### Installation

```bash
# Clone the repository
git clone https://github.com/lblommesteyn/PulseBMS.git
cd PulseBMS-Enhanced

# Install Python dependencies
pip install -r requirements.txt

# Start MongoDB and Redis services
# (Instructions vary by OS - see docs/SETUP.md)
```

### Running the System

```bash
# 1. Start the backend API server
cd backend && python main.py

# 2. Start the digital twin service
cd digital-twin && python digital_twin_service.py

# 3. Start the coordinator service
cd coordination && python coordinator_service.py

# 4. Run edge device simulator
cd edge && python edge_device.py

# 5. Run integration tests
cd testing && python integration_tests.py
```

### Testing the System

```bash
# Run the comprehensive integration test suite
python testing/integration_tests.py

# Run hardware-in-the-loop tests
python testing/hil_testing_framework.py
```

## Project Structure

```
PulseBMS-Enhanced/
├── docs/                   # Design documentation
├── backend/               # FastAPI backend services
├── edge/                  # Edge device code (RTOS simulation)
├── digital-twin/          # PyBaMM integration and physics-based models
├── optimization/          # MPC allocator and RL policy algorithms
├── coordination/          # Fleet coordination service integrating all optimization
├── testing/               # Integration tests and HIL testing framework
├── routers/               # API routing and telemetry handlers
├── backend/               # FastAPI backend services
├── dashboard/             # React-based live dashboard (coming soon)
├── docs/                  # Comprehensive design documentation
└── requirements.txt       # Python dependencies
```

## Development Status

- ✅ Project initialization and architecture design
- ✅ MQTT telemetry streaming implementation
- ✅ Edge device SoC/SoH estimators and simulation
- ✅ Digital twin service with PyBaMM integration
- ✅ MPC baseline allocator with safety constraints
- ✅ Safe RL policy training and deployment
- ✅ Coordinator service integrating MPC, RL, and Digital Twin
- ✅ Hardware-in-the-loop (HIL) testing framework
- ✅ Comprehensive integration test suite
- 📋 React-based live dashboard (in progress)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for contribution guidelines.
