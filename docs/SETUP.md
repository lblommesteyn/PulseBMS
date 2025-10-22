# PulseBMS Enhanced - Setup Guide

## System Requirements

### Hardware Requirements
- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space
- **Network**: Stable internet connection for external dependencies

### Software Requirements
- **Python**: 3.8 or higher
- **MongoDB**: 4.4 or higher
- **Redis**: 6.0 or higher
- **Git**: Latest version

## Installation Guide

### 1. Database Setup

#### MongoDB Installation

**Windows:**
```bash
# Download MongoDB Community Server from mongodb.com
# Run the installer and follow setup wizard
# Start MongoDB service
net start MongoDB
```

**macOS:**
```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb/brew/mongodb-community
```

**Linux (Ubuntu/Debian):**
```bash
# Import MongoDB GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-4.4.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-4.4.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### Redis Installation

**Windows:**
```bash
# Download Redis from https://github.com/microsoftarchive/redis/releases
# Run the installer
# Start Redis service
redis-server
```

**macOS:**
```bash
# Using Homebrew
brew install redis
brew services start redis
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

#### Environment Variables
Create a `.env` file in the project root:

```bash
# Database Configuration
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379

# Service URLs
BACKEND_URL=http://localhost:8000
DIGITAL_TWIN_URL=http://localhost:8001
COORDINATOR_URL=http://localhost:8002

# MQTT Configuration (optional)
MQTT_BROKER=localhost
MQTT_PORT=1883

# Safety Configuration
EMERGENCY_STOP_ENABLED=true
SAFETY_VOLTAGE_MIN=300.0
SAFETY_VOLTAGE_MAX=450.0
SAFETY_TEMPERATURE_MAX=60.0
```

#### MongoDB Database Setup
```bash
# Connect to MongoDB
mongosh

# Create database and collections
use pulsebms_enhanced

# Create indexes for better performance
db.telemetry.createIndex({"device_id": 1, "timestamp": -1})
db.devices.createIndex({"device_id": 1})
db.predictions.createIndex({"device_id": 1, "timestamp": -1})
```

### 4. Verification

#### Database Connectivity Test
```bash
# Test MongoDB connection
python -c "import motor.motor_asyncio; print('MongoDB driver: OK')"

# Test Redis connection
python -c "import redis; r=redis.Redis(); r.ping(); print('Redis: OK')"
```

#### Service Health Check
```bash
# Start all services in separate terminals

# Terminal 1: Backend
cd backend && python main.py

# Terminal 2: Digital Twin
cd digital-twin && python digital_twin_service.py

# Terminal 3: Coordinator
cd coordination && python coordinator_service.py

# Terminal 4: Run health check
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health
```

## Development Setup

### Code Quality Tools
```bash
# Install development dependencies
pip install black flake8 pytest mypy

# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

### Testing
```bash
# Run unit tests
pytest testing/

# Run integration tests
python testing/integration_tests.py

# Run HIL tests
python testing/hil_testing_framework.py
```

## Production Deployment

### Docker Setup (Optional)
```bash
# Build containers
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
```

### Performance Tuning
- **MongoDB**: Configure appropriate connection pool sizes
- **Redis**: Set memory limits and persistence settings
- **Python**: Use production WSGI server (e.g., Gunicorn)

## Troubleshooting

### Common Issues

**MongoDB Connection Failed:**
- Verify MongoDB service is running
- Check firewall settings
- Ensure correct connection string

**Redis Connection Failed:**
- Verify Redis service is running
- Check Redis configuration file
- Ensure correct Redis URL

**Import Errors:**
- Verify virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt --force-reinstall`

**Port Conflicts:**
- Check if ports 8000, 8001, 8002 are available
- Modify service URLs in configuration if needed

### Logs and Debugging
- **Application logs**: Check console output for each service
- **MongoDB logs**: Check MongoDB log files
- **Redis logs**: Check Redis log files
- **System logs**: Check system event logs for service issues

## Support

For additional support:
1. Check the [documentation](docs/)
2. Review [API specifications](docs/API_SPECIFICATIONS.md)
3. Consult [troubleshooting guide](docs/TROUBLESHOOTING.md)
4. Open an issue on GitHub

## Security Considerations

- **Database Security**: Enable authentication in production
- **Network Security**: Use TLS/SSL for external connections
- **API Security**: Implement authentication and rate limiting
- **Data Privacy**: Encrypt sensitive battery data
