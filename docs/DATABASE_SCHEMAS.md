# PulseBMS Enhanced - Database Schemas

## Overview

PulseBMS Enhanced uses a hybrid database approach:
- **MongoDB**: High-volume telemetry data, time-series optimization
- **PostgreSQL**: Configuration, relationships, ACID transactions
- **Redis**: Caching, session management, real-time data

## MongoDB Collections

### Telemetry Collection

```javascript
// Collection: telemetry_data
{
  "_id": ObjectId("..."),
  "site_id": "site_001",
  "device_id": "battery_pack_001",
  "timestamp": ISODate("2024-01-15T14:30:00Z"),
  
  // Pack-level measurements
  "measurements": {
    "voltage": 400.5,           // V
    "current": -25.3,           // A (negative = charging)
    "power": -10126.5,          // W
    "temperature": 23.7,        // °C
    "soc": 67.5,               // %
    "soh": 87.2,               // %
    "internal_resistance": 0.045 // Ohm
  },
  
  // Cell-level data (compressed for storage efficiency)
  "cell_data": {
    "voltages": [3.45, 3.46, 3.44, ...], // Individual cell voltages
    "temperatures": [23.1, 23.9, 22.8, ...], // Individual cell temps
    "balancing_status": [0, 1, 0, ...]  // 0=inactive, 1=balancing
  },
  
  // Safety and alarms
  "safety": {
    "alarm_flags": ["high_temp_warning"],
    "protection_status": {
      "overvoltage": false,
      "undervoltage": false,
      "overcurrent": false,
      "overtemperature": false
    }
  },
  
  // Metadata and quality
  "metadata": {
    "data_quality": 0.98,      // Quality score 0-1
    "source": "edge_device",
    "firmware_version": "1.2.3",
    "communication_latency": 85 // ms
  },
  
  // Indexes for time-series queries
  "day": "2024-01-15",          // For daily partitioning
  "hour": 14                    // For hourly aggregation
}

// Indexes
db.telemetry_data.createIndex({ "site_id": 1, "device_id": 1, "timestamp": -1 })
db.telemetry_data.createIndex({ "day": 1, "hour": 1 })
db.telemetry_data.createIndex({ "timestamp": -1 }, { expireAfterSeconds: 7776000 }) // 90 days
```

### Aggregated Metrics Collection

```javascript
// Collection: metrics_hourly
{
  "_id": ObjectId("..."),
  "site_id": "site_001",
  "device_id": "battery_pack_001",
  "timestamp": ISODate("2024-01-15T14:00:00Z"),
  "period": "hourly",
  
  "aggregates": {
    "voltage": {
      "min": 398.2,
      "max": 402.1,
      "avg": 400.1,
      "std": 1.2
    },
    "current": {
      "min": -30.5,
      "max": -20.1,
      "avg": -25.3,
      "std": 2.8
    },
    "power": {
      "min": -12240,
      "max": -8045,
      "avg": -10126,
      "std": 987
    },
    "temperature": {
      "min": 22.1,
      "max": 25.3,
      "avg": 23.7,
      "std": 0.8
    },
    "soc": {
      "min": 65.2,
      "max": 69.8,
      "avg": 67.5,
      "std": 1.1
    }
  },
  
  "energy_throughput": {
    "charged": 10.5,    // kWh
    "discharged": 8.3,  // kWh
    "net": 2.2          // kWh
  },
  
  "cycle_data": {
    "partial_cycles": 0.15,
    "equivalent_full_cycles": 0.02
  },
  
  "alarm_summary": {
    "total_alarms": 3,
    "alarm_types": ["high_temp_warning", "voltage_imbalance"],
    "alarm_duration": 180 // seconds
  }
}
```

### Digital Twin Predictions

```javascript
// Collection: digital_twin_predictions
{
  "_id": ObjectId("..."),
  "device_id": "battery_pack_001",
  "prediction_timestamp": ISODate("2024-01-15T14:30:00Z"),
  "horizon_hours": 24,
  
  "initial_state": {
    "soc": 67.5,
    "soh": 87.2,
    "temperature": 23.7,
    "internal_resistance": 0.045
  },
  
  "power_profile": [-10000, -8000, -5000, 0, 5000, ...], // W for next 24h
  
  "predictions": {
    "voltage_profile": [400.1, 398.5, 396.2, ...],
    "temperature_profile": [23.7, 24.1, 24.8, ...],
    "soc_profile": [67.5, 69.2, 71.1, ...],
    "efficiency_profile": [0.95, 0.94, 0.93, ...]
  },
  
  "degradation_forecast": {
    "capacity_fade_percent": 0.012,    // Expected fade over horizon
    "resistance_growth_percent": 0.008,
    "cycle_life_consumed": 0.05,
    "calendar_life_consumed": 0.001
  },
  
  "model_metadata": {
    "model_type": "P2D",
    "calibration_date": ISODate("2024-01-14T12:00:00Z"),
    "calibration_score": 0.94,
    "parameters": {
      "diffusion_coefficient": 3.9e-14,
      "reaction_rate": 2.334e-11,
      "conductivity": 1.0
    }
  }
}
```

## PostgreSQL Tables

### Device Configuration

```sql
-- Table: device_configurations
CREATE TABLE device_configurations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) UNIQUE NOT NULL,
    site_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    
    -- Hardware specifications
    chemistry chemistry_enum NOT NULL,
    nominal_capacity DECIMAL(10,2) NOT NULL, -- Ah
    nominal_voltage DECIMAL(10,2) NOT NULL,  -- V
    max_charge_power DECIMAL(10,2) NOT NULL, -- W
    max_discharge_power DECIMAL(10,2) NOT NULL, -- W
    
    -- Physical configuration
    series_cells INTEGER NOT NULL,
    parallel_cells INTEGER NOT NULL,
    
    -- Location and metadata
    location VARCHAR(255),
    rack_position VARCHAR(50),
    installation_date DATE,
    manufacturing_date DATE,
    
    -- Status and versioning
    status device_status_enum DEFAULT 'offline',
    firmware_version VARCHAR(50),
    last_seen TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255),
    
    CONSTRAINT valid_capacity CHECK (nominal_capacity > 0),
    CONSTRAINT valid_voltage CHECK (nominal_voltage > 0),
    CONSTRAINT valid_cells CHECK (series_cells > 0 AND parallel_cells > 0)
);

-- Indexes
CREATE INDEX idx_device_site ON device_configurations(site_id);
CREATE INDEX idx_device_status ON device_configurations(status);
CREATE INDEX idx_device_last_seen ON device_configurations(last_seen);

-- Enums
CREATE TYPE chemistry_enum AS ENUM ('LFP', 'NMC', 'LCO', 'NCA');
CREATE TYPE device_status_enum AS ENUM ('online', 'offline', 'error', 'maintenance');
```

### Safety Constraints

```sql
-- Table: safety_constraints
CREATE TABLE safety_constraints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL REFERENCES device_configurations(device_id),
    
    -- Voltage constraints
    max_cell_voltage DECIMAL(5,3) DEFAULT 4.2,
    min_cell_voltage DECIMAL(5,3) DEFAULT 2.5,
    max_pack_voltage DECIMAL(8,2),
    min_pack_voltage DECIMAL(8,2),
    
    -- Current constraints
    max_charge_current DECIMAL(8,2) DEFAULT 50.0,
    max_discharge_current DECIMAL(8,2) DEFAULT 100.0,
    
    -- Temperature constraints
    max_cell_temperature DECIMAL(5,2) DEFAULT 60.0,
    min_cell_temperature DECIMAL(5,2) DEFAULT -20.0,
    max_temperature_delta DECIMAL(5,2) DEFAULT 10.0,
    
    -- SoC constraints
    max_soc DECIMAL(5,2) DEFAULT 95.0,
    min_soc DECIMAL(5,2) DEFAULT 5.0,
    
    -- Power constraints
    max_charge_power DECIMAL(10,2),
    max_discharge_power DECIMAL(10,2),
    
    -- Degradation constraints
    max_cycle_depth DECIMAL(5,2) DEFAULT 80.0,
    max_c_rate DECIMAL(5,2) DEFAULT 1.0,
    
    -- Metadata
    active BOOLEAN DEFAULT true,
    effective_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    effective_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    
    CONSTRAINT valid_voltage_range CHECK (min_cell_voltage < max_cell_voltage),
    CONSTRAINT valid_temp_range CHECK (min_cell_temperature < max_cell_temperature),
    CONSTRAINT valid_soc_range CHECK (min_soc < max_soc),
    CONSTRAINT valid_soc_bounds CHECK (min_soc >= 0 AND max_soc <= 100)
);
```

### Optimization Policies

```sql
-- Table: optimization_policies
CREATE TABLE optimization_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    policy_type policy_type_enum NOT NULL,
    
    -- Policy configuration
    config JSONB NOT NULL,
    
    -- RL-specific fields
    model_version VARCHAR(50),
    training_date TIMESTAMP WITH TIME ZONE,
    performance_metrics JSONB,
    
    -- MPC-specific fields
    horizon_hours INTEGER,
    update_interval_seconds INTEGER,
    
    -- Status and lifecycle
    status policy_status_enum DEFAULT 'inactive',
    deployed_at TIMESTAMP WITH TIME ZONE,
    performance_score DECIMAL(5,4), -- 0-1 score
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by VARCHAR(255),
    
    CONSTRAINT valid_horizon CHECK (horizon_hours > 0),
    CONSTRAINT valid_interval CHECK (update_interval_seconds > 0),
    CONSTRAINT valid_score CHECK (performance_score >= 0 AND performance_score <= 1)
);

CREATE TYPE policy_type_enum AS ENUM ('MPC', 'RL', 'HYBRID');
CREATE TYPE policy_status_enum AS ENUM ('active', 'inactive', 'testing', 'deprecated');
```

### Command History

```sql
-- Table: command_history
CREATE TABLE command_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    command_id VARCHAR(255) UNIQUE NOT NULL,
    site_id VARCHAR(255) NOT NULL,
    device_id VARCHAR(255) NOT NULL,
    
    -- Command details
    command_type command_type_enum NOT NULL,
    parameters JSONB,
    
    -- Execution tracking
    issued_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Results
    success BOOLEAN,
    response_message TEXT,
    response_data JSONB,
    
    -- Timeout and retry
    timeout_seconds INTEGER DEFAULT 30,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    
    -- Source tracking
    issued_by VARCHAR(255),
    source_system VARCHAR(100),
    
    CONSTRAINT valid_timeout CHECK (timeout_seconds > 0),
    CONSTRAINT valid_retries CHECK (retry_count >= 0 AND max_retries >= 0)
);

CREATE TYPE command_type_enum AS ENUM (
    'start_charge', 'stop_charge', 'start_discharge', 'stop_discharge',
    'emergency_shutdown', 'update_parameters', 'request_status'
);

-- Indexes for performance
CREATE INDEX idx_command_device ON command_history(device_id, issued_at);
CREATE INDEX idx_command_status ON command_history(success, completed_at);
```

## Redis Schema Patterns

### Real-time Device Status

```redis
# Pattern: device:status:{device_id}
# TTL: 300 seconds (5 minutes)
SET device:status:battery_pack_001 '{
  "online": true,
  "last_telemetry": "2024-01-15T14:30:15Z",
  "current_soc": 67.5,
  "current_power": -10126,
  "alarm_count": 1,
  "communication_quality": 0.98
}' EX 300
```

### Site Aggregations

```redis
# Pattern: site:metrics:{site_id}:{period}
# TTL: 3600 seconds (1 hour)
HSET site:metrics:site_001:current
  total_power -45680
  total_capacity 850.5
  average_soc 72.3
  device_count 12
  online_count 11
  alarm_count 3

EXPIRE site:metrics:site_001:current 3600
```

### Active Commands

```redis
# Pattern: commands:pending:{device_id}
# TTL: Based on command timeout
LPUSH commands:pending:battery_pack_001 '{
  "command_id": "cmd_12345",
  "type": "start_charge",
  "parameters": {"target_power": 5000},
  "issued_at": "2024-01-15T14:30:00Z",
  "timeout": 30
}'

EXPIRE commands:pending:battery_pack_001 30
```

### Session Management

```redis
# Pattern: session:{session_id}
# TTL: 3600 seconds (1 hour)
SET session:sess_abc123 '{
  "user_id": "user_001",
  "site_permissions": ["site_001", "site_002"],
  "role": "operator",
  "login_time": "2024-01-15T14:00:00Z",
  "last_activity": "2024-01-15T14:29:45Z"
}' EX 3600
```

## Time-Series Optimization

### MongoDB Time-Series Collections

```javascript
// Create time-series collection for telemetry
db.createCollection("telemetry_ts", {
   timeseries: {
      timeField: "timestamp",
      metaField: "device_meta",
      granularity: "seconds"
   }
});

// Document structure optimized for time-series
{
  "timestamp": ISODate("2024-01-15T14:30:00Z"),
  "device_meta": {
    "site_id": "site_001",
    "device_id": "battery_pack_001",
    "chemistry": "LFP"
  },
  "voltage": 400.5,
  "current": -25.3,
  "temperature": 23.7,
  "soc": 67.5,
  "soh": 87.2
}
```

### Partitioning Strategy

```sql
-- PostgreSQL partitioning for command history
CREATE TABLE command_history_y2024m01 PARTITION OF command_history
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE command_history_y2024m02 PARTITION OF command_history
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- Automatic partition management
CREATE OR REPLACE FUNCTION create_monthly_partition()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    table_name text;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE + interval '1 month');
    end_date := start_date + interval '1 month';
    table_name := 'command_history_y' || to_char(start_date, 'YYYY') || 'm' || to_char(start_date, 'MM');
    
    EXECUTE format('CREATE TABLE %I PARTITION OF command_history FOR VALUES FROM (%L) TO (%L)',
                   table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

## Data Retention Policies

### MongoDB TTL Indexes

```javascript
// Raw telemetry: 90 days
db.telemetry_data.createIndex(
  { "timestamp": 1 }, 
  { expireAfterSeconds: 7776000 }
);

// Hourly aggregates: 2 years
db.metrics_hourly.createIndex(
  { "timestamp": 1 }, 
  { expireAfterSeconds: 63072000 }
);

// Daily aggregates: 10 years
db.metrics_daily.createIndex(
  { "timestamp": 1 }, 
  { expireAfterSeconds: 315360000 }
);
```

### PostgreSQL Cleanup Jobs

```sql
-- Clean up old command history (keep 1 year)
DELETE FROM command_history 
WHERE issued_at < NOW() - INTERVAL '1 year';

-- Archive old device configurations
INSERT INTO device_configurations_archive 
SELECT * FROM device_configurations 
WHERE updated_at < NOW() - INTERVAL '5 years';
```
