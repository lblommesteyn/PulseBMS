# PulseBMS Enhanced - System Diagrams

## System Architecture Overview

```mermaid
graph TB
    subgraph "Physical Layer"
        BP1[Battery Pack 1<br/>LFP 100kWh]
        BP2[Battery Pack 2<br/>NMC 80kWh]
        BP3[Battery Pack 3<br/>LCO 60kWh]
        BP4[Battery Pack N<br/>Mixed Chemistry]
    end
    
    subgraph "Edge Layer"
        ED1[Edge Device 1<br/>QNX RTOS]
        ED2[Edge Device 2<br/>FreeRTOS]
        ED3[Edge Device 3<br/>QNX RTOS]
        ED4[Edge Device N<br/>RTOS]
    end
    
    subgraph "Communication Layer"
        MQTT[MQTT Broker<br/>Eclipse Mosquitto]
        WS[WebSocket Gateway]
    end
    
    subgraph "Cloud Services"
        API[FastAPI Backend<br/>REST + WebSocket]
        DT[Digital Twin Service<br/>PyBaMM Models]
        COORD[Coordinator Service<br/>MPC + RL Engine]
        DB1[(MongoDB<br/>Telemetry)]
        DB2[(PostgreSQL<br/>Configuration)]
    end
    
    subgraph "User Interface"
        DASH[React Dashboard<br/>Real-time UI]
        MOB[Mobile App<br/>Field Operations]
    end
    
    BP1 --> ED1
    BP2 --> ED2
    BP3 --> ED3
    BP4 --> ED4
    
    ED1 --> MQTT
    ED2 --> MQTT
    ED3 --> MQTT
    ED4 --> MQTT
    
    MQTT --> API
    MQTT --> DT
    MQTT --> COORD
    
    API --> DB1
    API --> DB2
    API --> WS
    
    DT --> COORD
    COORD --> MQTT
    
    WS --> DASH
    WS --> MOB
    
    style BP1 fill:#e1f5fe
    style BP2 fill:#e8f5e8
    style BP3 fill:#fff3e0
    style BP4 fill:#fce4ec
    style MQTT fill:#f3e5f5
    style API fill:#e8eaf6
    style DT fill:#e0f2f1
    style COORD fill:#fff8e1
```

## Data Flow Architecture

```mermaid
sequenceDiagram
    participant BP as Battery Pack
    participant ED as Edge Device
    participant MQTT as MQTT Broker
    participant API as Backend API
    participant DT as Digital Twin
    participant COORD as Coordinator
    participant DASH as Dashboard
    
    loop Every 1 Second
        BP->>ED: CAN Bus Telemetry
        ED->>ED: SoC/SoH Estimation
        ED->>ED: Safety Monitoring
        ED->>MQTT: Publish Telemetry
    end
    
    MQTT->>API: Telemetry Stream
    API->>DASH: WebSocket Update
    
    loop Every 5 Minutes
        API->>DT: Request Forecast
        DT->>DT: PyBaMM Simulation
        DT->>API: Degradation Forecast
        API->>COORD: Updated Forecasts
    end
    
    loop Every 10 Minutes
        COORD->>COORD: MPC Optimization
        COORD->>COORD: RL Policy Update
        COORD->>MQTT: Power Allocation
        MQTT->>ED: Allocation Commands
        ED->>BP: Control Signals
    end
    
    Note over ED: <1ms Safety Response
    Note over COORD: Multi-objective Optimization
    Note over DT: Physics-based Modeling
```

## Safety Architecture Diagram

```mermaid
graph TD
    subgraph "Safety Layers"
        L1[Hardware Layer<br/>Fuses, Relays, Contactors]
        L2[Edge Layer<br/>Real-time Monitoring<br/>&lt;1ms Response]
        L3[Cloud Layer<br/>Fleet Analytics<br/>Predictive Safety]
        L4[Human Layer<br/>Dashboard Alerts<br/>Manual Override]
    end
    
    subgraph "Safety Constraints"
        SC1[Voltage Limits<br/>2.5V - 4.2V per cell]
        SC2[Current Limits<br/>±100A max]
        SC3[Temperature Limits<br/>-20°C to 60°C]
        SC4[SoC Limits<br/>5% - 95%]
        SC5[Degradation Limits<br/>Max C-rate: 1.0]
    end
    
    subgraph "Emergency Response"
        ER1[Immediate Shutdown<br/>Hardware Level]
        ER2[Controlled Shutdown<br/>Edge Level]
        ER3[Load Rebalancing<br/>Cloud Level]
        ER4[Operator Alert<br/>Human Level]
    end
    
    L1 --> ER1
    L2 --> ER2
    L3 --> ER3
    L4 --> ER4
    
    SC1 --> L1
    SC1 --> L2
    SC2 --> L1
    SC2 --> L2
    SC3 --> L2
    SC3 --> L3
    SC4 --> L2
    SC4 --> L3
    SC5 --> L3
    
    style L1 fill:#ffebee
    style L2 fill:#fff3e0
    style L3 fill:#e8f5e8
    style L4 fill:#e3f2fd
    style ER1 fill:#f44336,color:#fff
    style ER2 fill:#ff9800,color:#fff
    style ER3 fill:#4caf50,color:#fff
    style ER4 fill:#2196f3,color:#fff
```

## MQTT Communication Schema

```mermaid
graph LR
    subgraph "Topic Hierarchy"
        ROOT[pulsebms/]
        SITE[{site_id}/]
        DEV[{device_id}/]
        COORD_PATH[coordinator/]
        
        ROOT --> SITE
        SITE --> DEV
        SITE --> COORD_PATH
    end
    
    subgraph "Message Types"
        TEL[telemetry]
        CMD[commands]
        ALLOC[allocation]
        STATUS[status]
        
        DEV --> TEL
        DEV --> CMD
        DEV --> STATUS
        COORD_PATH --> ALLOC
    end
    
    subgraph "QoS Levels"
        QOS0[QoS 0: Fire & Forget<br/>Status Updates]
        QOS1[QoS 1: At Least Once<br/>Telemetry Data]
        QOS2[QoS 2: Exactly Once<br/>Safety Commands]
    end
    
    TEL --> QOS1
    CMD --> QOS2
    ALLOC --> QOS1
    STATUS --> QOS0
    
    style ROOT fill:#e1f5fe
    style TEL fill:#e8f5e8
    style CMD fill:#fff3e0
    style ALLOC fill:#f3e5f5
    style QOS2 fill:#ffebee
```

## Reinforcement Learning Architecture

```mermaid
graph TB
    subgraph "RL Environment"
        ENV[Battery Fleet Simulator<br/>PyBaMM Integration]
        STATE[State Space<br/>SoC, SoH, Temp, Load]
        ACTION[Action Space<br/>Power Allocation]
        REWARD[Reward Function<br/>-Cost + Safety]
    end
    
    subgraph "Safe RL Agent"
        POLICY[Policy Network<br/>Actor-Critic]
        SAFETY[Safety Layer<br/>Constraint Validation]
        EXPERIENCE[Experience Buffer<br/>Prioritized Replay]
    end
    
    subgraph "Training Pipeline"
        SIM[Distributed Simulation]
        TRAIN[Gradient Updates]
        EVAL[Policy Evaluation]
        DEPLOY[Edge Deployment]
    end
    
    ENV --> STATE
    STATE --> POLICY
    POLICY --> SAFETY
    SAFETY --> ACTION
    ACTION --> ENV
    ENV --> REWARD
    REWARD --> EXPERIENCE
    
    EXPERIENCE --> TRAIN
    TRAIN --> POLICY
    POLICY --> EVAL
    EVAL --> DEPLOY
    
    SIM --> ENV
    
    style POLICY fill:#e8eaf6
    style SAFETY fill:#ffebee
    style REWARD fill:#e8f5e8
    style DEPLOY fill:#f3e5f5
```

## Digital Twin Integration Flow

```mermaid
graph TB
    subgraph "Real Battery"
        RB[Physical Battery Pack]
        SENSORS[Voltage, Current, Temp Sensors]
        CAN[CAN Bus Interface]
    end
    
    subgraph "Digital Twin"
        PYBAMM[PyBaMM Physics Model]
        PARAMS[Parameter Estimation]
        FORECAST[Degradation Forecast]
        CALIB[Model Calibration]
    end
    
    subgraph "Data Flow"
        TELEMETRY[Real-time Telemetry]
        SYNC[State Synchronization]
        PREDICT[Future State Prediction]
    end
    
    RB --> SENSORS
    SENSORS --> CAN
    CAN --> TELEMETRY
    
    TELEMETRY --> SYNC
    SYNC --> PYBAMM
    PYBAMM --> PARAMS
    PARAMS --> CALIB
    CALIB --> PYBAMM
    
    PYBAMM --> FORECAST
    FORECAST --> PREDICT
    PREDICT --> TELEMETRY
    
    style RB fill:#e1f5fe
    style PYBAMM fill:#e8f5e8
    style SYNC fill:#fff3e0
    style FORECAST fill:#f3e5f5
```

## Network Architecture

```mermaid
graph TB
    subgraph "Edge Network"
        EDGE1[Edge Device 1<br/>192.168.1.10]
        EDGE2[Edge Device 2<br/>192.168.1.11]
        EDGE3[Edge Device N<br/>192.168.1.1x]
        SWITCH[Industrial Switch<br/>Managed PoE]
    end
    
    subgraph "Gateway Layer"
        GATEWAY[Edge Gateway<br/>VPN + Firewall]
        ROUTER[Industrial Router<br/>4G/5G/Ethernet]
    end
    
    subgraph "Cloud Infrastructure"
        LB[Load Balancer<br/>NGINX/HAProxy]
        API1[API Server 1]
        API2[API Server 2]
        MQTTC[MQTT Cluster<br/>HA Configuration]
        DB[Database Cluster<br/>MongoDB + PostgreSQL]
    end
    
    EDGE1 --> SWITCH
    EDGE2 --> SWITCH
    EDGE3 --> SWITCH
    SWITCH --> GATEWAY
    GATEWAY --> ROUTER
    ROUTER --> LB
    
    LB --> API1
    LB --> API2
    LB --> MQTTC
    
    API1 --> DB
    API2 --> DB
    
    style SWITCH fill:#e1f5fe
    style GATEWAY fill:#fff3e0
    style LB fill:#e8f5e8
    style MQTTC fill:#f3e5f5
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV[Developer Laptop]
        GIT[Git Repository]
        CI[GitHub Actions]
    end
    
    subgraph "Staging"
        STAGE_K8S[Kubernetes Staging]
        STAGE_DB[Staging Database]
        STAGE_MQTT[Staging MQTT]
    end
    
    subgraph "Production"
        PROD_K8S[Kubernetes Production<br/>Multi-zone HA]
        PROD_DB[Production Database<br/>Replicated]
        PROD_MQTT[Production MQTT<br/>Clustered]
        MONITOR[Monitoring<br/>Prometheus + Grafana]
    end
    
    DEV --> GIT
    GIT --> CI
    CI --> STAGE_K8S
    STAGE_K8S --> STAGE_DB
    STAGE_K8S --> STAGE_MQTT
    
    CI --> PROD_K8S
    PROD_K8S --> PROD_DB
    PROD_K8S --> PROD_MQTT
    PROD_K8S --> MONITOR
    
    style CI fill:#e8f5e8
    style STAGE_K8S fill:#fff3e0
    style PROD_K8S fill:#e3f2fd
    style MONITOR fill:#f3e5f5
```
