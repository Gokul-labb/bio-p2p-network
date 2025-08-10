# Bio P2P Node CLI

A production-ready command-line interface and daemon for participating in the Bio P2P network. This tool provides headless P2P network participation with comprehensive monitoring, security, and biological behavior implementation.

## Features

### Core Functionality
- **Headless Daemon**: Background service for continuous network participation
- **Biological Behaviors**: Implementation of 80+ specialized biological node types
- **P2P Networking**: Full libp2p integration with biological routing protocols
- **Multi-layer Security**: 5-tier security framework with immune system principles
- **Resource Management**: Dynamic resource allocation with biological efficiency

### CLI Interface
- **Command-line Control**: Start, stop, status, and configuration management
- **Configuration Templates**: Minimal, production, and development templates
- **Environment Integration**: Support for environment variables and config files
- **Bootstrap Utilities**: Network peer discovery and connection testing

### Monitoring & Observability
- **Health Checks**: HTTP endpoints for readiness and liveness probes
- **Prometheus Metrics**: Comprehensive metrics for network, performance, and biological behaviors
- **Structured Logging**: Configurable logging with multiple output formats
- **Status Reporting**: Detailed daemon status and network information

### Production Features
- **Signal Handling**: Graceful shutdown with SIGTERM/SIGINT support
- **PID Management**: Process identification and duplicate detection
- **Service Integration**: systemd service files and Docker containerization
- **Configuration Validation**: Comprehensive config validation with helpful error messages

## Installation

### From Source

```bash
cd bio-p2p/bio-node-cli
cargo build --release
```

The binary will be available at `target/release/bio-node`.

### Using Cargo Install

```bash
cargo install --path bio-p2p/bio-node-cli
```

## Quick Start

### 1. Generate Configuration

```bash
# Generate minimal configuration
bio-node config --template minimal > node.toml

# Generate production configuration
bio-node config --template production > production.toml

# Generate development configuration  
bio-node config --template development > dev.toml
```

### 2. Start the Node

```bash
# Start with configuration file
bio-node start --config node.toml

# Start with environment variables
export BIO_BIND_ADDR=0.0.0.0
export BIO_BIND_PORT=8000
export BIO_MAX_CPU_CORES=8
bio-node start

# Start in daemon mode (Unix)
bio-node start --config node.toml --daemon
```

### 3. Check Status

```bash
# Basic status
bio-node status

# Detailed status with configuration file
bio-node status --config node.toml
```

### 4. Stop the Node

```bash
# Graceful shutdown
bio-node stop --config node.toml
```

## Configuration

### Configuration File Format

Bio P2P Node uses TOML configuration files:

```toml
[network]
# Network listening configuration
listen_addresses = ["/ip4/0.0.0.0/tcp/8000"]
bootstrap_peers = []
max_connections = 100
protocol_version = "bio-p2p/1.0.0"

[biological]
# Preferred biological roles
preferred_roles = ["CasteNode", "YoungNode", "HavocNode"]
learning_rate = 0.1
trust_building_rate = 0.05
enable_havoc = true

[resources]
# Resource allocation limits
max_cpu_cores = 4
max_memory_mb = 4096
max_disk_gb = 100
enable_compartmentalization = true

[security]
# Security configuration
enable_multi_layer = true
thermal_monitoring = true
behavior_analysis = true
reputation_threshold = 0.7

[monitoring]
# Health and metrics configuration
enable_health_check = true
health_port = 8080
enable_metrics = true
metrics_addr = "127.0.0.1"
metrics_port = 9090

[daemon]
# Daemon process configuration
pid_file = "/var/run/bio-node.pid"
working_directory = "/var/lib/bio-node"
```

### Environment Variables

All configuration options can be overridden with environment variables:

```bash
# Network configuration
export BIO_BIND_ADDR=192.168.1.100
export BIO_BIND_PORT=8000
export BIO_BOOTSTRAP_PEERS="/ip4/10.0.0.1/tcp/8000/p2p/12D3K..."
export BIO_NETWORK_KEY=/path/to/private/key.pem

# Resource limits
export BIO_MAX_CPU_CORES=16
export BIO_MAX_MEMORY_MB=16384
export BIO_MAX_DISK_GB=500

# Biological roles
export BIO_PREFERRED_ROLES="CasteNode,HavocNode,ThermalNode"

# Logging
export BIO_LOG_LEVEL=info
export BIO_LOG_FILE=/var/log/bio-node.log

# Daemon configuration
export BIO_DAEMON_MODE=true
export BIO_DATA_DIR=/opt/bio-node/data
export BIO_PID_FILE=/run/bio-node.pid
```

## Commands

### `start` - Start the daemon

```bash
bio-node start [OPTIONS]
```

**Options:**
- `--config, -c <FILE>`: Configuration file path
- `--daemon, -d`: Run in daemon mode (Unix only)
- `--log-level <LEVEL>`: Set log level (error, warn, info, debug, trace)
- `--log-file <FILE>`: Write logs to file
- `--quiet, -q`: Suppress console output

**Examples:**
```bash
# Start with config file
bio-node start --config /etc/bio-node/node.toml

# Start in daemon mode
bio-node start --daemon --config production.toml

# Start with debug logging
bio-node start --log-level debug --log-file debug.log
```

### `stop` - Stop running daemon

```bash
bio-node stop [OPTIONS]
```

**Options:**
- `--config, -c <FILE>`: Configuration file path (to locate PID file)

**Examples:**
```bash
# Stop daemon
bio-node stop --config node.toml

# Stop using default config locations
bio-node stop
```

### `status` - Check daemon status

```bash
bio-node status [OPTIONS]
```

**Options:**
- `--config, -c <FILE>`: Configuration file path

**Examples:**
```bash
# Basic status
bio-node status

# Detailed status
bio-node status --config production.toml
```

### `config` - Generate configuration templates

```bash
bio-node config [OPTIONS]
```

**Options:**
- `--template <TYPE>`: Template type (minimal, production, development)
- `--output, -o <FILE>`: Output file path

**Examples:**
```bash
# Generate minimal config to stdout
bio-node config --template minimal

# Generate production config to file
bio-node config --template production --output production.toml

# Generate development config
bio-node config --template development --output dev.toml
```

### `bootstrap` - Test peer connectivity

```bash
bio-node bootstrap [OPTIONS] <PEER_ADDR>
```

**Options:**
- `--config, -c <FILE>`: Configuration file path
- `--timeout <SECONDS>`: Connection timeout (default: 30)

**Examples:**
```bash
# Test connection to peer
bio-node bootstrap /ip4/10.0.0.1/tcp/8000/p2p/12D3KooW...

# Test with custom timeout
bio-node bootstrap --timeout 60 /ip4/example.com/tcp/8000/p2p/12D3KooW...

# Update config file with successful peer
bio-node bootstrap --config node.toml /ip4/10.0.0.1/tcp/8000/p2p/12D3KooW...
```

## Monitoring

### Health Checks

When enabled, the node provides HTTP health check endpoints:

```bash
# Readiness probe (Kubernetes-compatible)
curl http://localhost:8080/ready

# Liveness probe (Kubernetes-compatible)  
curl http://localhost:8080/live

# Detailed health information
curl http://localhost:8080/health
```

Health check response example:
```json
{
  "status": "healthy",
  "network": {
    "ready": true,
    "peer_id": "12D3KooWGjQvqoRwTwEhXGFbAHE7UqFJ2Cv1RRNx5XmvZJbB9vkZ",
    "connected_peers": 15,
    "listen_addresses": ["/ip4/192.168.1.100/tcp/8000"]
  },
  "resources": {
    "cpu_usage": 0.45,
    "memory_usage": 0.62,
    "disk_usage": 0.23
  },
  "biological": {
    "active_roles": ["CasteNode", "YoungNode", "HavocNode"],
    "role_count": 3
  },
  "timestamp": "2025-01-01T12:00:00Z"
}
```

### Prometheus Metrics

When enabled, comprehensive metrics are available:

```bash
# Get all metrics
curl http://localhost:9090/metrics
```

Key metric categories:
- **Network**: `bio_node_connected_peers`, `bio_node_messages_sent_total`, `bio_node_network_latency_seconds`
- **Resources**: `bio_node_cpu_usage_percentage`, `bio_node_memory_usage_bytes`, `bio_node_allocation_efficiency_ratio`
- **Biological**: `bio_node_active_roles_count`, `bio_node_cooperation_success_rate`, `bio_node_trust_relationships`
- **Performance**: `bio_node_tasks_completed_total`, `bio_node_task_duration_seconds`, `bio_node_error_rate`
- **Security**: `bio_node_security_events_total`, `bio_node_threats_detected_total`, `bio_node_quarantined_nodes`

### Structured Logging

Logs are available in multiple formats:

```bash
# JSON structured logs
bio-node start --log-level info 2>&1 | jq

# Traditional format
bio-node start --log-level info --log-file /var/log/bio-node.log
```

Log entry example:
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "level": "INFO",
  "module": "bio_node::daemon",
  "message": "Bio P2P Node daemon started successfully",
  "fields": {
    "peer_id": "12D3KooWGjQvqoRwTwEhXGFbAHE7UqFJ2Cv1RRNx5XmvZJbB9vkZ",
    "listen_addresses": ["/ip4/192.168.1.100/tcp/8000"],
    "active_roles": ["CasteNode", "YoungNode"]
  }
}
```

## Service Integration

### systemd Service

Create `/etc/systemd/system/bio-node.service`:

```ini
[Unit]
Description=Bio P2P Node Daemon
After=network-online.target
Wants=network-online.target

[Service]
Type=notify
User=bio-node
Group=bio-node
ExecStart=/usr/local/bin/bio-node start --config /etc/bio-node/production.toml
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=30
Restart=on-failure
RestartSec=10

# Security settings
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/bio-node /var/log/bio-node
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

Commands:
```bash
# Install and start service
sudo systemctl enable bio-node
sudo systemctl start bio-node

# Check service status
sudo systemctl status bio-node

# View service logs
sudo journalctl -u bio-node -f
```

### Docker Container

Create `Dockerfile`:

```dockerfile
FROM rust:1.75 AS builder

WORKDIR /app
COPY . .
RUN cargo build --release --bin bio-node

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 bio-node

COPY --from=builder /app/target/release/bio-node /usr/local/bin/

USER bio-node
WORKDIR /home/bio-node

EXPOSE 8000 8080 9090

ENTRYPOINT ["bio-node"]
CMD ["start", "--config", "/config/node.toml"]
```

Docker commands:
```bash
# Build image
docker build -t bio-node .

# Run container
docker run -d \
  --name bio-node \
  -p 8000:8000 \
  -p 8080:8080 \
  -p 9090:9090 \
  -v /host/config:/config:ro \
  -v /host/data:/data \
  bio-node

# Check container status
docker logs bio-node -f

# Stop container gracefully
docker stop bio-node
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  bio-node:
    build: .
    container_name: bio-node
    restart: unless-stopped
    ports:
      - "8000:8000"   # P2P networking
      - "8080:8080"   # Health checks
      - "9090:9090"   # Metrics
    volumes:
      - ./config:/config:ro
      - bio_data:/data
      - bio_logs:/logs
    environment:
      - BIO_LOG_LEVEL=info
      - BIO_DATA_DIR=/data
      - RUST_LOG=bio_node=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Optional: Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  # Optional: Grafana dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=bio-p2p-admin

volumes:
  bio_data:
  bio_logs:
  prometheus_data:
  grafana_data:
```

## Biological Roles

The Bio P2P Node can automatically adopt different biological roles based on network conditions and configuration:

### Learning & Adaptation Nodes
- **YoungNode**: Learn optimal paths from experienced neighbors
- **CasteNode**: Compartmentalize resources for specialized functions
- **ImitateNode**: Copy successful patterns from high-performing peers

### Coordination & Synchronization Nodes  
- **HatchNode**: Manage super-node lifecycle synchronization
- **SyncPhaseNode**: Handle node lifecycle phase management
- **HuddleNode**: Dynamic position rotation for load distribution

### Communication & Routing Nodes
- **MigrationNode**: Maintain generational memory of optimal routes
- **AddressNode**: Hierarchical addressing for scalable management
- **TunnelNode**: Secure encrypted tunnels between nodes
- **SignNode**: Routing waypoints and path optimization
- **ThermalNode**: Monitor network congestion and thermal signatures

### Security & Defense Nodes
- **DosNode**: Denial of service detection and prevention
- **InvestigationNode**: Network anomaly analysis and forensics
- **CasualtyNode**: Post-incident analysis and learning

### Resource Management Nodes
- **HavocNode**: Emergency resource reallocation during crises
- **StepUpNode/StepDownNode**: Dynamic capacity scaling
- **FriendshipNode**: Cooperative resource sharing
- **BuddyNode**: Fault-tolerant paired backup systems
- **TrustNode**: Social bonding and trust relationship management

### Support & Maintenance Nodes
- **MemoryNode**: Process state management and recovery
- **TelescopeNode**: Predictive behavior analysis
- **HealingNode**: Proactive network optimization

Role assignment is dynamic and based on:
- Network conditions and resource availability
- Node performance and reputation
- Geographic location and network topology
- Crisis situations and emergency response needs

## Troubleshooting

### Common Issues

**Node won't start:**
```bash
# Check configuration
bio-node config --template minimal > test.toml
bio-node start --config test.toml --log-level debug

# Verify network connectivity
bio-node bootstrap /ip4/127.0.0.1/tcp/8000/p2p/...
```

**High resource usage:**
```bash
# Check resource allocation
curl http://localhost:8080/health | jq '.resources'

# Adjust resource limits in config
[resources]
max_cpu_cores = 2
max_memory_mb = 2048
```

**Network connectivity issues:**
```bash
# Check firewall settings
sudo ufw allow 8000/tcp

# Test bootstrap peers
bio-node bootstrap --timeout 10 <peer_address>

# Check listening addresses
netstat -tlnp | grep 8000
```

**Performance problems:**
```bash
# Monitor metrics
curl http://localhost:9090/metrics | grep bio_node_task

# Check biological role performance
curl http://localhost:8080/health | jq '.biological'
```

### Debug Logging

Enable detailed debug logging:

```bash
# Start with debug logs
bio-node start --log-level debug --log-file debug.log

# Enable Rust debug logs
RUST_LOG=bio_node=debug bio-node start

# Enable libp2p debug logs
RUST_LOG=libp2p=debug,bio_node=info bio-node start
```

### Health Check Debugging

```bash
# Test health endpoints
curl -v http://localhost:8080/health
curl -v http://localhost:8080/ready  
curl -v http://localhost:8080/live

# Check metrics availability
curl -v http://localhost:9090/metrics
```

### Configuration Validation

```bash
# Validate config file
bio-node start --config node.toml --log-level debug --dry-run

# Generate and compare configs
bio-node config --template production > expected.toml
diff node.toml expected.toml
```

## Development

### Building from Source

```bash
# Clone repository
git clone https://github.com/bio-p2p/bio-p2p.git
cd bio-p2p/bio-node-cli

# Build development version
cargo build

# Build release version
cargo build --release

# Run tests
cargo test

# Run with development config
cargo run -- start --config examples/development.toml
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-biological-role`
3. Make changes and add tests
4. Run tests: `cargo test`
5. Run clippy: `cargo clippy`
6. Format code: `cargo fmt`
7. Submit pull request

### Performance Benchmarking

```bash
# Run benchmarks
cargo bench

# Profile performance
cargo build --release
perf record target/release/bio-node start --config bench.toml
perf report
```

## Security

### Network Security

The Bio P2P Node implements a 5-layer security framework:

1. **Multi-Layer Execution**: Randomized execution environments
2. **CBADU**: Clean Before and After Usage sanitization
3. **Illusion Layer**: Active deception against attackers
4. **Behavior Monitoring**: Continuous anomaly detection
5. **Thermal Detection**: Resource usage pattern analysis

### Process Security

- Process isolation and sandboxing
- Minimal privilege execution
- Secure key management
- Encrypted inter-node communication
- Trust-based reputation system

### Configuration Security

```bash
# Secure file permissions
chmod 600 /etc/bio-node/node.toml
chown bio-node:bio-node /etc/bio-node/node.toml

# Use environment variables for secrets
export BIO_NETWORK_KEY_PASSPHRASE=secure_passphrase

# Validate configuration
bio-node start --dry-run --config production.toml
```

## License

This project is licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Support

- **Documentation**: [https://docs.bio-p2p.org](https://docs.bio-p2p.org)
- **Repository**: [https://github.com/bio-p2p/bio-p2p](https://github.com/bio-p2p/bio-p2p)
- **Issues**: [https://github.com/bio-p2p/bio-p2p/issues](https://github.com/bio-p2p/bio-p2p/issues)
- **Discussions**: [https://github.com/bio-p2p/bio-p2p/discussions](https://github.com/bio-p2p/bio-p2p/discussions)

## Changelog

### v0.1.0 (Initial Release)

- Complete CLI interface with start/stop/status/config commands
- Production-ready daemon with signal handling and PID management
- Comprehensive monitoring with health checks and Prometheus metrics
- Full biological node taxonomy implementation
- Multi-layer security framework
- Docker and systemd integration
- Bootstrap peer discovery and testing
- Configuration templates and environment variable support