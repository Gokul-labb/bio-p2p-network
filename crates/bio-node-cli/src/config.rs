//! Configuration management for Bio P2P Node
//!
//! This module provides comprehensive configuration management including file parsing,
//! environment variable integration, validation, and template generation.

use anyhow::{anyhow, Context, Result};
use libp2p::Multiaddr;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Complete node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Biological behavior configuration  
    pub biological: BiologicalConfig,
    
    /// Resource management configuration
    pub resources: ResourceConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Storage configuration
    pub storage: StorageConfig,
    
    /// Monitoring and observability configuration
    pub monitoring: MonitoringConfig,
    
    /// Daemon configuration
    pub daemon: DaemonConfig,
    
    /// Economic configuration
    pub economics: EconomicsConfig,
}

/// Network-related configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Addresses to listen on for incoming connections
    pub listen_addresses: Vec<Multiaddr>,
    
    /// Bootstrap peer addresses for initial network discovery
    pub bootstrap_peers: Vec<Multiaddr>,
    
    /// Path to node private key file
    pub node_key_path: PathBuf,
    
    /// Maximum number of concurrent connections
    pub max_connections: usize,
    
    /// Enable mDNS peer discovery
    pub enable_mdns: bool,
    
    /// Connection timeout duration
    pub connection_timeout: Duration,
    
    /// Keep-alive interval for connections
    pub keep_alive_interval: Duration,
    
    /// Maximum packet size for network communication
    pub max_packet_size: usize,
    
    /// Network protocol version
    pub protocol_version: String,
}

/// Biological behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    /// Preferred biological roles for this node
    pub preferred_roles: Vec<BiologicalRole>,
    
    /// Enable social behavior patterns
    pub enable_social_behavior: bool,
    
    /// Trust building rate (0.0 to 1.0)
    pub trust_building_rate: f64,
    
    /// Cooperation threshold for partner selection
    pub cooperation_threshold: f64,
    
    /// Enable HAVOC crisis management behavior
    pub enable_havoc: bool,
    
    /// Crisis sensitivity threshold (0.0 to 1.0)
    pub crisis_sensitivity: f64,
    
    /// Learning rate for adaptive behaviors
    pub learning_rate: f64,
    
    /// Maximum number of peers to learn from (Young Node behavior)
    pub max_learning_peers: usize,
    
    /// Trust decay rate for inactive relationships
    pub trust_decay_rate: f64,
    
    /// Enable thermal monitoring and pheromone trails
    pub enable_thermal_monitoring: bool,
    
    /// Young Node convergence timeout
    pub young_node_convergence_timeout: Duration,
    
    /// Caste Node compartment rebalancing interval
    pub caste_rebalancing_interval: Duration,
}

/// Available biological roles
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BiologicalRole {
    // Learning and Adaptation
    YoungNode,
    CasteNode, 
    ImitateNode,
    
    // Coordination and Synchronization
    HatchNode,
    SyncPhaseNode,
    HuddleNode,
    DynamicKeyNode,
    KeyNode,
    
    // Communication and Routing
    MigrationNode,
    AddressNode,
    TunnelNode,
    SignNode,
    ThermalNode,
    
    // Security and Defense
    DosNode,
    InvestigationNode,
    CasualtyNode,
    ConfusionNode,
    WatchdogNode,
    
    // Resource Management
    HavocNode,
    StepUpNode,
    StepDownNode,
    
    // Social and Collaborative
    FriendshipNode,
    BuddyNode,
    TrustNode,
    
    // Coordination and Orchestration
    SyncNode,
    PacketNode,
    CroakNode,
    
    // Specialized Functions
    WebNode,
    HierarchyNode,
    AlphaNode,
    BravoNode,
    SuperNode,
    
    // Support and Maintenance
    MemoryNode,
    TelescopeNode,
    HealingNode,
    
    // Additional specialized roles
    LocalNode,
    CommandNode,
    QueueNode,
    RegionalBaseNode,
    PlanNode,
    DistributorNode,
    PackageNode,
    ClientNode,
    RandomNode,
    KnownNode,
    ExpertNode,
    ClusterNode,
    SupportNode,
    PropagandaNode,
    CultureNode,
    InfoNode,
    FollowUpNode,
    SurvivalNode,
    MixUpNode,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Maximum CPU cores to allocate
    pub max_cpu_cores: usize,
    
    /// Maximum memory in MB
    pub max_memory_mb: usize,
    
    /// Maximum disk space in GB
    pub max_disk_gb: usize,
    
    /// Resource allocation strategy
    pub allocation_strategy: AllocationStrategy,
    
    /// Enable dynamic resource scaling
    pub enable_dynamic_scaling: bool,
    
    /// Scaling sensitivity (0.0 to 1.0)
    pub scaling_sensitivity: f64,
    
    /// Minimum resource reserve percentage
    pub min_reserve_percentage: f64,
    
    /// CPU affinity settings
    pub cpu_affinity: Vec<usize>,
    
    /// Memory allocation limits per compartment
    pub compartment_memory_limits: CompartmentLimits,
    
    /// Resource monitoring interval
    pub monitoring_interval: Duration,
}

/// Resource allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Conservative resource allocation
    Conservative,
    /// Balanced resource allocation 
    Balanced,
    /// Aggressive resource allocation
    Aggressive,
    /// Custom allocation with specific parameters
    Custom {
        cpu_percentage: f64,
        memory_percentage: f64,
        disk_percentage: f64,
    },
}

/// Per-compartment resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentLimits {
    /// Training compartment memory limit (MB)
    pub training_mb: usize,
    /// Inference compartment memory limit (MB)
    pub inference_mb: usize,
    /// Storage compartment memory limit (MB)
    pub storage_mb: usize,
    /// Communication compartment memory limit (MB)
    pub communication_mb: usize,
    /// Security compartment memory limit (MB)
    pub security_mb: usize,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable behavior monitoring (Layer 4)
    pub enable_behavior_monitoring: bool,
    
    /// Enable thermal detection (Layer 5)
    pub enable_thermal_detection: bool,
    
    /// Enable illusion layer deception (Layer 3)
    pub enable_illusion_layer: bool,
    
    /// Enable CBADU sanitization (Layer 2)
    pub enable_cbadu: bool,
    
    /// Trusted node threshold (0.0 to 1.0)
    pub trusted_node_threshold: f64,
    
    /// Maximum ratio of malicious nodes to tolerate
    pub max_malicious_ratio: f64,
    
    /// Security monitoring interval
    pub monitoring_interval: Duration,
    
    /// Threat detection sensitivity
    pub threat_detection_sensitivity: f64,
    
    /// Enable forensic logging
    pub enable_forensic_logging: bool,
    
    /// Quarantine duration for suspicious nodes
    pub quarantine_duration: Duration,
    
    /// Encryption settings
    pub encryption: EncryptionConfig,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Symmetric encryption algorithm
    pub symmetric_algorithm: String,
    
    /// Asymmetric encryption algorithm
    pub asymmetric_algorithm: String,
    
    /// Hash algorithm
    pub hash_algorithm: String,
    
    /// Key derivation function
    pub key_derivation: String,
    
    /// Key rotation interval
    pub key_rotation_interval: Duration,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Data directory path
    pub data_dir: PathBuf,
    
    /// Cache directory path
    pub cache_dir: PathBuf,
    
    /// Log directory path
    pub log_dir: PathBuf,
    
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
    
    /// Cache eviction policy
    pub cache_eviction_policy: CacheEvictionPolicy,
    
    /// Data compression settings
    pub compression: CompressionConfig,
    
    /// Backup configuration
    pub backup: BackupConfig,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheEvictionPolicy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In, First Out
    FIFO,
    /// Random eviction
    Random,
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable data compression
    pub enabled: bool,
    
    /// Compression algorithm
    pub algorithm: String,
    
    /// Compression level (1-9)
    pub level: u8,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    /// Enable automatic backups
    pub enabled: bool,
    
    /// Backup interval
    pub interval: Duration,
    
    /// Number of backup copies to retain
    pub retention_count: usize,
    
    /// Backup directory path
    pub backup_dir: PathBuf,
}

/// Monitoring and observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable Prometheus metrics collection
    pub enable_metrics: bool,
    
    /// Metrics server bind address
    pub metrics_addr: String,
    
    /// Metrics server port
    pub metrics_port: u16,
    
    /// Enable health check endpoints
    pub enable_health_check: bool,
    
    /// Health check server port
    pub health_port: u16,
    
    /// Enable distributed tracing
    pub enable_tracing: bool,
    
    /// Tracing endpoint URL
    pub tracing_endpoint: Option<String>,
    
    /// Tracing service name
    pub tracing_service_name: String,
    
    /// Log configuration
    pub logging: LoggingConfig,
    
    /// Performance monitoring configuration
    pub performance: PerformanceConfig,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: String,
    
    /// Log format (json, pretty, compact)
    pub format: String,
    
    /// Log file path (optional)
    pub file_path: Option<PathBuf>,
    
    /// Maximum log file size in MB
    pub max_file_size_mb: usize,
    
    /// Number of log files to retain
    pub max_files: usize,
    
    /// Enable console output
    pub enable_console: bool,
    
    /// Enable structured JSON logging
    pub enable_json: bool,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable CPU monitoring
    pub enable_cpu_monitoring: bool,
    
    /// Enable memory monitoring
    pub enable_memory_monitoring: bool,
    
    /// Enable network monitoring
    pub enable_network_monitoring: bool,
    
    /// Enable disk I/O monitoring
    pub enable_disk_monitoring: bool,
    
    /// Monitoring sample interval
    pub sample_interval: Duration,
    
    /// Performance history retention period
    pub history_retention: Duration,
}

/// Daemon configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// PID file path
    pub pid_file: PathBuf,
    
    /// Enable graceful shutdown
    pub enable_graceful_shutdown: bool,
    
    /// Shutdown timeout
    pub shutdown_timeout: Duration,
    
    /// Startup timeout
    pub startup_timeout: Duration,
    
    /// User to run daemon as
    pub user: Option<String>,
    
    /// Group to run daemon as  
    pub group: Option<String>,
    
    /// Working directory for daemon
    pub working_directory: PathBuf,
    
    /// Environment variables for daemon
    pub environment: std::collections::HashMap<String, String>,
}

/// Economic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicsConfig {
    /// Enable token-based economics
    pub enable_token_economics: bool,
    
    /// Initial token balance
    pub initial_token_balance: u64,
    
    /// Token earning rates by activity type
    pub earning_rates: EarningRates,
    
    /// Token spending rates by service type
    pub spending_rates: SpendingRates,
    
    /// Staking configuration
    pub staking: StakingConfig,
    
    /// Reputation system configuration
    pub reputation: ReputationConfig,
}

/// Token earning rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarningRates {
    /// Tokens per compute cycle
    pub compute_rate: f64,
    
    /// Tokens per MB of bandwidth provided
    pub bandwidth_rate: f64,
    
    /// Tokens per GB of storage provided
    pub storage_rate: f64,
    
    /// Tokens per security event detected
    pub security_rate: f64,
    
    /// Bonus multiplier for reliability
    pub reliability_bonus: f64,
}

/// Token spending rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingRates {
    /// Tokens per compute cycle consumed
    pub compute_cost: f64,
    
    /// Tokens per MB of bandwidth consumed
    pub bandwidth_cost: f64,
    
    /// Tokens per GB of storage consumed
    pub storage_cost: f64,
    
    /// Premium multiplier for priority processing
    pub priority_multiplier: f64,
}

/// Staking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakingConfig {
    /// Minimum stake required for participation
    pub minimum_stake: u64,
    
    /// Staking reward rate
    pub reward_rate: f64,
    
    /// Slashing penalty rate for bad behavior
    pub slashing_rate: f64,
    
    /// Stake lock duration
    pub lock_duration: Duration,
}

/// Reputation system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Initial reputation score
    pub initial_score: f64,
    
    /// Reputation decay rate
    pub decay_rate: f64,
    
    /// Minimum reputation for trusted status
    pub trusted_threshold: f64,
    
    /// Reputation calculation weights
    pub weights: ReputationWeights,
}

/// Reputation calculation weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationWeights {
    /// Weight for task completion score
    pub task_completion: f64,
    
    /// Weight for uptime/lifetime score
    pub uptime: f64,
    
    /// Weight for behavioral consistency
    pub behavior_consistency: f64,
    
    /// Weight for security contributions
    pub security_contributions: f64,
    
    /// Penalty weight for failures
    pub failure_penalty: f64,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            network: NetworkConfig::default(),
            biological: BiologicalConfig::default(),
            resources: ResourceConfig::default(),
            security: SecurityConfig::default(),
            storage: StorageConfig::default(),
            monitoring: MonitoringConfig::default(),
            daemon: DaemonConfig::default(),
            economics: EconomicsConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec!["/ip4/0.0.0.0/tcp/7000".parse().unwrap()],
            bootstrap_peers: Vec::new(),
            node_key_path: PathBuf::from("node_key.pem"),
            max_connections: 100,
            enable_mdns: true,
            connection_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(30),
            max_packet_size: 1024 * 1024, // 1MB
            protocol_version: "/bio-p2p/1.0.0".to_string(),
        }
    }
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            preferred_roles: vec![
                BiologicalRole::YoungNode,
                BiologicalRole::CasteNode,
                BiologicalRole::ThermalNode,
            ],
            enable_social_behavior: true,
            trust_building_rate: 0.1,
            cooperation_threshold: 0.6,
            enable_havoc: true,
            crisis_sensitivity: 0.8,
            learning_rate: 0.05,
            max_learning_peers: 100,
            trust_decay_rate: 0.01,
            enable_thermal_monitoring: true,
            young_node_convergence_timeout: Duration::from_secs(300),
            caste_rebalancing_interval: Duration::from_secs(1),
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        let cpu_cores = num_cpus::get();
        let memory_mb = 4096; // Default 4GB
        
        Self {
            max_cpu_cores: cpu_cores,
            max_memory_mb: memory_mb,
            max_disk_gb: 100,
            allocation_strategy: AllocationStrategy::Balanced,
            enable_dynamic_scaling: true,
            scaling_sensitivity: 0.7,
            min_reserve_percentage: 0.1,
            cpu_affinity: Vec::new(),
            compartment_memory_limits: CompartmentLimits::default(),
            monitoring_interval: Duration::from_secs(5),
        }
    }
}

impl Default for CompartmentLimits {
    fn default() -> Self {
        Self {
            training_mb: 1024,
            inference_mb: 512,
            storage_mb: 1024,
            communication_mb: 256,
            security_mb: 256,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_behavior_monitoring: true,
            enable_thermal_detection: true,
            enable_illusion_layer: true,
            enable_cbadu: true,
            trusted_node_threshold: 0.8,
            max_malicious_ratio: 0.25,
            monitoring_interval: Duration::from_secs(1),
            threat_detection_sensitivity: 0.7,
            enable_forensic_logging: true,
            quarantine_duration: Duration::from_secs(300),
            encryption: EncryptionConfig::default(),
        }
    }
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            symmetric_algorithm: "AES-256-GCM".to_string(),
            asymmetric_algorithm: "Ed25519".to_string(),
            hash_algorithm: "SHA3-256".to_string(),
            key_derivation: "PBKDF2".to_string(),
            key_rotation_interval: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./data"),
            cache_dir: PathBuf::from("./cache"),
            log_dir: PathBuf::from("./logs"),
            max_cache_size_mb: 1024,
            cache_eviction_policy: CacheEvictionPolicy::LRU,
            compression: CompressionConfig::default(),
            backup: BackupConfig::default(),
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: "zstd".to_string(),
            level: 3,
        }
    }
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(3600), // 1 hour
            retention_count: 24,
            backup_dir: PathBuf::from("./backups"),
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_addr: "127.0.0.1".to_string(),
            metrics_port: 9090,
            enable_health_check: true,
            health_port: 8080,
            enable_tracing: false,
            tracing_endpoint: None,
            tracing_service_name: "bio-node".to_string(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: "pretty".to_string(),
            file_path: None,
            max_file_size_mb: 100,
            max_files: 10,
            enable_console: true,
            enable_json: false,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_cpu_monitoring: true,
            enable_memory_monitoring: true,
            enable_network_monitoring: true,
            enable_disk_monitoring: true,
            sample_interval: Duration::from_secs(5),
            history_retention: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            pid_file: PathBuf::from("/var/run/bio-node.pid"),
            enable_graceful_shutdown: true,
            shutdown_timeout: Duration::from_secs(30),
            startup_timeout: Duration::from_secs(60),
            user: None,
            group: None,
            working_directory: PathBuf::from("/var/lib/bio-node"),
            environment: std::collections::HashMap::new(),
        }
    }
}

impl Default for EconomicsConfig {
    fn default() -> Self {
        Self {
            enable_token_economics: true,
            initial_token_balance: 1000,
            earning_rates: EarningRates::default(),
            spending_rates: SpendingRates::default(),
            staking: StakingConfig::default(),
            reputation: ReputationConfig::default(),
        }
    }
}

impl Default for EarningRates {
    fn default() -> Self {
        Self {
            compute_rate: 10.0,
            bandwidth_rate: 0.1,
            storage_rate: 0.01,
            security_rate: 50.0,
            reliability_bonus: 1.5,
        }
    }
}

impl Default for SpendingRates {
    fn default() -> Self {
        Self {
            compute_cost: 12.0,
            bandwidth_cost: 0.12,
            storage_cost: 0.012,
            priority_multiplier: 2.0,
        }
    }
}

impl Default for StakingConfig {
    fn default() -> Self {
        Self {
            minimum_stake: 100,
            reward_rate: 0.05,
            slashing_rate: 0.1,
            lock_duration: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            initial_score: 500.0,
            decay_rate: 0.001,
            trusted_threshold: 700.0,
            weights: ReputationWeights::default(),
        }
    }
}

impl Default for ReputationWeights {
    fn default() -> Self {
        Self {
            task_completion: 1.0,
            uptime: 1.0,
            behavior_consistency: 0.3,
            security_contributions: 0.3,
            failure_penalty: 5.0,
        }
    }
}

impl NodeConfig {
    /// Load configuration from file
    pub async fn load_from_file(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let content = tokio::fs::read_to_string(&path).await
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        
        // Support both TOML and YAML
        let config = if path.extension().and_then(|s| s.to_str()) == Some("yaml") || 
                       path.extension().and_then(|s| s.to_str()) == Some("yml") {
            serde_yaml::from_str(&content)
                .with_context(|| format!("Failed to parse YAML config: {}", path.display()))?
        } else {
            toml::from_str(&content)
                .with_context(|| format!("Failed to parse TOML config: {}", path.display()))?
        };
        
        Ok(config)
    }
    
    /// Validate configuration for consistency and completeness
    pub fn validate(&self) -> Result<()> {
        // Validate network configuration
        if self.network.listen_addresses.is_empty() {
            return Err(anyhow!("At least one listen address must be specified"));
        }
        
        if self.network.max_connections == 0 {
            return Err(anyhow!("Max connections must be greater than 0"));
        }
        
        if !self.network.node_key_path.exists() {
            return Err(anyhow!("Node key file does not exist: {}", self.network.node_key_path.display()));
        }
        
        // Validate biological configuration
        if self.biological.preferred_roles.is_empty() {
            return Err(anyhow!("At least one preferred biological role must be specified"));
        }
        
        if !(0.0..=1.0).contains(&self.biological.trust_building_rate) {
            return Err(anyhow!("Trust building rate must be between 0.0 and 1.0"));
        }
        
        if !(0.0..=1.0).contains(&self.biological.cooperation_threshold) {
            return Err(anyhow!("Cooperation threshold must be between 0.0 and 1.0"));
        }
        
        if !(0.0..=1.0).contains(&self.biological.crisis_sensitivity) {
            return Err(anyhow!("Crisis sensitivity must be between 0.0 and 1.0"));
        }
        
        // Validate resource configuration
        if self.resources.max_cpu_cores == 0 {
            return Err(anyhow!("Max CPU cores must be greater than 0"));
        }
        
        if self.resources.max_memory_mb == 0 {
            return Err(anyhow!("Max memory must be greater than 0"));
        }
        
        if self.resources.max_disk_gb == 0 {
            return Err(anyhow!("Max disk space must be greater than 0"));
        }
        
        if !(0.0..=1.0).contains(&self.resources.scaling_sensitivity) {
            return Err(anyhow!("Scaling sensitivity must be between 0.0 and 1.0"));
        }
        
        if !(0.0..=1.0).contains(&self.resources.min_reserve_percentage) {
            return Err(anyhow!("Min reserve percentage must be between 0.0 and 1.0"));
        }
        
        // Validate security configuration
        if !(0.0..=1.0).contains(&self.security.trusted_node_threshold) {
            return Err(anyhow!("Trusted node threshold must be between 0.0 and 1.0"));
        }
        
        if !(0.0..=1.0).contains(&self.security.max_malicious_ratio) {
            return Err(anyhow!("Max malicious ratio must be between 0.0 and 1.0"));
        }
        
        if !(0.0..=1.0).contains(&self.security.threat_detection_sensitivity) {
            return Err(anyhow!("Threat detection sensitivity must be between 0.0 and 1.0"));
        }
        
        // Validate storage configuration
        if let Some(parent) = self.storage.data_dir.parent() {
            if !parent.exists() {
                return Err(anyhow!("Data directory parent does not exist: {}", parent.display()));
            }
        }
        
        if self.storage.max_cache_size_mb == 0 {
            return Err(anyhow!("Max cache size must be greater than 0"));
        }
        
        // Validate monitoring configuration
        if self.monitoring.metrics_port == 0 {
            return Err(anyhow!("Metrics port must be greater than 0"));
        }
        
        if self.monitoring.health_port == 0 {
            return Err(anyhow!("Health port must be greater than 0"));
        }
        
        if self.monitoring.metrics_port == self.monitoring.health_port {
            return Err(anyhow!("Metrics port and health port cannot be the same"));
        }
        
        // Validate daemon configuration
        if let Some(parent) = self.daemon.pid_file.parent() {
            if !parent.exists() {
                return Err(anyhow!("PID file directory does not exist: {}", parent.display()));
            }
        }
        
        // Validate economic configuration
        if self.economics.earning_rates.compute_rate < 0.0 {
            return Err(anyhow!("Compute earning rate cannot be negative"));
        }
        
        if self.economics.spending_rates.compute_cost <= 0.0 {
            return Err(anyhow!("Compute cost must be greater than 0"));
        }
        
        if !(0.0..=1.0).contains(&self.economics.reputation.decay_rate) {
            return Err(anyhow!("Reputation decay rate must be between 0.0 and 1.0"));
        }
        
        Ok(())
    }
    
    /// Create directories required by configuration
    pub async fn ensure_directories(&self) -> Result<()> {
        // Create storage directories
        tokio::fs::create_dir_all(&self.storage.data_dir).await
            .with_context(|| format!("Failed to create data directory: {}", self.storage.data_dir.display()))?;
        
        tokio::fs::create_dir_all(&self.storage.cache_dir).await
            .with_context(|| format!("Failed to create cache directory: {}", self.storage.cache_dir.display()))?;
        
        tokio::fs::create_dir_all(&self.storage.log_dir).await
            .with_context(|| format!("Failed to create log directory: {}", self.storage.log_dir.display()))?;
        
        if self.storage.backup.enabled {
            tokio::fs::create_dir_all(&self.storage.backup.backup_dir).await
                .with_context(|| format!("Failed to create backup directory: {}", self.storage.backup.backup_dir.display()))?;
        }
        
        // Create PID file directory
        if let Some(parent) = self.daemon.pid_file.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create PID file directory: {}", parent.display()))?;
        }
        
        Ok(())
    }
}

/// Configuration template content
pub const MINIMAL_CONFIG_TEMPLATE: &str = r#"
# Bio P2P Node - Minimal Configuration
# This is a minimal configuration suitable for development and testing

[network]
listen_addresses = ["/ip4/127.0.0.1/tcp/0"]
bootstrap_peers = []
node_key_path = "node_key.pem"
max_connections = 50
enable_mdns = true

[biological]
preferred_roles = ["YoungNode", "CasteNode"]
enable_social_behavior = true
trust_building_rate = 0.1
cooperation_threshold = 0.5
enable_havoc = false
crisis_sensitivity = 0.8
learning_rate = 0.05

[resources]
max_cpu_cores = 2
max_memory_mb = 2048
max_disk_gb = 10
allocation_strategy = "Conservative"
enable_dynamic_scaling = false

[security]
enable_behavior_monitoring = true
enable_thermal_detection = false
enable_illusion_layer = false
trusted_node_threshold = 0.7

[storage]
data_dir = "./dev_data"
cache_dir = "./dev_cache"
log_dir = "./dev_logs"
max_cache_size_mb = 256

[monitoring]
enable_metrics = true
metrics_port = 9091
enable_health_check = true
health_port = 8081

[daemon]
pid_file = "./bio-node.pid"
enable_graceful_shutdown = true

[economics]
enable_token_economics = false
"#;

pub const PRODUCTION_CONFIG_TEMPLATE: &str = r#"
# Bio P2P Node - Production Configuration
# Optimized for production deployment with enterprise-grade settings

[network]
listen_addresses = ["/ip4/0.0.0.0/tcp/7000"]
bootstrap_peers = [
    # Add bootstrap peers for your network
    # "/ip4/203.0.113.1/tcp/7000/p2p/12D3KooWExample"
]
node_key_path = "/etc/bio-node/node_key.pem"
max_connections = 100
enable_mdns = false
connection_timeout = "30s"
keep_alive_interval = "30s"
max_packet_size = 1048576
protocol_version = "/bio-p2p/1.0.0"

[biological]
preferred_roles = ["CasteNode", "HavocNode", "ThermalNode", "AddressNode"]
enable_social_behavior = true
trust_building_rate = 0.1
cooperation_threshold = 0.7
enable_havoc = true
crisis_sensitivity = 0.8
learning_rate = 0.05
max_learning_peers = 100
trust_decay_rate = 0.01
enable_thermal_monitoring = true
young_node_convergence_timeout = "300s"
caste_rebalancing_interval = "1s"

[resources]
max_cpu_cores = 8
max_memory_mb = 8192
max_disk_gb = 100
allocation_strategy = "Aggressive"
enable_dynamic_scaling = true
scaling_sensitivity = 0.7
min_reserve_percentage = 0.1
monitoring_interval = "5s"

[resources.compartment_memory_limits]
training_mb = 2048
inference_mb = 1024
storage_mb = 2048
communication_mb = 512
security_mb = 512

[security]
enable_behavior_monitoring = true
enable_thermal_detection = true
enable_illusion_layer = true
enable_cbadu = true
trusted_node_threshold = 0.8
max_malicious_ratio = 0.25
monitoring_interval = "1s"
threat_detection_sensitivity = 0.7
enable_forensic_logging = true
quarantine_duration = "300s"

[security.encryption]
symmetric_algorithm = "AES-256-GCM"
asymmetric_algorithm = "Ed25519"
hash_algorithm = "SHA3-256"
key_derivation = "PBKDF2"
key_rotation_interval = "86400s"

[storage]
data_dir = "/var/lib/bio-node/data"
cache_dir = "/var/lib/bio-node/cache"
log_dir = "/var/log/bio-node"
max_cache_size_mb = 4096
cache_eviction_policy = "LRU"

[storage.compression]
enabled = true
algorithm = "zstd"
level = 3

[storage.backup]
enabled = true
interval = "3600s"
retention_count = 48
backup_dir = "/var/lib/bio-node/backups"

[monitoring]
enable_metrics = true
metrics_addr = "0.0.0.0"
metrics_port = 9090
enable_health_check = true
health_port = 8080
enable_tracing = false
tracing_service_name = "bio-node"

[monitoring.logging]
level = "info"
format = "json"
max_file_size_mb = 100
max_files = 30
enable_console = false
enable_json = true

[monitoring.performance]
enable_cpu_monitoring = true
enable_memory_monitoring = true
enable_network_monitoring = true
enable_disk_monitoring = true
sample_interval = "5s"
history_retention = "86400s"

[daemon]
pid_file = "/var/run/bio-node.pid"
enable_graceful_shutdown = true
shutdown_timeout = "30s"
startup_timeout = "60s"
working_directory = "/var/lib/bio-node"

[economics]
enable_token_economics = true
initial_token_balance = 1000

[economics.earning_rates]
compute_rate = 10.0
bandwidth_rate = 0.1
storage_rate = 0.01
security_rate = 50.0
reliability_bonus = 1.5

[economics.spending_rates]
compute_cost = 12.0
bandwidth_cost = 0.12
storage_cost = 0.012
priority_multiplier = 2.0

[economics.staking]
minimum_stake = 100
reward_rate = 0.05
slashing_rate = 0.1
lock_duration = "604800s"

[economics.reputation]
initial_score = 500.0
decay_rate = 0.001
trusted_threshold = 700.0

[economics.reputation.weights]
task_completion = 1.0
uptime = 1.0
behavior_consistency = 0.3
security_contributions = 0.3
failure_penalty = 5.0
"#;

pub const DEVELOPMENT_CONFIG_TEMPLATE: &str = r#"
# Bio P2P Node - Development Configuration
# Optimized for development and debugging with verbose logging

[network]
listen_addresses = ["/ip4/127.0.0.1/tcp/0"]
bootstrap_peers = []
node_key_path = "dev_node_key.pem"
max_connections = 20
enable_mdns = true
connection_timeout = "10s"
keep_alive_interval = "15s"

[biological]
preferred_roles = ["YoungNode", "CasteNode", "ThermalNode"]
enable_social_behavior = true
trust_building_rate = 0.2
cooperation_threshold = 0.4
enable_havoc = true
crisis_sensitivity = 0.9
learning_rate = 0.1
max_learning_peers = 20

[resources]
max_cpu_cores = 4
max_memory_mb = 4096
max_disk_gb = 20
allocation_strategy = "Balanced"
enable_dynamic_scaling = true
scaling_sensitivity = 0.8

[security]
enable_behavior_monitoring = true
enable_thermal_detection = true
enable_illusion_layer = false
trusted_node_threshold = 0.6

[storage]
data_dir = "./dev_data"
cache_dir = "./dev_cache" 
log_dir = "./dev_logs"
max_cache_size_mb = 512

[monitoring]
enable_metrics = true
metrics_port = 9091
enable_health_check = true
health_port = 8081
enable_tracing = true
tracing_endpoint = "http://localhost:14268/api/traces"

[monitoring.logging]
level = "debug"
format = "pretty"
enable_console = true
enable_json = false

[daemon]
pid_file = "./dev_bio-node.pid"
enable_graceful_shutdown = true
shutdown_timeout = "10s"
working_directory = "."

[economics]
enable_token_economics = true
initial_token_balance = 10000
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_config_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_file = temp_dir.path().join("test.toml");
        
        tokio::fs::write(&config_file, MINIMAL_CONFIG_TEMPLATE).await.unwrap();
        
        let config = NodeConfig::load_from_file(&config_file).await;
        assert!(config.is_ok());
        
        let config = config.unwrap();
        assert!(!config.network.listen_addresses.is_empty());
        assert!(config.resources.max_cpu_cores > 0);
    }
    
    #[tokio::test] 
    async fn test_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        let node_key = temp_dir.path().join("node_key.pem");
        tokio::fs::write(&node_key, "dummy key content").await.unwrap();
        
        let mut config = NodeConfig::default();
        config.network.node_key_path = node_key;
        
        // Should pass validation
        assert!(config.validate().is_ok());
        
        // Test invalid configuration
        config.network.max_connections = 0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_biological_role_serialization() {
        let role = BiologicalRole::CasteNode;
        let serialized = serde_json::to_string(&role).unwrap();
        let deserialized: BiologicalRole = serde_json::from_str(&serialized).unwrap();
        assert_eq!(role, deserialized);
    }
    
    #[test]
    fn test_allocation_strategy_serialization() {
        let strategy = AllocationStrategy::Custom {
            cpu_percentage: 0.8,
            memory_percentage: 0.9,
            disk_percentage: 0.7,
        };
        
        let serialized = toml::to_string(&strategy).unwrap();
        let deserialized: AllocationStrategy = toml::from_str(&serialized).unwrap();
        
        if let AllocationStrategy::Custom { cpu_percentage, memory_percentage, disk_percentage } = deserialized {
            assert_eq!(cpu_percentage, 0.8);
            assert_eq!(memory_percentage, 0.9);
            assert_eq!(disk_percentage, 0.7);
        } else {
            panic!("Deserialization failed");
        }
    }
}