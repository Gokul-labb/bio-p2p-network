use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::net::SocketAddr;
use std::collections::HashMap;

/// Configuration for the P2P node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeConfig {
    /// Node identity configuration
    pub identity: IdentityConfig,
    
    /// Network transport configuration
    pub network: NetworkConfig,
    
    /// Discovery configuration
    pub discovery: DiscoveryConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Biological behavior configuration
    pub biological: BiologicalConfig,
    
    /// Resource management configuration
    pub resources: ResourceConfig,
    
    /// Protocol-specific configurations
    pub protocols: ProtocolConfig,
}

/// Node identity configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdentityConfig {
    /// Node name/alias (optional)
    pub name: Option<String>,
    
    /// Node description (optional)
    pub description: Option<String>,
    
    /// Whether to generate a new identity or load from file
    pub generate_new: bool,
    
    /// Path to identity file (keypair)
    pub identity_file: Option<String>,
    
    /// Node capabilities and specializations
    pub capabilities: Vec<String>,
}

/// Network transport configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// TCP listening addresses
    pub tcp_addresses: Vec<SocketAddr>,
    
    /// QUIC listening addresses (optional)
    pub quic_addresses: Vec<SocketAddr>,
    
    /// WebSocket listening addresses (optional)
    pub websocket_addresses: Vec<SocketAddr>,
    
    /// External addresses to announce (for NAT traversal)
    pub external_addresses: Vec<String>,
    
    /// Maximum number of concurrent connections
    pub max_connections: u32,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Keep-alive interval
    pub keep_alive_interval: Duration,
    
    /// Enable connection limits
    pub enable_connection_limits: bool,
    
    /// Enable AutoNAT for NAT detection
    pub enable_autonat: bool,
    
    /// Enable UPnP for port mapping
    pub enable_upnp: bool,
}

/// Discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable mDNS for local network discovery
    pub enable_mdns: bool,
    
    /// mDNS service name
    pub mdns_service_name: String,
    
    /// Enable Kademlia DHT
    pub enable_kademlia: bool,
    
    /// DHT replication factor
    pub kademlia_replication_factor: usize,
    
    /// Bootstrap nodes for DHT
    pub bootstrap_nodes: Vec<String>,
    
    /// Enable periodic bootstrap
    pub enable_periodic_bootstrap: bool,
    
    /// Bootstrap interval
    pub bootstrap_interval: Duration,
    
    /// Enable active discovery
    pub enable_active_discovery: bool,
    
    /// Discovery interval
    pub discovery_interval: Duration,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable Noise XX encryption
    pub enable_noise: bool,
    
    /// Trust score threshold for new connections
    pub trust_threshold: f64,
    
    /// Maximum Byzantine fault tolerance percentage
    pub max_byzantine_percentage: f64,
    
    /// Enable security validation
    pub enable_security_validation: bool,
    
    /// Security layer configurations
    pub security_layers: HashMap<String, bool>,
    
    /// Reputation scoring parameters
    pub reputation_config: ReputationConfig,
}

/// Reputation scoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationConfig {
    /// Weight for task completion score
    pub task_completion_weight: f64,
    
    /// Weight for node lifetime
    pub lifetime_weight: f64,
    
    /// Weight for security score
    pub security_weight: f64,
    
    /// Penalty for system shutdowns
    pub shutdown_penalty: f64,
    
    /// Minimum reputation for interaction
    pub minimum_reputation: f64,
    
    /// Reputation decay rate
    pub decay_rate: f64,
}

/// Biological behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    /// Primary biological role
    pub primary_role: String,
    
    /// Secondary biological roles
    pub secondary_roles: Vec<String>,
    
    /// Enable role switching
    pub enable_role_switching: bool,
    
    /// Role switching threshold
    pub role_switch_threshold: f64,
    
    /// Behavioral parameters
    pub behavior_params: HashMap<String, f64>,
    
    /// Hierarchy configuration
    pub hierarchy: HierarchyConfig,
    
    /// Swarm behavior settings
    pub swarm_behavior: SwarmConfig,
}

/// Hierarchical organization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyConfig {
    /// Enable hierarchical organization
    pub enabled: bool,
    
    /// Target hierarchy level (Alpha, Bravo, Super, etc.)
    pub target_level: String,
    
    /// Nodes per hierarchy level
    pub nodes_per_level: usize,
    
    /// Leadership election algorithm
    pub leadership_algorithm: String,
    
    /// Leadership timeout
    pub leadership_timeout: Duration,
}

/// Swarm behavior configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Enable swarm coordination
    pub enabled: bool,
    
    /// Swarm size preference
    pub preferred_swarm_size: usize,
    
    /// Maximum swarm size
    pub max_swarm_size: usize,
    
    /// Swarm formation algorithm
    pub formation_algorithm: String,
    
    /// Synchronization interval
    pub sync_interval: Duration,
    
    /// Cohesion factor
    pub cohesion_factor: f64,
    
    /// Separation factor
    pub separation_factor: f64,
    
    /// Alignment factor
    pub alignment_factor: f64,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// CPU allocation percentage (0.0-1.0)
    pub cpu_allocation: f64,
    
    /// Memory allocation in MB
    pub memory_allocation_mb: u64,
    
    /// Network bandwidth allocation in Mbps
    pub bandwidth_allocation_mbps: u64,
    
    /// Storage allocation in GB
    pub storage_allocation_gb: u64,
    
    /// Enable dynamic resource allocation
    pub enable_dynamic_allocation: bool,
    
    /// Resource monitoring interval
    pub monitoring_interval: Duration,
    
    /// HAVOC node emergency response threshold
    pub havoc_threshold: f64,
    
    /// Enable thermal monitoring
    pub enable_thermal_monitoring: bool,
}

/// Protocol-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolConfig {
    /// Gossipsub configuration
    pub gossipsub: GossipsubConfig,
    
    /// Custom protocol configurations
    pub custom_protocols: HashMap<String, ProtocolSettings>,
}

/// Gossipsub protocol configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GossipsubConfig {
    /// Message validation mode
    pub validation_mode: String,
    
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    
    /// History length
    pub history_length: usize,
    
    /// History gossip
    pub history_gossip: usize,
    
    /// Mesh size bounds (low, high)
    pub mesh_size_low: usize,
    pub mesh_size_high: usize,
    
    /// Outbound queue size
    pub outbound_queue_size: usize,
    
    /// Topics to subscribe to on startup
    pub default_topics: Vec<String>,
    
    /// Enable message signing
    pub enable_message_signing: bool,
}

/// Generic protocol settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolSettings {
    /// Protocol-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Enable this protocol
    pub enabled: bool,
    
    /// Protocol timeout
    pub timeout: Duration,
    
    /// Maximum concurrent streams
    pub max_concurrent_streams: usize,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            identity: IdentityConfig::default(),
            network: NetworkConfig::default(),
            discovery: DiscoveryConfig::default(),
            security: SecurityConfig::default(),
            biological: BiologicalConfig::default(),
            resources: ResourceConfig::default(),
            protocols: ProtocolConfig::default(),
        }
    }
}

impl Default for IdentityConfig {
    fn default() -> Self {
        Self {
            name: None,
            description: None,
            generate_new: true,
            identity_file: None,
            capabilities: vec!["general".to_string()],
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            tcp_addresses: vec!["0.0.0.0:0".parse().unwrap()],
            quic_addresses: vec![],
            websocket_addresses: vec![],
            external_addresses: vec![],
            max_connections: 100,
            connection_timeout: Duration::from_secs(10),
            keep_alive_interval: Duration::from_secs(30),
            enable_connection_limits: true,
            enable_autonat: true,
            enable_upnp: false,
        }
    }
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enable_mdns: true,
            mdns_service_name: "bio-p2p".to_string(),
            enable_kademlia: true,
            kademlia_replication_factor: 20,
            bootstrap_nodes: vec![],
            enable_periodic_bootstrap: true,
            bootstrap_interval: Duration::from_secs(300), // 5 minutes
            enable_active_discovery: true,
            discovery_interval: Duration::from_secs(60),
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_noise: true,
            trust_threshold: 0.5,
            max_byzantine_percentage: 0.25,
            enable_security_validation: true,
            security_layers: {
                let mut layers = HashMap::new();
                layers.insert("multi_layer_execution".to_string(), true);
                layers.insert("cbadu".to_string(), true);
                layers.insert("illusion_layer".to_string(), true);
                layers.insert("behavior_monitoring".to_string(), true);
                layers.insert("thermal_detection".to_string(), true);
                layers
            },
            reputation_config: ReputationConfig::default(),
        }
    }
}

impl Default for ReputationConfig {
    fn default() -> Self {
        Self {
            task_completion_weight: 1.0,
            lifetime_weight: 0.5,
            security_weight: 0.3,
            shutdown_penalty: 5.0,
            minimum_reputation: 0.0,
            decay_rate: 0.01,
        }
    }
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        Self {
            primary_role: "Young".to_string(), // Start as learning node
            secondary_roles: vec![],
            enable_role_switching: true,
            role_switch_threshold: 0.7,
            behavior_params: HashMap::new(),
            hierarchy: HierarchyConfig::default(),
            swarm_behavior: SwarmConfig::default(),
        }
    }
}

impl Default for HierarchyConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_level: "Individual".to_string(),
            nodes_per_level: 3,
            leadership_algorithm: "democratic".to_string(),
            leadership_timeout: Duration::from_secs(60),
        }
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            preferred_swarm_size: 5,
            max_swarm_size: 20,
            formation_algorithm: "boids".to_string(),
            sync_interval: Duration::from_secs(5),
            cohesion_factor: 1.0,
            separation_factor: 1.0,
            alignment_factor: 1.0,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            cpu_allocation: 0.5,
            memory_allocation_mb: 1024,
            bandwidth_allocation_mbps: 10,
            storage_allocation_gb: 10,
            enable_dynamic_allocation: true,
            monitoring_interval: Duration::from_secs(30),
            havoc_threshold: 0.8,
            enable_thermal_monitoring: true,
        }
    }
}

impl Default for ProtocolConfig {
    fn default() -> Self {
        Self {
            gossipsub: GossipsubConfig::default(),
            custom_protocols: HashMap::new(),
        }
    }
}

impl Default for GossipsubConfig {
    fn default() -> Self {
        Self {
            validation_mode: "permissive".to_string(),
            heartbeat_interval: Duration::from_millis(700),
            history_length: 5,
            history_gossip: 3,
            mesh_size_low: 4,
            mesh_size_high: 12,
            outbound_queue_size: 1024,
            default_topics: vec!["bio-p2p-general".to_string(), "pheromones".to_string()],
            enable_message_signing: true,
        }
    }
}

impl NodeConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::P2PError::ConfigurationError {
                field: "file_read".to_string(),
                reason: e.to_string(),
            })?;
        
        toml::from_str(&content)
            .map_err(|e| crate::P2PError::ConfigurationError {
                field: "toml_parse".to_string(),
                reason: e.to_string(),
            })
    }
    
    /// Save configuration to a TOML file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> crate::Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| crate::P2PError::ConfigurationError {
                field: "toml_serialize".to_string(),
                reason: e.to_string(),
            })?;
        
        std::fs::write(path, content)
            .map_err(|e| crate::P2PError::ConfigurationError {
                field: "file_write".to_string(),
                reason: e.to_string(),
            })
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        // Validate network configuration
        if self.network.max_connections == 0 {
            return Err(crate::P2PError::ConfigurationError {
                field: "network.max_connections".to_string(),
                reason: "Must be greater than 0".to_string(),
            });
        }
        
        // Validate security configuration
        if self.security.trust_threshold < 0.0 || self.security.trust_threshold > 1.0 {
            return Err(crate::P2PError::ConfigurationError {
                field: "security.trust_threshold".to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        
        if self.security.max_byzantine_percentage < 0.0 || self.security.max_byzantine_percentage > 0.5 {
            return Err(crate::P2PError::ConfigurationError {
                field: "security.max_byzantine_percentage".to_string(),
                reason: "Must be between 0.0 and 0.5".to_string(),
            });
        }
        
        // Validate resource configuration
        if self.resources.cpu_allocation < 0.0 || self.resources.cpu_allocation > 1.0 {
            return Err(crate::P2PError::ConfigurationError {
                field: "resources.cpu_allocation".to_string(),
                reason: "Must be between 0.0 and 1.0".to_string(),
            });
        }
        
        // Validate biological configuration
        if self.biological.hierarchy.nodes_per_level == 0 {
            return Err(crate::P2PError::ConfigurationError {
                field: "biological.hierarchy.nodes_per_level".to_string(),
                reason: "Must be greater than 0".to_string(),
            });
        }
        
        Ok(())
    }
    
    /// Merge with another configuration (other takes precedence)
    pub fn merge(mut self, other: Self) -> Self {
        // Simple merge - in a real implementation, you'd want more sophisticated merging
        // This is a placeholder that replaces entire sections
        self.identity = other.identity;
        self.network = other.network;
        self.discovery = other.discovery;
        self.security = other.security;
        self.biological = other.biological;
        self.resources = other.resources;
        self.protocols = other.protocols;
        self
    }
    
    /// Create a configuration optimized for testing
    pub fn for_testing() -> Self {
        let mut config = Self::default();
        config.network.tcp_addresses = vec!["127.0.0.1:0".parse().unwrap()];
        config.network.max_connections = 10;
        config.discovery.bootstrap_interval = Duration::from_secs(10);
        config.discovery.discovery_interval = Duration::from_secs(5);
        config.resources.monitoring_interval = Duration::from_secs(5);
        config.biological.swarm_behavior.sync_interval = Duration::from_secs(2);
        config
    }
    
    /// Create a lightweight configuration for resource-constrained environments
    pub fn lightweight() -> Self {
        let mut config = Self::default();
        config.network.max_connections = 20;
        config.resources.cpu_allocation = 0.2;
        config.resources.memory_allocation_mb = 256;
        config.resources.bandwidth_allocation_mbps = 5;
        config.resources.storage_allocation_gb = 2;
        config.discovery.enable_kademlia = false; // Reduce resource usage
        config.biological.swarm_behavior.preferred_swarm_size = 3;
        config.biological.swarm_behavior.max_swarm_size = 8;
        config
    }
}