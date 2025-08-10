use async_trait::async_trait;
use libp2p::PeerId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use core_protocol::{BiologicalRole, NetworkAddress, NodeMessage, ThermalSignature};
use crate::{Result, P2PError};

/// Trait for biological roles that nodes can assume
#[async_trait]
pub trait BiologicalBehavior: Send + Sync {
    /// Get the role type
    fn role(&self) -> BiologicalRole;
    
    /// Get the role name
    fn role_name(&self) -> &str;
    
    /// Get biological inspiration description
    fn biological_inspiration(&self) -> &str;
    
    /// Initialize the role with given parameters
    async fn initialize(&mut self, params: RoleParameters) -> Result<()>;
    
    /// Process incoming messages
    async fn handle_message(&mut self, message: NodeMessage, from: PeerId) -> Result<Vec<NodeMessage>>;
    
    /// Perform periodic behavior updates
    async fn update(&mut self) -> Result<Vec<BiologicalAction>>;
    
    /// Get current behavior metrics
    fn metrics(&self) -> BehaviorMetrics;
    
    /// Check if role can handle specific capability
    fn can_handle_capability(&self, capability: &str) -> bool;
    
    /// Assess compatibility with another role
    fn compatibility_score(&self, other: &BiologicalRole) -> f64;
    
    /// Determine if this role should switch to another
    fn should_switch_role(&self, network_state: &NetworkState) -> Option<BiologicalRole>;
}

/// Parameters for initializing biological roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleParameters {
    /// Node's network address
    pub network_address: NetworkAddress,
    
    /// Initial peer connections
    pub initial_peers: Vec<PeerId>,
    
    /// Role-specific configuration
    pub config: HashMap<String, f64>,
    
    /// Available resources
    pub resources: ResourceAllocation,
    
    /// Network capabilities
    pub capabilities: Vec<String>,
}

/// Resource allocation for biological roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU allocation (0.0-1.0)
    pub cpu: f64,
    
    /// Memory allocation in MB
    pub memory_mb: u64,
    
    /// Network bandwidth in Mbps
    pub bandwidth_mbps: u64,
    
    /// Storage allocation in GB
    pub storage_gb: u64,
}

/// Action that a biological role wants to perform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalAction {
    /// Send message to specific peer
    SendMessage { to: PeerId, message: NodeMessage },
    
    /// Broadcast message to all peers
    Broadcast { message: NodeMessage },
    
    /// Subscribe to a gossipsub topic
    Subscribe { topic: String },
    
    /// Unsubscribe from a gossipsub topic
    Unsubscribe { topic: String },
    
    /// Connect to a new peer
    ConnectToPeer { peer_id: PeerId, address: Option<String> },
    
    /// Disconnect from a peer
    DisconnectFromPeer { peer_id: PeerId },
    
    /// Update resource allocation
    UpdateResources { allocation: ResourceAllocation },
    
    /// Emit thermal signature
    EmitThermalSignature { signature: ThermalSignature },
    
    /// Form hierarchical group
    FormGroup { peers: Vec<PeerId>, group_type: String },
    
    /// Leave current group
    LeaveGroup,
    
    /// Switch to different role
    SwitchRole { new_role: BiologicalRole },
}

/// Metrics for biological behavior performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehaviorMetrics {
    /// Messages processed
    pub messages_processed: u64,
    
    /// Actions performed
    pub actions_performed: u64,
    
    /// Success rate (0.0-1.0)
    pub success_rate: f64,
    
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    
    /// Performance score
    pub performance_score: f64,
    
    /// Role-specific metrics
    pub role_metrics: HashMap<String, f64>,
}

/// Current resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,
    
    /// Memory usage in MB
    pub memory_usage_mb: u64,
    
    /// Network usage in Mbps
    pub network_usage_mbps: u64,
    
    /// Storage usage in GB
    pub storage_usage_gb: u64,
}

/// Current network state for role decision making
#[derive(Debug, Clone)]
pub struct NetworkState {
    /// Connected peers and their roles
    pub peers: HashMap<PeerId, BiologicalRole>,
    
    /// Network size
    pub network_size: usize,
    
    /// Network health metrics
    pub network_health: f64,
    
    /// Resource demand
    pub resource_demand: HashMap<String, f64>,
    
    /// Current network topology
    pub topology_metrics: TopologyMetrics,
}

/// Network topology metrics
#[derive(Debug, Clone)]
pub struct TopologyMetrics {
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    
    /// Average path length
    pub average_path_length: f64,
    
    /// Network density
    pub density: f64,
    
    /// Degree distribution
    pub degree_distribution: Vec<usize>,
}

/// Young Node implementation (Crow Culture Learning)
#[derive(Debug, Clone)]
pub struct YoungNode {
    /// Learning parameters
    discovery_radius: usize,
    convergence_time: Duration,
    memory_usage_mb: usize,
    
    /// Learning state
    experienced_peers: Vec<PeerId>,
    learned_routes: HashMap<String, Vec<PeerId>>,
    learning_progress: f64,
    
    /// Metrics
    metrics: BehaviorMetrics,
}

impl YoungNode {
    pub fn new() -> Self {
        Self {
            discovery_radius: 100,
            convergence_time: Duration::from_secs(30),
            memory_usage_mb: 50,
            experienced_peers: Vec::new(),
            learned_routes: HashMap::new(),
            learning_progress: 0.0,
            metrics: BehaviorMetrics::default(),
        }
    }
    
    /// Learn from experienced neighboring nodes
    async fn learn_from_neighbors(&mut self, peers: &[PeerId]) -> Result<Vec<BiologicalAction>> {
        let mut actions = Vec::new();
        
        // Find experienced peers to learn from
        for peer in peers.iter().take(self.discovery_radius) {
            if !self.experienced_peers.contains(peer) {
                self.experienced_peers.push(*peer);
                
                // Request routing information
                let message = NodeMessage::RouteRequest {
                    request_id: uuid::Uuid::new_v4().to_string(),
                    destination: "any".to_string(),
                    max_hops: 10,
                };
                
                actions.push(BiologicalAction::SendMessage {
                    to: *peer,
                    message,
                });
            }
        }
        
        Ok(actions)
    }
}

#[async_trait]
impl BiologicalBehavior for YoungNode {
    fn role(&self) -> BiologicalRole {
        BiologicalRole::Young
    }
    
    fn role_name(&self) -> &str {
        "Young Node (Crow Culture Learning)"
    }
    
    fn biological_inspiration(&self) -> &str {
        "Young crows learn hunting techniques, tool use, and territorial navigation by observing experienced adults within their social groups"
    }
    
    async fn initialize(&mut self, params: RoleParameters) -> Result<()> {
        // Configure based on parameters
        if let Some(&radius) = params.config.get("discovery_radius") {
            self.discovery_radius = radius as usize;
        }
        
        if let Some(&convergence) = params.config.get("convergence_time_secs") {
            self.convergence_time = Duration::from_secs(convergence as u64);
        }
        
        self.experienced_peers = params.initial_peers;
        
        Ok(())
    }
    
    async fn handle_message(&mut self, message: NodeMessage, from: PeerId) -> Result<Vec<NodeMessage>> {
        match message {
            NodeMessage::RouteResponse { routes, .. } => {
                // Learn routing information
                for route in routes {
                    self.learned_routes.entry(route.destination)
                        .or_insert_with(Vec::new)
                        .push(from);
                }
                
                self.learning_progress = (self.learned_routes.len() as f64 / 100.0).min(1.0);
                self.metrics.messages_processed += 1;
                
                Ok(Vec::new())
            },
            NodeMessage::RouteRequest { request_id, destination, .. } => {
                // Share learned routes if available
                if let Some(route_peers) = self.learned_routes.get(&destination) {
                    let routes = route_peers.iter().map(|peer| core_protocol::RouteInfo {
                        destination: destination.clone(),
                        next_hop: peer.to_string(),
                        distance: 1,
                        quality: 0.8,
                    }).collect();
                    
                    Ok(vec![NodeMessage::RouteResponse {
                        request_id,
                        routes,
                        source: "young_node".to_string(),
                    }])
                } else {
                    Ok(Vec::new())
                }
            },
            _ => Ok(Vec::new()),
        }
    }
    
    async fn update(&mut self) -> Result<Vec<BiologicalAction>> {
        let mut actions = Vec::new();
        
        // Periodic learning from experienced peers
        if !self.experienced_peers.is_empty() && self.learning_progress < 1.0 {
            let learn_actions = self.learn_from_neighbors(&self.experienced_peers.clone()).await?;
            actions.extend(learn_actions);
        }
        
        // Update metrics
        self.metrics.performance_score = self.learning_progress;
        self.metrics.role_metrics.insert("learning_progress".to_string(), self.learning_progress);
        self.metrics.role_metrics.insert("routes_learned".to_string(), self.learned_routes.len() as f64);
        
        Ok(actions)
    }
    
    fn metrics(&self) -> BehaviorMetrics {
        self.metrics.clone()
    }
    
    fn can_handle_capability(&self, capability: &str) -> bool {
        matches!(capability, "learning" | "routing" | "discovery")
    }
    
    fn compatibility_score(&self, other: &BiologicalRole) -> f64 {
        match other {
            BiologicalRole::Memory | BiologicalRole::Trust => 0.9, // High compatibility with mentors
            BiologicalRole::Young => 0.7, // Good compatibility with peers
            BiologicalRole::Caste => 0.6, // Can benefit from specialization
            _ => 0.5,
        }
    }
    
    fn should_switch_role(&self, network_state: &NetworkState) -> Option<BiologicalRole> {
        // Switch to Caste node when sufficiently learned
        if self.learning_progress > 0.8 && network_state.resource_demand.get("specialization").unwrap_or(&0.0) > &0.6 {
            return Some(BiologicalRole::Caste);
        }
        
        // Switch to Trust node if network needs trust management
        if self.learning_progress > 0.7 && network_state.network_size > 10 {
            if let Some(&trust_demand) = network_state.resource_demand.get("trust_management") {
                if trust_demand > 0.7 {
                    return Some(BiologicalRole::Trust);
                }
            }
        }
        
        None
    }
}

/// Caste Node implementation (Ant Colony Division of Labor)
#[derive(Debug, Clone)]
pub struct CasteNode {
    /// Compartments
    compartments: HashMap<String, CompartmentState>,
    dynamic_sizing: bool,
    utilization_target: f64,
    
    /// Metrics
    metrics: BehaviorMetrics,
}

/// State of a computational compartment
#[derive(Debug, Clone)]
pub struct CompartmentState {
    /// Compartment type
    compartment_type: String,
    
    /// Resource allocation
    resources: ResourceAllocation,
    
    /// Current utilization
    utilization: f64,
    
    /// Tasks in queue
    task_queue_size: usize,
    
    /// Performance metrics
    performance_score: f64,
}

impl CasteNode {
    pub fn new() -> Self {
        let mut compartments = HashMap::new();
        
        // Initialize default compartments
        compartments.insert("training".to_string(), CompartmentState {
            compartment_type: "training".to_string(),
            resources: ResourceAllocation {
                cpu: 0.4,
                memory_mb: 512,
                bandwidth_mbps: 5,
                storage_gb: 5,
            },
            utilization: 0.0,
            task_queue_size: 0,
            performance_score: 1.0,
        });
        
        compartments.insert("inference".to_string(), CompartmentState {
            compartment_type: "inference".to_string(),
            resources: ResourceAllocation {
                cpu: 0.3,
                memory_mb: 256,
                bandwidth_mbps: 3,
                storage_gb: 2,
            },
            utilization: 0.0,
            task_queue_size: 0,
            performance_score: 1.0,
        });
        
        compartments.insert("storage".to_string(), CompartmentState {
            compartment_type: "storage".to_string(),
            resources: ResourceAllocation {
                cpu: 0.1,
                memory_mb: 128,
                bandwidth_mbps: 1,
                storage_gb: 8,
            },
            utilization: 0.0,
            task_queue_size: 0,
            performance_score: 1.0,
        });
        
        compartments.insert("communication".to_string(), CompartmentState {
            compartment_type: "communication".to_string(),
            resources: ResourceAllocation {
                cpu: 0.1,
                memory_mb: 64,
                bandwidth_mbps: 6,
                storage_gb: 1,
            },
            utilization: 0.0,
            task_queue_size: 0,
            performance_score: 1.0,
        });
        
        compartments.insert("security".to_string(), CompartmentState {
            compartment_type: "security".to_string(),
            resources: ResourceAllocation {
                cpu: 0.1,
                memory_mb: 128,
                bandwidth_mbps: 1,
                storage_gb: 1,
            },
            utilization: 0.0,
            task_queue_size: 0,
            performance_score: 1.0,
        });
        
        Self {
            compartments,
            dynamic_sizing: true,
            utilization_target: 0.85,
            metrics: BehaviorMetrics::default(),
        }
    }
    
    /// Rebalance compartment resources based on demand
    fn rebalance_compartments(&mut self) -> Vec<BiologicalAction> {
        if !self.dynamic_sizing {
            return Vec::new();
        }
        
        let mut actions = Vec::new();
        let total_demand: f64 = self.compartments.values()
            .map(|c| c.utilization * c.task_queue_size as f64)
            .sum();
        
        if total_demand > 0.0 {
            for compartment in self.compartments.values_mut() {
                let demand_ratio = (compartment.utilization * compartment.task_queue_size as f64) / total_demand;
                
                // Adjust CPU allocation based on demand
                let new_cpu = (demand_ratio * 0.8).max(0.05); // Reserve 20% for overhead
                if (new_cpu - compartment.resources.cpu).abs() > 0.05 {
                    compartment.resources.cpu = new_cpu;
                    
                    actions.push(BiologicalAction::UpdateResources {
                        allocation: compartment.resources.clone(),
                    });
                }
            }
        }
        
        actions
    }
}

#[async_trait]
impl BiologicalBehavior for CasteNode {
    fn role(&self) -> BiologicalRole {
        BiologicalRole::Caste
    }
    
    fn role_name(&self) -> &str {
        "Caste Node (Ant Colony Division of Labor)"
    }
    
    fn biological_inspiration(&self) -> &str {
        "Ant colonies achieve remarkable efficiency through specialized castes (workers, soldiers, nurses) that perform distinct functions"
    }
    
    async fn initialize(&mut self, params: RoleParameters) -> Result<()> {
        // Configure compartments based on node capabilities
        for capability in &params.capabilities {
            if let Some(compartment) = self.compartments.get_mut(capability) {
                // Allocate more resources to matching capabilities
                compartment.resources.cpu *= 1.5;
                compartment.resources.memory_mb = (compartment.resources.memory_mb as f64 * 1.5) as u64;
            }
        }
        
        if let Some(&dynamic) = params.config.get("dynamic_sizing") {
            self.dynamic_sizing = dynamic > 0.5;
        }
        
        if let Some(&target) = params.config.get("utilization_target") {
            self.utilization_target = target;
        }
        
        Ok(())
    }
    
    async fn handle_message(&mut self, message: NodeMessage, _from: PeerId) -> Result<Vec<NodeMessage>> {
        match message {
            NodeMessage::ComputeRequest { task_type, .. } => {
                // Route to appropriate compartment
                if let Some(compartment) = self.compartments.get_mut(&task_type) {
                    compartment.task_queue_size += 1;
                    compartment.utilization = (compartment.task_queue_size as f64 * 0.1).min(1.0);
                }
                
                self.metrics.messages_processed += 1;
                Ok(Vec::new())
            },
            NodeMessage::ResourceUpdate { utilization } => {
                // Update compartment utilization
                for (name, util) in utilization {
                    if let Some(compartment) = self.compartments.get_mut(&name) {
                        compartment.utilization = util;
                    }
                }
                Ok(Vec::new())
            },
            _ => Ok(Vec::new()),
        }
    }
    
    async fn update(&mut self) -> Result<Vec<BiologicalAction>> {
        let mut actions = Vec::new();
        
        // Rebalance compartments
        let rebalance_actions = self.rebalance_compartments();
        actions.extend(rebalance_actions);
        
        // Calculate overall performance
        let avg_utilization: f64 = self.compartments.values()
            .map(|c| c.utilization)
            .sum::<f64>() / self.compartments.len() as f64;
        
        self.metrics.performance_score = avg_utilization;
        self.metrics.resource_utilization.cpu_usage = avg_utilization;
        self.metrics.role_metrics.insert("avg_utilization".to_string(), avg_utilization);
        self.metrics.role_metrics.insert("compartment_count".to_string(), self.compartments.len() as f64);
        
        Ok(actions)
    }
    
    fn metrics(&self) -> BehaviorMetrics {
        self.metrics.clone()
    }
    
    fn can_handle_capability(&self, capability: &str) -> bool {
        self.compartments.contains_key(capability)
    }
    
    fn compatibility_score(&self, other: &BiologicalRole) -> f64 {
        match other {
            BiologicalRole::Hatch | BiologicalRole::Sync => 0.9, // High compatibility with coordinators
            BiologicalRole::HAVOC => 0.8, // Good for emergency resource allocation
            BiologicalRole::Thermal => 0.8, // Works well with resource monitoring
            _ => 0.6,
        }
    }
    
    fn should_switch_role(&self, network_state: &NetworkState) -> Option<BiologicalRole> {
        let avg_utilization = self.compartments.values()
            .map(|c| c.utilization)
            .sum::<f64>() / self.compartments.len() as f64;
        
        // Switch to HAVOC if under severe stress
        if avg_utilization > 0.95 && network_state.network_health < 0.5 {
            return Some(BiologicalRole::HAVOC);
        }
        
        None
    }
}

impl Default for BehaviorMetrics {
    fn default() -> Self {
        Self {
            messages_processed: 0,
            actions_performed: 0,
            success_rate: 1.0,
            resource_utilization: ResourceUtilization {
                cpu_usage: 0.0,
                memory_usage_mb: 0,
                network_usage_mbps: 0,
                storage_usage_gb: 0,
            },
            performance_score: 1.0,
            role_metrics: HashMap::new(),
        }
    }
}

/// Factory for creating biological behavior implementations
pub struct BiologicalBehaviorFactory;

impl BiologicalBehaviorFactory {
    /// Create a new biological behavior based on role
    pub fn create_behavior(role: &BiologicalRole) -> Result<Box<dyn BiologicalBehavior>> {
        match role {
            BiologicalRole::Young => Ok(Box::new(YoungNode::new())),
            BiologicalRole::Caste => Ok(Box::new(CasteNode::new())),
            // TODO: Implement other roles
            _ => Err(P2PError::InvalidBiologicalRole {
                role: format!("{:?}", role),
            }),
        }
    }
    
    /// List all available biological roles
    pub fn available_roles() -> Vec<BiologicalRole> {
        vec![
            BiologicalRole::Young,
            BiologicalRole::Caste,
            // TODO: Add other roles as implemented
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_young_node_initialization() {
        let mut young_node = YoungNode::new();
        
        let params = RoleParameters {
            network_address: core_protocol::NetworkAddress::new(1, 2, 3).unwrap(),
            initial_peers: vec![PeerId::random()],
            config: {
                let mut config = HashMap::new();
                config.insert("discovery_radius".to_string(), 50.0);
                config.insert("convergence_time_secs".to_string(), 60.0);
                config
            },
            resources: ResourceAllocation {
                cpu: 0.5,
                memory_mb: 1024,
                bandwidth_mbps: 10,
                storage_gb: 5,
            },
            capabilities: vec!["learning".to_string()],
        };
        
        let result = young_node.initialize(params).await;
        assert!(result.is_ok());
        assert_eq!(young_node.discovery_radius, 50);
        assert_eq!(young_node.convergence_time, Duration::from_secs(60));
    }
    
    #[tokio::test]
    async fn test_caste_node_compartments() {
        let mut caste_node = CasteNode::new();
        
        // Verify default compartments
        assert!(caste_node.compartments.contains_key("training"));
        assert!(caste_node.compartments.contains_key("inference"));
        assert!(caste_node.compartments.contains_key("storage"));
        assert!(caste_node.compartments.contains_key("communication"));
        assert!(caste_node.compartments.contains_key("security"));
        
        // Test message handling
        let message = NodeMessage::ComputeRequest {
            request_id: "test".to_string(),
            task_type: "training".to_string(),
            payload: Vec::new(),
            priority: 5,
            deadline: None,
        };
        
        let response = caste_node.handle_message(message, PeerId::random()).await;
        assert!(response.is_ok());
        
        // Verify task was queued
        assert_eq!(caste_node.compartments["training"].task_queue_size, 1);
    }
    
    #[test]
    fn test_behavior_factory() {
        let young_behavior = BiologicalBehaviorFactory::create_behavior(&BiologicalRole::Young);
        assert!(young_behavior.is_ok());
        
        let caste_behavior = BiologicalBehaviorFactory::create_behavior(&BiologicalRole::Caste);
        assert!(caste_behavior.is_ok());
        
        let invalid_behavior = BiologicalBehaviorFactory::create_behavior(&BiologicalRole::Memory);
        assert!(invalid_behavior.is_err());
    }
    
    #[test]
    fn test_role_compatibility() {
        let young_node = YoungNode::new();
        
        assert_eq!(young_node.compatibility_score(&BiologicalRole::Memory), 0.9);
        assert_eq!(young_node.compatibility_score(&BiologicalRole::Trust), 0.9);
        assert_eq!(young_node.compatibility_score(&BiologicalRole::Young), 0.7);
        assert_eq!(young_node.compatibility_score(&BiologicalRole::Caste), 0.6);
    }
}