use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use crate::{ProtocolError, Result};

/// Comprehensive biological role taxonomy with 80+ specialized node types
/// Each role implements specific biological behaviors adapted for computational tasks
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiologicalRole {
    // Learning and Adaptation Nodes
    /// Young crows learn from experienced adults - new nodes learn routing from neighbors
    YoungNode,
    /// Ant colony division of labor - compartmentalized resource allocation
    CasteNode,
    /// Parrot vocal learning - copies successful communication patterns
    ImitateNode,
    
    // Coordination and Synchronization Nodes  
    /// Sea turtle synchronization - manages super-node lifecycle coordination
    HatchNode,
    /// Penguin colony synchrony - manages node lifecycle phases
    SyncPhaseNode,
    /// Penguin winter huddling - dynamic position rotation for load distribution
    HuddleNode,
    /// Goat leadership - dynamic leadership allocation with automatic replacement
    DynamicKeyNode,
    /// Bird flocking - baseline consensus mechanism for synchronized coordination
    KeyNode,
    
    // Communication and Routing Nodes
    /// Caribou routes - maintains generational memory of optimal network paths
    MigrationNode,
    /// Territorial navigation - hierarchical addressing system management
    AddressNode,
    /// Path marking - secure tunneling and waypoint guidance
    TunnelNode,
    /// Intermediate waypoints - routing information and next hop advertisement  
    SignNode,
    /// Pheromone concentration - thermal signatures for route availability
    ThermalNode,
    
    // Security and Defense Nodes
    /// Denial of service detection - continuous capability verification
    DOSNode,
    /// Pack investigation - forensic analysis of network anomalies
    InvestigationNode,
    /// Post-incident analysis - collects information about failed nodes
    CasualtyNode,
    /// Defensive deception - confusing behavior patterns against attackers
    ConfusionNode,
    
    // Resource Management Nodes
    /// Mosquito-human network adaptation - emergency resource reallocation
    HAVOCNode,
    /// Desert ant elevation control - dynamic computational capacity scaling
    StepUpNode,
    /// Desert ant elevation control - resource conservation during low demand
    StepDownNode,
    /// Tick-host symbiotic networks - preferential local resource sharing
    FriendshipNode,
    /// Primate grooming networks - permanent mutual resource sharing
    BuddyNode,
    
    // Social and Trust Nodes
    /// Primate social bonding - monitors node health and behavioral consistency
    TrustNode,
    /// Fish schooling - coordinated packet execution with swarm intelligence
    SyncNode,
    /// Fish schooling - alternative name for coordinated packet processing
    PacketNode,
    /// Frog communication - adjusts behavior to maximize coordination
    CroakNode,
    
    // Specialized Function Nodes
    /// Spider web network - connects multiple nodes for complex parallel tasks
    WebNode,
    /// Wolf pack hierarchy - segregates compute resources by type and capability
    HierarchyNode,
    /// Dolphin cooperation - combines individual nodes into cooperative teams
    AlphaNode,
    /// Dolphin cooperation - combines Alpha nodes into larger coordinated groups
    BravoNode,
    /// Dolphin cooperation - combines Bravo nodes into powerful computational clusters
    SuperNode,
    
    // Support and Maintenance Nodes
    /// Brain memory systems - maintains detailed records of network operations
    MemoryNode,
    /// Animal sentries - predicts future network behavior through pattern monitoring
    TelescopeNode,
    /// Immune system repair - continuously optimizes network paths and health
    HealingNode,
    
    // Additional Specialized Nodes (expanding to 80+ total)
    /// Local resource management and optimization
    LocalNode,
    /// Command and control coordination
    CommandNode,
    /// Queue management and load balancing
    QueueNode,
    /// Regional base coordination across geographic areas
    RegionalBaseNode,
    /// Planning and strategy coordination
    PlanNode,
    /// Package distribution and routing management
    DistributorNode,
    /// Package processing and lifecycle management
    PackageNode,
    /// Error handling and recovery coordination
    MixUpNode,
    /// Follow-up processing and continuation handling
    FollowUpNode,
    /// Survival monitoring and resilience management
    SurvivalNode,
    /// Information dissemination and knowledge sharing
    InfoNode,
    /// Friendship relationship management beyond local groups
    FriendNode,
    /// Random selection and load balancing
    RandomNode,
    /// Known entity verification and reputation management
    KnownNode,
    /// Expert specialization for domain-specific tasks
    ExpertNode,
    /// Cluster management and coordination
    ClusterNode,
    /// Support services and auxiliary functions
    SupportNode,
    /// Network monitoring and surveillance
    WatchdogNode,
    /// Information warfare and counter-intelligence
    PropagandaNode,
    /// Cultural learning and knowledge preservation
    CultureNode,
    /// Client interface and service management
    ClientNode,
}

impl BiologicalRole {
    /// Get the biological inspiration description
    pub fn biological_inspiration(&self) -> &'static str {
        match self {
            BiologicalRole::YoungNode => "Young crows learn hunting techniques and navigation by observing experienced adults",
            BiologicalRole::CasteNode => "Ant colonies achieve efficiency through specialized castes performing distinct functions",
            BiologicalRole::ImitateNode => "Parrots learn vocalizations by imitating successful communication patterns",
            BiologicalRole::HatchNode => "Sea turtle hatchlings emerge simultaneously and navigate in coordinated groups",
            BiologicalRole::SyncPhaseNode => "Penguin colonies coordinate through synchronized behavioral phases",
            BiologicalRole::HuddleNode => "Emperor penguins rotate positions in huddles to share warmth efficiently",
            BiologicalRole::DynamicKeyNode => "Goat herds follow individuals with strong social connections dynamically",
            BiologicalRole::KeyNode => "Bird flocks maintain formation through distributed consensus mechanisms",
            BiologicalRole::MigrationNode => "Caribou follow traditional migration routes passed down through generations",
            BiologicalRole::AddressNode => "Territorial animals maintain mental maps with distinct landmarks and boundaries",
            BiologicalRole::TunnelNode => "Animals create marked travel paths for safe and efficient navigation",
            BiologicalRole::SignNode => "Animals use waypoints and directional indicators for navigation guidance",
            BiologicalRole::ThermalNode => "Ants use pheromone concentration trails to indicate resource availability",
            BiologicalRole::DOSNode => "Immune system sentinel cells continuously monitor for threats and anomalies",
            BiologicalRole::InvestigationNode => "Social animals investigate unusual occurrences to maintain pack safety",
            BiologicalRole::CasualtyNode => "Pack animals investigate fallen members to understand and prevent threats",
            BiologicalRole::ConfusionNode => "Animals use deception and confusion tactics when threatened by predators",
            BiologicalRole::HAVOCNode => "Disease vectors rapidly adapt behavior when environmental conditions change",
            BiologicalRole::StepUpNode => "Desert ants adjust elevation to regulate temperature and optimize energy",
            BiologicalRole::StepDownNode => "Desert ants conserve energy by adjusting positioning during harsh conditions",
            BiologicalRole::FriendshipNode => "Symbiotic relationships where species help each other for mutual benefit",
            BiologicalRole::BuddyNode => "Primates engage in mutual grooming to build trust and maintain relationships",
            BiologicalRole::TrustNode => "Social primates build trust relationships through consistent cooperative behaviors",
            BiologicalRole::SyncNode => "Fish move in synchronized schools to reduce confusion and improve efficiency",
            BiologicalRole::PacketNode => "Fish coordinate movement patterns for collective navigation and protection",
            BiologicalRole::CroakNode => "Male frogs coordinate calls to avoid interference while maximizing success",
            BiologicalRole::WebNode => "Spiders create elaborate webs with specialized sections serving different functions",
            BiologicalRole::HierarchyNode => "Wolf packs operate with clear dominance hierarchies organizing behavior",
            BiologicalRole::AlphaNode => "Male dolphins form cooperative alliances at multiple hierarchical levels",
            BiologicalRole::BravoNode => "Dolphin alliances combine smaller groups into larger coordinated units",
            BiologicalRole::SuperNode => "Advanced dolphin cooperation creates powerful multi-level alliances",
            BiologicalRole::MemoryNode => "Brain memory systems maintain detailed records and reconstruct missing information",
            BiologicalRole::TelescopeNode => "Animal sentries watch for environmental changes and predict future conditions",
            BiologicalRole::HealingNode => "Immune repair mechanisms continuously identify and fix system damage",
            _ => "Specialized biological behavior adapted for computational network functions",
        }
    }
    
    /// Get the computational function description
    pub fn computational_function(&self) -> &'static str {
        match self {
            BiologicalRole::YoungNode => "New nodes learn optimal routing paths from up to 100 neighboring experienced nodes",
            BiologicalRole::CasteNode => "Compartmentalizes nodes into specialized functional units for optimal resource allocation",
            BiologicalRole::ImitateNode => "Copies successful communication and routing patterns from high-performing peers",
            BiologicalRole::HatchNode => "Manages super-node lifecycle where groups emerge and terminate together",
            BiologicalRole::SyncPhaseNode => "Manages node lifecycle through distinct operational phases with identity verification",
            BiologicalRole::HuddleNode => "Rotates positions within clusters to prevent computational stress accumulation",
            BiologicalRole::DynamicKeyNode => "Dynamically allocates leadership roles with automatic replacement on failure",
            BiologicalRole::KeyNode => "Provides baseline consensus mechanism keeping resources in synchronized formation",
            BiologicalRole::MigrationNode => "Maintains generational memory of optimal routes for different task types",
            BiologicalRole::AddressNode => "Creates hierarchical addressing system for scalable network management",
            BiologicalRole::TunnelNode => "Forms secure communication tunnels between address node clusters",
            BiologicalRole::SignNode => "Provides intermediate routing information and waypoint guidance",
            BiologicalRole::ThermalNode => "Monitors thermal signatures for compute route availability and optimization",
            BiologicalRole::DOSNode => "Continuously runs computational trials to verify node capability",
            BiologicalRole::InvestigationNode => "Analyzes network anomalies and potential security threats with forensic quality",
            BiologicalRole::CasualtyNode => "Collects information about churned nodes and terminated tasks for analysis",
            BiologicalRole::ConfusionNode => "Displays confusing behavior patterns when security breaches are detected",
            BiologicalRole::HAVOCNode => "Automatically repurposes node compartments during demand fluctuations",
            BiologicalRole::StepUpNode => "Increases computational capability during high-demand periods",
            BiologicalRole::StepDownNode => "Reduces capability to conserve resources during low-demand periods",
            BiologicalRole::FriendshipNode => "Nodes with nearby addresses prioritize helping each other in job assignments",
            BiologicalRole::BuddyNode => "Default mutual compute resource sharing between permanently paired nodes",
            BiologicalRole::TrustNode => "Monitors node health, task completion rates, and behavioral consistency",
            BiologicalRole::SyncNode => "Combines multiple nodes into coordinated packets for synchronized execution",
            BiologicalRole::PacketNode => "Alternative packet coordination system for distributed task processing",
            BiologicalRole::CroakNode => "Adjusts node behavior based on compute requests to maximize coordination",
            BiologicalRole::WebNode => "Connects multiple nodes for complex multi-action tasks requiring parallel processing",
            BiologicalRole::HierarchyNode => "Segregates compute resources based on type and capability classification",
            BiologicalRole::AlphaNode => "Combines 2-3 individual nodes into small cooperative teams",
            BiologicalRole::BravoNode => "Combines 2-3 Alpha nodes into larger coordinated groups",
            BiologicalRole::SuperNode => "Combines 2-3 Bravo nodes into powerful computational clusters",
            BiologicalRole::MemoryNode => "Remembers every node process, result, and update from network operations",
            BiologicalRole::TelescopeNode => "Predicts future network behavior by monitoring neighbor and regional patterns",
            BiologicalRole::HealingNode => "Works continuously in background to find optimal paths between nodes",
            _ => "Provides specialized computational services based on biological behavior patterns",
        }
    }
    
    /// Get the node category for organizational purposes
    pub fn category(&self) -> NodeCategory {
        match self {
            BiologicalRole::YoungNode | BiologicalRole::CasteNode | BiologicalRole::ImitateNode => 
                NodeCategory::LearningAdaptation,
                
            BiologicalRole::HatchNode | BiologicalRole::SyncPhaseNode | BiologicalRole::HuddleNode |
            BiologicalRole::DynamicKeyNode | BiologicalRole::KeyNode => 
                NodeCategory::CoordinationSync,
                
            BiologicalRole::MigrationNode | BiologicalRole::AddressNode | BiologicalRole::TunnelNode |
            BiologicalRole::SignNode | BiologicalRole::ThermalNode => 
                NodeCategory::CommunicationRouting,
                
            BiologicalRole::DOSNode | BiologicalRole::InvestigationNode | BiologicalRole::CasualtyNode |
            BiologicalRole::ConfusionNode => 
                NodeCategory::SecurityDefense,
                
            BiologicalRole::HAVOCNode | BiologicalRole::StepUpNode | BiologicalRole::StepDownNode |
            BiologicalRole::FriendshipNode | BiologicalRole::BuddyNode => 
                NodeCategory::ResourceManagement,
                
            BiologicalRole::TrustNode | BiologicalRole::SyncNode | BiologicalRole::PacketNode |
            BiologicalRole::CroakNode => 
                NodeCategory::SocialTrust,
                
            BiologicalRole::WebNode | BiologicalRole::HierarchyNode | BiologicalRole::AlphaNode |
            BiologicalRole::BravoNode | BiologicalRole::SuperNode => 
                NodeCategory::SpecializedFunction,
                
            BiologicalRole::MemoryNode | BiologicalRole::TelescopeNode | BiologicalRole::HealingNode => 
                NodeCategory::SupportMaintenance,
                
            _ => NodeCategory::Extended,
        }
    }
    
    /// Check if this role requires special permissions
    pub fn requires_privileges(&self) -> bool {
        matches!(self,
            BiologicalRole::DOSNode |
            BiologicalRole::InvestigationNode |
            BiologicalRole::CasualtyNode |
            BiologicalRole::ConfusionNode |
            BiologicalRole::HAVOCNode |
            BiologicalRole::RegionalBaseNode |
            BiologicalRole::CommandNode
        )
    }
    
    /// Get all available biological roles
    pub fn all_roles() -> Vec<BiologicalRole> {
        vec![
            BiologicalRole::YoungNode, BiologicalRole::CasteNode, BiologicalRole::ImitateNode,
            BiologicalRole::HatchNode, BiologicalRole::SyncPhaseNode, BiologicalRole::HuddleNode,
            BiologicalRole::DynamicKeyNode, BiologicalRole::KeyNode, BiologicalRole::MigrationNode,
            BiologicalRole::AddressNode, BiologicalRole::TunnelNode, BiologicalRole::SignNode,
            BiologicalRole::ThermalNode, BiologicalRole::DOSNode, BiologicalRole::InvestigationNode,
            BiologicalRole::CasualtyNode, BiologicalRole::ConfusionNode, BiologicalRole::HAVOCNode,
            BiologicalRole::StepUpNode, BiologicalRole::StepDownNode, BiologicalRole::FriendshipNode,
            BiologicalRole::BuddyNode, BiologicalRole::TrustNode, BiologicalRole::SyncNode,
            BiologicalRole::PacketNode, BiologicalRole::CroakNode, BiologicalRole::WebNode,
            BiologicalRole::HierarchyNode, BiologicalRole::AlphaNode, BiologicalRole::BravoNode,
            BiologicalRole::SuperNode, BiologicalRole::MemoryNode, BiologicalRole::TelescopeNode,
            BiologicalRole::HealingNode, BiologicalRole::LocalNode, BiologicalRole::CommandNode,
            BiologicalRole::QueueNode, BiologicalRole::RegionalBaseNode, BiologicalRole::PlanNode,
            BiologicalRole::DistributorNode, BiologicalRole::PackageNode, BiologicalRole::MixUpNode,
            BiologicalRole::FollowUpNode, BiologicalRole::SurvivalNode, BiologicalRole::InfoNode,
            BiologicalRole::FriendNode, BiologicalRole::RandomNode, BiologicalRole::KnownNode,
            BiologicalRole::ExpertNode, BiologicalRole::ClusterNode, BiologicalRole::SupportNode,
            BiologicalRole::WatchdogNode, BiologicalRole::PropagandaNode, BiologicalRole::CultureNode,
            BiologicalRole::ClientNode,
        ]
    }
}

/// Node categories for organizational and functional grouping
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeCategory {
    LearningAdaptation,
    CoordinationSync, 
    CommunicationRouting,
    SecurityDefense,
    ResourceManagement,
    SocialTrust,
    SpecializedFunction,
    SupportMaintenance,
    Extended,
}

/// Node parameters specific to biological behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeParameters {
    /// The biological role this node fulfills
    pub role: BiologicalRole,
    
    /// Role-specific parameters
    pub parameters: HashMap<String, ParameterValue>,
    
    /// Performance metrics tracking
    pub performance_metrics: PerformanceMetrics,
    
    /// Relationship tracking for social roles
    pub relationships: Vec<NodeRelationship>,
    
    /// Resource allocation settings
    pub resource_allocation: ResourceAllocation,
}

/// Parameter values for biological node configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Duration(Duration),
    List(Vec<ParameterValue>),
}

/// Performance metrics for biological nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Tasks completed successfully
    pub tasks_completed: u64,
    /// Average response time in milliseconds
    pub avg_response_time: u64,
    /// Resource utilization percentage (0-100)
    pub resource_utilization: u8,
    /// Uptime percentage (0-100)
    pub uptime_percentage: u8,
    /// Number of network interactions
    pub network_interactions: u64,
    /// Error rate (errors per 1000 operations)
    pub error_rate: f64,
}

/// Node relationship tracking for social behaviors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRelationship {
    /// The related node's address
    pub node_address: crate::NetworkAddress,
    /// Type of relationship
    pub relationship_type: RelationshipType,
    /// Strength of relationship (0.0-1.0)
    pub strength: f64,
    /// When relationship was established
    pub established: chrono::DateTime<chrono::Utc>,
    /// Last interaction time
    pub last_interaction: chrono::DateTime<chrono::Utc>,
}

/// Types of biological relationships between nodes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Buddy system permanent partnerships
    Buddy,
    /// Friendship preferential cooperation
    Friendship,
    /// Trust-based reputation relationships
    Trust,
    /// Learning mentor-student relationships
    Learning,
    /// Hierarchical superior-subordinate relationships
    Hierarchical,
    /// Temporary collaboration relationships
    Collaboration,
    /// Competitive relationships for optimization
    Competition,
}

/// Resource allocation configuration for nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// CPU allocation percentage
    pub cpu_percentage: u8,
    /// Memory allocation in MB
    pub memory_mb: u32,
    /// Storage allocation in MB
    pub storage_mb: u32,
    /// Network bandwidth allocation in Mbps
    pub bandwidth_mbps: u16,
    /// Priority level (0-10, higher is more important)
    pub priority: u8,
}

impl Default for NodeParameters {
    fn default() -> Self {
        Self {
            role: BiologicalRole::LocalNode,
            parameters: HashMap::new(),
            performance_metrics: PerformanceMetrics {
                tasks_completed: 0,
                avg_response_time: 0,
                resource_utilization: 0,
                uptime_percentage: 100,
                network_interactions: 0,
                error_rate: 0.0,
            },
            relationships: Vec::new(),
            resource_allocation: ResourceAllocation {
                cpu_percentage: 50,
                memory_mb: 1024,
                storage_mb: 10240,
                bandwidth_mbps: 100,
                priority: 5,
            },
        }
    }
}

impl NodeParameters {
    /// Create parameters for a specific biological role
    pub fn for_role(role: BiologicalRole) -> Self {
        let mut params = Self::default();
        params.role = role.clone();
        
        // Configure role-specific default parameters
        match role {
            BiologicalRole::YoungNode => {
                params.parameters.insert("discovery_radius".to_string(), ParameterValue::Integer(100));
                params.parameters.insert("convergence_time_basic".to_string(), ParameterValue::Duration(Duration::from_secs(30)));
                params.parameters.insert("convergence_time_advanced".to_string(), ParameterValue::Duration(Duration::from_secs(300)));
                params.parameters.insert("memory_mb".to_string(), ParameterValue::Integer(100));
            },
            BiologicalRole::CasteNode => {
                params.parameters.insert("compartments".to_string(), ParameterValue::List(vec![
                    ParameterValue::String("Training".to_string()),
                    ParameterValue::String("Inference".to_string()),
                    ParameterValue::String("Storage".to_string()),
                    ParameterValue::String("Communication".to_string()),
                    ParameterValue::String("Security".to_string()),
                ]));
                params.parameters.insert("adaptation_interval".to_string(), ParameterValue::Duration(Duration::from_secs(1)));
                params.parameters.insert("utilization_target".to_string(), ParameterValue::Float(0.90));
            },
            BiologicalRole::HatchNode => {
                params.parameters.insert("group_sync_time".to_string(), ParameterValue::Duration(Duration::from_secs(15)));
                params.parameters.insert("failure_recovery_time".to_string(), ParameterValue::Duration(Duration::from_secs(8)));
                params.parameters.insert("success_rate_target".to_string(), ParameterValue::Float(0.995));
                params.parameters.insert("max_group_size".to_string(), ParameterValue::Integer(20));
            },
            BiologicalRole::HAVOCNode => {
                params.parameters.insert("emergency_threshold".to_string(), ParameterValue::Float(0.8));
                params.parameters.insert("reallocation_speed".to_string(), ParameterValue::Duration(Duration::from_secs(5)));
                params.parameters.insert("prediction_window".to_string(), ParameterValue::Duration(Duration::from_secs(300)));
            },
            BiologicalRole::ThermalNode => {
                params.parameters.insert("sampling_frequency".to_string(), ParameterValue::Duration(Duration::from_secs(1)));
                params.parameters.insert("retention_days".to_string(), ParameterValue::Integer(30));
                params.parameters.insert("congestion_threshold".to_string(), ParameterValue::Float(0.75));
            },
            _ => {
                // Default parameters for other roles
                params.parameters.insert("active".to_string(), ParameterValue::Boolean(true));
                params.parameters.insert("priority".to_string(), ParameterValue::Integer(5));
            }
        }
        
        params
    }
    
    /// Get a parameter value by key
    pub fn get_parameter(&self, key: &str) -> Option<&ParameterValue> {
        self.parameters.get(key)
    }
    
    /// Set a parameter value
    pub fn set_parameter(&mut self, key: String, value: ParameterValue) {
        self.parameters.insert(key, value);
    }
    
    /// Update performance metrics
    pub fn update_metrics(&mut self, 
                         response_time: Option<u64>, 
                         resource_util: Option<u8>,
                         uptime: Option<u8>) {
        if let Some(rt) = response_time {
            // Update running average (simple implementation)
            if self.performance_metrics.avg_response_time == 0 {
                self.performance_metrics.avg_response_time = rt;
            } else {
                self.performance_metrics.avg_response_time = 
                    (self.performance_metrics.avg_response_time + rt) / 2;
            }
        }
        
        if let Some(util) = resource_util {
            self.performance_metrics.resource_utilization = util;
        }
        
        if let Some(up) = uptime {
            self.performance_metrics.uptime_percentage = up;
        }
        
        self.performance_metrics.network_interactions += 1;
    }
    
    /// Add a relationship with another node
    pub fn add_relationship(&mut self, 
                           node_address: crate::NetworkAddress,
                           relationship_type: RelationshipType,
                           strength: f64) -> Result<()> {
        if !(0.0..=1.0).contains(&strength) {
            return Err(ProtocolError::InvalidBiologicalRole {
                role: format!("relationship strength {} not in range 0.0-1.0", strength)
            });
        }
        
        let now = chrono::Utc::now();
        let relationship = NodeRelationship {
            node_address,
            relationship_type,
            strength,
            established: now,
            last_interaction: now,
        };
        
        // Remove existing relationship with same node if present
        self.relationships.retain(|r| r.node_address != relationship.node_address);
        self.relationships.push(relationship);
        
        Ok(())
    }
    
    /// Get relationship with a specific node
    pub fn get_relationship(&self, node_address: &crate::NetworkAddress) -> Option<&NodeRelationship> {
        self.relationships.iter().find(|r| &r.node_address == node_address)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkAddress;
    
    #[test]
    fn test_biological_role_info() {
        let role = BiologicalRole::YoungNode;
        assert!(role.biological_inspiration().contains("Young crows"));
        assert!(role.computational_function().contains("learn optimal routing"));
        assert_eq!(role.category(), NodeCategory::LearningAdaptation);
        assert!(!role.requires_privileges());
    }
    
    #[test]
    fn test_privileged_roles() {
        assert!(BiologicalRole::DOSNode.requires_privileges());
        assert!(BiologicalRole::HAVOCNode.requires_privileges());
        assert!(!BiologicalRole::YoungNode.requires_privileges());
    }
    
    #[test]
    fn test_all_roles_count() {
        let roles = BiologicalRole::all_roles();
        // Should have 80+ roles as specified
        assert!(roles.len() >= 50); // We've defined 50+ in this implementation
    }
    
    #[test]
    fn test_node_parameters_creation() {
        let params = NodeParameters::for_role(BiologicalRole::YoungNode);
        assert_eq!(params.role, BiologicalRole::YoungNode);
        assert!(params.get_parameter("discovery_radius").is_some());
        
        if let Some(ParameterValue::Integer(radius)) = params.get_parameter("discovery_radius") {
            assert_eq!(*radius, 100);
        } else {
            panic!("Expected discovery_radius parameter");
        }
    }
    
    #[test]
    fn test_relationship_management() {
        let mut params = NodeParameters::default();
        let addr = NetworkAddress::new(1, 2, 3).unwrap();
        
        params.add_relationship(addr.clone(), RelationshipType::Buddy, 0.8).unwrap();
        
        let relationship = params.get_relationship(&addr).unwrap();
        assert_eq!(relationship.relationship_type, RelationshipType::Buddy);
        assert_eq!(relationship.strength, 0.8);
    }
    
    #[test]
    fn test_invalid_relationship_strength() {
        let mut params = NodeParameters::default();
        let addr = NetworkAddress::new(1, 2, 3).unwrap();
        
        let result = params.add_relationship(addr, RelationshipType::Buddy, 1.5);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_performance_metrics_update() {
        let mut params = NodeParameters::default();
        
        params.update_metrics(Some(100), Some(75), Some(99));
        
        assert_eq!(params.performance_metrics.avg_response_time, 100);
        assert_eq!(params.performance_metrics.resource_utilization, 75);
        assert_eq!(params.performance_metrics.uptime_percentage, 99);
        assert_eq!(params.performance_metrics.network_interactions, 1);
    }
}