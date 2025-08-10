use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use crate::{NetworkAddress, BiologicalRole, PackageState, ProtocolError, Result};

/// Comprehensive node message types for biological P2P network communication
/// Supports all biological behaviors and network coordination patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMessage {
    /// Unique message identifier
    pub id: Uuid,
    
    /// Message creation timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Source node address
    pub source: NetworkAddress,
    
    /// Destination address (None for broadcast)
    pub destination: Option<NetworkAddress>,
    
    /// Biological role of the sender
    pub sender_role: BiologicalRole,
    
    /// Message type and payload
    pub message_type: MessageType,
    
    /// Message priority (0-10, higher is more urgent)
    pub priority: u8,
    
    /// Time-to-live for message propagation
    pub ttl: u8,
    
    /// Routing path taken by this message
    pub routing_path: Vec<NetworkAddress>,
    
    /// Security signature for message authentication
    pub signature: Option<Vec<u8>>,
}

/// All message types supported by the biological network protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    // Learning and Adaptation Messages
    /// Young node learning request - crow culture learning behavior
    YoungNodeLearning {
        discovery_radius: u16,
        learning_targets: Vec<String>,
    },
    
    /// Learning response with routing knowledge
    LearningResponse {
        routing_knowledge: Vec<RouteKnowledge>,
        optimization_tips: Vec<String>,
    },
    
    /// Caste node compartment status - ant colony division of labor
    CasteCompartmentStatus {
        compartments: Vec<CompartmentStatus>,
        utilization_stats: UtilizationStats,
    },
    
    /// Imitate node pattern sharing - parrot vocal learning
    PatternSharing {
        pattern_type: PatternType,
        pattern_data: Vec<u8>,
        success_metrics: SuccessMetrics,
    },
    
    // Coordination and Synchronization Messages
    /// Hatch node synchronization - sea turtle coordination
    HatchSynchronization {
        group_id: Uuid,
        sync_phase: SyncPhase,
        participants: Vec<NetworkAddress>,
    },
    
    /// Sync phase lifecycle management - penguin colony synchrony
    SyncPhaseUpdate {
        current_phase: LifecyclePhase,
        phase_duration: chrono::Duration,
        next_phase_eta: DateTime<Utc>,
    },
    
    /// Huddle rotation coordination - penguin winter huddling
    HuddleRotation {
        cluster_id: Uuid,
        rotation_schedule: Vec<RotationSlot>,
        stress_metrics: StressMetrics,
    },
    
    /// Dynamic key leadership management - goat leadership
    DynamicKeyLeadership {
        leader_election: LeaderElection,
        leadership_handoff: Option<LeadershipHandoff>,
    },
    
    // Communication and Routing Messages
    /// Migration route sharing - caribou routes
    MigrationRoute {
        route_id: Uuid,
        route_segments: Vec<RouteSegment>,
        historical_performance: PerformanceHistory,
    },
    
    /// Address node hierarchy updates - territorial navigation
    AddressHierarchy {
        region: u16,
        group_updates: Vec<GroupUpdate>,
        topology_changes: Vec<TopologyChange>,
    },
    
    /// Tunnel establishment - secure path marking
    TunnelEstablishment {
        tunnel_id: Uuid,
        tunnel_endpoints: (NetworkAddress, NetworkAddress),
        encryption_params: EncryptionParameters,
    },
    
    /// Sign node waypoint information - path marking
    WaypointInformation {
        waypoint_id: Uuid,
        location: NetworkAddress,
        next_hops: Vec<NextHop>,
        conditions: Vec<RoutingCondition>,
    },
    
    /// Thermal signature reporting - pheromone concentration
    ThermalSignature {
        signature_id: Uuid,
        thermal_data: crate::ThermalSignature,
        congestion_levels: CongestionLevels,
    },
    
    // Security and Defense Messages
    /// DOS detection alert - immune system sentries
    DOSDetection {
        alert_id: Uuid,
        threat_level: ThreatLevel,
        attack_vectors: Vec<AttackVector>,
        mitigation_actions: Vec<String>,
    },
    
    /// Investigation report - pack investigation
    InvestigationReport {
        investigation_id: Uuid,
        anomaly_type: AnomalyType,
        evidence: Vec<Evidence>,
        recommendations: Vec<String>,
    },
    
    /// Casualty analysis - post-incident analysis
    CasualtyAnalysis {
        incident_id: Uuid,
        failed_nodes: Vec<NetworkAddress>,
        failure_patterns: Vec<FailurePattern>,
        prevention_measures: Vec<String>,
    },
    
    /// Confusion tactics - defensive deception
    ConfusionTactics {
        deception_id: Uuid,
        fake_topology: FakeTopologyInfo,
        misdirection_routes: Vec<NetworkAddress>,
    },
    
    // Resource Management Messages
    /// HAVOC emergency response - mosquito-human adaptation
    HAVOCResponse {
        emergency_id: Uuid,
        crisis_level: CrisisLevel,
        resource_reallocation: ResourceReallocation,
        emergency_actions: Vec<EmergencyAction>,
    },
    
    /// Step-up scaling request - desert ant elevation
    StepUpScaling {
        scaling_id: Uuid,
        target_capacity: f64,
        scaling_timeline: chrono::Duration,
        resource_requirements: crate::package::ResourceRequirements,
    },
    
    /// Step-down scaling request - resource conservation
    StepDownScaling {
        scaling_id: Uuid,
        target_capacity: f64,
        conservation_mode: ConservationMode,
    },
    
    /// Thermal monitoring data - pheromone signaling
    ThermalMonitoring {
        monitoring_id: Uuid,
        resource_availability: ResourceAvailability,
        performance_metrics: PerformanceMetrics,
        threshold_alerts: Vec<ThresholdAlert>,
    },
    
    // Social and Trust Messages
    /// Friendship cooperation - tick-host symbiotic networks
    FriendshipCooperation {
        cooperation_id: Uuid,
        assistance_request: AssistanceRequest,
        reciprocity_tracking: ReciprocityTracking,
    },
    
    /// Buddy system coordination - primate grooming networks
    BuddySystemCoordination {
        buddy_pair: (NetworkAddress, NetworkAddress),
        coordination_type: BuddyCoordinationType,
        sync_data: BuddySyncData,
    },
    
    /// Trust evaluation - primate social bonding
    TrustEvaluation {
        evaluation_id: Uuid,
        target_node: NetworkAddress,
        trust_metrics: TrustMetrics,
        behavioral_assessment: BehavioralAssessment,
    },
    
    // Package Processing Messages
    /// Package lifecycle update
    PackageLifecycleUpdate {
        package_id: Uuid,
        current_state: PackageState,
        processing_node: NetworkAddress,
        progress_percentage: u8,
    },
    
    /// Package routing request
    PackageRoutingRequest {
        package_id: Uuid,
        routing_requirements: RoutingRequirements,
        alternative_paths: Vec<Vec<NetworkAddress>>,
    },
    
    /// Package result delivery
    PackageResultDelivery {
        package_id: Uuid,
        result_data: Vec<u8>,
        processing_metadata: ProcessingMetadata,
        quality_metrics: QualityMetrics,
    },
    
    // Network Management Messages
    /// Heartbeat for liveness detection
    Heartbeat {
        node_status: NodeStatus,
        resource_availability: ResourceAvailability,
        performance_summary: PerformanceSummary,
    },
    
    /// Network topology discovery
    TopologyDiscovery {
        discovery_id: Uuid,
        hop_count: u8,
        discovered_nodes: Vec<DiscoveredNode>,
    },
    
    /// Consensus participation
    ConsensusParticipation {
        consensus_id: Uuid,
        proposal: ConsensusProposal,
        vote: Option<ConsensusVote>,
    },
    
    /// Error reporting and handling
    ErrorReport {
        error_id: Uuid,
        error_type: ErrorType,
        error_details: String,
        recovery_suggestions: Vec<String>,
    },
}

// Supporting types for message payloads

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteKnowledge {
    pub destination_pattern: String,
    pub optimal_path: Vec<NetworkAddress>,
    pub latency_ms: u64,
    pub reliability_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentStatus {
    pub compartment_name: String,
    pub current_utilization: f64,
    pub capacity: f64,
    pub active_tasks: u32,
    pub queue_length: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilizationStats {
    pub overall_utilization: f64,
    pub peak_utilization: f64,
    pub efficiency_score: f64,
    pub energy_consumption: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    Routing,
    Resource,
    Communication,
    Security,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuccessMetrics {
    pub usage_count: u64,
    pub success_rate: f64,
    pub performance_improvement: f64,
    pub adoption_rate: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncPhase {
    Initialization,
    Coordination,
    Execution,
    Completion,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LifecyclePhase {
    Initiation,
    Learning,
    Active,
    Maintenance,
    Retirement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationSlot {
    pub node: NetworkAddress,
    pub role: String,
    pub start_time: DateTime<Utc>,
    pub duration: chrono::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressMetrics {
    pub computational_stress: f64,
    pub network_stress: f64,
    pub thermal_stress: f64,
    pub overall_stress_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElection {
    pub candidate: NetworkAddress,
    pub election_round: u32,
    pub votes_received: u32,
    pub leadership_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadershipHandoff {
    pub current_leader: NetworkAddress,
    pub new_leader: NetworkAddress,
    pub transition_time: DateTime<Utc>,
    pub handoff_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteSegment {
    pub segment_id: u16,
    pub from: NetworkAddress,
    pub to: NetworkAddress,
    pub segment_metrics: SegmentMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentMetrics {
    pub latency_ms: u64,
    pub bandwidth_mbps: u32,
    pub reliability: f64,
    pub congestion_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    pub historical_latencies: Vec<u64>,
    pub usage_frequency: u64,
    pub last_used: DateTime<Utc>,
    pub seasonal_patterns: Vec<SeasonalPattern>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalPattern {
    pub time_pattern: String,
    pub performance_multiplier: f64,
    pub confidence: f64,
}

// Additional supporting types...

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupUpdate {
    pub group_id: u16,
    pub active_nodes: Vec<NetworkAddress>,
    pub group_capacity: f64,
    pub group_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyChange {
    pub change_type: String,
    pub affected_nodes: Vec<NetworkAddress>,
    pub timestamp: DateTime<Utc>,
    pub impact_assessment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionParameters {
    pub algorithm: String,
    pub key_length: u16,
    pub initialization_vector: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NextHop {
    pub destination_pattern: String,
    pub next_node: NetworkAddress,
    pub hop_count: u8,
    pub route_quality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingCondition {
    pub condition_type: String,
    pub threshold_value: f64,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CongestionLevels {
    pub cpu_congestion: f64,
    pub memory_congestion: f64,
    pub network_congestion: f64,
    pub overall_congestion: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackVector {
    pub vector_type: String,
    pub source_indicators: Vec<String>,
    pub severity: f64,
    pub confidence: f64,
}

// Continue with remaining types...

impl NodeMessage {
    /// Create a new node message
    pub fn new(
        source: NetworkAddress,
        destination: Option<NetworkAddress>,
        sender_role: BiologicalRole,
        message_type: MessageType,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            source,
            destination,
            sender_role,
            message_type,
            priority: 5,
            ttl: 64,
            routing_path: vec![source.clone()],
            signature: None,
        }
    }
    
    /// Create a broadcast message
    pub fn broadcast(
        source: NetworkAddress,
        sender_role: BiologicalRole,
        message_type: MessageType,
    ) -> Self {
        Self::new(source, None, sender_role, message_type)
    }
    
    /// Create a direct message to specific destination
    pub fn direct(
        source: NetworkAddress,
        destination: NetworkAddress,
        sender_role: BiologicalRole,
        message_type: MessageType,
    ) -> Self {
        Self::new(source, Some(destination), sender_role, message_type)
    }
    
    /// Add node to routing path
    pub fn add_to_routing_path(&mut self, node: NetworkAddress) {
        if !self.routing_path.contains(&node) {
            self.routing_path.push(node);
        }
    }
    
    /// Check if message has expired (TTL reached 0)
    pub fn is_expired(&self) -> bool {
        self.ttl == 0
    }
    
    /// Decrement TTL for message propagation
    pub fn decrement_ttl(&mut self) -> Result<()> {
        if self.ttl == 0 {
            return Err(ProtocolError::NetworkPartition { 
                partition_size: 100 // Message expired
            });
        }
        self.ttl -= 1;
        Ok(())
    }
    
    /// Set message priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }
    
    /// Set message TTL
    pub fn with_ttl(mut self, ttl: u8) -> Self {
        self.ttl = ttl;
        self
    }
    
    /// Get message size estimate in bytes
    pub fn estimated_size(&self) -> usize {
        // Rough estimate based on serialized size
        std::mem::size_of::<Self>() + 
        self.routing_path.len() * std::mem::size_of::<NetworkAddress>() +
        self.signature.as_ref().map_or(0, |s| s.len())
    }
    
    /// Check if this is a high-priority message
    pub fn is_high_priority(&self) -> bool {
        self.priority >= 8
    }
    
    /// Check if this is an emergency/security message
    pub fn is_emergency(&self) -> bool {
        matches!(self.message_type,
            MessageType::DOSDetection { .. } |
            MessageType::HAVOCResponse { .. } |
            MessageType::CasualtyAnalysis { .. } |
            MessageType::ErrorReport { .. }
        )
    }
    
    /// Get message category for routing and processing
    pub fn category(&self) -> MessageCategory {
        match &self.message_type {
            MessageType::YoungNodeLearning { .. } |
            MessageType::LearningResponse { .. } |
            MessageType::PatternSharing { .. } => MessageCategory::Learning,
            
            MessageType::HatchSynchronization { .. } |
            MessageType::SyncPhaseUpdate { .. } |
            MessageType::HuddleRotation { .. } => MessageCategory::Coordination,
            
            MessageType::MigrationRoute { .. } |
            MessageType::AddressHierarchy { .. } |
            MessageType::TunnelEstablishment { .. } => MessageCategory::Routing,
            
            MessageType::DOSDetection { .. } |
            MessageType::InvestigationReport { .. } |
            MessageType::CasualtyAnalysis { .. } => MessageCategory::Security,
            
            MessageType::HAVOCResponse { .. } |
            MessageType::StepUpScaling { .. } |
            MessageType::StepDownScaling { .. } => MessageCategory::ResourceManagement,
            
            MessageType::FriendshipCooperation { .. } |
            MessageType::BuddySystemCoordination { .. } |
            MessageType::TrustEvaluation { .. } => MessageCategory::Social,
            
            MessageType::PackageLifecycleUpdate { .. } |
            MessageType::PackageRoutingRequest { .. } |
            MessageType::PackageResultDelivery { .. } => MessageCategory::Package,
            
            MessageType::Heartbeat { .. } |
            MessageType::TopologyDiscovery { .. } |
            MessageType::ConsensusParticipation { .. } => MessageCategory::Network,
            
            _ => MessageCategory::General,
        }
    }
}

/// Message categories for routing and processing optimization
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageCategory {
    Learning,
    Coordination,
    Routing,
    Security,
    ResourceManagement,
    Social,
    Package,
    Network,
    General,
}

// Placeholder types for remaining message payload structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyType {
    pub category: String,
    pub severity: f64,
    pub frequency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    pub evidence_type: String,
    pub data: Vec<u8>,
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub pattern_type: String,
    pub affected_nodes: Vec<NetworkAddress>,
    pub correlation_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FakeTopologyInfo {
    pub fake_nodes: Vec<NetworkAddress>,
    pub fake_connections: Vec<(NetworkAddress, NetworkAddress)>,
    pub deception_level: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisLevel {
    Minor,
    Moderate,
    Major,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReallocation {
    pub source_compartments: Vec<String>,
    pub target_compartments: Vec<String>,
    pub reallocation_amount: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAction {
    pub action_type: String,
    pub target_nodes: Vec<NetworkAddress>,
    pub execution_time: DateTime<Utc>,
    pub priority: u8,
}

// Additional placeholder types would continue...

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NetworkAddress, BiologicalRole};
    
    #[test]
    fn test_message_creation() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let destination = NetworkAddress::new(4, 5, 6).unwrap();
        
        let message = NodeMessage::direct(
            source.clone(),
            destination.clone(),
            BiologicalRole::YoungNode,
            MessageType::YoungNodeLearning {
                discovery_radius: 100,
                learning_targets: vec!["routing".to_string()],
            },
        );
        
        assert_eq!(message.source, source);
        assert_eq!(message.destination, Some(destination));
        assert_eq!(message.sender_role, BiologicalRole::YoungNode);
        assert_eq!(message.ttl, 64);
    }
    
    #[test]
    fn test_broadcast_message() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        
        let message = NodeMessage::broadcast(
            source.clone(),
            BiologicalRole::ThermalNode,
            MessageType::ThermalMonitoring {
                monitoring_id: Uuid::new_v4(),
                resource_availability: ResourceAvailability {
                    cpu_available: 0.5,
                    memory_available: 0.7,
                    storage_available: 0.8,
                    bandwidth_available: 0.6,
                },
                performance_metrics: PerformanceMetrics {
                    throughput: 100.0,
                    latency_ms: 50,
                    error_rate: 0.01,
                    uptime_percentage: 99.9,
                },
                threshold_alerts: Vec::new(),
            },
        );
        
        assert_eq!(message.destination, None);
        assert_eq!(message.category(), MessageCategory::ResourceManagement);
    }
    
    #[test]
    fn test_message_ttl() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        
        let mut message = NodeMessage::broadcast(
            source,
            BiologicalRole::LocalNode,
            MessageType::Heartbeat {
                node_status: NodeStatus::Active,
                resource_availability: ResourceAvailability {
                    cpu_available: 0.5,
                    memory_available: 0.5,
                    storage_available: 0.5,
                    bandwidth_available: 0.5,
                },
                performance_summary: PerformanceSummary {
                    tasks_completed: 100,
                    avg_response_time: 200,
                    success_rate: 0.99,
                },
            },
        ).with_ttl(1);
        
        assert_eq!(message.ttl, 1);
        assert!(!message.is_expired());
        
        message.decrement_ttl().unwrap();
        assert_eq!(message.ttl, 0);
        assert!(message.is_expired());
        
        assert!(message.decrement_ttl().is_err());
    }
    
    #[test]
    fn test_message_priority() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        
        let message = NodeMessage::broadcast(
            source,
            BiologicalRole::DOSNode,
            MessageType::DOSDetection {
                alert_id: Uuid::new_v4(),
                threat_level: ThreatLevel::Critical,
                attack_vectors: Vec::new(),
                mitigation_actions: Vec::new(),
            },
        ).with_priority(10);
        
        assert_eq!(message.priority, 10);
        assert!(message.is_high_priority());
        assert!(message.is_emergency());
    }
    
    #[test]
    fn test_routing_path() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let hop1 = NetworkAddress::new(2, 3, 4).unwrap();
        let hop2 = NetworkAddress::new(3, 4, 5).unwrap();
        
        let mut message = NodeMessage::broadcast(
            source.clone(),
            BiologicalRole::LocalNode,
            MessageType::TopologyDiscovery {
                discovery_id: Uuid::new_v4(),
                hop_count: 0,
                discovered_nodes: Vec::new(),
            },
        );
        
        assert_eq!(message.routing_path, vec![source]);
        
        message.add_to_routing_path(hop1.clone());
        message.add_to_routing_path(hop2.clone());
        
        assert_eq!(message.routing_path, vec![source, hop1, hop2]);
        
        // Adding duplicate should not increase path
        message.add_to_routing_path(hop1.clone());
        assert_eq!(message.routing_path.len(), 3);
    }
}

// Additional supporting types for completeness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    pub cpu_available: f64,
    pub memory_available: f64,
    pub storage_available: f64,
    pub bandwidth_available: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency_ms: u64,
    pub error_rate: f64,
    pub uptime_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAlert {
    pub metric_name: String,
    pub threshold_value: f64,
    pub current_value: f64,
    pub alert_level: ThreatLevel,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Idle,
    Busy,
    Maintenance,
    Offline,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    pub tasks_completed: u64,
    pub avg_response_time: u64,
    pub success_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredNode {
    pub address: NetworkAddress,
    pub role: BiologicalRole,
    pub capabilities: Vec<String>,
    pub distance: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProposal {
    pub proposal_id: Uuid,
    pub proposal_type: String,
    pub proposal_data: Vec<u8>,
    pub voting_deadline: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusVote {
    pub voter: NetworkAddress,
    pub vote: bool,
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    NetworkError,
    ProcessingError,
    SecurityError,
    ResourceError,
    ProtocolError,
}

// Additional placeholder types for remaining message structures
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConservationMode {
    Light,
    Moderate,
    Aggressive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistanceRequest {
    pub request_id: Uuid,
    pub assistance_type: String,
    pub urgency: u8,
    pub estimated_duration: chrono::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReciprocityTracking {
    pub assistance_given: u64,
    pub assistance_received: u64,
    pub reciprocity_score: f64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuddyCoordinationType {
    ResourceSharing,
    LoadBalancing,
    Backup,
    Synchronization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuddySyncData {
    pub last_sync: DateTime<Utc>,
    pub sync_status: String,
    pub shared_resources: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustMetrics {
    pub reliability_score: f64,
    pub performance_score: f64,
    pub security_score: f64,
    pub cooperation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralAssessment {
    pub consistency_score: f64,
    pub cooperation_level: f64,
    pub anomaly_flags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingRequirements {
    pub max_hops: u8,
    pub latency_requirement: chrono::Duration,
    pub security_level: crate::package::SecurityLevel,
    pub preferred_paths: Vec<Vec<NetworkAddress>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_node: NetworkAddress,
    pub processing_time: chrono::Duration,
    pub resources_used: ResourceUsage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_hours: f64,
    pub memory_gb_hours: f64,
    pub storage_gb: f64,
    pub network_mb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub accuracy: f64,
    pub completeness: f64,
    pub processing_quality: f64,
    pub result_confidence: f64,
}