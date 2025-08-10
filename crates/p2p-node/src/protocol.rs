use libp2p::{
    PeerId,
    gossipsub::{self, IdentTopic},
    request_response,
    StreamProtocol,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use core_protocol::{BiologicalRole, NodeMessage, TrustScore, ReputationScore, ThermalSignature};
use crate::{Result, P2PError};

/// Custom protocol for biological role negotiation and coordination
pub const BIOLOGICAL_PROTOCOL: StreamProtocol = StreamProtocol::new("/bio-p2p/biological/1.0.0");
pub const PHEROMONE_PROTOCOL: StreamProtocol = StreamProtocol::new("/bio-p2p/pheromone/1.0.0");
pub const HIERARCHY_PROTOCOL: StreamProtocol = StreamProtocol::new("/bio-p2p/hierarchy/1.0.0");

/// Biological protocol message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiologicalProtocolMessage {
    /// Role announcement
    RoleAnnouncement {
        role: BiologicalRole,
        capabilities: Vec<String>,
        resources: ResourceAdvertisement,
        timestamp: SystemTime,
    },
    
    /// Role negotiation request
    RoleNegotiation {
        requested_role: BiologicalRole,
        current_role: BiologicalRole,
        reason: String,
        priority: u8,
    },
    
    /// Role negotiation response
    RoleNegotiationResponse {
        accepted: bool,
        alternative_role: Option<BiologicalRole>,
        reason: String,
    },
    
    /// Swarm formation request
    SwarmFormation {
        formation_type: String,
        target_size: usize,
        coordination_params: HashMap<String, f64>,
    },
    
    /// Swarm coordination message
    SwarmCoordination {
        command: SwarmCommand,
        parameters: HashMap<String, f64>,
        timestamp: SystemTime,
    },
    
    /// Hierarchy establishment
    HierarchyFormation {
        level: String, // Alpha, Bravo, Super
        parent_node: Option<PeerId>,
        child_nodes: Vec<PeerId>,
    },
    
    /// Leadership election
    LeadershipElection {
        candidate: PeerId,
        qualification_score: f64,
        supporters: Vec<PeerId>,
    },
    
    /// Resource sharing request
    ResourceRequest {
        resource_type: String,
        amount: f64,
        duration: Duration,
        compensation: f64,
    },
    
    /// Resource sharing response
    ResourceResponse {
        granted: bool,
        available_amount: f64,
        conditions: Vec<String>,
    },
    
    /// Trust evaluation
    TrustEvaluation {
        evaluated_peer: PeerId,
        trust_score: TrustScore,
        reputation_score: ReputationScore,
        evidence: Vec<TrustEvidence>,
    },
    
    /// Thermal signature broadcast
    ThermalBroadcast {
        signature: ThermalSignature,
        congestion_level: f64,
        route_quality: f64,
    },
    
    /// Emergency signal (HAVOC activation)
    EmergencySignal {
        emergency_type: String,
        severity: f64,
        affected_resources: Vec<String>,
        coordination_needed: bool,
    },
}

/// Resource advertisement for role announcements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAdvertisement {
    /// Available CPU (0.0-1.0)
    pub cpu_available: f64,
    
    /// Available memory in MB
    pub memory_available_mb: u64,
    
    /// Available bandwidth in Mbps
    pub bandwidth_available_mbps: u64,
    
    /// Available storage in GB
    pub storage_available_gb: u64,
    
    /// Specialized hardware
    pub specialized_hardware: Vec<String>,
    
    /// Quality of service metrics
    pub qos_metrics: HashMap<String, f64>,
}

/// Swarm coordination commands
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmCommand {
    /// Move towards target
    MoveTo { target: SwarmTarget },
    
    /// Maintain formation
    MaintainFormation,
    
    /// Avoid obstacle
    Avoid { obstacle: SwarmTarget },
    
    /// Split swarm
    Split { ratio: f64 },
    
    /// Merge with other swarm
    Merge { other_swarm: Vec<PeerId> },
    
    /// Emergency scatter
    Scatter,
    
    /// Synchronize state
    Synchronize { state_hash: String },
}

/// Swarm target for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmTarget {
    /// Network position
    NetworkPosition { region: u16, zone: u16 },
    
    /// Resource pool
    ResourcePool { pool_id: String },
    
    /// Peer node
    PeerNode { peer_id: PeerId },
    
    /// Geographic location (for edge computing)
    GeographicLocation { latitude: f64, longitude: f64 },
}

/// Evidence for trust evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEvidence {
    /// Type of evidence
    pub evidence_type: String,
    
    /// Evidence data
    pub data: serde_json::Value,
    
    /// Timestamp of evidence
    pub timestamp: SystemTime,
    
    /// Reliability of evidence source
    pub reliability: f64,
}

/// Pheromone message for stigmergic communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PheromoneMessage {
    /// Pheromone type (route quality, resource availability, etc.)
    pub pheromone_type: String,
    
    /// Concentration level (0.0-1.0)
    pub concentration: f64,
    
    /// Location/context of pheromone
    pub context: PheromoneContext,
    
    /// Decay rate
    pub decay_rate: f64,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Context for pheromone placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PheromoneContext {
    /// Network route
    Route { source: PeerId, destination: PeerId, path: Vec<PeerId> },
    
    /// Resource pool
    Resource { resource_type: String, location: String },
    
    /// Computational task
    Task { task_type: String, complexity: f64 },
    
    /// Network region
    Region { region_id: String },
}

/// Protocol handler for biological communications
pub struct BiologicalProtocolHandler {
    /// Current node role
    current_role: BiologicalRole,
    
    /// Known peers and their roles
    peer_roles: HashMap<PeerId, PeerRoleInfo>,
    
    /// Swarm memberships
    swarm_memberships: HashMap<String, SwarmMembership>,
    
    /// Hierarchy position
    hierarchy_position: Option<HierarchyPosition>,
    
    /// Trust scores for peers
    trust_scores: HashMap<PeerId, TrustScore>,
    
    /// Pheromone trails
    pheromone_trails: HashMap<String, Vec<PheromoneTrail>>,
    
    /// Event handlers
    event_handlers: Vec<Box<dyn BiologicalEventHandler>>,
}

/// Information about a peer's role
#[derive(Debug, Clone)]
pub struct PeerRoleInfo {
    /// Peer's biological role
    pub role: BiologicalRole,
    
    /// Peer's capabilities
    pub capabilities: Vec<String>,
    
    /// Resource advertisement
    pub resources: ResourceAdvertisement,
    
    /// Last seen timestamp
    pub last_seen: SystemTime,
    
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Swarm membership information
#[derive(Debug, Clone)]
pub struct SwarmMembership {
    /// Swarm identifier
    pub swarm_id: String,
    
    /// Formation type (boids, flocking, etc.)
    pub formation_type: String,
    
    /// Member nodes
    pub members: Vec<PeerId>,
    
    /// Current position in swarm
    pub position: SwarmPosition,
    
    /// Coordination parameters
    pub coordination_params: HashMap<String, f64>,
    
    /// Last coordination update
    pub last_update: SystemTime,
}

/// Position within a swarm
#[derive(Debug, Clone)]
pub struct SwarmPosition {
    /// Role within swarm (leader, follower, scout, etc.)
    pub swarm_role: String,
    
    /// Relative position vector
    pub position: (f64, f64, f64),
    
    /// Velocity vector
    pub velocity: (f64, f64, f64),
    
    /// Neighbors in formation
    pub neighbors: Vec<PeerId>,
}

/// Hierarchy position information
#[derive(Debug, Clone)]
pub struct HierarchyPosition {
    /// Hierarchy level (Individual, Alpha, Bravo, Super, Regional)
    pub level: String,
    
    /// Parent node (if any)
    pub parent: Option<PeerId>,
    
    /// Child nodes
    pub children: Vec<PeerId>,
    
    /// Leadership status
    pub is_leader: bool,
    
    /// Group identifier
    pub group_id: String,
}

/// Pheromone trail information
#[derive(Debug, Clone)]
pub struct PheromoneTrail {
    /// Trail identifier
    pub trail_id: String,
    
    /// Pheromone message
    pub message: PheromoneMessage,
    
    /// Current concentration (decays over time)
    pub current_concentration: f64,
    
    /// Last update time
    pub last_update: SystemTime,
}

/// Event handler for biological protocol events
pub trait BiologicalEventHandler: Send + Sync {
    /// Handle role change event
    fn on_role_change(&mut self, old_role: BiologicalRole, new_role: BiologicalRole);
    
    /// Handle swarm formation event
    fn on_swarm_formation(&mut self, swarm_id: String, members: Vec<PeerId>);
    
    /// Handle hierarchy change event
    fn on_hierarchy_change(&mut self, position: HierarchyPosition);
    
    /// Handle emergency signal
    fn on_emergency_signal(&mut self, signal: EmergencySignal);
    
    /// Handle pheromone detection
    fn on_pheromone_detected(&mut self, pheromone: PheromoneMessage);
}

/// Emergency signal details
#[derive(Debug, Clone)]
pub struct EmergencySignal {
    /// Signal type
    pub signal_type: String,
    
    /// Severity level (0.0-1.0)
    pub severity: f64,
    
    /// Source of emergency
    pub source: PeerId,
    
    /// Affected resources
    pub affected_resources: Vec<String>,
    
    /// Requires coordination
    pub coordination_needed: bool,
    
    /// Timestamp
    pub timestamp: SystemTime,
}

impl BiologicalProtocolHandler {
    /// Create a new biological protocol handler
    pub fn new(initial_role: BiologicalRole) -> Self {
        Self {
            current_role: initial_role,
            peer_roles: HashMap::new(),
            swarm_memberships: HashMap::new(),
            hierarchy_position: None,
            trust_scores: HashMap::new(),
            pheromone_trails: HashMap::new(),
            event_handlers: Vec::new(),
        }
    }
    
    /// Add an event handler
    pub fn add_event_handler(&mut self, handler: Box<dyn BiologicalEventHandler>) {
        self.event_handlers.push(handler);
    }
    
    /// Handle incoming biological protocol message
    pub async fn handle_message(
        &mut self,
        message: BiologicalProtocolMessage,
        from: PeerId,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        match message {
            BiologicalProtocolMessage::RoleAnnouncement { role, capabilities, resources, .. } => {
                self.handle_role_announcement(from, role, capabilities, resources).await
            },
            BiologicalProtocolMessage::RoleNegotiation { requested_role, current_role, reason, priority } => {
                self.handle_role_negotiation(from, requested_role, current_role, reason, priority).await
            },
            BiologicalProtocolMessage::SwarmFormation { formation_type, target_size, coordination_params } => {
                self.handle_swarm_formation(from, formation_type, target_size, coordination_params).await
            },
            BiologicalProtocolMessage::SwarmCoordination { command, parameters, .. } => {
                self.handle_swarm_coordination(from, command, parameters).await
            },
            BiologicalProtocolMessage::HierarchyFormation { level, parent_node, child_nodes } => {
                self.handle_hierarchy_formation(from, level, parent_node, child_nodes).await
            },
            BiologicalProtocolMessage::LeadershipElection { candidate, qualification_score, supporters } => {
                self.handle_leadership_election(from, candidate, qualification_score, supporters).await
            },
            BiologicalProtocolMessage::ResourceRequest { resource_type, amount, duration, compensation } => {
                self.handle_resource_request(from, resource_type, amount, duration, compensation).await
            },
            BiologicalProtocolMessage::TrustEvaluation { evaluated_peer, trust_score, reputation_score, evidence } => {
                self.handle_trust_evaluation(from, evaluated_peer, trust_score, reputation_score, evidence).await
            },
            BiologicalProtocolMessage::ThermalBroadcast { signature, congestion_level, route_quality } => {
                self.handle_thermal_broadcast(from, signature, congestion_level, route_quality).await
            },
            BiologicalProtocolMessage::EmergencySignal { emergency_type, severity, affected_resources, coordination_needed } => {
                self.handle_emergency_signal(from, emergency_type, severity, affected_resources, coordination_needed).await
            },
            _ => Ok(Vec::new()),
        }
    }
    
    /// Handle role announcement from peer
    async fn handle_role_announcement(
        &mut self,
        from: PeerId,
        role: BiologicalRole,
        capabilities: Vec<String>,
        resources: ResourceAdvertisement,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Update peer role information
        self.peer_roles.insert(from, PeerRoleInfo {
            role: role.clone(),
            capabilities,
            resources,
            last_seen: SystemTime::now(),
            performance_metrics: HashMap::new(),
        });
        
        // Check for role compatibility and potential cooperation
        let responses = self.evaluate_role_compatibility(&role, from).await?;
        
        Ok(responses)
    }
    
    /// Handle role negotiation request
    async fn handle_role_negotiation(
        &mut self,
        from: PeerId,
        requested_role: BiologicalRole,
        _current_role: BiologicalRole,
        reason: String,
        _priority: u8,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Evaluate if role change is beneficial for network
        let should_accept = self.should_accept_role_change(&requested_role, from, &reason).await?;
        
        let response = BiologicalProtocolMessage::RoleNegotiationResponse {
            accepted: should_accept,
            alternative_role: if !should_accept {
                self.suggest_alternative_role(from).await?
            } else {
                None
            },
            reason: if should_accept {
                "Role change accepted".to_string()
            } else {
                "Network would not benefit from this role change".to_string()
            },
        };
        
        Ok(vec![response])
    }
    
    /// Handle swarm formation request
    async fn handle_swarm_formation(
        &mut self,
        from: PeerId,
        formation_type: String,
        target_size: usize,
        coordination_params: HashMap<String, f64>,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Evaluate whether to join the swarm
        if self.should_join_swarm(&formation_type, target_size, &coordination_params).await? {
            let swarm_id = format!("swarm_{}_{}", from, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs());
            
            // Create swarm membership
            let membership = SwarmMembership {
                swarm_id: swarm_id.clone(),
                formation_type,
                members: vec![from], // Will be updated as others join
                position: SwarmPosition {
                    swarm_role: "follower".to_string(),
                    position: (0.0, 0.0, 0.0),
                    velocity: (0.0, 0.0, 0.0),
                    neighbors: vec![from],
                },
                coordination_params,
                last_update: SystemTime::now(),
            };
            
            self.swarm_memberships.insert(swarm_id.clone(), membership);
            
            // Notify event handlers
            for handler in &mut self.event_handlers {
                handler.on_swarm_formation(swarm_id.clone(), vec![from]);
            }
        }
        
        Ok(Vec::new())
    }
    
    /// Handle swarm coordination message
    async fn handle_swarm_coordination(
        &mut self,
        from: PeerId,
        command: SwarmCommand,
        parameters: HashMap<String, f64>,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Find relevant swarm membership
        for membership in self.swarm_memberships.values_mut() {
            if membership.members.contains(&from) {
                // Update swarm state based on command
                self.update_swarm_state(membership, &command, &parameters).await?;
                break;
            }
        }
        
        Ok(Vec::new())
    }
    
    /// Handle hierarchy formation
    async fn handle_hierarchy_formation(
        &mut self,
        _from: PeerId,
        level: String,
        parent_node: Option<PeerId>,
        child_nodes: Vec<PeerId>,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Update hierarchy position
        let position = HierarchyPosition {
            level,
            parent: parent_node,
            children: child_nodes,
            is_leader: false, // Will be determined by leadership election
            group_id: format!("group_{}", uuid::Uuid::new_v4()),
        };
        
        self.hierarchy_position = Some(position.clone());
        
        // Notify event handlers
        for handler in &mut self.event_handlers {
            handler.on_hierarchy_change(position.clone());
        }
        
        Ok(Vec::new())
    }
    
    /// Handle leadership election
    async fn handle_leadership_election(
        &mut self,
        _from: PeerId,
        candidate: PeerId,
        qualification_score: f64,
        _supporters: Vec<PeerId>,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Simple leadership election based on qualification score
        // In a real implementation, this would be more sophisticated
        if let Some(ref mut position) = self.hierarchy_position {
            if qualification_score > 0.7 {
                // Accept leadership if candidate is well qualified
                position.is_leader = candidate == libp2p::PeerId::random(); // This should be our own peer ID
            }
        }
        
        Ok(Vec::new())
    }
    
    /// Handle resource request
    async fn handle_resource_request(
        &mut self,
        _from: PeerId,
        resource_type: String,
        amount: f64,
        _duration: Duration,
        _compensation: f64,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Evaluate resource availability
        let available = self.get_available_resource(&resource_type).await?;
        let can_provide = available >= amount;
        
        let response = BiologicalProtocolMessage::ResourceResponse {
            granted: can_provide,
            available_amount: if can_provide { amount } else { available },
            conditions: if can_provide {
                vec!["Standard usage terms".to_string()]
            } else {
                vec!["Insufficient resources available".to_string()]
            },
        };
        
        Ok(vec![response])
    }
    
    /// Handle trust evaluation
    async fn handle_trust_evaluation(
        &mut self,
        _from: PeerId,
        evaluated_peer: PeerId,
        trust_score: TrustScore,
        _reputation_score: ReputationScore,
        _evidence: Vec<TrustEvidence>,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Store trust evaluation
        self.trust_scores.insert(evaluated_peer, trust_score);
        
        Ok(Vec::new())
    }
    
    /// Handle thermal broadcast (pheromone-like signaling)
    async fn handle_thermal_broadcast(
        &mut self,
        from: PeerId,
        signature: ThermalSignature,
        congestion_level: f64,
        route_quality: f64,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Create pheromone trail based on thermal information
        let pheromone = PheromoneMessage {
            pheromone_type: "thermal_route".to_string(),
            concentration: (1.0 - congestion_level) * route_quality,
            context: PheromoneContext::Route {
                source: from,
                destination: from, // Placeholder
                path: vec![from],
            },
            decay_rate: 0.1,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("thermal_signature".to_string(), serde_json::to_value(signature)?);
                meta
            },
            timestamp: SystemTime::now(),
        };
        
        // Add to pheromone trails
        let trail = PheromoneTrail {
            trail_id: uuid::Uuid::new_v4().to_string(),
            message: pheromone.clone(),
            current_concentration: pheromone.concentration,
            last_update: SystemTime::now(),
        };
        
        self.pheromone_trails
            .entry("thermal_routes".to_string())
            .or_insert_with(Vec::new)
            .push(trail);
        
        // Notify event handlers
        for handler in &mut self.event_handlers {
            handler.on_pheromone_detected(pheromone.clone());
        }
        
        Ok(Vec::new())
    }
    
    /// Handle emergency signal
    async fn handle_emergency_signal(
        &mut self,
        from: PeerId,
        emergency_type: String,
        severity: f64,
        affected_resources: Vec<String>,
        coordination_needed: bool,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        let signal = EmergencySignal {
            signal_type: emergency_type,
            severity,
            source: from,
            affected_resources,
            coordination_needed,
            timestamp: SystemTime::now(),
        };
        
        // Notify event handlers
        for handler in &mut self.event_handlers {
            handler.on_emergency_signal(signal.clone());
        }
        
        Ok(Vec::new())
    }
    
    /// Evaluate role compatibility with a peer
    async fn evaluate_role_compatibility(
        &self,
        peer_role: &BiologicalRole,
        _peer_id: PeerId,
    ) -> Result<Vec<BiologicalProtocolMessage>> {
        // Simple compatibility evaluation
        // In practice, this would be more sophisticated
        let compatibility = match (&self.current_role, peer_role) {
            (BiologicalRole::Young, BiologicalRole::Memory) => 0.9,
            (BiologicalRole::Caste, BiologicalRole::HAVOC) => 0.8,
            _ => 0.5,
        };
        
        if compatibility > 0.7 {
            // High compatibility - might suggest cooperation
            Ok(Vec::new()) // Placeholder
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Determine if a role change should be accepted
    async fn should_accept_role_change(
        &self,
        _requested_role: &BiologicalRole,
        _from: PeerId,
        _reason: &str,
    ) -> Result<bool> {
        // Simple acceptance criteria
        // In practice, this would consider network state, load balancing, etc.
        Ok(true)
    }
    
    /// Suggest alternative role for a peer
    async fn suggest_alternative_role(&self, _from: PeerId) -> Result<Option<BiologicalRole>> {
        // Simple suggestion logic
        Ok(Some(BiologicalRole::Young))
    }
    
    /// Determine if should join a swarm
    async fn should_join_swarm(
        &self,
        _formation_type: &str,
        _target_size: usize,
        _params: &HashMap<String, f64>,
    ) -> Result<bool> {
        // Simple join decision
        Ok(self.swarm_memberships.len() < 3) // Limit swarm participation
    }
    
    /// Update swarm state based on coordination command
    async fn update_swarm_state(
        &mut self,
        membership: &mut SwarmMembership,
        command: &SwarmCommand,
        _parameters: &HashMap<String, f64>,
    ) -> Result<()> {
        match command {
            SwarmCommand::MoveTo { target } => {
                // Update position based on target
                match target {
                    SwarmTarget::NetworkPosition { region, zone } => {
                        membership.position.position = (*region as f64, *zone as f64, 0.0);
                    },
                    _ => {}, // Handle other target types
                }
            },
            SwarmCommand::MaintainFormation => {
                // Keep current formation
            },
            SwarmCommand::Scatter => {
                // Emergency scatter - break formation
                membership.position.swarm_role = "scattered".to_string();
            },
            _ => {}, // Handle other commands
        }
        
        membership.last_update = SystemTime::now();
        Ok(())
    }
    
    /// Get available resource amount
    async fn get_available_resource(&self, resource_type: &str) -> Result<f64> {
        // Placeholder resource availability
        match resource_type {
            "cpu" => Ok(0.3),
            "memory" => Ok(0.5),
            "bandwidth" => Ok(0.4),
            "storage" => Ok(0.6),
            _ => Ok(0.0),
        }
    }
    
    /// Get current role
    pub fn current_role(&self) -> &BiologicalRole {
        &self.current_role
    }
    
    /// Change current role
    pub fn change_role(&mut self, new_role: BiologicalRole) {
        let old_role = self.current_role.clone();
        self.current_role = new_role.clone();
        
        // Notify event handlers
        for handler in &mut self.event_handlers {
            handler.on_role_change(old_role.clone(), new_role.clone());
        }
    }
    
    /// Get peer roles
    pub fn peer_roles(&self) -> &HashMap<PeerId, PeerRoleInfo> {
        &self.peer_roles
    }
    
    /// Get swarm memberships
    pub fn swarm_memberships(&self) -> &HashMap<String, SwarmMembership> {
        &self.swarm_memberships
    }
    
    /// Get hierarchy position
    pub fn hierarchy_position(&self) -> &Option<HierarchyPosition> {
        &self.hierarchy_position
    }
    
    /// Update pheromone trails (decay over time)
    pub fn update_pheromone_trails(&mut self) {
        let now = SystemTime::now();
        
        for trails in self.pheromone_trails.values_mut() {
            trails.retain(|trail| {
                let elapsed = now.duration_since(trail.last_update)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs_f64();
                
                let new_concentration = trail.current_concentration * 
                    (-trail.message.decay_rate * elapsed).exp();
                
                new_concentration > 0.01 // Remove very weak trails
            });
            
            // Update concentrations
            for trail in trails.iter_mut() {
                let elapsed = now.duration_since(trail.last_update)
                    .unwrap_or(Duration::from_secs(0))
                    .as_secs_f64();
                
                trail.current_concentration *= (-trail.message.decay_rate * elapsed).exp();
                trail.last_update = now;
            }
        }
    }
}

// Request-response codec implementations
impl request_response::Codec for BiologicalProtocolMessage {
    type Protocol = StreamProtocol;
    type Request = BiologicalProtocolMessage;
    type Response = BiologicalProtocolMessage;
    
    async fn read_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Request>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        
        bincode::deserialize(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    
    async fn read_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
    ) -> std::io::Result<Self::Response>
    where
        T: futures::AsyncRead + Unpin + Send,
    {
        use futures::AsyncReadExt;
        let mut buf = Vec::new();
        io.read_to_end(&mut buf).await?;
        
        bincode::deserialize(&buf)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
    
    async fn write_request<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        req: Self::Request,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let data = bincode::serialize(&req)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        io.write_all(&data).await?;
        io.close().await
    }
    
    async fn write_response<T>(
        &mut self,
        _protocol: &Self::Protocol,
        io: &mut T,
        res: Self::Response,
    ) -> std::io::Result<()>
    where
        T: futures::AsyncWrite + Unpin + Send,
    {
        use futures::AsyncWriteExt;
        let data = bincode::serialize(&res)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        
        io.write_all(&data).await?;
        io.close().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_biological_protocol_handler_creation() {
        let handler = BiologicalProtocolHandler::new(BiologicalRole::Young);
        assert_eq!(*handler.current_role(), BiologicalRole::Young);
        assert!(handler.peer_roles().is_empty());
        assert!(handler.swarm_memberships().is_empty());
    }
    
    #[tokio::test]
    async fn test_role_announcement_handling() {
        let mut handler = BiologicalProtocolHandler::new(BiologicalRole::Young);
        let peer_id = PeerId::random();
        
        let message = BiologicalProtocolMessage::RoleAnnouncement {
            role: BiologicalRole::Caste,
            capabilities: vec!["training".to_string(), "inference".to_string()],
            resources: ResourceAdvertisement {
                cpu_available: 0.8,
                memory_available_mb: 2048,
                bandwidth_available_mbps: 100,
                storage_available_gb: 50,
                specialized_hardware: vec!["GPU".to_string()],
                qos_metrics: HashMap::new(),
            },
            timestamp: SystemTime::now(),
        };
        
        let response = handler.handle_message(message, peer_id).await;
        assert!(response.is_ok());
        
        // Check that peer role was recorded
        assert!(handler.peer_roles().contains_key(&peer_id));
        assert_eq!(handler.peer_roles()[&peer_id].role, BiologicalRole::Caste);
    }
    
    #[tokio::test]
    async fn test_role_change() {
        let mut handler = BiologicalProtocolHandler::new(BiologicalRole::Young);
        
        handler.change_role(BiologicalRole::Caste);
        assert_eq!(*handler.current_role(), BiologicalRole::Caste);
    }
    
    #[tokio::test]
    async fn test_pheromone_trail_decay() {
        let mut handler = BiologicalProtocolHandler::new(BiologicalRole::Young);
        
        // Add a pheromone trail
        let trail = PheromoneTrail {
            trail_id: "test_trail".to_string(),
            message: PheromoneMessage {
                pheromone_type: "test".to_string(),
                concentration: 1.0,
                context: PheromoneContext::Region { region_id: "test_region".to_string() },
                decay_rate: 0.5,
                metadata: HashMap::new(),
                timestamp: SystemTime::now() - Duration::from_secs(2),
            },
            current_concentration: 1.0,
            last_update: SystemTime::now() - Duration::from_secs(2),
        };
        
        handler.pheromone_trails.insert("test".to_string(), vec![trail]);
        
        // Update trails (should decay)
        handler.update_pheromone_trails();
        
        let trails = &handler.pheromone_trails["test"];
        assert!(!trails.is_empty());
        assert!(trails[0].current_concentration < 1.0);
    }
}