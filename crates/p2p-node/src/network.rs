use libp2p::{
    PeerId,
    swarm::NetworkBehaviour,
    gossipsub::{self, IdentTopic, MessageId, ValidationMode},
    kad::{self, store::MemoryStore, Kademlia},
    mdns,
    noise,
    request_response::{self, ProtocolSupport},
    identify,
    ping,
    autonat,
    relay,
    dcutr,
    upnp,
    StreamProtocol,
};
use std::{
    collections::{HashMap, HashSet},
    time::{Duration, SystemTime},
    hash::{Hash, Hasher},
};
use serde::{Deserialize, Serialize};
use core_protocol::{BiologicalRole, NodeMessage, TrustScore, ReputationScore};
use crate::{
    protocols::{BiologicalProtocolMessage, BiologicalProtocolHandler, BIOLOGICAL_PROTOCOL},
    behavior::{BiologicalBehavior, BiologicalAction},
    config::NodeConfig,
    Result, P2PError,
};

/// Main network behavior combining all protocols
#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "NodeEvent")]
pub struct NodeBehaviour {
    /// Gossipsub for pub/sub messaging (pheromone trails)
    pub gossipsub: gossipsub::Behaviour,
    
    /// Kademlia DHT for peer discovery and routing
    pub kademlia: Kademlia<MemoryStore>,
    
    /// mDNS for local network discovery
    pub mdns: mdns::tokio::Behaviour,
    
    /// Request-response for biological protocol
    pub request_response: request_response::Behaviour<BiologicalProtocolMessage>,
    
    /// Identify protocol for peer information exchange
    pub identify: identify::Behaviour,
    
    /// Ping for connection keep-alive
    pub ping: ping::Behaviour,
    
    /// AutoNAT for NAT detection
    pub autonat: autonat::Behaviour,
    
    /// Relay for NAT traversal
    pub relay: relay::Behaviour,
    
    /// DCUtR for direct connection upgrade
    pub dcutr: dcutr::Behaviour,
    
    /// UPnP for automatic port mapping
    pub upnp: upnp::tokio::Behaviour,
}

/// Events emitted by the node behavior
#[derive(Debug)]
pub enum NodeEvent {
    /// Gossipsub event
    Gossipsub(gossipsub::Event),
    
    /// Kademlia event
    Kademlia(kad::Event),
    
    /// mDNS event
    Mdns(mdns::Event),
    
    /// Request-response event
    RequestResponse(request_response::Event<BiologicalProtocolMessage, BiologicalProtocolMessage>),
    
    /// Identify event
    Identify(identify::Event),
    
    /// Ping event
    Ping(ping::Event),
    
    /// AutoNAT event
    AutoNat(autonat::Event),
    
    /// Relay event
    Relay(relay::Event),
    
    /// DCUtR event
    DCutr(dcutr::Event),
    
    /// UPnP event
    Upnp(upnp::Event),
}

/// Peer information and trust management
#[derive(Debug, Clone)]
pub struct PeerInfo {
    /// Peer ID
    pub peer_id: PeerId,
    
    /// Biological role
    pub role: Option<BiologicalRole>,
    
    /// Trust score
    pub trust_score: Option<TrustScore>,
    
    /// Reputation score
    pub reputation_score: Option<ReputationScore>,
    
    /// Connection status
    pub connected: bool,
    
    /// Last seen timestamp
    pub last_seen: SystemTime,
    
    /// Supported protocols
    pub protocols: HashSet<String>,
    
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
    
    /// Connection quality metrics
    pub connection_quality: ConnectionQuality,
}

/// Connection quality metrics
#[derive(Debug, Clone)]
pub struct ConnectionQuality {
    /// Round-trip time
    pub rtt: Option<Duration>,
    
    /// Connection stability (0.0-1.0)
    pub stability: f64,
    
    /// Bandwidth estimate in Mbps
    pub bandwidth_estimate: Option<f64>,
    
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
}

/// Gossipsub message for biological communications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalMessage {
    /// Message type
    pub message_type: String,
    
    /// Source peer
    pub source: PeerId,
    
    /// Target peer (if directed)
    pub target: Option<PeerId>,
    
    /// Biological role of sender
    pub sender_role: BiologicalRole,
    
    /// Message payload
    pub payload: NodeMessage,
    
    /// Timestamp
    pub timestamp: SystemTime,
    
    /// Message priority (1-10)
    pub priority: u8,
    
    /// TTL (Time To Live) in hops
    pub ttl: u8,
}

/// Trust and reputation management
pub struct TrustManager {
    /// Trust scores for peers
    trust_scores: HashMap<PeerId, TrustScore>,
    
    /// Reputation scores for peers
    reputation_scores: HashMap<PeerId, ReputationScore>,
    
    /// Trust evaluation history
    trust_history: HashMap<PeerId, Vec<TrustEvaluation>>,
    
    /// Configuration
    config: TrustConfig,
}

/// Trust evaluation record
#[derive(Debug, Clone)]
pub struct TrustEvaluation {
    /// Evaluation timestamp
    pub timestamp: SystemTime,
    
    /// Evaluation type
    pub evaluation_type: String,
    
    /// Score awarded
    pub score: f64,
    
    /// Evidence or reason
    pub evidence: String,
}

/// Trust manager configuration
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Initial trust score for new peers
    pub initial_trust: f64,
    
    /// Minimum trust for interaction
    pub minimum_trust: f64,
    
    /// Trust decay rate per day
    pub decay_rate: f64,
    
    /// Maximum Byzantine tolerance
    pub max_byzantine_ratio: f64,
    
    /// Trust boost for successful interactions
    pub success_boost: f64,
    
    /// Trust penalty for failures
    pub failure_penalty: f64,
}

impl NodeBehaviour {
    /// Create a new node behavior with the given configuration
    pub fn new(
        local_peer_id: PeerId,
        config: &NodeConfig,
    ) -> Result<Self> {
        // Configure Gossipsub
        let gossipsub_config = gossipsub::ConfigBuilder::default()
            .heartbeat_interval(config.protocols.gossipsub.heartbeat_interval)
            .validation_mode(match config.protocols.gossipsub.validation_mode.as_str() {
                "strict" => ValidationMode::Strict,
                "permissive" => ValidationMode::Permissive,
                "anonymous" => ValidationMode::Anonymous,
                "none" => ValidationMode::None,
                _ => ValidationMode::Permissive,
            })
            .message_id_fn(|message| {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                message.data.hash(&mut hasher);
                MessageId::from(hasher.finish().to_string())
            })
            .build()
            .map_err(|e| P2PError::GossipsubError { reason: e.to_string() })?;
        
        let mut gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(libp2p::identity::Keypair::generate_ed25519()),
            gossipsub_config,
        ).map_err(|e| P2PError::GossipsubError { reason: e.to_string() })?;
        
        // Subscribe to default topics
        for topic_name in &config.protocols.gossipsub.default_topics {
            let topic = IdentTopic::new(topic_name);
            gossipsub.subscribe(&topic)
                .map_err(|e| P2PError::GossipsubError { reason: e.to_string() })?;
        }
        
        // Configure Kademlia
        let store = MemoryStore::new(local_peer_id);
        let mut kademlia = Kademlia::new(local_peer_id, store);
        
        // Set Kademlia mode based on configuration
        kademlia.set_mode(Some(kad::Mode::Server));
        
        // Add bootstrap nodes
        for bootstrap_addr in &config.discovery.bootstrap_nodes {
            if let Ok(addr) = bootstrap_addr.parse() {
                kademlia.add_address(&local_peer_id, addr);
            }
        }
        
        // Configure mDNS
        let mdns = mdns::tokio::Behaviour::new(
            mdns::Config::default(),
            local_peer_id,
        ).map_err(|e| P2PError::MdnsError { reason: e.to_string() })?;
        
        // Configure request-response for biological protocol
        let request_response = request_response::Behaviour::new(
            BiologicalProtocolMessage,
            [(BIOLOGICAL_PROTOCOL.clone(), ProtocolSupport::Full)],
            request_response::Config::default(),
        );
        
        // Configure Identify
        let identify = identify::Behaviour::new(
            identify::Config::new("/bio-p2p/1.0.0".to_string(), libp2p::identity::Keypair::generate_ed25519().public())
                .with_agent_version(format!("bio-p2p-node/{}", env!("CARGO_PKG_VERSION"))),
        );
        
        // Configure Ping
        let ping = ping::Behaviour::new(
            ping::Config::new()
                .with_keep_alive(true)
                .with_interval(config.network.keep_alive_interval),
        );
        
        // Configure AutoNAT
        let autonat = autonat::Behaviour::new(
            local_peer_id,
            autonat::Config::default(),
        );
        
        // Configure Relay
        let relay = relay::Behaviour::new(
            local_peer_id,
            relay::Config::default(),
        );
        
        // Configure DCUtR
        let dcutr = dcutr::Behaviour::new(local_peer_id);
        
        // Configure UPnP
        let upnp = upnp::tokio::Behaviour::default();
        
        Ok(Self {
            gossipsub,
            kademlia,
            mdns,
            request_response,
            identify,
            ping,
            autonat,
            relay,
            dcutr,
            upnp,
        })
    }
    
    /// Publish a biological message via gossipsub
    pub fn publish_biological_message(
        &mut self,
        topic: &str,
        message: BiologicalMessage,
    ) -> Result<MessageId> {
        let topic = IdentTopic::new(topic);
        let data = bincode::serialize(&message)
            .map_err(|e| P2PError::SerializationFailed { reason: e.to_string() })?;
        
        self.gossipsub
            .publish(topic, data)
            .map_err(|e| P2PError::GossipsubError { reason: e.to_string() })
    }
    
    /// Send a direct biological protocol message to a peer
    pub fn send_biological_request(
        &mut self,
        peer: PeerId,
        message: BiologicalProtocolMessage,
    ) -> request_response::RequestId {
        self.request_response.send_request(&peer, message)
    }
    
    /// Subscribe to a gossipsub topic
    pub fn subscribe_topic(&mut self, topic: &str) -> Result<bool> {
        let topic = IdentTopic::new(topic);
        self.gossipsub
            .subscribe(&topic)
            .map_err(|e| P2PError::GossipsubError { reason: e.to_string() })
    }
    
    /// Unsubscribe from a gossipsub topic
    pub fn unsubscribe_topic(&mut self, topic: &str) -> Result<bool> {
        let topic = IdentTopic::new(topic);
        self.gossipsub
            .unsubscribe(&topic)
            .map_err(|e| P2PError::GossipsubError { reason: e.to_string() })
    }
    
    /// Add a peer to Kademlia routing table
    pub fn add_peer_to_dht(&mut self, peer: PeerId, address: libp2p::Multiaddr) {
        self.kademlia.add_address(&peer, address);
    }
    
    /// Start a Kademlia bootstrap process
    pub fn bootstrap_dht(&mut self) -> Result<kad::QueryId> {
        self.kademlia
            .bootstrap()
            .map_err(|e| P2PError::KademliaError { reason: e.to_string() })
    }
    
    /// Get peers in a specific Kademlia bucket
    pub fn get_closest_peers(&mut self, target: PeerId) -> kad::QueryId {
        self.kademlia.get_closest_peers(target)
    }
    
    /// Put a value into the DHT
    pub fn dht_put(&mut self, key: kad::RecordKey, value: Vec<u8>) -> Result<kad::QueryId> {
        let record = kad::Record::new(key, value);
        Ok(self.kademlia.put_record(record, kad::Quorum::One)?)
    }
    
    /// Get a value from the DHT
    pub fn dht_get(&mut self, key: kad::RecordKey) -> kad::QueryId {
        self.kademlia.get_record(key, kad::Quorum::One)
    }
    
    /// Get connected peers from gossipsub
    pub fn get_gossipsub_peers(&self) -> Vec<&PeerId> {
        self.gossipsub.all_peers().collect()
    }
    
    /// Get mesh peers for a topic
    pub fn get_topic_mesh_peers(&self, topic: &str) -> Vec<&PeerId> {
        let topic = IdentTopic::new(topic).hash();
        self.gossipsub.mesh_peers(&topic).collect()
    }
    
    /// Get gossipsub topics
    pub fn get_gossipsub_topics(&self) -> Vec<&gossipsub::TopicHash> {
        self.gossipsub.topics().collect()
    }
}

impl TrustManager {
    /// Create a new trust manager
    pub fn new(config: TrustConfig) -> Self {
        Self {
            trust_scores: HashMap::new(),
            reputation_scores: HashMap::new(),
            trust_history: HashMap::new(),
            config,
        }
    }
    
    /// Get trust score for a peer
    pub fn get_trust_score(&self, peer: &PeerId) -> f64 {
        self.trust_scores
            .get(peer)
            .map(|score| score.overall_trust)
            .unwrap_or(self.config.initial_trust)
    }
    
    /// Get reputation score for a peer
    pub fn get_reputation_score(&self, peer: &PeerId) -> f64 {
        self.reputation_scores
            .get(peer)
            .map(|score| score.overall_reputation)
            .unwrap_or(0.0)
    }
    
    /// Update trust score based on interaction
    pub fn update_trust(
        &mut self,
        peer: PeerId,
        interaction_type: &str,
        success: bool,
        evidence: String,
    ) {
        let current_trust = self.get_trust_score(&peer);
        
        let adjustment = if success {
            self.config.success_boost
        } else {
            -self.config.failure_penalty
        };
        
        let new_trust = (current_trust + adjustment).max(0.0).min(1.0);
        
        // Update trust score
        let trust_score = TrustScore {
            peer_id: peer.to_string(),
            overall_trust: new_trust,
            interaction_count: 1, // Simplified
            success_rate: if success { 1.0 } else { 0.0 },
            last_interaction: SystemTime::now(),
            trust_factors: HashMap::new(),
        };
        
        self.trust_scores.insert(peer, trust_score);
        
        // Record evaluation
        let evaluation = TrustEvaluation {
            timestamp: SystemTime::now(),
            evaluation_type: interaction_type.to_string(),
            score: adjustment,
            evidence,
        };
        
        self.trust_history
            .entry(peer)
            .or_insert_with(Vec::new)
            .push(evaluation);
    }
    
    /// Apply daily trust decay
    pub fn apply_trust_decay(&mut self) {
        for trust_score in self.trust_scores.values_mut() {
            let days_elapsed = SystemTime::now()
                .duration_since(trust_score.last_interaction)
                .unwrap_or(Duration::from_secs(0))
                .as_secs_f64() / (24.0 * 3600.0);
            
            let decay_factor = (-self.config.decay_rate * days_elapsed).exp();
            trust_score.overall_trust *= decay_factor;
            
            // Remove peers with very low trust
            if trust_score.overall_trust < 0.01 {
                trust_score.overall_trust = 0.0;
            }
        }
    }
    
    /// Check if peer is trusted enough for interaction
    pub fn is_trusted(&self, peer: &PeerId) -> bool {
        self.get_trust_score(peer) >= self.config.minimum_trust
    }
    
    /// Get list of trusted peers
    pub fn get_trusted_peers(&self) -> Vec<PeerId> {
        self.trust_scores
            .iter()
            .filter(|(_, score)| score.overall_trust >= self.config.minimum_trust)
            .map(|(peer, _)| *peer)
            .collect()
    }
    
    /// Calculate network Byzantine ratio
    pub fn byzantine_ratio(&self) -> f64 {
        if self.trust_scores.is_empty() {
            return 0.0;
        }
        
        let untrusted_count = self.trust_scores
            .values()
            .filter(|score| score.overall_trust < self.config.minimum_trust)
            .count();
        
        untrusted_count as f64 / self.trust_scores.len() as f64
    }
    
    /// Check if network is secure (Byzantine ratio acceptable)
    pub fn is_network_secure(&self) -> bool {
        self.byzantine_ratio() <= self.config.max_byzantine_ratio
    }
    
    /// Get trust history for a peer
    pub fn get_trust_history(&self, peer: &PeerId) -> Option<&Vec<TrustEvaluation>> {
        self.trust_history.get(peer)
    }
    
    /// Clean old trust evaluations
    pub fn clean_old_evaluations(&mut self, max_age: Duration) {
        let cutoff = SystemTime::now() - max_age;
        
        for evaluations in self.trust_history.values_mut() {
            evaluations.retain(|eval| eval.timestamp >= cutoff);
        }
        
        // Remove empty histories
        self.trust_history.retain(|_, evaluations| !evaluations.is_empty());
    }
}

impl Default for TrustConfig {
    fn default() -> Self {
        Self {
            initial_trust: 0.5,
            minimum_trust: 0.3,
            decay_rate: 0.01, // 1% per day
            max_byzantine_ratio: 0.25,
            success_boost: 0.1,
            failure_penalty: 0.2,
        }
    }
}

impl Default for ConnectionQuality {
    fn default() -> Self {
        Self {
            rtt: None,
            stability: 1.0,
            bandwidth_estimate: None,
            error_rate: 0.0,
        }
    }
}

impl PeerInfo {
    /// Create new peer info
    pub fn new(peer_id: PeerId) -> Self {
        Self {
            peer_id,
            role: None,
            trust_score: None,
            reputation_score: None,
            connected: false,
            last_seen: SystemTime::now(),
            protocols: HashSet::new(),
            performance_metrics: HashMap::new(),
            connection_quality: ConnectionQuality::default(),
        }
    }
    
    /// Update peer role
    pub fn update_role(&mut self, role: BiologicalRole) {
        self.role = Some(role);
        self.last_seen = SystemTime::now();
    }
    
    /// Update trust score
    pub fn update_trust_score(&mut self, trust_score: TrustScore) {
        self.trust_score = Some(trust_score);
        self.last_seen = SystemTime::now();
    }
    
    /// Update connection status
    pub fn set_connected(&mut self, connected: bool) {
        self.connected = connected;
        self.last_seen = SystemTime::now();
    }
    
    /// Add supported protocol
    pub fn add_protocol(&mut self, protocol: String) {
        self.protocols.insert(protocol);
    }
    
    /// Update performance metric
    pub fn update_performance_metric(&mut self, metric: String, value: f64) {
        self.performance_metrics.insert(metric, value);
        self.last_seen = SystemTime::now();
    }
    
    /// Check if peer supports a protocol
    pub fn supports_protocol(&self, protocol: &str) -> bool {
        self.protocols.contains(protocol)
    }
    
    /// Check if peer is stale (not seen recently)
    pub fn is_stale(&self, max_age: Duration) -> bool {
        SystemTime::now()
            .duration_since(self.last_seen)
            .unwrap_or(Duration::from_secs(0)) > max_age
    }
}

/// Helper functions for biological message handling
impl BiologicalMessage {
    /// Create a new biological message
    pub fn new(
        message_type: String,
        source: PeerId,
        sender_role: BiologicalRole,
        payload: NodeMessage,
    ) -> Self {
        Self {
            message_type,
            source,
            target: None,
            sender_role,
            payload,
            timestamp: SystemTime::now(),
            priority: 5, // Default priority
            ttl: 10,     // Default TTL
        }
    }
    
    /// Create a directed biological message
    pub fn new_directed(
        message_type: String,
        source: PeerId,
        target: PeerId,
        sender_role: BiologicalRole,
        payload: NodeMessage,
    ) -> Self {
        let mut msg = Self::new(message_type, source, sender_role, payload);
        msg.target = Some(target);
        msg
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
    
    /// Check if message should be processed (not expired)
    pub fn is_valid(&self) -> bool {
        self.ttl > 0 && 
        SystemTime::now()
            .duration_since(self.timestamp)
            .unwrap_or(Duration::from_secs(0)) < Duration::from_secs(300) // 5 minute max age
    }
    
    /// Decrement TTL
    pub fn decrement_ttl(&mut self) -> bool {
        if self.ttl > 0 {
            self.ttl -= 1;
            self.ttl > 0
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::PeerId;
    use crate::config::NodeConfig;
    
    #[tokio::test]
    async fn test_node_behaviour_creation() {
        let local_peer_id = PeerId::random();
        let config = NodeConfig::for_testing();
        
        let behaviour = NodeBehaviour::new(local_peer_id, &config);
        assert!(behaviour.is_ok());
    }
    
    #[test]
    fn test_trust_manager() {
        let config = TrustConfig::default();
        let mut trust_manager = TrustManager::new(config);
        let peer = PeerId::random();
        
        // Initial trust should be default
        assert_eq!(trust_manager.get_trust_score(&peer), 0.5);
        
        // Update trust with successful interaction
        trust_manager.update_trust(
            peer,
            "message_delivery",
            true,
            "Message delivered successfully".to_string(),
        );
        
        // Trust should have increased
        assert!(trust_manager.get_trust_score(&peer) > 0.5);
        
        // Check if peer is trusted
        assert!(trust_manager.is_trusted(&peer));
    }
    
    #[test]
    fn test_peer_info() {
        let peer_id = PeerId::random();
        let mut peer_info = PeerInfo::new(peer_id);
        
        assert_eq!(peer_info.peer_id, peer_id);
        assert!(!peer_info.connected);
        assert!(peer_info.role.is_none());
        
        peer_info.update_role(BiologicalRole::Young);
        assert_eq!(peer_info.role, Some(BiologicalRole::Young));
        
        peer_info.add_protocol("test_protocol".to_string());
        assert!(peer_info.supports_protocol("test_protocol"));
    }
    
    #[test]
    fn test_biological_message() {
        let source = PeerId::random();
        let target = PeerId::random();
        let payload = NodeMessage::Heartbeat {
            node_id: "test".to_string(),
            timestamp: SystemTime::now(),
            role: BiologicalRole::Young,
        };
        
        let mut message = BiologicalMessage::new_directed(
            "test_message".to_string(),
            source,
            target,
            BiologicalRole::Young,
            payload,
        ).with_priority(8).with_ttl(5);
        
        assert_eq!(message.source, source);
        assert_eq!(message.target, Some(target));
        assert_eq!(message.priority, 8);
        assert_eq!(message.ttl, 5);
        assert!(message.is_valid());
        
        // Test TTL decrement
        assert!(message.decrement_ttl());
        assert_eq!(message.ttl, 4);
    }
    
    #[test]
    fn test_trust_decay() {
        let config = TrustConfig {
            decay_rate: 0.1,
            ..Default::default()
        };
        let mut trust_manager = TrustManager::new(config);
        let peer = PeerId::random();
        
        // Set high trust
        trust_manager.update_trust(
            peer,
            "test",
            true,
            "test evidence".to_string(),
        );
        
        let initial_trust = trust_manager.get_trust_score(&peer);
        
        // Apply decay
        trust_manager.apply_trust_decay();
        
        let decayed_trust = trust_manager.get_trust_score(&peer);
        assert!(decayed_trust <= initial_trust);
    }
}