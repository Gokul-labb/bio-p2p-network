use libp2p::{
    Swarm,
    SwarmBuilder,
    Transport,
    PeerId,
    identity::Keypair,
    core::upgrade,
    transport::{self, MemoryTransport},
    noise,
    yamux,
    tcp,
    quic,
    websocket,
    dns,
};
use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, SystemTime, Instant},
    sync::{Arc, RwLock},
    path::PathBuf,
};
use tokio::{
    sync::{mpsc, oneshot},
    time::{interval, sleep},
    task::JoinHandle,
};
use futures::{StreamExt, stream::SelectAll, select};
use tracing::{info, warn, error, debug, trace};
use serde::{Deserialize, Serialize};
use core_protocol::{BiologicalRole, NodeMessage, NetworkAddress, ThermalSignature};
use crate::{
    config::NodeConfig,
    network::{NodeBehaviour, NodeEvent, PeerInfo, TrustManager, BiologicalMessage},
    protocols::{BiologicalProtocolHandler, BiologicalProtocolMessage, BiologicalEventHandler},
    behavior::{BiologicalBehavior, BiologicalBehaviorFactory, BiologicalAction, RoleParameters},
    Result, P2PError,
};

/// Main P2P node implementation
pub struct Node {
    /// Node configuration
    config: NodeConfig,
    
    /// Local peer ID
    local_peer_id: PeerId,
    
    /// Keypair for cryptographic operations
    keypair: Keypair,
    
    /// libp2p swarm
    swarm: Swarm<NodeBehaviour>,
    
    /// Biological protocol handler
    biological_handler: BiologicalProtocolHandler,
    
    /// Current biological behavior
    current_behavior: Box<dyn BiologicalBehavior>,
    
    /// Trust manager
    trust_manager: TrustManager,
    
    /// Peer information database
    peers: HashMap<PeerId, PeerInfo>,
    
    /// Network address (for biological addressing)
    network_address: NetworkAddress,
    
    /// Message queue for outgoing messages
    outgoing_messages: VecDeque<OutgoingMessage>,
    
    /// Event channels
    event_sender: mpsc::UnboundedSender<NodeInternalEvent>,
    event_receiver: mpsc::UnboundedReceiver<NodeInternalEvent>,
    
    /// Running tasks
    tasks: Vec<JoinHandle<()>>,
    
    /// Node statistics
    stats: NodeStatistics,
    
    /// Shutdown signal
    shutdown_signal: Option<oneshot::Receiver<()>>,
    shutdown_sender: Option<oneshot::Sender<()>>,
}

/// Outgoing message queue entry
#[derive(Debug, Clone)]
pub struct OutgoingMessage {
    /// Target peer (None for broadcast)
    pub target: Option<PeerId>,
    
    /// Message content
    pub message: NodeMessage,
    
    /// Message priority
    pub priority: u8,
    
    /// Send timestamp
    pub timestamp: SystemTime,
    
    /// Retry count
    pub retry_count: u8,
    
    /// Maximum retries
    pub max_retries: u8,
}

/// Internal node events
#[derive(Debug)]
pub enum NodeInternalEvent {
    /// Peer connected
    PeerConnected(PeerId),
    
    /// Peer disconnected
    PeerDisconnected(PeerId),
    
    /// Message received
    MessageReceived {
        from: PeerId,
        message: NodeMessage,
    },
    
    /// Biological role changed
    RoleChanged {
        old_role: BiologicalRole,
        new_role: BiologicalRole,
    },
    
    /// Network topology changed
    TopologyChanged,
    
    /// Emergency signal received
    EmergencySignal {
        from: PeerId,
        signal_type: String,
        severity: f64,
    },
    
    /// Trust score updated
    TrustUpdated {
        peer: PeerId,
        new_trust: f64,
    },
}

/// Node statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatistics {
    /// Start time
    pub start_time: SystemTime,
    
    /// Messages sent
    pub messages_sent: u64,
    
    /// Messages received
    pub messages_received: u64,
    
    /// Connections established
    pub connections_established: u64,
    
    /// Connections dropped
    pub connections_dropped: u64,
    
    /// Current peer count
    pub current_peer_count: usize,
    
    /// Biological role switches
    pub role_switches: u64,
    
    /// Trust evaluations performed
    pub trust_evaluations: u64,
    
    /// Network performance metrics
    pub network_metrics: NetworkMetrics,
}

/// Network performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Average latency
    pub avg_latency_ms: f64,
    
    /// Message success rate
    pub message_success_rate: f64,
    
    /// Connection stability
    pub connection_stability: f64,
    
    /// Network health score
    pub network_health: f64,
}

/// Node builder for easy configuration
pub struct NodeBuilder {
    config: NodeConfig,
    keypair: Option<Keypair>,
    initial_role: BiologicalRole,
    network_address: Option<NetworkAddress>,
}

/// Custom event handler for biological events
struct NodeBiologicalEventHandler {
    event_sender: mpsc::UnboundedSender<NodeInternalEvent>,
}

impl BiologicalEventHandler for NodeBiologicalEventHandler {
    fn on_role_change(&mut self, old_role: BiologicalRole, new_role: BiologicalRole) {
        let _ = self.event_sender.send(NodeInternalEvent::RoleChanged { old_role, new_role });
    }
    
    fn on_swarm_formation(&mut self, _swarm_id: String, _members: Vec<PeerId>) {
        let _ = self.event_sender.send(NodeInternalEvent::TopologyChanged);
    }
    
    fn on_hierarchy_change(&mut self, _position: crate::protocols::HierarchyPosition) {
        let _ = self.event_sender.send(NodeInternalEvent::TopologyChanged);
    }
    
    fn on_emergency_signal(&mut self, signal: crate::protocols::EmergencySignal) {
        let _ = self.event_sender.send(NodeInternalEvent::EmergencySignal {
            from: signal.source,
            signal_type: signal.signal_type,
            severity: signal.severity,
        });
    }
    
    fn on_pheromone_detected(&mut self, _pheromone: crate::protocols::PheromoneMessage) {
        // Handle pheromone detection
    }
}

impl NodeBuilder {
    /// Create a new node builder
    pub fn new() -> Self {
        Self {
            config: NodeConfig::default(),
            keypair: None,
            initial_role: BiologicalRole::Young,
            network_address: None,
        }
    }
    
    /// Set the node configuration
    pub fn with_config(mut self, config: NodeConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Set the keypair
    pub fn with_keypair(mut self, keypair: Keypair) -> Self {
        self.keypair = Some(keypair);
        self
    }
    
    /// Set the initial biological role
    pub fn with_initial_role(mut self, role: BiologicalRole) -> Self {
        self.initial_role = role;
        self
    }
    
    /// Set the network address
    pub fn with_network_address(mut self, address: NetworkAddress) -> Self {
        self.network_address = Some(address);
        self
    }
    
    /// Load configuration from file
    pub fn with_config_file<P: AsRef<std::path::Path>>(mut self, path: P) -> Result<Self> {
        self.config = NodeConfig::from_file(path)?;
        Ok(self)
    }
    
    /// Build the node
    pub async fn build(self) -> Result<Node> {
        // Validate configuration
        self.config.validate()?;
        
        // Generate or load keypair
        let keypair = if let Some(keypair) = self.keypair {
            keypair
        } else if let Some(identity_file) = &self.config.identity.identity_file {
            load_keypair_from_file(identity_file)?
        } else {
            Keypair::generate_ed25519()
        };
        
        let local_peer_id = PeerId::from(keypair.public());
        
        // Create network address if not provided
        let network_address = if let Some(addr) = self.network_address {
            addr
        } else {
            NetworkAddress::new(1, 1, rand::random::<u8>() % 10)?
        };
        
        // Build transport
        let transport = Self::build_transport(&keypair).await?;
        
        // Create swarm
        let behaviour = NodeBehaviour::new(local_peer_id, &self.config)?;
        let mut swarm = SwarmBuilder::with_tokio_executor(transport, behaviour, local_peer_id)
            .build();
        
        // Listen on configured addresses
        for addr in &self.config.network.tcp_addresses {
            let listen_addr = format!("/ip4/{}/tcp/{}", addr.ip(), addr.port())
                .parse()
                .map_err(|e: libp2p::multiaddr::Error| P2PError::NetworkInitializationFailed {
                    reason: e.to_string(),
                })?;
            swarm.listen_on(listen_addr)?;
        }
        
        // Set up event channels
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (shutdown_sender, shutdown_receiver) = oneshot::channel();
        
        // Create biological protocol handler
        let mut biological_handler = BiologicalProtocolHandler::new(self.initial_role.clone());
        biological_handler.add_event_handler(Box::new(NodeBiologicalEventHandler {
            event_sender: event_sender.clone(),
        }));
        
        // Create initial biological behavior
        let current_behavior = BiologicalBehaviorFactory::create_behavior(&self.initial_role)?;
        
        // Create trust manager
        let trust_manager = TrustManager::new(Default::default());
        
        Ok(Node {
            config: self.config,
            local_peer_id,
            keypair,
            swarm,
            biological_handler,
            current_behavior,
            trust_manager,
            peers: HashMap::new(),
            network_address,
            outgoing_messages: VecDeque::new(),
            event_sender,
            event_receiver,
            tasks: Vec::new(),
            stats: NodeStatistics::new(),
            shutdown_signal: Some(shutdown_receiver),
            shutdown_sender: Some(shutdown_sender),
        })
    }
    
    /// Build the libp2p transport stack
    async fn build_transport(keypair: &Keypair) -> Result<transport::Boxed<(PeerId, libp2p::core::muxing::StreamMuxerBox)>> {
        let transport = {
            // TCP transport
            let tcp = tcp::tokio::Transport::new(tcp::Config::default().nodelay(true));
            
            // QUIC transport  
            let quic = quic::tokio::Transport::new(quic::Config::new(keypair));
            
            // WebSocket transport
            let ws_tcp = websocket::WsConfig::new(tcp.clone());
            
            // Combine transports
            tcp.or_transport(quic)
               .or_transport(ws_tcp)
        };
        
        // Add DNS resolution
        let transport = dns::tokio::Transport::system(transport)
            .map_err(|e| P2PError::NetworkInitializationFailed {
                reason: format!("DNS transport failed: {}", e),
            })?;
        
        // Add encryption and multiplexing
        let transport = transport
            .upgrade(upgrade::Version::V1)
            .authenticate(noise::Config::new(keypair).unwrap())
            .multiplex(yamux::Config::default())
            .boxed();
        
        Ok(transport)
    }
}

impl Node {
    /// Create a new node builder
    pub fn builder() -> NodeBuilder {
        NodeBuilder::new()
    }
    
    /// Start the node and run the main event loop
    pub async fn start(mut self) -> Result<()> {
        info!("Starting bio-p2p node with peer ID: {}", self.local_peer_id);
        info!("Network address: {}", self.network_address);
        info!("Initial biological role: {:?}", self.biological_handler.current_role());
        
        // Initialize biological behavior
        self.initialize_biological_behavior().await?;
        
        // Start periodic tasks
        self.start_periodic_tasks();
        
        // Start discovery
        if self.config.discovery.enable_kademlia {
            let _ = self.swarm.behaviour_mut().bootstrap_dht();
        }
        
        // Main event loop
        self.run_event_loop().await
    }
    
    /// Initialize the biological behavior with node parameters
    async fn initialize_biological_behavior(&mut self) -> Result<()> {
        let params = RoleParameters {
            network_address: self.network_address.clone(),
            initial_peers: self.peers.keys().cloned().collect(),
            config: self.biological_handler.current_role().get_config_defaults(),
            resources: crate::behavior::ResourceAllocation {
                cpu: self.config.resources.cpu_allocation,
                memory_mb: self.config.resources.memory_allocation_mb,
                bandwidth_mbps: self.config.resources.bandwidth_allocation_mbps,
                storage_gb: self.config.resources.storage_allocation_gb,
            },
            capabilities: self.config.identity.capabilities.clone(),
        };
        
        self.current_behavior.initialize(params).await?;
        
        info!("Initialized biological behavior: {}", self.current_behavior.role_name());
        Ok(())
    }
    
    /// Start periodic background tasks
    fn start_periodic_tasks(&mut self) {
        // Trust decay task
        let event_sender = self.event_sender.clone();
        let trust_decay_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Hourly
            loop {
                interval.tick().await;
                // Trust decay logic would go here
            }
        });
        self.tasks.push(trust_decay_task);
        
        // Network health monitoring task
        let event_sender = self.event_sender.clone();
        let health_monitor_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            loop {
                interval.tick().await;
                // Network health monitoring logic would go here
            }
        });
        self.tasks.push(health_monitor_task);
        
        // Biological behavior update task
        let event_sender = self.event_sender.clone();
        let behavior_update_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            loop {
                interval.tick().await;
                // Biological behavior updates would be triggered here
            }
        });
        self.tasks.push(behavior_update_task);
    }
    
    /// Main event loop
    async fn run_event_loop(mut self) -> Result<()> {
        let mut shutdown_signal = self.shutdown_signal.take().unwrap();
        
        loop {
            select! {
                // Handle shutdown signal
                _ = &mut shutdown_signal => {
                    info!("Shutdown signal received");
                    break;
                }
                
                // Handle swarm events
                swarm_event = self.swarm.select_next_some() => {
                    if let Err(e) = self.handle_swarm_event(swarm_event).await {
                        error!("Error handling swarm event: {}", e);
                    }
                }
                
                // Handle internal events
                internal_event = self.event_receiver.recv() => {
                    match internal_event {
                        Some(event) => {
                            if let Err(e) = self.handle_internal_event(event).await {
                                error!("Error handling internal event: {}", e);
                            }
                        }
                        None => {
                            warn!("Internal event channel closed");
                            break;
                        }
                    }
                }
                
                // Process outgoing messages
                _ = sleep(Duration::from_millis(100)) => {
                    self.process_outgoing_messages().await;
                }
                
                // Biological behavior updates
                _ = sleep(Duration::from_secs(1)) => {
                    if let Err(e) = self.update_biological_behavior().await {
                        error!("Error updating biological behavior: {}", e);
                    }
                }
            }
        }
        
        self.shutdown().await
    }
    
    /// Handle libp2p swarm events
    async fn handle_swarm_event(&mut self, event: libp2p::swarm::SwarmEvent<NodeEvent>) -> Result<()> {
        match event {
            libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                info!("Listening on: {}", address);
            }
            
            libp2p::swarm::SwarmEvent::ConnectionEstablished { peer_id, .. } => {
                info!("Connected to peer: {}", peer_id);
                self.handle_peer_connected(peer_id).await?;
            }
            
            libp2p::swarm::SwarmEvent::ConnectionClosed { peer_id, cause, .. } => {
                info!("Disconnected from peer: {} (cause: {:?})", peer_id, cause);
                self.handle_peer_disconnected(peer_id).await?;
            }
            
            libp2p::swarm::SwarmEvent::Behaviour(node_event) => {
                self.handle_behavior_event(node_event).await?;
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle behavior-specific events
    async fn handle_behavior_event(&mut self, event: NodeEvent) -> Result<()> {
        match event {
            NodeEvent::Gossipsub(gossipsub_event) => {
                self.handle_gossipsub_event(gossipsub_event).await?;
            }
            
            NodeEvent::RequestResponse(request_response_event) => {
                self.handle_request_response_event(request_response_event).await?;
            }
            
            NodeEvent::Kademlia(kad_event) => {
                self.handle_kademlia_event(kad_event).await?;
            }
            
            NodeEvent::Mdns(mdns_event) => {
                self.handle_mdns_event(mdns_event).await?;
            }
            
            NodeEvent::Identify(identify_event) => {
                self.handle_identify_event(identify_event).await?;
            }
            
            NodeEvent::Ping(ping_event) => {
                self.handle_ping_event(ping_event).await?;
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle Gossipsub events
    async fn handle_gossipsub_event(&mut self, event: libp2p::gossipsub::Event) -> Result<()> {
        match event {
            libp2p::gossipsub::Event::Message { message, .. } => {
                // Deserialize biological message
                if let Ok(bio_message) = bincode::deserialize::<BiologicalMessage>(&message.data) {
                    debug!("Received biological message: {:?}", bio_message.message_type);
                    
                    // Process through biological behavior
                    let responses = self.current_behavior.handle_message(bio_message.payload.clone(), bio_message.source).await?;
                    
                    // Send any response messages
                    for response in responses {
                        self.send_message(Some(bio_message.source), response).await?;
                    }
                    
                    self.stats.messages_received += 1;
                }
            }
            
            libp2p::gossipsub::Event::Subscribed { peer_id, topic } => {
                debug!("Peer {} subscribed to topic: {:?}", peer_id, topic);
            }
            
            libp2p::gossipsub::Event::Unsubscribed { peer_id, topic } => {
                debug!("Peer {} unsubscribed from topic: {:?}", peer_id, topic);
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle request-response events
    async fn handle_request_response_event(
        &mut self,
        event: libp2p::request_response::Event<BiologicalProtocolMessage, BiologicalProtocolMessage>,
    ) -> Result<()> {
        match event {
            libp2p::request_response::Event::Message { peer, message } => {
                match message {
                    libp2p::request_response::Message::Request { request, channel, .. } => {
                        // Handle biological protocol request
                        let responses = self.biological_handler.handle_message(request, peer).await?;
                        
                        // Send first response (if any)
                        if let Some(response) = responses.into_iter().next() {
                            let _ = self.swarm
                                .behaviour_mut()
                                .request_response
                                .send_response(channel, response);
                        }
                    }
                    
                    libp2p::request_response::Message::Response { response, .. } => {
                        // Handle biological protocol response
                        let _ = self.biological_handler.handle_message(response, peer).await?;
                    }
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle Kademlia events
    async fn handle_kademlia_event(&mut self, event: libp2p::kad::Event) -> Result<()> {
        match event {
            libp2p::kad::Event::OutboundQueryProgressed { result, .. } => {
                match result {
                    libp2p::kad::QueryResult::Bootstrap(Ok(_)) => {
                        info!("Kademlia bootstrap completed successfully");
                    }
                    libp2p::kad::QueryResult::GetClosestPeers(Ok(peers)) => {
                        debug!("Found {} closest peers", peers.peers.len());
                        for peer in peers.peers {
                            self.swarm.behaviour_mut().add_peer_to_dht(peer, "/ip4/0.0.0.0/tcp/0".parse().unwrap());
                        }
                    }
                    _ => {}
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle mDNS events
    async fn handle_mdns_event(&mut self, event: libp2p::mdns::Event) -> Result<()> {
        match event {
            libp2p::mdns::Event::Discovered(list) => {
                for (peer_id, multiaddr) in list {
                    debug!("mDNS discovered peer: {} at {}", peer_id, multiaddr);
                    self.swarm.behaviour_mut().add_peer_to_dht(peer_id, multiaddr);
                }
            }
            
            libp2p::mdns::Event::Expired(list) => {
                for (peer_id, _) in list {
                    debug!("mDNS peer expired: {}", peer_id);
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle Identify events
    async fn handle_identify_event(&mut self, event: libp2p::identify::Event) -> Result<()> {
        match event {
            libp2p::identify::Event::Received { peer_id, info } => {
                debug!("Received identify info from {}: {}", peer_id, info.agent_version);
                
                // Update peer info
                if let Some(peer_info) = self.peers.get_mut(&peer_id) {
                    for protocol in &info.protocols {
                        peer_info.add_protocol(protocol.to_string());
                    }
                }
            }
            
            _ => {}
        }
        
        Ok(())
    }
    
    /// Handle Ping events
    async fn handle_ping_event(&mut self, event: libp2p::ping::Event) -> Result<()> {
        match event {
            libp2p::ping::Event { peer, result } => {
                match result {
                    Ok(rtt) => {
                        debug!("Ping to {} successful: {:?}", peer, rtt);
                        if let Some(peer_info) = self.peers.get_mut(&peer) {
                            peer_info.connection_quality.rtt = Some(rtt);
                        }
                    }
                    Err(e) => {
                        warn!("Ping to {} failed: {}", peer, e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Handle internal node events
    async fn handle_internal_event(&mut self, event: NodeInternalEvent) -> Result<()> {
        match event {
            NodeInternalEvent::PeerConnected(peer_id) => {
                // Already handled in handle_peer_connected
            }
            
            NodeInternalEvent::PeerDisconnected(peer_id) => {
                // Already handled in handle_peer_disconnected
            }
            
            NodeInternalEvent::MessageReceived { from, message } => {
                let responses = self.current_behavior.handle_message(message, from).await?;
                for response in responses {
                    self.send_message(Some(from), response).await?;
                }
            }
            
            NodeInternalEvent::RoleChanged { old_role, new_role } => {
                info!("Biological role changed from {:?} to {:?}", old_role, new_role);
                self.stats.role_switches += 1;
                
                // Switch to new behavior
                let new_behavior = BiologicalBehaviorFactory::create_behavior(&new_role)?;
                self.current_behavior = new_behavior;
                self.initialize_biological_behavior().await?;
            }
            
            NodeInternalEvent::TopologyChanged => {
                debug!("Network topology changed");
            }
            
            NodeInternalEvent::EmergencySignal { from, signal_type, severity } => {
                warn!("Emergency signal received from {}: {} (severity: {})", from, signal_type, severity);
                
                // Handle emergency based on biological behavior
                if severity > 0.8 {
                    // High severity - might trigger role change
                    if let Some(new_role) = self.evaluate_emergency_role_change(&signal_type, severity) {
                        self.biological_handler.change_role(new_role);
                    }
                }
            }
            
            NodeInternalEvent::TrustUpdated { peer, new_trust } => {
                debug!("Trust updated for peer {}: {}", peer, new_trust);
            }
        }
        
        Ok(())
    }
    
    /// Handle peer connection
    async fn handle_peer_connected(&mut self, peer_id: PeerId) -> Result<()> {
        // Create or update peer info
        let peer_info = self.peers.entry(peer_id).or_insert_with(|| PeerInfo::new(peer_id));
        peer_info.set_connected(true);
        
        self.stats.connections_established += 1;
        self.stats.current_peer_count = self.peers.len();
        
        // Send internal event
        let _ = self.event_sender.send(NodeInternalEvent::PeerConnected(peer_id));
        
        // Send role announcement
        let announcement = BiologicalProtocolMessage::RoleAnnouncement {
            role: self.biological_handler.current_role().clone(),
            capabilities: self.config.identity.capabilities.clone(),
            resources: crate::protocols::ResourceAdvertisement {
                cpu_available: self.config.resources.cpu_allocation,
                memory_available_mb: self.config.resources.memory_allocation_mb,
                bandwidth_available_mbps: self.config.resources.bandwidth_allocation_mbps,
                storage_available_gb: self.config.resources.storage_allocation_gb,
                specialized_hardware: vec![], // TODO: Detect hardware
                qos_metrics: HashMap::new(),
            },
            timestamp: SystemTime::now(),
        };
        
        self.swarm.behaviour_mut().send_biological_request(peer_id, announcement);
        
        Ok(())
    }
    
    /// Handle peer disconnection
    async fn handle_peer_disconnected(&mut self, peer_id: PeerId) -> Result<()> {
        // Update peer info
        if let Some(peer_info) = self.peers.get_mut(&peer_id) {
            peer_info.set_connected(false);
        }
        
        self.stats.connections_dropped += 1;
        self.stats.current_peer_count = self.peers.values().filter(|p| p.connected).count();
        
        // Send internal event
        let _ = self.event_sender.send(NodeInternalEvent::PeerDisconnected(peer_id));
        
        Ok(())
    }
    
    /// Update biological behavior
    async fn update_biological_behavior(&mut self) -> Result<()> {
        let actions = self.current_behavior.update().await?;
        
        for action in actions {
            self.execute_biological_action(action).await?;
        }
        
        // Check for role switching
        let network_state = self.build_network_state();
        if let Some(new_role) = self.current_behavior.should_switch_role(&network_state) {
            if new_role != *self.biological_handler.current_role() {
                self.biological_handler.change_role(new_role);
            }
        }
        
        Ok(())
    }
    
    /// Execute a biological action
    async fn execute_biological_action(&mut self, action: BiologicalAction) -> Result<()> {
        match action {
            BiologicalAction::SendMessage { to, message } => {
                self.send_message(Some(to), message).await?;
            }
            
            BiologicalAction::Broadcast { message } => {
                self.send_message(None, message).await?;
            }
            
            BiologicalAction::Subscribe { topic } => {
                self.swarm.behaviour_mut().subscribe_topic(&topic)?;
            }
            
            BiologicalAction::Unsubscribe { topic } => {
                self.swarm.behaviour_mut().unsubscribe_topic(&topic)?;
            }
            
            BiologicalAction::ConnectToPeer { peer_id, address } => {
                if let Some(addr) = address {
                    if let Ok(multiaddr) = addr.parse() {
                        self.swarm.behaviour_mut().add_peer_to_dht(peer_id, multiaddr);
                    }
                }
            }
            
            BiologicalAction::DisconnectFromPeer { peer_id } => {
                // libp2p doesn't have direct disconnect - connections close naturally
                debug!("Requested disconnect from peer: {}", peer_id);
            }
            
            BiologicalAction::UpdateResources { allocation } => {
                // Update resource allocation configuration
                debug!("Updating resource allocation: {:?}", allocation);
            }
            
            BiologicalAction::EmitThermalSignature { signature } => {
                self.emit_thermal_signature(signature).await?;
            }
            
            BiologicalAction::FormGroup { peers, group_type } => {
                debug!("Forming group of type '{}' with {} peers", group_type, peers.len());
                // Group formation logic
            }
            
            BiologicalAction::LeaveGroup => {
                debug!("Leaving current group");
                // Group leaving logic
            }
            
            BiologicalAction::SwitchRole { new_role } => {
                self.biological_handler.change_role(new_role);
            }
        }
        
        Ok(())
    }
    
    /// Send a message
    async fn send_message(&mut self, target: Option<PeerId>, message: NodeMessage) -> Result<()> {
        self.outgoing_messages.push_back(OutgoingMessage {
            target,
            message,
            priority: 5,
            timestamp: SystemTime::now(),
            retry_count: 0,
            max_retries: 3,
        });
        
        Ok(())
    }
    
    /// Process outgoing message queue
    async fn process_outgoing_messages(&mut self) {
        let mut processed_messages = Vec::new();
        
        while let Some(msg) = self.outgoing_messages.pop_front() {
            match msg.target {
                Some(peer_id) => {
                    // Direct message
                    let bio_message = BiologicalMessage::new(
                        "direct_message".to_string(),
                        self.local_peer_id,
                        self.biological_handler.current_role().clone(),
                        msg.message,
                    );
                    
                    if let Ok(message_id) = self.swarm.behaviour_mut().publish_biological_message("direct", bio_message) {
                        trace!("Sent direct message to {}: {:?}", peer_id, message_id);
                        self.stats.messages_sent += 1;
                    } else if msg.retry_count < msg.max_retries {
                        // Retry
                        let mut retry_msg = msg;
                        retry_msg.retry_count += 1;
                        processed_messages.push(retry_msg);
                    }
                }
                None => {
                    // Broadcast message
                    let bio_message = BiologicalMessage::new(
                        "broadcast".to_string(),
                        self.local_peer_id,
                        self.biological_handler.current_role().clone(),
                        msg.message,
                    );
                    
                    if let Ok(message_id) = self.swarm.behaviour_mut().publish_biological_message("bio-p2p-general", bio_message) {
                        trace!("Broadcasted message: {:?}", message_id);
                        self.stats.messages_sent += 1;
                    } else if msg.retry_count < msg.max_retries {
                        // Retry
                        let mut retry_msg = msg;
                        retry_msg.retry_count += 1;
                        processed_messages.push(retry_msg);
                    }
                }
            }
        }
        
        // Re-queue retry messages
        for msg in processed_messages {
            self.outgoing_messages.push_back(msg);
        }
    }
    
    /// Emit thermal signature for pheromone-like communication
    async fn emit_thermal_signature(&mut self, signature: ThermalSignature) -> Result<()> {
        let thermal_message = BiologicalProtocolMessage::ThermalBroadcast {
            signature,
            congestion_level: 0.5, // TODO: Calculate actual congestion
            route_quality: 0.8,    // TODO: Calculate actual route quality
        };
        
        // Send to random peers
        let peers: Vec<PeerId> = self.peers.keys().take(5).cloned().collect();
        for peer in peers {
            self.swarm.behaviour_mut().send_biological_request(peer, thermal_message.clone());
        }
        
        Ok(())
    }
    
    /// Build current network state for biological decision making
    fn build_network_state(&self) -> crate::behavior::NetworkState {
        let peers = self.peers.iter().map(|(peer_id, peer_info)| {
            (*peer_id, peer_info.role.clone().unwrap_or(BiologicalRole::Young))
        }).collect();
        
        crate::behavior::NetworkState {
            peers,
            network_size: self.peers.len(),
            network_health: self.calculate_network_health(),
            resource_demand: HashMap::new(), // TODO: Calculate actual demand
            topology_metrics: crate::behavior::TopologyMetrics {
                clustering_coefficient: 0.0,
                average_path_length: 0.0,
                density: 0.0,
                degree_distribution: vec![],
            },
        }
    }
    
    /// Calculate network health score
    fn calculate_network_health(&self) -> f64 {
        if self.peers.is_empty() {
            return 0.0;
        }
        
        let connected_peers = self.peers.values().filter(|p| p.connected).count();
        let connection_ratio = connected_peers as f64 / self.peers.len() as f64;
        
        // Simple health calculation - could be more sophisticated
        connection_ratio * 0.7 + 0.3 // Base health component
    }
    
    /// Evaluate if emergency requires role change
    fn evaluate_emergency_role_change(&self, signal_type: &str, severity: f64) -> Option<BiologicalRole> {
        match signal_type {
            "resource_shortage" if severity > 0.8 => Some(BiologicalRole::HAVOC),
            "network_partition" if severity > 0.7 => Some(BiologicalRole::HAVOC),
            "security_threat" if severity > 0.9 => Some(BiologicalRole::DOS),
            _ => None,
        }
    }
    
    /// Get node statistics
    pub fn stats(&self) -> &NodeStatistics {
        &self.stats
    }
    
    /// Get current biological role
    pub fn current_role(&self) -> &BiologicalRole {
        self.biological_handler.current_role()
    }
    
    /// Get local peer ID
    pub fn peer_id(&self) -> PeerId {
        self.local_peer_id
    }
    
    /// Get network address
    pub fn network_address(&self) -> &NetworkAddress {
        &self.network_address
    }
    
    /// Get connected peers count
    pub fn connected_peers_count(&self) -> usize {
        self.peers.values().filter(|p| p.connected).count()
    }
    
    /// Graceful shutdown
    pub async fn shutdown(mut self) -> Result<()> {
        info!("Shutting down bio-p2p node");
        
        // Cancel all tasks
        for task in self.tasks {
            task.abort();
        }
        
        // Close swarm connections
        // libp2p swarm will handle connection cleanup automatically
        
        info!("Bio-p2p node shutdown complete");
        Ok(())
    }
    
    /// Trigger shutdown from external signal
    pub fn trigger_shutdown(&mut self) {
        if let Some(sender) = self.shutdown_sender.take() {
            let _ = sender.send(());
        }
    }
}

impl NodeStatistics {
    /// Create new statistics
    fn new() -> Self {
        Self {
            start_time: SystemTime::now(),
            messages_sent: 0,
            messages_received: 0,
            connections_established: 0,
            connections_dropped: 0,
            current_peer_count: 0,
            role_switches: 0,
            trust_evaluations: 0,
            network_metrics: NetworkMetrics::default(),
        }
    }
}

impl Default for NetworkMetrics {
    fn default() -> Self {
        Self {
            avg_latency_ms: 0.0,
            message_success_rate: 1.0,
            connection_stability: 1.0,
            network_health: 1.0,
        }
    }
}

// Helper extension trait for BiologicalRole
trait BiologicalRoleExt {
    fn get_config_defaults(&self) -> HashMap<String, f64>;
}

impl BiologicalRoleExt for BiologicalRole {
    fn get_config_defaults(&self) -> HashMap<String, f64> {
        let mut config = HashMap::new();
        
        match self {
            BiologicalRole::Young => {
                config.insert("discovery_radius".to_string(), 100.0);
                config.insert("convergence_time_secs".to_string(), 30.0);
            }
            BiologicalRole::Caste => {
                config.insert("dynamic_sizing".to_string(), 1.0);
                config.insert("utilization_target".to_string(), 0.85);
            }
            _ => {}
        }
        
        config
    }
}

/// Load keypair from file
fn load_keypair_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Keypair> {
    let content = std::fs::read(path)
        .map_err(|e| P2PError::ConfigurationError {
            field: "identity_file".to_string(),
            reason: e.to_string(),
        })?;
    
    // Simple keypair loading - in practice you'd want proper key format handling
    if content.len() >= 32 {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&content[..32]);
        Ok(Keypair::ed25519_from_bytes(seed)?)
    } else {
        Err(P2PError::ConfigurationError {
            field: "identity_file".to_string(),
            reason: "Invalid keypair file format".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_node_builder() {
        let config = NodeConfig::for_testing();
        let result = Node::builder()
            .with_config(config)
            .with_initial_role(BiologicalRole::Young)
            .build()
            .await;
        
        assert!(result.is_ok());
        let node = result.unwrap();
        assert_eq!(*node.current_role(), BiologicalRole::Young);
    }
    
    #[tokio::test]
    async fn test_node_statistics() {
        let stats = NodeStatistics::new();
        assert_eq!(stats.messages_sent, 0);
        assert_eq!(stats.messages_received, 0);
        assert_eq!(stats.connections_established, 0);
    }
    
    #[test]
    fn test_biological_role_config() {
        let role = BiologicalRole::Young;
        let config = role.get_config_defaults();
        
        assert!(config.contains_key("discovery_radius"));
        assert!(config.contains_key("convergence_time_secs"));
    }
    
    #[tokio::test]
    async fn test_keypair_generation() {
        let keypair = Keypair::generate_ed25519();
        let peer_id = PeerId::from(keypair.public());
        
        assert_ne!(peer_id.to_string(), "");
    }
}