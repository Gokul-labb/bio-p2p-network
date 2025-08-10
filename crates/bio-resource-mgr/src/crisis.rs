//! Crisis Detection and Emergency Response System
//! 
//! Implements the HAVOC Node (Mosquito-Human Network Adaptation) for crisis detection,
//! emergency resource reallocation, and network-wide coordination during emergencies.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, RwLock as TokioRwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{ResourceMetrics, CrisisMetrics};

/// Crisis severity levels following biological alarm systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CrisisLevel {
    /// Normal operation - no crisis detected
    Normal = 0,
    /// Minor stress - early warning indicators
    Minor = 1,
    /// Moderate crisis - resource constraints detected
    Moderate = 2,
    /// Major crisis - significant system stress
    Major = 3,
    /// Critical crisis - system failure imminent
    Critical = 4,
    /// Emergency - immediate intervention required
    Emergency = 5,
}

impl CrisisLevel {
    /// Convert to numeric value for calculations
    pub fn to_numeric(&self) -> u8 {
        *self as u8
    }
    
    /// Check if this level requires HAVOC response
    pub fn requires_havoc_response(&self) -> bool {
        *self >= CrisisLevel::Moderate
    }
    
    /// Check if this level requires network-wide coordination
    pub fn requires_network_coordination(&self) -> bool {
        *self >= CrisisLevel::Major
    }
    
    /// Get response urgency multiplier
    pub fn urgency_multiplier(&self) -> f64 {
        match self {
            CrisisLevel::Normal => 0.0,
            CrisisLevel::Minor => 1.1,
            CrisisLevel::Moderate => 1.5,
            CrisisLevel::Major => 2.0,
            CrisisLevel::Critical => 3.0,
            CrisisLevel::Emergency => 5.0,
        }
    }
}

impl std::fmt::Display for CrisisLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrisisLevel::Normal => write!(f, "NORMAL"),
            CrisisLevel::Minor => write!(f, "MINOR"),
            CrisisLevel::Moderate => write!(f, "MODERATE"),
            CrisisLevel::Major => write!(f, "MAJOR"),
            CrisisLevel::Critical => write!(f, "CRITICAL"),
            CrisisLevel::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Crisis event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisEvent {
    /// Unique event identifier
    pub id: Uuid,
    /// Crisis level
    pub level: CrisisLevel,
    /// Event type description
    pub event_type: String,
    /// Detailed description
    pub description: String,
    /// Affected nodes
    pub affected_nodes: Vec<String>,
    /// Timestamp when event occurred
    pub timestamp: DateTime<Utc>,
    /// Event source information
    pub source: CrisisSource,
    /// Metrics snapshot at time of crisis
    pub metrics_snapshot: Option<ResourceMetrics>,
}

/// Source of crisis detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrisisSource {
    /// Resource monitoring detected the crisis
    ResourceMonitor,
    /// Thermal analysis detected the crisis
    ThermalAnalysis,
    /// Node failure cascade detected
    NodeFailure,
    /// Network partition detected
    NetworkPartition,
    /// External alert received
    ExternalAlert,
    /// Prediction model forecast
    PredictiveModel,
}

/// Crisis response action plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisResponse {
    /// Response identifier
    pub id: Uuid,
    /// Related crisis event
    pub crisis_id: Uuid,
    /// Response strategy
    pub strategy: ResponseStrategy,
    /// Target nodes for response actions
    pub target_nodes: Vec<String>,
    /// Resource reallocation instructions
    pub resource_actions: Vec<ResourceAction>,
    /// Expected completion time
    pub estimated_completion: Duration,
    /// Priority level
    pub priority: u8,
}

/// Response strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResponseStrategy {
    /// Redistribute resources across available nodes
    ResourceRedistribution,
    /// Scale up available resources
    ScaleUp,
    /// Scale down to preserve critical functions
    ScaleDown,
    /// Activate backup systems
    ActivateBackups,
    /// Isolate problematic nodes
    Isolate,
    /// Emergency shutdown procedures
    EmergencyShutdown,
}

/// Resource action for crisis response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAction {
    /// Action type
    pub action_type: ActionType,
    /// Source node (for transfers)
    pub source_node: Option<String>,
    /// Target node
    pub target_node: String,
    /// Resource amount to transfer/allocate
    pub amount: f64,
    /// Resource type
    pub resource_type: String,
    /// Action priority
    pub priority: u8,
}

/// Types of resource actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Transfer resources between nodes
    Transfer,
    /// Allocate additional resources
    Allocate,
    /// Deallocate resources
    Deallocate,
    /// Reserve resources for critical functions
    Reserve,
    /// Release reserved resources
    Release,
}

/// HAVOC Node - Crisis Detection and Emergency Response
/// 
/// Named after the rapid behavioral adaptation observed in disease-vector mosquitoes
/// responding to environmental changes. Serves as the network's crisis response coordinator.
pub struct HAVOCNode {
    /// Node identifier
    pub id: String,
    /// Crisis detection configuration
    config: CrisisConfig,
    /// Current crisis state
    crisis_state: Arc<TokioRwLock<CrisisState>>,
    /// Active crisis events
    active_crises: Arc<DashMap<Uuid, CrisisEvent>>,
    /// Crisis response history
    response_history: Arc<RwLock<VecDeque<CrisisResponse>>>,
    /// Metrics for crisis analysis
    metrics: Arc<RwLock<CrisisMetrics>>,
    /// Event broadcast channel
    event_sender: broadcast::Sender<CrisisEvent>,
    /// Response coordination channel
    response_sender: mpsc::Sender<CrisisResponse>,
    /// Network stress monitoring
    stress_monitor: Arc<StressMonitor>,
    /// Predictive crisis detection
    #[cfg(feature = "ml-prediction")]
    predictor: Arc<CrisisPredictor>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Crisis detection configuration
#[derive(Debug, Clone)]
pub struct CrisisConfig {
    /// Threshold for crisis detection (0.0-1.0)
    pub detection_threshold: f64,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Maximum active crises to track
    pub max_active_crises: usize,
    /// Response history size
    pub response_history_size: usize,
    /// Network stress calculation window
    pub stress_window_size: usize,
    /// Enable predictive crisis detection
    pub predictive_detection: bool,
    /// Emergency response timeout
    pub emergency_timeout: Duration,
}

impl Default for CrisisConfig {
    fn default() -> Self {
        Self {
            detection_threshold: 0.8,
            monitoring_interval: Duration::from_secs(5),
            max_active_crises: 100,
            response_history_size: 1000,
            stress_window_size: 60,
            predictive_detection: true,
            emergency_timeout: Duration::from_secs(300),
        }
    }
}

/// Current crisis state
#[derive(Debug, Default)]
struct CrisisState {
    /// Overall network crisis level
    network_level: CrisisLevel,
    /// Last update timestamp
    last_update: Option<Instant>,
    /// Network stress level (0.0-1.0)
    network_stress: f64,
    /// Available resources estimate
    available_resources: f64,
    /// Active response count
    active_responses: usize,
}

/// Network stress monitoring component
struct StressMonitor {
    /// Stress history for analysis
    stress_history: RwLock<VecDeque<StressReading>>,
    /// Window size for calculations
    window_size: usize,
}

/// Individual stress reading
#[derive(Debug, Clone)]
struct StressReading {
    timestamp: Instant,
    cpu_stress: f64,
    memory_stress: f64,
    network_stress: f64,
    node_failures: u32,
    overall_stress: f64,
}

#[cfg(feature = "ml-prediction")]
/// Machine learning based crisis prediction
struct CrisisPredictor {
    // Implementation would use candle-core for ML models
    // Placeholder for now
    prediction_horizon: Duration,
    confidence_threshold: f64,
}

impl HAVOCNode {
    /// Create a new HAVOC node
    pub fn new(id: String, config: CrisisConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let (response_sender, _) = mpsc::channel(1000);
        
        Self {
            id,
            config: config.clone(),
            crisis_state: Arc::new(TokioRwLock::new(CrisisState::default())),
            active_crises: Arc::new(DashMap::new()),
            response_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.response_history_size))),
            metrics: Arc::new(RwLock::new(CrisisMetrics::default())),
            event_sender,
            response_sender,
            stress_monitor: Arc::new(StressMonitor::new(config.stress_window_size)),
            #[cfg(feature = "ml-prediction")]
            predictor: Arc::new(CrisisPredictor::new()),
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start crisis monitoring
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::crisis_error("HAVOC node already running"));
            }
            *running = true;
        }
        
        info!("Starting HAVOC node: {}", self.id);
        
        // Start monitoring task
        self.start_monitoring_task().await;
        
        // Start response coordination task
        self.start_response_task().await;
        
        Ok(())
    }
    
    /// Stop crisis monitoring
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping HAVOC node: {}", self.id);
        Ok(())
    }
    
    /// Report a crisis event
    pub async fn report_crisis(&self, 
        level: CrisisLevel, 
        event_type: String,
        description: String,
        affected_nodes: Vec<String>,
        source: CrisisSource,
        metrics_snapshot: Option<ResourceMetrics>
    ) -> ResourceResult<Uuid> {
        let event = CrisisEvent {
            id: Uuid::new_v4(),
            level,
            event_type,
            description,
            affected_nodes,
            timestamp: Utc::now(),
            source,
            metrics_snapshot,
        };
        
        info!("Crisis reported: {} - {} ({})", event.id, event.event_type, event.level);
        
        // Store active crisis
        self.active_crises.insert(event.id, event.clone());
        
        // Update crisis state
        self.update_crisis_state(&event).await?;
        
        // Broadcast event
        if let Err(e) = self.event_sender.send(event.clone()) {
            warn!("Failed to broadcast crisis event: {}", e);
        }
        
        // Generate response if needed
        if event.level.requires_havoc_response() {
            self.generate_crisis_response(&event).await?;
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_crises += 1;
            metrics.crises_by_level.entry(event.level).and_modify(|e| *e += 1).or_insert(1);
        }
        
        Ok(event.id)
    }
    
    /// Update network stress metrics
    pub async fn update_stress(&self, 
        cpu_stress: f64, 
        memory_stress: f64, 
        network_stress: f64, 
        node_failures: u32
    ) -> ResourceResult<()> {
        let overall_stress = (cpu_stress + memory_stress + network_stress) / 3.0;
        
        let reading = StressReading {
            timestamp: Instant::now(),
            cpu_stress,
            memory_stress,
            network_stress,
            node_failures,
            overall_stress,
        };
        
        self.stress_monitor.add_reading(reading);
        
        // Check for crisis conditions
        if overall_stress > self.config.detection_threshold {
            let crisis_level = self.calculate_crisis_level(overall_stress, node_failures);
            
            if crisis_level > CrisisLevel::Normal {
                self.report_crisis(
                    crisis_level,
                    "High Network Stress".to_string(),
                    format!("Network stress level: {:.2}%, Node failures: {}", overall_stress * 100.0, node_failures),
                    vec![], // Will be populated based on monitoring data
                    CrisisSource::ResourceMonitor,
                    None
                ).await?;
            }
        }
        
        Ok(())
    }
    
    /// Get current crisis level
    pub async fn current_crisis_level(&self) -> CrisisLevel {
        let state = self.crisis_state.read().await;
        state.network_level
    }
    
    /// Get active crises
    pub fn get_active_crises(&self) -> Vec<CrisisEvent> {
        self.active_crises.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Get crisis metrics
    pub fn get_metrics(&self) -> CrisisMetrics {
        self.metrics.read().clone()
    }
    
    /// Subscribe to crisis events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<CrisisEvent> {
        self.event_sender.subscribe()
    }
    
    /// Calculate HAVOC response strength using biological formula
    pub fn calculate_havoc_response(&self, 
        network_stress: f64, 
        available_resources: f64, 
        criticality_factor: f64
    ) -> f64 {
        crate::math::havoc_response_strength(
            self.config.detection_threshold,
            network_stress,
            available_resources,
            criticality_factor
        )
    }
    
    // Private methods
    
    async fn start_monitoring_task(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let stress_monitor = Arc::clone(&self.stress_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Perform periodic crisis detection and cleanup
                // This would include system resource monitoring,
                // cleanup of resolved crises, and predictive analysis
                
                debug!("HAVOC monitoring cycle completed");
            }
        });
    }
    
    async fn start_response_task(&self) {
        let running = Arc::clone(&self.running);
        let mut response_receiver = self.response_sender.subscribe();
        
        tokio::spawn(async move {
            while *running.read() {
                // Handle crisis responses
                // This would coordinate with other nodes
                // to implement crisis response actions
                
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        });
    }
    
    async fn update_crisis_state(&self, event: &CrisisEvent) -> ResourceResult<()> {
        let mut state = self.crisis_state.write().await;
        
        // Update network level to highest active crisis
        if event.level > state.network_level {
            state.network_level = event.level;
        }
        
        state.last_update = Some(Instant::now());
        
        // Update stress metrics
        let current_stress = self.stress_monitor.current_stress_level();
        state.network_stress = current_stress;
        
        Ok(())
    }
    
    async fn generate_crisis_response(&self, event: &CrisisEvent) -> ResourceResult<()> {
        let response_id = Uuid::new_v4();
        
        // Determine appropriate response strategy
        let strategy = match event.level {
            CrisisLevel::Moderate => ResponseStrategy::ResourceRedistribution,
            CrisisLevel::Major => ResponseStrategy::ScaleUp,
            CrisisLevel::Critical => ResponseStrategy::ActivateBackups,
            CrisisLevel::Emergency => ResponseStrategy::EmergencyShutdown,
            _ => return Ok(()),
        };
        
        let response = CrisisResponse {
            id: response_id,
            crisis_id: event.id,
            strategy,
            target_nodes: event.affected_nodes.clone(),
            resource_actions: self.generate_resource_actions(event).await?,
            estimated_completion: Duration::from_secs(60), // Based on crisis level
            priority: event.level.to_numeric(),
        };
        
        // Store response
        {
            let mut history = self.response_history.write();
            history.push_back(response.clone());
            
            // Maintain size limit
            while history.len() > self.config.response_history_size {
                history.pop_front();
            }
        }
        
        // Send response for execution
        if let Err(e) = self.response_sender.send(response).await {
            error!("Failed to send crisis response: {}", e);
        }
        
        Ok(())
    }
    
    async fn generate_resource_actions(&self, event: &CrisisEvent) -> ResourceResult<Vec<ResourceAction>> {
        let mut actions = Vec::new();
        
        // Generate appropriate resource actions based on crisis type and level
        match event.level {
            CrisisLevel::Moderate => {
                // Redistribute resources
                actions.push(ResourceAction {
                    action_type: ActionType::Transfer,
                    source_node: None, // Will be determined by allocator
                    target_node: "emergency_pool".to_string(),
                    amount: 0.2, // 20% additional resources
                    resource_type: "cpu".to_string(),
                    priority: 2,
                });
            },
            CrisisLevel::Major => {
                // Scale up resources
                actions.push(ResourceAction {
                    action_type: ActionType::Allocate,
                    source_node: None,
                    target_node: "network".to_string(),
                    amount: 0.5, // 50% additional resources
                    resource_type: "memory".to_string(),
                    priority: 3,
                });
            },
            CrisisLevel::Critical | CrisisLevel::Emergency => {
                // Emergency resource allocation
                actions.push(ResourceAction {
                    action_type: ActionType::Reserve,
                    source_node: None,
                    target_node: "critical_functions".to_string(),
                    amount: 0.8, // Reserve 80% for critical functions
                    resource_type: "all".to_string(),
                    priority: 5,
                });
            },
            _ => {}
        }
        
        Ok(actions)
    }
    
    fn calculate_crisis_level(&self, stress_level: f64, node_failures: u32) -> CrisisLevel {
        if stress_level >= 0.95 || node_failures >= 10 {
            CrisisLevel::Emergency
        } else if stress_level >= 0.9 || node_failures >= 5 {
            CrisisLevel::Critical
        } else if stress_level >= 0.85 || node_failures >= 3 {
            CrisisLevel::Major
        } else if stress_level >= 0.8 || node_failures >= 1 {
            CrisisLevel::Moderate
        } else if stress_level >= 0.75 {
            CrisisLevel::Minor
        } else {
            CrisisLevel::Normal
        }
    }
}

impl StressMonitor {
    fn new(window_size: usize) -> Self {
        Self {
            stress_history: RwLock::new(VecDeque::with_capacity(window_size)),
            window_size,
        }
    }
    
    fn add_reading(&self, reading: StressReading) {
        let mut history = self.stress_history.write();
        
        history.push_back(reading);
        
        // Maintain window size
        while history.len() > self.window_size {
            history.pop_front();
        }
    }
    
    fn current_stress_level(&self) -> f64 {
        let history = self.stress_history.read();
        
        if history.is_empty() {
            return 0.0;
        }
        
        // Calculate average stress over window
        let sum: f64 = history.iter().map(|r| r.overall_stress).sum();
        sum / history.len() as f64
    }
}

#[cfg(feature = "ml-prediction")]
impl CrisisPredictor {
    fn new() -> Self {
        Self {
            prediction_horizon: Duration::from_minutes(5),
            confidence_threshold: 0.8,
        }
    }
    
    // Placeholder for ML-based crisis prediction
    async fn predict_crisis(&self, _historical_data: &[StressReading]) -> Option<(CrisisLevel, f64)> {
        // Implementation would use ML models to predict upcoming crises
        None
    }
}

/// Crisis detector utility for external integration
pub struct CrisisDetector {
    havoc_nodes: Vec<Arc<HAVOCNode>>,
}

impl CrisisDetector {
    /// Create a new crisis detector with HAVOC nodes
    pub fn new() -> Self {
        Self {
            havoc_nodes: Vec::new(),
        }
    }
    
    /// Add a HAVOC node to the detector
    pub fn add_havoc_node(&mut self, node: Arc<HAVOCNode>) {
        self.havoc_nodes.push(node);
    }
    
    /// Check for network-wide crisis conditions
    pub async fn check_network_crisis(&self) -> ResourceResult<Option<CrisisLevel>> {
        let mut max_level = CrisisLevel::Normal;
        
        for node in &self.havoc_nodes {
            let level = node.current_crisis_level().await;
            if level > max_level {
                max_level = level;
            }
        }
        
        Ok(if max_level > CrisisLevel::Normal {
            Some(max_level)
        } else {
            None
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_havoc_node_creation() {
        let config = CrisisConfig::default();
        let node = HAVOCNode::new("test-havoc".to_string(), config);
        
        assert_eq!(node.id, "test-havoc");
        assert_eq!(node.current_crisis_level().await, CrisisLevel::Normal);
    }
    
    #[tokio::test]
    async fn test_crisis_reporting() {
        let config = CrisisConfig::default();
        let node = HAVOCNode::new("test-havoc".to_string(), config);
        
        let crisis_id = node.report_crisis(
            CrisisLevel::Major,
            "Test Crisis".to_string(),
            "Test crisis description".to_string(),
            vec!["node1".to_string()],
            CrisisSource::ResourceMonitor,
            None
        ).await.unwrap();
        
        assert!(!crisis_id.is_nil());
        
        let active_crises = node.get_active_crises();
        assert_eq!(active_crises.len(), 1);
        assert_eq!(active_crises[0].level, CrisisLevel::Major);
    }
    
    #[tokio::test]
    async fn test_stress_monitoring() {
        let config = CrisisConfig::default();
        let node = HAVOCNode::new("test-havoc".to_string(), config);
        
        node.update_stress(0.9, 0.8, 0.85, 2).await.unwrap();
        
        // High stress should trigger crisis detection
        let active_crises = node.get_active_crises();
        assert!(!active_crises.is_empty());
    }
    
    #[test]
    fn test_crisis_level_properties() {
        assert!(CrisisLevel::Major.requires_havoc_response());
        assert!(!CrisisLevel::Minor.requires_havoc_response());
        
        assert!(CrisisLevel::Critical.requires_network_coordination());
        assert!(!CrisisLevel::Moderate.requires_network_coordination());
        
        assert_eq!(CrisisLevel::Emergency.urgency_multiplier(), 5.0);
    }
    
    #[test]
    fn test_havoc_response_calculation() {
        let config = CrisisConfig::default();
        let node = HAVOCNode::new("test".to_string(), config);
        
        let response = node.calculate_havoc_response(0.9, 100.0, 2.0);
        assert!(response > 0.0);
    }
}