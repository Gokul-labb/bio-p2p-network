//! HAVOC Node Implementation
//! 
//! Implements the HAVOC (Mosquito-Human Network) node for crisis detection and emergency
//! resource reallocation, inspired by rapid behavioral adaptation observed in disease vectors.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{HavocMetrics, CrisisMetrics};
use crate::allocation::{ResourceProvider, AllocationStrategy};
use crate::thermal::ThermalSignature;

/// Crisis severity levels for HAVOC response
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CrisisSeverity {
    /// Low-level stress, monitoring required
    Low = 1,
    /// Moderate stress, proactive measures
    Moderate = 2,
    /// High stress, active intervention
    High = 3,
    /// Critical stress, emergency response
    Critical = 4,
    /// Catastrophic failure, full emergency protocol
    Catastrophic = 5,
}

impl CrisisSeverity {
    /// Get response priority for crisis level
    pub fn response_priority(&self) -> u8 {
        match self {
            CrisisSeverity::Low => 3,
            CrisisSeverity::Moderate => 5,
            CrisisSeverity::High => 7,
            CrisisSeverity::Critical => 9,
            CrisisSeverity::Catastrophic => 10,
        }
    }
    
    /// Get resource reallocation percentage
    pub fn reallocation_percentage(&self) -> f64 {
        match self {
            CrisisSeverity::Low => 0.05,
            CrisisSeverity::Moderate => 0.15,
            CrisisSeverity::High => 0.35,
            CrisisSeverity::Critical => 0.60,
            CrisisSeverity::Catastrophic => 0.80,
        }
    }
    
    /// Get response time requirement
    pub fn response_time(&self) -> Duration {
        match self {
            CrisisSeverity::Low => Duration::from_secs(300),      // 5 minutes
            CrisisSeverity::Moderate => Duration::from_secs(120), // 2 minutes
            CrisisSeverity::High => Duration::from_secs(30),      // 30 seconds
            CrisisSeverity::Critical => Duration::from_secs(10),  // 10 seconds
            CrisisSeverity::Catastrophic => Duration::from_secs(5), // 5 seconds
        }
    }
}

impl std::fmt::Display for CrisisSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrisisSeverity::Low => write!(f, "LOW"),
            CrisisSeverity::Moderate => write!(f, "MODERATE"),
            CrisisSeverity::High => write!(f, "HIGH"),
            CrisisSeverity::Critical => write!(f, "CRITICAL"),
            CrisisSeverity::Catastrophic => write!(f, "CATASTROPHIC"),
        }
    }
}

/// Crisis event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisEvent {
    /// Unique crisis identifier
    pub id: Uuid,
    /// Crisis severity level
    pub severity: CrisisSeverity,
    /// Affected nodes
    pub affected_nodes: Vec<String>,
    /// Crisis type
    pub crisis_type: CrisisType,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Crisis description
    pub description: String,
    /// Predicted impact
    pub predicted_impact: CrisisImpact,
    /// Current status
    pub status: CrisisStatus,
}

/// Types of network crises
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisType {
    /// Resource shortage crisis
    ResourceShortage,
    /// Network partition crisis
    NetworkPartition,
    /// Node failure cascade
    CascadingFailure,
    /// Security breach
    SecurityBreach,
    /// Performance degradation
    PerformanceDegradation,
    /// Overload condition
    SystemOverload,
    /// Unknown crisis type
    Unknown,
}

/// Predicted crisis impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrisisImpact {
    /// Expected duration
    pub expected_duration: Duration,
    /// Estimated affected capacity
    pub affected_capacity_percentage: f64,
    /// Performance impact (0.0-1.0)
    pub performance_impact: f64,
    /// Cascading failure probability
    pub cascading_probability: f64,
    /// Recovery difficulty (1-10)
    pub recovery_difficulty: u8,
}

/// Crisis resolution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisStatus {
    /// Crisis detected, response pending
    Detected,
    /// Response in progress
    Responding,
    /// Crisis contained, monitoring
    Contained,
    /// Crisis resolved
    Resolved,
    /// Crisis escalated to higher severity
    Escalated,
    /// Response failed, crisis ongoing
    Failed,
}

/// Emergency response action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyAction {
    /// Action identifier
    pub id: Uuid,
    /// Related crisis
    pub crisis_id: Uuid,
    /// Action type
    pub action_type: ActionType,
    /// Target nodes for action
    pub target_nodes: Vec<String>,
    /// Resources involved
    pub resources: HashMap<String, f64>,
    /// Action priority (1-10)
    pub priority: u8,
    /// Expected completion time
    pub expected_completion: Duration,
    /// Action status
    pub status: ActionStatus,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Types of emergency actions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    /// Reallocate resources from healthy to stressed nodes
    ResourceReallocation,
    /// Activate backup systems
    BackupActivation,
    /// Load balancing adjustment
    LoadBalancing,
    /// Network traffic rerouting
    TrafficRerouting,
    /// Node isolation for containment
    NodeIsolation,
    /// Emergency scaling
    EmergencyScaling,
    /// Data evacuation
    DataEvacuation,
}

/// Action execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionStatus {
    /// Action is pending execution
    Pending,
    /// Action is currently executing
    Executing,
    /// Action completed successfully
    Completed,
    /// Action failed
    Failed,
    /// Action was cancelled
    Cancelled,
}

/// HAVOC Node - Crisis detection and emergency resource reallocation
/// 
/// Inspired by mosquito-human network adaptation, this node automatically
/// repurposes node compartments based on demand fluctuations and crisis detection.
pub struct HavocNode {
    /// Node identifier
    pub id: String,
    /// HAVOC configuration
    config: HavocConfig,
    /// Crisis detection system
    crisis_detector: Arc<CrisisDetector>,
    /// Emergency response coordinator
    response_coordinator: Arc<EmergencyCoordinator>,
    /// Network state monitor
    network_monitor: Arc<NetworkMonitor>,
    /// Active crises
    active_crises: Arc<DashMap<Uuid, CrisisEvent>>,
    /// Crisis history
    crisis_history: Arc<RwLock<VecDeque<CrisisEvent>>>,
    /// HAVOC metrics
    metrics: Arc<RwLock<HavocMetrics>>,
    /// Crisis event broadcaster
    crisis_events: broadcast::Sender<CrisisEvent>,
    /// Action event broadcaster
    action_events: broadcast::Sender<EmergencyAction>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// HAVOC node configuration
#[derive(Debug, Clone)]
pub struct HavocConfig {
    /// Crisis detection sensitivity (0.0-1.0)
    pub detection_sensitivity: f64,
    /// Minimum crisis duration before response
    pub min_crisis_duration: Duration,
    /// Resource reallocation aggressiveness (0.0-1.0)
    pub reallocation_aggressiveness: f64,
    /// Enable predictive crisis detection
    pub predictive_detection: bool,
    /// Crisis history size
    pub history_size: usize,
    /// Network monitoring interval
    pub monitoring_interval: Duration,
    /// Emergency response timeout
    pub response_timeout: Duration,
    /// Cascading failure threshold
    pub cascading_threshold: f64,
}

impl Default for HavocConfig {
    fn default() -> Self {
        Self {
            detection_sensitivity: crate::constants::CRISIS_DETECTION_SENSITIVITY,
            min_crisis_duration: Duration::from_secs(30),
            reallocation_aggressiveness: 0.7,
            predictive_detection: true,
            history_size: 1000,
            monitoring_interval: Duration::from_secs(5),
            response_timeout: Duration::from_secs(60),
            cascading_threshold: 0.6,
        }
    }
}

/// Crisis detection system
pub struct CrisisDetector {
    /// Detection algorithms
    detection_algorithms: Vec<Box<dyn CrisisDetectionAlgorithm>>,
    /// Detection thresholds
    thresholds: RwLock<HashMap<CrisisType, f64>>,
    /// Historical patterns
    pattern_history: RwLock<VecDeque<NetworkPattern>>,
}

/// Network pattern for crisis prediction
#[derive(Debug, Clone)]
struct NetworkPattern {
    timestamp: DateTime<Utc>,
    resource_utilization: f64,
    node_failure_rate: f64,
    response_times: f64,
    thermal_signature: f64,
}

/// Crisis detection algorithm trait
pub trait CrisisDetectionAlgorithm: Send + Sync {
    fn detect_crisis(&self, pattern: &NetworkPattern) -> Option<(CrisisType, f64)>;
    fn algorithm_name(&self) -> &str;
}

/// Emergency response coordinator
pub struct EmergencyCoordinator {
    /// Active emergency actions
    active_actions: DashMap<Uuid, EmergencyAction>,
    /// Response strategies
    response_strategies: HashMap<CrisisType, Vec<ActionType>>,
    /// Resource pools for emergency use
    emergency_resource_pools: RwLock<HashMap<String, f64>>,
}

/// Network state monitor
pub struct NetworkMonitor {
    /// Node states
    node_states: DashMap<String, NodeState>,
    /// Network topology
    network_topology: RwLock<NetworkTopology>,
    /// Performance metrics
    performance_metrics: RwLock<NetworkPerformanceMetrics>,
}

/// Individual node state
#[derive(Debug, Clone)]
struct NodeState {
    node_id: String,
    last_heartbeat: DateTime<Utc>,
    resource_utilization: f64,
    performance_score: f64,
    failure_count: u32,
    thermal_signature: Option<ThermalSignature>,
}

/// Network topology representation
#[derive(Debug, Default)]
struct NetworkTopology {
    nodes: HashSet<String>,
    connections: HashMap<String, Vec<String>>,
    partition_groups: Vec<Vec<String>>,
}

/// Network performance metrics
#[derive(Debug, Default)]
struct NetworkPerformanceMetrics {
    average_response_time: f64,
    throughput: f64,
    error_rate: f64,
    availability: f64,
    last_update: Option<DateTime<Utc>>,
}

impl HavocNode {
    /// Create a new HAVOC node
    pub fn new(id: String, config: HavocConfig) -> Self {
        let (crisis_events, _) = broadcast::channel(1000);
        let (action_events, _) = broadcast::channel(1000);
        
        Self {
            id,
            config: config.clone(),
            crisis_detector: Arc::new(CrisisDetector::new(config.detection_sensitivity)),
            response_coordinator: Arc::new(EmergencyCoordinator::new()),
            network_monitor: Arc::new(NetworkMonitor::new()),
            active_crises: Arc::new(DashMap::new()),
            crisis_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            metrics: Arc::new(RwLock::new(HavocMetrics::default())),
            crisis_events,
            action_events,
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the HAVOC node
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::havoc_error("HAVOC node already running"));
            }
            *running = true;
        }
        
        info!("Starting HAVOC node: {}", self.id);
        
        // Start monitoring and detection
        self.start_crisis_monitoring().await;
        
        Ok(())
    }
    
    /// Stop the HAVOC node
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping HAVOC node: {}", self.id);
        Ok(())
    }
    
    /// Update node state for crisis detection
    pub async fn update_node_state(
        &self,
        node_id: String,
        utilization: f64,
        performance: f64,
        thermal_signature: Option<ThermalSignature>,
    ) -> ResourceResult<()> {
        let node_state = NodeState {
            node_id: node_id.clone(),
            last_heartbeat: Utc::now(),
            resource_utilization: utilization.clamp(0.0, 1.0),
            performance_score: performance.clamp(0.0, 1.0),
            failure_count: 0,
            thermal_signature,
        };
        
        self.network_monitor.node_states.insert(node_id.clone(), node_state);
        
        // Trigger crisis detection check
        if let Some((crisis_type, severity_score)) = self.check_for_crisis(&node_id).await {
            self.handle_detected_crisis(crisis_type, severity_score, vec![node_id]).await?;
        }
        
        Ok(())
    }
    
    /// Manually trigger emergency resource reallocation
    pub async fn trigger_emergency_reallocation(
        &self,
        crisis_type: CrisisType,
        affected_nodes: Vec<String>,
        target_reallocation: f64,
    ) -> ResourceResult<Uuid> {
        let crisis = CrisisEvent {
            id: Uuid::new_v4(),
            severity: self.determine_severity_from_reallocation(target_reallocation),
            affected_nodes: affected_nodes.clone(),
            crisis_type,
            detected_at: Utc::now(),
            description: "Manual emergency reallocation triggered".to_string(),
            predicted_impact: CrisisImpact {
                expected_duration: Duration::from_minutes(10),
                affected_capacity_percentage: target_reallocation,
                performance_impact: target_reallocation * 0.5,
                cascading_probability: 0.2,
                recovery_difficulty: 5,
            },
            status: CrisisStatus::Detected,
        };
        
        let crisis_id = crisis.id;
        self.active_crises.insert(crisis_id, crisis.clone());
        
        // Execute emergency response
        self.execute_emergency_response(crisis).await?;
        
        Ok(crisis_id)
    }
    
    /// Get current network stress level
    pub fn get_network_stress_level(&self) -> f64 {
        let node_states: Vec<_> = self.network_monitor.node_states
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        if node_states.is_empty() {
            return 0.0;
        }
        
        // Calculate average utilization
        let avg_utilization: f64 = node_states.iter()
            .map(|state| state.resource_utilization)
            .sum::<f64>() / node_states.len() as f64;
        
        // Calculate performance degradation
        let avg_performance: f64 = node_states.iter()
            .map(|state| state.performance_score)
            .sum::<f64>() / node_states.len() as f64;
        
        let performance_stress = 1.0 - avg_performance;
        
        // Calculate failure rate
        let total_failures: u32 = node_states.iter()
            .map(|state| state.failure_count)
            .sum();
        let failure_stress = (total_failures as f64 / node_states.len() as f64).min(1.0);
        
        // Combine stress factors using biological stress response model
        crate::math::network_stress_calculation(
            avg_utilization,
            performance_stress,
            0.5, // Bandwidth stress placeholder
            failure_stress / 10.0, // Normalize failure rate
            0.1, // Temporal factor placeholder
        )
    }
    
    /// Get active crises
    pub fn get_active_crises(&self) -> Vec<CrisisEvent> {
        self.active_crises.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Get HAVOC metrics
    pub fn get_metrics(&self) -> HavocMetrics {
        self.metrics.read().clone()
    }
    
    /// Subscribe to crisis events
    pub fn subscribe_to_crisis_events(&self) -> broadcast::Receiver<CrisisEvent> {
        self.crisis_events.subscribe()
    }
    
    /// Subscribe to action events
    pub fn subscribe_to_action_events(&self) -> broadcast::Receiver<EmergencyAction> {
        self.action_events.subscribe()
    }
    
    /// Calculate HAVOC response using mathematical model
    pub fn calculate_havoc_response(&self, network_stress: f64, available_resources: f64) -> f64 {
        let emergency_threshold = self.config.reallocation_aggressiveness;
        let criticality_factor = 1.5; // Based on current crisis severity
        
        crate::math::havoc_response_strength(
            emergency_threshold,
            network_stress,
            available_resources,
            criticality_factor,
        )
    }
    
    // Private methods
    
    async fn start_crisis_monitoring(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let crisis_detector = Arc::clone(&self.crisis_detector);
        let network_monitor = Arc::clone(&self.network_monitor);
        let node_id = self.id.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Collect network pattern
                let pattern = NetworkPattern {
                    timestamp: Utc::now(),
                    resource_utilization: Self::calculate_avg_utilization(&network_monitor),
                    node_failure_rate: Self::calculate_failure_rate(&network_monitor),
                    response_times: Self::calculate_avg_response_time(&network_monitor),
                    thermal_signature: Self::calculate_avg_thermal(&network_monitor),
                };
                
                // Detect potential crises
                if let Some((crisis_type, severity)) = crisis_detector.detect_pattern_crisis(&pattern) {
                    debug!("HAVOC crisis detected: {:?} (severity: {:.2})", crisis_type, severity);
                    
                    // This would normally trigger crisis handling
                    // For the monitoring task, we just log the detection
                }
            }
        });
    }
    
    fn calculate_avg_utilization(network_monitor: &NetworkMonitor) -> f64 {
        let states: Vec<_> = network_monitor.node_states.iter().map(|entry| entry.value().clone()).collect();
        if states.is_empty() {
            return 0.0;
        }
        states.iter().map(|s| s.resource_utilization).sum::<f64>() / states.len() as f64
    }
    
    fn calculate_failure_rate(network_monitor: &NetworkMonitor) -> f64 {
        let states: Vec<_> = network_monitor.node_states.iter().map(|entry| entry.value().clone()).collect();
        if states.is_empty() {
            return 0.0;
        }
        let total_failures: u32 = states.iter().map(|s| s.failure_count).sum();
        total_failures as f64 / states.len() as f64
    }
    
    fn calculate_avg_response_time(_network_monitor: &NetworkMonitor) -> f64 {
        // Placeholder - would calculate from actual metrics
        0.5
    }
    
    fn calculate_avg_thermal(network_monitor: &NetworkMonitor) -> f64 {
        let states: Vec<_> = network_monitor.node_states.iter().map(|entry| entry.value().clone()).collect();
        
        let thermal_values: Vec<f64> = states.iter()
            .filter_map(|s| s.thermal_signature.as_ref())
            .map(|ts| ts.signature_value)
            .collect();
        
        if thermal_values.is_empty() {
            return 0.0;
        }
        
        thermal_values.iter().sum::<f64>() / thermal_values.len() as f64
    }
    
    async fn check_for_crisis(&self, _node_id: &str) -> Option<(CrisisType, f64)> {
        let network_stress = self.get_network_stress_level();
        
        if network_stress > self.config.cascading_threshold {
            Some((CrisisType::CascadingFailure, network_stress))
        } else if network_stress > 0.8 {
            Some((CrisisType::SystemOverload, network_stress))
        } else if network_stress > 0.6 {
            Some((CrisisType::PerformanceDegradation, network_stress))
        } else {
            None
        }
    }
    
    async fn handle_detected_crisis(
        &self,
        crisis_type: CrisisType,
        severity_score: f64,
        affected_nodes: Vec<String>,
    ) -> ResourceResult<()> {
        let severity = self.determine_severity(severity_score);
        
        let crisis = CrisisEvent {
            id: Uuid::new_v4(),
            severity,
            affected_nodes: affected_nodes.clone(),
            crisis_type,
            detected_at: Utc::now(),
            description: format!("Crisis detected: {:?} with severity {:.2}", crisis_type, severity_score),
            predicted_impact: self.predict_crisis_impact(crisis_type, severity_score, &affected_nodes),
            status: CrisisStatus::Detected,
        };
        
        self.active_crises.insert(crisis.id, crisis.clone());
        
        // Broadcast crisis event
        if let Err(e) = self.crisis_events.send(crisis.clone()) {
            warn!("Failed to broadcast crisis event: {}", e);
        }
        
        // Execute emergency response
        self.execute_emergency_response(crisis).await?;
        
        Ok(())
    }
    
    async fn execute_emergency_response(&self, crisis: CrisisEvent) -> ResourceResult<()> {
        let reallocation_percentage = crisis.severity.reallocation_percentage();
        
        let action = EmergencyAction {
            id: Uuid::new_v4(),
            crisis_id: crisis.id,
            action_type: ActionType::ResourceReallocation,
            target_nodes: crisis.affected_nodes.clone(),
            resources: HashMap::from([("cpu".to_string(), reallocation_percentage)]),
            priority: crisis.severity.response_priority(),
            expected_completion: crisis.severity.response_time(),
            status: ActionStatus::Pending,
            timestamp: Utc::now(),
        };
        
        // Store action
        self.response_coordinator.active_actions.insert(action.id, action.clone());
        
        // Broadcast action event
        if let Err(e) = self.action_events.send(action.clone()) {
            warn!("Failed to broadcast action event: {}", e);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_crises_detected += 1;
            metrics.emergency_responses_triggered += 1;
        }
        
        info!("Executed emergency response for crisis {} with {}% resource reallocation", 
            crisis.id, reallocation_percentage * 100.0);
        
        Ok(())
    }
    
    fn determine_severity(&self, severity_score: f64) -> CrisisSeverity {
        if severity_score >= 0.95 {
            CrisisSeverity::Catastrophic
        } else if severity_score >= 0.85 {
            CrisisSeverity::Critical
        } else if severity_score >= 0.7 {
            CrisisSeverity::High
        } else if severity_score >= 0.5 {
            CrisisSeverity::Moderate
        } else {
            CrisisSeverity::Low
        }
    }
    
    fn determine_severity_from_reallocation(&self, reallocation: f64) -> CrisisSeverity {
        if reallocation >= 0.8 {
            CrisisSeverity::Catastrophic
        } else if reallocation >= 0.6 {
            CrisisSeverity::Critical
        } else if reallocation >= 0.35 {
            CrisisSeverity::High
        } else if reallocation >= 0.15 {
            CrisisSeverity::Moderate
        } else {
            CrisisSeverity::Low
        }
    }
    
    fn predict_crisis_impact(&self, crisis_type: CrisisType, severity_score: f64, _affected_nodes: &[String]) -> CrisisImpact {
        let base_duration = Duration::from_minutes(5);
        let severity_multiplier = 1.0 + severity_score * 3.0;
        
        CrisisImpact {
            expected_duration: Duration::from_millis((base_duration.as_millis() as f64 * severity_multiplier) as u64),
            affected_capacity_percentage: severity_score * 0.6,
            performance_impact: severity_score * 0.4,
            cascading_probability: match crisis_type {
                CrisisType::CascadingFailure => severity_score * 0.9,
                CrisisType::SystemOverload => severity_score * 0.7,
                _ => severity_score * 0.3,
            },
            recovery_difficulty: (severity_score * 10.0) as u8,
        }
    }
}

impl CrisisDetector {
    fn new(sensitivity: f64) -> Self {
        let mut detector = Self {
            detection_algorithms: Vec::new(),
            thresholds: RwLock::new(HashMap::new()),
            pattern_history: RwLock::new(VecDeque::with_capacity(1000)),
        };
        
        // Initialize default thresholds
        {
            let mut thresholds = detector.thresholds.write();
            thresholds.insert(CrisisType::ResourceShortage, 0.85 * sensitivity);
            thresholds.insert(CrisisType::SystemOverload, 0.9 * sensitivity);
            thresholds.insert(CrisisType::CascadingFailure, 0.7 * sensitivity);
            thresholds.insert(CrisisType::PerformanceDegradation, 0.75 * sensitivity);
        }
        
        detector
    }
    
    fn detect_pattern_crisis(&self, pattern: &NetworkPattern) -> Option<(CrisisType, f64)> {
        // Store pattern in history
        {
            let mut history = self.pattern_history.write();
            history.push_back(pattern.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        
        let thresholds = self.thresholds.read();
        
        // Check for resource shortage
        if pattern.resource_utilization > *thresholds.get(&CrisisType::ResourceShortage).unwrap_or(&0.85) {
            return Some((CrisisType::ResourceShortage, pattern.resource_utilization));
        }
        
        // Check for system overload
        if pattern.resource_utilization > *thresholds.get(&CrisisType::SystemOverload).unwrap_or(&0.9) {
            return Some((CrisisType::SystemOverload, pattern.resource_utilization));
        }
        
        // Check for performance degradation
        if pattern.response_times > *thresholds.get(&CrisisType::PerformanceDegradation).unwrap_or(&0.75) {
            return Some((CrisisType::PerformanceDegradation, pattern.response_times));
        }
        
        // Check for cascading failure
        if pattern.node_failure_rate > *thresholds.get(&CrisisType::CascadingFailure).unwrap_or(&0.7) {
            return Some((CrisisType::CascadingFailure, pattern.node_failure_rate));
        }
        
        None
    }
}

impl EmergencyCoordinator {
    fn new() -> Self {
        let mut response_strategies = HashMap::new();
        
        response_strategies.insert(
            CrisisType::ResourceShortage,
            vec![ActionType::ResourceReallocation, ActionType::LoadBalancing],
        );
        response_strategies.insert(
            CrisisType::SystemOverload,
            vec![ActionType::EmergencyScaling, ActionType::TrafficRerouting],
        );
        response_strategies.insert(
            CrisisType::CascadingFailure,
            vec![ActionType::NodeIsolation, ActionType::BackupActivation, ActionType::DataEvacuation],
        );
        
        Self {
            active_actions: DashMap::new(),
            response_strategies,
            emergency_resource_pools: RwLock::new(HashMap::new()),
        }
    }
}

impl NetworkMonitor {
    fn new() -> Self {
        Self {
            node_states: DashMap::new(),
            network_topology: RwLock::new(NetworkTopology::default()),
            performance_metrics: RwLock::new(NetworkPerformanceMetrics::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_crisis_severity_properties() {
        assert_eq!(CrisisSeverity::Catastrophic.response_priority(), 10);
        assert_eq!(CrisisSeverity::Low.response_priority(), 3);
        
        assert_eq!(CrisisSeverity::Catastrophic.reallocation_percentage(), 0.80);
        assert_eq!(CrisisSeverity::Low.reallocation_percentage(), 0.05);
        
        assert!(CrisisSeverity::Critical.response_time() < CrisisSeverity::Low.response_time());
    }
    
    #[tokio::test]
    async fn test_havoc_node_creation() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        assert_eq!(node.id, "havoc-test");
        assert_eq!(node.get_active_crises().len(), 0);
    }
    
    #[tokio::test]
    async fn test_node_state_update() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        node.update_node_state(
            "test-node".to_string(),
            0.85,
            0.7,
            None,
        ).await.unwrap();
        
        // Check if node state was recorded
        assert!(node.network_monitor.node_states.contains_key("test-node"));
    }
    
    #[tokio::test]
    async fn test_emergency_reallocation() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        let crisis_id = node.trigger_emergency_reallocation(
            CrisisType::ResourceShortage,
            vec!["node1".to_string(), "node2".to_string()],
            0.6,
        ).await.unwrap();
        
        assert!(!crisis_id.is_nil());
        assert_eq!(node.get_active_crises().len(), 1);
        
        let metrics = node.get_metrics();
        assert_eq!(metrics.total_crises_detected, 1);
        assert_eq!(metrics.emergency_responses_triggered, 1);
    }
    
    #[test]
    fn test_network_stress_calculation() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        // Add some node states
        let thermal_sig = crate::thermal::ThermalSignature::new(
            "test-node".to_string(),
            0.8, 0.7, 0.6, 0.5,
            "test".to_string(),
        );
        
        let node_state = NodeState {
            node_id: "test-node".to_string(),
            last_heartbeat: Utc::now(),
            resource_utilization: 0.8,
            performance_score: 0.6,
            failure_count: 2,
            thermal_signature: Some(thermal_sig),
        };
        
        node.network_monitor.node_states.insert("test-node".to_string(), node_state);
        
        let stress = node.get_network_stress_level();
        assert!(stress > 0.0);
        assert!(stress <= 1.0);
    }
    
    #[test]
    fn test_havoc_response_calculation() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        let response = node.calculate_havoc_response(0.8, 2.0);
        assert!(response > 0.0);
        
        // Higher stress should produce higher response
        let high_stress_response = node.calculate_havoc_response(0.95, 2.0);
        assert!(high_stress_response > response);
    }
    
    #[test]
    fn test_crisis_detector() {
        let detector = CrisisDetector::new(1.0);
        
        let pattern = NetworkPattern {
            timestamp: Utc::now(),
            resource_utilization: 0.9,
            node_failure_rate: 0.1,
            response_times: 0.3,
            thermal_signature: 0.7,
        };
        
        let detection = detector.detect_pattern_crisis(&pattern);
        assert!(detection.is_some());
        
        if let Some((crisis_type, severity)) = detection {
            assert_eq!(crisis_type, CrisisType::SystemOverload);
            assert_eq!(severity, 0.9);
        }
    }
    
    #[test]
    fn test_crisis_impact_prediction() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        let impact = node.predict_crisis_impact(
            CrisisType::CascadingFailure,
            0.8,
            &vec!["node1".to_string()],
        );
        
        assert!(impact.cascading_probability > 0.5); // High for cascading failure
        assert!(impact.affected_capacity_percentage > 0.0);
        assert!(impact.expected_duration > Duration::from_secs(0));
    }
    
    #[tokio::test]
    async fn test_crisis_event_broadcasting() {
        let config = HavocConfig::default();
        let node = HavocNode::new("havoc-test".to_string(), config);
        
        let mut crisis_receiver = node.subscribe_to_crisis_events();
        let mut action_receiver = node.subscribe_to_action_events();
        
        // Trigger a crisis
        let _crisis_id = node.trigger_emergency_reallocation(
            CrisisType::SystemOverload,
            vec!["test-node".to_string()],
            0.4,
        ).await.unwrap();
        
        // Should receive crisis and action events
        let crisis_event = tokio::time::timeout(Duration::from_millis(100), crisis_receiver.recv()).await;
        assert!(crisis_event.is_ok());
        
        let action_event = tokio::time::timeout(Duration::from_millis(100), action_receiver.recv()).await;
        assert!(action_event.is_ok());
    }
}