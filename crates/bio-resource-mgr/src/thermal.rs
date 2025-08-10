//! Thermal Monitoring and Management System
//! 
//! Implements biological thermal sensing inspired by snake heat detection and bat echolocation
//! for real-time resource monitoring, congestion detection, and performance optimization.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, warn, error};

#[cfg(feature = "thermal-monitoring")]
use sysinfo::{System, SystemExt, CpuExt, ProcessExt};

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::ThermalMetrics;

/// Thermal signature representing resource usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSignature {
    /// Unique identifier for this signature
    pub id: String,
    /// Timestamp when signature was captured
    pub timestamp: DateTime<Utc>,
    /// CPU usage pattern (0.0-1.0)
    pub cpu_usage: f64,
    /// Memory usage pattern (0.0-1.0)  
    pub memory_pattern: f64,
    /// Network bandwidth utilization (0.0-1.0)
    pub network_bandwidth: f64,
    /// Storage I/O access rate (0.0-1.0)
    pub storage_access: f64,
    /// Combined thermal signature value
    pub signature_value: f64,
    /// Node identifier that generated this signature
    pub node_id: String,
    /// Resource type being monitored
    pub resource_type: String,
}

impl ThermalSignature {
    /// Create a new thermal signature
    pub fn new(
        node_id: String,
        cpu_usage: f64,
        memory_pattern: f64,
        network_bandwidth: f64,
        storage_access: f64,
        resource_type: String,
    ) -> Self {
        let signature_value = crate::math::thermal_signature(
            cpu_usage,
            memory_pattern,
            network_bandwidth,
            storage_access,
        );
        
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            cpu_usage,
            memory_pattern,
            network_bandwidth,
            storage_access,
            signature_value,
            node_id,
            resource_type,
        }
    }
    
    /// Calculate signature similarity to another signature
    pub fn similarity(&self, other: &ThermalSignature) -> f64 {
        let cpu_diff = (self.cpu_usage - other.cpu_usage).abs();
        let mem_diff = (self.memory_pattern - other.memory_pattern).abs();
        let net_diff = (self.network_bandwidth - other.network_bandwidth).abs();
        let storage_diff = (self.storage_access - other.storage_access).abs();
        
        let total_diff = cpu_diff + mem_diff + net_diff + storage_diff;
        1.0 - (total_diff / 4.0) // Normalize to 0-1 range
    }
    
    /// Check if signature indicates high resource stress
    pub fn is_stressed(&self, threshold: f64) -> bool {
        self.signature_value > threshold
    }
    
    /// Get dominant resource type
    pub fn dominant_resource(&self) -> String {
        let max_usage = self.cpu_usage
            .max(self.memory_pattern)
            .max(self.network_bandwidth)
            .max(self.storage_access);
            
        if (self.cpu_usage - max_usage).abs() < f64::EPSILON {
            "cpu".to_string()
        } else if (self.memory_pattern - max_usage).abs() < f64::EPSILON {
            "memory".to_string()
        } else if (self.network_bandwidth - max_usage).abs() < f64::EPSILON {
            "network".to_string()
        } else {
            "storage".to_string()
        }
    }
}

/// Thermal threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalThreshold {
    /// Threshold name
    pub name: String,
    /// Warning threshold (0.0-1.0)
    pub warning: f64,
    /// Critical threshold (0.0-1.0)
    pub critical: f64,
    /// Emergency threshold (0.0-1.0)
    pub emergency: f64,
    /// Resource type this threshold applies to
    pub resource_type: String,
}

impl Default for ThermalThreshold {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            warning: 0.7,
            critical: 0.85,
            emergency: 0.95,
            resource_type: "all".to_string(),
        }
    }
}

impl ThermalThreshold {
    /// Check threshold level for given value
    pub fn check_level(&self, value: f64) -> ThermalLevel {
        if value >= self.emergency {
            ThermalLevel::Emergency
        } else if value >= self.critical {
            ThermalLevel::Critical
        } else if value >= self.warning {
            ThermalLevel::Warning
        } else {
            ThermalLevel::Normal
        }
    }
}

/// Thermal alert levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThermalLevel {
    /// Normal thermal conditions
    Normal = 0,
    /// Warning level - attention needed
    Warning = 1,
    /// Critical level - action required
    Critical = 2,
    /// Emergency level - immediate intervention
    Emergency = 3,
}

impl std::fmt::Display for ThermalLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermalLevel::Normal => write!(f, "NORMAL"),
            ThermalLevel::Warning => write!(f, "WARNING"),
            ThermalLevel::Critical => write!(f, "CRITICAL"),
            ThermalLevel::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

/// Thermal alert event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalAlert {
    /// Alert identifier
    pub id: String,
    /// Node that triggered the alert
    pub node_id: String,
    /// Alert level
    pub level: ThermalLevel,
    /// Thermal signature that triggered the alert
    pub signature: ThermalSignature,
    /// Alert message
    pub message: String,
    /// Timestamp of alert
    pub timestamp: DateTime<Utc>,
}

/// Thermal route information for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalRoute {
    /// Route identifier
    pub route_id: String,
    /// Source node
    pub source_node: String,
    /// Destination node
    pub destination_node: String,
    /// Current thermal load on route
    pub thermal_load: f64,
    /// Capacity remaining
    pub capacity_remaining: f64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Route reliability score (0.0-1.0)
    pub reliability: f64,
    /// Last thermal signature update
    pub last_update: DateTime<Utc>,
}

impl ThermalRoute {
    /// Check if route is congested
    pub fn is_congested(&self, threshold: f64) -> bool {
        self.thermal_load > threshold || self.capacity_remaining < (1.0 - threshold)
    }
    
    /// Calculate route efficiency score
    pub fn efficiency_score(&self) -> f64 {
        let load_factor = 1.0 - self.thermal_load;
        let capacity_factor = self.capacity_remaining;
        let reliability_factor = self.reliability;
        let time_factor = 1.0 / (1.0 + self.avg_response_time.as_millis() as f64 / 1000.0);
        
        (load_factor + capacity_factor + reliability_factor + time_factor) / 4.0
    }
}

/// Thermal Node - Real-time resource monitoring and thermal signature analysis
/// 
/// Inspired by ant colony pheromone concentration trails to indicate resource availability
pub struct ThermalNode {
    /// Node identifier
    pub id: String,
    /// Thermal monitoring configuration
    config: ThermalConfig,
    /// Current thermal state
    thermal_state: Arc<RwLock<ThermalState>>,
    /// Thermal signature history
    signature_history: Arc<RwLock<VecDeque<ThermalSignature>>>,
    /// Thermal thresholds
    thresholds: Arc<RwLock<HashMap<String, ThermalThreshold>>>,
    /// Route thermal information
    route_info: Arc<DashMap<String, ThermalRoute>>,
    /// Alert broadcast channel
    alert_sender: broadcast::Sender<ThermalAlert>,
    /// System monitor (if available)
    #[cfg(feature = "thermal-monitoring")]
    system_monitor: Arc<RwLock<System>>,
    /// Thermal metrics
    metrics: Arc<RwLock<ThermalMetrics>>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Thermal monitoring configuration
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Monitoring sampling frequency
    pub sampling_frequency: Duration,
    /// Signature history size
    pub history_size: usize,
    /// Default thermal thresholds
    pub default_thresholds: ThermalThreshold,
    /// Route congestion threshold
    pub congestion_threshold: f64,
    /// Enable predictive thermal analysis
    pub predictive_analysis: bool,
    /// Thermal signature compression enabled
    pub signature_compression: bool,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            sampling_frequency: crate::constants::THERMAL_MONITORING_INTERVAL,
            history_size: 1000,
            default_thresholds: ThermalThreshold::default(),
            congestion_threshold: 0.8,
            predictive_analysis: true,
            signature_compression: true,
        }
    }
}

/// Current thermal state
#[derive(Debug, Default)]
struct ThermalState {
    /// Current overall thermal level
    thermal_level: ThermalLevel,
    /// Latest thermal signature
    latest_signature: Option<ThermalSignature>,
    /// Last update timestamp
    last_update: Option<Instant>,
    /// Active thermal alerts
    active_alerts: HashMap<String, ThermalAlert>,
}

impl ThermalNode {
    /// Create a new thermal monitoring node
    pub fn new(id: String, config: ThermalConfig) -> Self {
        let (alert_sender, _) = broadcast::channel(1000);
        
        Self {
            id,
            config: config.clone(),
            thermal_state: Arc::new(RwLock::new(ThermalState::default())),
            signature_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            thresholds: Arc::new(RwLock::new(HashMap::new())),
            route_info: Arc::new(DashMap::new()),
            alert_sender,
            #[cfg(feature = "thermal-monitoring")]
            system_monitor: Arc::new(RwLock::new(System::new_all())),
            metrics: Arc::new(RwLock::new(ThermalMetrics::default())),
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start thermal monitoring
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::thermal_error("Thermal node already running"));
            }
            *running = true;
        }
        
        info!("Starting thermal monitoring node: {}", self.id);
        
        // Initialize default thresholds
        {
            let mut thresholds = self.thresholds.write();
            thresholds.insert("default".to_string(), self.config.default_thresholds.clone());
        }
        
        // Start monitoring task
        self.start_monitoring_task().await;
        
        Ok(())
    }
    
    /// Stop thermal monitoring
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping thermal monitoring node: {}", self.id);
        Ok(())
    }
    
    /// Record a thermal signature
    pub async fn record_signature(&self, signature: ThermalSignature) -> ResourceResult<()> {
        // Store signature in history
        {
            let mut history = self.signature_history.write();
            history.push_back(signature.clone());
            
            // Maintain history size
            while history.len() > self.config.history_size {
                history.pop_front();
            }
        }
        
        // Update thermal state
        {
            let mut state = self.thermal_state.write();
            state.latest_signature = Some(signature.clone());
            state.last_update = Some(Instant::now());
            
            // Check for thermal alerts
            let threshold = self.thresholds.read()
                .get("default")
                .unwrap_or(&self.config.default_thresholds)
                .clone();
            
            let level = threshold.check_level(signature.signature_value);
            
            if level > ThermalLevel::Normal {
                let alert = ThermalAlert {
                    id: uuid::Uuid::new_v4().to_string(),
                    node_id: signature.node_id.clone(),
                    level,
                    signature: signature.clone(),
                    message: format!("Thermal threshold exceeded: {} ({})", signature.signature_value, level),
                    timestamp: Utc::now(),
                };
                
                state.active_alerts.insert(alert.id.clone(), alert.clone());
                
                // Broadcast alert
                if let Err(e) = self.alert_sender.send(alert) {
                    warn!("Failed to broadcast thermal alert: {}", e);
                }
            }
            
            state.thermal_level = level;
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_signatures += 1;
            metrics.average_signature_value = (metrics.average_signature_value * (metrics.total_signatures - 1) as f64 + signature.signature_value) / metrics.total_signatures as f64;
            
            if signature.signature_value > metrics.peak_signature_value {
                metrics.peak_signature_value = signature.signature_value;
            }
        }
        
        Ok(())
    }
    
    /// Update route thermal information
    pub async fn update_route_thermal(&self, route: ThermalRoute) -> ResourceResult<()> {
        self.route_info.insert(route.route_id.clone(), route);
        Ok(())
    }
    
    /// Get optimal routes based on thermal signatures
    pub fn get_optimal_routes(&self, count: usize) -> Vec<ThermalRoute> {
        let mut routes: Vec<_> = self.route_info
            .iter()
            .map(|entry| entry.value().clone())
            .collect();
        
        // Sort by efficiency score (highest first)
        routes.sort_by(|a, b| b.efficiency_score().partial_cmp(&a.efficiency_score()).unwrap());
        
        routes.into_iter().take(count).collect()
    }
    
    /// Get congested routes requiring attention
    pub fn get_congested_routes(&self) -> Vec<ThermalRoute> {
        self.route_info
            .iter()
            .filter_map(|entry| {
                let route = entry.value();
                if route.is_congested(self.config.congestion_threshold) {
                    Some(route.clone())
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get current thermal level
    pub fn current_thermal_level(&self) -> ThermalLevel {
        let state = self.thermal_state.read();
        state.thermal_level
    }
    
    /// Get latest thermal signature
    pub fn latest_signature(&self) -> Option<ThermalSignature> {
        let state = self.thermal_state.read();
        state.latest_signature.clone()
    }
    
    /// Get thermal signature history
    pub fn get_signature_history(&self, limit: Option<usize>) -> Vec<ThermalSignature> {
        let history = self.signature_history.read();
        
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.iter().cloned().collect(),
        }
    }
    
    /// Subscribe to thermal alerts
    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<ThermalAlert> {
        self.alert_sender.subscribe()
    }
    
    /// Get thermal metrics
    pub fn get_metrics(&self) -> ThermalMetrics {
        self.metrics.read().clone()
    }
    
    /// Set custom thermal threshold
    pub fn set_threshold(&self, name: String, threshold: ThermalThreshold) {
        let mut thresholds = self.thresholds.write();
        thresholds.insert(name, threshold);
    }
    
    /// Analyze thermal patterns for predictions
    pub fn analyze_thermal_patterns(&self) -> Option<ThermalPrediction> {
        let history = self.signature_history.read();
        
        if history.len() < 10 {
            return None;
        }
        
        // Simple trend analysis - could be enhanced with ML
        let recent_signatures: Vec<_> = history.iter().rev().take(10).collect();
        
        let trend = self.calculate_trend(&recent_signatures);
        let predicted_level = self.predict_next_level(&recent_signatures);
        
        Some(ThermalPrediction {
            trend,
            predicted_level,
            confidence: 0.7, // Simple prediction confidence
            time_horizon: Duration::from_minutes(5),
        })
    }
    
    // Private methods
    
    async fn start_monitoring_task(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let node_id = self.id.clone();
        
        #[cfg(feature = "thermal-monitoring")]
        let system_monitor = Arc::clone(&self.system_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.sampling_frequency);
            
            while *running.read() {
                interval.tick().await;
                
                #[cfg(feature = "thermal-monitoring")]
                {
                    // Update system information
                    system_monitor.write().refresh_all();
                    let system = system_monitor.read();
                    
                    // Calculate thermal signature
                    let cpu_usage = system.global_cpu_info().cpu_usage() as f64 / 100.0;
                    let memory_pattern = (system.used_memory() as f64) / (system.total_memory() as f64);
                    
                    // Network and storage would require additional monitoring
                    let network_bandwidth = 0.5; // Placeholder
                    let storage_access = 0.3; // Placeholder
                    
                    let signature = ThermalSignature::new(
                        node_id.clone(),
                        cpu_usage,
                        memory_pattern,
                        network_bandwidth,
                        storage_access,
                        "system".to_string(),
                    );
                    
                    // This would normally be sent back to the main node
                    debug!("Thermal signature: {:.3} (CPU: {:.1}%, Mem: {:.1}%)", 
                        signature.signature_value, cpu_usage * 100.0, memory_pattern * 100.0);
                }
                
                #[cfg(not(feature = "thermal-monitoring"))]
                {
                    // Placeholder monitoring for testing
                    debug!("Thermal monitoring cycle (placeholder mode)");
                }
            }
        });
    }
    
    fn calculate_trend(&self, signatures: &[&ThermalSignature]) -> ThermalTrend {
        if signatures.len() < 2 {
            return ThermalTrend::Stable;
        }
        
        let mut increases = 0;
        let mut decreases = 0;
        
        for i in 1..signatures.len() {
            if signatures[i].signature_value > signatures[i-1].signature_value {
                increases += 1;
            } else if signatures[i].signature_value < signatures[i-1].signature_value {
                decreases += 1;
            }
        }
        
        if increases > decreases * 2 {
            ThermalTrend::Increasing
        } else if decreases > increases * 2 {
            ThermalTrend::Decreasing
        } else {
            ThermalTrend::Stable
        }
    }
    
    fn predict_next_level(&self, signatures: &[&ThermalSignature]) -> ThermalLevel {
        if signatures.is_empty() {
            return ThermalLevel::Normal;
        }
        
        // Simple prediction based on recent average
        let avg = signatures.iter().map(|s| s.signature_value).sum::<f64>() / signatures.len() as f64;
        
        let threshold = self.thresholds.read()
            .get("default")
            .unwrap_or(&self.config.default_thresholds)
            .clone();
        
        threshold.check_level(avg * 1.1) // Slight upward bias for prediction
    }
}

/// Thermal trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThermalTrend {
    /// Thermal signatures are increasing
    Increasing,
    /// Thermal signatures are decreasing
    Decreasing,
    /// Thermal signatures are stable
    Stable,
}

/// Thermal prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalPrediction {
    /// Predicted trend
    pub trend: ThermalTrend,
    /// Predicted thermal level
    pub predicted_level: ThermalLevel,
    /// Prediction confidence (0.0-1.0)
    pub confidence: f64,
    /// Time horizon for prediction
    pub time_horizon: Duration,
}

/// Thermal analyzer utility for advanced analysis
pub struct ThermalAnalyzer {
    /// Historical thermal data for analysis
    thermal_data: Arc<RwLock<Vec<ThermalSignature>>>,
}

impl ThermalAnalyzer {
    /// Create a new thermal analyzer
    pub fn new() -> Self {
        Self {
            thermal_data: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Add thermal data for analysis
    pub fn add_data(&self, signatures: Vec<ThermalSignature>) {
        let mut data = self.thermal_data.write();
        data.extend(signatures);
    }
    
    /// Analyze thermal patterns across multiple nodes
    pub fn analyze_network_thermal_patterns(&self) -> NetworkThermalAnalysis {
        let data = self.thermal_data.read();
        
        if data.is_empty() {
            return NetworkThermalAnalysis::default();
        }
        
        // Group by node
        let mut node_signatures: HashMap<String, Vec<&ThermalSignature>> = HashMap::new();
        for signature in data.iter() {
            node_signatures.entry(signature.node_id.clone())
                .or_insert_with(Vec::new)
                .push(signature);
        }
        
        // Calculate network-wide metrics
        let total_nodes = node_signatures.len();
        let avg_signature_value = data.iter().map(|s| s.signature_value).sum::<f64>() / data.len() as f64;
        
        let hotspots = node_signatures.iter()
            .filter_map(|(node_id, signatures)| {
                let avg = signatures.iter().map(|s| s.signature_value).sum::<f64>() / signatures.len() as f64;
                if avg > 0.8 {
                    Some(node_id.clone())
                } else {
                    None
                }
            })
            .collect();
        
        NetworkThermalAnalysis {
            total_nodes,
            avg_signature_value,
            hotspots,
            analysis_timestamp: Utc::now(),
        }
    }
}

/// Network-wide thermal analysis result
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct NetworkThermalAnalysis {
    /// Total number of nodes analyzed
    pub total_nodes: usize,
    /// Average signature value across network
    pub avg_signature_value: f64,
    /// List of thermal hotspot nodes
    pub hotspots: Vec<String>,
    /// Timestamp of analysis
    pub analysis_timestamp: DateTime<Utc>,
}

/// Thermal monitor utility for external integration
pub struct ThermalMonitor {
    thermal_nodes: Vec<Arc<ThermalNode>>,
    analyzer: ThermalAnalyzer,
}

impl ThermalMonitor {
    /// Create a new thermal monitor
    pub fn new() -> Self {
        Self {
            thermal_nodes: Vec::new(),
            analyzer: ThermalAnalyzer::new(),
        }
    }
    
    /// Add a thermal node to monitor
    pub fn add_thermal_node(&mut self, node: Arc<ThermalNode>) {
        self.thermal_nodes.push(node);
    }
    
    /// Get network-wide thermal status
    pub async fn get_network_thermal_status(&self) -> NetworkThermalStatus {
        let mut total_signatures = 0;
        let mut avg_thermal_level = 0.0;
        let mut critical_nodes = Vec::new();
        
        for node in &self.thermal_nodes {
            if let Some(signature) = node.latest_signature() {
                total_signatures += 1;
                avg_thermal_level += signature.signature_value;
                
                if node.current_thermal_level() >= ThermalLevel::Critical {
                    critical_nodes.push(node.id.clone());
                }
            }
        }
        
        if total_signatures > 0 {
            avg_thermal_level /= total_signatures as f64;
        }
        
        NetworkThermalStatus {
            total_nodes: self.thermal_nodes.len(),
            active_nodes: total_signatures,
            avg_thermal_level,
            critical_nodes,
            timestamp: Utc::now(),
        }
    }
}

/// Network thermal status summary
#[derive(Debug, Serialize, Deserialize)]
pub struct NetworkThermalStatus {
    /// Total thermal nodes
    pub total_nodes: usize,
    /// Nodes currently reporting
    pub active_nodes: usize,
    /// Average thermal level across network
    pub avg_thermal_level: f64,
    /// Nodes in critical thermal state
    pub critical_nodes: Vec<String>,
    /// Status timestamp
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_thermal_signature_creation() {
        let signature = ThermalSignature::new(
            "test-node".to_string(),
            0.8,
            0.9,
            0.7,
            0.6,
            "test".to_string(),
        );
        
        assert_eq!(signature.node_id, "test-node");
        assert_eq!(signature.cpu_usage, 0.8);
        assert!(signature.signature_value > 0.0);
        assert_eq!(signature.dominant_resource(), "memory");
    }
    
    #[tokio::test]
    async fn test_thermal_node_creation() {
        let config = ThermalConfig::default();
        let node = ThermalNode::new("thermal-test".to_string(), config);
        
        assert_eq!(node.id, "thermal-test");
        assert_eq!(node.current_thermal_level(), ThermalLevel::Normal);
    }
    
    #[tokio::test]
    async fn test_thermal_signature_recording() {
        let config = ThermalConfig::default();
        let node = ThermalNode::new("thermal-test".to_string(), config);
        
        let signature = ThermalSignature::new(
            "test-node".to_string(),
            0.5,
            0.4,
            0.3,
            0.2,
            "test".to_string(),
        );
        
        node.record_signature(signature).await.unwrap();
        
        let history = node.get_signature_history(None);
        assert_eq!(history.len(), 1);
    }
    
    #[test]
    fn test_thermal_threshold() {
        let threshold = ThermalThreshold::default();
        
        assert_eq!(threshold.check_level(0.6), ThermalLevel::Normal);
        assert_eq!(threshold.check_level(0.75), ThermalLevel::Warning);
        assert_eq!(threshold.check_level(0.9), ThermalLevel::Critical);
        assert_eq!(threshold.check_level(0.98), ThermalLevel::Emergency);
    }
    
    #[test]
    fn test_thermal_route_efficiency() {
        let route = ThermalRoute {
            route_id: "test-route".to_string(),
            source_node: "node1".to_string(),
            destination_node: "node2".to_string(),
            thermal_load: 0.3,
            capacity_remaining: 0.7,
            avg_response_time: Duration::from_millis(50),
            reliability: 0.95,
            last_update: Utc::now(),
        };
        
        assert!(!route.is_congested(0.8));
        assert!(route.efficiency_score() > 0.5);
    }
    
    #[test]
    fn test_signature_similarity() {
        let sig1 = ThermalSignature::new("node1".to_string(), 0.8, 0.7, 0.6, 0.5, "test".to_string());
        let sig2 = ThermalSignature::new("node2".to_string(), 0.8, 0.7, 0.6, 0.5, "test".to_string());
        let sig3 = ThermalSignature::new("node3".to_string(), 0.2, 0.3, 0.4, 0.1, "test".to_string());
        
        assert!(sig1.similarity(&sig2) > 0.9);
        assert!(sig1.similarity(&sig3) < 0.5);
    }
}