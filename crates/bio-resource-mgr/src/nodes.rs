//! Biological Node Types for Resource Management
//! 
//! Implements Step-up/Step-down nodes inspired by desert ant elevation control
//! and other biological resource management behaviors.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{PerformanceMetrics, ResourceMetrics};
use crate::thermal::ThermalSignature;
use crate::allocation::{ResourceProvider, AllocationStrategy};
use crate::compartments::{CompartmentType, QoSLevel};

/// Step-up Node - Increases computational capability during high-demand periods
/// 
/// Inspired by desert ants' elevation adjustment behavior for temperature regulation
/// and energy optimization. Provides bidirectional scaling capabilities.
pub struct StepUpNode {
    /// Node identifier
    pub id: String,
    /// Current scaling configuration
    config: ScalingConfig,
    /// Current resource state
    resource_state: Arc<RwLock<ResourceState>>,
    /// Scaling history for analysis
    scaling_history: Arc<RwLock<VecDeque<ScalingEvent>>>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Scaling event broadcaster
    scaling_events: broadcast::Sender<ScalingEvent>,
    /// Demand monitoring
    demand_monitor: Arc<DemandMonitor>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Step-down Node - Reduces computational capability to conserve resources during low-demand periods
/// 
/// Complementary to StepUpNode, implementing the conservation aspect of desert ant behavior.
pub struct StepDownNode {
    /// Node identifier
    pub id: String,
    /// Current scaling configuration
    config: ScalingConfig,
    /// Current resource state
    resource_state: Arc<RwLock<ResourceState>>,
    /// Scaling history for analysis
    scaling_history: Arc<RwLock<VecDeque<ScalingEvent>>>,
    /// Performance metrics
    metrics: Arc<RwLock<PerformanceMetrics>>,
    /// Scaling event broadcaster
    scaling_events: broadcast::Sender<ScalingEvent>,
    /// Conservation monitor
    conservation_monitor: Arc<ConservationMonitor>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Scaling configuration for step nodes
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Minimum scaling factor
    pub min_scaling_factor: f64,
    /// Maximum scaling factor
    pub max_scaling_factor: f64,
    /// Scaling step size
    pub scaling_step_size: f64,
    /// Demand threshold for scaling up
    pub scale_up_threshold: f64,
    /// Demand threshold for scaling down
    pub scale_down_threshold: f64,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Scaling cooldown period
    pub cooldown_period: Duration,
    /// Enable predictive scaling
    pub predictive_scaling: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_scaling_factor: 0.1,
            max_scaling_factor: 5.0,
            scaling_step_size: crate::constants::DEFAULT_SCALING_STEP,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            monitoring_interval: Duration::from_secs(10),
            cooldown_period: Duration::from_secs(60),
            predictive_scaling: true,
        }
    }
}

/// Current resource state of a scaling node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceState {
    /// Current scaling factor (1.0 = baseline)
    pub current_scaling_factor: f64,
    /// Baseline resource capacity
    pub baseline_capacity: f64,
    /// Current effective capacity
    pub effective_capacity: f64,
    /// Current utilization level (0.0-1.0)
    pub utilization_level: f64,
    /// Current demand level (0.0-1.0)
    pub demand_level: f64,
    /// Energy efficiency rating (0.0-1.0)
    pub energy_efficiency: f64,
    /// Last scaling timestamp
    pub last_scaling_time: Option<DateTime<Utc>>,
    /// Node status
    pub status: NodeStatus,
}

impl Default for ResourceState {
    fn default() -> Self {
        Self {
            current_scaling_factor: 1.0,
            baseline_capacity: 1.0,
            effective_capacity: 1.0,
            utilization_level: 0.0,
            demand_level: 0.0,
            energy_efficiency: 1.0,
            last_scaling_time: None,
            status: NodeStatus::Active,
        }
    }
}

/// Node operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeStatus {
    /// Node is initializing
    Initializing,
    /// Node is active and ready
    Active,
    /// Node is scaling up
    ScalingUp,
    /// Node is scaling down
    ScalingDown,
    /// Node is in conservation mode
    Conservation,
    /// Node is under maintenance
    Maintenance,
    /// Node has failed
    Failed,
}

/// Scaling event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    /// Event identifier
    pub id: Uuid,
    /// Node that performed scaling
    pub node_id: String,
    /// Event type
    pub event_type: ScalingEventType,
    /// Previous scaling factor
    pub previous_factor: f64,
    /// New scaling factor
    pub new_factor: f64,
    /// Demand level that triggered scaling
    pub demand_level: f64,
    /// Utilization level at time of scaling
    pub utilization_level: f64,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Scaling reason
    pub reason: String,
}

/// Types of scaling events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingEventType {
    /// Scale up event
    ScaleUp,
    /// Scale down event
    ScaleDown,
    /// Automatic scaling event
    AutoScale,
    /// Predictive scaling event
    PredictiveScale,
    /// Conservation mode activation
    ConservationMode,
    /// Emergency scaling event
    EmergencyScale,
}

/// Demand monitoring component
pub struct DemandMonitor {
    /// Historical demand data
    demand_history: RwLock<VecDeque<DemandReading>>,
    /// Prediction window size
    prediction_window: usize,
    /// Demand spike detection
    spike_threshold: f64,
}

/// Individual demand reading
#[derive(Debug, Clone)]
pub struct DemandReading {
    timestamp: DateTime<Utc>,
    demand_level: f64,
    utilization: f64,
    response_time: Duration,
}

/// Resource conservation monitoring
pub struct ConservationMonitor {
    /// Conservation opportunities
    conservation_opportunities: RwLock<Vec<ConservationOpportunity>>,
    /// Energy savings history
    energy_savings: RwLock<VecDeque<EnergySaving>>,
    /// Conservation targets
    conservation_targets: RwLock<ConservationTargets>,
}

/// Conservation opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationOpportunity {
    /// Opportunity identifier
    pub id: Uuid,
    /// Opportunity type
    pub opportunity_type: ConservationType,
    /// Estimated energy savings (0.0-1.0)
    pub estimated_savings: f64,
    /// Implementation cost
    pub implementation_cost: f64,
    /// Priority level (1-10)
    pub priority: u8,
    /// Description
    pub description: String,
}

/// Types of conservation opportunities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConservationType {
    /// CPU frequency scaling
    CpuScaling,
    /// Memory optimization
    MemoryOptimization,
    /// Network bandwidth reduction
    NetworkOptimization,
    /// Storage access optimization
    StorageOptimization,
    /// Idle state optimization
    IdleOptimization,
}

/// Energy saving record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergySaving {
    /// Saving identifier
    pub id: Uuid,
    /// Conservation type that produced savings
    pub conservation_type: ConservationType,
    /// Energy saved (normalized 0.0-1.0)
    pub energy_saved: f64,
    /// Duration of conservation
    pub duration: Duration,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Conservation targets and goals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationTargets {
    /// Target energy efficiency (0.0-1.0)
    pub target_efficiency: f64,
    /// Maximum acceptable performance reduction (0.0-1.0)
    pub max_performance_reduction: f64,
    /// Conservation priority (1-10)
    pub conservation_priority: u8,
}

impl Default for ConservationTargets {
    fn default() -> Self {
        Self {
            target_efficiency: 0.85,
            max_performance_reduction: 0.2,
            conservation_priority: 5,
        }
    }
}

impl StepUpNode {
    /// Create a new step-up node
    pub fn new(id: String, config: ScalingConfig) -> Self {
        let (scaling_events, _) = broadcast::channel(1000);
        
        Self {
            id,
            config: config.clone(),
            resource_state: Arc::new(RwLock::new(ResourceState::default())),
            scaling_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            scaling_events,
            demand_monitor: Arc::new(DemandMonitor::new(100, 0.9)),
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the step-up node
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::allocation_failed("StepUp node already running"));
            }
            *running = true;
        }
        
        info!("Starting step-up node: {}", self.id);
        
        // Start demand monitoring
        self.start_demand_monitoring().await;
        
        Ok(())
    }
    
    /// Stop the step-up node
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping step-up node: {}", self.id);
        Ok(())
    }
    
    /// Scale up resources based on demand
    pub async fn scale_up(&self, target_factor: Option<f64>) -> ResourceResult<()> {
        let mut state = self.resource_state.write();
        
        // Check cooldown period
        if let Some(last_scaling) = state.last_scaling_time {
            let elapsed = Utc::now().signed_duration_since(last_scaling);
            if elapsed < chrono::Duration::from_std(self.config.cooldown_period)? {
                return Err(ResourceError::allocation_failed("Still in cooldown period"));
            }
        }
        
        let previous_factor = state.current_scaling_factor;
        let new_factor = target_factor.unwrap_or_else(|| {
            (previous_factor + self.config.scaling_step_size).min(self.config.max_scaling_factor)
        });
        
        if new_factor <= previous_factor {
            return Err(ResourceError::allocation_failed("Cannot scale up to lower or equal factor"));
        }
        
        // Apply scaling
        state.current_scaling_factor = new_factor;
        state.effective_capacity = state.baseline_capacity * new_factor;
        state.last_scaling_time = Some(Utc::now());
        state.status = NodeStatus::ScalingUp;
        
        // Calculate energy efficiency impact
        state.energy_efficiency = self.calculate_energy_efficiency(new_factor);
        
        // Record scaling event
        let event = ScalingEvent {
            id: Uuid::new_v4(),
            node_id: self.id.clone(),
            event_type: ScalingEventType::ScaleUp,
            previous_factor,
            new_factor,
            demand_level: state.demand_level,
            utilization_level: state.utilization_level,
            timestamp: Utc::now(),
            reason: "Demand-based scaling up".to_string(),
        };
        
        // Store in history
        {
            let mut history = self.scaling_history.write();
            history.push_back(event.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        
        // Broadcast event
        if let Err(e) = self.scaling_events.send(event) {
            warn!("Failed to broadcast scaling event: {}", e);
        }
        
        info!("Scaled up node {} from {:.2}x to {:.2}x capacity", 
            self.id, previous_factor, new_factor);
        
        Ok(())
    }
    
    /// Get current resource state
    pub fn get_resource_state(&self) -> ResourceState {
        self.resource_state.read().clone()
    }
    
    /// Update demand level
    pub async fn update_demand(&self, demand_level: f64, utilization: f64) -> ResourceResult<()> {
        {
            let mut state = self.resource_state.write();
            state.demand_level = demand_level.clamp(0.0, 1.0);
            state.utilization_level = utilization.clamp(0.0, 1.0);
        }
        
        // Record demand reading
        let reading = DemandReading {
            timestamp: Utc::now(),
            demand_level,
            utilization,
            response_time: Duration::from_millis(50), // Placeholder
        };
        
        self.demand_monitor.add_reading(reading);
        
        // Check for automatic scaling
        if demand_level > self.config.scale_up_threshold {
            self.scale_up(None).await.ok(); // Ignore errors for automatic scaling
        }
        
        Ok(())
    }
    
    /// Get scaling factor calculation using biological transformer-like characteristics
    pub fn calculate_scaling_factor(&self, demand_ratio: f64) -> f64 {
        let capacity_factor = self.resource_state.read().baseline_capacity;
        crate::math::scaling_factor(demand_ratio, capacity_factor)
    }
    
    /// Subscribe to scaling events
    pub fn subscribe_to_scaling_events(&self) -> broadcast::Receiver<ScalingEvent> {
        self.scaling_events.subscribe()
    }
    
    // Private methods
    
    async fn start_demand_monitoring(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let resource_state = Arc::clone(&self.resource_state);
        let demand_monitor = Arc::clone(&self.demand_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Monitor demand patterns and predict scaling needs
                if let Some(prediction) = demand_monitor.predict_demand() {
                    debug!("Demand prediction: {:.2} (current: {:.2})", 
                        prediction, resource_state.read().demand_level);
                }
            }
        });
    }
    
    fn calculate_energy_efficiency(&self, scaling_factor: f64) -> f64 {
        // Energy efficiency decreases with higher scaling factors
        // Based on biological energy trade-offs
        let base_efficiency = 1.0;
        let efficiency_loss = (scaling_factor - 1.0) * 0.15;
        (base_efficiency - efficiency_loss).max(0.1)
    }
}

impl StepDownNode {
    /// Create a new step-down node
    pub fn new(id: String, config: ScalingConfig) -> Self {
        let (scaling_events, _) = broadcast::channel(1000);
        
        Self {
            id,
            config: config.clone(),
            resource_state: Arc::new(RwLock::new(ResourceState::default())),
            scaling_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            scaling_events,
            conservation_monitor: Arc::new(ConservationMonitor::new()),
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the step-down node
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::allocation_failed("StepDown node already running"));
            }
            *running = true;
        }
        
        info!("Starting step-down node: {}", self.id);
        
        // Start conservation monitoring
        self.start_conservation_monitoring().await;
        
        Ok(())
    }
    
    /// Stop the step-down node
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping step-down node: {}", self.id);
        Ok(())
    }
    
    /// Scale down resources to conserve energy
    pub async fn scale_down(&self, target_factor: Option<f64>) -> ResourceResult<()> {
        let mut state = self.resource_state.write();
        
        // Check cooldown period
        if let Some(last_scaling) = state.last_scaling_time {
            let elapsed = Utc::now().signed_duration_since(last_scaling);
            if elapsed < chrono::Duration::from_std(self.config.cooldown_period)? {
                return Err(ResourceError::allocation_failed("Still in cooldown period"));
            }
        }
        
        let previous_factor = state.current_scaling_factor;
        let new_factor = target_factor.unwrap_or_else(|| {
            (previous_factor - self.config.scaling_step_size).max(self.config.min_scaling_factor)
        });
        
        if new_factor >= previous_factor {
            return Err(ResourceError::allocation_failed("Cannot scale down to higher or equal factor"));
        }
        
        // Apply scaling
        state.current_scaling_factor = new_factor;
        state.effective_capacity = state.baseline_capacity * new_factor;
        state.last_scaling_time = Some(Utc::now());
        state.status = NodeStatus::ScalingDown;
        
        // Calculate energy efficiency improvement
        state.energy_efficiency = self.calculate_energy_efficiency(new_factor);
        
        // Record scaling event
        let event = ScalingEvent {
            id: Uuid::new_v4(),
            node_id: self.id.clone(),
            event_type: ScalingEventType::ScaleDown,
            previous_factor,
            new_factor,
            demand_level: state.demand_level,
            utilization_level: state.utilization_level,
            timestamp: Utc::now(),
            reason: "Conservation-based scaling down".to_string(),
        };
        
        // Store in history
        {
            let mut history = self.scaling_history.write();
            history.push_back(event.clone());
            if history.len() > 1000 {
                history.pop_front();
            }
        }
        
        // Broadcast event
        if let Err(e) = self.scaling_events.send(event) {
            warn!("Failed to broadcast scaling event: {}", e);
        }
        
        // Record energy savings
        let energy_saving = EnergySaving {
            id: Uuid::new_v4(),
            conservation_type: ConservationType::CpuScaling,
            energy_saved: (previous_factor - new_factor) / previous_factor,
            duration: Duration::from_secs(3600), // Assume 1 hour duration
            timestamp: Utc::now(),
        };
        
        self.conservation_monitor.record_energy_saving(energy_saving);
        
        info!("Scaled down node {} from {:.2}x to {:.2}x capacity (energy savings: {:.1}%)", 
            self.id, previous_factor, new_factor, ((previous_factor - new_factor) / previous_factor) * 100.0);
        
        Ok(())
    }
    
    /// Enter conservation mode
    pub async fn enter_conservation_mode(&self) -> ResourceResult<()> {
        let mut state = self.resource_state.write();
        
        // Scale down to minimum factor
        let previous_factor = state.current_scaling_factor;
        state.current_scaling_factor = self.config.min_scaling_factor;
        state.effective_capacity = state.baseline_capacity * self.config.min_scaling_factor;
        state.status = NodeStatus::Conservation;
        state.last_scaling_time = Some(Utc::now());
        
        // Maximize energy efficiency
        state.energy_efficiency = 0.95;
        
        info!("Node {} entered conservation mode (scaled from {:.2}x to {:.2}x)", 
            self.id, previous_factor, self.config.min_scaling_factor);
        
        Ok(())
    }
    
    /// Get conservation opportunities
    pub fn get_conservation_opportunities(&self) -> Vec<ConservationOpportunity> {
        self.conservation_monitor.get_opportunities()
    }
    
    /// Get energy savings history
    pub fn get_energy_savings_history(&self) -> Vec<EnergySaving> {
        self.conservation_monitor.get_energy_savings()
    }
    
    /// Update demand level
    pub async fn update_demand(&self, demand_level: f64, utilization: f64) -> ResourceResult<()> {
        {
            let mut state = self.resource_state.write();
            state.demand_level = demand_level.clamp(0.0, 1.0);
            state.utilization_level = utilization.clamp(0.0, 1.0);
        }
        
        // Check for automatic scaling down
        if demand_level < self.config.scale_down_threshold {
            self.scale_down(None).await.ok(); // Ignore errors for automatic scaling
        }
        
        Ok(())
    }
    
    /// Subscribe to scaling events
    pub fn subscribe_to_scaling_events(&self) -> broadcast::Receiver<ScalingEvent> {
        self.scaling_events.subscribe()
    }
    
    // Private methods
    
    async fn start_conservation_monitoring(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let conservation_monitor = Arc::clone(&self.conservation_monitor);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.monitoring_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Identify conservation opportunities
                conservation_monitor.identify_opportunities();
            }
        });
    }
    
    fn calculate_energy_efficiency(&self, scaling_factor: f64) -> f64 {
        // Energy efficiency increases with lower scaling factors
        let base_efficiency = 0.7;
        let efficiency_gain = (1.0 - scaling_factor) * 0.25;
        (base_efficiency + efficiency_gain).min(0.95)
    }
}

impl DemandMonitor {
    fn new(window_size: usize, spike_threshold: f64) -> Self {
        Self {
            demand_history: RwLock::new(VecDeque::with_capacity(window_size)),
            prediction_window: window_size,
            spike_threshold,
        }
    }
    
    fn add_reading(&self, reading: DemandReading) {
        let mut history = self.demand_history.write();
        
        history.push_back(reading);
        
        // Maintain window size
        while history.len() > self.prediction_window {
            history.pop_front();
        }
    }
    
    fn predict_demand(&self) -> Option<f64> {
        let history = self.demand_history.read();
        
        if history.len() < 10 {
            return None;
        }
        
        // Simple moving average prediction
        let recent_readings: Vec<_> = history.iter().rev().take(10).collect();
        let avg_demand = recent_readings.iter().map(|r| r.demand_level).sum::<f64>() / recent_readings.len() as f64;
        
        Some(avg_demand)
    }
}

impl ConservationMonitor {
    fn new() -> Self {
        Self {
            conservation_opportunities: RwLock::new(Vec::new()),
            energy_savings: RwLock::new(VecDeque::with_capacity(1000)),
            conservation_targets: RwLock::new(ConservationTargets::default()),
        }
    }
    
    fn identify_opportunities(&self) {
        let mut opportunities = self.conservation_opportunities.write();
        opportunities.clear();
        
        // Identify CPU scaling opportunities
        opportunities.push(ConservationOpportunity {
            id: Uuid::new_v4(),
            opportunity_type: ConservationType::CpuScaling,
            estimated_savings: 0.2,
            implementation_cost: 0.05,
            priority: 8,
            description: "Reduce CPU frequency during low demand periods".to_string(),
        });
        
        // Identify memory optimization opportunities
        opportunities.push(ConservationOpportunity {
            id: Uuid::new_v4(),
            opportunity_type: ConservationType::MemoryOptimization,
            estimated_savings: 0.15,
            implementation_cost: 0.02,
            priority: 6,
            description: "Optimize memory allocation and garbage collection".to_string(),
        });
    }
    
    fn record_energy_saving(&self, saving: EnergySaving) {
        let mut savings = self.energy_savings.write();
        
        savings.push_back(saving);
        
        // Maintain history size
        while savings.len() > 1000 {
            savings.pop_front();
        }
    }
    
    fn get_opportunities(&self) -> Vec<ConservationOpportunity> {
        self.conservation_opportunities.read().clone()
    }
    
    fn get_energy_savings(&self) -> Vec<EnergySaving> {
        self.energy_savings.read().iter().cloned().collect()
    }
}

/// Thermal Node implementation for resource monitoring
/// 
/// This is separate from the thermal module's ThermalNode to focus on
/// resource-specific thermal monitoring and management.
pub struct ResourceThermalNode {
    /// Node identifier
    pub id: String,
    /// Current thermal state
    thermal_state: Arc<RwLock<ResourceThermalState>>,
    /// Thermal monitoring configuration
    config: ThermalConfig,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Resource-specific thermal state
#[derive(Debug, Default)]
struct ResourceThermalState {
    /// CPU thermal level (0.0-1.0)
    cpu_thermal: f64,
    /// Memory thermal level (0.0-1.0)
    memory_thermal: f64,
    /// Network thermal level (0.0-1.0)
    network_thermal: f64,
    /// Storage thermal level (0.0-1.0)
    storage_thermal: f64,
    /// Combined thermal signature
    thermal_signature: f64,
    /// Last update timestamp
    last_update: Option<DateTime<Utc>>,
}

/// Thermal monitoring configuration
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Monitoring frequency
    pub monitoring_frequency: Duration,
    /// Thermal threshold for scaling
    pub thermal_threshold: f64,
    /// Cooling threshold
    pub cooling_threshold: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            monitoring_frequency: Duration::from_secs(5),
            thermal_threshold: 0.8,
            cooling_threshold: 0.4,
        }
    }
}

impl ResourceThermalNode {
    /// Create a new resource thermal node
    pub fn new(id: String, config: ThermalConfig) -> Self {
        Self {
            id,
            thermal_state: Arc::new(RwLock::new(ResourceThermalState::default())),
            config,
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Update thermal readings
    pub fn update_thermal(&self, cpu: f64, memory: f64, network: f64, storage: f64) -> ResourceResult<()> {
        let mut state = self.thermal_state.write();
        
        state.cpu_thermal = cpu.clamp(0.0, 1.0);
        state.memory_thermal = memory.clamp(0.0, 1.0);
        state.network_thermal = network.clamp(0.0, 1.0);
        state.storage_thermal = storage.clamp(0.0, 1.0);
        
        // Calculate combined thermal signature
        state.thermal_signature = crate::math::thermal_signature(
            state.cpu_thermal,
            state.memory_thermal,
            state.network_thermal,
            state.storage_thermal,
        );
        
        state.last_update = Some(Utc::now());
        
        Ok(())
    }
    
    /// Get current thermal signature
    pub fn get_thermal_signature(&self) -> f64 {
        self.thermal_state.read().thermal_signature
    }
    
    /// Check if thermal scaling is needed
    pub fn needs_thermal_scaling(&self) -> Option<crate::compartments::ScalingDirection> {
        let state = self.thermal_state.read();
        
        if state.thermal_signature > self.config.thermal_threshold {
            Some(crate::compartments::ScalingDirection::Down) // Scale down to cool
        } else if state.thermal_signature < self.config.cooling_threshold {
            Some(crate::compartments::ScalingDirection::Up) // Scale up when cool
        } else {
            None
        }
    }
}

/// Resource Node trait for common resource management operations
pub trait ResourceNode {
    /// Get node identifier
    fn id(&self) -> &str;
    
    /// Get current resource state
    fn resource_state(&self) -> ResourceState;
    
    /// Update resource metrics
    fn update_metrics(&self, metrics: ResourceMetrics) -> ResourceResult<()>;
    
    /// Get performance metrics
    fn performance_metrics(&self) -> PerformanceMetrics;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_step_up_node_creation() {
        let config = ScalingConfig::default();
        let node = StepUpNode::new("step-up-test".to_string(), config);
        
        assert_eq!(node.id, "step-up-test");
        
        let state = node.get_resource_state();
        assert_eq!(state.current_scaling_factor, 1.0);
        assert_eq!(state.status, NodeStatus::Active);
    }
    
    #[tokio::test]
    async fn test_step_down_node_creation() {
        let config = ScalingConfig::default();
        let node = StepDownNode::new("step-down-test".to_string(), config);
        
        assert_eq!(node.id, "step-down-test");
        
        let state = node.resource_state.read().clone();
        assert_eq!(state.current_scaling_factor, 1.0);
        assert_eq!(state.status, NodeStatus::Active);
    }
    
    #[tokio::test]
    async fn test_step_up_scaling() {
        let config = ScalingConfig::default();
        let node = StepUpNode::new("step-up-test".to_string(), config);
        
        // Initial scaling up
        node.scale_up(Some(1.5)).await.unwrap();
        
        let state = node.get_resource_state();
        assert_eq!(state.current_scaling_factor, 1.5);
        assert!(state.last_scaling_time.is_some());
    }
    
    #[tokio::test]
    async fn test_step_down_scaling() {
        let config = ScalingConfig::default();
        let node = StepDownNode::new("step-down-test".to_string(), config);
        
        // Start with higher scaling factor
        {
            let mut state = node.resource_state.write();
            state.current_scaling_factor = 1.5;
            state.effective_capacity = 1.5;
        }
        
        // Scale down
        node.scale_down(Some(0.8)).await.unwrap();
        
        let state = node.resource_state.read().clone();
        assert_eq!(state.current_scaling_factor, 0.8);
        assert!(state.last_scaling_time.is_some());
    }
    
    #[tokio::test]
    async fn test_conservation_mode() {
        let config = ScalingConfig::default();
        let node = StepDownNode::new("conservation-test".to_string(), config.clone());
        
        node.enter_conservation_mode().await.unwrap();
        
        let state = node.resource_state.read().clone();
        assert_eq!(state.current_scaling_factor, config.min_scaling_factor);
        assert_eq!(state.status, NodeStatus::Conservation);
        assert!(state.energy_efficiency > 0.9);
    }
    
    #[test]
    fn test_scaling_factor_calculation() {
        let config = ScalingConfig::default();
        let node = StepUpNode::new("calc-test".to_string(), config);
        
        let factor = node.calculate_scaling_factor(0.8);
        assert!(factor > 0.0);
    }
    
    #[test]
    fn test_thermal_node() {
        let config = ThermalConfig::default();
        let thermal_node = ResourceThermalNode::new("thermal-test".to_string(), config);
        
        thermal_node.update_thermal(0.8, 0.7, 0.6, 0.5).unwrap();
        
        let signature = thermal_node.get_thermal_signature();
        assert!(signature > 0.0);
        
        let scaling_needed = thermal_node.needs_thermal_scaling();
        assert!(scaling_needed.is_some());
    }
    
    #[test]
    fn test_conservation_opportunities() {
        let monitor = ConservationMonitor::new();
        
        monitor.identify_opportunities();
        let opportunities = monitor.get_opportunities();
        
        assert!(!opportunities.is_empty());
        assert!(opportunities.iter().any(|o| matches!(o.opportunity_type, ConservationType::CpuScaling)));
    }
    
    #[test]
    fn test_demand_monitoring() {
        let monitor = DemandMonitor::new(10, 0.9);
        
        // Add some readings
        for i in 0..5 {
            let reading = DemandReading {
                timestamp: Utc::now(),
                demand_level: 0.5 + i as f64 * 0.1,
                utilization: 0.6,
                response_time: Duration::from_millis(100),
            };
            monitor.add_reading(reading);
        }
        
        // Prediction should be None with too few readings
        assert!(monitor.predict_demand().is_none());
    }
}