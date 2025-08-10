//! Compartmentalization System for Resource Management
//! 
//! Implements biological compartmentalization inspired by ant colony division of labor
//! for dynamic resource allocation across specialized functional units.

use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{CompartmentMetrics, PerformanceMetrics};
use crate::allocation::{ResourceProvider, AllocationStrategy};
use crate::thermal::ThermalSignature;

/// Types of compartments based on biological specialization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompartmentType {
    /// Training compartment - optimized for gradient computation and model parameter updates
    Training,
    /// Inference compartment - specialized for low-latency prediction serving
    Inference,
    /// Storage compartment - manages distributed data storage and retrieval
    Storage,
    /// Communication compartment - handles inter-node messaging and coordination
    Communication,
    /// Security compartment - performs validation, encryption, and threat detection
    Security,
    /// General purpose compartment for flexible workloads
    General,
}

impl CompartmentType {
    /// Get compartment priority for resource allocation
    pub fn priority(&self) -> u8 {
        match self {
            CompartmentType::Security => 10,
            CompartmentType::Communication => 8,
            CompartmentType::Training => 7,
            CompartmentType::Inference => 6,
            CompartmentType::Storage => 5,
            CompartmentType::General => 3,
        }
    }
    
    /// Get default resource allocation percentage
    pub fn default_allocation(&self) -> f64 {
        match self {
            CompartmentType::Training => 0.4,
            CompartmentType::Inference => 0.25,
            CompartmentType::Storage => 0.15,
            CompartmentType::Communication => 0.1,
            CompartmentType::Security => 0.05,
            CompartmentType::General => 0.05,
        }
    }
    
    /// Check if compartment can be scaled dynamically
    pub fn is_scalable(&self) -> bool {
        match self {
            CompartmentType::Security => false, // Security compartment maintains minimum allocation
            _ => true,
        }
    }
    
    /// Get compartment description
    pub fn description(&self) -> &'static str {
        match self {
            CompartmentType::Training => "Machine learning model training and parameter optimization",
            CompartmentType::Inference => "Real-time prediction serving and model inference",
            CompartmentType::Storage => "Distributed data storage, retrieval, and management",
            CompartmentType::Communication => "Inter-node messaging, coordination, and networking",
            CompartmentType::Security => "Validation, encryption, threat detection, and security monitoring",
            CompartmentType::General => "General purpose computing for flexible workloads",
        }
    }
}

impl std::fmt::Display for CompartmentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompartmentType::Training => write!(f, "TRAINING"),
            CompartmentType::Inference => write!(f, "INFERENCE"),
            CompartmentType::Storage => write!(f, "STORAGE"),
            CompartmentType::Communication => write!(f, "COMMUNICATION"),
            CompartmentType::Security => write!(f, "SECURITY"),
            CompartmentType::General => write!(f, "GENERAL"),
        }
    }
}

/// Individual compartment within a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compartment {
    /// Unique compartment identifier
    pub id: Uuid,
    /// Parent node identifier
    pub node_id: String,
    /// Compartment type
    pub compartment_type: CompartmentType,
    /// Allocated CPU resources (0.0-1.0)
    pub cpu_allocation: f64,
    /// Allocated memory resources (0.0-1.0)
    pub memory_allocation: f64,
    /// Allocated network bandwidth (0.0-1.0)
    pub network_allocation: f64,
    /// Allocated storage (0.0-1.0)
    pub storage_allocation: f64,
    /// Current utilization level (0.0-1.0)
    pub utilization_level: f64,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Custom pricing per compartment
    pub pricing: CompartmentPricing,
    /// Quality of service level
    pub qos_level: QoSLevel,
    /// Current thermal signature
    pub thermal_signature: Option<ThermalSignature>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
    /// Compartment status
    pub status: CompartmentStatus,
}

/// Compartment pricing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentPricing {
    /// Price per CPU unit per hour
    pub cpu_price_per_hour: f64,
    /// Price per memory unit per hour
    pub memory_price_per_hour: f64,
    /// Price per network unit per hour
    pub network_price_per_hour: f64,
    /// Price per storage unit per hour
    pub storage_price_per_hour: f64,
    /// Quality multiplier for premium services
    pub quality_multiplier: f64,
}

impl Default for CompartmentPricing {
    fn default() -> Self {
        Self {
            cpu_price_per_hour: 0.1,
            memory_price_per_hour: 0.05,
            network_price_per_hour: 0.02,
            storage_price_per_hour: 0.01,
            quality_multiplier: 1.0,
        }
    }
}

impl CompartmentPricing {
    /// Calculate hourly cost for compartment
    pub fn calculate_hourly_cost(&self, compartment: &Compartment) -> f64 {
        let cpu_cost = compartment.cpu_allocation * self.cpu_price_per_hour;
        let memory_cost = compartment.memory_allocation * self.memory_price_per_hour;
        let network_cost = compartment.network_allocation * self.network_price_per_hour;
        let storage_cost = compartment.storage_allocation * self.storage_price_per_hour;
        
        (cpu_cost + memory_cost + network_cost + storage_cost) * self.quality_multiplier
    }
}

/// Quality of service levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum QoSLevel {
    /// Basic service level
    Basic = 1,
    /// Standard service level
    Standard = 2,
    /// Premium service level
    Premium = 3,
    /// Enterprise service level
    Enterprise = 4,
}

impl QoSLevel {
    /// Get resource allocation priority
    pub fn resource_priority(&self) -> f64 {
        match self {
            QoSLevel::Basic => 1.0,
            QoSLevel::Standard => 1.2,
            QoSLevel::Premium => 1.5,
            QoSLevel::Enterprise => 2.0,
        }
    }
    
    /// Get service level agreement guarantees
    pub fn sla_guarantees(&self) -> SLAGuarantees {
        match self {
            QoSLevel::Basic => SLAGuarantees {
                uptime_percentage: 95.0,
                max_response_time_ms: 1000,
                availability_percentage: 98.0,
            },
            QoSLevel::Standard => SLAGuarantees {
                uptime_percentage: 99.0,
                max_response_time_ms: 500,
                availability_percentage: 99.5,
            },
            QoSLevel::Premium => SLAGuarantees {
                uptime_percentage: 99.5,
                max_response_time_ms: 200,
                availability_percentage: 99.9,
            },
            QoSLevel::Enterprise => SLAGuarantees {
                uptime_percentage: 99.99,
                max_response_time_ms: 100,
                availability_percentage: 99.99,
            },
        }
    }
}

/// Service Level Agreement guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLAGuarantees {
    /// Minimum uptime percentage
    pub uptime_percentage: f64,
    /// Maximum response time in milliseconds
    pub max_response_time_ms: u64,
    /// Service availability percentage
    pub availability_percentage: f64,
}

/// Compartment operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompartmentStatus {
    /// Compartment is initializing
    Initializing,
    /// Compartment is active and ready
    Active,
    /// Compartment is scaling resources
    Scaling,
    /// Compartment is under maintenance
    Maintenance,
    /// Compartment is shutting down
    ShuttingDown,
    /// Compartment has failed
    Failed,
}

impl Compartment {
    /// Create a new compartment
    pub fn new(
        node_id: String,
        compartment_type: CompartmentType,
        qos_level: QoSLevel,
    ) -> Self {
        let default_allocation = compartment_type.default_allocation();
        
        Self {
            id: Uuid::new_v4(),
            node_id,
            compartment_type,
            cpu_allocation: default_allocation,
            memory_allocation: default_allocation,
            network_allocation: default_allocation * 0.5, // Network typically needs less
            storage_allocation: default_allocation * 0.3, // Storage allocation varies
            utilization_level: 0.0,
            performance_metrics: PerformanceMetrics::default(),
            pricing: CompartmentPricing::default(),
            qos_level,
            thermal_signature: None,
            created_at: Utc::now(),
            last_updated: Utc::now(),
            status: CompartmentStatus::Initializing,
        }
    }
    
    /// Calculate total resource allocation
    pub fn total_allocation(&self) -> f64 {
        (self.cpu_allocation + self.memory_allocation + self.network_allocation + self.storage_allocation) / 4.0
    }
    
    /// Calculate resource efficiency
    pub fn efficiency(&self) -> f64 {
        if self.total_allocation() == 0.0 {
            0.0
        } else {
            self.utilization_level / self.total_allocation()
        }
    }
    
    /// Check if compartment needs scaling
    pub fn needs_scaling(&self) -> Option<ScalingDirection> {
        if !self.compartment_type.is_scalable() {
            return None;
        }
        
        // Scale up if utilization is consistently high
        if self.utilization_level > 0.85 {
            Some(ScalingDirection::Up)
        }
        // Scale down if utilization is consistently low
        else if self.utilization_level < 0.3 && self.total_allocation() > 0.1 {
            Some(ScalingDirection::Down)
        } else {
            None
        }
    }
    
    /// Update compartment utilization
    pub fn update_utilization(&mut self, new_utilization: f64) {
        self.utilization_level = new_utilization.clamp(0.0, 1.0);
        self.last_updated = Utc::now();
        
        // Update thermal signature if high utilization
        if new_utilization > 0.8 {
            self.thermal_signature = Some(ThermalSignature::new(
                self.node_id.clone(),
                new_utilization,
                self.memory_allocation,
                self.network_allocation,
                self.storage_allocation,
                format!("{:?}", self.compartment_type),
            ));
        }
    }
    
    /// Scale compartment resources
    pub fn scale(&mut self, direction: ScalingDirection, factor: f64) -> ResourceResult<()> {
        if !self.compartment_type.is_scalable() {
            return Err(ResourceError::compartment_error(
                format!("Compartment type {:?} is not scalable", self.compartment_type)
            ));
        }
        
        let scaling_factor = match direction {
            ScalingDirection::Up => 1.0 + factor,
            ScalingDirection::Down => 1.0 - factor,
        };
        
        // Apply scaling with bounds checking
        self.cpu_allocation = (self.cpu_allocation * scaling_factor).clamp(0.01, 1.0);
        self.memory_allocation = (self.memory_allocation * scaling_factor).clamp(0.01, 1.0);
        self.network_allocation = (self.network_allocation * scaling_factor).clamp(0.005, 1.0);
        self.storage_allocation = (self.storage_allocation * scaling_factor).clamp(0.005, 1.0);
        
        self.last_updated = Utc::now();
        self.status = CompartmentStatus::Scaling;
        
        info!("Scaled compartment {} ({:?}) {} by factor {:.2}", 
            self.id, self.compartment_type, direction, factor);
        
        Ok(())
    }
    
    /// Get hourly operational cost
    pub fn hourly_cost(&self) -> f64 {
        self.pricing.calculate_hourly_cost(self)
    }
}

/// Scaling direction for compartments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingDirection {
    /// Scale up resources
    Up,
    /// Scale down resources
    Down,
}

impl std::fmt::Display for ScalingDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalingDirection::Up => write!(f, "UP"),
            ScalingDirection::Down => write!(f, "DOWN"),
        }
    }
}

/// Compartment Manager - Manages all compartments within a node
pub struct CompartmentManager {
    /// Node identifier
    pub node_id: String,
    /// Active compartments by type
    compartments: Arc<DashMap<CompartmentType, Compartment>>,
    /// Compartment metrics
    metrics: Arc<RwLock<CompartmentMetrics>>,
    /// Manager configuration
    config: CompartmentConfig,
    /// Scaling event broadcaster
    scaling_events: broadcast::Sender<ScalingEvent>,
    /// Resource constraints
    resource_constraints: Arc<RwLock<ResourceConstraints>>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Compartment manager configuration
#[derive(Debug, Clone)]
pub struct CompartmentConfig {
    /// Enable dynamic scaling
    pub enable_dynamic_scaling: bool,
    /// Scaling check interval
    pub scaling_interval: Duration,
    /// Default scaling step size
    pub default_scaling_step: f64,
    /// Minimum compartment allocation
    pub min_compartment_allocation: f64,
    /// Maximum compartment allocation
    pub max_compartment_allocation: f64,
    /// Resource rebalancing threshold
    pub rebalancing_threshold: f64,
}

impl Default for CompartmentConfig {
    fn default() -> Self {
        Self {
            enable_dynamic_scaling: true,
            scaling_interval: Duration::from_secs(30),
            default_scaling_step: crate::constants::DEFAULT_SCALING_STEP,
            min_compartment_allocation: 0.01,
            max_compartment_allocation: 0.8,
            rebalancing_threshold: 0.1,
        }
    }
}

/// Resource constraints for the node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    /// Maximum CPU allocation across all compartments
    pub max_cpu: f64,
    /// Maximum memory allocation across all compartments
    pub max_memory: f64,
    /// Maximum network allocation across all compartments
    pub max_network: f64,
    /// Maximum storage allocation across all compartments
    pub max_storage: f64,
}

impl Default for ResourceConstraints {
    fn default() -> Self {
        Self {
            max_cpu: 1.0,
            max_memory: 1.0,
            max_network: 1.0,
            max_storage: 1.0,
        }
    }
}

/// Scaling event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    /// Event identifier
    pub id: Uuid,
    /// Compartment that was scaled
    pub compartment_id: Uuid,
    /// Compartment type
    pub compartment_type: CompartmentType,
    /// Scaling direction
    pub direction: ScalingDirection,
    /// Scaling factor applied
    pub factor: f64,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Reason for scaling
    pub reason: String,
}

impl CompartmentManager {
    /// Create a new compartment manager
    pub fn new(node_id: String, config: CompartmentConfig) -> Self {
        let (scaling_events, _) = broadcast::channel(1000);
        
        Self {
            node_id,
            compartments: Arc::new(DashMap::new()),
            metrics: Arc::new(RwLock::new(CompartmentMetrics::default())),
            config,
            scaling_events,
            resource_constraints: Arc::new(RwLock::new(ResourceConstraints::default())),
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the compartment manager
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::compartment_error("Compartment manager already running"));
            }
            *running = true;
        }
        
        info!("Starting compartment manager for node: {}", self.node_id);
        
        // Initialize default compartments
        self.initialize_default_compartments().await?;
        
        // Start scaling monitoring if enabled
        if self.config.enable_dynamic_scaling {
            self.start_scaling_monitor().await;
        }
        
        Ok(())
    }
    
    /// Stop the compartment manager
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping compartment manager for node: {}", self.node_id);
        Ok(())
    }
    
    /// Create a new compartment
    pub async fn create_compartment(
        &self, 
        compartment_type: CompartmentType,
        qos_level: QoSLevel,
    ) -> ResourceResult<Uuid> {
        // Check if compartment type already exists
        if self.compartments.contains_key(&compartment_type) {
            return Err(ResourceError::compartment_error(
                format!("Compartment type {:?} already exists", compartment_type)
            ));
        }
        
        let compartment = Compartment::new(self.node_id.clone(), compartment_type, qos_level);
        let compartment_id = compartment.id;
        
        // Validate resource constraints
        if !self.validate_resource_allocation(&compartment).await? {
            return Err(ResourceError::compartment_error(
                "Insufficient resources for new compartment"
            ));
        }
        
        info!("Creating compartment: {:?} with QoS level {:?}", compartment_type, qos_level);
        self.compartments.insert(compartment_type, compartment);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_compartments = self.compartments.len();
        }
        
        Ok(compartment_id)
    }
    
    /// Remove a compartment
    pub async fn remove_compartment(&self, compartment_type: CompartmentType) -> ResourceResult<()> {
        if let Some((_, mut compartment)) = self.compartments.remove(&compartment_type) {
            compartment.status = CompartmentStatus::ShuttingDown;
            
            info!("Removing compartment: {:?}", compartment_type);
            
            // Update metrics
            {
                let mut metrics = self.metrics.write();
                metrics.total_compartments = self.compartments.len();
            }
            
            Ok(())
        } else {
            Err(ResourceError::compartment_error(
                format!("Compartment type {:?} not found", compartment_type)
            ))
        }
    }
    
    /// Get compartment by type
    pub fn get_compartment(&self, compartment_type: &CompartmentType) -> Option<Compartment> {
        self.compartments.get(compartment_type).map(|entry| entry.value().clone())
    }
    
    /// Get all compartments
    pub fn get_all_compartments(&self) -> Vec<Compartment> {
        self.compartments.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Update compartment utilization
    pub async fn update_compartment_utilization(
        &self, 
        compartment_type: CompartmentType, 
        utilization: f64
    ) -> ResourceResult<()> {
        if let Some(mut compartment_entry) = self.compartments.get_mut(&compartment_type) {
            compartment_entry.value_mut().update_utilization(utilization);
            
            // Check if scaling is needed
            if let Some(direction) = compartment_entry.value().needs_scaling() {
                self.scale_compartment(compartment_type, direction, self.config.default_scaling_step).await?;
            }
            
            Ok(())
        } else {
            Err(ResourceError::compartment_error(
                format!("Compartment type {:?} not found", compartment_type)
            ))
        }
    }
    
    /// Scale a compartment
    pub async fn scale_compartment(
        &self,
        compartment_type: CompartmentType,
        direction: ScalingDirection,
        factor: f64,
    ) -> ResourceResult<()> {
        if let Some(mut compartment_entry) = self.compartments.get_mut(&compartment_type) {
            let compartment = compartment_entry.value_mut();
            
            // Validate scaling constraints
            if !self.validate_scaling(compartment, direction, factor).await? {
                return Err(ResourceError::compartment_error(
                    "Scaling would violate resource constraints"
                ));
            }
            
            // Apply scaling
            compartment.scale(direction, factor)?;
            
            // Broadcast scaling event
            let scaling_event = ScalingEvent {
                id: Uuid::new_v4(),
                compartment_id: compartment.id,
                compartment_type,
                direction,
                factor,
                timestamp: Utc::now(),
                reason: "Dynamic utilization-based scaling".to_string(),
            };
            
            if let Err(e) = self.scaling_events.send(scaling_event) {
                warn!("Failed to broadcast scaling event: {}", e);
            }
            
            // Update metrics
            {
                let mut metrics = self.metrics.write();
                match direction {
                    ScalingDirection::Up => metrics.scale_up_events += 1,
                    ScalingDirection::Down => metrics.scale_down_events += 1,
                }
            }
            
            Ok(())
        } else {
            Err(ResourceError::compartment_error(
                format!("Compartment type {:?} not found", compartment_type)
            ))
        }
    }
    
    /// Rebalance resources across compartments
    pub async fn rebalance_resources(&self) -> ResourceResult<()> {
        info!("Rebalancing resources across compartments");
        
        let compartments = self.get_all_compartments();
        if compartments.is_empty() {
            return Ok(());
        }
        
        // Calculate total utilization and identify over/under utilized compartments
        let mut over_utilized = Vec::new();
        let mut under_utilized = Vec::new();
        
        for compartment in &compartments {
            let efficiency = compartment.efficiency();
            
            if efficiency > 1.2 {
                over_utilized.push(compartment.compartment_type);
            } else if efficiency < 0.5 && compartment.total_allocation() > 0.1 {
                under_utilized.push(compartment.compartment_type);
            }
        }
        
        // Scale up over-utilized compartments
        for compartment_type in over_utilized {
            self.scale_compartment(
                compartment_type, 
                ScalingDirection::Up, 
                self.config.default_scaling_step
            ).await?;
        }
        
        // Scale down under-utilized compartments
        for compartment_type in under_utilized {
            self.scale_compartment(
                compartment_type, 
                ScalingDirection::Down, 
                self.config.default_scaling_step
            ).await?;
        }
        
        Ok(())
    }
    
    /// Get resource utilization summary
    pub fn get_resource_summary(&self) -> ResourceSummary {
        let compartments = self.get_all_compartments();
        
        if compartments.is_empty() {
            return ResourceSummary::default();
        }
        
        let total_cpu: f64 = compartments.iter().map(|c| c.cpu_allocation).sum();
        let total_memory: f64 = compartments.iter().map(|c| c.memory_allocation).sum();
        let total_network: f64 = compartments.iter().map(|c| c.network_allocation).sum();
        let total_storage: f64 = compartments.iter().map(|c| c.storage_allocation).sum();
        
        let avg_utilization: f64 = compartments.iter()
            .map(|c| c.utilization_level)
            .sum::<f64>() / compartments.len() as f64;
        
        let total_hourly_cost: f64 = compartments.iter()
            .map(|c| c.hourly_cost())
            .sum();
        
        ResourceSummary {
            total_compartments: compartments.len(),
            total_cpu_allocated: total_cpu,
            total_memory_allocated: total_memory,
            total_network_allocated: total_network,
            total_storage_allocated: total_storage,
            average_utilization: avg_utilization,
            total_hourly_cost,
            timestamp: Utc::now(),
        }
    }
    
    /// Get compartment metrics
    pub fn get_metrics(&self) -> CompartmentMetrics {
        let mut metrics = self.metrics.write();
        metrics.total_compartments = self.compartments.len();
        metrics.clone()
    }
    
    /// Subscribe to scaling events
    pub fn subscribe_to_scaling_events(&self) -> broadcast::Receiver<ScalingEvent> {
        self.scaling_events.subscribe()
    }
    
    /// Set resource constraints
    pub async fn set_resource_constraints(&self, constraints: ResourceConstraints) {
        *self.resource_constraints.write() = constraints;
    }
    
    /// Get current resource constraints
    pub fn get_resource_constraints(&self) -> ResourceConstraints {
        self.resource_constraints.read().clone()
    }
    
    // Private methods
    
    async fn initialize_default_compartments(&self) -> ResourceResult<()> {
        let default_compartments = vec![
            (CompartmentType::Training, QoSLevel::Standard),
            (CompartmentType::Inference, QoSLevel::Premium),
            (CompartmentType::Storage, QoSLevel::Standard),
            (CompartmentType::Communication, QoSLevel::Standard),
            (CompartmentType::Security, QoSLevel::Enterprise),
        ];
        
        for (compartment_type, qos_level) in default_compartments {
            self.create_compartment(compartment_type, qos_level).await?;
        }
        
        Ok(())
    }
    
    async fn start_scaling_monitor(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let compartments = Arc::clone(&self.compartments);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.scaling_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Check compartments for scaling needs
                for compartment_entry in compartments.iter() {
                    let compartment = compartment_entry.value();
                    
                    if let Some(direction) = compartment.needs_scaling() {
                        debug!("Compartment {:?} needs scaling {:?} (utilization: {:.2}%)", 
                            compartment.compartment_type, 
                            direction, 
                            compartment.utilization_level * 100.0
                        );
                    }
                }
            }
        });
    }
    
    async fn validate_resource_allocation(&self, compartment: &Compartment) -> ResourceResult<bool> {
        let constraints = self.resource_constraints.read();
        let current_totals = self.get_current_resource_totals();
        
        let new_cpu_total = current_totals.0 + compartment.cpu_allocation;
        let new_memory_total = current_totals.1 + compartment.memory_allocation;
        let new_network_total = current_totals.2 + compartment.network_allocation;
        let new_storage_total = current_totals.3 + compartment.storage_allocation;
        
        Ok(new_cpu_total <= constraints.max_cpu &&
           new_memory_total <= constraints.max_memory &&
           new_network_total <= constraints.max_network &&
           new_storage_total <= constraints.max_storage)
    }
    
    async fn validate_scaling(
        &self, 
        compartment: &Compartment, 
        direction: ScalingDirection, 
        factor: f64
    ) -> ResourceResult<bool> {
        let constraints = self.resource_constraints.read();
        
        let scaling_factor = match direction {
            ScalingDirection::Up => 1.0 + factor,
            ScalingDirection::Down => 1.0 - factor,
        };
        
        let new_cpu = compartment.cpu_allocation * scaling_factor;
        let new_memory = compartment.memory_allocation * scaling_factor;
        
        // Check bounds
        let within_bounds = new_cpu <= constraints.max_cpu &&
                           new_memory <= constraints.max_memory &&
                           new_cpu >= self.config.min_compartment_allocation &&
                           new_memory >= self.config.min_compartment_allocation;
        
        Ok(within_bounds)
    }
    
    fn get_current_resource_totals(&self) -> (f64, f64, f64, f64) {
        let compartments = self.get_all_compartments();
        
        let cpu_total: f64 = compartments.iter().map(|c| c.cpu_allocation).sum();
        let memory_total: f64 = compartments.iter().map(|c| c.memory_allocation).sum();
        let network_total: f64 = compartments.iter().map(|c| c.network_allocation).sum();
        let storage_total: f64 = compartments.iter().map(|c| c.storage_allocation).sum();
        
        (cpu_total, memory_total, network_total, storage_total)
    }
}

/// Resource summary across all compartments
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResourceSummary {
    /// Total number of compartments
    pub total_compartments: usize,
    /// Total CPU allocated across compartments
    pub total_cpu_allocated: f64,
    /// Total memory allocated across compartments
    pub total_memory_allocated: f64,
    /// Total network allocated across compartments
    pub total_network_allocated: f64,
    /// Total storage allocated across compartments
    pub total_storage_allocated: f64,
    /// Average utilization across compartments
    pub average_utilization: f64,
    /// Total hourly operational cost
    pub total_hourly_cost: f64,
    /// Summary timestamp
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compartment_type_properties() {
        assert_eq!(CompartmentType::Security.priority(), 10);
        assert_eq!(CompartmentType::Training.priority(), 7);
        assert_eq!(CompartmentType::General.priority(), 3);
        
        assert_eq!(CompartmentType::Training.default_allocation(), 0.4);
        assert_eq!(CompartmentType::Security.default_allocation(), 0.05);
        
        assert!(!CompartmentType::Security.is_scalable());
        assert!(CompartmentType::Training.is_scalable());
    }
    
    #[test]
    fn test_compartment_creation() {
        let compartment = Compartment::new(
            "test-node".to_string(),
            CompartmentType::Training,
            QoSLevel::Premium,
        );
        
        assert_eq!(compartment.compartment_type, CompartmentType::Training);
        assert_eq!(compartment.qos_level, QoSLevel::Premium);
        assert_eq!(compartment.status, CompartmentStatus::Initializing);
        assert!(compartment.total_allocation() > 0.0);
    }
    
    #[test]
    fn test_compartment_scaling() {
        let mut compartment = Compartment::new(
            "test-node".to_string(),
            CompartmentType::Inference,
            QoSLevel::Standard,
        );
        
        let initial_allocation = compartment.total_allocation();
        
        compartment.scale(ScalingDirection::Up, 0.2).unwrap();
        let scaled_allocation = compartment.total_allocation();
        
        assert!(scaled_allocation > initial_allocation);
        assert_eq!(compartment.status, CompartmentStatus::Scaling);
    }
    
    #[test]
    fn test_qos_level_properties() {
        assert!(QoSLevel::Enterprise.resource_priority() > QoSLevel::Basic.resource_priority());
        
        let enterprise_sla = QoSLevel::Enterprise.sla_guarantees();
        let basic_sla = QoSLevel::Basic.sla_guarantees();
        
        assert!(enterprise_sla.uptime_percentage > basic_sla.uptime_percentage);
        assert!(enterprise_sla.max_response_time_ms < basic_sla.max_response_time_ms);
    }
    
    #[tokio::test]
    async fn test_compartment_manager_creation() {
        let config = CompartmentConfig::default();
        let manager = CompartmentManager::new("test-node".to_string(), config);
        
        assert_eq!(manager.node_id, "test-node");
        assert_eq!(manager.get_all_compartments().len(), 0);
    }
    
    #[tokio::test]
    async fn test_compartment_creation_and_retrieval() {
        let config = CompartmentConfig::default();
        let manager = CompartmentManager::new("test-node".to_string(), config);
        
        let compartment_id = manager.create_compartment(
            CompartmentType::Training,
            QoSLevel::Premium,
        ).await.unwrap();
        
        assert!(!compartment_id.is_nil());
        
        let compartment = manager.get_compartment(&CompartmentType::Training);
        assert!(compartment.is_some());
        assert_eq!(compartment.unwrap().qos_level, QoSLevel::Premium);
    }
    
    #[test]
    fn test_compartment_pricing() {
        let compartment = Compartment::new(
            "test-node".to_string(),
            CompartmentType::Training,
            QoSLevel::Premium,
        );
        
        let cost = compartment.hourly_cost();
        assert!(cost > 0.0);
    }
    
    #[test]
    fn test_compartment_needs_scaling() {
        let mut compartment = Compartment::new(
            "test-node".to_string(),
            CompartmentType::Inference,
            QoSLevel::Standard,
        );
        
        // Low utilization - should need scaling down
        compartment.update_utilization(0.2);
        assert_eq!(compartment.needs_scaling(), Some(ScalingDirection::Down));
        
        // High utilization - should need scaling up
        compartment.update_utilization(0.9);
        assert_eq!(compartment.needs_scaling(), Some(ScalingDirection::Up));
        
        // Medium utilization - no scaling needed
        compartment.update_utilization(0.6);
        assert_eq!(compartment.needs_scaling(), None);
    }
    
    #[tokio::test]
    async fn test_resource_constraints() {
        let config = CompartmentConfig::default();
        let manager = CompartmentManager::new("test-node".to_string(), config);
        
        let constraints = ResourceConstraints {
            max_cpu: 0.8,
            max_memory: 0.8,
            max_network: 0.8,
            max_storage: 0.8,
        };
        
        manager.set_resource_constraints(constraints.clone()).await;
        
        let retrieved_constraints = manager.get_resource_constraints();
        assert_eq!(retrieved_constraints.max_cpu, 0.8);
    }
}