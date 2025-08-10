//! Resource Management Metrics and Monitoring
//! 
//! Comprehensive metrics collection for biological resource management system
//! including performance tracking, social metrics, and crisis monitoring.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Performance metrics for individual nodes and compartments
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Number of tasks completed successfully
    pub tasks_completed: u64,
    /// Number of tasks that failed
    pub tasks_failed: u64,
    /// Average task completion time in milliseconds
    pub avg_completion_time_ms: f64,
    /// Current CPU utilization (0.0-1.0)
    pub cpu_utilization: f64,
    /// Current memory utilization (0.0-1.0)
    pub memory_utilization: f64,
    /// Current network utilization (0.0-1.0)
    pub network_utilization: f64,
    /// Current storage utilization (0.0-1.0)
    pub storage_utilization: f64,
    /// Energy efficiency rating (0.0-1.0)
    pub energy_efficiency: f64,
    /// Throughput in tasks per second
    pub throughput: f64,
    /// Error rate percentage (0.0-1.0)
    pub error_rate: f64,
    /// Availability percentage (0.0-1.0)
    pub availability: f64,
    /// Last metrics update timestamp
    pub last_updated: DateTime<Utc>,
}

impl PerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Update metrics with new values
    pub fn update(
        &mut self,
        tasks_completed: u64,
        tasks_failed: u64,
        completion_time_ms: f64,
        cpu: f64,
        memory: f64,
        network: f64,
        storage: f64,
    ) {
        self.tasks_completed = tasks_completed;
        self.tasks_failed = tasks_failed;
        self.avg_completion_time_ms = completion_time_ms;
        self.cpu_utilization = cpu.clamp(0.0, 1.0);
        self.memory_utilization = memory.clamp(0.0, 1.0);
        self.network_utilization = network.clamp(0.0, 1.0);
        self.storage_utilization = storage.clamp(0.0, 1.0);
        self.last_updated = Utc::now();
        
        // Calculate derived metrics
        self.calculate_derived_metrics();
    }
    
    /// Calculate derived performance metrics
    fn calculate_derived_metrics(&mut self) {
        // Calculate success rate
        let total_tasks = self.tasks_completed + self.tasks_failed;
        if total_tasks > 0 {
            self.error_rate = self.tasks_failed as f64 / total_tasks as f64;
            self.availability = self.tasks_completed as f64 / total_tasks as f64;
        }
        
        // Calculate throughput (simplified - tasks per hour)
        if self.avg_completion_time_ms > 0.0 {
            self.throughput = 3600000.0 / self.avg_completion_time_ms; // tasks per hour
        }
        
        // Calculate energy efficiency based on resource utilization
        let avg_utilization = (self.cpu_utilization + self.memory_utilization + 
                              self.network_utilization + self.storage_utilization) / 4.0;
        self.energy_efficiency = crate::math::energy_efficiency_calculation(
            self.throughput,
            avg_utilization * 100.0, // Normalize to energy units
            avg_utilization * 0.1,   // Thermal overhead
            1.0 - avg_utilization,   // Idle ratio
        );
    }
    
    /// Get overall performance score (0.0-1.0)
    pub fn performance_score(&self) -> f64 {
        let availability_weight = 0.4;
        let efficiency_weight = 0.3;
        let throughput_weight = 0.2;
        let error_weight = 0.1;
        
        let normalized_throughput = (self.throughput / 1000.0).min(1.0); // Normalize throughput
        let error_penalty = 1.0 - self.error_rate;
        
        self.availability * availability_weight +
        self.energy_efficiency * efficiency_weight +
        normalized_throughput * throughput_weight +
        error_penalty * error_weight
    }
}

/// Resource utilization metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Total CPU capacity
    pub total_cpu_capacity: f64,
    /// Used CPU capacity
    pub used_cpu_capacity: f64,
    /// Total memory capacity
    pub total_memory_capacity: f64,
    /// Used memory capacity
    pub used_memory_capacity: f64,
    /// Total network bandwidth
    pub total_network_bandwidth: f64,
    /// Used network bandwidth
    pub used_network_bandwidth: f64,
    /// Total storage capacity
    pub total_storage_capacity: f64,
    /// Used storage capacity
    pub used_storage_capacity: f64,
    /// Number of active resource providers
    pub active_providers: u64,
    /// Resource allocation efficiency (0.0-1.0)
    pub allocation_efficiency: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl ResourceMetrics {
    /// Create new resource metrics
    pub fn new() -> Self {
        Self {
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Update resource metrics
    pub fn update(&mut self, cpu_used: f64, cpu_total: f64, memory_used: f64, memory_total: f64) {
        self.used_cpu_capacity = cpu_used;
        self.total_cpu_capacity = cpu_total;
        self.used_memory_capacity = memory_used;
        self.total_memory_capacity = memory_total;
        self.last_updated = Utc::now();
        
        self.calculate_efficiency();
    }
    
    /// Calculate allocation efficiency
    fn calculate_efficiency(&mut self) {
        let cpu_efficiency = if self.total_cpu_capacity > 0.0 {
            self.used_cpu_capacity / self.total_cpu_capacity
        } else { 0.0 };
        
        let memory_efficiency = if self.total_memory_capacity > 0.0 {
            self.used_memory_capacity / self.total_memory_capacity
        } else { 0.0 };
        
        self.allocation_efficiency = (cpu_efficiency + memory_efficiency) / 2.0;
    }
    
    /// Get CPU utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        if self.total_cpu_capacity > 0.0 {
            (self.used_cpu_capacity / self.total_cpu_capacity).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Get memory utilization percentage
    pub fn memory_utilization(&self) -> f64 {
        if self.total_memory_capacity > 0.0 {
            (self.used_memory_capacity / self.total_memory_capacity).min(1.0)
        } else {
            0.0
        }
    }
}

/// Compartment-specific metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CompartmentMetrics {
    /// Total number of compartments
    pub total_compartments: usize,
    /// Number of scale-up events
    pub scale_up_events: u64,
    /// Number of scale-down events
    pub scale_down_events: u64,
    /// Average compartment utilization
    pub avg_compartment_utilization: f64,
    /// Compartment efficiency rating
    pub compartment_efficiency: f64,
    /// Resource rebalancing operations
    pub rebalancing_operations: u64,
    /// Last metrics update
    pub last_updated: DateTime<Utc>,
}

impl CompartmentMetrics {
    /// Create new compartment metrics
    pub fn new() -> Self {
        Self {
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Update compartment metrics
    pub fn update(&mut self, total_compartments: usize, avg_utilization: f64) {
        self.total_compartments = total_compartments;
        self.avg_compartment_utilization = avg_utilization.clamp(0.0, 1.0);
        self.last_updated = Utc::now();
        
        // Calculate efficiency based on utilization and scaling events
        let scaling_stability = if self.scale_up_events + self.scale_down_events > 0 {
            1.0 / (1.0 + (self.scale_up_events + self.scale_down_events) as f64 * 0.01)
        } else {
            1.0
        };
        
        self.compartment_efficiency = self.avg_compartment_utilization * scaling_stability;
    }
}

/// Social relationship metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SocialMetrics {
    /// Total number of relationships
    pub total_relationships: usize,
    /// Number of active friendships
    pub active_friendships: usize,
    /// Number of active buddy relationships
    pub active_buddy_relationships: usize,
    /// Total resources shared
    pub total_resources_shared: f64,
    /// Total resources received
    pub total_resources_received: f64,
    /// Number of successful sharing transactions
    pub successful_shares: u64,
    /// Number of successful reciprocations
    pub successful_reciprocations: u64,
    /// Average trust level across relationships
    pub avg_trust_level: f64,
    /// Cooperation efficiency (0.0-1.0)
    pub cooperation_efficiency: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl SocialMetrics {
    /// Create new social metrics
    pub fn new() -> Self {
        Self {
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Calculate cooperation efficiency
    pub fn calculate_cooperation_efficiency(&mut self) {
        if self.successful_shares > 0 {
            let reciprocation_rate = self.successful_reciprocations as f64 / self.successful_shares as f64;
            let sharing_balance = if self.total_resources_shared > 0.0 {
                (self.total_resources_received / self.total_resources_shared).min(1.0)
            } else {
                0.0
            };
            
            self.cooperation_efficiency = (reciprocation_rate + sharing_balance) / 2.0;
        } else {
            self.cooperation_efficiency = 0.0;
        }
        
        self.last_updated = Utc::now();
    }
    
    /// Get sharing ratio (received / given)
    pub fn sharing_ratio(&self) -> f64 {
        if self.total_resources_shared > 0.0 {
            self.total_resources_received / self.total_resources_shared
        } else if self.total_resources_received > 0.0 {
            f64::INFINITY // Only received, never shared
        } else {
            1.0 // No sharing activity
        }
    }
}

/// HAVOC node specific metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct HavocMetrics {
    /// Total crises detected
    pub total_crises_detected: u64,
    /// Emergency responses triggered
    pub emergency_responses_triggered: u64,
    /// Successful crisis resolutions
    pub successful_resolutions: u64,
    /// Failed crisis responses
    pub failed_responses: u64,
    /// Average response time in seconds
    pub avg_response_time_secs: f64,
    /// Network stability score (0.0-1.0)
    pub network_stability: f64,
    /// Resource reallocation efficiency
    pub reallocation_efficiency: f64,
    /// Current network stress level
    pub current_network_stress: f64,
    /// Last metrics update
    pub last_updated: DateTime<Utc>,
}

impl HavocMetrics {
    /// Create new HAVOC metrics
    pub fn new() -> Self {
        Self {
            network_stability: 1.0, // Start assuming stable
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Update HAVOC metrics
    pub fn update(&mut self, network_stress: f64, response_time: f64) {
        self.current_network_stress = network_stress.clamp(0.0, 1.0);
        self.network_stability = 1.0 - network_stress;
        
        // Update rolling average of response time
        if self.emergency_responses_triggered > 0 {
            let weight = 1.0 / (self.emergency_responses_triggered as f64).min(100.0);
            self.avg_response_time_secs = self.avg_response_time_secs * (1.0 - weight) + response_time * weight;
        } else {
            self.avg_response_time_secs = response_time;
        }
        
        self.last_updated = Utc::now();
        
        self.calculate_reallocation_efficiency();
    }
    
    /// Calculate resource reallocation efficiency
    fn calculate_reallocation_efficiency(&mut self) {
        if self.emergency_responses_triggered > 0 {
            let success_rate = self.successful_resolutions as f64 / self.emergency_responses_triggered as f64;
            let response_speed_factor = if self.avg_response_time_secs > 0.0 {
                (60.0 / self.avg_response_time_secs).min(1.0) // Faster responses are better
            } else {
                0.0
            };
            
            self.reallocation_efficiency = (success_rate + response_speed_factor) / 2.0;
        } else {
            self.reallocation_efficiency = 0.0;
        }
    }
    
    /// Get crisis resolution rate
    pub fn resolution_rate(&self) -> f64 {
        if self.total_crises_detected > 0 {
            self.successful_resolutions as f64 / self.total_crises_detected as f64
        } else {
            0.0
        }
    }
}

/// Crisis-specific metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CrisisMetrics {
    /// Crisis identifier
    pub crisis_id: String,
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    /// Response start timestamp
    pub response_started_at: Option<DateTime<Utc>>,
    /// Resolution timestamp
    pub resolved_at: Option<DateTime<Utc>>,
    /// Number of nodes affected
    pub nodes_affected: usize,
    /// Resource impact percentage
    pub resource_impact: f64,
    /// Response actions taken
    pub actions_taken: u32,
    /// Final outcome
    pub outcome: CrisisOutcome,
}

/// Crisis resolution outcomes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrisisOutcome {
    /// Crisis was successfully resolved
    Resolved,
    /// Crisis was contained but not fully resolved
    Contained,
    /// Crisis response failed
    Failed,
    /// Crisis is still ongoing
    Ongoing,
    /// Crisis escalated to higher severity
    Escalated,
}

impl Default for CrisisOutcome {
    fn default() -> Self {
        CrisisOutcome::Ongoing
    }
}

/// Thermal monitoring metrics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ThermalMetrics {
    /// Current thermal signature
    pub current_thermal_signature: f64,
    /// Peak thermal signature observed
    pub peak_thermal_signature: f64,
    /// Average thermal signature over time
    pub avg_thermal_signature: f64,
    /// Number of thermal alerts generated
    pub thermal_alerts: u64,
    /// Cooling actions triggered
    pub cooling_actions: u64,
    /// Thermal efficiency (0.0-1.0)
    pub thermal_efficiency: f64,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl ThermalMetrics {
    /// Create new thermal metrics
    pub fn new() -> Self {
        Self {
            thermal_efficiency: 1.0, // Start assuming optimal
            last_updated: Utc::now(),
            ..Default::default()
        }
    }
    
    /// Update thermal metrics
    pub fn update(&mut self, current_signature: f64) {
        self.current_thermal_signature = current_signature.clamp(0.0, 1.0);
        self.peak_thermal_signature = self.peak_thermal_signature.max(current_signature);
        
        // Update rolling average
        let weight = 0.1; // 10% weighting for new value
        self.avg_thermal_signature = self.avg_thermal_signature * (1.0 - weight) + current_signature * weight;
        
        // Calculate thermal efficiency (inverse of thermal stress)
        self.thermal_efficiency = 1.0 - current_signature;
        
        self.last_updated = Utc::now();
    }
    
    /// Check if thermal alert should be triggered
    pub fn should_alert(&self, threshold: f64) -> bool {
        self.current_thermal_signature > threshold
    }
}

/// Metrics aggregator for the entire resource management system
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// Resource utilization metrics
    pub resources: ResourceMetrics,
    /// Compartment metrics
    pub compartments: CompartmentMetrics,
    /// Social relationship metrics
    pub social: SocialMetrics,
    /// HAVOC crisis management metrics
    pub havoc: HavocMetrics,
    /// Thermal monitoring metrics
    pub thermal: ThermalMetrics,
    /// System uptime
    pub uptime: Duration,
    /// System start time
    pub started_at: DateTime<Utc>,
    /// Last comprehensive update
    pub last_updated: DateTime<Utc>,
}

impl SystemMetrics {
    /// Create new system metrics
    pub fn new() -> Self {
        Self {
            performance: PerformanceMetrics::new(),
            resources: ResourceMetrics::new(),
            compartments: CompartmentMetrics::new(),
            social: SocialMetrics::new(),
            havoc: HavocMetrics::new(),
            thermal: ThermalMetrics::new(),
            started_at: Utc::now(),
            last_updated: Utc::now(),
            uptime: Duration::from_secs(0),
        }
    }
    
    /// Update system metrics with current values
    pub fn update_all(&mut self) {
        let now = Utc::now();
        self.uptime = now.signed_duration_since(self.started_at).to_std().unwrap_or_default();
        self.last_updated = now;
        
        // Update individual metric timestamps
        self.performance.last_updated = now;
        self.resources.last_updated = now;
        self.compartments.last_updated = now;
        self.social.last_updated = now;
        self.havoc.last_updated = now;
        self.thermal.last_updated = now;
    }
    
    /// Get overall system health score (0.0-1.0)
    pub fn system_health_score(&self) -> f64 {
        let performance_weight = 0.25;
        let resource_weight = 0.20;
        let compartment_weight = 0.15;
        let social_weight = 0.15;
        let havoc_weight = 0.15;
        let thermal_weight = 0.10;
        
        self.performance.performance_score() * performance_weight +
        self.resources.allocation_efficiency * resource_weight +
        self.compartments.compartment_efficiency * compartment_weight +
        self.social.cooperation_efficiency * social_weight +
        self.havoc.network_stability * havoc_weight +
        self.thermal.thermal_efficiency * thermal_weight
    }
    
    /// Get system status summary
    pub fn status_summary(&self) -> SystemStatusSummary {
        SystemStatusSummary {
            overall_health: self.system_health_score(),
            uptime_hours: self.uptime.as_secs() as f64 / 3600.0,
            active_compartments: self.compartments.total_compartments,
            resource_efficiency: self.resources.allocation_efficiency,
            crisis_resolution_rate: self.havoc.resolution_rate(),
            social_cooperation_rate: self.social.cooperation_efficiency,
            thermal_status: if self.thermal.current_thermal_signature > 0.8 {
                ThermalStatus::Hot
            } else if self.thermal.current_thermal_signature > 0.6 {
                ThermalStatus::Warm
            } else {
                ThermalStatus::Cool
            },
            last_updated: self.last_updated,
        }
    }
}

/// System status summary for dashboards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatusSummary {
    /// Overall system health (0.0-1.0)
    pub overall_health: f64,
    /// System uptime in hours
    pub uptime_hours: f64,
    /// Number of active compartments
    pub active_compartments: usize,
    /// Resource allocation efficiency
    pub resource_efficiency: f64,
    /// Crisis resolution success rate
    pub crisis_resolution_rate: f64,
    /// Social cooperation efficiency
    pub social_cooperation_rate: f64,
    /// Current thermal status
    pub thermal_status: ThermalStatus,
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Thermal status indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThermalStatus {
    /// System is running cool (< 0.6 thermal signature)
    Cool,
    /// System is running warm (0.6-0.8 thermal signature)
    Warm,
    /// System is running hot (> 0.8 thermal signature)
    Hot,
}

impl std::fmt::Display for ThermalStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThermalStatus::Cool => write!(f, "COOL"),
            ThermalStatus::Warm => write!(f, "WARM"),
            ThermalStatus::Hot => write!(f, "HOT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_metrics() {
        let mut metrics = PerformanceMetrics::new();
        
        metrics.update(100, 5, 250.0, 0.8, 0.7, 0.6, 0.5);
        
        assert_eq!(metrics.tasks_completed, 100);
        assert_eq!(metrics.tasks_failed, 5);
        assert_eq!(metrics.avg_completion_time_ms, 250.0);
        
        let score = metrics.performance_score();
        assert!(score >= 0.0 && score <= 1.0);
        assert!(metrics.error_rate > 0.0); // Should have calculated error rate
    }
    
    #[test]
    fn test_resource_metrics() {
        let mut metrics = ResourceMetrics::new();
        
        metrics.update(8.0, 10.0, 6.0, 8.0);
        
        assert_eq!(metrics.cpu_utilization(), 0.8);
        assert_eq!(metrics.memory_utilization(), 0.75);
        assert!(metrics.allocation_efficiency > 0.0);
    }
    
    #[test]
    fn test_social_metrics() {
        let mut metrics = SocialMetrics::new();
        
        metrics.total_resources_shared = 100.0;
        metrics.total_resources_received = 80.0;
        metrics.successful_shares = 10;
        metrics.successful_reciprocations = 8;
        
        metrics.calculate_cooperation_efficiency();
        
        assert_eq!(metrics.sharing_ratio(), 0.8);
        assert!(metrics.cooperation_efficiency > 0.0);
    }
    
    #[test]
    fn test_havoc_metrics() {
        let mut metrics = HavocMetrics::new();
        
        metrics.update(0.7, 30.0);
        
        assert_eq!(metrics.current_network_stress, 0.7);
        assert_eq!(metrics.network_stability, 0.3);
        assert_eq!(metrics.avg_response_time_secs, 30.0);
    }
    
    #[test]
    fn test_thermal_metrics() {
        let mut metrics = ThermalMetrics::new();
        
        metrics.update(0.8);
        
        assert_eq!(metrics.current_thermal_signature, 0.8);
        assert_eq!(metrics.peak_thermal_signature, 0.8);
        assert!(metrics.should_alert(0.7));
        assert!(!metrics.should_alert(0.9));
    }
    
    #[test]
    fn test_system_metrics() {
        let mut system_metrics = SystemMetrics::new();
        
        // Update individual metrics
        system_metrics.performance.update(50, 2, 200.0, 0.7, 0.6, 0.5, 0.4);
        system_metrics.resources.update(6.0, 10.0, 4.0, 8.0);
        system_metrics.compartments.update(5, 0.75);
        system_metrics.thermal.update(0.6);
        
        let health_score = system_metrics.system_health_score();
        assert!(health_score >= 0.0 && health_score <= 1.0);
        
        let summary = system_metrics.status_summary();
        assert_eq!(summary.active_compartments, 5);
        assert_eq!(summary.thermal_status, ThermalStatus::Warm);
    }
    
    #[test]
    fn test_compartment_metrics() {
        let mut metrics = CompartmentMetrics::new();
        
        metrics.scale_up_events = 5;
        metrics.scale_down_events = 3;
        metrics.update(8, 0.85);
        
        assert_eq!(metrics.total_compartments, 8);
        assert_eq!(metrics.avg_compartment_utilization, 0.85);
        assert!(metrics.compartment_efficiency > 0.0);
    }
    
    #[test]
    fn test_thermal_status_display() {
        assert_eq!(ThermalStatus::Cool.to_string(), "COOL");
        assert_eq!(ThermalStatus::Warm.to_string(), "WARM");
        assert_eq!(ThermalStatus::Hot.to_string(), "HOT");
    }
    
    #[test]
    fn test_crisis_metrics() {
        let metrics = CrisisMetrics {
            crisis_id: "test-crisis".to_string(),
            detected_at: Utc::now(),
            nodes_affected: 3,
            resource_impact: 0.4,
            actions_taken: 2,
            outcome: CrisisOutcome::Resolved,
            ..Default::default()
        };
        
        assert_eq!(metrics.outcome, CrisisOutcome::Resolved);
        assert_eq!(metrics.nodes_affected, 3);
    }
}