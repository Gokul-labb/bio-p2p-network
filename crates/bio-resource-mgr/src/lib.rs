//! Biological Resource Manager Crate
//! 
//! A comprehensive resource management system inspired by biological processes
//! including crisis detection, social collaboration, thermal management,
//! and adaptive scaling mechanisms.

pub mod allocation;
pub mod compartments;
pub mod constants;
pub mod errors;
pub mod havoc;
pub mod math;
pub mod metrics;
pub mod nodes;
pub mod social;
pub mod thermal;

// Re-export commonly used types
pub use allocation::{AllocationStrategy, ResourceAllocation, ResourceProvider, ResourceRequest};
pub use compartments::{
    Compartment, CompartmentManager, CompartmentType, QoSLevel, ScalingDirection
};
pub use constants::*;
pub use errors::{ResourceError, ResourceResult};
pub use havoc::{
    CrisisEvent, CrisisType, CrisisSeverity, EmergencyAction, HavocNode, HavocConfig
};
pub use math::*;
pub use metrics::{
    PerformanceMetrics, ResourceMetrics, SocialMetrics, HavocMetrics, 
    SystemMetrics, ThermalMetrics
};
pub use nodes::{
    StepUpNode, StepDownNode, ResourceThermalNode, ScalingConfig, 
    ResourceState, NodeStatus, ScalingEvent
};
pub use social::{
    FriendshipNode, BuddyNode, SocialRelationship, RelationshipType,
    TrustLevel, SharingTransaction, BuddyRelationship
};
pub use thermal::{ThermalSignature, ThermalNode, ThermalState};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// Biological Resource Manager
/// 
/// Main entry point for the biological resource management system.
/// Coordinates all biological nodes and provides unified management interface.
#[derive(Debug)]
pub struct BiologicalResourceManager {
    /// HAVOC node for crisis management
    havoc_node: Option<HavocNode>,
    /// Step-up nodes for performance scaling
    step_up_nodes: Vec<StepUpNode>,
    /// Step-down nodes for energy conservation
    step_down_nodes: Vec<StepDownNode>,
    /// Social friendship nodes
    friendship_nodes: Vec<FriendshipNode>,
    /// Social buddy nodes
    buddy_nodes: Vec<BuddyNode>,
    /// Thermal management nodes
    thermal_nodes: Vec<ResourceThermalNode>,
    /// System metrics
    system_metrics: SystemMetrics,
    /// Manager configuration
    config: ManagerConfig,
}

/// Configuration for the biological resource manager
#[derive(Debug, Clone)]
pub struct ManagerConfig {
    /// Enable HAVOC crisis management
    pub enable_havoc: bool,
    /// Enable social collaboration features
    pub enable_social: bool,
    /// Enable thermal management
    pub enable_thermal: bool,
    /// Enable automatic scaling
    pub enable_auto_scaling: bool,
    /// Metrics collection interval
    pub metrics_interval: std::time::Duration,
    /// Maximum number of nodes per type
    pub max_nodes_per_type: usize,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            enable_havoc: true,
            enable_social: true,
            enable_thermal: true,
            enable_auto_scaling: true,
            metrics_interval: std::time::Duration::from_secs(60),
            max_nodes_per_type: 10,
        }
    }
}

impl BiologicalResourceManager {
    /// Create a new biological resource manager
    pub fn new(config: ManagerConfig) -> Self {
        Self {
            havoc_node: None,
            step_up_nodes: Vec::new(),
            step_down_nodes: Vec::new(),
            friendship_nodes: Vec::new(),
            buddy_nodes: Vec::new(),
            thermal_nodes: Vec::new(),
            system_metrics: SystemMetrics::new(),
            config,
        }
    }
    
    /// Initialize the resource manager with default components
    pub async fn initialize(&mut self, manager_id: String) -> ResourceResult<()> {
        // Initialize HAVOC node if enabled
        if self.config.enable_havoc {
            let havoc_config = HavocConfig::default();
            let havoc_node = HavocNode::new(format!("{}-havoc", manager_id), havoc_config);
            havoc_node.start().await?;
            self.havoc_node = Some(havoc_node);
        }
        
        Ok(())
    }
    
    /// Add a step-up node for performance scaling
    pub async fn add_step_up_node(&mut self, node_id: String) -> ResourceResult<()> {
        if self.step_up_nodes.len() >= self.config.max_nodes_per_type {
            return Err(ResourceError::allocation_failed("Maximum step-up nodes reached"));
        }
        
        let config = ScalingConfig::default();
        let node = StepUpNode::new(node_id, config);
        
        if self.config.enable_auto_scaling {
            node.start().await?;
        }
        
        self.step_up_nodes.push(node);
        Ok(())
    }
    
    /// Add a step-down node for energy conservation
    pub async fn add_step_down_node(&mut self, node_id: String) -> ResourceResult<()> {
        if self.step_down_nodes.len() >= self.config.max_nodes_per_type {
            return Err(ResourceError::allocation_failed("Maximum step-down nodes reached"));
        }
        
        let config = ScalingConfig::default();
        let node = StepDownNode::new(node_id, config);
        
        if self.config.enable_auto_scaling {
            node.start().await?;
        }
        
        self.step_down_nodes.push(node);
        Ok(())
    }
    
    /// Add a friendship node for social collaboration
    pub async fn add_friendship_node(&mut self, node_id: String, address: String) -> ResourceResult<()> {
        if !self.config.enable_social {
            return Err(ResourceError::configuration_error("social", "Social features disabled"));
        }
        
        if self.friendship_nodes.len() >= self.config.max_nodes_per_type {
            return Err(ResourceError::allocation_failed("Maximum friendship nodes reached"));
        }
        
        let social_config = social::SocialConfig::default();
        let node = FriendshipNode::new(node_id, address, social_config);
        node.start().await?;
        
        self.friendship_nodes.push(node);
        Ok(())
    }
    
    /// Add a buddy node for permanent partnerships
    pub async fn add_buddy_node(&mut self, node_id: String) -> ResourceResult<()> {
        if !self.config.enable_social {
            return Err(ResourceError::configuration_error("social", "Social features disabled"));
        }
        
        if self.buddy_nodes.len() >= self.config.max_nodes_per_type {
            return Err(ResourceError::allocation_failed("Maximum buddy nodes reached"));
        }
        
        let social_config = social::SocialConfig::default();
        let node = BuddyNode::new(node_id, social_config);
        node.start().await?;
        
        self.buddy_nodes.push(node);
        Ok(())
    }
    
    /// Add a thermal management node
    pub fn add_thermal_node(&mut self, node_id: String) -> ResourceResult<()> {
        if !self.config.enable_thermal {
            return Err(ResourceError::configuration_error("thermal", "Thermal management disabled"));
        }
        
        if self.thermal_nodes.len() >= self.config.max_nodes_per_type {
            return Err(ResourceError::allocation_failed("Maximum thermal nodes reached"));
        }
        
        let thermal_config = nodes::ThermalConfig::default();
        let node = ResourceThermalNode::new(node_id, thermal_config);
        
        self.thermal_nodes.push(node);
        Ok(())
    }
    
    /// Trigger emergency resource reallocation
    pub async fn trigger_emergency_reallocation(
        &self,
        crisis_type: CrisisType,
        affected_nodes: Vec<String>,
        reallocation_percentage: f64,
    ) -> ResourceResult<uuid::Uuid> {
        if let Some(havoc_node) = &self.havoc_node {
            havoc_node.trigger_emergency_reallocation(crisis_type, affected_nodes, reallocation_percentage).await
        } else {
            Err(ResourceError::havoc_error("HAVOC node not initialized"))
        }
    }
    
    /// Get current network stress level
    pub fn get_network_stress(&self) -> f64 {
        if let Some(havoc_node) = &self.havoc_node {
            havoc_node.get_network_stress_level()
        } else {
            0.0
        }
    }
    
    /// Get system metrics
    pub fn get_system_metrics(&mut self) -> SystemMetrics {
        self.system_metrics.update_all();
        
        // Update metrics from individual components
        if let Some(havoc_node) = &self.havoc_node {
            self.system_metrics.havoc = havoc_node.get_metrics();
        }
        
        // Update social metrics from friendship nodes
        if !self.friendship_nodes.is_empty() {
            let mut total_social_metrics = SocialMetrics::new();
            for node in &self.friendship_nodes {
                let metrics = node.get_metrics();
                total_social_metrics.total_relationships += metrics.total_relationships;
                total_social_metrics.active_friendships += metrics.active_friendships;
                total_social_metrics.total_resources_shared += metrics.total_resources_shared;
                total_social_metrics.successful_shares += metrics.successful_shares;
            }
            total_social_metrics.calculate_cooperation_efficiency();
            self.system_metrics.social = total_social_metrics;
        }
        
        self.system_metrics.clone()
    }
    
    /// Get system status summary
    pub fn get_status_summary(&mut self) -> metrics::SystemStatusSummary {
        let metrics = self.get_system_metrics();
        metrics.status_summary()
    }
    
    /// Scale up resources across all step-up nodes
    pub async fn scale_up_all(&self, target_factor: f64) -> ResourceResult<Vec<ResourceResult<()>>> {
        let mut results = Vec::new();
        
        for node in &self.step_up_nodes {
            let result = node.scale_up(Some(target_factor)).await;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Scale down resources across all step-down nodes
    pub async fn scale_down_all(&self, target_factor: f64) -> ResourceResult<Vec<ResourceResult<()>>> {
        let mut results = Vec::new();
        
        for node in &self.step_down_nodes {
            let result = node.scale_down(Some(target_factor)).await;
            results.push(result);
        }
        
        Ok(results)
    }
    
    /// Update thermal readings for all thermal nodes
    pub fn update_thermal_readings(&self, readings: &[(String, f64, f64, f64, f64)]) -> ResourceResult<()> {
        for (node_id, cpu, memory, network, storage) in readings {
            if let Some(thermal_node) = self.thermal_nodes.iter().find(|n| n.id == *node_id) {
                thermal_node.update_thermal(*cpu, *memory, *network, *storage)?;
            }
        }
        Ok(())
    }
    
    /// Get comprehensive health report
    pub fn get_health_report(&mut self) -> HealthReport {
        let metrics = self.get_system_metrics();
        
        HealthReport {
            overall_health: metrics.system_health_score(),
            component_health: ComponentHealth {
                havoc: self.havoc_node.as_ref().map(|_| metrics.havoc.network_stability).unwrap_or(1.0),
                scaling: if !self.step_up_nodes.is_empty() || !self.step_down_nodes.is_empty() {
                    0.9 // Placeholder for scaling health
                } else {
                    1.0
                },
                social: metrics.social.cooperation_efficiency,
                thermal: metrics.thermal.thermal_efficiency,
            },
            active_nodes: ActiveNodeCounts {
                havoc_nodes: if self.havoc_node.is_some() { 1 } else { 0 },
                step_up_nodes: self.step_up_nodes.len(),
                step_down_nodes: self.step_down_nodes.len(),
                friendship_nodes: self.friendship_nodes.len(),
                buddy_nodes: self.buddy_nodes.len(),
                thermal_nodes: self.thermal_nodes.len(),
            },
            recommendations: self.generate_recommendations(&metrics),
            timestamp: chrono::Utc::now(),
        }
    }
    
    /// Generate optimization recommendations
    fn generate_recommendations(&self, metrics: &SystemMetrics) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if metrics.havoc.current_network_stress > 0.8 {
            recommendations.push("High network stress detected - consider scaling up resources".to_string());
        }
        
        if metrics.thermal.current_thermal_signature > 0.8 {
            recommendations.push("Thermal stress detected - reduce load or improve cooling".to_string());
        }
        
        if metrics.social.cooperation_efficiency < 0.5 {
            recommendations.push("Low social cooperation - review relationship management".to_string());
        }
        
        if self.step_up_nodes.is_empty() && metrics.performance.cpu_utilization > 0.9 {
            recommendations.push("Add step-up nodes for better performance scaling".to_string());
        }
        
        if self.step_down_nodes.is_empty() && metrics.performance.cpu_utilization < 0.3 {
            recommendations.push("Add step-down nodes for energy conservation".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("System operating optimally".to_string());
        }
        
        recommendations
    }
    
    /// Shutdown all nodes gracefully
    pub async fn shutdown(&mut self) -> ResourceResult<()> {
        // Stop HAVOC node
        if let Some(havoc_node) = &self.havoc_node {
            havoc_node.stop().await?;
        }
        
        // Stop step-up nodes
        for node in &self.step_up_nodes {
            node.stop().await?;
        }
        
        // Stop step-down nodes
        for node in &self.step_down_nodes {
            node.stop().await?;
        }
        
        // Stop friendship nodes
        for node in &self.friendship_nodes {
            node.stop().await?;
        }
        
        // Stop buddy nodes
        for node in &self.buddy_nodes {
            node.stop().await?;
        }
        
        Ok(())
    }
}

/// System health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall system health score (0.0-1.0)
    pub overall_health: f64,
    /// Health of individual components
    pub component_health: ComponentHealth,
    /// Count of active nodes by type
    pub active_nodes: ActiveNodeCounts,
    /// Optimization recommendations
    pub recommendations: Vec<String>,
    /// Report timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Health scores for individual components
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// HAVOC crisis management health
    pub havoc: f64,
    /// Scaling system health
    pub scaling: f64,
    /// Social collaboration health
    pub social: f64,
    /// Thermal management health
    pub thermal: f64,
}

/// Count of active nodes by type
#[derive(Debug, Clone)]
pub struct ActiveNodeCounts {
    /// Number of HAVOC nodes
    pub havoc_nodes: usize,
    /// Number of step-up nodes
    pub step_up_nodes: usize,
    /// Number of step-down nodes
    pub step_down_nodes: usize,
    /// Number of friendship nodes
    pub friendship_nodes: usize,
    /// Number of buddy nodes
    pub buddy_nodes: usize,
    /// Number of thermal nodes
    pub thermal_nodes: usize,
}

impl ActiveNodeCounts {
    /// Get total number of active nodes
    pub fn total(&self) -> usize {
        self.havoc_nodes + self.step_up_nodes + self.step_down_nodes +
        self.friendship_nodes + self.buddy_nodes + self.thermal_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_manager_creation() {
        let config = ManagerConfig::default();
        let manager = BiologicalResourceManager::new(config);
        
        assert_eq!(manager.step_up_nodes.len(), 0);
        assert_eq!(manager.step_down_nodes.len(), 0);
        assert!(manager.havoc_node.is_none());
    }
    
    #[tokio::test]
    async fn test_manager_initialization() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.initialize("test-manager".to_string()).await.unwrap();
        
        assert!(manager.havoc_node.is_some());
    }
    
    #[tokio::test]
    async fn test_add_step_up_node() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_step_up_node("step-up-1".to_string()).await.unwrap();
        
        assert_eq!(manager.step_up_nodes.len(), 1);
        assert_eq!(manager.step_up_nodes[0].id, "step-up-1");
    }
    
    #[tokio::test]
    async fn test_add_step_down_node() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_step_down_node("step-down-1".to_string()).await.unwrap();
        
        assert_eq!(manager.step_down_nodes.len(), 1);
        assert_eq!(manager.step_down_nodes[0].id, "step-down-1");
    }
    
    #[tokio::test]
    async fn test_add_friendship_node() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_friendship_node(
            "friend-1".to_string(),
            "127.0.0.1".to_string(),
        ).await.unwrap();
        
        assert_eq!(manager.friendship_nodes.len(), 1);
        assert_eq!(manager.friendship_nodes[0].id, "friend-1");
    }
    
    #[tokio::test]
    async fn test_add_buddy_node() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_buddy_node("buddy-1".to_string()).await.unwrap();
        
        assert_eq!(manager.buddy_nodes.len(), 1);
        assert_eq!(manager.buddy_nodes[0].id, "buddy-1");
    }
    
    #[test]
    fn test_add_thermal_node() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_thermal_node("thermal-1".to_string()).unwrap();
        
        assert_eq!(manager.thermal_nodes.len(), 1);
        assert_eq!(manager.thermal_nodes[0].id, "thermal-1");
    }
    
    #[tokio::test]
    async fn test_emergency_reallocation() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        manager.initialize("test".to_string()).await.unwrap();
        
        let crisis_id = manager.trigger_emergency_reallocation(
            CrisisType::ResourceShortage,
            vec!["node1".to_string()],
            0.5,
        ).await.unwrap();
        
        assert!(!crisis_id.is_nil());
    }
    
    #[test]
    fn test_health_report() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        let report = manager.get_health_report();
        
        assert!(report.overall_health >= 0.0 && report.overall_health <= 1.0);
        assert_eq!(report.active_nodes.total(), 0);
        assert!(!report.recommendations.is_empty());
    }
    
    #[test]
    fn test_system_metrics() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        let metrics = manager.get_system_metrics();
        let summary = manager.get_status_summary();
        
        assert!(metrics.system_health_score() >= 0.0);
        assert_eq!(summary.active_compartments, 0);
    }
    
    #[test]
    fn test_thermal_readings_update() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        manager.add_thermal_node("thermal-1".to_string()).unwrap();
        
        let readings = vec![
            ("thermal-1".to_string(), 0.8, 0.7, 0.6, 0.5),
        ];
        
        manager.update_thermal_readings(&readings).unwrap();
        
        let signature = manager.thermal_nodes[0].get_thermal_signature();
        assert!(signature > 0.0);
    }
    
    #[test]
    fn test_manager_config() {
        let config = ManagerConfig {
            enable_havoc: false,
            enable_social: false,
            enable_thermal: false,
            enable_auto_scaling: false,
            metrics_interval: std::time::Duration::from_secs(30),
            max_nodes_per_type: 5,
        };
        
        let manager = BiologicalResourceManager::new(config.clone());
        
        assert!(!manager.config.enable_havoc);
        assert!(!manager.config.enable_social);
        assert_eq!(manager.config.max_nodes_per_type, 5);
    }
    
    #[test]
    fn test_version_info() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "bio-resource-mgr");
    }
}