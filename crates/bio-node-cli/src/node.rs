//! Biological node implementation integrating all network components
//!
//! This module provides the main biological node that coordinates P2P networking,
//! biological behaviors, security systems, and resource management.

use anyhow::{Context, Result};
use libp2p::PeerId;
use tokio::sync::{broadcast, RwLock};
use tracing::{info, debug, error, warn, span, Level};
use std::sync::Arc;
use std::collections::HashMap;

use bio_p2p_core::{
    protocol::BiologicalProtocol,
    package::{Package, PackageManager},
    reputation::ReputationManager,
};

use bio_p2p_p2p::{
    node::P2PNode,
    discovery::PeerDiscovery,
    routing::BiologicalRouter,
};

use bio_p2p_security::{
    manager::SecurityManager,
    layers::{SecurityLayer, LayerType},
};

use bio_p2p_resource::{
    manager::ResourceManager,
    allocation::ResourceAllocator,
    monitoring::ResourceMonitor,
};

use crate::{
    config::{NodeConfig, BiologicalRole},
    daemon::{NetworkInfo, ResourceUsage},
};

/// Main biological node coordinating all network components
#[derive(Debug)]
pub struct BiologicalNode {
    /// Node configuration
    config: NodeConfig,
    
    /// P2P networking layer
    p2p_node: Arc<P2PNode>,
    
    /// Biological protocol handler
    protocol: Arc<BiologicalProtocol>,
    
    /// Package processing manager
    package_manager: Arc<PackageManager>,
    
    /// Security management system
    security_manager: Arc<SecurityManager>,
    
    /// Resource management system
    resource_manager: Arc<ResourceManager>,
    
    /// Reputation system
    reputation_manager: Arc<ReputationManager>,
    
    /// Peer discovery system
    peer_discovery: Arc<PeerDiscovery>,
    
    /// Biological routing system
    router: Arc<BiologicalRouter>,
    
    /// Currently active biological roles
    active_roles: Arc<RwLock<Vec<BiologicalRole>>>,
    
    /// Network health status
    network_health: Arc<RwLock<bool>>,
    
    /// Node statistics
    node_stats: Arc<RwLock<NodeStatistics>>,
}

/// Node operational statistics
#[derive(Debug, Clone, Default)]
pub struct NodeStatistics {
    /// Tasks completed successfully
    pub tasks_completed: u64,
    /// Tasks failed
    pub tasks_failed: u64,
    /// Total uptime in seconds
    pub uptime_seconds: u64,
    /// Bytes sent over network
    pub bytes_sent: u64,
    /// Bytes received over network
    pub bytes_received: u64,
    /// Current reputation score
    pub reputation_score: f64,
    /// Trust relationships count
    pub trust_relationships: usize,
    /// Crisis events handled
    pub crisis_events: u64,
}

/// Network bandwidth information
#[derive(Debug, Clone)]
pub struct BandwidthInfo {
    /// Inbound bandwidth (bytes/sec)
    pub inbound: u64,
    /// Outbound bandwidth (bytes/sec)  
    pub outbound: u64,
}

impl BiologicalNode {
    /// Create new biological node
    pub async fn new(config: &NodeConfig) -> Result<Self> {
        let span = span!(Level::INFO, "biological_node_init");
        let _enter = span.enter();
        
        info!("Initializing biological node");
        
        // Initialize P2P node
        let p2p_node = Arc::new(P2PNode::new(
            &config.network.listen_addresses,
            &config.network.node_key_path,
            config.network.max_connections
        ).await.context("Failed to create P2P node")?);
        
        // Initialize biological protocol
        let protocol = Arc::new(BiologicalProtocol::new(
            config.biological.preferred_roles.clone(),
            config.biological.learning_rate,
            config.biological.trust_building_rate
        ).await.context("Failed to create biological protocol")?);
        
        // Initialize package manager
        let package_manager = Arc::new(PackageManager::new(
            p2p_node.clone(),
            protocol.clone()
        ).await.context("Failed to create package manager")?);
        
        // Initialize security manager
        let security_manager = Arc::new(SecurityManager::new(
            &config.security,
            p2p_node.clone()
        ).await.context("Failed to create security manager")?);
        
        // Initialize resource manager
        let resource_manager = Arc::new(ResourceManager::new(
            &config.resources,
            config.biological.preferred_roles.clone()
        ).await.context("Failed to create resource manager")?);
        
        // Initialize reputation manager
        let reputation_manager = Arc::new(ReputationManager::new(
            config.economics.reputation.initial_score,
            config.economics.reputation.decay_rate,
            config.economics.reputation.weights.clone()
        ).await.context("Failed to create reputation manager")?);
        
        // Initialize peer discovery
        let peer_discovery = Arc::new(PeerDiscovery::new(
            p2p_node.clone(),
            config.network.enable_mdns,
            config.network.bootstrap_peers.clone()
        ).await.context("Failed to create peer discovery")?);
        
        // Initialize biological router
        let router = Arc::new(BiologicalRouter::new(
            p2p_node.clone(),
            protocol.clone()
        ).await.context("Failed to create biological router")?);
        
        let active_roles = Arc::new(RwLock::new(Vec::new()));
        let network_health = Arc::new(RwLock::new(false));
        let node_stats = Arc::new(RwLock::new(NodeStatistics::default()));
        
        info!("Biological node initialization completed");
        
        Ok(Self {
            config: config.clone(),
            p2p_node,
            protocol,
            package_manager,
            security_manager,
            resource_manager,
            reputation_manager,
            peer_discovery,
            router,
            active_roles,
            network_health,
            node_stats,
        })
    }
    
    /// Start P2P networking
    pub async fn start_p2p_network(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Starting P2P network");
        
        // Start P2P node
        let p2p_handle = {
            let node = self.p2p_node.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                node.start(shutdown).await
                    .context("P2P node failed")
            })
        };
        
        // Start peer discovery
        let discovery_handle = {
            let discovery = self.peer_discovery.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                discovery.start(shutdown).await
                    .context("Peer discovery failed")
            })
        };
        
        // Start router
        let router_handle = {
            let router = self.router.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                router.start(shutdown).await
                    .context("Biological router failed")
            })
        };
        
        // Monitor network health
        let health_handle = {
            let network_health = self.network_health.clone();
            let p2p_node = self.p2p_node.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                Self::monitor_network_health(network_health, p2p_node, shutdown).await
                    .context("Network health monitoring failed")
            })
        };
        
        // Wait for shutdown
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("P2P network shutting down");
            }
            result = p2p_handle => {
                error!("P2P node exited: {:?}", result);
            }
            result = discovery_handle => {
                error!("Peer discovery exited: {:?}", result);
            }
            result = router_handle => {
                error!("Router exited: {:?}", result);
            }
            result = health_handle => {
                error!("Health monitor exited: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    /// Activate biological behaviors
    pub async fn activate_biological_behaviors(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Activating biological behaviors");
        
        // Start biological protocol
        let protocol_handle = {
            let protocol = self.protocol.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                protocol.start(shutdown).await
                    .context("Biological protocol failed")
            })
        };
        
        // Activate preferred roles
        self.activate_preferred_roles().await?;
        
        // Start role management
        let role_handle = {
            let active_roles = self.active_roles.clone();
            let protocol = self.protocol.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                Self::manage_biological_roles(active_roles, protocol, shutdown).await
                    .context("Role management failed")
            })
        };
        
        // Wait for shutdown
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("Biological behaviors shutting down");
            }
            result = protocol_handle => {
                error!("Biological protocol exited: {:?}", result);
            }
            result = role_handle => {
                error!("Role management exited: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    /// Start resource management systems
    pub async fn start_resource_management(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Starting resource management");
        
        // Start resource manager
        let resource_handle = {
            let manager = self.resource_manager.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                manager.start(shutdown).await
                    .context("Resource manager failed")
            })
        };
        
        // Start package manager
        let package_handle = {
            let manager = self.package_manager.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                manager.start(shutdown).await
                    .context("Package manager failed")
            })
        };
        
        // Wait for shutdown
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("Resource management shutting down");
            }
            result = resource_handle => {
                error!("Resource manager exited: {:?}", result);
            }
            result = package_handle => {
                error!("Package manager exited: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    /// Start security systems
    pub async fn start_security_systems(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Starting security systems");
        
        // Start security manager
        let security_handle = {
            let manager = self.security_manager.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                manager.start(shutdown).await
                    .context("Security manager failed")
            })
        };
        
        // Start reputation manager
        let reputation_handle = {
            let manager = self.reputation_manager.clone();
            let shutdown = shutdown_rx.resubscribe();
            
            tokio::spawn(async move {
                manager.start(shutdown).await
                    .context("Reputation manager failed")
            })
        };
        
        // Wait for shutdown
        tokio::select! {
            _ = shutdown_rx.recv() => {
                info!("Security systems shutting down");
            }
            result = security_handle => {
                error!("Security manager exited: {:?}", result);
            }
            result = reputation_handle => {
                error!("Reputation manager exited: {:?}", result);
            }
        }
        
        Ok(())
    }
    
    /// Check if network is ready
    pub async fn is_network_ready(&self) -> bool {
        *self.network_health.read().await
    }
    
    /// Check if node is healthy
    pub async fn is_healthy(&self) -> bool {
        self.is_network_ready().await && 
        self.security_manager.is_healthy().await &&
        self.resource_manager.is_healthy().await
    }
    
    /// Check if security systems are healthy
    pub async fn is_security_healthy(&self) -> bool {
        self.security_manager.is_healthy().await
    }
    
    /// Get network information
    pub async fn get_network_info(&self) -> Option<NetworkInfo> {
        if let Ok(local_peer_id) = self.p2p_node.local_peer_id().await {
            if let Ok(listen_addrs) = self.p2p_node.listen_addresses().await {
                if let Ok(connected_peers) = self.p2p_node.connected_peer_count().await {
                    return Some(NetworkInfo {
                        local_peer_id: local_peer_id.to_string(),
                        listen_addresses: listen_addrs.iter().map(|a| a.to_string()).collect(),
                        connected_peers,
                        protocol_version: self.config.network.protocol_version.clone(),
                    });
                }
            }
        }
        
        None
    }
    
    /// Get resource usage information
    pub async fn get_resource_usage(&self) -> Option<ResourceUsage> {
        self.resource_manager.get_current_usage().await.ok()
    }
    
    /// Get currently active biological roles
    pub async fn get_active_roles(&self) -> Vec<BiologicalRole> {
        self.active_roles.read().await.clone()
    }
    
    /// Get health metrics
    pub async fn get_health_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        if let Ok(uptime) = self.get_uptime_seconds().await {
            metrics.insert("uptime_seconds".to_string(), uptime as f64);
        }
        
        if let Ok(task_rate) = self.get_task_completion_rate().await {
            metrics.insert("task_completion_rate".to_string(), task_rate);
        }
        
        if let Ok(reputation) = self.get_reputation_score().await {
            metrics.insert("reputation_score".to_string(), reputation);
        }
        
        metrics
    }
    
    /// Get network metrics
    pub async fn get_network_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        if let Some(info) = self.get_network_info().await {
            metrics.insert("connected_peers".to_string(), info.connected_peers as f64);
        }
        
        if let Ok(latency) = self.get_average_latency().await {
            metrics.insert("avg_latency_ms".to_string(), latency);
        }
        
        if let Ok(packet_loss) = self.get_packet_loss().await {
            metrics.insert("packet_loss".to_string(), packet_loss);
        }
        
        metrics
    }
    
    /// Get security metrics
    pub async fn get_security_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        if let Ok(quarantined) = self.get_quarantined_node_count().await {
            metrics.insert("quarantined_nodes".to_string(), quarantined as f64);
        }
        
        if let Ok(anomaly_rate) = self.get_anomaly_detection_rate().await {
            metrics.insert("anomaly_rate".to_string(), anomaly_rate);
        }
        
        metrics
    }
    
    /// Activate preferred biological roles
    async fn activate_preferred_roles(&self) -> Result<()> {
        let mut active_roles = self.active_roles.write().await;
        
        for role in &self.config.biological.preferred_roles {
            match self.protocol.activate_role(role.clone()).await {
                Ok(()) => {
                    active_roles.push(role.clone());
                    info!("Activated biological role: {:?}", role);
                }
                Err(e) => {
                    warn!("Failed to activate role {:?}: {}", role, e);
                }
            }
        }
        
        if active_roles.is_empty() {
            return Err(anyhow::anyhow!("No biological roles could be activated"));
        }
        
        info!("Activated {} biological roles", active_roles.len());
        Ok(())
    }
    
    /// Monitor network health
    async fn monitor_network_health(
        network_health: Arc<RwLock<bool>>,
        p2p_node: Arc<P2PNode>,
        mut shutdown_rx: broadcast::Receiver<()>
    ) -> Result<()> {
        let mut health_interval = tokio::time::interval(Duration::from_secs(10));
        
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                
                _ = health_interval.tick() => {
                    let is_healthy = match p2p_node.connected_peer_count().await {
                        Ok(count) => count > 0,
                        Err(_) => false,
                    };
                    
                    {
                        let mut health = network_health.write().await;
                        *health = is_healthy;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Manage biological role lifecycle
    async fn manage_biological_roles(
        active_roles: Arc<RwLock<Vec<BiologicalRole>>>,
        protocol: Arc<BiologicalProtocol>,
        mut shutdown_rx: broadcast::Receiver<()>
    ) -> Result<()> {
        let mut role_interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    break;
                }
                
                _ = role_interval.tick() => {
                    // Check role health and performance
                    let roles = active_roles.read().await.clone();
                    
                    for role in &roles {
                        if let Err(e) = protocol.check_role_health(role).await {
                            warn!("Role {:?} health check failed: {}", role, e);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    // Health and status methods
    
    /// Get current average latency
    pub async fn get_average_latency(&self) -> Result<f64> {
        self.p2p_node.get_average_latency().await
    }
    
    /// Get packet loss percentage
    pub async fn get_packet_loss(&self) -> Result<f64> {
        self.p2p_node.get_packet_loss().await
    }
    
    /// Get bootstrap connectivity ratio
    pub async fn get_bootstrap_connectivity(&self) -> Result<f64> {
        self.peer_discovery.get_bootstrap_connectivity().await
    }
    
    /// Get routing efficiency
    pub async fn get_routing_efficiency(&self) -> Result<f64> {
        self.router.get_efficiency().await
    }
    
    /// Get current bandwidth usage
    pub async fn get_current_bandwidth(&self) -> Option<BandwidthInfo> {
        if let Ok((inbound, outbound)) = self.p2p_node.get_bandwidth_usage().await {
            Some(BandwidthInfo { inbound, outbound })
        } else {
            None
        }
    }
    
    /// Get average CPU usage
    pub async fn get_avg_cpu_usage(&self) -> Result<f64> {
        self.resource_manager.get_avg_cpu_usage().await
    }
    
    /// Get peak CPU usage
    pub async fn get_peak_cpu_usage(&self) -> Result<f64> {
        self.resource_manager.get_peak_cpu_usage().await
    }
    
    /// Get CPU usage trend
    pub async fn get_cpu_trend(&self) -> Result<f64> {
        self.resource_manager.get_cpu_trend().await
    }
    
    /// Get average memory usage
    pub async fn get_avg_memory_usage(&self) -> Result<f64> {
        self.resource_manager.get_avg_memory_usage().await
    }
    
    /// Get peak memory usage
    pub async fn get_peak_memory_usage(&self) -> Result<f64> {
        self.resource_manager.get_peak_memory_usage().await
    }
    
    /// Get memory usage trend
    pub async fn get_memory_trend(&self) -> Result<f64> {
        self.resource_manager.get_memory_trend().await
    }
    
    /// Get average disk usage
    pub async fn get_avg_disk_usage(&self) -> Result<f64> {
        self.resource_manager.get_avg_disk_usage().await
    }
    
    /// Get peak disk usage
    pub async fn get_peak_disk_usage(&self) -> Result<f64> {
        self.resource_manager.get_peak_disk_usage().await
    }
    
    /// Get disk usage trend
    pub async fn get_disk_trend(&self) -> Result<f64> {
        self.resource_manager.get_disk_trend().await
    }
    
    /// Get average network usage
    pub async fn get_avg_network_usage(&self) -> Result<f64> {
        self.p2p_node.get_avg_bandwidth().await
    }
    
    /// Get peak network usage
    pub async fn get_peak_network_usage(&self) -> Result<f64> {
        self.p2p_node.get_peak_bandwidth().await
    }
    
    /// Get network usage trend
    pub async fn get_network_trend(&self) -> Result<f64> {
        self.p2p_node.get_bandwidth_trend().await
    }
    
    /// Get network capacity
    pub async fn get_network_capacity(&self) -> Result<f64> {
        self.p2p_node.get_bandwidth_capacity().await
    }
    
    /// Get allocation efficiency
    pub async fn get_allocation_efficiency(&self) -> Result<f64> {
        self.resource_manager.get_allocation_efficiency().await
    }
    
    /// Get role performance
    pub async fn get_role_performance(&self, role: &BiologicalRole) -> Result<f64> {
        self.protocol.get_role_performance(role).await
    }
    
    /// Get trust relationship count
    pub async fn get_trust_relationship_count(&self) -> Result<usize> {
        self.reputation_manager.get_trust_relationship_count().await
    }
    
    /// Get learning efficiency
    pub async fn get_learning_efficiency(&self) -> Result<f64> {
        self.protocol.get_learning_efficiency().await
    }
    
    /// Get cooperation rate
    pub async fn get_cooperation_rate(&self) -> Result<f64> {
        self.protocol.get_cooperation_rate().await
    }
    
    /// Get task completion rate
    pub async fn get_task_completion_rate(&self) -> Result<f64> {
        let stats = self.node_stats.read().await;
        let total_tasks = stats.tasks_completed + stats.tasks_failed;
        
        if total_tasks > 0 {
            Ok(stats.tasks_completed as f64 / total_tasks as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Get average task duration
    pub async fn get_avg_task_duration(&self) -> Result<f64> {
        self.package_manager.get_avg_task_duration().await
    }
    
    /// Get throughput
    pub async fn get_throughput(&self) -> Result<f64> {
        self.package_manager.get_throughput().await
    }
    
    /// Get error rate
    pub async fn get_error_rate(&self) -> Result<f64> {
        let stats = self.node_stats.read().await;
        let total_tasks = stats.tasks_completed + stats.tasks_failed;
        
        if total_tasks > 0 {
            Ok(stats.tasks_failed as f64 / total_tasks as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Get response time
    pub async fn get_response_time(&self) -> Result<f64> {
        self.package_manager.get_avg_response_time().await
    }
    
    /// Get queue depth
    pub async fn get_queue_depth(&self) -> Result<usize> {
        self.package_manager.get_queue_depth().await
    }
    
    /// Get active task count
    pub async fn get_active_task_count(&self) -> Result<usize> {
        self.package_manager.get_active_task_count().await
    }
    
    /// Get quarantined node count
    pub async fn get_quarantined_node_count(&self) -> Result<usize> {
        self.security_manager.get_quarantined_node_count().await
    }
    
    /// Get anomaly detection rate
    pub async fn get_anomaly_detection_rate(&self) -> Result<f64> {
        self.security_manager.get_anomaly_detection_rate().await
    }
    
    /// Get token balance
    pub async fn get_token_balance(&self) -> Result<f64> {
        if self.config.economics.enable_token_economics {
            // Integration point for token system
            Ok(1000.0) // Placeholder
        } else {
            Ok(0.0)
        }
    }
    
    /// Get reputation score
    pub async fn get_reputation_score(&self) -> Result<f64> {
        self.reputation_manager.get_reputation_score().await
    }
    
    /// Get staking amount
    pub async fn get_staking_amount(&self) -> Result<f64> {
        if self.config.economics.enable_token_economics {
            // Integration point for staking system
            Ok(100.0) // Placeholder
        } else {
            Ok(0.0)
        }
    }
    
    /// Get uptime in seconds
    pub async fn get_uptime_seconds(&self) -> Result<u64> {
        let stats = self.node_stats.read().await;
        Ok(stats.uptime_seconds)
    }
    
    /// Get compartment usage
    pub async fn get_compartment_usage(&self) -> Option<HashMap<String, f64>> {
        self.resource_manager.get_compartment_usage().await.ok()
    }
    
    /// Get health status for API
    pub async fn get_health_status(&self) -> Result<serde_json::Value> {
        let network_info = self.get_network_info().await;
        let resource_usage = self.get_resource_usage().await;
        let active_roles = self.get_active_roles().await;
        
        let health_status = serde_json::json!({
            "status": if self.is_healthy().await { "healthy" } else { "unhealthy" },
            "network": {
                "ready": self.is_network_ready().await,
                "peer_id": network_info.as_ref().map(|i| &i.local_peer_id),
                "connected_peers": network_info.as_ref().map(|i| i.connected_peers).unwrap_or(0),
                "listen_addresses": network_info.as_ref().map(|i| &i.listen_addresses).unwrap_or(&vec![]),
            },
            "resources": {
                "cpu_usage": resource_usage.as_ref().map(|r| r.cpu_usage).unwrap_or(0.0),
                "memory_usage": resource_usage.as_ref().map(|r| r.memory_usage).unwrap_or(0.0),
                "disk_usage": resource_usage.as_ref().map(|r| r.disk_usage).unwrap_or(0.0),
            },
            "biological": {
                "active_roles": active_roles,
                "role_count": active_roles.len(),
            },
            "timestamp": chrono::Utc::now().to_rfc3339(),
        });
        
        Ok(health_status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeConfig;
    use tempfile::TempDir;
    
    async fn create_test_biological_node() -> (BiologicalNode, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = NodeConfig::default();
        
        // Configure test paths
        config.storage.data_dir = temp_dir.path().join("data");
        config.storage.cache_dir = temp_dir.path().join("cache");
        config.storage.log_dir = temp_dir.path().join("logs");
        config.network.node_key_path = temp_dir.path().join("node_key.pem");
        config.network.listen_addresses = vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()];
        
        let biological_node = BiologicalNode::new(&config).await.unwrap();
        
        (biological_node, temp_dir)
    }
    
    #[tokio::test]
    async fn test_biological_node_creation() {
        let (biological_node, _temp_dir) = create_test_biological_node().await;
        
        // Test initial state
        assert!(!biological_node.is_network_ready().await);
        assert!(biological_node.get_active_roles().await.is_empty());
    }
    
    #[tokio::test]
    async fn test_health_metrics() {
        let (biological_node, _temp_dir) = create_test_biological_node().await;
        
        let health_metrics = biological_node.get_health_metrics().await;
        assert!(!health_metrics.is_empty());
        
        let network_metrics = biological_node.get_network_metrics().await;
        // Network metrics may be empty before network starts
        
        let security_metrics = biological_node.get_security_metrics().await;
        assert!(!security_metrics.is_empty());
    }
    
    #[tokio::test]
    async fn test_health_status_json() {
        let (biological_node, _temp_dir) = create_test_biological_node().await;
        
        let health_status = biological_node.get_health_status().await.unwrap();
        
        // Verify JSON structure
        assert!(health_status.get("status").is_some());
        assert!(health_status.get("network").is_some());
        assert!(health_status.get("resources").is_some());
        assert!(health_status.get("biological").is_some());
        assert!(health_status.get("timestamp").is_some());
    }
}