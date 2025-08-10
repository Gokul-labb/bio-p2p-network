//! Health checking and status monitoring for Bio P2P Node
//!
//! This module provides comprehensive health monitoring including component health,
//! network connectivity, resource usage, and biological behavior validation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tokio::{sync::broadcast, time::{interval, Duration}};
use tracing::{debug, error, info, warn};
use std::sync::Arc;
use std::collections::HashMap;

use crate::{
    config::{NodeConfig, BiologicalRole},
    node::BiologicalNode,
    daemon::{ComponentStatus, NetworkInfo, ResourceUsage},
};

/// Health checking system for comprehensive node monitoring
#[derive(Debug)]
pub struct HealthChecker {
    /// Node configuration
    config: NodeConfig,
    
    /// Reference to biological node
    biological_node: Arc<BiologicalNode>,
    
    /// HTTP server for health endpoints
    health_server: Option<Arc<HealthHttpServer>>,
    
    /// Last health check results
    last_health_check: Arc<tokio::sync::RwLock<Option<HealthCheckResult>>>,
    
    /// Health check history for trending
    health_history: Arc<tokio::sync::RwLock<Vec<HealthCheckResult>>>,
}

/// Complete health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    /// Timestamp of health check
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Overall health status
    pub overall_status: HealthStatus,
    
    /// Individual component health
    pub components: HashMap<String, ComponentHealth>,
    
    /// Network connectivity health
    pub network: NetworkHealth,
    
    /// Resource usage health
    pub resources: ResourceHealth,
    
    /// Biological behavior health
    pub biological: BiologicalHealth,
    
    /// Performance metrics
    pub performance: PerformanceHealth,
}

/// Overall health status levels
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All systems operating normally
    Healthy,
    /// Some degradation but still functional
    Degraded,
    /// Significant issues affecting functionality
    Unhealthy,
    /// Critical failures requiring intervention
    Critical,
}

/// Individual component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component status
    pub status: HealthStatus,
    
    /// Health check message
    pub message: String,
    
    /// Last successful operation timestamp
    pub last_success: Option<chrono::DateTime<chrono::Utc>>,
    
    /// Error count in last hour
    pub error_count: u32,
    
    /// Component-specific metrics
    pub metrics: HashMap<String, f64>,
}

/// Network connectivity health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkHealth {
    /// Network status
    pub status: HealthStatus,
    
    /// Number of connected peers
    pub connected_peers: usize,
    
    /// Network latency metrics (ms)
    pub avg_latency_ms: f64,
    
    /// Packet loss percentage
    pub packet_loss_percentage: f64,
    
    /// Bootstrap peer connectivity
    pub bootstrap_connectivity: f64,
    
    /// Address resolution health
    pub address_resolution: HealthStatus,
    
    /// Routing efficiency
    pub routing_efficiency: f64,
}

/// Resource usage health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceHealth {
    /// Resource status
    pub status: HealthStatus,
    
    /// CPU usage metrics
    pub cpu: ResourceMetric,
    
    /// Memory usage metrics
    pub memory: ResourceMetric,
    
    /// Disk usage metrics  
    pub disk: ResourceMetric,
    
    /// Network bandwidth metrics
    pub network_bandwidth: ResourceMetric,
    
    /// Resource allocation efficiency
    pub allocation_efficiency: f64,
}

/// Individual resource metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetric {
    /// Current usage percentage (0.0 to 1.0)
    pub current_usage: f64,
    
    /// Average usage over last hour
    pub avg_usage: f64,
    
    /// Peak usage in last hour
    pub peak_usage: f64,
    
    /// Available capacity
    pub available_capacity: f64,
    
    /// Usage trend (positive = increasing)
    pub trend: f64,
}

/// Biological behavior health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalHealth {
    /// Biological systems status
    pub status: HealthStatus,
    
    /// Active biological roles
    pub active_roles: Vec<BiologicalRole>,
    
    /// Role performance metrics
    pub role_performance: HashMap<String, f64>,
    
    /// Trust relationship health
    pub trust_relationships: usize,
    
    /// Learning efficiency metrics
    pub learning_efficiency: f64,
    
    /// Cooperation success rate
    pub cooperation_rate: f64,
    
    /// Crisis response capability
    pub crisis_response: HealthStatus,
}

/// Performance health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHealth {
    /// Performance status
    pub status: HealthStatus,
    
    /// Task completion rate
    pub task_completion_rate: f64,
    
    /// Average task duration (seconds)
    pub avg_task_duration: f64,
    
    /// Throughput (tasks per second)
    pub throughput: f64,
    
    /// Error rate percentage
    pub error_rate: f64,
    
    /// Response time (ms)
    pub response_time_ms: f64,
    
    /// Queue depth
    pub queue_depth: usize,
}

impl HealthChecker {
    /// Create new health checker
    pub async fn new(config: &NodeConfig, biological_node: Arc<BiologicalNode>) -> Result<Self> {
        let health_server = if config.monitoring.enable_health_check {
            Some(Arc::new(HealthHttpServer::new(
                &config.monitoring.metrics_addr,
                config.monitoring.health_port,
                biological_node.clone()
            ).await.context("Failed to create health HTTP server")?))
        } else {
            None
        };
        
        Ok(Self {
            config: config.clone(),
            biological_node,
            health_server,
            last_health_check: Arc::new(tokio::sync::RwLock::new(None)),
            health_history: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        })
    }
    
    /// Start health checker background tasks
    pub async fn start(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Starting health checker");
        
        // Start HTTP server if enabled
        if let Some(ref server) = self.health_server {
            let server_shutdown = shutdown_rx.resubscribe();
            let server = server.clone();
            
            tokio::spawn(async move {
                if let Err(e) = server.start(server_shutdown).await {
                    error!("Health HTTP server failed: {}", e);
                }
            });
        }
        
        // Start health check loop
        let mut check_interval = interval(Duration::from_secs(30));
        
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Health checker shutting down");
                    break;
                }
                
                _ = check_interval.tick() => {
                    if let Err(e) = self.perform_health_check().await {
                        error!("Health check failed: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform comprehensive health check
    pub async fn perform_health_check(&self) -> Result<HealthCheckResult> {
        debug!("Performing comprehensive health check");
        
        let timestamp = chrono::Utc::now();
        
        // Check individual components
        let components = self.check_component_health().await?;
        
        // Check network health
        let network = self.check_network_health().await?;
        
        // Check resource health
        let resources = self.check_resource_health().await?;
        
        // Check biological behavior health
        let biological = self.check_biological_health().await?;
        
        // Check performance health
        let performance = self.check_performance_health().await?;
        
        // Determine overall health status
        let overall_status = self.determine_overall_status(&components, &network, &resources, &biological, &performance);
        
        let health_result = HealthCheckResult {
            timestamp,
            overall_status,
            components,
            network,
            resources,
            biological,
            performance,
        };
        
        // Store results
        {
            let mut last_check = self.last_health_check.write().await;
            *last_check = Some(health_result.clone());
        }
        
        // Add to history (keep last 100 checks)
        {
            let mut history = self.health_history.write().await;
            history.push(health_result.clone());
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        debug!("Health check completed with status: {:?}", health_result.overall_status);
        
        Ok(health_result)
    }
    
    /// Check component health status
    async fn check_component_health(&self) -> Result<HashMap<String, ComponentHealth>> {
        let mut components = HashMap::new();
        
        // Check biological node
        let bio_health = ComponentHealth {
            status: if self.biological_node.is_healthy().await {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: "Biological node operational".to_string(),
            last_success: Some(chrono::Utc::now()),
            error_count: 0,
            metrics: self.biological_node.get_health_metrics().await,
        };
        components.insert("biological_node".to_string(), bio_health);
        
        // Check P2P network
        let p2p_health = ComponentHealth {
            status: if self.biological_node.is_network_ready().await {
                HealthStatus::Healthy
            } else {
                HealthStatus::Unhealthy
            },
            message: "P2P network operational".to_string(),
            last_success: Some(chrono::Utc::now()),
            error_count: 0,
            metrics: self.biological_node.get_network_metrics().await,
        };
        components.insert("p2p_network".to_string(), p2p_health);
        
        // Check security systems
        let security_health = ComponentHealth {
            status: if self.biological_node.is_security_healthy().await {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
            message: "Security systems operational".to_string(),
            last_success: Some(chrono::Utc::now()),
            error_count: 0,
            metrics: self.biological_node.get_security_metrics().await,
        };
        components.insert("security".to_string(), security_health);
        
        Ok(components)
    }
    
    /// Check network connectivity health
    async fn check_network_health(&self) -> Result<NetworkHealth> {
        let network_info = self.biological_node.get_network_info().await;
        
        let status = if let Some(ref info) = network_info {
            if info.connected_peers >= 3 {
                HealthStatus::Healthy
            } else if info.connected_peers >= 1 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Unhealthy
            }
        } else {
            HealthStatus::Critical
        };
        
        let connected_peers = network_info.as_ref().map(|i| i.connected_peers).unwrap_or(0);
        
        Ok(NetworkHealth {
            status,
            connected_peers,
            avg_latency_ms: self.biological_node.get_average_latency().await.unwrap_or(0.0),
            packet_loss_percentage: self.biological_node.get_packet_loss().await.unwrap_or(0.0),
            bootstrap_connectivity: self.biological_node.get_bootstrap_connectivity().await.unwrap_or(0.0),
            address_resolution: if connected_peers > 0 { HealthStatus::Healthy } else { HealthStatus::Unhealthy },
            routing_efficiency: self.biological_node.get_routing_efficiency().await.unwrap_or(0.0),
        })
    }
    
    /// Check resource usage health
    async fn check_resource_health(&self) -> Result<ResourceHealth> {
        let resource_usage = self.biological_node.get_resource_usage().await;
        
        let status = if let Some(ref usage) = resource_usage {
            if usage.cpu_usage > 0.95 || usage.memory_usage > 0.95 || usage.disk_usage > 0.95 {
                HealthStatus::Critical
            } else if usage.cpu_usage > 0.8 || usage.memory_usage > 0.8 || usage.disk_usage > 0.8 {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            }
        } else {
            HealthStatus::Unhealthy
        };
        
        let default_usage = ResourceUsage {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            disk_usage: 0.0,
            network_usage: 0,
        };
        
        let usage = resource_usage.as_ref().unwrap_or(&default_usage);
        
        Ok(ResourceHealth {
            status,
            cpu: ResourceMetric {
                current_usage: usage.cpu_usage,
                avg_usage: self.biological_node.get_avg_cpu_usage().await.unwrap_or(0.0),
                peak_usage: self.biological_node.get_peak_cpu_usage().await.unwrap_or(0.0),
                available_capacity: 1.0 - usage.cpu_usage,
                trend: self.biological_node.get_cpu_trend().await.unwrap_or(0.0),
            },
            memory: ResourceMetric {
                current_usage: usage.memory_usage,
                avg_usage: self.biological_node.get_avg_memory_usage().await.unwrap_or(0.0),
                peak_usage: self.biological_node.get_peak_memory_usage().await.unwrap_or(0.0),
                available_capacity: 1.0 - usage.memory_usage,
                trend: self.biological_node.get_memory_trend().await.unwrap_or(0.0),
            },
            disk: ResourceMetric {
                current_usage: usage.disk_usage,
                avg_usage: self.biological_node.get_avg_disk_usage().await.unwrap_or(0.0),
                peak_usage: self.biological_node.get_peak_disk_usage().await.unwrap_or(0.0),
                available_capacity: 1.0 - usage.disk_usage,
                trend: self.biological_node.get_disk_trend().await.unwrap_or(0.0),
            },
            network_bandwidth: ResourceMetric {
                current_usage: usage.network_usage as f64 / (1024.0 * 1024.0), // Convert to MB/s
                avg_usage: self.biological_node.get_avg_network_usage().await.unwrap_or(0.0),
                peak_usage: self.biological_node.get_peak_network_usage().await.unwrap_or(0.0),
                available_capacity: self.biological_node.get_network_capacity().await.unwrap_or(100.0),
                trend: self.biological_node.get_network_trend().await.unwrap_or(0.0),
            },
            allocation_efficiency: self.biological_node.get_allocation_efficiency().await.unwrap_or(0.0),
        })
    }
    
    /// Check biological behavior health
    async fn check_biological_health(&self) -> Result<BiologicalHealth> {
        let active_roles = self.biological_node.get_active_roles().await;
        
        let status = if active_roles.is_empty() {
            HealthStatus::Unhealthy
        } else if active_roles.len() >= self.config.biological.preferred_roles.len() / 2 {
            HealthStatus::Healthy
        } else {
            HealthStatus::Degraded
        };
        
        let mut role_performance = HashMap::new();
        for role in &active_roles {
            let performance = self.biological_node.get_role_performance(role).await.unwrap_or(0.0);
            role_performance.insert(format!("{:?}", role), performance);
        }
        
        Ok(BiologicalHealth {
            status,
            active_roles,
            role_performance,
            trust_relationships: self.biological_node.get_trust_relationship_count().await.unwrap_or(0),
            learning_efficiency: self.biological_node.get_learning_efficiency().await.unwrap_or(0.0),
            cooperation_rate: self.biological_node.get_cooperation_rate().await.unwrap_or(0.0),
            crisis_response: if self.config.biological.enable_havoc {
                HealthStatus::Healthy
            } else {
                HealthStatus::Degraded
            },
        })
    }
    
    /// Check performance health
    async fn check_performance_health(&self) -> Result<PerformanceHealth> {
        let task_completion_rate = self.biological_node.get_task_completion_rate().await.unwrap_or(0.0);
        
        let status = if task_completion_rate > 0.95 {
            HealthStatus::Healthy
        } else if task_completion_rate > 0.8 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Unhealthy
        };
        
        Ok(PerformanceHealth {
            status,
            task_completion_rate,
            avg_task_duration: self.biological_node.get_avg_task_duration().await.unwrap_or(0.0),
            throughput: self.biological_node.get_throughput().await.unwrap_or(0.0),
            error_rate: self.biological_node.get_error_rate().await.unwrap_or(0.0),
            response_time_ms: self.biological_node.get_response_time().await.unwrap_or(0.0),
            queue_depth: self.biological_node.get_queue_depth().await.unwrap_or(0),
        })
    }
    
    /// Determine overall health status from component results
    fn determine_overall_status(
        &self,
        components: &HashMap<String, ComponentHealth>,
        network: &NetworkHealth,
        resources: &ResourceHealth,
        biological: &BiologicalHealth,
        performance: &PerformanceHealth,
    ) -> HealthStatus {
        // Count health status occurrences
        let mut status_counts = HashMap::new();
        
        // Add component statuses
        for component in components.values() {
            *status_counts.entry(component.status.clone()).or_insert(0) += 1;
        }
        
        // Add major subsystem statuses
        *status_counts.entry(network.status.clone()).or_insert(0) += 2; // Network is critical
        *status_counts.entry(resources.status.clone()).or_insert(0) += 2; // Resources are critical
        *status_counts.entry(biological.status.clone()).or_insert(0) += 1;
        *status_counts.entry(performance.status.clone()).or_insert(0) += 1;
        
        // Determine overall status with precedence: Critical > Unhealthy > Degraded > Healthy
        if status_counts.get(&HealthStatus::Critical).unwrap_or(&0) > &0 {
            HealthStatus::Critical
        } else if status_counts.get(&HealthStatus::Unhealthy).unwrap_or(&0) > &0 {
            HealthStatus::Unhealthy
        } else if status_counts.get(&HealthStatus::Degraded).unwrap_or(&0) > &1 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
    
    /// Get latest health check result
    pub async fn get_latest_health(&self) -> Option<HealthCheckResult> {
        self.last_health_check.read().await.clone()
    }
    
    /// Get health check history
    pub async fn get_health_history(&self, limit: Option<usize>) -> Vec<HealthCheckResult> {
        let history = self.health_history.read().await;
        
        match limit {
            Some(limit) => {
                let start = if history.len() > limit { history.len() - limit } else { 0 };
                history[start..].to_vec()
            }
            None => history.clone(),
        }
    }
    
    /// Check if node is healthy for readiness probe
    pub async fn is_ready(&self) -> bool {
        if let Some(health) = self.get_latest_health().await {
            matches!(health.overall_status, HealthStatus::Healthy | HealthStatus::Degraded)
        } else {
            false
        }
    }
    
    /// Check if node is alive for liveness probe
    pub async fn is_alive(&self) -> bool {
        if let Some(health) = self.get_latest_health().await {
            !matches!(health.overall_status, HealthStatus::Critical)
        } else {
            true // If no health check yet, assume alive
        }
    }
}

/// HTTP server for health endpoints
#[derive(Debug)]
pub struct HealthHttpServer {
    /// Server bind address
    addr: String,
    /// Server port
    port: u16,
    /// Reference to biological node
    biological_node: Arc<BiologicalNode>,
}

impl HealthHttpServer {
    /// Create new health HTTP server
    pub async fn new(addr: &str, port: u16, biological_node: Arc<BiologicalNode>) -> Result<Self> {
        Ok(Self {
            addr: addr.to_string(),
            port,
            biological_node,
        })
    }
    
    /// Start HTTP server
    pub async fn start(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        use warp::Filter;
        
        info!("Starting health HTTP server on {}:{}", self.addr, self.port);
        
        let biological_node = self.biological_node.clone();
        
        // Health endpoint
        let health = warp::path("health")
            .and(warp::get())
            .and_then(move || {
                let node = biological_node.clone();
                async move {
                    match node.get_health_status().await {
                        Ok(health) => Ok(warp::reply::json(&health)),
                        Err(_) => Err(warp::reject::not_found()),
                    }
                }
            });
        
        let biological_node = self.biological_node.clone();
        
        // Readiness endpoint
        let ready = warp::path("ready")
            .and(warp::get())
            .and_then(move || {
                let node = biological_node.clone();
                async move {
                    if node.is_network_ready().await {
                        Ok(warp::reply::with_status("OK", warp::http::StatusCode::OK))
                    } else {
                        Ok(warp::reply::with_status("Not Ready", warp::http::StatusCode::SERVICE_UNAVAILABLE))
                    }
                }
            });
        
        let biological_node = self.biological_node.clone();
        
        // Liveness endpoint
        let live = warp::path("live")
            .and(warp::get())
            .and_then(move || {
                let node = biological_node.clone();
                async move {
                    if node.is_healthy().await {
                        Ok(warp::reply::with_status("OK", warp::http::StatusCode::OK))
                    } else {
                        Ok(warp::reply::with_status("Unhealthy", warp::http::StatusCode::SERVICE_UNAVAILABLE))
                    }
                }
            });
        
        let routes = health.or(ready).or(live);
        
        let addr: std::net::SocketAddr = format!("{}:{}", self.addr, self.port).parse()
            .context("Invalid server address")?;
        
        let (_, server) = warp::serve(routes)
            .bind_with_graceful_shutdown(addr, async move {
                let _ = shutdown_rx.recv().await;
                info!("Health HTTP server shutting down");
            });
        
        server.await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeConfig;
    use tempfile::TempDir;
    
    async fn create_test_health_checker() -> (HealthChecker, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = NodeConfig::default();
        
        // Configure test paths
        config.storage.data_dir = temp_dir.path().join("data");
        config.storage.cache_dir = temp_dir.path().join("cache");
        config.storage.log_dir = temp_dir.path().join("logs");
        config.network.node_key_path = temp_dir.path().join("node_key.pem");
        config.monitoring.enable_health_check = false; // Disable HTTP server for tests
        
        // Create biological node
        let biological_node = Arc::new(BiologicalNode::new(&config).await.unwrap());
        
        let health_checker = HealthChecker::new(&config, biological_node).await.unwrap();
        
        (health_checker, temp_dir)
    }
    
    #[tokio::test]
    async fn test_health_checker_creation() {
        let (health_checker, _temp_dir) = create_test_health_checker().await;
        
        // Basic validation
        assert!(health_checker.last_health_check.read().await.is_none());
        assert!(health_checker.health_history.read().await.is_empty());
    }
    
    #[tokio::test]
    async fn test_health_check_execution() {
        let (health_checker, _temp_dir) = create_test_health_checker().await;
        
        let result = health_checker.perform_health_check().await;
        assert!(result.is_ok());
        
        let health_result = result.unwrap();
        assert!(!health_result.components.is_empty());
        
        // Check that health was stored
        let latest = health_checker.get_latest_health().await;
        assert!(latest.is_some());
    }
    
    #[tokio::test]
    async fn test_health_status_determination() {
        let (health_checker, _temp_dir) = create_test_health_checker().await;
        
        // Test with healthy components
        let mut components = HashMap::new();
        components.insert("test".to_string(), ComponentHealth {
            status: HealthStatus::Healthy,
            message: "Test component".to_string(),
            last_success: Some(chrono::Utc::now()),
            error_count: 0,
            metrics: HashMap::new(),
        });
        
        let network = NetworkHealth {
            status: HealthStatus::Healthy,
            connected_peers: 5,
            avg_latency_ms: 10.0,
            packet_loss_percentage: 0.0,
            bootstrap_connectivity: 1.0,
            address_resolution: HealthStatus::Healthy,
            routing_efficiency: 0.95,
        };
        
        let resources = ResourceHealth {
            status: HealthStatus::Healthy,
            cpu: ResourceMetric {
                current_usage: 0.5,
                avg_usage: 0.4,
                peak_usage: 0.7,
                available_capacity: 0.5,
                trend: 0.0,
            },
            memory: ResourceMetric {
                current_usage: 0.6,
                avg_usage: 0.5,
                peak_usage: 0.8,
                available_capacity: 0.4,
                trend: 0.0,
            },
            disk: ResourceMetric {
                current_usage: 0.3,
                avg_usage: 0.3,
                peak_usage: 0.5,
                available_capacity: 0.7,
                trend: 0.0,
            },
            network_bandwidth: ResourceMetric {
                current_usage: 50.0,
                avg_usage: 40.0,
                peak_usage: 80.0,
                available_capacity: 50.0,
                trend: 0.0,
            },
            allocation_efficiency: 0.92,
        };
        
        let biological = BiologicalHealth {
            status: HealthStatus::Healthy,
            active_roles: vec![BiologicalRole::CasteNode, BiologicalRole::YoungNode],
            role_performance: HashMap::new(),
            trust_relationships: 10,
            learning_efficiency: 0.85,
            cooperation_rate: 0.90,
            crisis_response: HealthStatus::Healthy,
        };
        
        let performance = PerformanceHealth {
            status: HealthStatus::Healthy,
            task_completion_rate: 0.98,
            avg_task_duration: 15.0,
            throughput: 100.0,
            error_rate: 0.02,
            response_time_ms: 50.0,
            queue_depth: 5,
        };
        
        let overall = health_checker.determine_overall_status(&components, &network, &resources, &biological, &performance);
        assert_eq!(overall, HealthStatus::Healthy);
    }
    
    #[tokio::test]
    async fn test_readiness_and_liveness() {
        let (health_checker, _temp_dir) = create_test_health_checker().await;
        
        // Before any health check
        assert!(!health_checker.is_ready().await);
        assert!(health_checker.is_alive().await); // Assumes alive until proven otherwise
        
        // After health check
        let _ = health_checker.perform_health_check().await;
        // Note: Actual readiness depends on biological node implementation
    }
}