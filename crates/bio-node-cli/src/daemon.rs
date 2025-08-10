//! Main daemon implementation for Bio P2P Node
//!
//! This module coordinates all system components including P2P networking,
//! biological behaviors, security, resource management, and monitoring.

use anyhow::{anyhow, Context, Result};
use tokio::{select, signal, sync::broadcast, task::JoinHandle};
use tracing::{error, info, warn, debug, span, Level};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::collections::HashMap;

use crate::{
    config::{NodeConfig, BiologicalRole},
    health::HealthChecker,
    metrics::MetricsCollector,
    node::BiologicalNode,
    signals::SignalHandler,
};

/// Main daemon coordinator for Bio P2P Node
#[derive(Debug)]
pub struct BioNodeDaemon {
    /// Node configuration
    config: NodeConfig,
    
    /// P2P biological node instance
    biological_node: Arc<BiologicalNode>,
    
    /// Health checking system
    health_checker: Arc<HealthChecker>,
    
    /// Metrics collection system
    metrics_collector: Arc<MetricsCollector>,
    
    /// Signal handler for graceful shutdown
    signal_handler: SignalHandler,
    
    /// Shutdown coordination
    shutdown_tx: broadcast::Sender<()>,
    shutdown_rx: broadcast::Receiver<()>,
    
    /// Running state flag
    is_running: Arc<AtomicBool>,
    
    /// Background task handles
    tasks: Vec<JoinHandle<Result<()>>>,
    
    /// Component status tracking
    component_status: Arc<tokio::sync::RwLock<HashMap<String, ComponentStatus>>>,
}

/// Status of individual daemon components
#[derive(Debug, Clone, PartialEq)]
pub enum ComponentStatus {
    /// Component is starting up
    Starting,
    /// Component is running normally
    Running,
    /// Component is shutting down
    Stopping,
    /// Component has stopped
    Stopped,
    /// Component failed with error
    Failed(String),
}

/// Daemon startup phases
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StartupPhase {
    /// Initializing configuration and directories
    Initialization,
    /// Starting P2P networking
    NetworkStartup,
    /// Activating biological behaviors
    BiologicalActivation,
    /// Starting security systems
    SecurityActivation,
    /// Starting monitoring systems
    MonitoringStartup,
    /// Daemon fully operational
    Ready,
}

impl BioNodeDaemon {
    /// Create new Bio P2P Node daemon
    pub async fn new(config: NodeConfig) -> Result<Self> {
        let span = span!(Level::INFO, "daemon_init");
        let _enter = span.enter();
        
        info!("Initializing Bio P2P Node daemon");
        
        // Validate configuration
        config.validate()
            .context("Configuration validation failed")?;
        
        // Ensure required directories exist
        config.ensure_directories().await
            .context("Failed to create required directories")?;
        
        // Initialize shutdown coordination
        let (shutdown_tx, shutdown_rx) = broadcast::channel(16);
        let is_running = Arc::new(AtomicBool::new(false));
        
        // Initialize components
        let biological_node = Arc::new(BiologicalNode::new(&config).await
            .context("Failed to create biological node")?);
        
        let health_checker = Arc::new(HealthChecker::new(&config, biological_node.clone()).await
            .context("Failed to create health checker")?);
        
        let metrics_collector = Arc::new(MetricsCollector::new(&config, biological_node.clone()).await
            .context("Failed to create metrics collector")?);
        
        let signal_handler = SignalHandler::new(shutdown_tx.clone())
            .context("Failed to create signal handler")?;
        
        let component_status = Arc::new(tokio::sync::RwLock::new(HashMap::new()));
        
        // Initialize component status
        {
            let mut status = component_status.write().await;
            status.insert("biological_node".to_string(), ComponentStatus::Stopped);
            status.insert("health_checker".to_string(), ComponentStatus::Stopped);
            status.insert("metrics_collector".to_string(), ComponentStatus::Stopped);
            status.insert("signal_handler".to_string(), ComponentStatus::Stopped);
        }
        
        info!("Daemon initialization completed successfully");
        
        Ok(Self {
            config,
            biological_node,
            health_checker,
            metrics_collector,
            signal_handler,
            shutdown_tx,
            shutdown_rx,
            is_running,
            tasks: Vec::new(),
            component_status,
        })
    }
    
    /// Start the daemon and all components
    pub async fn start(&mut self) -> Result<()> {
        let span = span!(Level::INFO, "daemon_start");
        let _enter = span.enter();
        
        info!("Starting Bio P2P Node daemon");
        
        if self.is_running.load(Ordering::SeqCst) {
            return Err(anyhow!("Daemon is already running"));
        }
        
        // Set running state
        self.is_running.store(true, Ordering::SeqCst);
        
        // Start components in phases
        self.run_startup_phase(StartupPhase::Initialization).await?;
        self.run_startup_phase(StartupPhase::NetworkStartup).await?;
        self.run_startup_phase(StartupPhase::BiologicalActivation).await?;
        self.run_startup_phase(StartupPhase::SecurityActivation).await?;
        self.run_startup_phase(StartupPhase::MonitoringStartup).await?;
        self.run_startup_phase(StartupPhase::Ready).await?;
        
        info!("Bio P2P Node daemon started successfully");
        
        Ok(())
    }
    
    /// Run a specific startup phase
    async fn run_startup_phase(&mut self, phase: StartupPhase) -> Result<()> {
        info!("Entering startup phase: {:?}", phase);
        
        match phase {
            StartupPhase::Initialization => {
                // Create PID file
                self.create_pid_file().await?;
                
                // Initialize node key if needed
                self.ensure_node_key().await?;
                
                info!("Initialization phase completed");
            }
            
            StartupPhase::NetworkStartup => {
                // Start P2P networking
                self.update_component_status("biological_node", ComponentStatus::Starting).await;
                
                let node_handle = {
                    let node = self.biological_node.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        node.start_p2p_network(shutdown_rx).await
                            .context("Failed to start P2P network")
                    })
                };
                
                self.tasks.push(node_handle);
                
                // Wait for network to be ready
                tokio::time::timeout(
                    self.config.daemon.startup_timeout,
                    self.wait_for_network_ready()
                ).await
                    .context("Network startup timeout")?
                    .context("Network startup failed")?;
                
                self.update_component_status("biological_node", ComponentStatus::Running).await;
                
                info!("Network startup phase completed");
            }
            
            StartupPhase::BiologicalActivation => {
                // Activate biological behaviors
                let behaviors_handle = {
                    let node = self.biological_node.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        node.activate_biological_behaviors(shutdown_rx).await
                            .context("Failed to activate biological behaviors")
                    })
                };
                
                self.tasks.push(behaviors_handle);
                
                // Start resource management
                let resource_handle = {
                    let node = self.biological_node.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        node.start_resource_management(shutdown_rx).await
                            .context("Failed to start resource management")
                    })
                };
                
                self.tasks.push(resource_handle);
                
                info!("Biological activation phase completed");
            }
            
            StartupPhase::SecurityActivation => {
                // Start security systems
                let security_handle = {
                    let node = self.biological_node.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        node.start_security_systems(shutdown_rx).await
                            .context("Failed to start security systems")
                    })
                };
                
                self.tasks.push(security_handle);
                
                info!("Security activation phase completed");
            }
            
            StartupPhase::MonitoringStartup => {
                // Start health checker
                self.update_component_status("health_checker", ComponentStatus::Starting).await;
                
                let health_handle = {
                    let checker = self.health_checker.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        checker.start(shutdown_rx).await
                            .context("Failed to start health checker")
                    })
                };
                
                self.tasks.push(health_handle);
                self.update_component_status("health_checker", ComponentStatus::Running).await;
                
                // Start metrics collector
                self.update_component_status("metrics_collector", ComponentStatus::Starting).await;
                
                let metrics_handle = {
                    let collector = self.metrics_collector.clone();
                    let shutdown_rx = self.shutdown_tx.subscribe();
                    
                    tokio::spawn(async move {
                        collector.start(shutdown_rx).await
                            .context("Failed to start metrics collector")
                    })
                };
                
                self.tasks.push(metrics_handle);
                self.update_component_status("metrics_collector", ComponentStatus::Running).await;
                
                info!("Monitoring startup phase completed");
            }
            
            StartupPhase::Ready => {
                // Start signal handler
                self.update_component_status("signal_handler", ComponentStatus::Starting).await;
                
                let signal_handle = {
                    let handler = self.signal_handler.clone();
                    
                    tokio::spawn(async move {
                        handler.run().await
                            .context("Signal handler failed")
                    })
                };
                
                self.tasks.push(signal_handle);
                self.update_component_status("signal_handler", ComponentStatus::Running).await;
                
                // Log daemon ready status
                self.log_ready_status().await;
                
                info!("Bio P2P Node daemon is ready and operational");
            }
        }
        
        Ok(())
    }
    
    /// Run the daemon event loop
    pub async fn run(&mut self) -> Result<()> {
        let span = span!(Level::INFO, "daemon_run");
        let _enter = span.enter();
        
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(anyhow!("Daemon is not running"));
        }
        
        info!("Entering daemon event loop");
        
        // Main event loop
        loop {
            select! {
                // Handle shutdown signal
                _ = self.shutdown_rx.recv() => {
                    info!("Shutdown signal received");
                    break;
                }
                
                // Check for failed tasks
                result = futures::future::select_all(&mut self.tasks) => {
                    let (task_result, index, remaining_tasks) = result;
                    self.tasks = remaining_tasks;
                    
                    match task_result {
                        Ok(Ok(())) => {
                            debug!("Task {} completed successfully", index);
                        }
                        Ok(Err(e)) => {
                            error!("Task {} failed: {}", index, e);
                            self.handle_task_failure(index, e).await;
                        }
                        Err(e) => {
                            error!("Task {} panicked: {}", index, e);
                            self.handle_task_panic(index, e).await;
                        }
                    }
                    
                    // If all critical tasks have failed, initiate shutdown
                    if self.tasks.is_empty() {
                        warn!("All tasks completed, initiating shutdown");
                        break;
                    }
                }
                
                // Periodic health checks
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(30)) => {
                    self.perform_health_check().await;
                }
            }
        }
        
        info!("Exiting daemon event loop");
        Ok(())
    }
    
    /// Gracefully shut down the daemon
    pub async fn shutdown(&mut self) -> Result<()> {
        let span = span!(Level::INFO, "daemon_shutdown");
        let _enter = span.enter();
        
        info!("Shutting down Bio P2P Node daemon");
        
        if !self.is_running.load(Ordering::SeqCst) {
            info!("Daemon is not running, shutdown already complete");
            return Ok(());
        }
        
        // Update running state
        self.is_running.store(false, Ordering::SeqCst);
        
        // Send shutdown signal to all components
        if let Err(e) = self.shutdown_tx.send(()) {
            warn!("Failed to send shutdown signal: {}", e);
        }
        
        // Update component statuses
        self.update_all_component_status(ComponentStatus::Stopping).await;
        
        // Wait for tasks to complete with timeout
        let shutdown_timeout = self.config.daemon.shutdown_timeout;
        let shutdown_future = async {
            futures::future::join_all(self.tasks.drain(..)).await;
        };
        
        if let Err(_) = tokio::time::timeout(shutdown_timeout, shutdown_future).await {
            warn!("Shutdown timeout reached, forcing termination");
        }
        
        // Final cleanup
        self.cleanup().await?;
        
        // Update final status
        self.update_all_component_status(ComponentStatus::Stopped).await;
        
        info!("Bio P2P Node daemon shutdown completed");
        Ok(())
    }
    
    /// Get current daemon status
    pub async fn get_status(&self) -> DaemonStatus {
        let component_status = self.component_status.read().await.clone();
        
        DaemonStatus {
            is_running: self.is_running.load(Ordering::SeqCst),
            components: component_status,
            network_info: self.biological_node.get_network_info().await,
            resource_usage: self.biological_node.get_resource_usage().await,
            active_roles: self.biological_node.get_active_roles().await,
        }
    }
    
    /// Wait for network to be ready
    async fn wait_for_network_ready(&self) -> Result<()> {
        let mut attempts = 0;
        const MAX_ATTEMPTS: u32 = 30;
        
        while attempts < MAX_ATTEMPTS {
            if self.biological_node.is_network_ready().await {
                return Ok(());
            }
            
            attempts += 1;
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        
        Err(anyhow!("Network failed to become ready within timeout"))
    }
    
    /// Create PID file
    async fn create_pid_file(&self) -> Result<()> {
        let pid = std::process::id();
        let pid_content = pid.to_string();
        
        tokio::fs::write(&self.config.daemon.pid_file, pid_content).await
            .with_context(|| format!("Failed to create PID file: {}", self.config.daemon.pid_file.display()))?;
        
        info!("Created PID file: {} (PID: {})", self.config.daemon.pid_file.display(), pid);
        Ok(())
    }
    
    /// Ensure node key exists or create it
    async fn ensure_node_key(&self) -> Result<()> {
        if !self.config.network.node_key_path.exists() {
            info!("Generating new node key: {}", self.config.network.node_key_path.display());
            
            // Generate new Ed25519 key pair
            let keypair = libp2p::identity::Keypair::generate_ed25519();
            let private_key = keypair.to_protobuf_encoding()
                .context("Failed to encode private key")?;
            
            // Write to file with appropriate permissions
            tokio::fs::write(&self.config.network.node_key_path, private_key).await
                .context("Failed to write node key file")?;
            
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = tokio::fs::metadata(&self.config.network.node_key_path).await
                    .context("Failed to get key file metadata")?
                    .permissions();
                perms.set_mode(0o600);
                tokio::fs::set_permissions(&self.config.network.node_key_path, perms).await
                    .context("Failed to set key file permissions")?;
            }
            
            info!("Node key generated successfully");
        }
        
        Ok(())
    }
    
    /// Update component status
    async fn update_component_status(&self, component: &str, status: ComponentStatus) {
        let mut status_map = self.component_status.write().await;
        status_map.insert(component.to_string(), status);
    }
    
    /// Update all component statuses
    async fn update_all_component_status(&self, status: ComponentStatus) {
        let mut status_map = self.component_status.write().await;
        for (_, component_status) in status_map.iter_mut() {
            *component_status = status.clone();
        }
    }
    
    /// Handle task failure
    async fn handle_task_failure(&self, task_index: usize, error: anyhow::Error) {
        error!("Critical task {} failed: {}", task_index, error);
        
        // Determine which component failed and update status
        match task_index {
            0 => self.update_component_status("biological_node", ComponentStatus::Failed(error.to_string())).await,
            1 => self.update_component_status("health_checker", ComponentStatus::Failed(error.to_string())).await,
            2 => self.update_component_status("metrics_collector", ComponentStatus::Failed(error.to_string())).await,
            3 => self.update_component_status("signal_handler", ComponentStatus::Failed(error.to_string())).await,
            _ => warn!("Unknown task index: {}", task_index),
        }
        
        // For critical components, initiate shutdown
        if task_index <= 1 {
            warn!("Critical component failed, initiating emergency shutdown");
            let _ = self.shutdown_tx.send(());
        }
    }
    
    /// Handle task panic
    async fn handle_task_panic(&self, task_index: usize, error: tokio::task::JoinError) {
        error!("Critical task {} panicked: {}", task_index, error);
        
        // All panics are considered critical
        warn!("Task panic detected, initiating emergency shutdown");
        let _ = self.shutdown_tx.send(());
    }
    
    /// Perform periodic health checks
    async fn perform_health_check(&self) {
        debug!("Performing periodic health check");
        
        // Check component health
        let status = self.get_status().await;
        
        for (component, status) in &status.components {
            match status {
                ComponentStatus::Failed(error) => {
                    warn!("Component {} is in failed state: {}", component, error);
                }
                ComponentStatus::Running => {
                    debug!("Component {} is healthy", component);
                }
                _ => {
                    debug!("Component {} status: {:?}", component, status);
                }
            }
        }
        
        // Log resource usage
        if let Some(ref resource_usage) = status.resource_usage {
            if resource_usage.cpu_usage > 0.9 {
                warn!("High CPU usage detected: {:.1}%", resource_usage.cpu_usage * 100.0);
            }
            
            if resource_usage.memory_usage > 0.9 {
                warn!("High memory usage detected: {:.1}%", resource_usage.memory_usage * 100.0);
            }
        }
    }
    
    /// Log ready status information
    async fn log_ready_status(&self) {
        let status = self.get_status().await;
        
        info!("=== Bio P2P Node Ready ===");
        
        if let Some(ref network_info) = status.network_info {
            info!("Local Peer ID: {}", network_info.local_peer_id);
            info!("Listen Addresses: {:?}", network_info.listen_addresses);
            info!("Connected Peers: {}", network_info.connected_peers);
        }
        
        info!("Active Biological Roles: {:?}", status.active_roles);
        
        if let Some(ref resource_usage) = status.resource_usage {
            info!("Resource Usage - CPU: {:.1}%, Memory: {:.1}%, Disk: {:.1}%", 
                resource_usage.cpu_usage * 100.0,
                resource_usage.memory_usage * 100.0,
                resource_usage.disk_usage * 100.0
            );
        }
        
        if self.config.monitoring.enable_health_check {
            info!("Health Check: http://{}:{}/health", 
                "localhost", self.config.monitoring.health_port);
        }
        
        if self.config.monitoring.enable_metrics {
            info!("Metrics: http://{}:{}/metrics", 
                self.config.monitoring.metrics_addr, self.config.monitoring.metrics_port);
        }
        
        info!("=== Ready for P2P Computing ===");
    }
    
    /// Cleanup daemon resources
    async fn cleanup(&self) -> Result<()> {
        // Remove PID file
        if self.config.daemon.pid_file.exists() {
            if let Err(e) = tokio::fs::remove_file(&self.config.daemon.pid_file).await {
                warn!("Failed to remove PID file: {}", e);
            } else {
                info!("Removed PID file: {}", self.config.daemon.pid_file.display());
            }
        }
        
        // Additional cleanup as needed
        
        Ok(())
    }
}

/// Overall daemon status information
#[derive(Debug, Clone)]
pub struct DaemonStatus {
    /// Whether daemon is running
    pub is_running: bool,
    /// Component status map
    pub components: HashMap<String, ComponentStatus>,
    /// Network information
    pub network_info: Option<NetworkInfo>,
    /// Resource usage information
    pub resource_usage: Option<ResourceUsage>,
    /// Currently active biological roles
    pub active_roles: Vec<BiologicalRole>,
}

/// Network status information
#[derive(Debug, Clone)]
pub struct NetworkInfo {
    /// Local peer ID
    pub local_peer_id: String,
    /// Listen addresses
    pub listen_addresses: Vec<String>,
    /// Number of connected peers
    pub connected_peers: usize,
    /// Network protocol version
    pub protocol_version: String,
}

/// Resource usage information
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    /// CPU usage percentage (0.0 to 1.0)
    pub cpu_usage: f64,
    /// Memory usage percentage (0.0 to 1.0)
    pub memory_usage: f64,
    /// Disk usage percentage (0.0 to 1.0)
    pub disk_usage: f64,
    /// Network bandwidth usage (bytes/sec)
    pub network_usage: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NodeConfig, NetworkConfig};
    use tempfile::TempDir;
    
    async fn create_test_config() -> (NodeConfig, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = NodeConfig::default();
        
        // Set test paths
        config.storage.data_dir = temp_dir.path().join("data");
        config.storage.cache_dir = temp_dir.path().join("cache");
        config.storage.log_dir = temp_dir.path().join("logs");
        config.daemon.pid_file = temp_dir.path().join("test.pid");
        config.network.node_key_path = temp_dir.path().join("node_key.pem");
        
        // Use random ports to avoid conflicts
        config.network.listen_addresses = vec!["/ip4/127.0.0.1/tcp/0".parse().unwrap()];
        config.monitoring.health_port = 0; // Let OS assign
        config.monitoring.metrics_port = 0; // Let OS assign
        
        (config, temp_dir)
    }
    
    #[tokio::test]
    async fn test_daemon_creation() {
        let (config, _temp_dir) = create_test_config().await;
        
        let daemon = BioNodeDaemon::new(config).await;
        assert!(daemon.is_ok());
        
        let daemon = daemon.unwrap();
        assert!(!daemon.is_running.load(Ordering::SeqCst));
    }
    
    #[tokio::test]
    async fn test_daemon_status() {
        let (config, _temp_dir) = create_test_config().await;
        
        let daemon = BioNodeDaemon::new(config).await.unwrap();
        let status = daemon.get_status().await;
        
        assert!(!status.is_running);
        assert!(!status.components.is_empty());
    }
    
    #[tokio::test]
    async fn test_pid_file_creation() {
        let (config, temp_dir) = create_test_config().await;
        
        let daemon = BioNodeDaemon::new(config).await.unwrap();
        daemon.create_pid_file().await.unwrap();
        
        assert!(temp_dir.path().join("test.pid").exists());
    }
    
    #[tokio::test]
    async fn test_node_key_generation() {
        let (config, temp_dir) = create_test_config().await;
        
        let daemon = BioNodeDaemon::new(config).await.unwrap();
        daemon.ensure_node_key().await.unwrap();
        
        assert!(temp_dir.path().join("node_key.pem").exists());
    }
}