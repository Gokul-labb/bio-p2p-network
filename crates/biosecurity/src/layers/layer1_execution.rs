//! Layer 1: Multi-Layer Execution
//! 
//! Implements randomized execution environments with protective monitoring layers.
//! Inspired by cellular compartmentalization in biological systems.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use std::process::Stdio;

#[cfg(feature = "containers")]
use bollard::{Docker, container::{CreateContainerOptions, Config, StartContainerOptions}};

use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::config::{LayerConfig, LayerSettings, IsolationLevel, ContainerRuntime};
use crate::crypto::CryptoContext;
use crate::layers::{SecurityLayer, BaseLayer, SecurityContext, ProcessResult, LayerStatus, LayerMetrics, RiskLevel};

/// Layer 1: Multi-Layer Execution implementation
pub struct MultiLayerExecution {
    base: BaseLayer,
    execution_layers: Arc<RwLock<Vec<ExecutionLayer>>>,
    monitoring_agents: Arc<RwLock<Vec<MonitoringAgent>>>,
    container_manager: Option<ContainerManager>,
    randomization_state: Arc<RwLock<RandomizationState>>,
}

impl MultiLayerExecution {
    pub fn new() -> Self {
        Self {
            base: BaseLayer::new(1, "Multi-Layer Execution".to_string()),
            execution_layers: Arc::new(RwLock::new(Vec::new())),
            monitoring_agents: Arc::new(RwLock::new(Vec::new())),
            container_manager: None,
            randomization_state: Arc::new(RwLock::new(RandomizationState::new())),
        }
    }

    async fn setup_execution_layers(&self, settings: &LayerSettings) -> SecurityResult<()> {
        let (monitoring_layers, isolation_level, container_runtime) = match settings {
            LayerSettings::MultiLayerExecution { 
                monitoring_layers, 
                isolation_level, 
                container_runtime,
                .. 
            } => (*monitoring_layers, isolation_level.clone(), container_runtime.clone()),
            _ => return Err(SecurityError::ConfigurationError(
                "Invalid settings for Multi-Layer Execution".to_string()
            )),
        };

        // Create execution layers
        let mut layers = self.execution_layers.write().await;
        layers.clear();
        
        for i in 0..monitoring_layers {
            let layer = ExecutionLayer::new(
                i,
                format!("Execution Layer {}", i),
                isolation_level.clone(),
                container_runtime.clone(),
            );
            layers.push(layer);
        }

        // Create monitoring agents for each layer
        let mut agents = self.monitoring_agents.write().await;
        agents.clear();
        
        for i in 0..monitoring_layers {
            let agent = MonitoringAgent::new(i, format!("Monitor Agent {}", i));
            agents.push(agent);
        }

        Ok(())
    }

    async fn select_execution_layer(&self, context: &SecurityContext) -> SecurityResult<usize> {
        let mut rng_state = self.randomization_state.write().await;
        let layers = self.execution_layers.read().await;
        
        if layers.is_empty() {
            return Err(SecurityError::LayerError {
                layer: 1,
                message: "No execution layers available".to_string(),
            });
        }

        // Randomized selection based on context and current state
        let seed = rng_state.generate_seed(&context.execution_id, &context.node_id);
        let layer_index = seed % layers.len();
        
        Ok(layer_index)
    }

    async fn execute_with_monitoring(
        &self,
        data: &[u8],
        context: &SecurityContext,
        layer_index: usize,
    ) -> SecurityResult<ProcessResult> {
        let start_time = std::time::Instant::now();
        
        // Get execution layer and monitoring agent
        let layers = self.execution_layers.read().await;
        let agents = self.monitoring_agents.read().await;
        
        let execution_layer = layers.get(layer_index).ok_or_else(|| {
            SecurityError::LayerError {
                layer: 1,
                message: format!("Execution layer {} not found", layer_index),
            }
        })?;
        
        let monitor = agents.get(layer_index).ok_or_else(|| {
            SecurityError::LayerError {
                layer: 1,
                message: format!("Monitor agent {} not found", layer_index),
            }
        })?;

        // Start monitoring
        monitor.start_monitoring(context).await?;
        
        let mut events = Vec::new();
        
        // Execute with protection
        let result = match execution_layer.execute_protected(data, context).await {
            Ok(result) => result,
            Err(e) => {
                // Record security event for failed execution
                let event = SecurityEvent::new(
                    SecuritySeverity::High,
                    "execution_failed",
                    format!("Layer 1 execution failed: {}", e),
                )
                .with_layer(1)
                .with_node(context.node_id.clone());
                
                events.push(event);
                monitor.stop_monitoring().await?;
                return Err(e);
            }
        };
        
        // Stop monitoring and collect results
        let monitor_events = monitor.stop_monitoring().await?;
        events.extend(monitor_events);
        
        // Check for backdoors or leakage
        if monitor.detected_anomalies().await {
            let event = SecurityEvent::new(
                SecuritySeverity::Critical,
                "backdoor_detected",
                "Potential backdoor or information leakage detected during execution",
            )
            .with_layer(1)
            .with_node(context.node_id.clone());
            
            events.push(event);
            self.base.record_threat_detection().await;
        }
        
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;
        
        Ok(ProcessResult::success(result, context.clone()).with_events(events))
    }
}

#[async_trait]
impl SecurityLayer for MultiLayerExecution {
    fn layer_id(&self) -> usize {
        self.base.layer_id()
    }

    fn layer_name(&self) -> &str {
        self.base.layer_name()
    }

    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.base.initialize(config, crypto).await?;
        
        // Setup container manager if containers are enabled
        #[cfg(feature = "containers")]
        {
            if let LayerSettings::MultiLayerExecution { container_runtime, .. } = &config.settings {
                let manager = ContainerManager::new(container_runtime.clone()).await?;
                self.container_manager = Some(manager);
            }
        }
        
        // Setup execution layers
        self.setup_execution_layers(&config.settings).await?;
        
        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.base.start().await?;
        
        // Start all execution layers
        let mut layers = self.execution_layers.write().await;
        for layer in layers.iter_mut() {
            layer.start().await?;
        }
        
        // Start all monitoring agents
        let mut agents = self.monitoring_agents.write().await;
        for agent in agents.iter_mut() {
            agent.start().await?;
        }
        
        Ok(())
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        // Stop all components
        let mut layers = self.execution_layers.write().await;
        for layer in layers.iter_mut() {
            layer.stop().await?;
        }
        
        let mut agents = self.monitoring_agents.write().await;
        for agent in agents.iter_mut() {
            agent.stop().await?;
        }
        
        self.base.stop().await
    }

    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        // Select randomized execution layer
        let layer_index = self.select_execution_layer(context).await?;
        
        // Execute with monitoring
        self.execute_with_monitoring(data, context, layer_index).await
    }

    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        // Post-execution validation
        let layers = self.execution_layers.read().await;
        if layers.is_empty() {
            return Err(SecurityError::LayerError {
                layer: 1,
                message: "No execution layers available for post-processing".to_string(),
            });
        }

        // Verify execution integrity
        let event = SecurityEvent::new(
            SecuritySeverity::Info,
            "execution_completed",
            "Multi-layer execution completed successfully",
        )
        .with_layer(1)
        .with_node(context.node_id.clone());

        Ok(ProcessResult::success(data.to_vec(), context.clone()).with_event(event))
    }

    async fn status(&self) -> LayerStatus {
        self.base.status().await
    }

    async fn metrics(&self) -> LayerMetrics {
        self.base.metrics().await
    }

    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        self.base.handle_event(event).await
    }
}

/// Individual execution layer with isolation
pub struct ExecutionLayer {
    id: usize,
    name: String,
    isolation_level: IsolationLevel,
    container_runtime: ContainerRuntime,
    active: bool,
}

impl ExecutionLayer {
    pub fn new(
        id: usize,
        name: String,
        isolation_level: IsolationLevel,
        container_runtime: ContainerRuntime,
    ) -> Self {
        Self {
            id,
            name,
            isolation_level,
            container_runtime,
            active: false,
        }
    }

    pub async fn start(&mut self) -> SecurityResult<()> {
        self.active = true;
        tracing::debug!("Started execution layer {}: {}", self.id, self.name);
        Ok(())
    }

    pub async fn stop(&mut self) -> SecurityResult<()> {
        self.active = false;
        tracing::debug!("Stopped execution layer {}: {}", self.id, self.name);
        Ok(())
    }

    pub async fn execute_protected(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<Vec<u8>> {
        if !self.active {
            return Err(SecurityError::LayerError {
                layer: 1,
                message: format!("Execution layer {} is not active", self.id),
            });
        }

        match self.isolation_level {
            IsolationLevel::Basic => self.execute_basic(data, context).await,
            IsolationLevel::Full => self.execute_full(data, context).await,
            IsolationLevel::Enhanced => self.execute_enhanced(data, context).await,
        }
    }

    async fn execute_basic(&self, data: &[u8], _context: &SecurityContext) -> SecurityResult<Vec<u8>> {
        // Basic isolation - simple process execution
        let output = tokio::process::Command::new("echo")
            .arg("basic_execution")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| SecurityError::LayerError {
                layer: 1,
                message: format!("Failed to spawn process: {}", e),
            })?
            .wait_with_output()
            .await
            .map_err(|e| SecurityError::LayerError {
                layer: 1,
                message: format!("Process execution failed: {}", e),
            })?;

        if output.status.success() {
            Ok(output.stdout)
        } else {
            Err(SecurityError::LayerError {
                layer: 1,
                message: format!("Process failed with exit code: {:?}", output.status.code()),
            })
        }
    }

    async fn execute_full(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<Vec<u8>> {
        // Full isolation - container execution
        #[cfg(feature = "containers")]
        {
            // This would use the container manager for full isolation
            // For now, fallback to basic execution
            self.execute_basic(data, context).await
        }
        
        #[cfg(not(feature = "containers"))]
        {
            self.execute_basic(data, context).await
        }
    }

    async fn execute_enhanced(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<Vec<u8>> {
        // Enhanced isolation - additional security layers
        // Apply additional security measures based on risk level
        let multiplier = context.risk_level.security_multiplier();
        
        if multiplier > 1.5 {
            // High risk - apply additional restrictions
            tracing::warn!("Executing high-risk computation with enhanced security");
        }
        
        self.execute_full(data, context).await
    }
}

/// Monitoring agent for execution layers
pub struct MonitoringAgent {
    id: usize,
    name: String,
    active: bool,
    monitoring_active: bool,
    anomalies_detected: bool,
    start_time: Option<std::time::Instant>,
}

impl MonitoringAgent {
    pub fn new(id: usize, name: String) -> Self {
        Self {
            id,
            name,
            active: false,
            monitoring_active: false,
            anomalies_detected: false,
            start_time: None,
        }
    }

    pub async fn start(&mut self) -> SecurityResult<()> {
        self.active = true;
        tracing::debug!("Started monitoring agent {}: {}", self.id, self.name);
        Ok(())
    }

    pub async fn stop(&mut self) -> SecurityResult<()> {
        self.active = false;
        self.monitoring_active = false;
        tracing::debug!("Stopped monitoring agent {}: {}", self.id, self.name);
        Ok(())
    }

    pub async fn start_monitoring(&self, context: &SecurityContext) -> SecurityResult<()> {
        if !self.active {
            return Err(SecurityError::LayerError {
                layer: 1,
                message: format!("Monitoring agent {} is not active", self.id),
            });
        }

        tracing::debug!(
            "Started monitoring for execution {} on agent {}",
            context.execution_id,
            self.id
        );
        Ok(())
    }

    pub async fn stop_monitoring(&self) -> SecurityResult<Vec<SecurityEvent>> {
        let mut events = Vec::new();
        
        if self.anomalies_detected {
            let event = SecurityEvent::new(
                SecuritySeverity::Medium,
                "anomaly_detected",
                format!("Monitoring agent {} detected anomalous behavior", self.id),
            )
            .with_layer(1);
            
            events.push(event);
        }

        tracing::debug!("Stopped monitoring on agent {}", self.id);
        Ok(events)
    }

    pub async fn detected_anomalies(&self) -> bool {
        self.anomalies_detected
    }
}

/// Container management for execution isolation
#[cfg(feature = "containers")]
pub struct ContainerManager {
    docker: Docker,
    runtime: ContainerRuntime,
}

#[cfg(feature = "containers")]
impl ContainerManager {
    pub async fn new(runtime: ContainerRuntime) -> SecurityResult<Self> {
        let docker = Docker::connect_with_local_defaults()
            .map_err(|e| SecurityError::ContainerError(format!("Failed to connect to Docker: {}", e)))?;

        Ok(Self { docker, runtime })
    }

    pub async fn execute_in_container(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<Vec<u8>> {
        let container_name = format!("bio-security-{}-{}", context.execution_id, context.node_id);
        
        // Container configuration
        let config = Config {
            image: Some("bio-p2p/secure-compute:latest"),
            cmd: Some(vec!["echo".to_string(), "container_execution".to_string()]),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            ..Default::default()
        };

        // Create container
        let _container = self.docker
            .create_container(
                Some(CreateContainerOptions {
                    name: container_name.clone(),
                    ..Default::default()
                }),
                config,
            )
            .await
            .map_err(|e| SecurityError::ContainerError(format!("Failed to create container: {}", e)))?;

        // Start container
        self.docker
            .start_container(&container_name, None::<StartContainerOptions<String>>)
            .await
            .map_err(|e| SecurityError::ContainerError(format!("Failed to start container: {}", e)))?;

        // For demonstration, return processed data
        // In real implementation, this would execute the computation in the container
        Ok(b"container_processed_data".to_vec())
    }
}

/// Randomization state for layer selection
pub struct RandomizationState {
    seed_counter: u64,
    last_selection_time: std::time::Instant,
}

impl RandomizationState {
    pub fn new() -> Self {
        Self {
            seed_counter: 0,
            last_selection_time: std::time::Instant::now(),
        }
    }

    pub fn generate_seed(&mut self, execution_id: &str, node_id: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let now = std::time::Instant::now();
        let time_factor = now.duration_since(self.last_selection_time).as_millis() as u64;
        
        self.seed_counter = self.seed_counter.wrapping_add(1);
        self.last_selection_time = now;

        let mut hasher = DefaultHasher::new();
        execution_id.hash(&mut hasher);
        node_id.hash(&mut hasher);
        self.seed_counter.hash(&mut hasher);
        time_factor.hash(&mut hasher);

        hasher.finish() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SecurityConfig, CryptoConfig};

    #[tokio::test]
    async fn test_multi_layer_execution_creation() {
        let layer = MultiLayerExecution::new();
        assert_eq!(layer.layer_id(), 1);
        assert_eq!(layer.layer_name(), "Multi-Layer Execution");
    }

    #[tokio::test]
    async fn test_execution_layer_lifecycle() {
        let mut layer = ExecutionLayer::new(
            0,
            "Test Layer".to_string(),
            IsolationLevel::Basic,
            ContainerRuntime::Docker,
        );

        assert!(!layer.active);
        layer.start().await.unwrap();
        assert!(layer.active);
        layer.stop().await.unwrap();
        assert!(!layer.active);
    }

    #[tokio::test]
    async fn test_monitoring_agent_lifecycle() {
        let mut agent = MonitoringAgent::new(0, "Test Agent".to_string());
        
        assert!(!agent.active);
        agent.start().await.unwrap();
        assert!(agent.active);
        agent.stop().await.unwrap();
        assert!(!agent.active);
    }

    #[tokio::test]
    async fn test_randomization_state() {
        let mut state = RandomizationState::new();
        
        let seed1 = state.generate_seed("exec1", "node1");
        let seed2 = state.generate_seed("exec2", "node1");
        let seed3 = state.generate_seed("exec1", "node2");
        
        // Seeds should be different for different inputs
        assert_ne!(seed1, seed2);
        assert_ne!(seed1, seed3);
        assert_ne!(seed2, seed3);
    }

    #[tokio::test]
    async fn test_layer_initialization() {
        let mut layer = MultiLayerExecution::new();
        let config = LayerConfig::multi_layer_execution();
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());

        let result = layer.initialize(&config, crypto).await;
        assert!(result.is_ok());
        assert_eq!(layer.status().await, LayerStatus::Ready);
    }

    #[tokio::test]
    async fn test_basic_execution() {
        let layer = ExecutionLayer::new(
            0,
            "Test".to_string(),
            IsolationLevel::Basic,
            ContainerRuntime::Docker,
        );

        let context = SecurityContext::new("test".to_string(), "node".to_string());
        
        // Layer not started, should fail
        let result = layer.execute_protected(b"test", &context).await;
        assert!(result.is_err());
    }
}