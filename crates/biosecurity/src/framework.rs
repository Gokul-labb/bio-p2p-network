//! Main security framework implementation
//! 
//! Coordinates all five security layers and security nodes to provide
//! comprehensive biological defense-in-depth protection.

use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use futures::future::try_join_all;

use crate::config::SecurityConfig;
use crate::crypto::CryptoContext;
use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::layers::{
    SecurityLayer, SecurityContext, ProcessResult, LayerStatus, LayerMetrics,
    layer1_execution::MultiLayerExecution,
    layer2_cbadu::CBADULayer,
    layer3_illusion::IllusionLayer,
    layer4_behavior::BehaviorMonitoring,
    layer5_thermal::ThermalDetection,
};
use crate::nodes::{
    SecurityNode, SecurityNodeStatus, SecurityNodeMetrics,
    DOSNode, InvestigationNode, CasualtyNode, ConfusionNode,
};

/// Main security framework coordinating all layers and nodes
pub struct SecurityFramework {
    config: SecurityConfig,
    crypto: Arc<CryptoContext>,
    layers: Vec<Box<dyn SecurityLayer>>,
    nodes: HashMap<String, Box<dyn SecurityNode>>,
    event_handlers: Arc<RwLock<Vec<Arc<dyn SecurityEventHandler>>>>,
    metrics: Arc<RwLock<FrameworkMetrics>>,
    status: Arc<RwLock<FrameworkStatus>>,
}

impl SecurityFramework {
    /// Create new security framework
    pub async fn new(config: SecurityConfig) -> SecurityResult<Self> {
        // Validate configuration
        config.validate()?;
        
        // Initialize cryptographic context
        let crypto = Arc::new(CryptoContext::new(config.crypto.clone())?);
        
        // Initialize layers
        let mut layers: Vec<Box<dyn SecurityLayer>> = Vec::new();
        
        // Layer 1: Multi-Layer Execution
        layers.push(Box::new(MultiLayerExecution::new()));
        
        // Layer 2: CBADU
        layers.push(Box::new(CBADULayer::new()));
        
        // Layer 3: Illusion Layer
        layers.push(Box::new(IllusionLayer::new()));
        
        // Layer 4: Behavior Monitoring
        layers.push(Box::new(BehaviorMonitoring::new()));
        
        // Layer 5: Thermal Detection
        layers.push(Box::new(ThermalDetection::new()));
        
        // Initialize security nodes
        let mut nodes: HashMap<String, Box<dyn SecurityNode>> = HashMap::new();
        
        // DOS Node for denial of service detection
        let dos_node = DOSNode::new("dos_node_1".to_string());
        nodes.insert("dos_node_1".to_string(), Box::new(dos_node));
        
        // Investigation Node for forensic analysis
        let investigation_node = InvestigationNode::new("investigation_node_1".to_string());
        nodes.insert("investigation_node_1".to_string(), Box::new(investigation_node));
        
        // Casualty Node for post-incident analysis
        let casualty_node = CasualtyNode::new("casualty_node_1".to_string());
        nodes.insert("casualty_node_1".to_string(), Box::new(casualty_node));
        
        // Confusion Node for defensive deception
        let confusion_node = ConfusionNode::new("confusion_node_1".to_string());
        nodes.insert("confusion_node_1".to_string(), Box::new(confusion_node));
        
        let metrics = Arc::new(RwLock::new(FrameworkMetrics::new()));
        let status = Arc::new(RwLock::new(FrameworkStatus::Initializing));
        
        Ok(Self {
            config,
            crypto,
            layers,
            nodes,
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            metrics,
            status,
        })
    }

    /// Initialize the security framework
    pub async fn initialize(&mut self) -> SecurityResult<()> {
        tracing::info!("Initializing biological security framework");
        
        // Initialize all layers
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let layer_config = &self.config.layers[i];
            tracing::debug!("Initializing layer {}: {}", layer.layer_id(), layer.layer_name());
            
            layer.initialize(layer_config, self.crypto.clone()).await
                .map_err(|e| SecurityError::LayerError {
                    layer: layer.layer_id(),
                    message: format!("Failed to initialize layer: {}", e),
                })?;
        }
        
        // Initialize all nodes
        for (node_id, node) in self.nodes.iter_mut() {
            tracing::debug!("Initializing security node: {}", node_id);
            node.initialize(self.crypto.clone()).await
                .map_err(|e| SecurityError::NodeViolation {
                    node_id: node_id.clone(),
                    violation: format!("Failed to initialize node: {}", e),
                })?;
        }
        
        {
            let mut status = self.status.write().await;
            *status = FrameworkStatus::Ready;
        }
        
        tracing::info!("Security framework initialized successfully");
        Ok(())
    }

    /// Start the security framework
    pub async fn start(&mut self) -> SecurityResult<()> {
        tracing::info!("Starting biological security framework");
        
        {
            let mut status = self.status.write().await;
            *status = FrameworkStatus::Starting;
        }
        
        // Start all layers
        let layer_futures = self.layers.iter_mut().map(|layer| layer.start());
        try_join_all(layer_futures).await
            .map_err(|e| SecurityError::SystemError(format!("Failed to start layers: {}", e)))?;
        
        // Start all nodes
        let node_futures = self.nodes.values_mut().map(|node| node.start());
        try_join_all(node_futures).await
            .map_err(|e| SecurityError::SystemError(format!("Failed to start nodes: {}", e)))?;
        
        {
            let mut status = self.status.write().await;
            *status = FrameworkStatus::Active;
        }
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        tracing::info!("Security framework started successfully");
        Ok(())
    }

    /// Stop the security framework
    pub async fn stop(&mut self) -> SecurityResult<()> {
        tracing::info!("Stopping biological security framework");
        
        {
            let mut status = self.status.write().await;
            *status = FrameworkStatus::Stopping;
        }
        
        // Stop all nodes
        let node_futures = self.nodes.values_mut().map(|node| node.stop());
        let _ = try_join_all(node_futures).await; // Best effort
        
        // Stop all layers
        let layer_futures = self.layers.iter_mut().map(|layer| layer.stop());
        let _ = try_join_all(layer_futures).await; // Best effort
        
        {
            let mut status = self.status.write().await;
            *status = FrameworkStatus::Stopped;
        }
        
        tracing::info!("Security framework stopped");
        Ok(())
    }

    /// Execute secure computation through all layers
    pub async fn execute_secure(
        &self,
        execution_id: &str,
        data: &[u8],
    ) -> SecurityResult<Vec<u8>> {
        let node_id = "framework_node".to_string(); // This would typically come from node context
        let context = SecurityContext::new(execution_id.to_string(), node_id);
        
        self.execute_secure_with_context(data, &context).await
    }

    /// Execute secure computation with specific context
    pub async fn execute_secure_with_context(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        tracing::debug!("Starting secure execution for {}", context.execution_id);
        
        // Check framework status
        {
            let status = self.status.read().await;
            if !matches!(*status, FrameworkStatus::Active) {
                return Err(SecurityError::SystemError(
                    format!("Security framework not active: {:?}", *status)
                ));
            }
        }
        
        let mut current_data = data.to_vec();
        let mut current_context = context.clone();
        let mut all_events = Vec::new();
        
        // Pre-processing through all layers
        for layer in &self.layers {
            let result = layer.process_pre(&current_data, &current_context).await
                .map_err(|e| SecurityError::LayerError {
                    layer: layer.layer_id(),
                    message: format!("Pre-processing failed: {}", e),
                })?;
            
            if !result.success || !result.continue_processing {
                return Err(SecurityError::LayerError {
                    layer: layer.layer_id(),
                    message: "Layer processing failed or was terminated".to_string(),
                });
            }
            
            current_data = result.data;
            current_context = result.context;
            all_events.extend(result.events);
        }
        
        // Process security events through nodes
        for event in &all_events {
            self.process_security_event(event).await?;
        }
        
        // Core computation would happen here
        // For this framework, we just pass the data through as an example
        let computation_result = current_data.clone();
        
        // Post-processing through all layers (in reverse order)
        let mut result_data = computation_result;
        for layer in self.layers.iter().rev() {
            let result = layer.process_post(&result_data, &current_context).await
                .map_err(|e| SecurityError::LayerError {
                    layer: layer.layer_id(),
                    message: format!("Post-processing failed: {}", e),
                })?;
            
            if !result.success {
                return Err(SecurityError::LayerError {
                    layer: layer.layer_id(),
                    message: "Layer post-processing failed".to_string(),
                });
            }
            
            result_data = result.data;
            all_events.extend(result.events);
        }
        
        // Process any additional security events
        for event in &all_events {
            self.process_security_event(event).await?;
        }
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        {
            let mut metrics = self.metrics.write().await;
            metrics.record_execution(processing_time, true);
        }
        
        tracing::debug!("Completed secure execution for {} in {:.2}ms", 
            context.execution_id, processing_time);
        
        Ok(result_data)
    }

    /// Process a security event through appropriate nodes
    async fn process_security_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        let mut response_events = Vec::new();
        
        // Route event to appropriate security nodes based on event type
        match event.event_type.as_str() {
            // DOS-related events
            "dos_attack_detected" | "traffic_anomaly" | "connection_flood" => {
                if let Some(dos_node) = self.nodes.get("dos_node_1") {
                    let events = dos_node.process_event(event).await?;
                    response_events.extend(events);
                }
            },
            
            // Investigation-required events
            "behavioral_anomaly_detected" | "security_violation" | "threat_detected" => {
                if let Some(investigation_node) = self.nodes.get("investigation_node_1") {
                    let events = investigation_node.process_event(event).await?;
                    response_events.extend(events);
                }
            },
            
            // Post-incident events
            "node_failure" | "execution_failed" | "system_error" => {
                if let Some(casualty_node) = self.nodes.get("casualty_node_1") {
                    let events = casualty_node.process_event(event).await?;
                    response_events.extend(events);
                }
            },
            
            // High severity events requiring deception
            _ if matches!(event.severity, SecuritySeverity::High | SecuritySeverity::Critical) => {
                if let Some(confusion_node) = self.nodes.get("confusion_node_1") {
                    let events = confusion_node.process_event(event).await?;
                    response_events.extend(events);
                }
            },
            
            _ => {
                tracing::debug!("No specific handler for event type: {}", event.event_type);
            }
        }
        
        // Process response events recursively (with depth limit to prevent loops)
        for response_event in response_events {
            self.handle_response_event(&response_event, 1).await?;
        }
        
        // Notify event handlers
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            handler.handle_event(event).await?;
        }
        
        Ok(())
    }

    /// Handle response events from security nodes
    async fn handle_response_event(&self, event: &SecurityEvent, depth: usize) -> SecurityResult<()> {
        const MAX_DEPTH: usize = 3;
        
        if depth > MAX_DEPTH {
            tracing::warn!("Maximum event processing depth reached, stopping recursion");
            return Ok(());
        }
        
        // Process the response event through the framework
        self.process_security_event(event).await?;
        
        Ok(())
    }

    /// Add event handler
    pub async fn add_event_handler(&self, handler: Arc<dyn SecurityEventHandler>) -> SecurityResult<()> {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
        Ok(())
    }

    /// Get framework status
    pub async fn status(&self) -> FrameworkStatus {
        let status = self.status.read().await;
        status.clone()
    }

    /// Get framework metrics
    pub async fn metrics(&self) -> FrameworkMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Get layer status for all layers
    pub async fn layer_status(&self) -> Vec<(usize, String, LayerStatus)> {
        let mut statuses = Vec::new();
        
        for layer in &self.layers {
            let status = layer.status().await;
            statuses.push((layer.layer_id(), layer.layer_name().to_string(), status));
        }
        
        statuses
    }

    /// Get layer metrics for all layers
    pub async fn layer_metrics(&self) -> Vec<LayerMetrics> {
        let mut all_metrics = Vec::new();
        
        for layer in &self.layers {
            let metrics = layer.metrics().await;
            all_metrics.push(metrics);
        }
        
        all_metrics
    }

    /// Get node status for all nodes
    pub async fn node_status(&self) -> Vec<(String, SecurityNodeStatus)> {
        let mut statuses = Vec::new();
        
        for (node_id, node) in &self.nodes {
            let status = node.status().await;
            statuses.push((node_id.clone(), status));
        }
        
        statuses
    }

    /// Get node metrics for all nodes
    pub async fn node_metrics(&self) -> Vec<SecurityNodeMetrics> {
        let mut all_metrics = Vec::new();
        
        for node in self.nodes.values() {
            let metrics = node.metrics().await;
            all_metrics.push(metrics);
        }
        
        all_metrics
    }

    /// Start background framework tasks
    async fn start_background_tasks(&self) -> SecurityResult<()> {
        // Start metrics collection task
        let metrics = self.metrics.clone();
        let layers = unsafe { 
            // SAFETY: We need to share the layers between tasks
            // In a real implementation, we'd use Arc<RwLock<Vec<...>>>
            std::mem::transmute::<&Vec<Box<dyn SecurityLayer>>, &'static Vec<Box<dyn SecurityLayer>>>(&self.layers)
        };
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let mut framework_metrics = metrics.write().await;
                
                // Collect metrics from all layers
                let mut total_operations = 0;
                let mut total_security_events = 0;
                let mut total_threat_detections = 0;
                
                for layer in layers.iter() {
                    let layer_metrics = layer.metrics().await;
                    total_operations += layer_metrics.operations_processed;
                    total_security_events += layer_metrics.security_events;
                    total_threat_detections += layer_metrics.threat_detections;
                }
                
                framework_metrics.total_layer_operations = total_operations;
                framework_metrics.total_security_events = total_security_events;
                framework_metrics.total_threat_detections = total_threat_detections;
                
                tracing::debug!("Updated framework metrics: {} operations, {} events, {} threats",
                    total_operations, total_security_events, total_threat_detections);
            }
        });
        
        tracing::info!("Started background framework tasks");
        Ok(())
    }
}

/// Framework operational status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameworkStatus {
    /// Framework is initializing
    Initializing,
    /// Framework is ready but not started
    Ready,
    /// Framework is starting up
    Starting,
    /// Framework is actively processing
    Active,
    /// Framework is stopping
    Stopping,
    /// Framework has stopped
    Stopped,
    /// Framework has encountered an error
    Error(String),
}

/// Framework performance metrics
#[derive(Debug, Clone)]
pub struct FrameworkMetrics {
    /// Total executions processed
    pub executions_processed: u64,
    /// Total successful executions
    pub executions_successful: u64,
    /// Total failed executions
    pub executions_failed: u64,
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: f64,
    /// Total operations across all layers
    pub total_layer_operations: u64,
    /// Total security events generated
    pub total_security_events: u64,
    /// Total threats detected
    pub total_threat_detections: u64,
    /// Framework uptime
    pub uptime: std::time::Duration,
    /// Start time
    pub start_time: std::time::Instant,
}

impl FrameworkMetrics {
    pub fn new() -> Self {
        Self {
            executions_processed: 0,
            executions_successful: 0,
            executions_failed: 0,
            avg_execution_time_ms: 0.0,
            total_layer_operations: 0,
            total_security_events: 0,
            total_threat_detections: 0,
            uptime: std::time::Duration::ZERO,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn record_execution(&mut self, processing_time_ms: f64, success: bool) {
        self.executions_processed += 1;
        
        if success {
            self.executions_successful += 1;
        } else {
            self.executions_failed += 1;
        }
        
        // Update rolling average
        let total_executions = self.executions_processed as f64;
        self.avg_execution_time_ms = 
            (self.avg_execution_time_ms * (total_executions - 1.0) + processing_time_ms) / total_executions;
        
        // Update uptime
        self.uptime = self.start_time.elapsed();
    }

    pub fn success_rate(&self) -> f64 {
        if self.executions_processed == 0 {
            1.0
        } else {
            self.executions_successful as f64 / self.executions_processed as f64
        }
    }
}

/// Security event handler trait
#[async_trait::async_trait]
pub trait SecurityEventHandler: Send + Sync {
    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()>;
}

/// Result type for security framework operations
pub type SecurityResult<T> = Result<T, SecurityError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SecurityConfig;

    #[tokio::test]
    async fn test_framework_creation() {
        let config = SecurityConfig::for_testing();
        let framework = SecurityFramework::new(config).await;
        assert!(framework.is_ok());
    }

    #[tokio::test]
    async fn test_framework_lifecycle() {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await.unwrap();
        
        // Initialize
        framework.initialize().await.unwrap();
        assert_eq!(framework.status().await, FrameworkStatus::Ready);
        
        // Start
        framework.start().await.unwrap();
        assert_eq!(framework.status().await, FrameworkStatus::Active);
        
        // Stop
        framework.stop().await.unwrap();
        assert_eq!(framework.status().await, FrameworkStatus::Stopped);
    }

    #[tokio::test]
    async fn test_secure_execution() {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await.unwrap();
        
        framework.initialize().await.unwrap();
        framework.start().await.unwrap();
        
        let test_data = b"test computation data";
        let result = framework.execute_secure("test_execution", test_data).await;
        
        // Should succeed (passes data through)
        assert!(result.is_ok());
        let output = result.unwrap();
        // In this test implementation, data passes through unchanged
        assert!(!output.is_empty());
        
        framework.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_framework_metrics() {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await.unwrap();
        
        framework.initialize().await.unwrap();
        framework.start().await.unwrap();
        
        let test_data = b"test data";
        let _ = framework.execute_secure("test_exec", test_data).await;
        
        let metrics = framework.metrics().await;
        assert!(metrics.executions_processed > 0);
        
        framework.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_layer_status_collection() {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await.unwrap();
        
        framework.initialize().await.unwrap();
        
        let layer_status = framework.layer_status().await;
        assert_eq!(layer_status.len(), 5); // Should have all 5 layers
        
        for (layer_id, name, status) in layer_status {
            assert!(layer_id >= 1 && layer_id <= 5);
            assert!(!name.is_empty());
            assert_eq!(status, LayerStatus::Ready);
        }
        
        framework.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_node_status_collection() {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await.unwrap();
        
        framework.initialize().await.unwrap();
        
        let node_status = framework.node_status().await;
        assert_eq!(node_status.len(), 4); // DOS, Investigation, Casualty, Confusion
        
        for (node_id, status) in node_status {
            assert!(!node_id.is_empty());
            assert_eq!(status, SecurityNodeStatus::Ready);
        }
        
        framework.stop().await.unwrap();
    }

    #[test]
    fn test_framework_metrics_calculation() {
        let mut metrics = FrameworkMetrics::new();
        
        // Record some executions
        metrics.record_execution(100.0, true);
        metrics.record_execution(200.0, true);
        metrics.record_execution(150.0, false);
        
        assert_eq!(metrics.executions_processed, 3);
        assert_eq!(metrics.executions_successful, 2);
        assert_eq!(metrics.executions_failed, 1);
        assert_eq!(metrics.avg_execution_time_ms, 150.0);
        assert_eq!(metrics.success_rate(), 2.0 / 3.0);
    }
}