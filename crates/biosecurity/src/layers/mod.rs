//! Security layer implementations
//! 
//! This module contains the five biological security layers that form the
//! defense-in-depth architecture of the framework.

pub mod layer1_execution;
pub mod layer2_cbadu;
pub mod layer3_illusion;
pub mod layer4_behavior;
pub mod layer5_thermal;

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::config::LayerConfig;
use crate::crypto::CryptoContext;

/// Trait for all security layers
#[async_trait]
pub trait SecurityLayer: Send + Sync {
    /// Layer identifier (1-5)
    fn layer_id(&self) -> usize;
    
    /// Layer name
    fn layer_name(&self) -> &str;
    
    /// Initialize the security layer
    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()>;
    
    /// Start the security layer
    async fn start(&mut self) -> SecurityResult<()>;
    
    /// Stop the security layer
    async fn stop(&mut self) -> SecurityResult<()>;
    
    /// Process data through this security layer (pre-execution)
    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult>;
    
    /// Process data through this security layer (post-execution)
    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult>;
    
    /// Get current layer status
    async fn status(&self) -> LayerStatus;
    
    /// Get layer metrics
    async fn metrics(&self) -> LayerMetrics;
    
    /// Handle security event
    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()>;
}

/// Security context passed between layers
#[derive(Debug, Clone)]
pub struct SecurityContext {
    /// Unique execution identifier
    pub execution_id: String,
    /// Node identifier
    pub node_id: String,
    /// Timestamp of execution start
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Security metadata
    pub metadata: std::collections::HashMap<String, String>,
    /// Trust score for the requesting node
    pub trust_score: f64,
    /// Risk level assessment
    pub risk_level: RiskLevel,
}

impl SecurityContext {
    pub fn new(execution_id: String, node_id: String) -> Self {
        Self {
            execution_id,
            node_id,
            timestamp: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
            trust_score: 0.5, // neutral trust
            risk_level: RiskLevel::Medium,
        }
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn with_trust_score(mut self, score: f64) -> Self {
        self.trust_score = score.clamp(0.0, 1.0);
        self
    }

    pub fn with_risk_level(mut self, level: RiskLevel) -> Self {
        self.risk_level = level;
        self
    }
}

/// Risk level assessment for computations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    /// Low risk - trusted computation
    Low,
    /// Medium risk - standard computation
    Medium,
    /// High risk - requires additional monitoring
    High,
    /// Critical risk - maximum security measures
    Critical,
}

impl RiskLevel {
    pub fn from_trust_score(trust_score: f64) -> Self {
        if trust_score >= 0.8 {
            RiskLevel::Low
        } else if trust_score >= 0.6 {
            RiskLevel::Medium
        } else if trust_score >= 0.3 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    pub fn security_multiplier(&self) -> f64 {
        match self {
            RiskLevel::Low => 0.8,
            RiskLevel::Medium => 1.0,
            RiskLevel::High => 1.5,
            RiskLevel::Critical => 2.0,
        }
    }
}

/// Result of layer processing
#[derive(Debug)]
pub struct ProcessResult {
    /// Processed data
    pub data: Vec<u8>,
    /// Whether processing was successful
    pub success: bool,
    /// Security events generated
    pub events: Vec<SecurityEvent>,
    /// Updated context
    pub context: SecurityContext,
    /// Whether to continue to next layer
    pub continue_processing: bool,
}

impl ProcessResult {
    pub fn success(data: Vec<u8>, context: SecurityContext) -> Self {
        Self {
            data,
            success: true,
            events: Vec::new(),
            context,
            continue_processing: true,
        }
    }

    pub fn failure(reason: String, context: SecurityContext) -> Self {
        let event = SecurityEvent::new(
            SecuritySeverity::High,
            "layer_processing_failed",
            reason,
        );

        Self {
            data: Vec::new(),
            success: false,
            events: vec![event],
            context,
            continue_processing: false,
        }
    }

    pub fn with_event(mut self, event: SecurityEvent) -> Self {
        self.events.push(event);
        self
    }

    pub fn with_events(mut self, mut events: Vec<SecurityEvent>) -> Self {
        self.events.append(&mut events);
        self
    }
}

/// Security layer operational status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerStatus {
    /// Layer is not initialized
    Uninitialized,
    /// Layer is initializing
    Initializing,
    /// Layer is ready but not active
    Ready,
    /// Layer is actively processing
    Active,
    /// Layer is stopping
    Stopping,
    /// Layer has stopped
    Stopped,
    /// Layer has encountered an error
    Error(String),
}

impl LayerStatus {
    pub fn is_operational(&self) -> bool {
        matches!(self, LayerStatus::Ready | LayerStatus::Active)
    }
}

/// Security layer performance metrics
#[derive(Debug, Clone)]
pub struct LayerMetrics {
    /// Layer identifier
    pub layer_id: usize,
    /// Number of operations processed
    pub operations_processed: u64,
    /// Number of successful operations
    pub operations_successful: u64,
    /// Number of failed operations
    pub operations_failed: u64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Security events generated
    pub security_events: u64,
    /// Threat detections
    pub threat_detections: u64,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Current status
    pub status: LayerStatus,
}

impl Default for LayerMetrics {
    fn default() -> Self {
        Self {
            layer_id: 0,
            operations_processed: 0,
            operations_successful: 0,
            operations_failed: 0,
            avg_processing_time_ms: 0.0,
            security_events: 0,
            threat_detections: 0,
            last_activity: chrono::Utc::now(),
            status: LayerStatus::Uninitialized,
        }
    }
}

impl LayerMetrics {
    pub fn new(layer_id: usize) -> Self {
        Self {
            layer_id,
            ..Default::default()
        }
    }

    pub fn record_operation(&mut self, processing_time_ms: f64, success: bool) {
        self.operations_processed += 1;
        if success {
            self.operations_successful += 1;
        } else {
            self.operations_failed += 1;
        }

        // Update rolling average
        let total_ops = self.operations_processed as f64;
        self.avg_processing_time_ms = 
            (self.avg_processing_time_ms * (total_ops - 1.0) + processing_time_ms) / total_ops;
        
        self.last_activity = chrono::Utc::now();
    }

    pub fn record_security_event(&mut self) {
        self.security_events += 1;
    }

    pub fn record_threat_detection(&mut self) {
        self.threat_detections += 1;
        self.security_events += 1;
    }

    pub fn success_rate(&self) -> f64 {
        if self.operations_processed == 0 {
            1.0
        } else {
            self.operations_successful as f64 / self.operations_processed as f64
        }
    }
}

/// Base implementation for security layers
pub struct BaseLayer {
    layer_id: usize,
    layer_name: String,
    config: LayerConfig,
    crypto: Option<Arc<CryptoContext>>,
    status: Arc<RwLock<LayerStatus>>,
    metrics: Arc<RwLock<LayerMetrics>>,
}

impl BaseLayer {
    pub fn new(layer_id: usize, layer_name: String) -> Self {
        Self {
            layer_id,
            layer_name: layer_name.clone(),
            config: LayerConfig::multi_layer_execution(), // Default config
            crypto: None,
            status: Arc::new(RwLock::new(LayerStatus::Uninitialized)),
            metrics: Arc::new(RwLock::new(LayerMetrics::new(layer_id))),
        }
    }

    pub async fn set_status(&self, status: LayerStatus) {
        let mut current_status = self.status.write().await;
        *current_status = status;
        
        let mut metrics = self.metrics.write().await;
        metrics.status = current_status.clone();
    }

    pub async fn record_operation(&self, processing_time_ms: f64, success: bool) {
        let mut metrics = self.metrics.write().await;
        metrics.record_operation(processing_time_ms, success);
    }

    pub async fn record_security_event(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.record_security_event();
    }

    pub async fn record_threat_detection(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.record_threat_detection();
    }

    pub fn config(&self) -> &LayerConfig {
        &self.config
    }

    pub fn crypto(&self) -> Option<&Arc<CryptoContext>> {
        self.crypto.as_ref()
    }
}

#[async_trait]
impl SecurityLayer for BaseLayer {
    fn layer_id(&self) -> usize {
        self.layer_id
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.set_status(LayerStatus::Initializing).await;
        self.config = config.clone();
        self.crypto = Some(crypto);
        self.set_status(LayerStatus::Ready).await;
        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.set_status(LayerStatus::Active).await;
        Ok(())
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        self.set_status(LayerStatus::Stopping).await;
        self.set_status(LayerStatus::Stopped).await;
        Ok(())
    }

    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        // Default implementation - pass through
        Ok(ProcessResult::success(data.to_vec(), context.clone()))
    }

    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        // Default implementation - pass through  
        Ok(ProcessResult::success(data.to_vec(), context.clone()))
    }

    async fn status(&self) -> LayerStatus {
        self.status.read().await.clone()
    }

    async fn metrics(&self) -> LayerMetrics {
        self.metrics.read().await.clone()
    }

    async fn handle_event(&self, _event: &SecurityEvent) -> SecurityResult<()> {
        // Default implementation - log event
        tracing::info!("Layer {} handling event: {}", self.layer_id, _event.event_type);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{SecurityConfig, CryptoConfig};

    #[tokio::test]
    async fn test_base_layer_lifecycle() {
        let mut layer = BaseLayer::new(1, "Test Layer".to_string());
        
        // Initial status should be uninitialized
        assert_eq!(layer.status().await, LayerStatus::Uninitialized);
        
        // Initialize layer
        let config = LayerConfig::multi_layer_execution();
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());
        
        layer.initialize(&config, crypto).await.unwrap();
        assert_eq!(layer.status().await, LayerStatus::Ready);
        
        // Start layer
        layer.start().await.unwrap();
        assert_eq!(layer.status().await, LayerStatus::Active);
        
        // Stop layer
        layer.stop().await.unwrap();
        assert_eq!(layer.status().await, LayerStatus::Stopped);
    }

    #[tokio::test]
    async fn test_security_context() {
        let context = SecurityContext::new(
            "test_exec_123".to_string(),
            "node_abc".to_string(),
        )
        .with_trust_score(0.8)
        .with_risk_level(RiskLevel::Low)
        .with_metadata("test_key".to_string(), "test_value".to_string());

        assert_eq!(context.execution_id, "test_exec_123");
        assert_eq!(context.node_id, "node_abc");
        assert_eq!(context.trust_score, 0.8);
        assert_eq!(context.risk_level, RiskLevel::Low);
        assert_eq!(context.metadata.get("test_key"), Some(&"test_value".to_string()));
    }

    #[tokio::test]
    async fn test_process_result() {
        let context = SecurityContext::new("test".to_string(), "node".to_string());
        let data = b"test data".to_vec();
        
        let result = ProcessResult::success(data.clone(), context.clone());
        assert!(result.success);
        assert_eq!(result.data, data);
        assert!(result.continue_processing);
        
        let failure = ProcessResult::failure("test error".to_string(), context);
        assert!(!failure.success);
        assert!(!failure.continue_processing);
        assert!(!failure.events.is_empty());
    }

    #[test]
    fn test_risk_level_from_trust_score() {
        assert_eq!(RiskLevel::from_trust_score(0.9), RiskLevel::Low);
        assert_eq!(RiskLevel::from_trust_score(0.7), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_trust_score(0.5), RiskLevel::High);
        assert_eq!(RiskLevel::from_trust_score(0.1), RiskLevel::Critical);
    }

    #[test]
    fn test_layer_metrics() {
        let mut metrics = LayerMetrics::new(1);
        
        // Initial state
        assert_eq!(metrics.operations_processed, 0);
        assert_eq!(metrics.success_rate(), 1.0);
        
        // Record successful operation
        metrics.record_operation(100.0, true);
        assert_eq!(metrics.operations_processed, 1);
        assert_eq!(metrics.operations_successful, 1);
        assert_eq!(metrics.avg_processing_time_ms, 100.0);
        assert_eq!(metrics.success_rate(), 1.0);
        
        // Record failed operation
        metrics.record_operation(150.0, false);
        assert_eq!(metrics.operations_processed, 2);
        assert_eq!(metrics.operations_failed, 1);
        assert_eq!(metrics.avg_processing_time_ms, 125.0);
        assert_eq!(metrics.success_rate(), 0.5);
    }
}