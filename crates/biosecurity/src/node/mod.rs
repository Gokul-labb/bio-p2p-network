//! Security node implementations
//! 
//! This module contains the security nodes that implement biological behaviors
//! for threat detection, investigation, and defensive responses.

pub mod dos_node;
pub mod investigation_node;
pub mod casualty_node;
pub mod confusion_node;

use async_trait::async_trait;
use std::sync::Arc;

use crate::errors::{SecurityResult, SecurityEvent};
use crate::crypto::CryptoContext;
use crate::layers::SecurityContext;

/// Base trait for all security nodes
#[async_trait]
pub trait SecurityNode: Send + Sync {
    /// Node type identifier
    fn node_type(&self) -> &'static str;
    
    /// Node identifier
    fn node_id(&self) -> &str;
    
    /// Initialize the security node
    async fn initialize(&mut self, crypto: Arc<CryptoContext>) -> SecurityResult<()>;
    
    /// Start the security node
    async fn start(&mut self) -> SecurityResult<()>;
    
    /// Stop the security node
    async fn stop(&mut self) -> SecurityResult<()>;
    
    /// Process a security event
    async fn process_event(&self, event: &SecurityEvent) -> SecurityResult<Vec<SecurityEvent>>;
    
    /// Get current node status
    async fn status(&self) -> SecurityNodeStatus;
    
    /// Get node metrics
    async fn metrics(&self) -> SecurityNodeMetrics;
}

/// Security node operational status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityNodeStatus {
    /// Node is not initialized
    Uninitialized,
    /// Node is initializing
    Initializing,
    /// Node is ready but not active
    Ready,
    /// Node is actively processing
    Active,
    /// Node is investigating a threat
    Investigating,
    /// Node is responding to an incident
    Responding,
    /// Node is stopping
    Stopping,
    /// Node has stopped
    Stopped,
    /// Node has encountered an error
    Error(String),
}

/// Security node performance metrics
#[derive(Debug, Clone)]
pub struct SecurityNodeMetrics {
    /// Node identifier
    pub node_id: String,
    /// Node type
    pub node_type: String,
    /// Events processed
    pub events_processed: u64,
    /// Threats detected
    pub threats_detected: u64,
    /// Investigations completed
    pub investigations_completed: u64,
    /// Response actions taken
    pub response_actions: u64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Current status
    pub status: SecurityNodeStatus,
}

impl Default for SecurityNodeMetrics {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            node_type: String::new(),
            events_processed: 0,
            threats_detected: 0,
            investigations_completed: 0,
            response_actions: 0,
            avg_processing_time_ms: 0.0,
            last_activity: chrono::Utc::now(),
            status: SecurityNodeStatus::Uninitialized,
        }
    }
}

impl SecurityNodeMetrics {
    pub fn new(node_id: String, node_type: String) -> Self {
        Self {
            node_id,
            node_type,
            ..Default::default()
        }
    }

    pub fn record_event(&mut self, processing_time_ms: f64) {
        self.events_processed += 1;
        
        // Update rolling average
        let total_events = self.events_processed as f64;
        self.avg_processing_time_ms = 
            (self.avg_processing_time_ms * (total_events - 1.0) + processing_time_ms) / total_events;
        
        self.last_activity = chrono::Utc::now();
    }

    pub fn record_threat_detection(&mut self) {
        self.threats_detected += 1;
    }

    pub fn record_investigation_completed(&mut self) {
        self.investigations_completed += 1;
    }

    pub fn record_response_action(&mut self) {
        self.response_actions += 1;
    }
}

/// Base implementation for security nodes
pub struct BaseSecurityNode {
    node_type: &'static str,
    node_id: String,
    crypto: Option<Arc<CryptoContext>>,
    status: SecurityNodeStatus,
    metrics: SecurityNodeMetrics,
}

impl BaseSecurityNode {
    pub fn new(node_type: &'static str, node_id: String) -> Self {
        let metrics = SecurityNodeMetrics::new(node_id.clone(), node_type.to_string());
        
        Self {
            node_type,
            node_id: node_id.clone(),
            crypto: None,
            status: SecurityNodeStatus::Uninitialized,
            metrics,
        }
    }

    pub fn set_status(&mut self, status: SecurityNodeStatus) {
        self.status = status.clone();
        self.metrics.status = status;
    }

    pub fn record_event(&mut self, processing_time_ms: f64) {
        self.metrics.record_event(processing_time_ms);
    }

    pub fn record_threat_detection(&mut self) {
        self.metrics.record_threat_detection();
    }

    pub fn record_investigation_completed(&mut self) {
        self.metrics.record_investigation_completed();
    }

    pub fn record_response_action(&mut self) {
        self.metrics.record_response_action();
    }

    pub fn crypto(&self) -> Option<&Arc<CryptoContext>> {
        self.crypto.as_ref()
    }
}

#[async_trait]
impl SecurityNode for BaseSecurityNode {
    fn node_type(&self) -> &'static str {
        self.node_type
    }

    fn node_id(&self) -> &str {
        &self.node_id
    }

    async fn initialize(&mut self, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.set_status(SecurityNodeStatus::Initializing);
        self.crypto = Some(crypto);
        self.set_status(SecurityNodeStatus::Ready);
        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.set_status(SecurityNodeStatus::Active);
        Ok(())
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        self.set_status(SecurityNodeStatus::Stopping);
        self.set_status(SecurityNodeStatus::Stopped);
        Ok(())
    }

    async fn process_event(&self, _event: &SecurityEvent) -> SecurityResult<Vec<SecurityEvent>> {
        // Base implementation - no processing
        Ok(Vec::new())
    }

    async fn status(&self) -> SecurityNodeStatus {
        self.status.clone()
    }

    async fn metrics(&self) -> SecurityNodeMetrics {
        self.metrics.clone()
    }
}

// Re-export specific nodes
pub use dos_node::DOSNode;
pub use investigation_node::InvestigationNode;
pub use casualty_node::CasualtyNode;
pub use confusion_node::ConfusionNode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CryptoConfig;
    use crate::errors::SecuritySeverity;

    #[tokio::test]
    async fn test_base_security_node() {
        let mut node = BaseSecurityNode::new("test_node", "test_id".to_string());
        
        assert_eq!(node.node_type(), "test_node");
        assert_eq!(node.node_id(), "test_id");
        assert_eq!(node.status().await, SecurityNodeStatus::Uninitialized);
    }

    #[tokio::test]
    async fn test_node_lifecycle() {
        let mut node = BaseSecurityNode::new("test_node", "test_id".to_string());
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());

        // Initialize
        node.initialize(crypto).await.unwrap();
        assert_eq!(node.status().await, SecurityNodeStatus::Ready);

        // Start
        node.start().await.unwrap();
        assert_eq!(node.status().await, SecurityNodeStatus::Active);

        // Stop
        node.stop().await.unwrap();
        assert_eq!(node.status().await, SecurityNodeStatus::Stopped);
    }

    #[tokio::test]
    async fn test_node_metrics() {
        let mut node = BaseSecurityNode::new("test_node", "test_id".to_string());
        
        // Initial metrics
        let metrics = node.metrics().await;
        assert_eq!(metrics.events_processed, 0);
        assert_eq!(metrics.threats_detected, 0);
        
        // Record some activity
        node.record_event(100.0);
        node.record_threat_detection();
        
        let updated_metrics = node.metrics().await;
        assert_eq!(updated_metrics.events_processed, 1);
        assert_eq!(updated_metrics.threats_detected, 1);
        assert_eq!(updated_metrics.avg_processing_time_ms, 100.0);
    }

    #[tokio::test]
    async fn test_event_processing() {
        let node = BaseSecurityNode::new("test_node", "test_id".to_string());
        
        let event = SecurityEvent::new(
            SecuritySeverity::Medium,
            "test_event",
            "Test event message",
        );
        
        let result_events = node.process_event(&event).await.unwrap();
        assert!(result_events.is_empty()); // Base implementation returns empty
    }

    #[test]
    fn test_security_node_metrics_calculation() {
        let mut metrics = SecurityNodeMetrics::new("test".to_string(), "test_type".to_string());
        
        // Record multiple events
        metrics.record_event(100.0);
        metrics.record_event(200.0);
        metrics.record_event(150.0);
        
        assert_eq!(metrics.events_processed, 3);
        assert_eq!(metrics.avg_processing_time_ms, 150.0); // (100 + 200 + 150) / 3
    }
}