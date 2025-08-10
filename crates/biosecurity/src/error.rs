//! Security error types and results

use thiserror::Error;
use std::fmt;

/// Main security error type
#[derive(Error, Debug)]
pub enum SecurityError {
    #[error("Cryptographic operation failed: {0}")]
    Cryptographic(String),
    
    #[error("Layer {layer} failed: {message}")]
    LayerError { layer: usize, message: String },
    
    #[error("Container isolation failed: {0}")]
    ContainerError(String),
    
    #[error("Sanitization failed: {0}")]
    SanitizationError(String),
    
    #[error("Anomaly detected: {threat_type} with confidence {confidence}")]
    AnomalyDetected { threat_type: String, confidence: f32 },
    
    #[error("Thermal signature violation: {resource} usage {actual} exceeds threshold {threshold}")]
    ThermalViolation { resource: String, actual: f64, threshold: f64 },
    
    #[error("Behavioral analysis failed: {0}")]
    BehaviorAnalysisError(String),
    
    #[error("Deception layer activation failed: {0}")]
    DeceptionError(String),
    
    #[error("Node security violation: {node_id} - {violation}")]
    NodeViolation { node_id: String, violation: String },
    
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("I/O operation failed: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("System error: {0}")]
    SystemError(String),
    
    #[error("Byzantine behavior detected from node {node_id}: {behavior}")]
    ByzantineBehavior { node_id: String, behavior: String },
    
    #[error("Security layer timeout: layer {layer} exceeded {timeout_ms}ms")]
    TimeoutError { layer: usize, timeout_ms: u64 },
}

/// Result type for security operations
pub type SecurityResult<T> = Result<T, SecurityError>;

/// Security event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecuritySeverity {
    /// Informational events
    Info,
    /// Low severity threats
    Low,
    /// Medium severity threats  
    Medium,
    /// High severity threats requiring immediate attention
    High,
    /// Critical security incidents
    Critical,
}

impl fmt::Display for SecuritySeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecuritySeverity::Info => write!(f, "INFO"),
            SecuritySeverity::Low => write!(f, "LOW"),
            SecuritySeverity::Medium => write!(f, "MEDIUM"), 
            SecuritySeverity::High => write!(f, "HIGH"),
            SecuritySeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Security event for logging and monitoring
#[derive(Debug, Clone)]
pub struct SecurityEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub severity: SecuritySeverity,
    pub layer: Option<usize>,
    pub node_id: Option<String>,
    pub event_type: String,
    pub message: String,
    pub metadata: serde_json::Value,
}

impl SecurityEvent {
    pub fn new(
        severity: SecuritySeverity,
        event_type: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            severity,
            layer: None,
            node_id: None,
            event_type: event_type.into(),
            message: message.into(),
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_layer(mut self, layer: usize) -> Self {
        self.layer = Some(layer);
        self
    }

    pub fn with_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_error_creation() {
        let error = SecurityError::AnomalyDetected {
            threat_type: "DDoS".to_string(),
            confidence: 0.95,
        };
        assert!(error.to_string().contains("DDoS"));
        assert!(error.to_string().contains("0.95"));
    }

    #[test]
    fn test_security_event_creation() {
        let event = SecurityEvent::new(
            SecuritySeverity::High,
            "intrusion_detected",
            "Suspicious activity from node ABC123"
        )
        .with_layer(4)
        .with_node("ABC123");

        assert_eq!(event.severity, SecuritySeverity::High);
        assert_eq!(event.layer, Some(4));
        assert_eq!(event.node_id, Some("ABC123".to_string()));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(SecuritySeverity::Critical > SecuritySeverity::High);
        assert!(SecuritySeverity::High > SecuritySeverity::Medium);
        assert!(SecuritySeverity::Medium > SecuritySeverity::Low);
        assert!(SecuritySeverity::Low > SecuritySeverity::Info);
    }
}