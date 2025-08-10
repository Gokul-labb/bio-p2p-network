//! Resource Management Error Types
//! 
//! Comprehensive error handling for the biological resource management system.

use std::error::Error as StdError;
use std::fmt;
use std::result;

/// Result type for resource management operations
pub type ResourceResult<T> = result::Result<T, ResourceError>;

/// Comprehensive error types for resource management operations
#[derive(Debug, Clone)]
pub enum ResourceError {
    /// Resource allocation failed
    AllocationFailed {
        reason: String,
        resource_type: Option<String>,
        requested_amount: Option<f64>,
    },
    
    /// Insufficient resources available
    InsufficientResources {
        resource_type: String,
        requested: f64,
        available: f64,
    },
    
    /// Compartmentalization error
    CompartmentError {
        compartment_type: Option<String>,
        message: String,
    },
    
    /// Social relationship error
    SocialError {
        relationship_type: Option<String>,
        nodes_involved: Vec<String>,
        message: String,
    },
    
    /// HAVOC crisis management error
    HavocError {
        crisis_type: Option<String>,
        severity: Option<String>,
        message: String,
    },
    
    /// Thermal management error
    ThermalError {
        node_id: Option<String>,
        thermal_level: Option<f64>,
        message: String,
    },
    
    /// Scaling operation error
    ScalingError {
        node_id: Option<String>,
        scaling_factor: Option<f64>,
        message: String,
    },
    
    /// Network communication error
    NetworkError {
        endpoint: Option<String>,
        operation: String,
        message: String,
    },
    
    /// Configuration error
    ConfigurationError {
        parameter: String,
        value: Option<String>,
        message: String,
    },
    
    /// Validation error
    ValidationError {
        field: String,
        value: String,
        constraint: String,
    },
    
    /// Serialization/Deserialization error
    SerializationError {
        data_type: String,
        operation: String, // "serialize" or "deserialize"
        message: String,
    },
    
    /// Timeout error
    TimeoutError {
        operation: String,
        timeout_duration: std::time::Duration,
    },
    
    /// Concurrency error
    ConcurrencyError {
        resource: String,
        operation: String,
        message: String,
    },
    
    /// Metrics error
    MetricsError {
        metric_type: String,
        message: String,
    },
    
    /// Internal system error
    InternalError {
        component: String,
        message: String,
    },
    
    /// External dependency error
    ExternalError {
        service: String,
        operation: String,
        status_code: Option<u16>,
        message: String,
    },
}

impl ResourceError {
    /// Create a new allocation failed error
    pub fn allocation_failed<S: Into<String>>(reason: S) -> Self {
        Self::AllocationFailed {
            reason: reason.into(),
            resource_type: None,
            requested_amount: None,
        }
    }
    
    /// Create a new allocation failed error with details
    pub fn allocation_failed_detailed<S: Into<String>>(
        reason: S, 
        resource_type: S, 
        requested_amount: f64
    ) -> Self {
        Self::AllocationFailed {
            reason: reason.into(),
            resource_type: Some(resource_type.into()),
            requested_amount: Some(requested_amount),
        }
    }
    
    /// Create a new insufficient resources error
    pub fn insufficient_resources<S: Into<String>>(
        resource_type: S, 
        requested: f64, 
        available: f64
    ) -> Self {
        Self::InsufficientResources {
            resource_type: resource_type.into(),
            requested,
            available,
        }
    }
    
    /// Create a new compartment error
    pub fn compartment_error<S: Into<String>>(message: S) -> Self {
        Self::CompartmentError {
            compartment_type: None,
            message: message.into(),
        }
    }
    
    /// Create a new compartment error with type
    pub fn compartment_error_typed<S: Into<String>>(compartment_type: S, message: S) -> Self {
        Self::CompartmentError {
            compartment_type: Some(compartment_type.into()),
            message: message.into(),
        }
    }
    
    /// Create a new social error
    pub fn social_error<S: Into<String>>(message: S) -> Self {
        Self::SocialError {
            relationship_type: None,
            nodes_involved: Vec::new(),
            message: message.into(),
        }
    }
    
    /// Create a new social error with details
    pub fn social_error_detailed<S: Into<String>>(
        relationship_type: S, 
        nodes_involved: Vec<String>, 
        message: S
    ) -> Self {
        Self::SocialError {
            relationship_type: Some(relationship_type.into()),
            nodes_involved,
            message: message.into(),
        }
    }
    
    /// Create a new HAVOC error
    pub fn havoc_error<S: Into<String>>(message: S) -> Self {
        Self::HavocError {
            crisis_type: None,
            severity: None,
            message: message.into(),
        }
    }
    
    /// Create a new HAVOC error with crisis details
    pub fn havoc_error_detailed<S: Into<String>>(
        crisis_type: S, 
        severity: S, 
        message: S
    ) -> Self {
        Self::HavocError {
            crisis_type: Some(crisis_type.into()),
            severity: Some(severity.into()),
            message: message.into(),
        }
    }
    
    /// Create a new thermal error
    pub fn thermal_error<S: Into<String>>(message: S) -> Self {
        Self::ThermalError {
            node_id: None,
            thermal_level: None,
            message: message.into(),
        }
    }
    
    /// Create a new thermal error with details
    pub fn thermal_error_detailed<S: Into<String>>(
        node_id: S, 
        thermal_level: f64, 
        message: S
    ) -> Self {
        Self::ThermalError {
            node_id: Some(node_id.into()),
            thermal_level: Some(thermal_level),
            message: message.into(),
        }
    }
    
    /// Create a new scaling error
    pub fn scaling_error<S: Into<String>>(message: S) -> Self {
        Self::ScalingError {
            node_id: None,
            scaling_factor: None,
            message: message.into(),
        }
    }
    
    /// Create a new scaling error with details
    pub fn scaling_error_detailed<S: Into<String>>(
        node_id: S, 
        scaling_factor: f64, 
        message: S
    ) -> Self {
        Self::ScalingError {
            node_id: Some(node_id.into()),
            scaling_factor: Some(scaling_factor),
            message: message.into(),
        }
    }
    
    /// Create a new network error
    pub fn network_error<S: Into<String>>(operation: S, message: S) -> Self {
        Self::NetworkError {
            endpoint: None,
            operation: operation.into(),
            message: message.into(),
        }
    }
    
    /// Create a new network error with endpoint
    pub fn network_error_with_endpoint<S: Into<String>>(
        endpoint: S, 
        operation: S, 
        message: S
    ) -> Self {
        Self::NetworkError {
            endpoint: Some(endpoint.into()),
            operation: operation.into(),
            message: message.into(),
        }
    }
    
    /// Create a new configuration error
    pub fn configuration_error<S: Into<String>>(parameter: S, message: S) -> Self {
        Self::ConfigurationError {
            parameter: parameter.into(),
            value: None,
            message: message.into(),
        }
    }
    
    /// Create a new configuration error with value
    pub fn configuration_error_with_value<S: Into<String>>(
        parameter: S, 
        value: S, 
        message: S
    ) -> Self {
        Self::ConfigurationError {
            parameter: parameter.into(),
            value: Some(value.into()),
            message: message.into(),
        }
    }
    
    /// Create a new validation error
    pub fn validation_error<S: Into<String>>(field: S, value: S, constraint: S) -> Self {
        Self::ValidationError {
            field: field.into(),
            value: value.into(),
            constraint: constraint.into(),
        }
    }
    
    /// Create a new serialization error
    pub fn serialization_error<S: Into<String>>(data_type: S, operation: S, message: S) -> Self {
        Self::SerializationError {
            data_type: data_type.into(),
            operation: operation.into(),
            message: message.into(),
        }
    }
    
    /// Create a new timeout error
    pub fn timeout_error<S: Into<String>>(operation: S, timeout_duration: std::time::Duration) -> Self {
        Self::TimeoutError {
            operation: operation.into(),
            timeout_duration,
        }
    }
    
    /// Create a new concurrency error
    pub fn concurrency_error<S: Into<String>>(resource: S, operation: S, message: S) -> Self {
        Self::ConcurrencyError {
            resource: resource.into(),
            operation: operation.into(),
            message: message.into(),
        }
    }
    
    /// Create a new metrics error
    pub fn metrics_error<S: Into<String>>(metric_type: S, message: S) -> Self {
        Self::MetricsError {
            metric_type: metric_type.into(),
            message: message.into(),
        }
    }
    
    /// Create a new internal error
    pub fn internal_error<S: Into<String>>(component: S, message: S) -> Self {
        Self::InternalError {
            component: component.into(),
            message: message.into(),
        }
    }
    
    /// Create a new external error
    pub fn external_error<S: Into<String>>(service: S, operation: S, message: S) -> Self {
        Self::ExternalError {
            service: service.into(),
            operation: operation.into(),
            status_code: None,
            message: message.into(),
        }
    }
    
    /// Create a new external error with status code
    pub fn external_error_with_status<S: Into<String>>(
        service: S, 
        operation: S, 
        status_code: u16, 
        message: S
    ) -> Self {
        Self::ExternalError {
            service: service.into(),
            operation: operation.into(),
            status_code: Some(status_code),
            message: message.into(),
        }
    }
    
    /// Get error category for logging and metrics
    pub fn category(&self) -> &'static str {
        match self {
            Self::AllocationFailed { .. } => "allocation",
            Self::InsufficientResources { .. } => "resources",
            Self::CompartmentError { .. } => "compartment",
            Self::SocialError { .. } => "social",
            Self::HavocError { .. } => "havoc",
            Self::ThermalError { .. } => "thermal",
            Self::ScalingError { .. } => "scaling",
            Self::NetworkError { .. } => "network",
            Self::ConfigurationError { .. } => "configuration",
            Self::ValidationError { .. } => "validation",
            Self::SerializationError { .. } => "serialization",
            Self::TimeoutError { .. } => "timeout",
            Self::ConcurrencyError { .. } => "concurrency",
            Self::MetricsError { .. } => "metrics",
            Self::InternalError { .. } => "internal",
            Self::ExternalError { .. } => "external",
        }
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::AllocationFailed { .. } => ErrorSeverity::High,
            Self::InsufficientResources { .. } => ErrorSeverity::High,
            Self::CompartmentError { .. } => ErrorSeverity::Medium,
            Self::SocialError { .. } => ErrorSeverity::Low,
            Self::HavocError { .. } => ErrorSeverity::Critical,
            Self::ThermalError { .. } => ErrorSeverity::High,
            Self::ScalingError { .. } => ErrorSeverity::Medium,
            Self::NetworkError { .. } => ErrorSeverity::Medium,
            Self::ConfigurationError { .. } => ErrorSeverity::High,
            Self::ValidationError { .. } => ErrorSeverity::Medium,
            Self::SerializationError { .. } => ErrorSeverity::Low,
            Self::TimeoutError { .. } => ErrorSeverity::Medium,
            Self::ConcurrencyError { .. } => ErrorSeverity::Medium,
            Self::MetricsError { .. } => ErrorSeverity::Low,
            Self::InternalError { .. } => ErrorSeverity::Critical,
            Self::ExternalError { .. } => ErrorSeverity::Medium,
        }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::AllocationFailed { .. } => true,
            Self::InsufficientResources { .. } => true,
            Self::CompartmentError { .. } => true,
            Self::SocialError { .. } => true,
            Self::HavocError { .. } => false, // Crisis situations need manual intervention
            Self::ThermalError { .. } => true,
            Self::ScalingError { .. } => true,
            Self::NetworkError { .. } => true,
            Self::ConfigurationError { .. } => false, // Configuration issues need fixing
            Self::ValidationError { .. } => false, // Validation failures need correction
            Self::SerializationError { .. } => false,
            Self::TimeoutError { .. } => true,
            Self::ConcurrencyError { .. } => true,
            Self::MetricsError { .. } => true,
            Self::InternalError { .. } => false,
            Self::ExternalError { .. } => true,
        }
    }
    
    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<&'static str> {
        match self {
            Self::AllocationFailed { .. } => vec!["Retry with different parameters", "Check resource availability"],
            Self::InsufficientResources { .. } => vec!["Wait for resources to become available", "Scale up capacity"],
            Self::CompartmentError { .. } => vec!["Restart compartment", "Check compartment configuration"],
            Self::SocialError { .. } => vec!["Re-establish relationship", "Check node connectivity"],
            Self::HavocError { .. } => vec!["Manual intervention required", "Escalate to administrator"],
            Self::ThermalError { .. } => vec!["Reduce load", "Check cooling systems"],
            Self::ScalingError { .. } => vec!["Check scaling constraints", "Verify resource availability"],
            Self::NetworkError { .. } => vec!["Check network connectivity", "Retry operation"],
            Self::ConfigurationError { .. } => vec!["Fix configuration", "Restart service"],
            Self::ValidationError { .. } => vec!["Correct input data", "Check validation rules"],
            Self::SerializationError { .. } => vec!["Check data format", "Update serialization version"],
            Self::TimeoutError { .. } => vec!["Increase timeout", "Check system performance"],
            Self::ConcurrencyError { .. } => vec!["Retry operation", "Reduce concurrent access"],
            Self::MetricsError { .. } => vec!["Check metrics collection", "Restart metrics service"],
            Self::InternalError { .. } => vec!["Restart system", "Contact support"],
            Self::ExternalError { .. } => vec!["Check external service", "Implement fallback"],
        }
    }
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - warning level
    Low = 1,
    /// Medium severity - error level
    Medium = 2,
    /// High severity - major error
    High = 3,
    /// Critical severity - system failure
    Critical = 4,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl fmt::Display for ResourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllocationFailed { reason, resource_type, requested_amount } => {
                let resource_info = match (resource_type, requested_amount) {
                    (Some(rt), Some(ra)) => format!(" (resource: {}, requested: {})", rt, ra),
                    (Some(rt), None) => format!(" (resource: {})", rt),
                    _ => String::new(),
                };
                write!(f, "Resource allocation failed: {}{}", reason, resource_info)
            }
            Self::InsufficientResources { resource_type, requested, available } => {
                write!(f, "Insufficient {} resources: requested {}, available {}", 
                    resource_type, requested, available)
            }
            Self::CompartmentError { compartment_type, message } => {
                let compartment_info = compartment_type
                    .as_ref()
                    .map(|ct| format!(" ({})", ct))
                    .unwrap_or_default();
                write!(f, "Compartment error{}: {}", compartment_info, message)
            }
            Self::SocialError { relationship_type, nodes_involved, message } => {
                let relationship_info = relationship_type
                    .as_ref()
                    .map(|rt| format!(" ({})", rt))
                    .unwrap_or_default();
                let nodes_info = if !nodes_involved.is_empty() {
                    format!(" involving nodes: {}", nodes_involved.join(", "))
                } else {
                    String::new()
                };
                write!(f, "Social error{}{}: {}", relationship_info, nodes_info, message)
            }
            Self::HavocError { crisis_type, severity, message } => {
                let crisis_info = match (crisis_type, severity) {
                    (Some(ct), Some(s)) => format!(" (crisis: {}, severity: {})", ct, s),
                    (Some(ct), None) => format!(" (crisis: {})", ct),
                    (None, Some(s)) => format!(" (severity: {})", s),
                    _ => String::new(),
                };
                write!(f, "HAVOC crisis error{}: {}", crisis_info, message)
            }
            Self::ThermalError { node_id, thermal_level, message } => {
                let thermal_info = match (node_id, thermal_level) {
                    (Some(ni), Some(tl)) => format!(" (node: {}, level: {:.2})", ni, tl),
                    (Some(ni), None) => format!(" (node: {})", ni),
                    (None, Some(tl)) => format!(" (level: {:.2})", tl),
                    _ => String::new(),
                };
                write!(f, "Thermal error{}: {}", thermal_info, message)
            }
            Self::ScalingError { node_id, scaling_factor, message } => {
                let scaling_info = match (node_id, scaling_factor) {
                    (Some(ni), Some(sf)) => format!(" (node: {}, factor: {:.2})", ni, sf),
                    (Some(ni), None) => format!(" (node: {})", ni),
                    (None, Some(sf)) => format!(" (factor: {:.2})", sf),
                    _ => String::new(),
                };
                write!(f, "Scaling error{}: {}", scaling_info, message)
            }
            Self::NetworkError { endpoint, operation, message } => {
                let endpoint_info = endpoint
                    .as_ref()
                    .map(|ep| format!(" (endpoint: {})", ep))
                    .unwrap_or_default();
                write!(f, "Network error during {}{}: {}", operation, endpoint_info, message)
            }
            Self::ConfigurationError { parameter, value, message } => {
                let value_info = value
                    .as_ref()
                    .map(|v| format!(" = {}", v))
                    .unwrap_or_default();
                write!(f, "Configuration error for parameter {}{}: {}", parameter, value_info, message)
            }
            Self::ValidationError { field, value, constraint } => {
                write!(f, "Validation error: field '{}' with value '{}' violates constraint: {}", 
                    field, value, constraint)
            }
            Self::SerializationError { data_type, operation, message } => {
                write!(f, "Serialization error during {} of {}: {}", operation, data_type, message)
            }
            Self::TimeoutError { operation, timeout_duration } => {
                write!(f, "Timeout error: {} exceeded {} seconds", 
                    operation, timeout_duration.as_secs())
            }
            Self::ConcurrencyError { resource, operation, message } => {
                write!(f, "Concurrency error accessing {} during {}: {}", resource, operation, message)
            }
            Self::MetricsError { metric_type, message } => {
                write!(f, "Metrics error for {}: {}", metric_type, message)
            }
            Self::InternalError { component, message } => {
                write!(f, "Internal error in {}: {}", component, message)
            }
            Self::ExternalError { service, operation, status_code, message } => {
                let status_info = status_code
                    .map(|sc| format!(" (status: {})", sc))
                    .unwrap_or_default();
                write!(f, "External error from {} during {}{}: {}", 
                    service, operation, status_info, message)
            }
        }
    }
}

impl StdError for ResourceError {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        // This could be extended to wrap underlying errors
        None
    }
}

// Conversion from common error types
impl From<std::io::Error> for ResourceError {
    fn from(err: std::io::Error) -> Self {
        ResourceError::internal_error("io", err.to_string())
    }
}

impl From<serde_json::Error> for ResourceError {
    fn from(err: serde_json::Error) -> Self {
        ResourceError::serialization_error("json", "deserialize", err.to_string())
    }
}

impl From<tokio::time::error::Elapsed> for ResourceError {
    fn from(_: tokio::time::error::Elapsed) -> Self {
        ResourceError::timeout_error("operation", std::time::Duration::from_secs(30))
    }
}

impl From<chrono::OutOfRangeError> for ResourceError {
    fn from(err: chrono::OutOfRangeError) -> Self {
        ResourceError::validation_error("timestamp", err.to_string(), "valid timestamp range")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_allocation_failed_error() {
        let error = ResourceError::allocation_failed("Not enough memory");
        assert_eq!(error.category(), "allocation");
        assert_eq!(error.severity(), ErrorSeverity::High);
        assert!(error.is_recoverable());
        
        let error_detailed = ResourceError::allocation_failed_detailed(
            "Out of CPU", "cpu", 2.5
        );
        assert!(error_detailed.to_string().contains("cpu"));
        assert!(error_detailed.to_string().contains("2.5"));
    }
    
    #[test]
    fn test_insufficient_resources_error() {
        let error = ResourceError::insufficient_resources("memory", 8.0, 4.0);
        assert!(error.to_string().contains("memory"));
        assert!(error.to_string().contains("requested 8"));
        assert!(error.to_string().contains("available 4"));
    }
    
    #[test]
    fn test_compartment_error() {
        let error = ResourceError::compartment_error("Compartment failed to start");
        assert_eq!(error.category(), "compartment");
        assert_eq!(error.severity(), ErrorSeverity::Medium);
        
        let typed_error = ResourceError::compartment_error_typed(
            "Training", "Out of GPU memory"
        );
        assert!(typed_error.to_string().contains("Training"));
    }
    
    #[test]
    fn test_social_error() {
        let error = ResourceError::social_error("Trust relationship failed");
        assert_eq!(error.category(), "social");
        assert_eq!(error.severity(), ErrorSeverity::Low);
        
        let detailed_error = ResourceError::social_error_detailed(
            "Friendship",
            vec!["node1".to_string(), "node2".to_string()],
            "Connection lost"
        );
        assert!(detailed_error.to_string().contains("node1, node2"));
    }
    
    #[test]
    fn test_havoc_error() {
        let error = ResourceError::havoc_error("Crisis detected");
        assert_eq!(error.category(), "havoc");
        assert_eq!(error.severity(), ErrorSeverity::Critical);
        assert!(!error.is_recoverable());
        
        let detailed_error = ResourceError::havoc_error_detailed(
            "ResourceShortage", "Critical", "Network overload"
        );
        assert!(detailed_error.to_string().contains("ResourceShortage"));
        assert!(detailed_error.to_string().contains("Critical"));
    }
    
    #[test]
    fn test_thermal_error() {
        let error = ResourceError::thermal_error("Overheating detected");
        assert_eq!(error.category(), "thermal");
        
        let detailed_error = ResourceError::thermal_error_detailed(
            "node-123", 0.95, "Critical temperature"
        );
        assert!(detailed_error.to_string().contains("node-123"));
        assert!(detailed_error.to_string().contains("0.95"));
    }
    
    #[test]
    fn test_scaling_error() {
        let error = ResourceError::scaling_error("Failed to scale");
        assert_eq!(error.category(), "scaling");
        
        let detailed_error = ResourceError::scaling_error_detailed(
            "worker-node", 2.5, "Insufficient capacity"
        );
        assert!(detailed_error.to_string().contains("worker-node"));
        assert!(detailed_error.to_string().contains("2.5"));
    }
    
    #[test]
    fn test_network_error() {
        let error = ResourceError::network_error("connection", "Network unreachable");
        assert_eq!(error.category(), "network");
        
        let endpoint_error = ResourceError::network_error_with_endpoint(
            "192.168.1.100:8080", "GET", "Connection timeout"
        );
        assert!(endpoint_error.to_string().contains("192.168.1.100:8080"));
    }
    
    #[test]
    fn test_configuration_error() {
        let error = ResourceError::configuration_error("max_connections", "Invalid parameter");
        assert_eq!(error.category(), "configuration");
        assert!(!error.is_recoverable());
        
        let value_error = ResourceError::configuration_error_with_value(
            "timeout", "invalid", "Must be a number"
        );
        assert!(value_error.to_string().contains("timeout"));
        assert!(value_error.to_string().contains("invalid"));
    }
    
    #[test]
    fn test_validation_error() {
        let error = ResourceError::validation_error(
            "cpu_cores", "-1", "must be positive"
        );
        assert!(error.to_string().contains("cpu_cores"));
        assert!(error.to_string().contains("-1"));
        assert!(error.to_string().contains("must be positive"));
    }
    
    #[test]
    fn test_timeout_error() {
        let error = ResourceError::timeout_error("database_query", std::time::Duration::from_secs(30));
        assert!(error.to_string().contains("database_query"));
        assert!(error.to_string().contains("30 seconds"));
    }
    
    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }
    
    #[test]
    fn test_error_recovery_suggestions() {
        let error = ResourceError::allocation_failed("Test failure");
        let suggestions = error.recovery_suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("Retry")));
    }
    
    #[test]
    fn test_error_from_conversions() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let resource_error = ResourceError::from(io_error);
        assert_eq!(resource_error.category(), "internal");
        
        let json_error = serde_json::from_str::<serde_json::Value>("invalid json").unwrap_err();
        let resource_error = ResourceError::from(json_error);
        assert_eq!(resource_error.category(), "serialization");
    }
    
    #[test]
    fn test_error_display() {
        let error = ResourceError::insufficient_resources("cpu", 4.0, 2.0);
        let display = error.to_string();
        assert!(display.contains("Insufficient cpu resources"));
        assert!(display.contains("requested 4"));
        assert!(display.contains("available 2"));
    }
}