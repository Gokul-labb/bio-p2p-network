use thiserror::Error;

/// Core protocol errors for the biological P2P network
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ProtocolError {
    /// Invalid network address format
    #[error("Invalid network address format: {address}")]
    InvalidAddress { address: String },
    
    /// Package size exceeds maximum allowed
    #[error("Package size {size} exceeds maximum {max_size}")]
    PackageTooLarge { size: usize, max_size: usize },
    
    /// Invalid biological role assignment
    #[error("Invalid biological role: {role}")]
    InvalidBiologicalRole { role: String },
    
    /// Security validation failed
    #[error("Security validation failed: {reason}")]
    SecurityValidationFailed { reason: String },
    
    /// Reputation score calculation error
    #[error("Reputation score calculation failed: {reason}")]
    ReputationCalculationFailed { reason: String },
    
    /// Thermal signature invalid
    #[error("Invalid thermal signature: {reason}")]
    InvalidThermalSignature { reason: String },
    
    /// Node synchronization timeout
    #[error("Node synchronization timeout after {timeout}s")]
    SynchronizationTimeout { timeout: u64 },
    
    /// Package lifecycle violation
    #[error("Package lifecycle violation: expected {expected}, got {actual}")]
    PackageLifecycleViolation { expected: String, actual: String },
    
    /// Message serialization error
    #[error("Message serialization error: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },
    
    /// Cryptographic error
    #[error("Cryptographic error: {reason}")]
    CryptographicError { reason: String },
    
    /// Network partition detected
    #[error("Network partition detected: {partition_size}% of network unreachable")]
    NetworkPartition { partition_size: u8 },
    
    /// Byzantine behavior detected
    #[error("Byzantine behavior detected from node {node_id}")]
    ByzantineNode { node_id: String },
    
    /// Resource exhaustion
    #[error("Resource exhaustion: {resource} at {usage}% capacity")]
    ResourceExhaustion { resource: String, usage: u8 },
    
    /// Consensus failure
    #[error("Consensus failure: insufficient responses ({responses}/{required})")]
    ConsensusFailed { responses: usize, required: usize },
}

/// Specialized Result type for protocol operations
pub type Result<T> = std::result::Result<T, ProtocolError>;

/// Security-related errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SecurityError {
    #[error("Authentication failed")]
    AuthenticationFailed,
    
    #[error("Authorization denied for operation {operation}")]
    AuthorizationDenied { operation: String },
    
    #[error("Encryption failed: {reason}")]
    EncryptionFailed { reason: String },
    
    #[error("Integrity check failed")]
    IntegrityCheckFailed,
    
    #[error("Malicious behavior detected: {behavior}")]
    MaliciousBehavior { behavior: String },
}

/// Network-related errors
#[derive(Error, Debug, Clone, PartialEq)]
pub enum NetworkError {
    #[error("Connection timeout to {address}")]
    ConnectionTimeout { address: String },
    
    #[error("Network unreachable: {reason}")]
    NetworkUnreachable { reason: String },
    
    #[error("Protocol version mismatch: local {local}, remote {remote}")]
    ProtocolVersionMismatch { local: String, remote: String },
    
    #[error("Node capacity exceeded: {current}/{max}")]
    CapacityExceeded { current: usize, max: usize },
}

impl ProtocolError {
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        matches!(self, 
            ProtocolError::SynchronizationTimeout { .. } |
            ProtocolError::NetworkPartition { partition_size } if *partition_size < 60 |
            ProtocolError::ConsensusFailed { .. } |
            ProtocolError::ResourceExhaustion { .. }
        )
    }
    
    /// Get error severity level
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ProtocolError::SecurityValidationFailed { .. } |
            ProtocolError::ByzantineNode { .. } => ErrorSeverity::Critical,
            
            ProtocolError::NetworkPartition { .. } |
            ProtocolError::ConsensusFailed { .. } => ErrorSeverity::High,
            
            ProtocolError::SynchronizationTimeout { .. } |
            ProtocolError::ResourceExhaustion { .. } => ErrorSeverity::Medium,
            
            ProtocolError::InvalidAddress { .. } |
            ProtocolError::PackageTooLarge { .. } => ErrorSeverity::Low,
            
            _ => ErrorSeverity::Medium,
        }
    }
}

/// Error severity levels for logging and handling
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_recovery() {
        let recoverable = ProtocolError::SynchronizationTimeout { timeout: 30 };
        assert!(recoverable.is_recoverable());
        
        let non_recoverable = ProtocolError::ByzantineNode { 
            node_id: "test".to_string() 
        };
        assert!(!non_recoverable.is_recoverable());
    }
    
    #[test]
    fn test_error_severity() {
        let critical = ProtocolError::ByzantineNode { 
            node_id: "test".to_string() 
        };
        assert_eq!(critical.severity(), ErrorSeverity::Critical);
        
        let low = ProtocolError::InvalidAddress { 
            address: "invalid".to_string() 
        };
        assert_eq!(low.severity(), ErrorSeverity::Low);
    }
}