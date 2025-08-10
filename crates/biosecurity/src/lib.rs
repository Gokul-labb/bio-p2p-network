//! Biological P2P Security Framework
//! 
//! A comprehensive 5-tier security framework inspired by biological immune systems
//! and natural defense mechanisms. Implements multi-layer execution, sanitization,
//! deception, behavioral monitoring, and thermal detection.

pub mod config;
pub mod crypto;
pub mod errors;
pub mod framework;
pub mod layers;
pub mod nodes;

// Re-export main types for convenience
pub use framework::{SecurityFramework, FrameworkStatus, FrameworkMetrics, SecurityEventHandler};
pub use config::{SecurityConfig, LayerConfig, CryptoConfig};
pub use errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
pub use layers::{SecurityContext, RiskLevel};
pub use crypto::CryptoContext;

/// Current version of the bio-security crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Biological security framework constants
pub mod constants {
    use std::time::Duration;

    /// Default execution timeout for security operations
    pub const DEFAULT_EXECUTION_TIMEOUT: Duration = Duration::from_secs(300);
    
    /// Default DoD 5220.22-M sanitization passes
    pub const DOD_SANITIZATION_PASSES: usize = 3;
    
    /// Default 3-sigma threshold for anomaly detection
    pub const THREE_SIGMA_THRESHOLD: f64 = 3.0;
    
    /// Default thermal monitoring sampling frequency
    pub const THERMAL_SAMPLING_FREQUENCY: Duration = Duration::from_secs(1);
    
    /// Maximum number of layers in the biological security stack
    pub const MAX_SECURITY_LAYERS: usize = 5;
    
    /// Default honeypot count for illusion layer
    pub const DEFAULT_HONEYPOT_COUNT: usize = 5;
    
    /// Default false topology complexity
    pub const DEFAULT_FALSE_TOPOLOGY_COMPLEXITY: f32 = 0.7;
}

/// Biological security prelude for common imports
pub mod prelude {
    pub use crate::{
        SecurityFramework,
        SecurityConfig,
        SecurityResult,
        SecurityEvent,
        SecuritySeverity,
        SecurityContext,
        RiskLevel,
        CryptoContext,
    };
    
    pub use crate::framework::{FrameworkStatus, FrameworkMetrics, SecurityEventHandler};
    pub use crate::config::{LayerConfig, CryptoConfig, FrameworkConfig};
    pub use crate::errors::SecurityError;
    pub use crate::layers::{SecurityLayer, LayerStatus, LayerMetrics};
    pub use crate::nodes::{SecurityNode, SecurityNodeStatus, SecurityNodeMetrics};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_constant() {
        assert!(!VERSION.is_empty());
        println!("Bio-Security Framework Version: {}", VERSION);
    }
    
    #[test]
    fn test_constants() {
        use constants::*;
        
        assert_eq!(DEFAULT_EXECUTION_TIMEOUT.as_secs(), 300);
        assert_eq!(DOD_SANITIZATION_PASSES, 3);
        assert_eq!(THREE_SIGMA_THRESHOLD, 3.0);
        assert_eq!(MAX_SECURITY_LAYERS, 5);
        assert_eq!(DEFAULT_HONEYPOT_COUNT, 5);
        assert_eq!(DEFAULT_FALSE_TOPOLOGY_COMPLEXITY, 0.7);
    }
    
    #[tokio::test]
    async fn test_basic_framework_creation() {
        let config = SecurityConfig::for_testing();
        let result = SecurityFramework::new(config).await;
        
        assert!(result.is_ok(), "Should be able to create framework with test config");
        
        let framework = result.unwrap();
        assert_eq!(framework.status().await, FrameworkStatus::Initializing);
    }
    
    #[test]
    fn test_prelude_imports() {
        use prelude::*;
        
        // Test that all prelude imports are accessible
        let _config = SecurityConfig::for_testing();
        let _context = SecurityContext::new("test".to_string(), "node".to_string());
        let _event = SecurityEvent::new(SecuritySeverity::Info, "test", "Test event");
        
        // If this compiles, all prelude imports work correctly
    }
}