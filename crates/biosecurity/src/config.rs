//! Configuration structures for the biological security framework
//! 
//! Defines all configuration options for layers, nodes, cryptography, and
//! framework-wide settings with validation and defaults.

use std::time::Duration;
use serde::{Deserialize, Serialize};

use crate::errors::{SecurityError, SecurityResult};

/// Complete security framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Cryptographic configuration
    pub crypto: CryptoConfig,
    /// Configuration for each security layer
    pub layers: Vec<LayerConfig>,
    /// Framework-wide settings
    pub framework: FrameworkConfig,
}

impl SecurityConfig {
    /// Create default configuration for production use
    pub fn default() -> Self {
        Self {
            crypto: CryptoConfig::default(),
            layers: vec![
                LayerConfig::multi_layer_execution(),
                LayerConfig::cbadu(),
                LayerConfig::illusion_layer(),
                LayerConfig::behavior_monitoring(),
                LayerConfig::thermal_detection(),
            ],
            framework: FrameworkConfig::default(),
        }
    }

    /// Create configuration optimized for testing
    pub fn for_testing() -> Self {
        Self {
            crypto: CryptoConfig::for_testing(),
            layers: vec![
                LayerConfig::multi_layer_execution_testing(),
                LayerConfig::cbadu_testing(),
                LayerConfig::illusion_layer_testing(),
                LayerConfig::behavior_monitoring_testing(),
                LayerConfig::thermal_detection_testing(),
            ],
            framework: FrameworkConfig::for_testing(),
        }
    }

    /// Create high-security configuration
    pub fn high_security() -> Self {
        Self {
            crypto: CryptoConfig::high_security(),
            layers: vec![
                LayerConfig::multi_layer_execution_high_security(),
                LayerConfig::cbadu_high_security(),
                LayerConfig::illusion_layer_high_security(),
                LayerConfig::behavior_monitoring_high_security(),
                LayerConfig::thermal_detection_high_security(),
            ],
            framework: FrameworkConfig::high_security(),
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> SecurityResult<()> {
        // Validate crypto config
        self.crypto.validate()?;
        
        // Validate framework config
        self.framework.validate()?;
        
        // Validate layer configs
        if self.layers.len() != 5 {
            return Err(SecurityError::ConfigurationError(
                "Must have exactly 5 security layers".to_string()
            ));
        }
        
        for (i, layer_config) in self.layers.iter().enumerate() {
            layer_config.validate(i + 1)?;
        }
        
        Ok(())
    }
}

/// Cryptographic configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptoConfig {
    /// Hash algorithm to use
    pub hash_algorithm: HashAlgorithm,
    /// Key derivation function iterations
    pub kdf_iterations: u32,
    /// Salt size for key derivation
    pub salt_size: usize,
    /// Enable secure key erasure
    pub secure_erasure: bool,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Sha3_256,
            kdf_iterations: 100_000,
            salt_size: 32,
            secure_erasure: true,
        }
    }
}

impl CryptoConfig {
    /// Configuration for testing (faster but less secure)
    pub fn for_testing() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Blake3,
            kdf_iterations: 1_000,  // Much faster for tests
            salt_size: 16,
            secure_erasure: false,  // Skip secure erasure in tests
        }
    }

    /// High security configuration
    pub fn high_security() -> Self {
        Self {
            hash_algorithm: HashAlgorithm::Sha3_512,
            kdf_iterations: 200_000,
            salt_size: 64,
            secure_erasure: true,
        }
    }

    /// Validate cryptographic configuration
    pub fn validate(&self) -> SecurityResult<()> {
        if self.kdf_iterations < 1000 {
            return Err(SecurityError::ConfigurationError(
                "KDF iterations must be at least 1000".to_string()
            ));
        }
        
        if self.salt_size < 16 {
            return Err(SecurityError::ConfigurationError(
                "Salt size must be at least 16 bytes".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Supported hash algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HashAlgorithm {
    /// SHA-3 256-bit (NIST standard)
    Sha3_256,
    /// SHA-3 512-bit (Higher security)
    Sha3_512,
    /// BLAKE3 (High performance)
    Blake3,
}

/// Security layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer identifier (1-5)
    pub layer_id: usize,
    /// Layer-specific settings
    pub settings: LayerSettings,
    /// Whether layer is enabled
    pub enabled: bool,
}

impl LayerConfig {
    /// Layer 1: Multi-Layer Execution configuration
    pub fn multi_layer_execution() -> Self {
        Self {
            layer_id: 1,
            settings: LayerSettings::MultiLayerExecution {
                monitoring_layers: 3,
                isolation_level: IsolationLevel::Full,
                container_runtime: ContainerRuntime::Docker,
                randomization_enabled: true,
            },
            enabled: true,
        }
    }

    pub fn multi_layer_execution_testing() -> Self {
        Self {
            layer_id: 1,
            settings: LayerSettings::MultiLayerExecution {
                monitoring_layers: 1,
                isolation_level: IsolationLevel::Basic,
                container_runtime: ContainerRuntime::Native,
                randomization_enabled: false,
            },
            enabled: true,
        }
    }

    pub fn multi_layer_execution_high_security() -> Self {
        Self {
            layer_id: 1,
            settings: LayerSettings::MultiLayerExecution {
                monitoring_layers: 5,
                isolation_level: IsolationLevel::Enhanced,
                container_runtime: ContainerRuntime::Docker,
                randomization_enabled: true,
            },
            enabled: true,
        }
    }

    /// Layer 2: CBADU configuration
    pub fn cbadu() -> Self {
        Self {
            layer_id: 2,
            settings: LayerSettings::CBADU {
                sanitization_passes: 3,
                verification_enabled: true,
                secure_overwrite: true,
                memory_clearing: true,
            },
            enabled: true,
        }
    }

    pub fn cbadu_testing() -> Self {
        Self {
            layer_id: 2,
            settings: LayerSettings::CBADU {
                sanitization_passes: 1,
                verification_enabled: false,
                secure_overwrite: false,
                memory_clearing: false,
            },
            enabled: true,
        }
    }

    pub fn cbadu_high_security() -> Self {
        Self {
            layer_id: 2,
            settings: LayerSettings::CBADU {
                sanitization_passes: 7,
                verification_enabled: true,
                secure_overwrite: true,
                memory_clearing: true,
            },
            enabled: true,
        }
    }

    /// Layer 3: Illusion Layer configuration
    pub fn illusion_layer() -> Self {
        Self {
            layer_id: 3,
            settings: LayerSettings::IllusionLayer {
                deception_enabled: true,
                honeypot_count: 5,
                false_topology_complexity: 0.7,
                misdirection_probability: 0.3,
            },
            enabled: true,
        }
    }

    pub fn illusion_layer_testing() -> Self {
        Self {
            layer_id: 3,
            settings: LayerSettings::IllusionLayer {
                deception_enabled: false,
                honeypot_count: 1,
                false_topology_complexity: 0.1,
                misdirection_probability: 0.0,
            },
            enabled: true,
        }
    }

    pub fn illusion_layer_high_security() -> Self {
        Self {
            layer_id: 3,
            settings: LayerSettings::IllusionLayer {
                deception_enabled: true,
                honeypot_count: 15,
                false_topology_complexity: 0.9,
                misdirection_probability: 0.6,
            },
            enabled: true,
        }
    }

    /// Layer 4: Behavior Monitoring configuration
    pub fn behavior_monitoring() -> Self {
        Self {
            layer_id: 4,
            settings: LayerSettings::BehaviorMonitoring {
                baseline_learning_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
                anomaly_threshold: 3.0,
                ml_enabled: true,
                pattern_window_size: 1000,
            },
            enabled: true,
        }
    }

    pub fn behavior_monitoring_testing() -> Self {
        Self {
            layer_id: 4,
            settings: LayerSettings::BehaviorMonitoring {
                baseline_learning_period: Duration::from_secs(60), // 1 minute
                anomaly_threshold: 5.0, // Higher threshold for less noise in tests
                ml_enabled: false,
                pattern_window_size: 10,
            },
            enabled: true,
        }
    }

    pub fn behavior_monitoring_high_security() -> Self {
        Self {
            layer_id: 4,
            settings: LayerSettings::BehaviorMonitoring {
                baseline_learning_period: Duration::from_secs(60 * 24 * 60 * 60), // 60 days
                anomaly_threshold: 2.0, // More sensitive
                ml_enabled: true,
                pattern_window_size: 5000,
            },
            enabled: true,
        }
    }

    /// Layer 5: Thermal Detection configuration
    pub fn thermal_detection() -> Self {
        Self {
            layer_id: 5,
            settings: LayerSettings::ThermalDetection {
                sampling_frequency: Duration::from_secs(1),
                history_retention: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
                cpu_threshold: 0.9,
                memory_threshold: 0.9,
                network_threshold: 100_000_000, // 100 MB/s
                storage_threshold: 50_000_000,  // 50 MB/s
            },
            enabled: true,
        }
    }

    pub fn thermal_detection_testing() -> Self {
        Self {
            layer_id: 5,
            settings: LayerSettings::ThermalDetection {
                sampling_frequency: Duration::from_secs(10),
                history_retention: Duration::from_secs(60), // 1 minute
                cpu_threshold: 1.0, // No threshold in tests
                memory_threshold: 1.0,
                network_threshold: u64::MAX,
                storage_threshold: u64::MAX,
            },
            enabled: true,
        }
    }

    pub fn thermal_detection_high_security() -> Self {
        Self {
            layer_id: 5,
            settings: LayerSettings::ThermalDetection {
                sampling_frequency: Duration::from_millis(500),
                history_retention: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
                cpu_threshold: 0.8, // More sensitive
                memory_threshold: 0.8,
                network_threshold: 50_000_000,  // 50 MB/s
                storage_threshold: 25_000_000,  // 25 MB/s
            },
            enabled: true,
        }
    }

    /// Validate layer configuration
    pub fn validate(&self, expected_id: usize) -> SecurityResult<()> {
        if self.layer_id != expected_id {
            return Err(SecurityError::ConfigurationError(
                format!("Layer ID mismatch: expected {}, got {}", expected_id, self.layer_id)
            ));
        }

        match &self.settings {
            LayerSettings::MultiLayerExecution { monitoring_layers, .. } => {
                if *monitoring_layers == 0 {
                    return Err(SecurityError::ConfigurationError(
                        "Must have at least 1 monitoring layer".to_string()
                    ));
                }
            },
            LayerSettings::CBADU { sanitization_passes, .. } => {
                if *sanitization_passes == 0 {
                    return Err(SecurityError::ConfigurationError(
                        "Must have at least 1 sanitization pass".to_string()
                    ));
                }
            },
            LayerSettings::IllusionLayer { false_topology_complexity, misdirection_probability, .. } => {
                if *false_topology_complexity < 0.0 || *false_topology_complexity > 1.0 {
                    return Err(SecurityError::ConfigurationError(
                        "False topology complexity must be between 0.0 and 1.0".to_string()
                    ));
                }
                if *misdirection_probability < 0.0 || *misdirection_probability > 1.0 {
                    return Err(SecurityError::ConfigurationError(
                        "Misdirection probability must be between 0.0 and 1.0".to_string()
                    ));
                }
            },
            LayerSettings::BehaviorMonitoring { anomaly_threshold, pattern_window_size, .. } => {
                if *anomaly_threshold <= 0.0 {
                    return Err(SecurityError::ConfigurationError(
                        "Anomaly threshold must be positive".to_string()
                    ));
                }
                if *pattern_window_size == 0 {
                    return Err(SecurityError::ConfigurationError(
                        "Pattern window size must be positive".to_string()
                    ));
                }
            },
            LayerSettings::ThermalDetection { cpu_threshold, memory_threshold, .. } => {
                if *cpu_threshold <= 0.0 || *cpu_threshold > 1.0 {
                    return Err(SecurityError::ConfigurationError(
                        "CPU threshold must be between 0.0 and 1.0".to_string()
                    ));
                }
                if *memory_threshold <= 0.0 || *memory_threshold > 1.0 {
                    return Err(SecurityError::ConfigurationError(
                        "Memory threshold must be between 0.0 and 1.0".to_string()
                    ));
                }
            },
        }

        Ok(())
    }
}

/// Layer-specific configuration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerSettings {
    /// Multi-Layer Execution settings
    MultiLayerExecution {
        monitoring_layers: usize,
        isolation_level: IsolationLevel,
        container_runtime: ContainerRuntime,
        randomization_enabled: bool,
    },
    /// CBADU settings
    CBADU {
        sanitization_passes: usize,
        verification_enabled: bool,
        secure_overwrite: bool,
        memory_clearing: bool,
    },
    /// Illusion Layer settings
    IllusionLayer {
        deception_enabled: bool,
        honeypot_count: usize,
        false_topology_complexity: f32,
        misdirection_probability: f32,
    },
    /// Behavior Monitoring settings
    BehaviorMonitoring {
        baseline_learning_period: Duration,
        anomaly_threshold: f64,
        ml_enabled: bool,
        pattern_window_size: usize,
    },
    /// Thermal Detection settings
    ThermalDetection {
        sampling_frequency: Duration,
        history_retention: Duration,
        cpu_threshold: f64,
        memory_threshold: f64,
        network_threshold: u64,
        storage_threshold: u64,
    },
}

/// Isolation levels for execution environments
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IsolationLevel {
    /// Basic process isolation
    Basic,
    /// Full containerized isolation
    Full,
    /// Enhanced isolation with additional security measures
    Enhanced,
}

/// Container runtime options
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContainerRuntime {
    /// Native process execution (no containers)
    Native,
    /// Docker container runtime
    Docker,
    /// Podman container runtime
    Podman,
    /// Containerd runtime
    Containerd,
}

/// Framework-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameworkConfig {
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Maximum concurrent executions
    pub max_concurrent_executions: usize,
    /// Execution timeout
    pub execution_timeout: Duration,
    /// Event processing timeout
    pub event_processing_timeout: Duration,
    /// Metrics collection interval
    pub metrics_interval: Duration,
}

impl Default for FrameworkConfig {
    fn default() -> Self {
        Self {
            detailed_logging: true,
            max_concurrent_executions: 100,
            execution_timeout: Duration::from_secs(300), // 5 minutes
            event_processing_timeout: Duration::from_secs(30),
            metrics_interval: Duration::from_secs(60),
        }
    }
}

impl FrameworkConfig {
    /// Configuration for testing
    pub fn for_testing() -> Self {
        Self {
            detailed_logging: false,
            max_concurrent_executions: 10,
            execution_timeout: Duration::from_secs(30),
            event_processing_timeout: Duration::from_secs(5),
            metrics_interval: Duration::from_secs(10),
        }
    }

    /// High security configuration
    pub fn high_security() -> Self {
        Self {
            detailed_logging: true,
            max_concurrent_executions: 50, // Lower limit for security
            execution_timeout: Duration::from_secs(120), // Shorter timeout
            event_processing_timeout: Duration::from_secs(10),
            metrics_interval: Duration::from_secs(30),
        }
    }

    /// Validate framework configuration
    pub fn validate(&self) -> SecurityResult<()> {
        if self.max_concurrent_executions == 0 {
            return Err(SecurityError::ConfigurationError(
                "Max concurrent executions must be positive".to_string()
            ));
        }

        if self.execution_timeout.as_secs() == 0 {
            return Err(SecurityError::ConfigurationError(
                "Execution timeout must be positive".to_string()
            ));
        }

        if self.event_processing_timeout.as_secs() == 0 {
            return Err(SecurityError::ConfigurationError(
                "Event processing timeout must be positive".to_string()
            ));
        }

        if self.metrics_interval.as_secs() == 0 {
            return Err(SecurityError::ConfigurationError(
                "Metrics interval must be positive".to_string()
            ));
        }

        Ok(())
    }
}

/// Load configuration from file
pub fn load_from_file(path: &str) -> SecurityResult<SecurityConfig> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| SecurityError::ConfigurationError(format!("Failed to read config file: {}", e)))?;
    
    let config: SecurityConfig = toml::from_str(&contents)
        .map_err(|e| SecurityError::ConfigurationError(format!("Failed to parse config: {}", e)))?;
    
    config.validate()?;
    Ok(config)
}

/// Save configuration to file
pub fn save_to_file(config: &SecurityConfig, path: &str) -> SecurityResult<()> {
    config.validate()?;
    
    let contents = toml::to_string_pretty(config)
        .map_err(|e| SecurityError::ConfigurationError(format!("Failed to serialize config: {}", e)))?;
    
    std::fs::write(path, contents)
        .map_err(|e| SecurityError::ConfigurationError(format!("Failed to write config file: {}", e)))?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn test_default_config_validation() {
        let config = SecurityConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_testing_config_validation() {
        let config = SecurityConfig::for_testing();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_high_security_config_validation() {
        let config = SecurityConfig::high_security();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_crypto_config_validation() {
        let mut config = CryptoConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid iterations
        config.kdf_iterations = 500;
        assert!(config.validate().is_err());

        // Test invalid salt size
        config.kdf_iterations = 10000;
        config.salt_size = 8;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_framework_config_validation() {
        let mut config = FrameworkConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid max concurrent executions
        config.max_concurrent_executions = 0;
        assert!(config.validate().is_err());

        // Test invalid timeout
        config.max_concurrent_executions = 10;
        config.execution_timeout = Duration::ZERO;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_layer_config_validation() {
        // Test valid layer config
        let config = LayerConfig::multi_layer_execution();
        assert!(config.validate(1).is_ok());

        // Test invalid layer ID
        assert!(config.validate(2).is_err());

        // Test invalid monitoring layers
        let mut invalid_config = config.clone();
        if let LayerSettings::MultiLayerExecution { ref mut monitoring_layers, .. } = invalid_config.settings {
            *monitoring_layers = 0;
        }
        assert!(invalid_config.validate(1).is_err());
    }

    #[test]
    fn test_illusion_layer_validation() {
        let mut config = LayerConfig::illusion_layer();
        
        // Test invalid complexity
        if let LayerSettings::IllusionLayer { ref mut false_topology_complexity, .. } = config.settings {
            *false_topology_complexity = 1.5;
        }
        assert!(config.validate(3).is_err());

        // Test invalid probability
        config = LayerConfig::illusion_layer();
        if let LayerSettings::IllusionLayer { ref mut misdirection_probability, .. } = config.settings {
            *misdirection_probability = -0.1;
        }
        assert!(config.validate(3).is_err());
    }

    #[test]
    fn test_thermal_detection_validation() {
        let mut config = LayerConfig::thermal_detection();
        
        // Test invalid CPU threshold
        if let LayerSettings::ThermalDetection { ref mut cpu_threshold, .. } = config.settings {
            *cpu_threshold = 1.5;
        }
        assert!(config.validate(5).is_err());

        // Test invalid memory threshold
        config = LayerConfig::thermal_detection();
        if let LayerSettings::ThermalDetection { ref mut memory_threshold, .. } = config.settings {
            *memory_threshold = -0.1;
        }
        assert!(config.validate(5).is_err());
    }

    #[test]
    fn test_config_file_operations() {
        let config = SecurityConfig::for_testing();
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();
        
        // Save config to file
        save_to_file(&config, file_path).unwrap();
        
        // Load config from file
        let loaded_config = load_from_file(file_path).unwrap();
        
        // Configs should be equivalent (can't directly compare due to Duration serialization)
        assert_eq!(loaded_config.layers.len(), config.layers.len());
        assert_eq!(loaded_config.crypto.hash_algorithm, config.crypto.hash_algorithm);
    }

    #[test]
    fn test_invalid_config_file() {
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"invalid toml content").unwrap();
        
        let file_path = temp_file.path().to_str().unwrap();
        let result = load_from_file(file_path);
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SecurityError::ConfigurationError(_) => {},
            _ => panic!("Expected configuration error"),
        }
    }

    #[test]
    fn test_hash_algorithm_serialization() {
        assert_eq!(HashAlgorithm::Sha3_256, HashAlgorithm::Sha3_256);
        assert_ne!(HashAlgorithm::Sha3_256, HashAlgorithm::Blake3);
    }

    #[test]
    fn test_isolation_level_enum() {
        let basic = IsolationLevel::Basic;
        let full = IsolationLevel::Full;
        let enhanced = IsolationLevel::Enhanced;
        
        assert_ne!(basic, full);
        assert_ne!(full, enhanced);
        assert_ne!(basic, enhanced);
    }

    #[test]
    fn test_container_runtime_enum() {
        let native = ContainerRuntime::Native;
        let docker = ContainerRuntime::Docker;
        
        assert_ne!(native, docker);
        assert_eq!(docker, ContainerRuntime::Docker);
    }
}