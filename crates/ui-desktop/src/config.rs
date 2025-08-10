//! Application configuration management for Bio P2P Desktop
//! 
//! This module handles loading, saving, and validation of application
//! configuration including network settings, UI preferences, and biological parameters.

use anyhow::{Context, Result};
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::fs;
use tracing::{info, warn, error};

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// Application metadata
    pub app: AppInfo,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// UI configuration
    pub ui: UIConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Resource management configuration
    pub resource: ResourceConfig,
    
    /// Biological system configuration
    pub biological: BiologicalConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
}

/// Application information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppInfo {
    pub name: String,
    pub version: String,
    pub build_date: String,
    pub git_commit: Option<String>,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// P2P networking settings
    pub listen_addresses: Vec<String>,
    pub bootstrap_nodes: Vec<String>,
    pub enable_mdns: bool,
    pub enable_upnp: bool,
    pub enable_relay: bool,
    pub enable_dcutr: bool,
    
    /// Connection settings
    pub connection_timeout_ms: u64,
    pub keepalive_interval_ms: u64,
    pub max_connections: usize,
    pub max_pending_connections: usize,
    
    /// Biological networking parameters
    pub young_node_learning_radius: usize,
    pub trust_decay_rate: f64,
    pub reputation_threshold: f64,
    pub adaptation_sensitivity: f64,
    
    /// Bandwidth and resource limits
    pub max_bandwidth_mbps: Option<f64>,
    pub max_storage_mb: Option<u64>,
}

/// User interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIConfig {
    /// Window settings
    pub window_width: f32,
    pub window_height: f32,
    pub window_maximized: bool,
    pub start_minimized: bool,
    pub minimize_to_tray: bool,
    
    /// Theme and appearance
    pub theme_name: String,
    pub biological_metaphor_level: String,
    pub enable_animations: bool,
    pub enable_sound_effects: bool,
    
    /// Panel layout
    pub default_visible_panels: Vec<String>,
    pub panel_layout: PanelLayoutConfig,
    
    /// Chart and visualization
    pub chart_update_interval_ms: u64,
    pub chart_history_length: usize,
    pub enable_real_time_updates: bool,
    
    /// Education and learning
    pub show_biological_tooltips: bool,
    pub enable_guided_tours: bool,
    pub learning_pace: String,
    pub track_learning_progress: bool,
}

/// Panel layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelLayoutConfig {
    pub layout_name: String,
    pub panel_positions: HashMap<String, PanelLayoutPosition>,
    pub splitter_positions: HashMap<String, f32>,
    pub auto_hide_panels: bool,
}

/// Panel layout position
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelLayoutPosition {
    pub area: String, // "left", "right", "top", "bottom", "center", "floating"
    pub size: f32,
    pub order: usize,
    pub floating_position: Option<(f32, f32)>,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Multi-layer security settings
    pub enable_all_layers: bool,
    pub enable_clean_before_after_usage: bool,
    pub enable_illusion_layer: bool,
    pub enable_behavior_monitoring: bool,
    pub enable_thermal_detection: bool,
    
    /// Threat detection settings
    pub threat_detection_sensitivity: f64,
    pub auto_quarantine_threats: bool,
    pub enable_immune_response: bool,
    pub false_positive_tolerance: f64,
    
    /// Biological security parameters
    pub immune_system_aggressiveness: f64,
    pub pack_investigation_threshold: f64,
    pub dos_detection_window_ms: u64,
    
    /// Encryption and cryptography
    pub enable_end_to_end_encryption: bool,
    pub key_rotation_interval_hours: u64,
    pub require_peer_authentication: bool,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Resource limits
    pub max_cpu_usage_percent: f64,
    pub max_memory_usage_mb: u64,
    pub max_disk_usage_mb: u64,
    pub max_network_bandwidth_mbps: f64,
    
    /// Biological resource management
    pub enable_caste_specialization: bool,
    pub enable_havoc_responses: bool,
    pub thermal_monitoring_enabled: bool,
    pub resource_sharing_enabled: bool,
    
    /// Compartmentalization settings
    pub training_compartment_size_percent: f64,
    pub inference_compartment_size_percent: f64,
    pub storage_compartment_size_percent: f64,
    pub communication_compartment_size_percent: f64,
    pub security_compartment_size_percent: f64,
    
    /// Adaptation parameters
    pub adaptation_aggressiveness: f64,
    pub resource_reallocation_threshold: f64,
    pub emergency_response_threshold: f64,
}

/// Biological system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalConfig {
    /// Auto-adaptation settings
    pub enable_auto_adaptation: bool,
    pub adaptation_frequency_minutes: u64,
    pub learning_rate: f64,
    pub exploration_vs_exploitation_ratio: f64,
    
    /// Role activation settings
    pub auto_activate_roles: bool,
    pub role_activation_threshold: f64,
    pub max_concurrent_roles: usize,
    pub role_transition_cooldown_minutes: u64,
    
    /// Biological behavior parameters
    pub young_node_learning_enabled: bool,
    pub swarm_intelligence_enabled: bool,
    pub social_learning_enabled: bool,
    pub immune_system_learning_enabled: bool,
    
    /// Specific role configurations
    pub role_configs: HashMap<String, RoleConfig>,
    
    /// Evolution and adaptation
    pub evolution_pressure: f64,
    pub mutation_rate: f64,
    pub selection_pressure: f64,
}

/// Individual role configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleConfig {
    pub enabled: bool,
    pub auto_activate: bool,
    pub performance_threshold: f64,
    pub energy_efficiency_weight: f64,
    pub specialization_level: f64,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Global logging settings
    pub global_log_level: String,
    pub enable_console_logging: bool,
    pub enable_file_logging: bool,
    pub colored_output: bool,
    
    /// Module-specific log levels
    pub module_log_levels: HashMap<String, String>,
    
    /// File logging settings
    pub log_file_max_size_mb: u64,
    pub log_file_max_count: usize,
    pub log_rotation: String, // "daily", "weekly", "monthly", "size"
    
    /// Specialized logging
    pub enable_biological_event_logging: bool,
    pub enable_security_event_logging: bool,
    pub enable_performance_logging: bool,
    pub enable_network_event_logging: bool,
    
    /// Log filtering
    pub filter_sensitive_data: bool,
    pub include_biological_context: bool,
    pub log_format: String, // "json", "text", "structured"
}

impl AppConfig {
    /// Load configuration from file or create default
    pub async fn load_or_default() -> Result<Self> {
        match Self::load().await {
            Ok(config) => {
                info!("Loaded configuration from file");
                Ok(config)
            }
            Err(e) => {
                warn!("Failed to load configuration: {}, using defaults", e);
                let config = Self::default();
                // Try to save default configuration
                if let Err(save_err) = config.save().await {
                    warn!("Failed to save default configuration: {}", save_err);
                }
                Ok(config)
            }
        }
    }

    /// Load configuration from file
    pub async fn load() -> Result<Self> {
        let config_path = Self::get_config_file_path()?;
        
        if !config_path.exists() {
            return Err(anyhow::anyhow!("Configuration file does not exist: {:?}", config_path));
        }

        let config_data = fs::read_to_string(&config_path).await
            .with_context(|| format!("Failed to read config file: {:?}", config_path))?;

        let config: AppConfig = toml::from_str(&config_data)
            .with_context(|| format!("Failed to parse config file: {:?}", config_path))?;

        config.validate()?;

        info!("Configuration loaded from: {:?}", config_path);
        Ok(config)
    }

    /// Save configuration to file
    pub async fn save(&self) -> Result<()> {
        let config_path = Self::get_config_file_path()?;
        
        // Ensure directory exists
        if let Some(parent) = config_path.parent() {
            fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create config directory: {:?}", parent))?;
        }

        let config_data = toml::to_string_pretty(self)
            .context("Failed to serialize configuration")?;

        fs::write(&config_path, config_data).await
            .with_context(|| format!("Failed to write config file: {:?}", config_path))?;

        info!("Configuration saved to: {:?}", config_path);
        Ok(())
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate network configuration
        if self.network.listen_addresses.is_empty() {
            return Err(anyhow::anyhow!("At least one listen address must be specified"));
        }

        if self.network.max_connections == 0 {
            return Err(anyhow::anyhow!("Max connections must be greater than 0"));
        }

        if self.network.young_node_learning_radius == 0 {
            return Err(anyhow::anyhow!("Young node learning radius must be greater than 0"));
        }

        // Validate UI configuration
        if self.ui.window_width <= 0.0 || self.ui.window_height <= 0.0 {
            return Err(anyhow::anyhow!("Window dimensions must be positive"));
        }

        if self.ui.chart_update_interval_ms == 0 {
            return Err(anyhow::anyhow!("Chart update interval must be greater than 0"));
        }

        // Validate security configuration
        if self.security.threat_detection_sensitivity < 0.0 || self.security.threat_detection_sensitivity > 1.0 {
            return Err(anyhow::anyhow!("Threat detection sensitivity must be between 0.0 and 1.0"));
        }

        // Validate resource configuration
        if self.resource.max_cpu_usage_percent <= 0.0 || self.resource.max_cpu_usage_percent > 100.0 {
            return Err(anyhow::anyhow!("CPU usage limit must be between 0.1 and 100.0"));
        }

        let total_compartment_size = self.resource.training_compartment_size_percent +
            self.resource.inference_compartment_size_percent +
            self.resource.storage_compartment_size_percent +
            self.resource.communication_compartment_size_percent +
            self.resource.security_compartment_size_percent;

        if (total_compartment_size - 100.0).abs() > 0.01 {
            return Err(anyhow::anyhow!("Compartment sizes must total 100%, got {:.2}%", total_compartment_size));
        }

        // Validate biological configuration
        if self.biological.learning_rate < 0.0 || self.biological.learning_rate > 1.0 {
            return Err(anyhow::anyhow!("Learning rate must be between 0.0 and 1.0"));
        }

        if self.biological.max_concurrent_roles == 0 {
            return Err(anyhow::anyhow!("Max concurrent roles must be greater than 0"));
        }

        // Validate logging configuration
        if !["trace", "debug", "info", "warn", "error"].contains(&self.logging.global_log_level.as_str()) {
            return Err(anyhow::anyhow!("Invalid log level: {}", self.logging.global_log_level));
        }

        if self.logging.log_file_max_size_mb == 0 {
            return Err(anyhow::anyhow!("Log file max size must be greater than 0"));
        }

        info!("Configuration validation completed successfully");
        Ok(())
    }

    /// Get configuration file path
    pub fn get_config_file_path() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("org", "bio-p2p", "bio-p2p-desktop")
            .context("Failed to determine project directories")?;

        let config_dir = project_dirs.config_dir();
        Ok(config_dir.join("config.toml"))
    }

    /// Get application data directory
    pub fn get_data_dir() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("org", "bio-p2p", "bio-p2p-desktop")
            .context("Failed to determine project directories")?;

        Ok(project_dirs.data_dir().to_path_buf())
    }

    /// Get application cache directory
    pub fn get_cache_dir() -> Result<PathBuf> {
        let project_dirs = ProjectDirs::from("org", "bio-p2p", "bio-p2p-desktop")
            .context("Failed to determine project directories")?;

        Ok(project_dirs.cache_dir().to_path_buf())
    }

    /// Update configuration value
    pub fn update_value(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        match key {
            "ui.theme_name" => {
                if let Some(theme) = value.as_str() {
                    self.ui.theme_name = theme.to_string();
                }
            }
            "ui.biological_metaphor_level" => {
                if let Some(level) = value.as_str() {
                    self.ui.biological_metaphor_level = level.to_string();
                }
            }
            "network.enable_mdns" => {
                if let Some(enable) = value.as_bool() {
                    self.network.enable_mdns = enable;
                }
            }
            "security.enable_all_layers" => {
                if let Some(enable) = value.as_bool() {
                    self.security.enable_all_layers = enable;
                }
            }
            "biological.enable_auto_adaptation" => {
                if let Some(enable) = value.as_bool() {
                    self.biological.enable_auto_adaptation = enable;
                }
            }
            _ => {
                return Err(anyhow::anyhow!("Unknown configuration key: {}", key));
            }
        }
        
        Ok(())
    }

    /// Get configuration value
    pub fn get_value(&self, key: &str) -> Result<serde_json::Value> {
        match key {
            "ui.theme_name" => Ok(serde_json::Value::String(self.ui.theme_name.clone())),
            "ui.biological_metaphor_level" => Ok(serde_json::Value::String(self.ui.biological_metaphor_level.clone())),
            "network.enable_mdns" => Ok(serde_json::Value::Bool(self.network.enable_mdns)),
            "security.enable_all_layers" => Ok(serde_json::Value::Bool(self.security.enable_all_layers)),
            "biological.enable_auto_adaptation" => Ok(serde_json::Value::Bool(self.biological.enable_auto_adaptation)),
            _ => Err(anyhow::anyhow!("Unknown configuration key: {}", key)),
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            app: AppInfo {
                name: "Bio P2P Desktop".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                build_date: env!("VERGEN_BUILD_DATE").unwrap_or("unknown").to_string(),
                git_commit: option_env!("VERGEN_GIT_SHA").map(|s| s.to_string()),
            },
            network: NetworkConfig::default(),
            ui: UIConfig::default(),
            security: SecurityConfig::default(),
            resource: ResourceConfig::default(),
            biological: BiologicalConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addresses: vec![
                "/ip4/0.0.0.0/tcp/0".to_string(),
                "/ip6/::/tcp/0".to_string(),
            ],
            bootstrap_nodes: vec![
                "/dnsaddr/bootstrap.bio-p2p.org".to_string(),
            ],
            enable_mdns: true,
            enable_upnp: true,
            enable_relay: true,
            enable_dcutr: true,
            connection_timeout_ms: 30000,
            keepalive_interval_ms: 60000,
            max_connections: 100,
            max_pending_connections: 25,
            young_node_learning_radius: 100,
            trust_decay_rate: 0.01,
            reputation_threshold: 0.5,
            adaptation_sensitivity: 0.3,
            max_bandwidth_mbps: None,
            max_storage_mb: Some(10000),
        }
    }
}

impl Default for UIConfig {
    fn default() -> Self {
        Self {
            window_width: 1200.0,
            window_height: 800.0,
            window_maximized: false,
            start_minimized: false,
            minimize_to_tray: true,
            theme_name: "biological_forest".to_string(),
            biological_metaphor_level: "intermediate".to_string(),
            enable_animations: true,
            enable_sound_effects: false,
            default_visible_panels: vec![
                "network".to_string(),
                "biological".to_string(),
                "resources".to_string(),
            ],
            panel_layout: PanelLayoutConfig::default(),
            chart_update_interval_ms: 5000,
            chart_history_length: 1000,
            enable_real_time_updates: true,
            show_biological_tooltips: true,
            enable_guided_tours: true,
            learning_pace: "normal".to_string(),
            track_learning_progress: true,
        }
    }
}

impl Default for PanelLayoutConfig {
    fn default() -> Self {
        let mut panel_positions = HashMap::new();
        panel_positions.insert("navigation".to_string(), PanelLayoutPosition {
            area: "left".to_string(),
            size: 250.0,
            order: 0,
            floating_position: None,
        });
        panel_positions.insert("main_content".to_string(), PanelLayoutPosition {
            area: "center".to_string(),
            size: 0.0, // Auto-size
            order: 0,
            floating_position: None,
        });

        Self {
            layout_name: "default".to_string(),
            panel_positions,
            splitter_positions: HashMap::new(),
            auto_hide_panels: false,
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_all_layers: true,
            enable_clean_before_after_usage: true,
            enable_illusion_layer: true,
            enable_behavior_monitoring: true,
            enable_thermal_detection: true,
            threat_detection_sensitivity: 0.7,
            auto_quarantine_threats: true,
            enable_immune_response: true,
            false_positive_tolerance: 0.05,
            immune_system_aggressiveness: 0.6,
            pack_investigation_threshold: 0.8,
            dos_detection_window_ms: 60000,
            enable_end_to_end_encryption: true,
            key_rotation_interval_hours: 24,
            require_peer_authentication: true,
        }
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_cpu_usage_percent: 80.0,
            max_memory_usage_mb: 4096,
            max_disk_usage_mb: 10240,
            max_network_bandwidth_mbps: 100.0,
            enable_caste_specialization: true,
            enable_havoc_responses: true,
            thermal_monitoring_enabled: true,
            resource_sharing_enabled: true,
            training_compartment_size_percent: 30.0,
            inference_compartment_size_percent: 25.0,
            storage_compartment_size_percent: 20.0,
            communication_compartment_size_percent: 15.0,
            security_compartment_size_percent: 10.0,
            adaptation_aggressiveness: 0.5,
            resource_reallocation_threshold: 0.8,
            emergency_response_threshold: 0.9,
        }
    }
}

impl Default for BiologicalConfig {
    fn default() -> Self {
        let mut role_configs = HashMap::new();
        
        // Default role configurations
        role_configs.insert("YoungNode".to_string(), RoleConfig {
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.6,
            energy_efficiency_weight: 0.8,
            specialization_level: 0.7,
            parameters: HashMap::new(),
        });

        role_configs.insert("TrustNode".to_string(), RoleConfig {
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.7,
            energy_efficiency_weight: 0.6,
            specialization_level: 0.8,
            parameters: HashMap::new(),
        });

        Self {
            enable_auto_adaptation: true,
            adaptation_frequency_minutes: 30,
            learning_rate: 0.1,
            exploration_vs_exploitation_ratio: 0.3,
            auto_activate_roles: true,
            role_activation_threshold: 0.6,
            max_concurrent_roles: 10,
            role_transition_cooldown_minutes: 5,
            young_node_learning_enabled: true,
            swarm_intelligence_enabled: true,
            social_learning_enabled: true,
            immune_system_learning_enabled: true,
            role_configs,
            evolution_pressure: 0.1,
            mutation_rate: 0.05,
            selection_pressure: 0.3,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        let mut module_log_levels = HashMap::new();
        module_log_levels.insert("bio_p2p_core".to_string(), "info".to_string());
        module_log_levels.insert("bio_p2p_p2p".to_string(), "info".to_string());
        module_log_levels.insert("bio_p2p_security".to_string(), "warn".to_string());
        module_log_levels.insert("bio_p2p_resource".to_string(), "info".to_string());
        module_log_levels.insert("libp2p".to_string(), "warn".to_string());

        Self {
            global_log_level: "info".to_string(),
            enable_console_logging: true,
            enable_file_logging: true,
            colored_output: true,
            module_log_levels,
            log_file_max_size_mb: 100,
            log_file_max_count: 5,
            log_rotation: "daily".to_string(),
            enable_biological_event_logging: true,
            enable_security_event_logging: true,
            enable_performance_logging: true,
            enable_network_event_logging: true,
            filter_sensitive_data: true,
            include_biological_context: true,
            log_format: "structured".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_default_config_validation() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[tokio::test]
    async fn test_config_serialization() {
        let config = AppConfig::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: AppConfig = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.app.name, deserialized.app.name);
        assert_eq!(config.network.max_connections, deserialized.network.max_connections);
        assert_eq!(config.ui.window_width, deserialized.ui.window_width);
    }

    #[test]
    fn test_config_value_update() {
        let mut config = AppConfig::default();
        
        let result = config.update_value("ui.theme_name", serde_json::Value::String("dark".to_string()));
        assert!(result.is_ok());
        assert_eq!(config.ui.theme_name, "dark");
        
        let result = config.update_value("network.enable_mdns", serde_json::Value::Bool(false));
        assert!(result.is_ok());
        assert_eq!(config.network.enable_mdns, false);
    }

    #[test]
    fn test_config_value_get() {
        let config = AppConfig::default();
        
        let value = config.get_value("ui.theme_name").unwrap();
        assert_eq!(value, serde_json::Value::String("biological_forest".to_string()));
        
        let value = config.get_value("network.enable_mdns").unwrap();
        assert_eq!(value, serde_json::Value::Bool(true));
    }

    #[test]
    fn test_invalid_config_validation() {
        let mut config = AppConfig::default();
        
        // Test invalid CPU usage
        config.resource.max_cpu_usage_percent = 150.0;
        assert!(config.validate().is_err());
        
        // Reset and test invalid compartment sizes
        config.resource.max_cpu_usage_percent = 80.0;
        config.resource.training_compartment_size_percent = 60.0;
        assert!(config.validate().is_err());
    }
}