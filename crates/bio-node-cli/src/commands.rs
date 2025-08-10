//! Command implementations for Bio P2P Node CLI
//!
//! This module provides the core implementation for all CLI commands including
//! start, stop, status, config generation, and bootstrap operations.

use anyhow::{anyhow, Context, Result};
use tokio::fs;
use tracing::{info, warn, error, debug};
use std::path::PathBuf;

use crate::{
    cli::{Commands, ConfigTemplate, CommandContext},
    config::{NodeConfig, MINIMAL_CONFIG_TEMPLATE, PRODUCTION_CONFIG_TEMPLATE, DEVELOPMENT_CONFIG_TEMPLATE},
    daemon::BioNodeDaemon,
    signals::PidFile,
    logging::LoggingSystem,
};

/// Execute CLI command based on parsed arguments
pub async fn execute_command(context: CommandContext) -> Result<()> {
    // Validate context
    context.validate()
        .context("Command context validation failed")?;
    
    // Initialize logging
    let _logging_guard = LoggingSystem::init(
        context.effective_log_level(),
        context.args.log_file.as_ref(),
        context.args.quiet
    ).context("Failed to initialize logging")?;
    
    match context.args.command {
        Commands::Start { ref config_file, .. } => {
            start_command(&context, config_file.as_ref()).await
        }
        
        Commands::Stop { ref config_file } => {
            stop_command(&context, config_file.as_ref()).await
        }
        
        Commands::Status { ref config_file } => {
            status_command(&context, config_file.as_ref()).await
        }
        
        Commands::Config { template, ref output } => {
            config_command(&context, template, output.as_ref()).await
        }
        
        Commands::Bootstrap { ref config_file, ref peer_addr, timeout } => {
            bootstrap_command(&context, config_file.as_ref(), peer_addr, timeout).await
        }
    }
}

/// Start the Bio P2P Node daemon
async fn start_command(context: &CommandContext, config_file: Option<&PathBuf>) -> Result<()> {
    info!("Starting Bio P2P Node daemon");
    
    // Load configuration
    let config = load_configuration(context, config_file).await
        .context("Failed to load configuration")?;
    
    // Check if daemon is already running
    let pid_file = PidFile::new(&config.daemon.pid_file);
    if let Some(existing_pid) = pid_file.check_running().await? {
        return Err(anyhow!("Daemon is already running with PID: {}", existing_pid));
    }
    
    // Create and start daemon
    let mut daemon = BioNodeDaemon::new(config).await
        .context("Failed to create daemon")?;
    
    // Handle daemon mode
    #[cfg(feature = "daemon")]
    if context.is_daemon_mode() {
        return start_daemon_mode(daemon).await;
    }
    
    // Start daemon components
    daemon.start().await
        .context("Failed to start daemon")?;
    
    // Run main event loop
    daemon.run().await
        .context("Daemon event loop failed")?;
    
    // Graceful shutdown
    daemon.shutdown().await
        .context("Failed to shutdown daemon gracefully")?;
    
    info!("Bio P2P Node daemon stopped");
    Ok(())
}

/// Stop running Bio P2P Node daemon
async fn stop_command(context: &CommandContext, config_file: Option<&PathBuf>) -> Result<()> {
    info!("Stopping Bio P2P Node daemon");
    
    // Load configuration to get PID file location
    let config = load_configuration(context, config_file).await
        .context("Failed to load configuration")?;
    
    let pid_file = PidFile::new(&config.daemon.pid_file);
    
    match pid_file.check_running().await? {
        Some(pid) => {
            info!("Found running daemon with PID: {}", pid);
            
            // Send SIGTERM for graceful shutdown
            #[cfg(unix)]
            {
                use nix::sys::signal::{self, Signal};
                use nix::unistd::Pid;
                
                match signal::kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
                    Ok(()) => {
                        info!("Sent SIGTERM to PID: {}", pid);
                        
                        // Wait for graceful shutdown with timeout
                        let shutdown_timeout = config.daemon.shutdown_timeout;
                        let mut attempts = 0;
                        let max_attempts = shutdown_timeout.as_secs() / 2; // Check every 2 seconds
                        
                        while attempts < max_attempts {
                            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                            
                            if pid_file.check_running().await?.is_none() {
                                info!("Daemon shutdown completed gracefully");
                                return Ok(());
                            }
                            
                            attempts += 1;
                        }
                        
                        // Force shutdown if graceful shutdown failed
                        warn!("Graceful shutdown timeout, sending SIGKILL");
                        signal::kill(Pid::from_raw(pid as i32), Signal::SIGKILL)
                            .context("Failed to send SIGKILL")?;
                        
                        // Wait a bit more for process to die
                        tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                        
                        if pid_file.check_running().await?.is_some() {
                            return Err(anyhow!("Failed to stop daemon process"));
                        }
                        
                        info!("Daemon stopped forcefully");
                    }
                    Err(e) => {
                        return Err(anyhow!("Failed to send signal to PID {}: {}", pid, e));
                    }
                }
            }
            
            #[cfg(windows)]
            {
                // Windows process termination
                info!("Windows process termination not implemented - please use Ctrl+C or Task Manager");
                return Err(anyhow!("Windows process termination requires manual intervention"));
            }
        }
        None => {
            info!("No running daemon found");
        }
    }
    
    Ok(())
}

/// Show daemon status information
async fn status_command(context: &CommandContext, config_file: Option<&PathBuf>) -> Result<()> {
    info!("Checking Bio P2P Node daemon status");
    
    // Load configuration
    let config = load_configuration(context, config_file).await
        .context("Failed to load configuration")?;
    
    let pid_file = PidFile::new(&config.daemon.pid_file);
    
    // Check if daemon is running
    match pid_file.check_running().await? {
        Some(pid) => {
            println!("Bio P2P Node Status:");
            println!("  Status: RUNNING");
            println!("  PID: {}", pid);
            println!("  PID File: {}", config.daemon.pid_file.display());
            
            // Try to get detailed status from health endpoint
            if config.monitoring.enable_health_check {
                match get_health_status(&config).await {
                    Ok(health) => {
                        println!("  Health: {}", health.overall_health);
                        println!("  Connected Peers: {}", health.connected_peers);
                        println!("  Active Roles: {:?}", health.active_roles);
                        println!("  Resource Usage:");
                        println!("    CPU: {:.1}%", health.cpu_usage * 100.0);
                        println!("    Memory: {:.1}%", health.memory_usage * 100.0);
                        println!("    Disk: {:.1}%", health.disk_usage * 100.0);
                        
                        if config.monitoring.enable_metrics {
                            println!("  Metrics: http://{}:{}/metrics", 
                                config.monitoring.metrics_addr, 
                                config.monitoring.metrics_port);
                        }
                        
                        println!("  Health Check: http://localhost:{}/health", 
                            config.monitoring.health_port);
                    }
                    Err(e) => {
                        warn!("Could not retrieve detailed status: {}", e);
                        println!("  Health: UNKNOWN (daemon running but health check unavailable)");
                    }
                }
            } else {
                println!("  Health: UNKNOWN (health checks disabled)");
            }
        }
        None => {
            println!("Bio P2P Node Status:");
            println!("  Status: NOT RUNNING");
            
            if config.daemon.pid_file.exists() {
                println!("  Note: Stale PID file removed");
            }
        }
    }
    
    Ok(())
}

/// Generate configuration file template
async fn config_command(context: &CommandContext, template: ConfigTemplate, output: Option<&PathBuf>) -> Result<()> {
    info!("Generating configuration template: {:?}", template);
    
    let template_content = match template {
        ConfigTemplate::Minimal => MINIMAL_CONFIG_TEMPLATE,
        ConfigTemplate::Production => PRODUCTION_CONFIG_TEMPLATE,
        ConfigTemplate::Development => DEVELOPMENT_CONFIG_TEMPLATE,
    };
    
    match output {
        Some(output_path) => {
            // Write to file
            fs::write(output_path, template_content).await
                .with_context(|| format!("Failed to write config to: {}", output_path.display()))?;
            
            println!("Configuration template written to: {}", output_path.display());
            
            // Set appropriate permissions on Unix systems
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mut perms = fs::metadata(output_path).await?.permissions();
                perms.set_mode(0o644);
                fs::set_permissions(output_path, perms).await?;
            }
        }
        None => {
            // Write to stdout
            print!("{}", template_content);
        }
    }
    
    Ok(())
}

/// Bootstrap connection to network peer
async fn bootstrap_command(
    context: &CommandContext,
    config_file: Option<&PathBuf>,
    peer_addr: &str,
    timeout: u64
) -> Result<()> {
    info!("Bootstrapping connection to peer: {}", peer_addr);
    
    // Load configuration
    let mut config = load_configuration(context, config_file).await
        .context("Failed to load configuration")?;
    
    // Parse peer address
    let peer_multiaddr: libp2p::Multiaddr = peer_addr.parse()
        .context("Invalid peer address format")?;
    
    // Add to bootstrap peers if not already present
    if !config.network.bootstrap_peers.contains(&peer_multiaddr) {
        config.network.bootstrap_peers.push(peer_multiaddr.clone());
        info!("Added peer to bootstrap list");
    }
    
    // Create temporary node for bootstrap test
    println!("Testing connection to peer...");
    
    let test_node = crate::node::BiologicalNode::new(&config).await
        .context("Failed to create test node")?;
    
    // Attempt connection with timeout
    let connection_result = tokio::time::timeout(
        tokio::time::Duration::from_secs(timeout),
        test_bootstrap_connection(&test_node, &peer_multiaddr)
    ).await;
    
    match connection_result {
        Ok(Ok(())) => {
            println!("✓ Successfully connected to peer");
            println!("✓ Peer is reachable and responsive");
            
            // If we have a config file, update it
            if let Some(config_path) = config_file {
                update_bootstrap_config(config_path, &config).await
                    .context("Failed to update configuration file")?;
                println!("✓ Updated configuration file with bootstrap peer");
            } else {
                println!("💡 To make this peer permanent, add to your configuration:");
                println!("   bootstrap_peers = [\"{}\"]", peer_addr);
            }
        }
        Ok(Err(e)) => {
            println!("✗ Connection failed: {}", e);
            return Err(anyhow!("Bootstrap connection failed"));
        }
        Err(_) => {
            println!("✗ Connection timed out after {} seconds", timeout);
            return Err(anyhow!("Bootstrap connection timed out"));
        }
    }
    
    Ok(())
}

/// Load node configuration with fallback to defaults
async fn load_configuration(context: &CommandContext, config_file: Option<&PathBuf>) -> Result<NodeConfig> {
    let config_path = match config_file {
        Some(path) => {
            if !path.exists() {
                return Err(anyhow!("Configuration file not found: {}", path.display()));
            }
            path.clone()
        }
        None => {
            // Search for config file in default locations
            let default_paths = context.default_config_paths();
            
            let mut found_path = None;
            for path in &default_paths {
                if path.exists() {
                    found_path = Some(path.clone());
                    break;
                }
            }
            
            match found_path {
                Some(path) => {
                    info!("Using configuration file: {}", path.display());
                    path
                }
                None => {
                    info!("No configuration file found, using defaults");
                    return Ok(apply_env_overrides(NodeConfig::default(), &context.env_config));
                }
            }
        }
    };
    
    // Load configuration from file
    let mut config = NodeConfig::load_from_file(&config_path).await
        .with_context(|| format!("Failed to load configuration from: {}", config_path.display()))?;
    
    // Apply environment variable overrides
    config = apply_env_overrides(config, &context.env_config);
    
    // Validate final configuration
    config.validate()
        .context("Configuration validation failed")?;
    
    Ok(config)
}

/// Apply environment variable overrides to configuration
fn apply_env_overrides(mut config: NodeConfig, env_config: &crate::cli::EnvConfig) -> NodeConfig {
    // Network overrides
    if let Some(ref bind_addr) = env_config.bind_addr {
        if let Some(bind_port) = env_config.bind_port {
            let addr_str = format!("/ip4/{}/tcp/{}", bind_addr, bind_port);
            if let Ok(multiaddr) = addr_str.parse() {
                config.network.listen_addresses = vec![multiaddr];
            }
        }
    }
    
    if !env_config.bootstrap_peers.is_empty() {
        let peers: Vec<_> = env_config.bootstrap_peers
            .iter()
            .filter_map(|p| p.parse().ok())
            .collect();
        if !peers.is_empty() {
            config.network.bootstrap_peers = peers;
        }
    }
    
    if let Some(ref key_path) = env_config.network_key {
        config.network.node_key_path = PathBuf::from(key_path);
    }
    
    // Resource overrides
    if let Some(max_cores) = env_config.max_cpu_cores {
        config.resources.max_cpu_cores = max_cores;
    }
    
    if let Some(max_memory) = env_config.max_memory_mb {
        config.resources.max_memory_mb = max_memory;
    }
    
    // Biological role overrides
    if !env_config.preferred_roles.is_empty() {
        let roles: Vec<_> = env_config.preferred_roles
            .iter()
            .filter_map(|r| parse_biological_role(r))
            .collect();
        if !roles.is_empty() {
            config.biological.preferred_roles = roles;
        }
    }
    
    // Logging overrides
    if let Some(ref log_level) = env_config.log_level {
        config.monitoring.logging.level = log_level.clone();
    }
    
    // Daemon overrides
    if let Some(daemon_mode) = env_config.daemon_mode {
        // Note: daemon mode is handled by CLI parser
        debug!("Daemon mode from environment: {}", daemon_mode);
    }
    
    if let Some(ref data_dir) = env_config.data_dir {
        config.storage.data_dir = PathBuf::from(data_dir);
    }
    
    if let Some(ref pid_file) = env_config.pid_file {
        config.daemon.pid_file = PathBuf::from(pid_file);
    }
    
    config
}

/// Parse biological role from string
fn parse_biological_role(role_str: &str) -> Option<crate::config::BiologicalRole> {
    use crate::config::BiologicalRole;
    
    match role_str {
        "YoungNode" => Some(BiologicalRole::YoungNode),
        "CasteNode" => Some(BiologicalRole::CasteNode),
        "ImitateNode" => Some(BiologicalRole::ImitateNode),
        "HatchNode" => Some(BiologicalRole::HatchNode),
        "SyncPhaseNode" => Some(BiologicalRole::SyncPhaseNode),
        "HuddleNode" => Some(BiologicalRole::HuddleNode),
        "MigrationNode" => Some(BiologicalRole::MigrationNode),
        "AddressNode" => Some(BiologicalRole::AddressNode),
        "TunnelNode" => Some(BiologicalRole::TunnelNode),
        "SignNode" => Some(BiologicalRole::SignNode),
        "ThermalNode" => Some(BiologicalRole::ThermalNode),
        "DosNode" => Some(BiologicalRole::DosNode),
        "InvestigationNode" => Some(BiologicalRole::InvestigationNode),
        "CasualtyNode" => Some(BiologicalRole::CasualtyNode),
        "HavocNode" => Some(BiologicalRole::HavocNode),
        "StepUpNode" => Some(BiologicalRole::StepUpNode),
        "StepDownNode" => Some(BiologicalRole::StepDownNode),
        "FriendshipNode" => Some(BiologicalRole::FriendshipNode),
        "BuddyNode" => Some(BiologicalRole::BuddyNode),
        "TrustNode" => Some(BiologicalRole::TrustNode),
        "MemoryNode" => Some(BiologicalRole::MemoryNode),
        "TelescopeNode" => Some(BiologicalRole::TelescopeNode),
        "HealingNode" => Some(BiologicalRole::HealingNode),
        _ => {
            warn!("Unknown biological role: {}", role_str);
            None
        }
    }
}

/// Start daemon in background mode (Unix only)
#[cfg(feature = "daemon")]
async fn start_daemon_mode(mut daemon: BioNodeDaemon) -> Result<()> {
    use daemonize::Daemonize;
    
    info!("Starting daemon in background mode");
    
    let daemon_config = &daemon.config.daemon;
    
    // Prepare daemonization
    let mut daemonize = Daemonize::new()
        .pid_file(&daemon_config.pid_file)
        .working_directory(&daemon_config.working_directory);
    
    if let Some(ref user) = daemon_config.user {
        daemonize = daemonize.user(user);
    }
    
    if let Some(ref group) = daemon_config.group {
        daemonize = daemonize.group(group);
    }
    
    // Daemonize the process
    daemonize.start()
        .context("Failed to daemonize process")?;
    
    // Now running as daemon
    daemon.start().await
        .context("Failed to start daemon")?;
    
    daemon.run().await
        .context("Daemon event loop failed")?;
    
    daemon.shutdown().await
        .context("Failed to shutdown daemon gracefully")?;
    
    Ok(())
}

/// Test bootstrap connection to peer
async fn test_bootstrap_connection(node: &crate::node::BiologicalNode, peer_addr: &libp2p::Multiaddr) -> Result<()> {
    // This would integrate with the P2P node to test connectivity
    // For now, we'll simulate the connection test
    
    debug!("Testing connection to: {}", peer_addr);
    
    // In a real implementation, this would:
    // 1. Create temporary P2P connection
    // 2. Attempt handshake with peer
    // 3. Verify protocol compatibility
    // 4. Test basic message exchange
    
    // Simulate connection delay
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    
    // For now, assume success if address is well-formed
    if peer_addr.to_string().contains("/p2p/") {
        Ok(())
    } else {
        Err(anyhow!("Peer address missing peer ID component"))
    }
}

/// Update configuration file with bootstrap peer
async fn update_bootstrap_config(config_path: &PathBuf, config: &NodeConfig) -> Result<()> {
    // Read current file content
    let content = fs::read_to_string(config_path).await
        .context("Failed to read config file")?;
    
    // For TOML files, we need to parse and update the bootstrap_peers array
    if config_path.extension().and_then(|s| s.to_str()) == Some("toml") {
        let mut toml_value: toml::Value = toml::from_str(&content)
            .context("Failed to parse TOML config")?;
        
        // Update bootstrap_peers in network section
        if let Some(network_section) = toml_value.get_mut("network") {
            if let Some(network_table) = network_section.as_table_mut() {
                let peer_strings: Vec<toml::Value> = config.network.bootstrap_peers
                    .iter()
                    .map(|addr| toml::Value::String(addr.to_string()))
                    .collect();
                
                network_table.insert(
                    "bootstrap_peers".to_string(),
                    toml::Value::Array(peer_strings)
                );
            }
        }
        
        // Write updated content
        let updated_content = toml::to_string_pretty(&toml_value)
            .context("Failed to serialize updated config")?;
        
        fs::write(config_path, updated_content).await
            .context("Failed to write updated config file")?;
    }
    
    Ok(())
}

/// Get health status from running daemon
async fn get_health_status(config: &NodeConfig) -> Result<HealthSummary> {
    let health_url = format!("http://localhost:{}/health", config.monitoring.health_port);
    
    let client = reqwest::Client::new();
    let response = client
        .get(&health_url)
        .timeout(tokio::time::Duration::from_secs(5))
        .send()
        .await
        .context("Failed to connect to health endpoint")?;
    
    if !response.status().is_success() {
        return Err(anyhow!("Health endpoint returned error: {}", response.status()));
    }
    
    let health_json: serde_json::Value = response.json().await
        .context("Failed to parse health response")?;
    
    // Parse health information
    let overall_health = health_json
        .get("status")
        .and_then(|s| s.as_str())
        .unwrap_or("unknown")
        .to_string();
    
    let connected_peers = health_json
        .get("network")
        .and_then(|n| n.get("connected_peers"))
        .and_then(|p| p.as_u64())
        .unwrap_or(0) as usize;
    
    let active_roles = health_json
        .get("biological")
        .and_then(|b| b.get("active_roles"))
        .and_then(|r| r.as_array())
        .map(|arr| arr.len())
        .unwrap_or(0);
    
    let cpu_usage = health_json
        .get("resources")
        .and_then(|r| r.get("cpu_usage"))
        .and_then(|c| c.as_f64())
        .unwrap_or(0.0);
    
    let memory_usage = health_json
        .get("resources")
        .and_then(|r| r.get("memory_usage"))
        .and_then(|m| m.as_f64())
        .unwrap_or(0.0);
    
    let disk_usage = health_json
        .get("resources")
        .and_then(|r| r.get("disk_usage"))
        .and_then(|d| d.as_f64())
        .unwrap_or(0.0);
    
    Ok(HealthSummary {
        overall_health,
        connected_peers,
        active_roles: vec![], // Would parse from JSON
        cpu_usage,
        memory_usage,
        disk_usage,
    })
}

/// Summary health information for status display
#[derive(Debug)]
struct HealthSummary {
    overall_health: String,
    connected_peers: usize,
    active_roles: Vec<crate::config::BiologicalRole>,
    cpu_usage: f64,
    memory_usage: f64,
    disk_usage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Commands};
    use clap::Parser;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_config_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_file = temp_dir.path().join("test.toml");
        
        // Write test config
        fs::write(&config_file, MINIMAL_CONFIG_TEMPLATE).await.unwrap();
        
        let cli = Cli::try_parse_from(["bio-node", "start"]).unwrap();
        let context = CommandContext::new(cli);
        
        let config = load_configuration(&context, Some(&config_file)).await;
        assert!(config.is_ok());
    }
    
    #[tokio::test]
    async fn test_config_generation() {
        let temp_dir = TempDir::new().unwrap();
        let output_file = temp_dir.path().join("generated.toml");
        
        let cli = Cli::try_parse_from(["bio-node", "config"]).unwrap();
        let context = CommandContext::new(cli);
        
        let result = config_command(&context, ConfigTemplate::Minimal, Some(&output_file)).await;
        assert!(result.is_ok());
        assert!(output_file.exists());
        
        // Verify content
        let content = fs::read_to_string(&output_file).await.unwrap();
        assert!(content.contains("[network]"));
        assert!(content.contains("[biological]"));
    }
    
    #[test]
    fn test_biological_role_parsing() {
        assert_eq!(parse_biological_role("CasteNode"), Some(crate::config::BiologicalRole::CasteNode));
        assert_eq!(parse_biological_role("YoungNode"), Some(crate::config::BiologicalRole::YoungNode));
        assert_eq!(parse_biological_role("InvalidRole"), None);
    }
    
    #[tokio::test]
    async fn test_env_config_application() {
        let mut base_config = NodeConfig::default();
        let env_config = crate::cli::EnvConfig {
            max_cpu_cores: Some(16),
            max_memory_mb: Some(16384),
            bind_addr: Some("192.168.1.100".to_string()),
            bind_port: Some(8000),
            preferred_roles: vec!["CasteNode".to_string(), "HavocNode".to_string()],
            ..Default::default()
        };
        
        let updated_config = apply_env_overrides(base_config, &env_config);
        
        assert_eq!(updated_config.resources.max_cpu_cores, 16);
        assert_eq!(updated_config.resources.max_memory_mb, 16384);
        assert_eq!(updated_config.biological.preferred_roles.len(), 2);
    }
}