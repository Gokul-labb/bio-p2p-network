//! Tauri command handlers for Bio P2P Desktop Application
//! 
//! This module contains all the Tauri command functions that bridge
//! between the frontend UI and the embedded Rust backend node.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tauri::{AppHandle, Manager, State};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};

use crate::node::EmbeddedNode;
use crate::state::AppState;

// Command result types
type CommandResult<T> = Result<T, String>;

/// Network status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStatus {
    pub connected: bool,
    pub peer_count: usize,
    pub network_id: String,
    pub local_peer_id: String,
    pub listen_addresses: Vec<String>,
    pub uptime_seconds: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub connection_quality: String,
}

/// Biological roles information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalRoles {
    pub active_roles: Vec<BiologicalRole>,
    pub available_roles: Vec<String>,
    pub role_transitions: Vec<RoleTransition>,
    pub adaptation_status: String,
}

/// Individual biological role information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalRole {
    pub role_type: String,
    pub biological_inspiration: String,
    pub description: String,
    pub active: bool,
    pub performance_metrics: HashMap<String, f64>,
    pub specialization_level: f64,
    pub energy_efficiency: f64,
    pub adaptation_rate: f64,
}

/// Role transition event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTransition {
    pub from_role: String,
    pub to_role: String,
    pub timestamp: DateTime<Utc>,
    pub reason: String,
    pub success: bool,
}

/// Peer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub peer_id: String,
    pub multiaddrs: Vec<String>,
    pub connection_status: String,
    pub reputation_score: f64,
    pub trust_score: f64,
    pub performance_score: f64,
    pub biological_roles: Vec<String>,
    pub last_seen: DateTime<Utc>,
    pub latency_ms: u64,
    pub bandwidth_mbps: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub memory_available_mb: f64,
    pub disk_usage_mb: f64,
    pub network_upload_mbps: f64,
    pub network_download_mbps: f64,
    pub thermal_signatures: Vec<ThermalSignature>,
    pub compartment_usage: HashMap<String, CompartmentUsage>,
}

/// Thermal signature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermalSignature {
    pub component: String,
    pub temperature: f64,
    pub thermal_load: f64,
    pub efficiency_rating: f64,
}

/// Compartment resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentUsage {
    pub compartment_type: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_tasks: usize,
    pub efficiency: f64,
}

/// Security status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStatus {
    pub security_level: String,
    pub active_threats: Vec<SecurityThreat>,
    pub layer_status: Vec<SecurityLayerStatus>,
    pub immune_response_active: bool,
    pub threat_detection_rate: f64,
    pub false_positive_rate: f64,
    pub last_security_scan: DateTime<Utc>,
}

/// Security threat information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityThreat {
    pub threat_id: String,
    pub threat_type: String,
    pub severity: String,
    pub source: String,
    pub detected_at: DateTime<Utc>,
    pub mitigation_status: String,
    pub biological_response: String,
}

/// Security layer status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityLayerStatus {
    pub layer_name: String,
    pub layer_number: u8,
    pub status: String,
    pub effectiveness: f64,
    pub biological_inspiration: String,
}

/// Package processing queue information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageQueue {
    pub active_packages: Vec<PackageInfo>,
    pub queued_packages: Vec<PackageInfo>,
    pub completed_packages: Vec<PackageInfo>,
    pub failed_packages: Vec<PackageInfo>,
    pub processing_stats: PackageProcessingStats,
}

/// Individual package information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub package_id: String,
    pub package_type: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub processed_at: Option<DateTime<Utc>>,
    pub processing_time_ms: Option<u64>,
    pub size_bytes: u64,
    pub priority: u8,
    pub biological_processor: Option<String>,
    pub security_validated: bool,
}

/// Package processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageProcessingStats {
    pub total_processed: u64,
    pub success_rate: f64,
    pub average_processing_time_ms: f64,
    pub throughput_per_minute: f64,
    pub queue_length: usize,
}

/// Node log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeLog {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub module: String,
    pub message: String,
    pub biological_context: Option<String>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub uptime_seconds: u64,
    pub tasks_completed: u64,
    pub success_rate: f64,
    pub average_response_time_ms: f64,
    pub network_efficiency: f64,
    pub biological_adaptation_score: f64,
    pub energy_efficiency_rating: f64,
}

// Network Commands

#[tauri::command]
pub async fn get_network_status(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<NetworkStatus> {
    info!("Getting network status");
    
    let node = embedded_node.read().await;
    let status = node.get_network_status().await
        .map_err(|e| format!("Failed to get network status: {}", e))?;
    
    Ok(status)
}

#[tauri::command]
pub async fn add_peer(
    peer_address: String,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Adding peer: {}", peer_address);
    
    let mut node = embedded_node.write().await;
    let success = node.add_peer(peer_address).await
        .map_err(|e| format!("Failed to add peer: {}", e))?;
    
    Ok(success)
}

#[tauri::command]
pub async fn remove_peer(
    peer_id: String,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Removing peer: {}", peer_id);
    
    let mut node = embedded_node.write().await;
    let success = node.remove_peer(peer_id).await
        .map_err(|e| format!("Failed to remove peer: {}", e))?;
    
    Ok(success)
}

#[tauri::command]
pub async fn export_network_topology(
    format: String,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<String> {
    info!("Exporting network topology in format: {}", format);
    
    let node = embedded_node.read().await;
    let topology = node.export_topology(format).await
        .map_err(|e| format!("Failed to export topology: {}", e))?;
    
    Ok(topology)
}

// Biological Role Commands

#[tauri::command]
pub async fn get_biological_roles(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<BiologicalRoles> {
    info!("Getting biological roles");
    
    let node = embedded_node.read().await;
    let roles = node.get_biological_roles().await
        .map_err(|e| format!("Failed to get biological roles: {}", e))?;
    
    Ok(roles)
}

#[tauri::command]
pub async fn set_biological_role(
    role_type: String,
    enable: bool,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Setting biological role: {} -> {}", role_type, enable);
    
    let mut node = embedded_node.write().await;
    let success = node.set_biological_role(role_type, enable).await
        .map_err(|e| format!("Failed to set biological role: {}", e))?;
    
    Ok(success)
}

#[tauri::command]
pub async fn get_available_biological_roles() -> CommandResult<Vec<String>> {
    info!("Getting available biological roles");
    
    let roles = vec![
        "YoungNode".to_string(),
        "CasteNode".to_string(),
        "ImitateNode".to_string(),
        "HatchNode".to_string(),
        "SyncPhaseNode".to_string(),
        "HuddleNode".to_string(),
        "MigrationNode".to_string(),
        "AddressNode".to_string(),
        "TunnelNode".to_string(),
        "SignNode".to_string(),
        "DOSNode".to_string(),
        "InvestigationNode".to_string(),
        "CasualtyNode".to_string(),
        "HAVOCNode".to_string(),
        "StepUpNode".to_string(),
        "StepDownNode".to_string(),
        "ThermalNode".to_string(),
        "FriendshipNode".to_string(),
        "BuddyNode".to_string(),
        "TrustNode".to_string(),
    ];
    
    Ok(roles)
}

#[tauri::command]
pub async fn get_biological_role_info(role_type: String) -> CommandResult<HashMap<String, String>> {
    info!("Getting biological role info for: {}", role_type);
    
    let mut info = HashMap::new();
    
    match role_type.as_str() {
        "YoungNode" => {
            info.insert("biological_inspiration".to_string(), 
                "Young crows learn hunting techniques and territorial navigation by observing experienced adults".to_string());
            info.insert("description".to_string(), 
                "New nodes learn optimal routing paths from up to 100 neighboring experienced nodes".to_string());
            info.insert("key_features".to_string(), 
                "60-80% reduction in initialization overhead, 40-70% improvement in path discovery".to_string());
        }
        "CasteNode" => {
            info.insert("biological_inspiration".to_string(), 
                "Ant colonies achieve efficiency through specialized castes (workers, soldiers, nurses)".to_string());
            info.insert("description".to_string(), 
                "Compartmentalizes nodes into specialized functional units for different tasks".to_string());
            info.insert("key_features".to_string(), 
                "85-95% resource utilization, dynamic compartment scaling".to_string());
        }
        "HAVOCNode" => {
            info.insert("biological_inspiration".to_string(), 
                "Mosquitoes rapidly adapt behavior and resource allocation to environmental changes".to_string());
            info.insert("description".to_string(), 
                "Emergency resource reallocation preventing cascading network failures".to_string());
            info.insert("key_features".to_string(), 
                "Crisis management, rapid adaptation, network stability maintenance".to_string());
        }
        _ => {
            info.insert("biological_inspiration".to_string(), 
                "Various biological behaviors inspire computational efficiency".to_string());
            info.insert("description".to_string(), 
                "Specialized network behavior based on natural systems".to_string());
            info.insert("key_features".to_string(), 
                "Enhanced network performance through biological principles".to_string());
        }
    }
    
    Ok(info)
}

#[tauri::command]
pub async fn trigger_havoc_response(
    emergency_type: String,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    warn!("Triggering HAVOC response for: {}", emergency_type);
    
    let mut node = embedded_node.write().await;
    let success = node.trigger_havoc_response(emergency_type).await
        .map_err(|e| format!("Failed to trigger HAVOC response: {}", e))?;
    
    Ok(success)
}

// Peer Management Commands

#[tauri::command]
pub async fn get_peer_list(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<Vec<PeerInfo>> {
    info!("Getting peer list");
    
    let node = embedded_node.read().await;
    let peers = node.get_peer_list().await
        .map_err(|e| format!("Failed to get peer list: {}", e))?;
    
    Ok(peers)
}

#[tauri::command]
pub async fn validate_peer_address(address: String) -> CommandResult<bool> {
    info!("Validating peer address: {}", address);
    
    // Basic multiaddr validation
    match address.parse::<libp2p::Multiaddr>() {
        Ok(_) => Ok(true),
        Err(e) => {
            warn!("Invalid peer address: {}", e);
            Ok(false)
        }
    }
}

// Resource Management Commands

#[tauri::command]
pub async fn get_resource_usage(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<ResourceUsage> {
    let node = embedded_node.read().await;
    let usage = node.get_resource_usage().await
        .map_err(|e| format!("Failed to get resource usage: {}", e))?;
    
    Ok(usage)
}

#[tauri::command]
pub async fn get_thermal_signatures(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<Vec<ThermalSignature>> {
    let node = embedded_node.read().await;
    let signatures = node.get_thermal_signatures().await
        .map_err(|e| format!("Failed to get thermal signatures: {}", e))?;
    
    Ok(signatures)
}

#[tauri::command]
pub async fn update_resource_limits(
    cpu_limit: Option<f64>,
    memory_limit_mb: Option<u64>,
    bandwidth_limit_mbps: Option<f64>,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Updating resource limits");
    
    let mut node = embedded_node.write().await;
    let success = node.update_resource_limits(cpu_limit, memory_limit_mb, bandwidth_limit_mbps).await
        .map_err(|e| format!("Failed to update resource limits: {}", e))?;
    
    Ok(success)
}

// Security Commands

#[tauri::command]
pub async fn get_security_status(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<SecurityStatus> {
    let node = embedded_node.read().await;
    let status = node.get_security_status().await
        .map_err(|e| format!("Failed to get security status: {}", e))?;
    
    Ok(status)
}

// Package Processing Commands

#[tauri::command]
pub async fn get_package_queue(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<PackageQueue> {
    let node = embedded_node.read().await;
    let queue = node.get_package_queue().await
        .map_err(|e| format!("Failed to get package queue: {}", e))?;
    
    Ok(queue)
}

// Node Control Commands

#[tauri::command]
pub async fn start_node(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Starting embedded node");
    
    let mut node = embedded_node.write().await;
    node.start().await
        .map_err(|e| format!("Failed to start node: {}", e))?;
    
    Ok(true)
}

#[tauri::command]
pub async fn stop_node(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Stopping embedded node");
    
    let mut node = embedded_node.write().await;
    node.stop().await
        .map_err(|e| format!("Failed to stop node: {}", e))?;
    
    Ok(true)
}

#[tauri::command]
pub async fn restart_node(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<bool> {
    info!("Restarting embedded node");
    
    let mut node = embedded_node.write().await;
    node.restart().await
        .map_err(|e| format!("Failed to restart node: {}", e))?;
    
    Ok(true)
}

// Configuration Commands

#[tauri::command]
pub async fn import_configuration(
    config_data: String,
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<bool> {
    info!("Importing configuration");
    
    let mut state = app_state.write().await;
    let success = state.import_configuration(config_data).await
        .map_err(|e| format!("Failed to import configuration: {}", e))?;
    
    Ok(success)
}

#[tauri::command]
pub async fn export_configuration(
    app_state: State<'_, Arc<RwLock<AppState>>>,
) -> CommandResult<String> {
    info!("Exporting configuration");
    
    let state = app_state.read().await;
    let config_data = state.export_configuration().await
        .map_err(|e| format!("Failed to export configuration: {}", e))?;
    
    Ok(config_data)
}

// Utility Commands

#[tauri::command]
pub async fn get_node_logs(
    limit: Option<usize>,
    level_filter: Option<String>,
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<Vec<NodeLog>> {
    let node = embedded_node.read().await;
    let logs = node.get_logs(limit, level_filter).await
        .map_err(|e| format!("Failed to get node logs: {}", e))?;
    
    Ok(logs)
}

#[tauri::command]
pub async fn get_performance_metrics(
    embedded_node: State<'_, Arc<RwLock<EmbeddedNode>>>,
) -> CommandResult<PerformanceMetrics> {
    let node = embedded_node.read().await;
    let metrics = node.get_performance_metrics().await
        .map_err(|e| format!("Failed to get performance metrics: {}", e))?;
    
    Ok(metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_peer_address_valid() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let result = validate_peer_address("/ip4/127.0.0.1/tcp/4001".to_string()).await;
            assert!(result.is_ok());
            assert!(result.unwrap());
        });
    }

    #[test]
    fn test_validate_peer_address_invalid() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let result = validate_peer_address("invalid-address".to_string()).await;
            assert!(result.is_ok());
            assert!(!result.unwrap());
        });
    }

    #[test]
    fn test_get_available_biological_roles() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let result = get_available_biological_roles().await;
            assert!(result.is_ok());
            let roles = result.unwrap();
            assert!(!roles.is_empty());
            assert!(roles.contains(&"YoungNode".to_string()));
        });
    }

    #[test]
    fn test_get_biological_role_info() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        
        rt.block_on(async {
            let result = get_biological_role_info("YoungNode".to_string()).await;
            assert!(result.is_ok());
            let info = result.unwrap();
            assert!(info.contains_key("biological_inspiration"));
            assert!(info.contains_key("description"));
        });
    }
}