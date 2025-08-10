//! Event management system for Bio P2P Desktop UI
//! 
//! This module handles event communication between the embedded node
//! and the UI components, enabling real-time updates and notifications.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::{AppHandle, Manager};
use tokio::sync::mpsc;
use tracing::{info, warn, error, debug};
use chrono::{DateTime, Utc};

use crate::commands::{ResourceUsage, BiologicalRole, SecurityThreat};

/// Event manager for coordinating UI updates
pub struct EventManager {
    app_handle: AppHandle,
    event_receiver: Option<mpsc::UnboundedReceiver<UIEvent>>,
    event_sender: mpsc::UnboundedSender<UIEvent>,
}

/// Events that can be emitted to update the UI
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum UIEvent {
    /// Node lifecycle events
    NodeStarted,
    NodeStopped,
    NodeError { message: String },
    
    /// Network events
    PeerConnected { peer_id: String, multiaddr: String },
    PeerDisconnected { peer_id: String, reason: String },
    NetworkStatusChanged { connected: bool, peer_count: usize },
    ConnectionQualityChanged { quality: f64 },
    
    /// Biological role events
    BiologicalRoleChanged { role: String, active: bool, reason: String },
    BiologicalAdaptation { adaptation_type: String, effectiveness: f64 },
    LearningProgress { concept: String, progress: f64 },
    RolePerformanceUpdate { role: String, metrics: RolePerformanceMetrics },
    
    /// Resource events
    ResourceUsageUpdate { usage: ResourceUsage },
    ResourceLimitExceeded { resource_type: String, usage: f64, limit: f64 },
    ThermalAlert { component: String, temperature: f64, threshold: f64 },
    CompartmentReallocation { compartments: Vec<CompartmentAllocation> },
    
    /// Security events
    SecurityThreatDetected { threat: SecurityThreat },
    SecurityThreatResolved { threat_id: String, resolution: String },
    SecurityLayerActivated { layer_name: String, reason: String },
    ImmuneResponseTriggered { response_type: String, effectiveness: f64 },
    
    /// Package processing events
    PackageReceived { package_id: String, package_type: String, size: u64 },
    PackageProcessed { package_id: String, processing_time_ms: u64, success: bool },
    PackageQueueBacklog { queue_size: usize, estimated_delay_ms: u64 },
    
    /// Performance events
    PerformanceAlert { alert_type: String, severity: String, message: String },
    MetricsUpdate { timestamp: DateTime<Utc>, metrics: PerformanceSnapshot },
    
    /// UI interaction events
    PanelVisibilityChanged { panel: String, visible: bool },
    UserActionCompleted { action: String, success: bool, message: Option<String> },
    NotificationDismissed { notification_id: String },
    
    /// Educational events
    BiologicalConceptExplored { concept: String, depth: String },
    LearningMilestoneReached { milestone: String, concept: String },
    TutorialCompleted { tutorial_id: String, completion_time: u64 },
    
    /// System events
    ConfigurationChanged { section: String, key: String },
    ApplicationShutdown { reason: String },
    ErrorOccurred { component: String, error: String, severity: String },
}

/// Performance metrics for biological roles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolePerformanceMetrics {
    pub efficiency: f64,
    pub energy_consumption: f64,
    pub specialization_level: f64,
    pub adaptation_rate: f64,
    pub cooperation_score: f64,
}

/// Compartment allocation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompartmentAllocation {
    pub compartment_name: String,
    pub previous_size_percent: f64,
    pub new_size_percent: f64,
    pub reallocation_reason: String,
}

/// Performance snapshot for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_throughput: f64,
    pub biological_efficiency: f64,
    pub adaptation_score: f64,
}

/// Notification for UI display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UINotification {
    pub id: String,
    pub title: String,
    pub message: String,
    pub severity: NotificationSeverity,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: Option<u64>,
    pub action_button: Option<NotificationAction>,
    pub biological_context: Option<String>,
}

/// Notification severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationSeverity {
    Info,
    Success,
    Warning,
    Error,
    BiologicalEvent,
}

/// Notification action button
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationAction {
    pub label: String,
    pub action_type: String,
    pub parameters: Option<serde_json::Value>,
}

impl EventManager {
    /// Create new event manager
    pub fn new(app_handle: AppHandle) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();

        Self {
            app_handle,
            event_receiver: Some(event_receiver),
            event_sender,
        }
    }

    /// Get event sender for other components to emit events
    pub fn get_event_sender(&self) -> mpsc::UnboundedSender<UIEvent> {
        self.event_sender.clone()
    }

    /// Start the event processing loop
    pub async fn start_event_loop(&mut self) -> Result<()> {
        info!("Starting UI event processing loop");

        let mut receiver = self.event_receiver.take()
            .ok_or_else(|| anyhow::anyhow!("Event receiver already taken"))?;

        while let Some(event) = receiver.recv().await {
            if let Err(e) = self.handle_event(event).await {
                error!("Failed to handle UI event: {}", e);
            }
        }

        info!("Event processing loop ended");
        Ok(())
    }

    /// Handle individual UI event
    async fn handle_event(&self, event: UIEvent) -> Result<()> {
        debug!("Handling UI event: {:?}", event);

        match &event {
            UIEvent::NodeStarted => {
                self.emit_to_frontend("node-started", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Node Started".to_string(),
                    message: "🧬 Bio P2P node is now active and connecting to the network".to_string(),
                    severity: NotificationSeverity::Success,
                    timestamp: Utc::now(),
                    duration_ms: Some(5000),
                    action_button: None,
                    biological_context: Some("Like a seed sprouting into a plant, your node is beginning to grow its network connections".to_string()),
                }).await?;
            }

            UIEvent::NodeStopped => {
                self.emit_to_frontend("node-stopped", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Node Stopped".to_string(),
                    message: "🔴 Bio P2P node has been stopped".to_string(),
                    severity: NotificationSeverity::Info,
                    timestamp: Utc::now(),
                    duration_ms: Some(3000),
                    action_button: None,
                    biological_context: None,
                }).await?;
            }

            UIEvent::PeerConnected { peer_id, multiaddr } => {
                self.emit_to_frontend("peer-connected", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "New Peer Connected".to_string(),
                    message: format!("🤝 Connected to peer: {}", &peer_id[..12]),
                    severity: NotificationSeverity::Info,
                    timestamp: Utc::now(),
                    duration_ms: Some(3000),
                    action_button: None,
                    biological_context: Some("Like animals forming social bonds, nodes create connections that strengthen the network".to_string()),
                }).await?;
            }

            UIEvent::BiologicalRoleChanged { role, active, reason } => {
                self.emit_to_frontend("biological-role-changed", &event).await?;
                let action = if *active { "activated" } else { "deactivated" };
                let emoji = self.get_role_emoji(role);
                
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: format!("Biological Role {}", if *active { "Activated" } else { "Deactivated" }),
                    message: format!("{} {} {} - {}", emoji, role, action, reason),
                    severity: NotificationSeverity::BiologicalEvent,
                    timestamp: Utc::now(),
                    duration_ms: Some(4000),
                    action_button: Some(NotificationAction {
                        label: "View Details".to_string(),
                        action_type: "show_biological_panel".to_string(),
                        parameters: Some(serde_json::json!({"role": role})),
                    }),
                    biological_context: Some(self.get_biological_context(role)),
                }).await?;
            }

            UIEvent::SecurityThreatDetected { threat } => {
                self.emit_to_frontend("security-threat-detected", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Security Alert".to_string(),
                    message: format!("🚨 {} threat detected: {}", threat.severity, threat.threat_type),
                    severity: NotificationSeverity::Error,
                    timestamp: Utc::now(),
                    duration_ms: None, // Persistent until dismissed
                    action_button: Some(NotificationAction {
                        label: "View Security Dashboard".to_string(),
                        action_type: "show_security_panel".to_string(),
                        parameters: None,
                    }),
                    biological_context: Some("The immune system has detected a potential threat and is activating defense mechanisms".to_string()),
                }).await?;
            }

            UIEvent::ResourceLimitExceeded { resource_type, usage, limit } => {
                self.emit_to_frontend("resource-limit-exceeded", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Resource Limit Exceeded".to_string(),
                    message: format!("⚠️ {} usage ({:.1}%) exceeded limit ({:.1}%)", resource_type, usage, limit),
                    severity: NotificationSeverity::Warning,
                    timestamp: Utc::now(),
                    duration_ms: Some(8000),
                    action_button: Some(NotificationAction {
                        label: "Adjust Limits".to_string(),
                        action_type: "show_resource_panel".to_string(),
                        parameters: Some(serde_json::json!({"resource": resource_type})),
                    }),
                    biological_context: Some("Like animals conserving energy during resource scarcity, the node needs to manage its resources more carefully".to_string()),
                }).await?;
            }

            UIEvent::ThermalAlert { component, temperature, threshold } => {
                self.emit_to_frontend("thermal-alert", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Thermal Alert".to_string(),
                    message: format!("🌡️ {} temperature ({:.1}°C) exceeded threshold ({:.1}°C)", component, temperature, threshold),
                    severity: NotificationSeverity::Warning,
                    timestamp: Utc::now(),
                    duration_ms: Some(6000),
                    action_button: None,
                    biological_context: Some("Like animals seeking shade in hot weather, the system is monitoring thermal conditions for optimal performance".to_string()),
                }).await?;
            }

            UIEvent::LearningMilestoneReached { milestone, concept } => {
                self.emit_to_frontend("learning-milestone-reached", &event).await?;
                self.show_notification(UINotification {
                    id: uuid::Uuid::new_v4().to_string(),
                    title: "Learning Milestone Reached!".to_string(),
                    message: format!("🎓 Completed: {} for concept '{}'", milestone, concept),
                    severity: NotificationSeverity::Success,
                    timestamp: Utc::now(),
                    duration_ms: Some(6000),
                    action_button: Some(NotificationAction {
                        label: "Continue Learning".to_string(),
                        action_type: "show_education_panel".to_string(),
                        parameters: Some(serde_json::json!({"concept": concept})),
                    }),
                    biological_context: Some("Like young animals mastering new skills, you're developing expertise in biological computing concepts".to_string()),
                }).await?;
            }

            _ => {
                // For other events, just emit to frontend
                let event_name = self.get_event_name(&event);
                self.emit_to_frontend(&event_name, &event).await?;
            }
        }

        Ok(())
    }

    /// Emit event to frontend
    async fn emit_to_frontend(&self, event_name: &str, event_data: &UIEvent) -> Result<()> {
        if let Err(e) = self.app_handle.emit_all(event_name, event_data) {
            error!("Failed to emit event '{}' to frontend: {}", event_name, e);
            return Err(e.into());
        }

        debug!("Emitted event '{}' to frontend", event_name);
        Ok(())
    }

    /// Show notification to user
    async fn show_notification(&self, notification: UINotification) -> Result<()> {
        // Emit notification event
        if let Err(e) = self.app_handle.emit_all("notification", &notification) {
            error!("Failed to emit notification: {}", e);
            return Err(e.into());
        }

        // Also show system notification if enabled
        if let Err(e) = self.show_system_notification(&notification).await {
            warn!("Failed to show system notification: {}", e);
        }

        Ok(())
    }

    /// Show system-level notification (desktop notification)
    async fn show_system_notification(&self, notification: &UINotification) -> Result<()> {
        let mut title = notification.title.clone();
        let mut body = notification.message.clone();

        // Add biological context if available and short enough
        if let Some(context) = &notification.biological_context {
            if context.len() < 100 {
                body = format!("{}\n\n💡 {}", body, context);
            }
        }

        // Create system notification
        #[cfg(feature = "notification")]
        {
            use tauri::api::notification::Notification;
            
            Notification::new(&self.app_handle.config().tauri.bundle.identifier)
                .title(&title)
                .body(&body)
                .show()?;
        }

        Ok(())
    }

    /// Get emoji for biological role
    fn get_role_emoji(&self, role: &str) -> &'static str {
        match role {
            "YoungNode" => "🐦",
            "CasteNode" => "🐜",
            "ImitateNode" => "🦜",
            "HatchNode" => "🐢",
            "SyncPhaseNode" => "🐧",
            "HuddleNode" => "🐧",
            "MigrationNode" => "🦌",
            "TrustNode" => "🤝",
            "HAVOCNode" => "🦟",
            "ThermalNode" => "🌡️",
            "DOSNode" => "🛡️",
            "InvestigationNode" => "🔍",
            "BuddyNode" => "👫",
            "FriendshipNode" => "🤝",
            _ => "🧬",
        }
    }

    /// Get biological context explanation for role
    fn get_biological_context(&self, role: &str) -> String {
        match role {
            "YoungNode" => "Like young crows learning from their elders, this node observes and learns from experienced network peers".to_string(),
            "CasteNode" => "Similar to ant colonies with specialized castes, this role compartmentalizes functions for maximum efficiency".to_string(),
            "TrustNode" => "Like primates building social bonds through grooming, this role manages trust relationships with other nodes".to_string(),
            "HAVOCNode" => "Inspired by mosquitoes' rapid adaptation to environmental changes, this role handles emergency resource reallocation".to_string(),
            "ThermalNode" => "Like ants using pheromone trails to indicate food quality, this role monitors and signals resource availability".to_string(),
            "DOSNode" => "Similar to immune system sentinels, this role continuously monitors for potential threats and attacks".to_string(),
            _ => "This biological role applies natural intelligence patterns to enhance network performance".to_string(),
        }
    }

    /// Get frontend event name for UIEvent
    fn get_event_name(&self, event: &UIEvent) -> String {
        match event {
            UIEvent::NodeStarted => "node-started",
            UIEvent::NodeStopped => "node-stopped",
            UIEvent::NodeError { .. } => "node-error",
            UIEvent::PeerConnected { .. } => "peer-connected",
            UIEvent::PeerDisconnected { .. } => "peer-disconnected",
            UIEvent::NetworkStatusChanged { .. } => "network-status-changed",
            UIEvent::ConnectionQualityChanged { .. } => "connection-quality-changed",
            UIEvent::BiologicalRoleChanged { .. } => "biological-role-changed",
            UIEvent::BiologicalAdaptation { .. } => "biological-adaptation",
            UIEvent::LearningProgress { .. } => "learning-progress",
            UIEvent::RolePerformanceUpdate { .. } => "role-performance-update",
            UIEvent::ResourceUsageUpdate { .. } => "resource-usage-update",
            UIEvent::ResourceLimitExceeded { .. } => "resource-limit-exceeded",
            UIEvent::ThermalAlert { .. } => "thermal-alert",
            UIEvent::CompartmentReallocation { .. } => "compartment-reallocation",
            UIEvent::SecurityThreatDetected { .. } => "security-threat-detected",
            UIEvent::SecurityThreatResolved { .. } => "security-threat-resolved",
            UIEvent::SecurityLayerActivated { .. } => "security-layer-activated",
            UIEvent::ImmuneResponseTriggered { .. } => "immune-response-triggered",
            UIEvent::PackageReceived { .. } => "package-received",
            UIEvent::PackageProcessed { .. } => "package-processed",
            UIEvent::PackageQueueBacklog { .. } => "package-queue-backlog",
            UIEvent::PerformanceAlert { .. } => "performance-alert",
            UIEvent::MetricsUpdate { .. } => "metrics-update",
            UIEvent::PanelVisibilityChanged { .. } => "panel-visibility-changed",
            UIEvent::UserActionCompleted { .. } => "user-action-completed",
            UIEvent::NotificationDismissed { .. } => "notification-dismissed",
            UIEvent::BiologicalConceptExplored { .. } => "biological-concept-explored",
            UIEvent::LearningMilestoneReached { .. } => "learning-milestone-reached",
            UIEvent::TutorialCompleted { .. } => "tutorial-completed",
            UIEvent::ConfigurationChanged { .. } => "configuration-changed",
            UIEvent::ApplicationShutdown { .. } => "application-shutdown",
            UIEvent::ErrorOccurred { .. } => "error-occurred",
        }.to_string()
    }

    /// Emit a custom event
    pub async fn emit_event(&self, event: UIEvent) -> Result<()> {
        self.event_sender.send(event)?;
        Ok(())
    }

    /// Create and emit a biological role change event
    pub async fn emit_role_change(&self, role: String, active: bool, reason: String) -> Result<()> {
        let event = UIEvent::BiologicalRoleChanged {
            role,
            active,
            reason,
        };
        self.emit_event(event).await
    }

    /// Create and emit a peer connection event
    pub async fn emit_peer_connected(&self, peer_id: String, multiaddr: String) -> Result<()> {
        let event = UIEvent::PeerConnected {
            peer_id,
            multiaddr,
        };
        self.emit_event(event).await
    }

    /// Create and emit a security threat event
    pub async fn emit_security_threat(&self, threat: SecurityThreat) -> Result<()> {
        let event = UIEvent::SecurityThreatDetected {
            threat,
        };
        self.emit_event(event).await
    }

    /// Create and emit a resource usage update event
    pub async fn emit_resource_update(&self, usage: ResourceUsage) -> Result<()> {
        let event = UIEvent::ResourceUsageUpdate {
            usage,
        };
        self.emit_event(event).await
    }

    /// Create and emit a learning progress event
    pub async fn emit_learning_progress(&self, concept: String, progress: f64) -> Result<()> {
        let event = UIEvent::LearningProgress {
            concept,
            progress,
        };
        self.emit_event(event).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_name_generation() {
        let event_manager = create_test_event_manager();
        
        let event = UIEvent::NodeStarted;
        assert_eq!(event_manager.get_event_name(&event), "node-started");
        
        let event = UIEvent::BiologicalRoleChanged {
            role: "YoungNode".to_string(),
            active: true,
            reason: "Test".to_string(),
        };
        assert_eq!(event_manager.get_event_name(&event), "biological-role-changed");
    }

    #[test]
    fn test_role_emoji_mapping() {
        let event_manager = create_test_event_manager();
        
        assert_eq!(event_manager.get_role_emoji("YoungNode"), "🐦");
        assert_eq!(event_manager.get_role_emoji("CasteNode"), "🐜");
        assert_eq!(event_manager.get_role_emoji("UnknownRole"), "🧬");
    }

    #[test]
    fn test_biological_context() {
        let event_manager = create_test_event_manager();
        
        let context = event_manager.get_biological_context("YoungNode");
        assert!(context.contains("young crows"));
        assert!(context.contains("learning"));
        
        let context = event_manager.get_biological_context("CasteNode");
        assert!(context.contains("ant colonies"));
        assert!(context.contains("specialized"));
    }

    fn create_test_event_manager() -> EventManager {
        // Create a mock AppHandle for testing
        // This would need to be implemented properly in a real test environment
        todo!("Implement test event manager creation")
    }
}