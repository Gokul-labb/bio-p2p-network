//! Main egui application for Bio P2P Desktop UI
//! 
//! This module contains the primary egui application structure that manages
//! the overall UI layout, panel coordination, and biological network visualization.

use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

use crate::state::{AppState, UIState, PanelState, BiologicalMetaphorLevel};
use crate::node::EmbeddedNode;
use crate::ui::{
    NetworkStatusPanel, BiologicalRolesPanel, PeerManagementPanel,
    ResourceMonitorPanel, SecurityDashboardPanel, PackageProcessingPanel,
    NetworkTopologyPanel, LogViewerPanel,
};

/// Main Bio P2P Desktop Application
pub struct BioP2PApp {
    /// Application state
    app_state: Arc<RwLock<AppState>>,
    
    /// Embedded Bio P2P node
    embedded_node: Arc<RwLock<EmbeddedNode>>,
    
    /// Current UI state
    ui_state: UIState,
    
    /// Panel visibility and layout state
    panel_state: PanelState,
    
    /// UI panels
    network_panel: NetworkStatusPanel,
    biological_panel: BiologicalRolesPanel,
    peer_panel: PeerManagementPanel,
    resource_panel: ResourceMonitorPanel,
    security_panel: SecurityDashboardPanel,
    package_panel: PackageProcessingPanel,
    topology_panel: NetworkTopologyPanel,
    logs_panel: LogViewerPanel,
    
    /// Application theme and styling
    theme: AppTheme,
    
    /// Status and notifications
    status_message: String,
    last_update: DateTime<Utc>,
    
    /// Biological education system
    biological_education: BiologicalEducationSystem,
}

/// Application theme configuration
#[derive(Debug, Clone)]
pub struct AppTheme {
    primary_color: egui::Color32,
    secondary_color: egui::Color32,
    accent_color: egui::Color32,
    background_color: egui::Color32,
    text_color: egui::Color32,
    success_color: egui::Color32,
    warning_color: egui::Color32,
    error_color: egui::Color32,
    biological_color: egui::Color32,
}

/// Educational system for biological concepts
#[derive(Debug)]
pub struct BiologicalEducationSystem {
    tooltip_provider: HashMap<String, BiologicalTooltip>,
    learning_progress: HashMap<String, f64>,
    show_hints: bool,
    metaphor_level: BiologicalMetaphorLevel,
}

/// Biological concept tooltip
#[derive(Debug, Clone)]
pub struct BiologicalTooltip {
    concept: String,
    simple_explanation: String,
    detailed_explanation: String,
    biological_example: String,
    technical_implementation: String,
}

impl BioP2PApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        app_state: Arc<RwLock<AppState>>,
        embedded_node: Arc<RwLock<EmbeddedNode>>,
    ) -> Self {
        // Configure egui style for biological theme
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(240, 255, 240);
        style.visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(220, 255, 220);
        style.visuals.widgets.active.bg_fill = egui::Color32::from_rgb(34, 139, 34);
        cc.egui_ctx.set_style(style);

        let theme = AppTheme::biological_forest();
        
        Self {
            network_panel: NetworkStatusPanel::new(app_state.clone()),
            biological_panel: BiologicalRolesPanel::new(app_state.clone()),
            peer_panel: PeerManagementPanel::new(app_state.clone()),
            resource_panel: ResourceMonitorPanel::new(app_state.clone()),
            security_panel: SecurityDashboardPanel::new(app_state.clone()),
            package_panel: PackageProcessingPanel::new(app_state.clone()),
            topology_panel: NetworkTopologyPanel::new(app_state.clone()),
            logs_panel: LogViewerPanel::new(app_state.clone()),
            app_state,
            embedded_node,
            ui_state: UIState::new(),
            panel_state: PanelState::default(),
            theme,
            status_message: "Starting Bio P2P Network...".to_string(),
            last_update: Utc::now(),
            biological_education: BiologicalEducationSystem::new(),
        }
    }

    /// Update application state from embedded node
    async fn update_from_node(&mut self) {
        if let Ok(node) = self.embedded_node.try_read() {
            // Update network status
            if let Ok(network_status) = node.get_network_status().await {
                self.ui_state.network_connected = network_status.connected;
                self.ui_state.connected_peers = network_status.peer_count;
                self.ui_state.connection_quality = network_status.connection_quality.parse().unwrap_or(0.0);
            }

            // Update resource usage
            if let Ok(resource_usage) = node.get_resource_usage().await {
                self.ui_state.cpu_usage = resource_usage.cpu_usage_percent;
                self.ui_state.memory_usage_mb = resource_usage.memory_usage_mb;
                self.ui_state.network_upload_mbps = resource_usage.network_upload_mbps;
                self.ui_state.network_download_mbps = resource_usage.network_download_mbps;
            }

            // Update security status
            if let Ok(security_status) = node.get_security_status().await {
                self.ui_state.security_level = security_status.security_level;
                self.ui_state.security_threats = security_status.active_threats.len();
                self.ui_state.last_security_scan = Some(security_status.last_security_scan);
            }

            // Update package processing
            if let Ok(package_queue) = node.get_package_queue().await {
                self.ui_state.package_queue_size = package_queue.active_packages.len();
                self.ui_state.packages_processed = package_queue.processing_stats.total_processed;
                self.ui_state.processing_rate = package_queue.processing_stats.throughput_per_minute;
            }
        }

        self.last_update = Utc::now();
    }
}

impl eframe::App for BioP2PApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Periodic updates
        ctx.request_repaint_after(std::time::Duration::from_millis(1000));

        // Update from embedded node (async context needed)
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            // This would need proper async handling
            // For now, we'll simulate updates
        });

        // Top panel with main navigation
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            self.show_top_panel(ui);
        });

        // Bottom panel with status information
        egui::TopBottomPanel::bottom("bottom_panel").show(ctx, |ui| {
            self.show_bottom_panel(ui);
        });

        // Left panel for navigation and quick controls
        egui::SidePanel::left("left_panel")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                self.show_navigation_panel(ui);
            });

        // Central panel content area
        egui::CentralPanel::default().show(ctx, |ui| {
            self.show_main_content(ui);
        });

        // Show floating windows/dialogs
        self.show_floating_windows(ctx);

        // Handle biological education tooltips
        self.handle_biological_education(ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        // Save panel state
        eframe::set_value(storage, "panel_state", &self.panel_state);
        
        // Save UI preferences
        if let Ok(app_state) = self.app_state.try_read() {
            eframe::set_value(storage, "ui_preferences", &app_state.ui_preferences);
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // Perform cleanup when application exits
        let rt = tokio::runtime::Handle::current();
        let embedded_node = self.embedded_node.clone();
        
        rt.spawn(async move {
            if let Ok(mut node) = embedded_node.try_write() {
                let _ = node.stop().await;
            }
        });
    }
}

impl BioP2PApp {
    fn show_top_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Application title and icon
            ui.label(egui::RichText::new("🧬 Bio P2P Network").size(18.0).strong());
            
            ui.separator();
            
            // Connection status indicator
            let status_text = if self.ui_state.network_connected {
                format!("● Connected ({} peers)", self.ui_state.connected_peers)
            } else {
                "○ Disconnected".to_string()
            };
            let status_color = if self.ui_state.network_connected { 
                egui::Color32::GREEN 
            } else { 
                egui::Color32::RED 
            };
            ui.colored_label(status_color, status_text);
            
            ui.separator();
            
            // Quick biological status
            if !self.ui_state.active_roles.is_empty() {
                ui.label(format!("🐜 {} active roles", self.ui_state.active_roles.len()));
            } else {
                ui.colored_label(egui::Color32::GRAY, "🐜 No active roles");
            }
            
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Application controls
                if ui.button("⚙ Settings").clicked() {
                    // Open settings
                }
                
                if ui.button("❓ Help").clicked() {
                    // Open help
                }
                
                // Node control buttons
                if self.ui_state.node_running {
                    if ui.button("⏸ Stop Node").clicked() {
                        // TODO: Stop node command
                    }
                } else {
                    if ui.button("▶ Start Node").clicked() {
                        // TODO: Start node command
                    }
                }
            });
        });
    }

    fn show_bottom_panel(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Status message
            ui.label(&self.status_message);
            
            ui.separator();
            
            // Resource usage summary
            ui.label(format!("CPU: {:.1}%", self.ui_state.cpu_usage));
            ui.label(format!("RAM: {:.1} MB", self.ui_state.memory_usage_mb));
            ui.label(format!("Net: ↑{:.1}/↓{:.1} MB/s", 
                self.ui_state.network_upload_mbps, 
                self.ui_state.network_download_mbps));
            
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                // Last update time
                let elapsed = (Utc::now() - self.last_update).num_seconds();
                ui.small(format!("Updated {}s ago", elapsed));
                
                ui.separator();
                
                // Security status
                match self.ui_state.security_threats {
                    0 => ui.colored_label(egui::Color32::GREEN, "🛡️ Secure"),
                    1 => ui.colored_label(egui::Color32::YELLOW, "⚠️ 1 Alert"),
                    n => ui.colored_label(egui::Color32::RED, &format!("🚨 {} Alerts", n)),
                };
            });
        });
    }

    fn show_navigation_panel(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("Network");
            
            // Panel selection buttons
            if ui.selectable_label(self.panel_state.show_network, "🌐 Network Status").clicked() {
                self.panel_state.show_network = !self.panel_state.show_network;
            }
            
            if ui.selectable_label(self.panel_state.show_biological, "🐜 Biological Roles").clicked() {
                self.panel_state.show_biological = !self.panel_state.show_biological;
            }
            
            if ui.selectable_label(self.panel_state.show_peers, "👥 Peer Management").clicked() {
                self.panel_state.show_peers = !self.panel_state.show_peers;
            }
            
            ui.separator();
            ui.heading("System");
            
            if ui.selectable_label(self.panel_state.show_resources, "📊 Resource Monitor").clicked() {
                self.panel_state.show_resources = !self.panel_state.show_resources;
            }
            
            if ui.selectable_label(self.panel_state.show_security, "🛡️ Security Dashboard").clicked() {
                self.panel_state.show_security = !self.panel_state.show_security;
            }
            
            if ui.selectable_label(self.panel_state.show_packages, "📦 Package Processing").clicked() {
                self.panel_state.show_packages = !self.panel_state.show_packages;
            }
            
            ui.separator();
            ui.heading("Analysis");
            
            if ui.selectable_label(self.panel_state.show_topology, "🗺️ Network Topology").clicked() {
                self.panel_state.show_topology = !self.panel_state.show_topology;
            }
            
            if ui.selectable_label(self.panel_state.show_logs, "📝 Logs").clicked() {
                self.panel_state.show_logs = !self.panel_state.show_logs;
            }

            ui.add_space(20.0);
            
            // Quick actions
            ui.heading("Quick Actions");
            
            if ui.button("🔍 Add Peer").clicked() {
                // Open add peer dialog
            }
            
            if ui.button("🧠 Auto-Adapt Roles").clicked() {
                // Trigger biological adaptation
            }
            
            if ui.button("📋 Export Config").clicked() {
                // Export configuration
            }

            ui.add_space(20.0);
            
            // Biological education
            self.show_biological_education_section(ui);
        });
    }

    fn show_main_content(&mut self, ui: &mut egui::Ui) {
        if !self.panel_state.any_panel_open() {
            // Show welcome/dashboard view
            self.show_dashboard(ui);
            return;
        }

        egui::ScrollArea::vertical().show(ui, |ui| {
            // Show active panels
            if self.panel_state.show_network {
                ui.group(|ui| {
                    self.network_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_biological {
                ui.group(|ui| {
                    self.biological_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_peers {
                ui.group(|ui| {
                    self.peer_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_resources {
                ui.group(|ui| {
                    self.resource_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_security {
                ui.group(|ui| {
                    self.security_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_packages {
                ui.group(|ui| {
                    self.package_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_topology {
                ui.group(|ui| {
                    self.topology_panel.show(ui);
                });
                ui.add_space(10.0);
            }

            if self.panel_state.show_logs {
                ui.group(|ui| {
                    self.logs_panel.show(ui);
                });
                ui.add_space(10.0);
            }
        });
    }

    fn show_dashboard(&mut self, ui: &mut egui::Ui) {
        ui.vertical_centered(|ui| {
            ui.add_space(50.0);
            
            ui.heading("🧬 Welcome to Bio P2P Network");
            ui.add_space(10.0);
            ui.label("A revolutionary peer-to-peer platform inspired by biological systems");
            
            ui.add_space(30.0);
            
            // Quick start cards
            ui.horizontal(|ui| {
                if ui.button("🚀 Start Your Node").clicked() {
                    // Start node
                }
                
                if ui.button("🐜 Explore Biology").clicked() {
                    self.panel_state.show_biological = true;
                }
                
                if ui.button("🌐 View Network").clicked() {
                    self.panel_state.show_network = true;
                }
            });

            ui.add_space(30.0);
            
            // Getting started information
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.strong("Getting Started:");
                    ui.label("1. Start your node to join the biological P2P network");
                    ui.label("2. Watch as Young Node learns from experienced peers");
                    ui.label("3. Enable biological roles for specialized functions");
                    ui.label("4. Monitor network health and adaptation progress");
                });
            });
        });
    }

    fn show_biological_education_section(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("🎓 Learn Biology", |ui| {
            ui.checkbox(&mut self.biological_education.show_hints, "Show Hints");
            
            ui.horizontal(|ui| {
                ui.label("Level:");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.biological_education.metaphor_level))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.biological_education.metaphor_level, BiologicalMetaphorLevel::Simple, "Simple");
                        ui.selectable_value(&mut self.biological_education.metaphor_level, BiologicalMetaphorLevel::Intermediate, "Intermediate");
                        ui.selectable_value(&mut self.biological_education.metaphor_level, BiologicalMetaphorLevel::Advanced, "Advanced");
                        ui.selectable_value(&mut self.biological_education.metaphor_level, BiologicalMetaphorLevel::Expert, "Expert");
                    });
            });
            
            ui.separator();
            
            ui.small("Today's Bio Concept:");
            ui.label("🐦 Young Node Learning");
            ui.small("Crows learn by observing elders");
            
            if ui.small_button("Learn More").clicked() {
                // Show detailed biological explanation
            }
        });
    }

    fn show_floating_windows(&mut self, ctx: &egui::Context) {
        // Show any floating dialogs or windows
        // This would include configuration dialogs, help windows, etc.
    }

    fn handle_biological_education(&mut self, ctx: &egui::Context) {
        // Handle educational tooltips and hints
        if self.biological_education.show_hints {
            // Show contextual hints based on current panel
        }
    }
}

impl AppTheme {
    pub fn biological_forest() -> Self {
        Self {
            primary_color: egui::Color32::from_rgb(34, 139, 34),     // Forest green
            secondary_color: egui::Color32::from_rgb(144, 238, 144), // Light green
            accent_color: egui::Color32::from_rgb(255, 215, 0),      // Gold
            background_color: egui::Color32::from_rgb(248, 255, 248), // Very light green
            text_color: egui::Color32::from_rgb(0, 50, 0),           // Dark green
            success_color: egui::Color32::from_rgb(0, 128, 0),       // Green
            warning_color: egui::Color32::from_rgb(255, 140, 0),     // Dark orange
            error_color: egui::Color32::from_rgb(220, 20, 60),       // Crimson
            biological_color: egui::Color32::from_rgb(46, 125, 50),  // Bio green
        }
    }
}

impl BiologicalEducationSystem {
    pub fn new() -> Self {
        let mut tooltip_provider = HashMap::new();
        
        // Add biological tooltips
        tooltip_provider.insert("YoungNode".to_string(), BiologicalTooltip {
            concept: "Young Node Learning".to_string(),
            simple_explanation: "New nodes learn from experienced neighbors".to_string(),
            detailed_explanation: "Young crows observe and imitate successful behaviors of adult crows in their social group".to_string(),
            biological_example: "A young crow watches elders use tools to extract insects from tree bark".to_string(),
            technical_implementation: "Nodes observe routing patterns and resource allocation strategies from up to 100 neighboring nodes".to_string(),
        });

        Self {
            tooltip_provider,
            learning_progress: HashMap::new(),
            show_hints: true,
            metaphor_level: BiologicalMetaphorLevel::Intermediate,
        }
    }
}