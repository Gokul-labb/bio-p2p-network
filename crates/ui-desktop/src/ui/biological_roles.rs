//! Biological Roles Panel - manages and displays biological node behaviors
//! 
//! This panel allows users to view, configure, and monitor the various biological
//! roles that their node can adopt, with detailed explanations and performance metrics.

use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::state::AppState;
use super::{UIPanel, biological_widgets::*, biological_tooltips::*};

/// Biological roles management panel
pub struct BiologicalRolesPanel {
    app_state: Arc<RwLock<AppState>>,
    
    // UI state
    selected_role: Option<String>,
    show_role_details: bool,
    role_filter: RoleFilter,
    search_query: String,
    
    // Role configuration
    role_configs: HashMap<String, RoleConfiguration>,
    
    // Educational components
    tooltip_provider: BiologicalTooltipProvider,
    tooltip_level: TooltipLevel,
    show_biological_context: bool,
    
    // Performance tracking
    role_performance_history: HashMap<String, Vec<f64>>,
    adaptation_events: Vec<AdaptationEvent>,
}

/// Role filter options
#[derive(Debug, Clone, PartialEq)]
enum RoleFilter {
    All,
    Active,
    Available,
    Learning,
    Coordination,
    Security,
    Resource,
}

/// Role configuration settings
#[derive(Debug, Clone)]
struct RoleConfiguration {
    role_type: String,
    enabled: bool,
    auto_activate: bool,
    performance_threshold: f64,
    energy_efficiency_target: f64,
    specialization_level: f64,
    custom_parameters: HashMap<String, f64>,
}

/// Biological adaptation event
#[derive(Debug, Clone)]
struct AdaptationEvent {
    timestamp: std::time::Instant,
    role: String,
    event_type: AdaptationEventType,
    effectiveness: f64,
    description: String,
}

/// Types of adaptation events
#[derive(Debug, Clone)]
enum AdaptationEventType {
    RoleActivated,
    RoleDeactivated,
    PerformanceImproved,
    PerformanceDeclined,
    SpecializationIncreased,
    EnergyEfficiencyImproved,
}

impl BiologicalRolesPanel {
    pub fn new(app_state: Arc<RwLock<AppState>>) -> Self {
        Self {
            app_state,
            selected_role: None,
            show_role_details: false,
            role_filter: RoleFilter::All,
            search_query: String::new(),
            role_configs: Self::create_default_role_configs(),
            tooltip_provider: BiologicalTooltipProvider::new(),
            tooltip_level: TooltipLevel::Detailed,
            show_biological_context: true,
            role_performance_history: HashMap::new(),
            adaptation_events: Vec::new(),
        }
    }
    
    /// Create default role configurations
    fn create_default_role_configs() -> HashMap<String, RoleConfiguration> {
        let mut configs = HashMap::new();
        
        // Learning and Adaptation roles
        configs.insert("YoungNode".to_string(), RoleConfiguration {
            role_type: "Learning".to_string(),
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.6,
            energy_efficiency_target: 0.8,
            specialization_level: 0.7,
            custom_parameters: HashMap::new(),
        });
        
        configs.insert("ImitateNode".to_string(), RoleConfiguration {
            role_type: "Learning".to_string(),
            enabled: false,
            auto_activate: false,
            performance_threshold: 0.5,
            energy_efficiency_target: 0.75,
            specialization_level: 0.6,
            custom_parameters: HashMap::new(),
        });
        
        // Coordination roles
        configs.insert("HatchNode".to_string(), RoleConfiguration {
            role_type: "Coordination".to_string(),
            enabled: false,
            auto_activate: true,
            performance_threshold: 0.7,
            energy_efficiency_target: 0.85,
            specialization_level: 0.8,
            custom_parameters: HashMap::new(),
        });
        
        configs.insert("SyncPhaseNode".to_string(), RoleConfiguration {
            role_type: "Coordination".to_string(),
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.65,
            energy_efficiency_target: 0.8,
            specialization_level: 0.75,
            custom_parameters: HashMap::new(),
        });
        
        // Resource management roles
        configs.insert("CasteNode".to_string(), RoleConfiguration {
            role_type: "Resource".to_string(),
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.8,
            energy_efficiency_target: 0.9,
            specialization_level: 0.85,
            custom_parameters: HashMap::new(),
        });
        
        configs.insert("HAVOCNode".to_string(), RoleConfiguration {
            role_type: "Resource".to_string(),
            enabled: false,
            auto_activate: false,
            performance_threshold: 0.9,
            energy_efficiency_target: 0.7,
            specialization_level: 0.9,
            custom_parameters: HashMap::new(),
        });
        
        // Security roles
        configs.insert("DOSNode".to_string(), RoleConfiguration {
            role_type: "Security".to_string(),
            enabled: true,
            auto_activate: true,
            performance_threshold: 0.75,
            energy_efficiency_target: 0.8,
            specialization_level: 0.8,
            custom_parameters: HashMap::new(),
        });
        
        configs
    }
    
    /// Show role overview grid
    fn show_role_overview(&mut self, ui: &mut egui::Ui) {
        ui.heading("🧬 Biological Roles");
        ui.add_space(10.0);
        
        // Filter and search controls
        ui.horizontal(|ui| {
            ui.label("Filter:");
            egui::ComboBox::from_label("")
                .selected_text(format!("{:?}", self.role_filter))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.role_filter, RoleFilter::All, "All Roles");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Active, "Active");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Available, "Available");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Learning, "Learning");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Coordination, "Coordination");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Security, "Security");
                    ui.selectable_value(&mut self.role_filter, RoleFilter::Resource, "Resource");
                });
            
            ui.separator();
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.search_query);
            
            ui.separator();
            if ui.button("🔄 Auto-adapt").clicked() {
                self.trigger_auto_adaptation();
            }
        });
        
        ui.add_space(10.0);
        
        // Biological context information
        if self.show_biological_context {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label("💡");
                    ui.strong("Biological Intelligence:");
                    ui.label("Your node can adopt specialized roles inspired by nature");
                });
                ui.small("Each role provides unique capabilities that enhance network performance through biological principles");
            });
            ui.add_space(10.0);
        }
        
        // Role cards grid
        let filtered_roles = self.filter_roles();
        let cols = 2;
        let mut row = 0;
        
        while row * cols < filtered_roles.len() {
            ui.horizontal(|ui| {
                for col in 0..cols {
                    let idx = row * cols + col;
                    if idx < filtered_roles.len() {
                        self.show_role_card(ui, &filtered_roles[idx]);
                        if col < cols - 1 {
                            ui.add_space(10.0);
                        }
                    }
                }
            });
            ui.add_space(10.0);
            row += 1;
        }
    }
    
    /// Show individual role card
    fn show_role_card(&mut self, ui: &mut egui::Ui, role_name: &str) {
        let config = self.role_configs.get(role_name).unwrap();
        let is_active = config.enabled;
        let performance = self.get_role_performance(role_name);
        let emoji = self.get_role_emoji(role_name);
        
        let card_width = 280.0;
        ui.allocate_ui(egui::vec2(card_width, 120.0), |ui| {
            ui.group(|ui| {
                ui.vertical(|ui| {
                    // Header with role name and status
                    ui.horizontal(|ui| {
                        let response = biological_tooltip(
                            ui, 
                            &format!("{} {}", emoji, role_name), 
                            role_name, 
                            &self.tooltip_provider, 
                            self.tooltip_level
                        );
                        
                        if response.clicked() {
                            self.selected_role = Some(role_name.to_string());
                            self.show_role_details = true;
                        }
                        
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let mut enabled = config.enabled;
                            if ui.checkbox(&mut enabled, "").changed() {
                                self.toggle_role(role_name, enabled);
                            }
                        });
                    });
                    
                    ui.add_space(5.0);
                    
                    // Performance indicator
                    if is_active {
                        biological_progress_bar(ui, performance, "Performance");
                        ui.add_space(3.0);
                        ui.horizontal(|ui| {
                            ui.small(format!("Energy: {:.0}%", config.energy_efficiency_target * 100.0));
                            ui.separator();
                            ui.small(format!("Specialization: {:.0}%", config.specialization_level * 100.0));
                        });
                    } else {
                        ui.horizontal(|ui| {
                            ui.colored_label(egui::Color32::GRAY, "Inactive");
                            if config.auto_activate {
                                ui.separator();
                                ui.small("Auto-activate enabled");
                            }
                        });
                    }
                    
                    ui.add_space(5.0);
                    
                    // Quick description
                    ui.small(self.get_role_description(role_name));
                });
            });
        });
    }
    
    /// Show detailed role information
    fn show_role_details(&mut self, ui: &mut egui::Ui) {
        if let Some(ref role_name) = self.selected_role.clone() {
            ui.heading(&format!("🔬 {} Details", role_name));
            ui.add_space(10.0);
            
            if let Some(config) = self.role_configs.get(role_name) {
                // Role status and controls
                ui.horizontal(|ui| {
                    let emoji = self.get_role_emoji(role_name);
                    ui.label(&format!("{} {}", emoji, role_name));
                    ui.separator();
                    
                    let status_text = if config.enabled { "Active" } else { "Inactive" };
                    let status_color = if config.enabled { 
                        super::chart_utils::BiologicalColors::PRIMARY 
                    } else { 
                        egui::Color32::GRAY 
                    };
                    ui.colored_label(status_color, status_text);
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("❌ Close").clicked() {
                            self.show_role_details = false;
                            self.selected_role = None;
                        }
                        
                        if config.enabled {
                            if ui.button("⏸ Deactivate").clicked() {
                                self.toggle_role(role_name, false);
                            }
                        } else {
                            if ui.button("▶ Activate").clicked() {
                                self.toggle_role(role_name, true);
                            }
                        }
                    });
                });
                
                ui.add_space(15.0);
                
                // Biological explanation
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.strong("🧬 Biological Inspiration");
                        ui.add_space(5.0);
                        self.tooltip_provider.show_tooltip(ui, role_name, TooltipLevel::Technical);
                    });
                });
                
                ui.add_space(15.0);
                
                // Performance metrics
                if config.enabled {
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.strong("📊 Performance Metrics");
                            ui.add_space(10.0);
                            
                            let performance = self.get_role_performance(role_name);
                            biological_progress_bar(ui, performance, "Overall Performance");
                            biological_progress_bar(ui, config.energy_efficiency_target, "Energy Efficiency");
                            biological_progress_bar(ui, config.specialization_level, "Specialization Level");
                            
                            ui.add_space(10.0);
                            
                            // Performance history chart
                            ui.strong("Performance History");
                            ui.add_space(5.0);
                            if let Some(history) = self.role_performance_history.get(role_name) {
                                // This would show a small performance trend chart
                                ui.small(&format!("Data points: {}", history.len()));
                                ui.small("📈 Performance trending upward");
                            }
                        });
                    });
                }
                
                ui.add_space(15.0);
                
                // Configuration options
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.strong("⚙️ Configuration");
                        ui.add_space(10.0);
                        
                        let mut temp_config = config.clone();
                        let mut changed = false;
                        
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut temp_config.auto_activate, "Auto-activate");
                            if temp_config.auto_activate != config.auto_activate {
                                changed = true;
                            }
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Performance threshold:");
                            if ui.add(egui::Slider::new(&mut temp_config.performance_threshold, 0.0..=1.0)
                                .text("%")).changed() {
                                changed = true;
                            }
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Energy efficiency target:");
                            if ui.add(egui::Slider::new(&mut temp_config.energy_efficiency_target, 0.0..=1.0)
                                .text("%")).changed() {
                                changed = true;
                            }
                        });
                        
                        if changed {
                            self.role_configs.insert(role_name.clone(), temp_config);
                        }
                    });
                });
            }
        }
    }
    
    /// Filter roles based on current filter and search query
    fn filter_roles(&self) -> Vec<String> {
        let mut roles: Vec<String> = self.role_configs.keys().cloned().collect();
        
        // Apply filter
        roles.retain(|role| {
            let config = self.role_configs.get(role).unwrap();
            match self.role_filter {
                RoleFilter::All => true,
                RoleFilter::Active => config.enabled,
                RoleFilter::Available => !config.enabled,
                RoleFilter::Learning => config.role_type == "Learning",
                RoleFilter::Coordination => config.role_type == "Coordination",
                RoleFilter::Security => config.role_type == "Security",
                RoleFilter::Resource => config.role_type == "Resource",
            }
        });
        
        // Apply search query
        if !self.search_query.is_empty() {
            let query = self.search_query.to_lowercase();
            roles.retain(|role| role.to_lowercase().contains(&query));
        }
        
        roles.sort();
        roles
    }
    
    /// Toggle a biological role on/off
    fn toggle_role(&mut self, role_name: &str, enable: bool) {
        if let Some(config) = self.role_configs.get_mut(role_name) {
            config.enabled = enable;
            
            // Record adaptation event
            let event = AdaptationEvent {
                timestamp: std::time::Instant::now(),
                role: role_name.to_string(),
                event_type: if enable { AdaptationEventType::RoleActivated } else { AdaptationEventType::RoleDeactivated },
                effectiveness: 0.8, // Placeholder
                description: format!("Role {} {}", role_name, if enable { "activated" } else { "deactivated" }),
            };
            self.adaptation_events.push(event);
        }
        
        // TODO: Send command to backend to actually toggle the role
    }
    
    /// Trigger automatic adaptation
    fn trigger_auto_adaptation(&mut self) {
        // Implement auto-adaptation logic
        // This would analyze current network conditions and automatically
        // activate/deactivate roles for optimal performance
    }
    
    /// Get role performance (simulated)
    fn get_role_performance(&self, role_name: &str) -> f32 {
        match role_name {
            "YoungNode" => 0.78,
            "CasteNode" => 0.91,
            "DOSNode" => 0.85,
            "SyncPhaseNode" => 0.72,
            "HAVOCNode" => 0.95,
            _ => 0.65,
        }
    }
    
    /// Get emoji for role
    fn get_role_emoji(&self, role_name: &str) -> &'static str {
        match role_name {
            "YoungNode" => "🐦",
            "CasteNode" => "🐜",
            "ImitateNode" => "🦜",
            "HatchNode" => "🐢",
            "SyncPhaseNode" => "🐧",
            "HuddleNode" => "🐧",
            "DOSNode" => "🛡️",
            "HAVOCNode" => "🦟",
            "TrustNode" => "🤝",
            "ThermalNode" => "🌡️",
            _ => "🧬",
        }
    }
    
    /// Get role description
    fn get_role_description(&self, role_name: &str) -> &'static str {
        match role_name {
            "YoungNode" => "Learn from experienced network peers",
            "CasteNode" => "Specialized resource compartmentalization",
            "ImitateNode" => "Copy successful communication patterns",
            "HatchNode" => "Coordinate group lifecycle management",
            "SyncPhaseNode" => "Manage node lifecycle synchronization",
            "HuddleNode" => "Dynamic position rotation for load balancing",
            "DOSNode" => "Detect and prevent denial of service attacks",
            "HAVOCNode" => "Emergency resource reallocation",
            "TrustNode" => "Monitor and manage peer relationships",
            "ThermalNode" => "Monitor resource availability signatures",
            _ => "Specialized biological behavior",
        }
    }
}

impl UIPanel for BiologicalRolesPanel {
    fn show(&mut self, ui: &mut egui::Ui) {
        if self.show_role_details {
            self.show_role_details(ui);
        } else {
            self.show_role_overview(ui);
        }
        
        // Education controls
        ui.add_space(15.0);
        ui.collapsing("🎓 Learning Controls", |ui| {
            ui.horizontal(|ui| {
                ui.label("Explanation level:");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?}", self.tooltip_level))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.tooltip_level, TooltipLevel::Simple, "Simple");
                        ui.selectable_value(&mut self.tooltip_level, TooltipLevel::Detailed, "Detailed");
                        ui.selectable_value(&mut self.tooltip_level, TooltipLevel::Technical, "Technical");
                    });
            });
            
            ui.checkbox(&mut self.show_biological_context, "Show biological context");
        });
    }
    
    fn update(&mut self) {
        // Update role performance metrics
        // This would normally get data from the backend
    }
    
    fn title(&self) -> &str {
        "Biological Roles"
    }
    
    fn is_visible(&self) -> bool {
        true
    }
    
    fn set_visible(&mut self, visible: bool) {
        // Update visibility
    }
}