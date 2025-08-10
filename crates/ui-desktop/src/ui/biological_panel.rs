//! Biological Roles Panel - Displays active biological behaviors and role management
//! 
//! This panel shows the current biological roles active on the node,
//! their performance metrics, adaptation status, and allows role configuration.

use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc};

use crate::state::AppState;
use crate::commands::BiologicalRole;

pub struct BiologicalRolesPanel {
    app_state: Arc<RwLock<AppState>>,
    active_roles: Vec<BiologicalRoleDisplay>,
    available_roles: Vec<String>,
    selected_role: Option<String>,
    show_role_details: bool,
    role_filter: String,
    sort_by: RoleSortBy,
    show_performance_charts: bool,
    learning_mode_enabled: bool,
}

#[derive(Debug, Clone)]
struct BiologicalRoleDisplay {
    role_type: String,
    biological_inspiration: String,
    description: String,
    active: bool,
    performance_score: f64,
    energy_efficiency: f64,
    adaptation_rate: f64,
    specialization_level: f64,
    activation_time: DateTime<Utc>,
    activity_history: Vec<(DateTime<Utc>, f64)>,
}

#[derive(Debug, Clone)]
enum RoleSortBy {
    Name,
    Performance,
    EnergyEfficiency,
    AdaptationRate,
    ActivationTime,
}

impl BiologicalRolesPanel {
    pub fn new(app_state: Arc<RwLock<AppState>>) -> Self {
        Self {
            app_state,
            active_roles: Vec::new(),
            available_roles: Self::get_all_available_roles(),
            selected_role: None,
            show_role_details: false,
            role_filter: String::new(),
            sort_by: RoleSortBy::Performance,
            show_performance_charts: false,
            learning_mode_enabled: true,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        // Header
        ui.horizontal(|ui| {
            ui.heading("🐜 Biological Roles");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_performance_charts, "Charts");
                ui.checkbox(&mut self.learning_mode_enabled, "Learning Mode");
                if ui.button("🔄 Refresh").clicked() {
                    self.refresh_roles();
                }
            });
        });

        ui.separator();

        // Role controls
        self.show_role_controls(ui);

        ui.add_space(10.0);

        // Active roles section
        self.show_active_roles(ui);

        ui.add_space(10.0);

        // Role management section
        self.show_role_management(ui);

        if self.show_role_details && self.selected_role.is_some() {
            ui.add_space(10.0);
            self.show_role_details(ui);
        }
    }

    fn show_role_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Search:");
            ui.text_edit_singleline(&mut self.role_filter);
            
            ui.separator();
            
            ui.label("Sort by:");
            egui::ComboBox::from_id_source("role_sort")
                .selected_text(format!("{:?}", self.sort_by))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.sort_by, RoleSortBy::Name, "Name");
                    ui.selectable_value(&mut self.sort_by, RoleSortBy::Performance, "Performance");
                    ui.selectable_value(&mut self.sort_by, RoleSortBy::EnergyEfficiency, "Energy Efficiency");
                    ui.selectable_value(&mut self.sort_by, RoleSortBy::AdaptationRate, "Adaptation Rate");
                    ui.selectable_value(&mut self.sort_by, RoleSortBy::ActivationTime, "Activation Time");
                });
        });
    }

    fn show_active_roles(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Active Biological Roles").strong());
                ui.add_space(5.0);

                if self.active_roles.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.label("🌱 No biological roles active");
                        ui.small("Enable auto-assignment or manually activate roles below");
                    });
                } else {
                    // Role overview cards
                    ui.horizontal_wrapped(|ui| {
                        for role in &mut self.active_roles {
                            if self.matches_filter(&role.role_type) {
                                self.show_role_card(ui, role);
                            }
                        }
                    });

                    ui.add_space(10.0);

                    // Adaptation status
                    self.show_adaptation_status(ui);
                }
            });
        });
    }

    fn show_role_card(&mut self, ui: &mut egui::Ui, role: &mut BiologicalRoleDisplay) {
        let card_size = egui::vec2(280.0, 160.0);
        ui.allocate_ui(card_size, |ui| {
            egui::Frame::group(ui.style())
                .fill(if role.active { egui::Color32::from_rgb(240, 255, 240) } else { egui::Color32::WHITE })
                .show(ui, |ui| {
                    ui.set_max_width(card_size.x);
                    
                    // Role header
                    ui.horizontal(|ui| {
                        ui.strong(&role.role_type);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let status_color = if role.active { egui::Color32::GREEN } else { egui::Color32::GRAY };
                            ui.colored_label(status_color, if role.active { "●" } else { "○" });
                        });
                    });

                    ui.separator();

                    // Biological inspiration
                    ui.label(egui::RichText::new(&role.biological_inspiration).italics().size(11.0));
                    
                    ui.add_space(5.0);

                    // Performance metrics
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.small("Performance");
                            ui.add(egui::ProgressBar::new(role.performance_score).text(format!("{:.1}%", role.performance_score * 100.0)));
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.small("Energy Efficiency");
                            ui.add(egui::ProgressBar::new(role.energy_efficiency).text(format!("{:.1}%", role.energy_efficiency * 100.0)));
                        });
                    });

                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.small("Adaptation Rate");
                            ui.add(egui::ProgressBar::new(role.adaptation_rate).text(format!("{:.1}%", role.adaptation_rate * 100.0)));
                        });
                    });

                    // Action buttons
                    ui.horizontal(|ui| {
                        if ui.small_button("📊 Details").clicked() {
                            self.selected_role = Some(role.role_type.clone());
                            self.show_role_details = true;
                        }
                        
                        let button_text = if role.active { "⏸ Pause" } else { "▶ Activate" };
                        if ui.small_button(button_text).clicked() {
                            role.active = !role.active;
                            // TODO: Send command to activate/deactivate role
                        }
                        
                        if ui.small_button("⚙ Configure").clicked() {
                            // TODO: Open role configuration dialog
                        }
                    });
                });
        });
    }

    fn show_adaptation_status(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("🧠 Adaptation Status", |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.strong("Learning Progress");
                    ui.horizontal(|ui| {
                        ui.label("Young Node Learning:");
                        ui.add(egui::ProgressBar::new(0.67).text("67%"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Pattern Recognition:");
                        ui.add(egui::ProgressBar::new(0.84).text("84%"));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Network Integration:");
                        ui.add(egui::ProgressBar::new(0.91).text("91%"));
                    });
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.strong("Adaptation Metrics");
                    ui.label("🎯 Role Transitions: 7 today");
                    ui.label("📈 Performance Improvement: +12%");
                    ui.label("🔄 Learning Cycles: 156 completed");
                    ui.label("🌱 Adaptation Score: 8.4/10");
                });
            });
        });
    }

    fn show_role_management(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Role Management").strong());
                ui.add_space(5.0);

                // Auto-assignment settings
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.learning_mode_enabled, "Auto Role Assignment");
                    ui.label("🧠 Automatically adapt roles based on network conditions");
                });

                ui.add_space(10.0);

                // Available roles
                ui.label("Available Biological Roles:");
                ui.separator();

                let filtered_roles: Vec<_> = self.available_roles.iter()
                    .filter(|role| self.matches_filter(role))
                    .collect();

                egui::ScrollArea::vertical()
                    .max_height(200.0)
                    .show(ui, |ui| {
                        for role_name in filtered_roles {
                            ui.horizontal(|ui| {
                                let is_active = self.active_roles.iter().any(|r| &r.role_type == role_name);
                                
                                // Role icon based on biological type
                                let icon = self.get_role_icon(role_name);
                                ui.label(icon);
                                
                                ui.label(role_name);
                                
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    if is_active {
                                        ui.colored_label(egui::Color32::GREEN, "Active");
                                        if ui.button("Deactivate").clicked() {
                                            self.deactivate_role(role_name);
                                        }
                                    } else {
                                        if ui.button("Activate").clicked() {
                                            self.activate_role(role_name);
                                        }
                                    }
                                    
                                    if ui.small_button("ℹ").on_hover_text("Role information").clicked() {
                                        self.selected_role = Some(role_name.clone());
                                        self.show_role_details = true;
                                    }
                                });
                            });
                            ui.separator();
                        }
                    });
            });
        });
    }

    fn show_role_details(&mut self, ui: &mut egui::Ui) {
        if let Some(role_name) = &self.selected_role.clone() {
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.heading(format!("🔬 {} Details", role_name));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.button("✖").clicked() {
                                self.show_role_details = false;
                                self.selected_role = None;
                            }
                        });
                    });

                    ui.separator();

                    // Role information
                    self.show_detailed_role_info(ui, role_name);

                    if self.show_performance_charts {
                        ui.add_space(10.0);
                        self.show_role_performance_chart(ui, role_name);
                    }
                });
            });
        }
    }

    fn show_detailed_role_info(&mut self, ui: &mut egui::Ui, role_name: &str) {
        if let Some(role_info) = self.get_role_detailed_info(role_name) {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.strong("Biological Inspiration:");
                    ui.label(role_info.biological_inspiration);
                    
                    ui.add_space(5.0);
                    
                    ui.strong("Technical Description:");
                    ui.label(role_info.description);
                    
                    ui.add_space(5.0);
                    
                    ui.strong("Key Features:");
                    for feature in role_info.key_features {
                        ui.label(format!("• {}", feature));
                    }
                });

                ui.separator();

                ui.vertical(|ui| {
                    ui.strong("Performance Metrics:");
                    
                    ui.horizontal(|ui| {
                        ui.label("Specialization Level:");
                        ui.add(egui::ProgressBar::new(role_info.specialization_level)
                            .text(format!("{:.1}%", role_info.specialization_level * 100.0)));
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Energy Efficiency:");
                        ui.add(egui::ProgressBar::new(role_info.energy_efficiency)
                            .text(format!("{:.1}%", role_info.energy_efficiency * 100.0)));
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Adaptation Rate:");
                        ui.add(egui::ProgressBar::new(role_info.adaptation_rate)
                            .text(format!("{:.1}%", role_info.adaptation_rate * 100.0)));
                    });

                    ui.add_space(10.0);

                    ui.strong("Related Concepts:");
                    for concept in role_info.related_concepts {
                        if ui.small_button(&concept).clicked() {
                            // TODO: Show tooltip or navigate to concept
                        }
                    }
                });
            });
        }
    }

    fn show_role_performance_chart(&mut self, ui: &mut egui::Ui, _role_name: &str) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Performance History").strong());
                
                // Simple chart placeholder
                let desired_size = egui::vec2(ui.available_width(), 150.0);
                let (rect, _response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
                
                if ui.is_rect_visible(rect) {
                    let painter = ui.painter();
                    
                    // Draw chart background
                    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(245));
                    painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::GRAY));
                    
                    // Draw sample performance line
                    let mut points = Vec::new();
                    for i in 0..20 {
                        let x = rect.min.x + (i as f32 / 19.0) * rect.width();
                        let y = rect.min.y + rect.height() * (0.3 + 0.4 * (i as f32 * 0.3).sin());
                        points.push(egui::pos2(x, y));
                    }
                    
                    if points.len() > 1 {
                        painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, egui::Color32::from_rgb(34, 139, 34))));
                    }
                }
            });
        });
    }

    fn matches_filter(&self, role_name: &str) -> bool {
        if self.role_filter.is_empty() {
            return true;
        }
        role_name.to_lowercase().contains(&self.role_filter.to_lowercase())
    }

    fn get_role_icon(&self, role_name: &str) -> &'static str {
        match role_name {
            "YoungNode" => "🐦",
            "CasteNode" => "🐜", 
            "ImitateNode" => "🦜",
            "HatchNode" => "🐢",
            "SyncPhaseNode" => "🐧",
            "HuddleNode" => "🐧",
            "MigrationNode" => "🦌",
            "AddressNode" => "🗺️",
            "TunnelNode" => "🕳️",
            "SignNode" => "🪧",
            "DOSNode" => "🛡️",
            "InvestigationNode" => "🔍",
            "CasualtyNode" => "📋",
            "HAVOCNode" => "🦟",
            "StepUpNode" => "⬆️",
            "StepDownNode" => "⬇️",
            "ThermalNode" => "🌡️",
            "FriendshipNode" => "🤝",
            "BuddyNode" => "👫",
            "TrustNode" => "🤝",
            _ => "🧬",
        }
    }

    fn get_all_available_roles() -> Vec<String> {
        vec![
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
            // Additional roles would be listed here...
        ]
    }

    fn get_role_detailed_info(&self, role_name: &str) -> Option<DetailedRoleInfo> {
        match role_name {
            "YoungNode" => Some(DetailedRoleInfo {
                biological_inspiration: "Young crows learn hunting techniques and territorial navigation by observing experienced adults within their social groups".to_string(),
                description: "New nodes learn optimal routing paths and resource allocation strategies from up to 100 neighboring experienced nodes".to_string(),
                key_features: vec![
                    "60-80% reduction in initialization overhead".to_string(),
                    "40-70% improvement in path discovery time".to_string(),
                    "Continuous learning from network experience".to_string(),
                ],
                specialization_level: 0.7,
                energy_efficiency: 0.8,
                adaptation_rate: 0.9,
                related_concepts: vec!["Swarm Learning".to_string(), "Peer Discovery".to_string()],
            }),
            "CasteNode" => Some(DetailedRoleInfo {
                biological_inspiration: "Ant colonies achieve remarkable efficiency through specialized castes (workers, soldiers, nurses) that perform distinct functions".to_string(),
                description: "Compartmentalizes single nodes into specialized functional units for training, inference, storage, communication, and security".to_string(),
                key_features: vec![
                    "85-95% resource utilization efficiency".to_string(),
                    "Dynamic compartment scaling".to_string(),
                    "Specialized processing optimization".to_string(),
                ],
                specialization_level: 0.95,
                energy_efficiency: 0.85,
                adaptation_rate: 0.6,
                related_concepts: vec!["Resource Allocation".to_string(), "Specialization".to_string()],
            }),
            "HAVOCNode" => Some(DetailedRoleInfo {
                biological_inspiration: "Disease-vector mosquitoes rapidly adapt behavior and resource allocation to environmental changes".to_string(),
                description: "Emergency resource reallocation preventing cascading failures through rapid behavioral adaptation".to_string(),
                key_features: vec![
                    "Prevents network death during critical shortages".to_string(),
                    "Rapid resource reallocation capabilities".to_string(),
                    "Crisis management coordination".to_string(),
                ],
                specialization_level: 0.9,
                energy_efficiency: 0.7,
                adaptation_rate: 0.95,
                related_concepts: vec!["Crisis Management".to_string(), "Emergency Response".to_string()],
            }),
            _ => None,
        }
    }

    fn activate_role(&mut self, role_name: &str) {
        // TODO: Send command to activate role
        println!("Activating role: {}", role_name);
    }

    fn deactivate_role(&mut self, role_name: &str) {
        // TODO: Send command to deactivate role  
        println!("Deactivating role: {}", role_name);
    }

    fn refresh_roles(&mut self) {
        // TODO: Fetch current roles from node
        // For now, populate with sample data
        self.active_roles = vec![
            BiologicalRoleDisplay {
                role_type: "YoungNode".to_string(),
                biological_inspiration: "Crow Culture Learning".to_string(),
                description: "Learning optimal paths from experienced peers".to_string(),
                active: true,
                performance_score: 0.78,
                energy_efficiency: 0.82,
                adaptation_rate: 0.91,
                specialization_level: 0.67,
                activation_time: Utc::now(),
                activity_history: Vec::new(),
            },
            BiologicalRoleDisplay {
                role_type: "TrustNode".to_string(),
                biological_inspiration: "Primate Social Bonding".to_string(),
                description: "Monitoring peer relationships and trust levels".to_string(),
                active: true,
                performance_score: 0.85,
                energy_efficiency: 0.75,
                adaptation_rate: 0.63,
                specialization_level: 0.89,
                activation_time: Utc::now(),
                activity_history: Vec::new(),
            },
        ];
    }
}

#[derive(Debug, Clone)]
struct DetailedRoleInfo {
    biological_inspiration: String,
    description: String,
    key_features: Vec<String>,
    specialization_level: f64,
    energy_efficiency: f64,
    adaptation_rate: f64,
    related_concepts: Vec<String>,
}