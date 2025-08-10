//! Network Status Panel - displays current network connectivity and peer information
//! 
//! This panel provides real-time information about network status, peer connections,
//! and overall network health with biological metaphors.

use eframe::egui;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use crate::state::AppState;
use super::{UIPanel, chart_utils::*, biological_widgets::*};

/// Network status panel showing connectivity and peer information
pub struct NetworkStatusPanel {
    app_state: Arc<RwLock<AppState>>,
    
    // Connection metrics
    connection_history: TimeSeries,
    peer_count_history: TimeSeries,
    quality_history: TimeSeries,
    
    // UI state
    show_advanced_metrics: bool,
    auto_refresh: bool,
    last_update: Instant,
    
    // Network visualization
    network_view_mode: NetworkViewMode,
}

/// Network visualization modes
#[derive(Debug, Clone, PartialEq)]
enum NetworkViewMode {
    Summary,
    Timeline,
    PeerMap,
    BiologicalView,
}

impl NetworkStatusPanel {
    pub fn new(app_state: Arc<RwLock<AppState>>) -> Self {
        Self {
            app_state,
            connection_history: TimeSeries::new("Connection Quality".to_string(), BiologicalColors::PRIMARY),
            peer_count_history: TimeSeries::new("Peer Count".to_string(), BiologicalColors::SECONDARY),
            quality_history: TimeSeries::new("Network Quality".to_string(), BiologicalColors::ACCENT),
            show_advanced_metrics: false,
            auto_refresh: true,
            last_update: Instant::now(),
            network_view_mode: NetworkViewMode::Summary,
        }
    }
    
    /// Update network metrics from current state
    fn update_metrics(&mut self) {
        // This would normally read from the app state
        // For now, we'll use simulated data
        let now = self.last_update.elapsed().as_secs_f64();
        
        // Simulate network metrics
        let connection_quality = 0.8 + 0.2 * (now * 0.1).sin();
        let peer_count = 5.0 + 3.0 * (now * 0.05).sin();
        let network_quality = 0.75 + 0.25 * (now * 0.08).cos();
        
        self.connection_history.add_point(now, connection_quality);
        self.peer_count_history.add_point(now, peer_count);
        self.quality_history.add_point(now, network_quality);
    }
    
    /// Show network status summary
    fn show_network_summary(&mut self, ui: &mut egui::Ui) {
        ui.heading("🌐 Network Status");
        ui.add_space(10.0);
        
        // Main status indicators
        ui.horizontal(|ui| {
            biological_metric_card(ui, "Connection", "Connected", "🟢", Some(2.5));
            ui.add_space(10.0);
            biological_metric_card(ui, "Peers", "8", "👥", Some(12.5));
            ui.add_space(10.0);
            biological_metric_card(ui, "Quality", "85%", "📊", Some(-1.2));
        });
        
        ui.add_space(15.0);
        
        // Biological context section
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("🧬");
                    ui.strong("Biological Network Behavior");
                });
                ui.add_space(5.0);
                
                ui.label("Your node is behaving like a healthy organism in an ecosystem:");
                ui.add_space(3.0);
                
                ui.horizontal(|ui| {
                    ui.label("🐦");
                    ui.label("Learning from 5 experienced neighbors (Young Node behavior)");
                });
                
                ui.horizontal(|ui| {
                    ui.label("🤝");
                    ui.label("Maintaining trust relationships with 8 peers");
                });
                
                ui.horizontal(|ui| {
                    ui.label("🔄");
                    ui.label("Adapting routing patterns based on network conditions");
                });
            });
        });
    }
    
    /// Show network timeline view
    fn show_network_timeline(&mut self, ui: &mut egui::Ui) {
        ui.heading("📈 Network Timeline");
        ui.add_space(10.0);
        
        // Chart controls
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.auto_refresh, "Auto-refresh");
            ui.separator();
            if ui.button("📊 Export Data").clicked() {
                // TODO: Export network data
            }
        });
        
        ui.add_space(10.0);
        
        // Connection quality chart
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.strong("Connection Quality Over Time");
                ui.add_space(5.0);
                
                let chart_size = egui::vec2(ui.available_width(), 200.0);
                show_line_chart(
                    ui,
                    chart_size,
                    &[&self.connection_history],
                    "Connection Quality",
                    "Quality %",
                );
            });
        });
        
        ui.add_space(10.0);
        
        // Peer count chart
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.strong("Peer Count Evolution");
                ui.add_space(5.0);
                
                let chart_size = egui::vec2(ui.available_width(), 150.0);
                show_line_chart(
                    ui,
                    chart_size,
                    &[&self.peer_count_history],
                    "Connected Peers",
                    "Count",
                );
            });
        });
    }
    
    /// Show peer map visualization
    fn show_peer_map(&mut self, ui: &mut egui::Ui) {
        ui.heading("🗺️ Peer Network Map");
        ui.add_space(10.0);
        
        // Network topology visualization placeholder
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(50.0);
                ui.heading("🕸️");
                ui.add_space(10.0);
                ui.label("Interactive peer network visualization");
                ui.small("Shows connections between peers with biological styling");
                ui.add_space(20.0);
                
                // Simple network diagram
                self.show_simple_network_diagram(ui);
                
                ui.add_space(50.0);
            });
        });
        
        ui.add_space(10.0);
        
        // Peer details
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.strong("Connected Peers");
                ui.separator();
                
                // Sample peer list
                for i in 0..5 {
                    ui.horizontal(|ui| {
                        ui.label("👤");
                        ui.label(format!("Peer {}", i + 1));
                        ui.separator();
                        role_badge(ui, "YoungNode", "🐦", true, 0.85);
                        ui.separator();
                        ui.small("Trust: 85%");
                        ui.separator();
                        ui.small("Latency: 45ms");
                    });
                }
            });
        });
    }
    
    /// Show biological network view
    fn show_biological_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("🧬 Biological Network View");
        ui.add_space(10.0);
        
        ui.label("Your network behaves like a living ecosystem:");
        ui.add_space(10.0);
        
        // Ecosystem health indicators
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.strong("🌱 Network Ecosystem Health");
                ui.add_space(5.0);
                
                biological_progress_bar(ui, 0.85, "🧬 Biodiversity (Role Variety)");
                biological_progress_bar(ui, 0.78, "🤝 Social Cohesion (Trust Network)");
                biological_progress_bar(ui, 0.92, "⚡ Energy Efficiency (Resource Usage)");
                biological_progress_bar(ui, 0.67, "🧠 Collective Intelligence (Adaptation)");
                biological_progress_bar(ui, 0.83, "🛡️ Immune Response (Security)");
            });
        });
        
        ui.add_space(15.0);
        
        // Biological behaviors observed
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.strong("🔬 Observed Biological Behaviors");
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("🐦");
                    ui.vertical(|ui| {
                        ui.strong("Young Node Learning");
                        ui.small("3 nodes are learning routing patterns from experienced peers");
                        ui.small("Learning efficiency: 78% (above average)");
                    });
                });
                
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("🐜");
                    ui.vertical(|ui| {
                        ui.strong("Caste Specialization");
                        ui.small("Resource compartments automatically adapting to workload");
                        ui.small("Current specialization: Training 35%, Inference 25%, Storage 20%");
                    });
                });
                
                ui.separator();
                
                ui.horizontal(|ui| {
                    ui.label("🦌");
                    ui.vertical(|ui| {
                        ui.strong("Migration Patterns");
                        ui.small("Optimal routes being reinforced through repeated successful use");
                        ui.small("Route efficiency improved 15% in the last hour");
                    });
                });
            });
        });
    }
    
    /// Simple network diagram visualization
    fn show_simple_network_diagram(&self, ui: &mut egui::Ui) {
        let (response, painter) = ui.allocate_painter(egui::vec2(300.0, 200.0), egui::Sense::hover());
        let rect = response.rect;
        let center = rect.center();
        
        // Draw central node (this node)
        painter.circle_filled(center, 20.0, BiologicalColors::PRIMARY);
        painter.text(
            center,
            egui::Align2::CENTER_CENTER,
            "You",
            egui::FontId::proportional(12.0),
            egui::Color32::WHITE,
        );
        
        // Draw connected peers in a circle
        let peer_count = 8;
        let radius = 80.0;
        
        for i in 0..peer_count {
            let angle = (i as f32 * 2.0 * std::f32::consts::PI) / peer_count as f32;
            let peer_pos = center + egui::vec2(
                radius * angle.cos(),
                radius * angle.sin(),
            );
            
            // Draw connection line
            painter.line_segment(
                [center, peer_pos],
                egui::Stroke::new(2.0, BiologicalColors::SECONDARY),
            );
            
            // Draw peer node
            painter.circle_filled(peer_pos, 15.0, BiologicalColors::ACCENT);
            painter.text(
                peer_pos,
                egui::Align2::CENTER_CENTER,
                &format!("P{}", i + 1),
                egui::FontId::proportional(10.0),
                egui::Color32::BLACK,
            );
        }
    }
}

impl UIPanel for NetworkStatusPanel {
    fn show(&mut self, ui: &mut egui::Ui) {
        // Update metrics if auto-refresh is enabled
        if self.auto_refresh && self.last_update.elapsed().as_secs() >= 5 {
            self.update_metrics();
            self.last_update = Instant::now();
        }
        
        // View mode selector
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.network_view_mode, NetworkViewMode::Summary, "📊 Summary");
            ui.selectable_value(&mut self.network_view_mode, NetworkViewMode::Timeline, "📈 Timeline");
            ui.selectable_value(&mut self.network_view_mode, NetworkViewMode::PeerMap, "🗺️ Peer Map");
            ui.selectable_value(&mut self.network_view_mode, NetworkViewMode::BiologicalView, "🧬 Biological");
        });
        
        ui.add_space(10.0);
        
        // Show content based on selected view mode
        match self.network_view_mode {
            NetworkViewMode::Summary => self.show_network_summary(ui),
            NetworkViewMode::Timeline => self.show_network_timeline(ui),
            NetworkViewMode::PeerMap => self.show_peer_map(ui),
            NetworkViewMode::BiologicalView => self.show_biological_view(ui),
        }
        
        ui.add_space(10.0);
        
        // Advanced options
        ui.collapsing("⚙️ Advanced Options", |ui| {
            ui.checkbox(&mut self.show_advanced_metrics, "Show advanced metrics");
            
            if self.show_advanced_metrics {
                ui.separator();
                ui.label("🔧 Advanced network diagnostics");
                ui.small("• Protocol statistics");
                ui.small("• Bandwidth analysis");
                ui.small("• Connection quality metrics");
                ui.small("• Peer reputation details");
            }
        });
    }
    
    fn update(&mut self) {
        self.update_metrics();
    }
    
    fn title(&self) -> &str {
        "Network Status"
    }
    
    fn is_visible(&self) -> bool {
        true // This would be controlled by panel state
    }
    
    fn set_visible(&mut self, visible: bool) {
        // Update panel visibility state
    }
}