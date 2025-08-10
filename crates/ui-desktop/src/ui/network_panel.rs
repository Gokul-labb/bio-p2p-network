//! Network Status Panel - Displays network connectivity and peer information
//! 
//! This panel shows the current network status, connection quality,
//! peer count, and network-level biological behaviors.

use eframe::egui;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};

use crate::state::AppState;
use crate::commands::NetworkStatus;

pub struct NetworkStatusPanel {
    app_state: Arc<RwLock<AppState>>,
    last_update: Option<DateTime<Utc>>,
    network_history: Vec<(DateTime<Utc>, NetworkMetrics)>,
    show_advanced_metrics: bool,
    selected_metric: String,
}

#[derive(Debug, Clone)]
struct NetworkMetrics {
    peer_count: usize,
    bytes_sent: u64,
    bytes_received: u64,
    connection_quality: f64,
}

impl NetworkStatusPanel {
    pub fn new(app_state: Arc<RwLock<AppState>>) -> Self {
        Self {
            app_state,
            last_update: None,
            network_history: Vec::new(),
            show_advanced_metrics: false,
            selected_metric: "peer_count".to_string(),
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui) {
        // Header
        ui.horizontal(|ui| {
            ui.heading("🌐 Network Status");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.checkbox(&mut self.show_advanced_metrics, "Advanced");
                if ui.button("🔄 Refresh").clicked() {
                    self.refresh_data();
                }
            });
        });

        ui.separator();

        // Connection status section
        self.show_connection_status(ui);
        
        ui.add_space(10.0);
        
        // Peer statistics
        self.show_peer_statistics(ui);
        
        ui.add_space(10.0);
        
        // Network performance metrics
        self.show_performance_metrics(ui);
        
        if self.show_advanced_metrics {
            ui.add_space(10.0);
            self.show_advanced_section(ui);
        }

        ui.add_space(10.0);

        // Network history chart
        self.show_network_chart(ui);
    }

    fn show_connection_status(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.set_min_height(80.0);
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Connection Status").strong());
                ui.add_space(5.0);
                
                ui.horizontal(|ui| {
                    // Connection indicator
                    let connected = true; // TODO: Get from actual network status
                    if connected {
                        ui.colored_label(egui::Color32::GREEN, "● Connected");
                        ui.label("to Bio P2P Network");
                    } else {
                        ui.colored_label(egui::Color32::RED, "● Disconnected");
                        ui.label("from network");
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Node ID:");
                    ui.monospace("12D3KooW...ABC123"); // Truncated node ID
                    if ui.small_button("📋").on_hover_text("Copy full node ID").clicked() {
                        // TODO: Copy to clipboard
                    }
                });

                ui.horizontal(|ui| {
                    ui.label("Uptime:");
                    ui.label("2h 34m 12s");
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label("Quality:");
                        ui.colored_label(egui::Color32::GREEN, "Excellent");
                    });
                });
            });
        });
    }

    fn show_peer_statistics(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Peer Statistics").strong());
                ui.add_space(5.0);
                
                ui.horizontal(|ui| {
                    // Connected peers
                    ui.vertical(|ui| {
                        ui.label("Connected Peers");
                        ui.label(egui::RichText::new("23").size(24.0).strong());
                        ui.small("🔗 Active connections");
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label("Discovered Peers");
                        ui.label(egui::RichText::new("157").size(24.0).strong());
                        ui.small("👁 In routing table");
                    });

                    ui.separator();

                    ui.vertical(|ui| {
                        ui.label("Biological Roles");
                        ui.label(egui::RichText::new("12").size(24.0).strong());
                        ui.small("🐜 Active in network");
                    });
                });

                ui.add_space(5.0);

                // Peer diversity indicator
                ui.horizontal(|ui| {
                    ui.label("Network Diversity:");
                    ui.add(egui::ProgressBar::new(0.78).text("High"));
                    ui.small("🌱 Biological diversity enhances resilience");
                });
            });
        });
    }

    fn show_performance_metrics(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(egui::RichText::new("Performance Metrics").strong());
                ui.add_space(5.0);
                
                ui.horizontal(|ui| {
                    // Data transfer
                    ui.vertical(|ui| {
                        ui.small("Data Sent");
                        ui.label("1.2 MB");
                    });
                    
                    ui.separator();
                    
                    ui.vertical(|ui| {
                        ui.small("Data Received");
                        ui.label("8.7 MB");
                    });
                    
                    ui.separator();
                    
                    ui.vertical(|ui| {
                        ui.small("Avg Latency");
                        ui.label("45 ms");
                    });
                    
                    ui.separator();
                    
                    ui.vertical(|ui| {
                        ui.small("Success Rate");
                        ui.colored_label(egui::Color32::GREEN, "99.2%");
                    });
                });

                ui.add_space(5.0);

                // Biological efficiency indicators
                ui.horizontal(|ui| {
                    ui.label("🐜 Swarm Efficiency:");
                    ui.add(egui::ProgressBar::new(0.85).text("85%"));
                    
                    ui.label("🧠 Learning Rate:");
                    ui.add(egui::ProgressBar::new(0.72).text("72%"));
                });
            });
        });
    }

    fn show_advanced_section(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("🔬 Advanced Network Metrics", |ui| {
            // Protocol statistics
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.strong("Protocol Stats");
                    ui.monospace("Gossipsub: 156 msgs/min");
                    ui.monospace("Kademlia: 23 queries/min");
                    ui.monospace("Identify: 12 exchanges/min");
                });
                
                ui.separator();
                
                ui.vertical(|ui| {
                    ui.strong("Biological Protocols");
                    ui.monospace("Migration: 8 routes/min");
                    ui.monospace("Young Node: 15 learns/min");
                    ui.monospace("Trust: 34 updates/min");
                });
            });

            ui.add_space(5.0);

            // Network topology metrics
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.strong("Topology");
                    ui.label("Clustering Coefficient: 0.67");
                    ui.label("Average Path Length: 3.2");
                    ui.label("Network Diameter: 7 hops");
                });
                
                ui.separator();
                
                ui.vertical(|ui| {
                    ui.strong("Biological Topology");
                    ui.label("Alpha Nodes: 3 clusters");
                    ui.label("Bravo Nodes: 1 cluster");
                    ui.label("Super Nodes: 0");
                });
            });
        });
    }

    fn show_network_chart(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Network Activity").strong());
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        egui::ComboBox::from_id_source("metric_selector")
                            .selected_text(&self.selected_metric)
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.selected_metric, "peer_count".to_string(), "Peer Count");
                                ui.selectable_value(&mut self.selected_metric, "bandwidth".to_string(), "Bandwidth");
                                ui.selectable_value(&mut self.selected_metric, "quality".to_string(), "Connection Quality");
                                ui.selectable_value(&mut self.selected_metric, "biological".to_string(), "Biological Activity");
                            });
                    });
                });
                
                ui.add_space(5.0);
                
                // Simple chart placeholder
                let desired_size = egui::vec2(ui.available_width(), 200.0);
                let (rect, response) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
                
                if ui.is_rect_visible(rect) {
                    let painter = ui.painter();
                    
                    // Draw chart background
                    painter.rect_filled(rect, 4.0, egui::Color32::from_gray(240));
                    
                    // Draw chart border
                    painter.rect_stroke(rect, 4.0, egui::Stroke::new(1.0, egui::Color32::GRAY));
                    
                    // Draw sample data line
                    if !self.network_history.is_empty() {
                        let points = self.generate_chart_points(&rect);
                        if points.len() > 1 {
                            painter.add(egui::Shape::line(points, egui::Stroke::new(2.0, egui::Color32::from_rgb(34, 139, 34))));
                        }
                    }
                    
                    // Draw axis labels
                    painter.text(
                        rect.left_bottom() + egui::vec2(5.0, -15.0),
                        egui::Align2::LEFT_BOTTOM,
                        "Time",
                        egui::FontId::proportional(12.0),
                        egui::Color32::GRAY,
                    );
                    
                    let metric_label = match self.selected_metric.as_str() {
                        "peer_count" => "Peers",
                        "bandwidth" => "MB/s",
                        "quality" => "Quality",
                        "biological" => "Activity",
                        _ => "Value",
                    };
                    
                    painter.text(
                        rect.left_top() + egui::vec2(5.0, 15.0),
                        egui::Align2::LEFT_TOP,
                        metric_label,
                        egui::FontId::proportional(12.0),
                        egui::Color32::GRAY,
                    );
                }
                
                // Chart controls
                ui.horizontal(|ui| {
                    if ui.button("1H").clicked() {
                        self.set_chart_timerange(Duration::hours(1));
                    }
                    if ui.button("6H").clicked() {
                        self.set_chart_timerange(Duration::hours(6));
                    }
                    if ui.button("24H").clicked() {
                        self.set_chart_timerange(Duration::hours(24));
                    }
                    if ui.button("7D").clicked() {
                        self.set_chart_timerange(Duration::days(7));
                    }
                    
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.small("🔄 Updates every 5s");
                    });
                });
            });
        });
    }

    fn generate_chart_points(&self, rect: &egui::Rect) -> Vec<egui::Pos2> {
        // Generate sample data points for demonstration
        let mut points = Vec::new();
        let width = rect.width();
        let height = rect.height();
        
        for i in 0..50 {
            let x = rect.min.x + (i as f32 / 49.0) * width;
            let y_factor = match self.selected_metric.as_str() {
                "peer_count" => (20.0 + (i as f32 * 0.1).sin() * 5.0) / 30.0,
                "bandwidth" => (0.5 + (i as f32 * 0.2).cos() * 0.3).max(0.0).min(1.0),
                "quality" => (0.8 + (i as f32 * 0.15).sin() * 0.15).max(0.0).min(1.0),
                "biological" => (0.6 + (i as f32 * 0.3).sin() * 0.3).max(0.0).min(1.0),
                _ => 0.5,
            };
            let y = rect.max.y - (y_factor * height);
            
            points.push(egui::pos2(x, y));
        }
        
        points
    }

    fn refresh_data(&mut self) {
        self.last_update = Some(Utc::now());
        
        // Add current metrics to history
        let metrics = NetworkMetrics {
            peer_count: 23, // TODO: Get from actual network status
            bytes_sent: 1200000,
            bytes_received: 8700000,
            connection_quality: 0.92,
        };
        
        self.network_history.push((Utc::now(), metrics));
        
        // Limit history size
        if self.network_history.len() > 1000 {
            self.network_history.remove(0);
        }
    }

    fn set_chart_timerange(&mut self, duration: Duration) {
        let cutoff = Utc::now() - duration;
        self.network_history.retain(|(time, _)| *time > cutoff);
    }
}