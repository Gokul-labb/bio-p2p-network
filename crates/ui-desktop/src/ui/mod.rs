//! UI module containing all the panel implementations
//! 
//! This module provides the individual UI panels that display different
//! aspects of the Bio P2P network, from network status to biological roles.

pub mod network_status;
pub mod biological_roles;
pub mod peer_management;
pub mod resource_monitor;
pub mod security_dashboard;
pub mod package_processing;
pub mod network_topology;
pub mod log_viewer;

// Re-export all panels for easy access
pub use network_status::NetworkStatusPanel;
pub use biological_roles::BiologicalRolesPanel;
pub use peer_management::PeerManagementPanel;
pub use resource_monitor::ResourceMonitorPanel;
pub use security_dashboard::SecurityDashboardPanel;
pub use package_processing::PackageProcessingPanel;
pub use network_topology::NetworkTopologyPanel;
pub use log_viewer::LogViewerPanel;

use eframe::egui;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::state::AppState;

/// Base trait for all UI panels
pub trait UIPanel {
    /// Show the panel UI
    fn show(&mut self, ui: &mut egui::Ui);
    
    /// Update panel with new data
    fn update(&mut self);
    
    /// Get panel title
    fn title(&self) -> &str;
    
    /// Check if panel is visible
    fn is_visible(&self) -> bool;
    
    /// Set panel visibility
    fn set_visible(&mut self, visible: bool);
}

/// Shared chart utilities for all panels
pub mod chart_utils {
    use eframe::egui;
    use plotters::prelude::*;
    use plotters_backend::DrawingBackend;
    use std::collections::VecDeque;
    
    /// Simple time series data point
    #[derive(Debug, Clone)]
    pub struct TimeSeriesPoint {
        pub timestamp: f64,
        pub value: f64,
    }
    
    /// Time series data container
    #[derive(Debug, Clone)]
    pub struct TimeSeries {
        pub name: String,
        pub data: VecDeque<TimeSeriesPoint>,
        pub max_points: usize,
        pub color: egui::Color32,
    }
    
    impl TimeSeries {
        pub fn new(name: String, color: egui::Color32) -> Self {
            Self {
                name,
                data: VecDeque::new(),
                max_points: 100,
                color,
            }
        }
        
        pub fn add_point(&mut self, timestamp: f64, value: f64) {
            self.data.push_back(TimeSeriesPoint { timestamp, value });
            
            if self.data.len() > self.max_points {
                self.data.pop_front();
            }
        }
        
        pub fn clear(&mut self) {
            self.data.clear();
        }
    }
    
    /// Simple line chart widget
    pub fn show_line_chart(
        ui: &mut egui::Ui,
        size: egui::Vec2,
        series: &[&TimeSeries],
        title: &str,
        y_label: &str,
    ) {
        let (response, painter) = ui.allocate_painter(size, egui::Sense::hover());
        let rect = response.rect;
        
        if series.is_empty() {
            ui.put(rect, egui::Label::new("No data"));
            return;
        }
        
        // Find data bounds
        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;
        
        for ts in series {
            for point in &ts.data {
                min_x = min_x.min(point.timestamp);
                max_x = max_x.max(point.timestamp);
                min_y = min_y.min(point.value);
                max_y = max_y.max(point.value);
            }
        }
        
        if min_x >= max_x || min_y >= max_y {
            ui.put(rect, egui::Label::new("Insufficient data"));
            return;
        }
        
        // Add some padding to y-axis
        let y_padding = (max_y - min_y) * 0.1;
        min_y -= y_padding;
        max_y += y_padding;
        
        // Draw background
        painter.rect_filled(rect, egui::Rounding::same(4.0), egui::Color32::from_gray(240));
        
        // Draw grid lines
        let grid_color = egui::Color32::from_gray(200);
        for i in 1..10 {
            let x = rect.min.x + (rect.width() * i as f32 / 10.0);
            painter.line_segment(
                [egui::pos2(x, rect.min.y), egui::pos2(x, rect.max.y)],
                egui::Stroke::new(1.0, grid_color),
            );
            
            let y = rect.min.y + (rect.height() * i as f32 / 10.0);
            painter.line_segment(
                [egui::pos2(rect.min.x, y), egui::pos2(rect.max.x, y)],
                egui::Stroke::new(1.0, grid_color),
            );
        }
        
        // Draw data series
        for ts in series {
            if ts.data.len() < 2 {
                continue;
            }
            
            let points: Vec<egui::Pos2> = ts.data.iter()
                .map(|point| {
                    let x = rect.min.x + ((point.timestamp - min_x) / (max_x - min_x)) as f32 * rect.width();
                    let y = rect.max.y - ((point.value - min_y) / (max_y - min_y)) as f32 * rect.height();
                    egui::pos2(x, y)
                })
                .collect();
            
            // Draw line segments
            for window in points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    egui::Stroke::new(2.0, ts.color),
                );
            }
            
            // Draw points
            for point in &points {
                painter.circle_filled(*point, 3.0, ts.color);
            }
        }
        
        // Draw title
        painter.text(
            egui::pos2(rect.center().x, rect.min.y + 10.0),
            egui::Align2::CENTER_TOP,
            title,
            egui::FontId::proportional(14.0),
            egui::Color32::BLACK,
        );
        
        // Draw y-axis label
        painter.text(
            egui::pos2(rect.min.x + 10.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            y_label,
            egui::FontId::proportional(12.0),
            egui::Color32::from_gray(100),
        );
    }
    
    /// Biological-themed color palette
    pub struct BiologicalColors;
    
    impl BiologicalColors {
        pub const PRIMARY: egui::Color32 = egui::Color32::from_rgb(34, 139, 34);    // Forest green
        pub const SECONDARY: egui::Color32 = egui::Color32::from_rgb(144, 238, 144); // Light green
        pub const ACCENT: egui::Color32 = egui::Color32::from_rgb(255, 215, 0);     // Gold
        pub const WARNING: egui::Color32 = egui::Color32::from_rgb(255, 140, 0);    // Dark orange
        pub const ERROR: egui::Color32 = egui::Color32::from_rgb(220, 20, 60);      // Crimson
        pub const INFO: egui::Color32 = egui::Color32::from_rgb(70, 130, 180);      // Steel blue
        
        pub fn get_series_color(index: usize) -> egui::Color32 {
            let colors = [
                Self::PRIMARY,
                Self::SECONDARY,
                Self::ACCENT,
                Self::INFO,
                Self::WARNING,
                Self::ERROR,
            ];
            colors[index % colors.len()]
        }
    }
}

/// Biological tooltip system
pub mod biological_tooltips {
    use eframe::egui;
    use std::collections::HashMap;
    
    /// Biological concept explanation
    #[derive(Debug, Clone)]
    pub struct BiologicalConcept {
        pub name: String,
        pub simple_explanation: String,
        pub detailed_explanation: String,
        pub biological_example: String,
        pub technical_implementation: String,
        pub related_concepts: Vec<String>,
    }
    
    /// Tooltip provider for biological concepts
    pub struct BiologicalTooltipProvider {
        concepts: HashMap<String, BiologicalConcept>,
    }
    
    impl BiologicalTooltipProvider {
        pub fn new() -> Self {
            let mut concepts = HashMap::new();
            
            // Add biological concepts
            concepts.insert("YoungNode".to_string(), BiologicalConcept {
                name: "Young Node Learning".to_string(),
                simple_explanation: "New nodes learn from experienced neighbors".to_string(),
                detailed_explanation: "Young crows observe and imitate successful behaviors of adult crows in their social group, learning hunting techniques, tool use, and territorial navigation through social learning".to_string(),
                biological_example: "A young crow watches elders use sticks to extract insects from tree bark, then practices the same technique".to_string(),
                technical_implementation: "Nodes observe routing patterns and resource allocation strategies from up to 100 neighboring nodes, reducing initialization overhead by 60-80%".to_string(),
                related_concepts: vec!["SocialLearning".to_string(), "NetworkAdaptation".to_string()],
            });
            
            concepts.insert("CasteNode".to_string(), BiologicalConcept {
                name: "Ant Colony Caste System".to_string(),
                simple_explanation: "Specialized roles for maximum efficiency".to_string(),
                detailed_explanation: "Ant colonies achieve remarkable efficiency through division of labor, with workers, soldiers, nurses, and foragers each performing specific functions that complement the whole colony".to_string(),
                biological_example: "Worker ants gather food, soldier ants defend the colony, and nurse ants care for larvae - each specialized for their role".to_string(),
                technical_implementation: "Node resources are compartmentalized into specialized units (training, inference, storage, communication, security) achieving 85-95% resource utilization".to_string(),
                related_concepts: vec!["ResourceAllocation".to_string(), "Specialization".to_string()],
            });
            
            concepts.insert("SwarmIntelligence".to_string(), BiologicalConcept {
                name: "Swarm Intelligence".to_string(),
                simple_explanation: "Collective behavior emerges from simple individual actions".to_string(),
                detailed_explanation: "Complex group behaviors emerge when many simple agents follow basic rules, creating intelligence that exceeds the sum of individual capabilities".to_string(),
                biological_example: "Bird flocks navigate obstacles and predators through simple rules: maintain distance from neighbors, align with group direction, stay with the flock".to_string(),
                technical_implementation: "Distributed consensus and coordination emerge from local node interactions without centralized control".to_string(),
                related_concepts: vec!["EmergentBehavior".to_string(), "DistributedCoordination".to_string()],
            });
            
            Self { concepts }
        }
        
        /// Show tooltip for a biological concept
        pub fn show_tooltip(&self, ui: &mut egui::Ui, concept_id: &str, level: TooltipLevel) {
            if let Some(concept) = self.concepts.get(concept_id) {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.strong(&concept.name);
                        ui.add_space(5.0);
                        
                        match level {
                            TooltipLevel::Simple => {
                                ui.label(&concept.simple_explanation);
                            }
                            TooltipLevel::Detailed => {
                                ui.label(&concept.detailed_explanation);
                                ui.add_space(5.0);
                                ui.small(format!("🔬 Example: {}", concept.biological_example));
                            }
                            TooltipLevel::Technical => {
                                ui.label(&concept.detailed_explanation);
                                ui.add_space(5.0);
                                ui.small(format!("🔬 Biology: {}", concept.biological_example));
                                ui.small(format!("⚙️ Implementation: {}", concept.technical_implementation));
                            }
                        }
                        
                        if !concept.related_concepts.is_empty() {
                            ui.add_space(5.0);
                            ui.horizontal_wrapped(|ui| {
                                ui.small("Related:");
                                for related in &concept.related_concepts {
                                    ui.small(related);
                                }
                            });
                        }
                    });
                });
            }
        }
    }
    
    /// Tooltip complexity level
    #[derive(Debug, Clone, Copy)]
    pub enum TooltipLevel {
        Simple,    // Basic explanation
        Detailed,  // Biological context
        Technical, // Implementation details
    }
    
    /// Widget that shows biological tooltip on hover
    pub fn biological_tooltip(
        ui: &mut egui::Ui,
        text: &str,
        concept_id: &str,
        provider: &BiologicalTooltipProvider,
        level: TooltipLevel,
    ) -> egui::Response {
        let response = ui.selectable_label(false, text);
        
        if response.hovered() {
            egui::show_tooltip_at_pointer(ui.ctx(), egui::Id::new(concept_id), |ui| {
                provider.show_tooltip(ui, concept_id, level);
            });
        }
        
        response
    }
}

/// Shared biological UI components
pub mod biological_widgets {
    use eframe::egui;
    use super::chart_utils::BiologicalColors;
    
    /// Status indicator with biological styling
    pub fn biological_status_indicator(
        ui: &mut egui::Ui,
        status: &str,
        active: bool,
        emoji: &str,
    ) -> egui::Response {
        let color = if active { BiologicalColors::PRIMARY } else { egui::Color32::GRAY };
        let text = format!("{} {}", emoji, status);
        
        ui.colored_label(color, text)
    }
    
    /// Progress bar with biological theming
    pub fn biological_progress_bar(
        ui: &mut egui::Ui,
        progress: f32,
        label: &str,
    ) {
        ui.horizontal(|ui| {
            ui.label(label);
            let progress_bar = egui::ProgressBar::new(progress)
                .fill(BiologicalColors::PRIMARY)
                .animate(true);
            ui.add(progress_bar);
            ui.label(format!("{:.1}%", progress * 100.0));
        });
    }
    
    /// Metric card with biological styling
    pub fn biological_metric_card(
        ui: &mut egui::Ui,
        title: &str,
        value: &str,
        icon: &str,
        trend: Option<f64>,
    ) {
        ui.group(|ui| {
            ui.vertical_centered(|ui| {
                ui.horizontal(|ui| {
                    ui.label(icon);
                    ui.strong(title);
                });
                
                ui.add_space(5.0);
                ui.heading(value);
                
                if let Some(trend_value) = trend {
                    let (trend_icon, trend_color) = if trend_value > 0.0 {
                        ("📈", BiologicalColors::PRIMARY)
                    } else if trend_value < 0.0 {
                        ("📉", BiologicalColors::ERROR)
                    } else {
                        ("➡️", egui::Color32::GRAY)
                    };
                    
                    ui.horizontal(|ui| {
                        ui.small(trend_icon);
                        ui.small(format!("{:+.1}%", trend_value));
                    });
                }
            });
        });
    }
    
    /// Biological role badge
    pub fn role_badge(
        ui: &mut egui::Ui,
        role_name: &str,
        emoji: &str,
        active: bool,
        performance: f64,
    ) -> egui::Response {
        let background_color = if active {
            BiologicalColors::PRIMARY.linear_multiply(0.2)
        } else {
            egui::Color32::from_gray(240)
        };
        
        let text_color = if active {
            BiologicalColors::PRIMARY
        } else {
            egui::Color32::GRAY
        };
        
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(emoji);
                ui.colored_label(text_color, role_name);
                
                if active {
                    ui.separator();
                    ui.small(format!("{:.0}%", performance * 100.0));
                }
            });
        }).response
    }
    
    /// Network connection visualization
    pub fn network_connection_widget(
        ui: &mut egui::Ui,
        connected: bool,
        peer_count: usize,
        quality: f64,
    ) {
        ui.horizontal(|ui| {
            let (status_icon, status_color) = if connected {
                ("🟢", BiologicalColors::PRIMARY)
            } else {
                ("🔴", BiologicalColors::ERROR)
            };
            
            ui.colored_label(status_color, status_icon);
            
            if connected {
                ui.label(format!("{} peers", peer_count));
                ui.separator();
                
                let quality_text = match quality {
                    q if q > 0.8 => "Excellent",
                    q if q > 0.6 => "Good", 
                    q if q > 0.4 => "Fair",
                    _ => "Poor",
                };
                
                ui.label(format!("Quality: {}", quality_text));
            } else {
                ui.colored_label(egui::Color32::GRAY, "Disconnected");
            }
        });
    }
}

/// Animation and transition utilities
pub mod animations {
    use std::time::{Duration, Instant};
    
    /// Simple easing function
    pub fn ease_in_out(t: f32) -> f32 {
        if t < 0.5 {
            2.0 * t * t
        } else {
            -1.0 + (4.0 - 2.0 * t) * t
        }
    }
    
    /// Animated value that transitions smoothly
    #[derive(Debug)]
    pub struct AnimatedValue {
        current: f32,
        target: f32,
        start_time: Instant,
        duration: Duration,
    }
    
    impl AnimatedValue {
        pub fn new(initial_value: f32) -> Self {
            Self {
                current: initial_value,
                target: initial_value,
                start_time: Instant::now(),
                duration: Duration::from_millis(300),
            }
        }
        
        pub fn set_target(&mut self, target: f32) {
            if (self.target - target).abs() > 0.001 {
                self.target = target;
                self.start_time = Instant::now();
            }
        }
        
        pub fn get_current(&mut self) -> f32 {
            let elapsed = self.start_time.elapsed();
            
            if elapsed >= self.duration {
                self.current = self.target;
            } else {
                let t = elapsed.as_secs_f32() / self.duration.as_secs_f32();
                let eased_t = ease_in_out(t);
                self.current = self.current + (self.target - self.current) * eased_t;
            }
            
            self.current
        }
        
        pub fn is_animating(&self) -> bool {
            self.start_time.elapsed() < self.duration && (self.current - self.target).abs() > 0.001
        }
    }
}