//! Application state management for Bio P2P Desktop UI
//! 
//! This module manages the persistent and runtime state of the desktop
//! application, including UI preferences, network status, and configuration.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::config::AppConfig;

/// Main application state container
pub struct AppState {
    /// Application configuration
    pub config: AppConfig,
    
    /// Current UI state
    pub ui_state: UIState,
    
    /// UI preferences and layout
    pub ui_preferences: UIPreferences,
    
    /// Biological learning progress
    pub biological_progress: BiologicalProgress,
    
    /// Network relationship tracking
    pub network_relationships: NetworkRelationships,
    
    /// Performance history
    pub performance_history: PerformanceHistory,
}

/// Runtime UI state information
#[derive(Debug, Clone)]
pub struct UIState {
    /// Network connectivity status
    pub network_connected: bool,
    pub connected_peers: usize,
    pub connection_quality: f64,
    pub network_uptime: u64,
    
    /// Node operational status
    pub node_running: bool,
    pub node_uptime: u64,
    
    /// Resource utilization
    pub cpu_usage: f64,
    pub memory_usage_mb: f64,
    pub network_upload_mbps: f64,
    pub network_download_mbps: f64,
    
    /// Biological roles and adaptation
    pub active_roles: Vec<String>,
    pub adaptation_progress: f64,
    pub learning_efficiency: f64,
    
    /// Security status
    pub security_level: String,
    pub security_threats: usize,
    pub last_security_scan: Option<DateTime<Utc>>,
    
    /// Package processing
    pub package_queue_size: usize,
    pub packages_processed: u64,
    pub processing_rate: f64,
    
    /// UI interaction state
    pub selected_panel: Option<String>,
    pub show_notifications: bool,
    pub notification_count: usize,
}

/// User interface preferences and layout settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UIPreferences {
    /// Theme and appearance
    pub theme: String,
    pub biological_metaphor_level: BiologicalMetaphorLevel,
    pub show_tooltips: bool,
    pub enable_animations: bool,
    pub color_scheme: ColorScheme,
    
    /// Window layout
    pub window_layout: WindowLayout,
    
    /// Panel visibility
    pub panel_visibility: PanelVisibility,
    
    /// Chart and visualization preferences
    pub chart_preferences: ChartPreferences,
    
    /// Notification preferences
    pub notification_preferences: NotificationPreferences,
    
    /// Learning and education preferences
    pub education_preferences: EducationPreferences,
}

/// Biological metaphor complexity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BiologicalMetaphorLevel {
    Simple,      // Basic analogies and simple explanations
    Intermediate, // Moderate detail with biological context
    Advanced,    // Detailed biological explanations
    Expert,      // Full scientific terminology and concepts
}

/// Application color scheme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub accent: String,
    pub background: String,
    pub text: String,
    pub success: String,
    pub warning: String,
    pub error: String,
    pub biological: String,
}

/// Window layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowLayout {
    pub main_window_size: (f32, f32),
    pub main_window_position: Option<(f32, f32)>,
    pub maximized: bool,
    pub panel_sizes: HashMap<String, f32>,
    pub panel_positions: HashMap<String, PanelPosition>,
    pub splitter_positions: HashMap<String, f32>,
}

/// Panel position configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelPosition {
    Left,
    Right,
    Top,
    Bottom,
    Center,
    Floating { x: f32, y: f32 },
}

/// Panel visibility settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelVisibility {
    pub network_panel: bool,
    pub biological_panel: bool,
    pub peer_panel: bool,
    pub resource_panel: bool,
    pub security_panel: bool,
    pub package_panel: bool,
    pub topology_panel: bool,
    pub logs_panel: bool,
    pub auto_hide_inactive: bool,
    pub pin_active_panels: bool,
}

/// Chart and visualization preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartPreferences {
    pub default_time_range: String,
    pub update_frequency_ms: u64,
    pub show_grid_lines: bool,
    pub show_data_points: bool,
    pub enable_zoom: bool,
    pub enable_pan: bool,
    pub biological_color_coding: bool,
    pub preferred_chart_types: HashMap<String, String>,
}

/// Notification preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationPreferences {
    pub enable_desktop_notifications: bool,
    pub enable_sound_effects: bool,
    pub notification_duration_ms: u64,
    pub filter_by_importance: bool,
    pub biological_event_notifications: bool,
    pub security_alert_notifications: bool,
    pub performance_alert_notifications: bool,
    pub network_event_notifications: bool,
    pub quiet_hours: Option<(u8, u8)>, // (start_hour, end_hour)
}

/// Education and learning preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EducationPreferences {
    pub show_biological_hints: bool,
    pub enable_guided_tours: bool,
    pub learning_pace: LearningPace,
    pub preferred_explanation_style: ExplanationStyle,
    pub track_learning_progress: bool,
    pub gamification_enabled: bool,
}

/// Learning pace for educational content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningPace {
    Slow,
    Normal,
    Fast,
    SelfPaced,
}

/// Explanation style for biological concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExplanationStyle {
    Analogies,      // Use analogies and metaphors
    Technical,      // Focus on technical implementation
    Biological,     // Emphasize biological inspiration
    Mixed,          // Combine all approaches
}

/// Biological learning progress tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalProgress {
    /// Concepts learned and understanding level (0.0 to 1.0)
    pub concept_understanding: HashMap<String, f64>,
    
    /// Total learning time spent
    pub total_learning_time_minutes: u64,
    
    /// Learning milestones achieved
    pub milestones_achieved: Vec<LearningMilestone>,
    
    /// Current learning goals
    pub active_learning_goals: Vec<LearningGoal>,
    
    /// Learning efficiency metrics
    pub learning_efficiency: f64,
    
    /// Favorite biological concepts
    pub favorite_concepts: Vec<String>,
}

/// Learning milestone achievement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningMilestone {
    pub milestone_id: String,
    pub name: String,
    pub description: String,
    pub achieved_at: DateTime<Utc>,
    pub difficulty_level: u8,
    pub biological_concept: String,
}

/// Learning goal tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningGoal {
    pub goal_id: String,
    pub name: String,
    pub description: String,
    pub target_understanding: f64,
    pub current_progress: f64,
    pub created_at: DateTime<Utc>,
    pub target_date: Option<DateTime<Utc>>,
    pub biological_concepts: Vec<String>,
}

/// Network relationship tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkRelationships {
    /// Peer relationship history
    pub peer_relationships: HashMap<String, PeerRelationship>,
    
    /// Trust network evolution
    pub trust_evolution: Vec<TrustEvolutionEvent>,
    
    /// Social network metrics
    pub social_metrics: SocialNetworkMetrics,
    
    /// Biological behavior observations
    pub behavior_observations: Vec<BiologicalBehaviorObservation>,
}

/// Individual peer relationship tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerRelationship {
    pub peer_id: String,
    pub first_contact: DateTime<Utc>,
    pub last_interaction: DateTime<Utc>,
    pub interaction_count: u64,
    pub trust_score_history: Vec<(DateTime<Utc>, f64)>,
    pub reputation_score_history: Vec<(DateTime<Utc>, f64)>,
    pub biological_roles_observed: Vec<String>,
    pub cooperation_events: u64,
    pub conflict_events: u64,
    pub relationship_type: RelationshipType,
}

/// Type of relationship with peer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    Unknown,
    Trusted,
    Friend,
    Buddy,
    Mentor,
    Student,
    Neutral,
    Suspicious,
}

/// Trust evolution event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustEvolutionEvent {
    pub timestamp: DateTime<Utc>,
    pub peer_id: String,
    pub event_type: String,
    pub trust_change: f64,
    pub reason: String,
    pub biological_context: Option<String>,
}

/// Social network metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialNetworkMetrics {
    pub clustering_coefficient: f64,
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub eigenvector_centrality: f64,
    pub social_influence_score: f64,
    pub cooperation_rate: f64,
}

/// Biological behavior observation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicalBehaviorObservation {
    pub timestamp: DateTime<Utc>,
    pub behavior_type: String,
    pub participants: Vec<String>,
    pub effectiveness_rating: f64,
    pub biological_inspiration: String,
    pub learned_insights: Vec<String>,
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Historical performance metrics
    pub metrics_history: Vec<PerformanceSnapshot>,
    
    /// Resource usage trends
    pub resource_trends: ResourceTrends,
    
    /// Network performance evolution
    pub network_performance_evolution: Vec<NetworkPerformanceSnapshot>,
    
    /// Biological adaptation effectiveness
    pub adaptation_effectiveness: Vec<AdaptationEffectivenessSnapshot>,
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_throughput: f64,
    pub task_completion_rate: f64,
    pub energy_efficiency: f64,
    pub biological_adaptation_score: f64,
}

/// Resource usage trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTrends {
    pub cpu_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub network_trend: TrendDirection,
    pub efficiency_trend: TrendDirection,
    pub predicted_resource_needs: ResourcePrediction,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Resource usage prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePrediction {
    pub predicted_cpu_usage: f64,
    pub predicted_memory_usage: f64,
    pub predicted_network_usage: f64,
    pub confidence_level: f64,
    pub prediction_horizon_hours: u64,
}

/// Network performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub peer_count: usize,
    pub connection_quality: f64,
    pub message_latency: f64,
    pub throughput_mbps: f64,
    pub packet_loss_rate: f64,
    pub biological_efficiency_score: f64,
}

/// Adaptation effectiveness measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEffectivenessSnapshot {
    pub timestamp: DateTime<Utc>,
    pub adaptation_type: String,
    pub before_performance: f64,
    pub after_performance: f64,
    pub adaptation_cost: f64,
    pub effectiveness_ratio: f64,
    pub biological_inspiration: String,
}

/// Panel state for UI layout
#[derive(Debug, Clone)]
pub struct PanelState {
    pub show_network: bool,
    pub show_biological: bool,
    pub show_peers: bool,
    pub show_resources: bool,
    pub show_security: bool,
    pub show_packages: bool,
    pub show_topology: bool,
    pub show_logs: bool,
}

impl AppState {
    /// Create new application state
    pub async fn new() -> Result<Self> {
        let config = AppConfig::load_or_default().await?;
        
        Ok(Self {
            config,
            ui_state: UIState::new(),
            ui_preferences: UIPreferences::default(),
            biological_progress: BiologicalProgress::new(),
            network_relationships: NetworkRelationships::new(),
            performance_history: PerformanceHistory::new(),
        })
    }

    /// Import configuration from string
    pub async fn import_configuration(&mut self, config_data: String) -> Result<bool> {
        // Parse the configuration data (assuming TOML format)
        let new_config: AppConfig = toml::from_str(&config_data)?;
        
        // Validate the configuration
        new_config.validate()?;
        
        // Update the configuration
        self.config = new_config;
        
        // Save to disk
        self.config.save().await?;
        
        Ok(true)
    }

    /// Export configuration as string
    pub async fn export_configuration(&self) -> Result<String> {
        let config_data = toml::to_string_pretty(&self.config)?;
        Ok(config_data)
    }

    /// Update biological learning progress
    pub fn update_biological_progress(&mut self, concept: String, understanding_increase: f64) {
        let current_understanding = self.biological_progress.concept_understanding
            .get(&concept)
            .copied()
            .unwrap_or(0.0);
        
        let new_understanding = (current_understanding + understanding_increase).min(1.0);
        self.biological_progress.concept_understanding.insert(concept, new_understanding);
    }

    /// Record peer interaction
    pub fn record_peer_interaction(&mut self, peer_id: String, interaction_type: String) {
        let relationship = self.network_relationships.peer_relationships
            .entry(peer_id.clone())
            .or_insert_with(|| PeerRelationship {
                peer_id: peer_id.clone(),
                first_contact: Utc::now(),
                last_interaction: Utc::now(),
                interaction_count: 0,
                trust_score_history: Vec::new(),
                reputation_score_history: Vec::new(),
                biological_roles_observed: Vec::new(),
                cooperation_events: 0,
                conflict_events: 0,
                relationship_type: RelationshipType::Unknown,
            });

        relationship.last_interaction = Utc::now();
        relationship.interaction_count += 1;

        if interaction_type == "cooperation" {
            relationship.cooperation_events += 1;
        } else if interaction_type == "conflict" {
            relationship.conflict_events += 1;
        }
    }
}

impl UIState {
    pub fn new() -> Self {
        Self {
            network_connected: false,
            connected_peers: 0,
            connection_quality: 0.0,
            network_uptime: 0,
            node_running: false,
            node_uptime: 0,
            cpu_usage: 0.0,
            memory_usage_mb: 0.0,
            network_upload_mbps: 0.0,
            network_download_mbps: 0.0,
            active_roles: Vec::new(),
            adaptation_progress: 0.0,
            learning_efficiency: 0.0,
            security_level: "Unknown".to_string(),
            security_threats: 0,
            last_security_scan: None,
            package_queue_size: 0,
            packages_processed: 0,
            processing_rate: 0.0,
            selected_panel: None,
            show_notifications: true,
            notification_count: 0,
        }
    }
}

impl Default for UIPreferences {
    fn default() -> Self {
        Self {
            theme: "biological".to_string(),
            biological_metaphor_level: BiologicalMetaphorLevel::Intermediate,
            show_tooltips: true,
            enable_animations: true,
            color_scheme: ColorScheme::biological_forest(),
            window_layout: WindowLayout::default(),
            panel_visibility: PanelVisibility::default(),
            chart_preferences: ChartPreferences::default(),
            notification_preferences: NotificationPreferences::default(),
            education_preferences: EducationPreferences::default(),
        }
    }
}

impl ColorScheme {
    pub fn biological_forest() -> Self {
        Self {
            primary: "#228B22".to_string(),      // Forest green
            secondary: "#90EE90".to_string(),    // Light green
            accent: "#FFD700".to_string(),       // Gold
            background: "#F8FFF8".to_string(),   // Mint cream
            text: "#003200".to_string(),         // Dark green
            success: "#008000".to_string(),      // Green
            warning: "#FF8C00".to_string(),      // Dark orange
            error: "#DC143C".to_string(),        // Crimson
            biological: "#2E7D32".to_string(),   // Bio green
        }
    }
}

impl Default for WindowLayout {
    fn default() -> Self {
        Self {
            main_window_size: (1200.0, 800.0),
            main_window_position: None,
            maximized: false,
            panel_sizes: HashMap::new(),
            panel_positions: HashMap::new(),
            splitter_positions: HashMap::new(),
        }
    }
}

impl Default for PanelVisibility {
    fn default() -> Self {
        Self {
            network_panel: true,
            biological_panel: true,
            peer_panel: false,
            resource_panel: true,
            security_panel: false,
            package_panel: false,
            topology_panel: false,
            logs_panel: false,
            auto_hide_inactive: false,
            pin_active_panels: true,
        }
    }
}

impl Default for ChartPreferences {
    fn default() -> Self {
        let mut chart_types = HashMap::new();
        chart_types.insert("network_activity".to_string(), "line".to_string());
        chart_types.insert("resource_usage".to_string(), "area".to_string());
        chart_types.insert("biological_activity".to_string(), "scatter".to_string());

        Self {
            default_time_range: "1h".to_string(),
            update_frequency_ms: 5000,
            show_grid_lines: true,
            show_data_points: false,
            enable_zoom: true,
            enable_pan: true,
            biological_color_coding: true,
            preferred_chart_types: chart_types,
        }
    }
}

impl Default for NotificationPreferences {
    fn default() -> Self {
        Self {
            enable_desktop_notifications: true,
            enable_sound_effects: false,
            notification_duration_ms: 5000,
            filter_by_importance: true,
            biological_event_notifications: true,
            security_alert_notifications: true,
            performance_alert_notifications: true,
            network_event_notifications: true,
            quiet_hours: None,
        }
    }
}

impl Default for EducationPreferences {
    fn default() -> Self {
        Self {
            show_biological_hints: true,
            enable_guided_tours: true,
            learning_pace: LearningPace::Normal,
            preferred_explanation_style: ExplanationStyle::Mixed,
            track_learning_progress: true,
            gamification_enabled: false,
        }
    }
}

impl BiologicalProgress {
    pub fn new() -> Self {
        Self {
            concept_understanding: HashMap::new(),
            total_learning_time_minutes: 0,
            milestones_achieved: Vec::new(),
            active_learning_goals: Vec::new(),
            learning_efficiency: 0.0,
            favorite_concepts: Vec::new(),
        }
    }
}

impl NetworkRelationships {
    pub fn new() -> Self {
        Self {
            peer_relationships: HashMap::new(),
            trust_evolution: Vec::new(),
            social_metrics: SocialNetworkMetrics {
                clustering_coefficient: 0.0,
                betweenness_centrality: 0.0,
                closeness_centrality: 0.0,
                eigenvector_centrality: 0.0,
                social_influence_score: 0.0,
                cooperation_rate: 0.0,
            },
            behavior_observations: Vec::new(),
        }
    }
}

impl PerformanceHistory {
    pub fn new() -> Self {
        Self {
            metrics_history: Vec::new(),
            resource_trends: ResourceTrends {
                cpu_trend: TrendDirection::Stable,
                memory_trend: TrendDirection::Stable,
                network_trend: TrendDirection::Stable,
                efficiency_trend: TrendDirection::Stable,
                predicted_resource_needs: ResourcePrediction {
                    predicted_cpu_usage: 50.0,
                    predicted_memory_usage: 2048.0,
                    predicted_network_usage: 10.0,
                    confidence_level: 0.7,
                    prediction_horizon_hours: 24,
                },
            },
            network_performance_evolution: Vec::new(),
            adaptation_effectiveness: Vec::new(),
        }
    }
}

impl Default for PanelState {
    fn default() -> Self {
        Self {
            show_network: false,
            show_biological: false,
            show_peers: false,
            show_resources: false,
            show_security: false,
            show_packages: false,
            show_topology: false,
            show_logs: false,
        }
    }
}

impl PanelState {
    pub fn any_panel_open(&self) -> bool {
        self.show_network || self.show_biological || self.show_peers ||
        self.show_resources || self.show_security || self.show_packages ||
        self.show_topology || self.show_logs
    }
}