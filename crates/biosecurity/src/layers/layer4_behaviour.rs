//! Layer 4: Behavior Monitoring
//! 
//! Implements continuous pattern analysis and anomaly detection using 3-sigma thresholds.
//! Inspired by social animal vigilance systems like meerkat sentries and wolf pack scouts.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use statrs::statistics::{Statistics, OrderStatistics};

#[cfg(feature = "ml-detection")]
use candle_core::{Tensor, Device};

use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::config::{LayerConfig, LayerSettings};
use crate::crypto::CryptoContext;
use crate::layers::{SecurityLayer, BaseLayer, SecurityContext, ProcessResult, LayerStatus, LayerMetrics, RiskLevel};

/// Layer 4: Behavior Monitoring implementation
pub struct BehaviorMonitoring {
    base: BaseLayer,
    monitoring_config: Arc<RwLock<MonitoringConfig>>,
    behavioral_baselines: Arc<RwLock<HashMap<String, BehavioralBaseline>>>,
    pattern_analyzer: Arc<RwLock<PatternAnalyzer>>,
    anomaly_detector: Arc<RwLock<AnomalyDetector>>,
    active_monitors: Arc<RwLock<HashMap<String, NodeMonitor>>>,
    #[cfg(feature = "ml-detection")]
    ml_classifier: Arc<RwLock<Option<MLClassifier>>>,
}

impl BehaviorMonitoring {
    pub fn new() -> Self {
        Self {
            base: BaseLayer::new(4, "Behavior Monitoring".to_string()),
            monitoring_config: Arc::new(RwLock::new(MonitoringConfig::default())),
            behavioral_baselines: Arc::new(RwLock::new(HashMap::new())),
            pattern_analyzer: Arc::new(RwLock::new(PatternAnalyzer::new())),
            anomaly_detector: Arc::new(RwLock::new(AnomalyDetector::new())),
            active_monitors: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "ml-detection")]
            ml_classifier: Arc::new(RwLock::new(None)),
        }
    }

    /// Start monitoring a node's behavior
    async fn start_node_monitoring(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = Instant::now();
        let config = self.monitoring_config.read().await;
        
        // Create or get existing monitor for this node
        let monitor = {
            let mut monitors = self.active_monitors.write().await;
            monitors.entry(context.node_id.clone()).or_insert_with(|| {
                NodeMonitor::new(context.node_id.clone())
            }).clone()
        };

        // Record behavioral observation
        let observation = BehavioralObservation {
            timestamp: chrono::Utc::now(),
            node_id: context.node_id.clone(),
            execution_id: context.execution_id.clone(),
            data_size: data.len(),
            trust_score: context.trust_score,
            risk_level: context.risk_level,
            communication_pattern: self.extract_communication_pattern(data).await,
            resource_usage: self.extract_resource_usage_pattern(context).await?,
            interaction_frequency: monitor.get_interaction_frequency().await,
            response_time_ms: 0, // Will be updated in post-processing
        };

        // Add observation to pattern analyzer
        {
            let mut analyzer = self.pattern_analyzer.write().await;
            analyzer.add_observation(&observation).await?;
        }

        // Check against baseline if available
        let mut events = Vec::new();
        {
            let baselines = self.behavioral_baselines.read().await;
            if let Some(baseline) = baselines.get(&context.node_id) {
                let anomaly_score = self.calculate_anomaly_score(&observation, baseline).await?;
                
                if anomaly_score > config.anomaly_threshold {
                    let event = SecurityEvent::new(
                        SecuritySeverity::Medium,
                        "behavioral_anomaly_detected",
                        format!(
                            "Node {} behavior anomaly detected (score: {:.2})", 
                            context.node_id, 
                            anomaly_score
                        ),
                    )
                    .with_layer(4)
                    .with_node(context.node_id.clone());
                    
                    events.push(event);
                    self.base.record_threat_detection().await;
                }
            }
        }

        // Update monitor with current observation
        {
            let mut monitors = self.active_monitors.write().await;
            if let Some(monitor) = monitors.get_mut(&context.node_id) {
                monitor.record_observation(&observation).await;
            }
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        Ok(ProcessResult::success(data.to_vec(), context.clone()).with_events(events))
    }

    /// Complete monitoring for a node's behavior
    async fn complete_node_monitoring(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = Instant::now();
        let mut events = Vec::new();

        // Update response time for the completed operation
        {
            let mut monitors = self.active_monitors.write().await;
            if let Some(monitor) = monitors.get_mut(&context.node_id) {
                monitor.complete_current_operation(start_time.elapsed()).await;
            }
        }

        // Perform comprehensive behavioral analysis
        let analysis_result = self.analyze_behavior_patterns(&context.node_id).await?;
        
        if analysis_result.anomalies_detected > 0 {
            let event = SecurityEvent::new(
                SecuritySeverity::High,
                "behavioral_analysis_complete",
                format!(
                    "Behavioral analysis completed for node {}: {} anomalies detected",
                    context.node_id,
                    analysis_result.anomalies_detected
                ),
            )
            .with_layer(4)
            .with_node(context.node_id.clone());
            
            events.push(event);
        }

        // Update baseline if learning period is active
        self.update_behavioral_baseline(&context.node_id).await?;

        // Perform ML-based classification if enabled
        #[cfg(feature = "ml-detection")]
        {
            let ml_classifier = self.ml_classifier.read().await;
            if let Some(ref classifier) = *ml_classifier {
                let classification_result = classifier.classify_behavior(&context.node_id).await?;
                
                if classification_result.threat_probability > 0.8 {
                    let event = SecurityEvent::new(
                        SecuritySeverity::Critical,
                        "ml_threat_detected",
                        format!(
                            "ML classifier detected high threat probability ({:.2}) for node {}",
                            classification_result.threat_probability,
                            context.node_id
                        ),
                    )
                    .with_layer(4)
                    .with_node(context.node_id.clone());
                    
                    events.push(event);
                    self.base.record_threat_detection().await;
                }
            }
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        Ok(ProcessResult::success(data.to_vec(), context.clone()).with_events(events))
    }

    /// Extract communication pattern from data
    async fn extract_communication_pattern(&self, data: &[u8]) -> CommunicationPattern {
        CommunicationPattern {
            message_size: data.len(),
            frequency_per_minute: 1.0, // Would be calculated based on historical data
            protocol_type: "tcp".to_string(), // Would be extracted from actual data
            destination_count: 1,
        }
    }

    /// Extract resource usage pattern from context
    async fn extract_resource_usage_pattern(&self, _context: &SecurityContext) -> SecurityResult<ResourceUsagePattern> {
        // In real implementation, this would query system metrics
        Ok(ResourceUsagePattern {
            cpu_usage: 0.5,
            memory_usage: 0.6,
            network_io_bytes: 1024,
            disk_io_bytes: 512,
        })
    }

    /// Calculate anomaly score using 3-sigma rule
    async fn calculate_anomaly_score(
        &self,
        observation: &BehavioralObservation,
        baseline: &BehavioralBaseline,
    ) -> SecurityResult<f64> {
        let mut anomaly_scores = Vec::new();

        // Check data size against baseline
        let data_size_z_score = (observation.data_size as f64 - baseline.avg_data_size) 
            / baseline.std_data_size.max(1.0);
        anomaly_scores.push(data_size_z_score.abs());

        // Check communication frequency
        let freq_z_score = (observation.communication_pattern.frequency_per_minute - baseline.avg_frequency) 
            / baseline.std_frequency.max(0.1);
        anomaly_scores.push(freq_z_score.abs());

        // Check resource usage
        let cpu_z_score = (observation.resource_usage.cpu_usage - baseline.avg_cpu_usage)
            / baseline.std_cpu_usage.max(0.01);
        anomaly_scores.push(cpu_z_score.abs());

        let memory_z_score = (observation.resource_usage.memory_usage - baseline.avg_memory_usage)
            / baseline.std_memory_usage.max(0.01);
        anomaly_scores.push(memory_z_score.abs());

        // Return maximum z-score as anomaly score
        Ok(anomaly_scores.iter().fold(0.0, |a, &b| a.max(b)))
    }

    /// Analyze behavior patterns for a specific node
    async fn analyze_behavior_patterns(&self, node_id: &str) -> SecurityResult<AnalysisResult> {
        let analyzer = self.pattern_analyzer.read().await;
        let observations = analyzer.get_observations_for_node(node_id);
        
        if observations.len() < 10 {
            // Not enough data for reliable analysis
            return Ok(AnalysisResult {
                node_id: node_id.to_string(),
                observations_count: observations.len(),
                anomalies_detected: 0,
                confidence_score: 0.0,
            });
        }

        let mut anomaly_count = 0;
        let config = self.monitoring_config.read().await;
        
        // Statistical analysis using 3-sigma rule
        let data_sizes: Vec<f64> = observations.iter().map(|o| o.data_size as f64).collect();
        let frequencies: Vec<f64> = observations.iter().map(|o| o.communication_pattern.frequency_per_minute).collect();
        
        let data_size_mean = data_sizes.mean();
        let data_size_std = data_sizes.std_dev();
        let freq_mean = frequencies.mean();
        let freq_std = frequencies.std_dev();
        
        for observation in &observations {
            let data_size_z = (observation.data_size as f64 - data_size_mean).abs() / data_size_std.max(1.0);
            let freq_z = (observation.communication_pattern.frequency_per_minute - freq_mean).abs() / freq_std.max(0.1);
            
            if data_size_z > config.anomaly_threshold || freq_z > config.anomaly_threshold {
                anomaly_count += 1;
            }
        }
        
        let confidence = if observations.len() >= config.pattern_window_size {
            1.0
        } else {
            observations.len() as f64 / config.pattern_window_size as f64
        };

        Ok(AnalysisResult {
            node_id: node_id.to_string(),
            observations_count: observations.len(),
            anomalies_detected: anomaly_count,
            confidence_score: confidence,
        })
    }

    /// Update behavioral baseline for a node
    async fn update_behavioral_baseline(&self, node_id: &str) -> SecurityResult<()> {
        let analyzer = self.pattern_analyzer.read().await;
        let observations = analyzer.get_observations_for_node(node_id);
        
        if observations.len() < 30 {
            // Need more observations to establish reliable baseline
            return Ok(());
        }

        // Calculate statistical baseline
        let data_sizes: Vec<f64> = observations.iter().map(|o| o.data_size as f64).collect();
        let frequencies: Vec<f64> = observations.iter().map(|o| o.communication_pattern.frequency_per_minute).collect();
        let cpu_usage: Vec<f64> = observations.iter().map(|o| o.resource_usage.cpu_usage).collect();
        let memory_usage: Vec<f64> = observations.iter().map(|o| o.resource_usage.memory_usage).collect();

        let baseline = BehavioralBaseline {
            node_id: node_id.to_string(),
            created_at: chrono::Utc::now(),
            observation_count: observations.len(),
            avg_data_size: data_sizes.mean(),
            std_data_size: data_sizes.std_dev(),
            avg_frequency: frequencies.mean(),
            std_frequency: frequencies.std_dev(),
            avg_cpu_usage: cpu_usage.mean(),
            std_cpu_usage: cpu_usage.std_dev(),
            avg_memory_usage: memory_usage.mean(),
            std_memory_usage: memory_usage.std_dev(),
        };

        // Update baseline
        {
            let mut baselines = self.behavioral_baselines.write().await;
            baselines.insert(node_id.to_string(), baseline);
        }

        tracing::debug!("Updated behavioral baseline for node {}", node_id);
        Ok(())
    }
}

#[async_trait]
impl SecurityLayer for BehaviorMonitoring {
    fn layer_id(&self) -> usize {
        self.base.layer_id()
    }

    fn layer_name(&self) -> &str {
        self.base.layer_name()
    }

    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.base.initialize(config, crypto).await?;
        
        // Extract behavior monitoring settings
        if let LayerSettings::BehaviorMonitoring { 
            baseline_learning_period,
            anomaly_threshold,
            ml_enabled,
            pattern_window_size,
        } = &config.settings {
            let mut monitoring_config = self.monitoring_config.write().await;
            monitoring_config.baseline_learning_period = *baseline_learning_period;
            monitoring_config.anomaly_threshold = *anomaly_threshold;
            monitoring_config.ml_enabled = *ml_enabled;
            monitoring_config.pattern_window_size = *pattern_window_size;
        }

        // Initialize ML classifier if enabled
        #[cfg(feature = "ml-detection")]
        {
            let config = self.monitoring_config.read().await;
            if config.ml_enabled {
                let classifier = MLClassifier::new().await?;
                let mut ml_classifier = self.ml_classifier.write().await;
                *ml_classifier = Some(classifier);
                tracing::info!("ML-based anomaly detection enabled");
            }
        }

        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.base.start().await?;
        
        // Start background tasks for pattern analysis
        let analyzer = self.pattern_analyzer.clone();
        let config = self.monitoring_config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                
                let config_guard = config.read().await;
                let mut analyzer_guard = analyzer.write().await;
                
                // Clean up old observations beyond learning period
                let cutoff_time = chrono::Utc::now() - chrono::Duration::from_std(config_guard.baseline_learning_period).unwrap();
                analyzer_guard.cleanup_old_observations(cutoff_time).await;
            }
        });

        Ok(())
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        // Clear all monitoring state
        {
            let mut monitors = self.active_monitors.write().await;
            monitors.clear();
        }

        {
            let mut baselines = self.behavioral_baselines.write().await;
            baselines.clear();
        }

        {
            let mut analyzer = self.pattern_analyzer.write().await;
            analyzer.clear_all_observations().await;
        }

        self.base.stop().await
    }

    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.start_node_monitoring(data, context).await
    }

    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.complete_node_monitoring(data, context).await
    }

    async fn status(&self) -> LayerStatus {
        self.base.status().await
    }

    async fn metrics(&self) -> LayerMetrics {
        let mut base_metrics = self.base.metrics().await;
        
        // Add behavior monitoring specific metrics
        let monitors = self.active_monitors.read().await;
        let baselines = self.behavioral_baselines.read().await;
        
        base_metrics.operations_processed = monitors.len() as u64;
        // Additional metrics would be calculated here
        
        base_metrics
    }

    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        // Adjust monitoring sensitivity based on security events
        match event.severity {
            SecuritySeverity::High | SecuritySeverity::Critical => {
                tracing::warn!("High severity event detected, increasing monitoring sensitivity");
                // Could temporarily lower anomaly threshold
            },
            _ => {}
        }
        
        self.base.handle_event(event).await
    }
}

/// Configuration for behavior monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub baseline_learning_period: Duration,
    pub anomaly_threshold: f64,
    pub ml_enabled: bool,
    pub pattern_window_size: usize,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            baseline_learning_period: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
            anomaly_threshold: 3.0, // 3-sigma rule
            ml_enabled: true,
            pattern_window_size: 1000,
        }
    }
}

/// Behavioral observation data point
#[derive(Debug, Clone)]
pub struct BehavioralObservation {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub node_id: String,
    pub execution_id: String,
    pub data_size: usize,
    pub trust_score: f64,
    pub risk_level: RiskLevel,
    pub communication_pattern: CommunicationPattern,
    pub resource_usage: ResourceUsagePattern,
    pub interaction_frequency: f64,
    pub response_time_ms: f64,
}

/// Communication pattern analysis
#[derive(Debug, Clone)]
pub struct CommunicationPattern {
    pub message_size: usize,
    pub frequency_per_minute: f64,
    pub protocol_type: String,
    pub destination_count: usize,
}

/// Resource usage pattern analysis
#[derive(Debug, Clone)]
pub struct ResourceUsagePattern {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_io_bytes: u64,
    pub disk_io_bytes: u64,
}

/// Behavioral baseline for a node
#[derive(Debug, Clone)]
pub struct BehavioralBaseline {
    pub node_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub observation_count: usize,
    pub avg_data_size: f64,
    pub std_data_size: f64,
    pub avg_frequency: f64,
    pub std_frequency: f64,
    pub avg_cpu_usage: f64,
    pub std_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub std_memory_usage: f64,
}

/// Pattern analyzer for behavioral data
#[derive(Debug)]
pub struct PatternAnalyzer {
    observations: HashMap<String, VecDeque<BehavioralObservation>>,
    max_observations_per_node: usize,
}

impl PatternAnalyzer {
    pub fn new() -> Self {
        Self {
            observations: HashMap::new(),
            max_observations_per_node: 10000,
        }
    }

    pub async fn add_observation(&mut self, observation: &BehavioralObservation) -> SecurityResult<()> {
        let node_observations = self.observations
            .entry(observation.node_id.clone())
            .or_insert_with(VecDeque::new);

        node_observations.push_back(observation.clone());

        // Keep only recent observations
        while node_observations.len() > self.max_observations_per_node {
            node_observations.pop_front();
        }

        Ok(())
    }

    pub fn get_observations_for_node(&self, node_id: &str) -> Vec<BehavioralObservation> {
        self.observations
            .get(node_id)
            .map(|obs| obs.iter().cloned().collect())
            .unwrap_or_else(Vec::new)
    }

    pub async fn cleanup_old_observations(&mut self, cutoff_time: chrono::DateTime<chrono::Utc>) {
        for observations in self.observations.values_mut() {
            observations.retain(|obs| obs.timestamp > cutoff_time);
        }
        
        // Remove empty entries
        self.observations.retain(|_, obs| !obs.is_empty());
    }

    pub async fn clear_all_observations(&mut self) {
        self.observations.clear();
    }
}

/// Anomaly detector using statistical methods
#[derive(Debug)]
pub struct AnomalyDetector {
    detection_algorithms: Vec<DetectionAlgorithm>,
}

impl AnomalyDetector {
    pub fn new() -> Self {
        Self {
            detection_algorithms: vec![
                DetectionAlgorithm::ThreeSigma,
                DetectionAlgorithm::IQR,
                DetectionAlgorithm::IsolationForest,
            ],
        }
    }
}

/// Detection algorithm types
#[derive(Debug, Clone)]
pub enum DetectionAlgorithm {
    ThreeSigma,
    IQR,
    IsolationForest,
}

/// Node monitor for tracking individual node behavior
#[derive(Debug, Clone)]
pub struct NodeMonitor {
    pub node_id: String,
    pub start_time: Instant,
    pub last_activity: Instant,
    pub operation_count: u64,
    pub total_processing_time: Duration,
    pub recent_observations: VecDeque<BehavioralObservation>,
}

impl NodeMonitor {
    pub fn new(node_id: String) -> Self {
        let now = Instant::now();
        Self {
            node_id,
            start_time: now,
            last_activity: now,
            operation_count: 0,
            total_processing_time: Duration::ZERO,
            recent_observations: VecDeque::with_capacity(100),
        }
    }

    pub async fn record_observation(&mut self, observation: &BehavioralObservation) {
        self.last_activity = Instant::now();
        self.operation_count += 1;
        
        self.recent_observations.push_back(observation.clone());
        
        // Keep only recent observations
        while self.recent_observations.len() > 100 {
            self.recent_observations.pop_front();
        }
    }

    pub async fn complete_current_operation(&mut self, duration: Duration) {
        self.total_processing_time += duration;
    }

    pub async fn get_interaction_frequency(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs() as f64 / 60.0; // minutes
        if elapsed > 0.0 {
            self.operation_count as f64 / elapsed
        } else {
            0.0
        }
    }
}

/// Analysis result for behavioral patterns
#[derive(Debug)]
pub struct AnalysisResult {
    pub node_id: String,
    pub observations_count: usize,
    pub anomalies_detected: usize,
    pub confidence_score: f64,
}

/// ML-based behavior classifier
#[cfg(feature = "ml-detection")]
#[derive(Debug)]
pub struct MLClassifier {
    model: Option<candle_core::Tensor>,
    device: candle_core::Device,
}

#[cfg(feature = "ml-detection")]
impl MLClassifier {
    pub async fn new() -> SecurityResult<Self> {
        let device = Device::Cpu;
        
        Ok(Self {
            model: None,
            device,
        })
    }

    pub async fn classify_behavior(&self, _node_id: &str) -> SecurityResult<ClassificationResult> {
        // Simplified ML classification
        // Real implementation would use trained model
        Ok(ClassificationResult {
            threat_probability: 0.1,
            classification: "normal".to_string(),
            confidence: 0.8,
        })
    }
}

/// ML classification result
#[cfg(feature = "ml-detection")]
#[derive(Debug)]
pub struct ClassificationResult {
    pub threat_probability: f64,
    pub classification: String,
    pub confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CryptoConfig;

    #[tokio::test]
    async fn test_behavior_monitoring_creation() {
        let layer = BehaviorMonitoring::new();
        assert_eq!(layer.layer_id(), 4);
        assert_eq!(layer.layer_name(), "Behavior Monitoring");
    }

    #[tokio::test]
    async fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert_eq!(config.anomaly_threshold, 3.0);
        assert!(config.ml_enabled);
        assert_eq!(config.pattern_window_size, 1000);
    }

    #[tokio::test]
    async fn test_behavioral_observation() {
        let observation = BehavioralObservation {
            timestamp: chrono::Utc::now(),
            node_id: "test-node".to_string(),
            execution_id: "exec-123".to_string(),
            data_size: 1024,
            trust_score: 0.8,
            risk_level: RiskLevel::Medium,
            communication_pattern: CommunicationPattern {
                message_size: 1024,
                frequency_per_minute: 5.0,
                protocol_type: "tcp".to_string(),
                destination_count: 3,
            },
            resource_usage: ResourceUsagePattern {
                cpu_usage: 0.5,
                memory_usage: 0.6,
                network_io_bytes: 2048,
                disk_io_bytes: 1024,
            },
            interaction_frequency: 2.0,
            response_time_ms: 150.0,
        };

        assert_eq!(observation.node_id, "test-node");
        assert_eq!(observation.data_size, 1024);
        assert_eq!(observation.trust_score, 0.8);
    }

    #[tokio::test]
    async fn test_pattern_analyzer() {
        let mut analyzer = PatternAnalyzer::new();
        
        let observation = BehavioralObservation {
            timestamp: chrono::Utc::now(),
            node_id: "test-node".to_string(),
            execution_id: "exec-123".to_string(),
            data_size: 1024,
            trust_score: 0.8,
            risk_level: RiskLevel::Medium,
            communication_pattern: CommunicationPattern {
                message_size: 1024,
                frequency_per_minute: 5.0,
                protocol_type: "tcp".to_string(),
                destination_count: 1,
            },
            resource_usage: ResourceUsagePattern {
                cpu_usage: 0.5,
                memory_usage: 0.6,
                network_io_bytes: 2048,
                disk_io_bytes: 1024,
            },
            interaction_frequency: 2.0,
            response_time_ms: 150.0,
        };

        analyzer.add_observation(&observation).await.unwrap();
        
        let observations = analyzer.get_observations_for_node("test-node");
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].node_id, "test-node");
    }

    #[tokio::test]
    async fn test_node_monitor() {
        let mut monitor = NodeMonitor::new("test-node".to_string());
        assert_eq!(monitor.node_id, "test-node");
        assert_eq!(monitor.operation_count, 0);
        
        let observation = BehavioralObservation {
            timestamp: chrono::Utc::now(),
            node_id: "test-node".to_string(),
            execution_id: "exec-123".to_string(),
            data_size: 1024,
            trust_score: 0.8,
            risk_level: RiskLevel::Medium,
            communication_pattern: CommunicationPattern {
                message_size: 1024,
                frequency_per_minute: 5.0,
                protocol_type: "tcp".to_string(),
                destination_count: 1,
            },
            resource_usage: ResourceUsagePattern {
                cpu_usage: 0.5,
                memory_usage: 0.6,
                network_io_bytes: 2048,
                disk_io_bytes: 1024,
            },
            interaction_frequency: 2.0,
            response_time_ms: 150.0,
        };

        monitor.record_observation(&observation).await;
        assert_eq!(monitor.operation_count, 1);
    }

    #[tokio::test]
    async fn test_layer_initialization() {
        let mut layer = BehaviorMonitoring::new();
        let config = LayerConfig::behavior_monitoring();
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());

        let result = layer.initialize(&config, crypto).await;
        assert!(result.is_ok());
        assert_eq!(layer.status().await, LayerStatus::Ready);
    }

    #[test]
    fn test_statistics_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = data.mean();
        let std_dev = data.std_dev();
        
        assert_eq!(mean, 3.0);
        assert!(std_dev > 0.0);
        
        // Test 3-sigma rule
        let z_score = (6.0 - mean).abs() / std_dev;
        assert!(z_score > 1.0); // 6.0 should be outside normal range
    }
}