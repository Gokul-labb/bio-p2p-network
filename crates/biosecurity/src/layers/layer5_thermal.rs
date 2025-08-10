//! Layer 5: Thermal Detection
//! 
//! Implements comprehensive resource usage monitoring and multi-dimensional anomaly detection.
//! Inspired by thermal sensing in biological systems like snake heat detection and bat echolocation.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{System, SystemExt, ComponentExt, ProcessExt, CpuExt};

#[cfg(feature = "advanced-monitoring")]
use procfs::{process::all_processes, Meminfo};

use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::config::{LayerConfig, LayerSettings};
use crate::crypto::CryptoContext;
use crate::layers::{SecurityLayer, BaseLayer, SecurityContext, ProcessResult, LayerStatus, LayerMetrics};

/// Layer 5: Thermal Detection implementation
pub struct ThermalDetection {
    base: BaseLayer,
    thermal_config: Arc<RwLock<ThermalConfig>>,
    system_monitor: Arc<RwLock<System>>,
    thermal_signatures: Arc<RwLock<HashMap<String, ThermalSignature>>>,
    resource_history: Arc<RwLock<HashMap<String, ResourceHistory>>>,
    anomaly_thresholds: Arc<RwLock<AnomalyThresholds>>,
    active_monitoring: Arc<RwLock<bool>>,
    sampling_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,
}

impl ThermalDetection {
    pub fn new() -> Self {
        Self {
            base: BaseLayer::new(5, "Thermal Detection".to_string()),
            thermal_config: Arc::new(RwLock::new(ThermalConfig::default())),
            system_monitor: Arc::new(RwLock::new(System::new_all())),
            thermal_signatures: Arc::new(RwLock::new(HashMap::new())),
            resource_history: Arc::new(RwLock::new(HashMap::new())),
            anomaly_thresholds: Arc::new(RwLock::new(AnomalyThresholds::default())),
            active_monitoring: Arc::new(RwLock::new(false)),
            sampling_handle: Arc::new(RwLock::new(None)),
        }
    }

    /// Start thermal monitoring for a computational process
    async fn start_thermal_monitoring(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = Instant::now();
        let session_id = format!("{}-{}", context.execution_id, context.node_id);
        
        // Create thermal signature entry
        let signature = ThermalSignature::new(session_id.clone(), data.len());
        {
            let mut signatures = self.thermal_signatures.write().await;
            signatures.insert(session_id.clone(), signature);
        }

        // Record baseline thermal state
        let baseline_reading = self.capture_thermal_reading(&session_id).await?;
        
        // Initialize resource history for this session
        {
            let mut history = self.resource_history.write().await;
            let resource_history = ResourceHistory::new(session_id.clone());
            history.insert(session_id.clone(), resource_history);
        }

        // Record initial thermal reading
        self.record_thermal_reading(&session_id, baseline_reading).await?;

        let mut events = Vec::new();
        
        // Check if baseline exceeds thresholds
        let baseline_violations = self.check_threshold_violations(&session_id, &baseline_reading).await?;
        if !baseline_violations.is_empty() {
            let event = SecurityEvent::new(
                SecuritySeverity::Medium,
                "baseline_thermal_violation",
                format!("Baseline thermal violations detected: {:?}", baseline_violations),
            )
            .with_layer(5)
            .with_node(context.node_id.clone());
            
            events.push(event);
            self.base.record_security_event().await;
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        tracing::debug!("Started thermal monitoring for session {}", session_id);
        
        Ok(ProcessResult::success(data.to_vec(), context.clone()).with_events(events))
    }

    /// Complete thermal monitoring and analyze patterns
    async fn complete_thermal_monitoring(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = Instant::now();
        let session_id = format!("{}-{}", context.execution_id, context.node_id);
        let mut events = Vec::new();

        // Capture final thermal reading
        let final_reading = self.capture_thermal_reading(&session_id).await?;
        self.record_thermal_reading(&session_id, final_reading.clone()).await?;

        // Perform comprehensive thermal analysis
        let analysis_result = self.analyze_thermal_patterns(&session_id).await?;
        
        // Check for anomalies in thermal patterns
        if analysis_result.anomalies_detected > 0 {
            let event = SecurityEvent::new(
                SecuritySeverity::High,
                "thermal_anomaly_detected",
                format!(
                    "Thermal analysis detected {} anomalies for session {}",
                    analysis_result.anomalies_detected,
                    session_id
                ),
            )
            .with_layer(5)
            .with_node(context.node_id.clone());
            
            events.push(event);
            self.base.record_threat_detection().await;
        }

        // Check for performance optimization opportunities
        let optimization_suggestions = self.generate_optimization_suggestions(&analysis_result).await;
        if !optimization_suggestions.is_empty() {
            let event = SecurityEvent::new(
                SecuritySeverity::Info,
                "performance_optimization_available",
                format!("Performance optimization suggestions: {:?}", optimization_suggestions),
            )
            .with_layer(5)
            .with_node(context.node_id.clone());
            
            events.push(event);
        }

        // Update thermal signature with final analysis
        {
            let mut signatures = self.thermal_signatures.write().await;
            if let Some(signature) = signatures.get_mut(&session_id) {
                signature.complete_analysis(analysis_result);
            }
        }

        // Cleanup session data if configured to do so
        self.cleanup_session_data(&session_id).await?;

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        tracing::debug!("Completed thermal monitoring for session {}", session_id);
        
        Ok(ProcessResult::success(data.to_vec(), context.clone()).with_events(events))
    }

    /// Capture current thermal reading from system
    async fn capture_thermal_reading(&self, session_id: &str) -> SecurityResult<ThermalReading> {
        let mut system = self.system_monitor.write().await;
        system.refresh_all();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // CPU usage
        let cpu_usage = system.global_cpu_info().cpu_usage() as f64 / 100.0;
        
        // Memory usage
        let total_memory = system.total_memory();
        let used_memory = system.used_memory();
        let memory_usage = if total_memory > 0 {
            used_memory as f64 / total_memory as f64
        } else {
            0.0
        };

        // Network I/O (simplified - would need more detailed implementation)
        let network_rx_bytes = self.get_network_rx_bytes().await.unwrap_or(0);
        let network_tx_bytes = self.get_network_tx_bytes().await.unwrap_or(0);

        // Disk I/O (simplified - would need more detailed implementation)
        let disk_read_bytes = self.get_disk_read_bytes().await.unwrap_or(0);
        let disk_write_bytes = self.get_disk_write_bytes().await.unwrap_or(0);

        // Temperature readings (if available)
        let mut temperatures = Vec::new();
        for component in system.components() {
            temperatures.push(component.temperature());
        }

        // Process-specific metrics (if available)
        let process_cpu = self.get_process_cpu_usage(session_id).await.unwrap_or(0.0);
        let process_memory = self.get_process_memory_usage(session_id).await.unwrap_or(0);

        Ok(ThermalReading {
            timestamp,
            session_id: session_id.to_string(),
            cpu_usage,
            memory_usage,
            network_rx_bytes,
            network_tx_bytes,
            disk_read_bytes,
            disk_write_bytes,
            temperatures,
            process_cpu_usage: process_cpu,
            process_memory_bytes: process_memory,
        })
    }

    /// Record thermal reading in history
    async fn record_thermal_reading(
        &self,
        session_id: &str,
        reading: ThermalReading,
    ) -> SecurityResult<()> {
        let config = self.thermal_config.read().await;
        let mut history = self.resource_history.write().await;
        
        if let Some(resource_history) = history.get_mut(session_id) {
            resource_history.add_reading(reading);
            
            // Keep only recent history
            let cutoff_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs() - config.history_retention.as_secs();
            
            resource_history.cleanup_old_readings(cutoff_time);
        }

        Ok(())
    }

    /// Check for threshold violations
    async fn check_threshold_violations(
        &self,
        session_id: &str,
        reading: &ThermalReading,
    ) -> SecurityResult<Vec<ThresholdViolation>> {
        let thresholds = self.anomaly_thresholds.read().await;
        let mut violations = Vec::new();

        // CPU threshold check
        if reading.cpu_usage > thresholds.cpu_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::CPU,
                threshold: thresholds.cpu_threshold,
                actual_value: reading.cpu_usage,
                severity: if reading.cpu_usage > thresholds.cpu_threshold * 1.5 {
                    SecuritySeverity::High
                } else {
                    SecuritySeverity::Medium
                },
            });
        }

        // Memory threshold check
        if reading.memory_usage > thresholds.memory_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::Memory,
                threshold: thresholds.memory_threshold,
                actual_value: reading.memory_usage,
                severity: if reading.memory_usage > thresholds.memory_threshold * 1.2 {
                    SecuritySeverity::High
                } else {
                    SecuritySeverity::Medium
                },
            });
        }

        // Network threshold checks
        if reading.network_rx_bytes > thresholds.network_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::NetworkRx,
                threshold: thresholds.network_threshold as f64,
                actual_value: reading.network_rx_bytes as f64,
                severity: SecuritySeverity::Medium,
            });
        }

        if reading.network_tx_bytes > thresholds.network_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::NetworkTx,
                threshold: thresholds.network_threshold as f64,
                actual_value: reading.network_tx_bytes as f64,
                severity: SecuritySeverity::Medium,
            });
        }

        // Storage threshold checks
        if reading.disk_read_bytes > thresholds.storage_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::DiskRead,
                threshold: thresholds.storage_threshold as f64,
                actual_value: reading.disk_read_bytes as f64,
                severity: SecuritySeverity::Medium,
            });
        }

        if reading.disk_write_bytes > thresholds.storage_threshold {
            violations.push(ThresholdViolation {
                resource_type: ResourceType::DiskWrite,
                threshold: thresholds.storage_threshold as f64,
                actual_value: reading.disk_write_bytes as f64,
                severity: SecuritySeverity::Medium,
            });
        }

        Ok(violations)
    }

    /// Analyze thermal patterns for anomalies and optimization
    async fn analyze_thermal_patterns(&self, session_id: &str) -> SecurityResult<ThermalAnalysisResult> {
        let history = self.resource_history.read().await;
        
        let resource_history = history.get(session_id).ok_or_else(|| {
            SecurityError::ThermalViolation {
                resource: "session".to_string(),
                actual: 0.0,
                threshold: 1.0,
            }
        })?;

        let readings = &resource_history.readings;
        
        if readings.len() < 2 {
            return Ok(ThermalAnalysisResult {
                session_id: session_id.to_string(),
                total_readings: readings.len(),
                anomalies_detected: 0,
                avg_cpu_usage: 0.0,
                avg_memory_usage: 0.0,
                peak_resource_usage: PeakResourceUsage::default(),
                efficiency_score: 1.0,
                recommendations: Vec::new(),
            });
        }

        // Calculate statistics
        let cpu_values: Vec<f64> = readings.iter().map(|r| r.cpu_usage).collect();
        let memory_values: Vec<f64> = readings.iter().map(|r| r.memory_usage).collect();
        
        let avg_cpu = cpu_values.iter().sum::<f64>() / cpu_values.len() as f64;
        let avg_memory = memory_values.iter().sum::<f64>() / memory_values.len() as f64;
        
        let max_cpu = cpu_values.iter().fold(0.0, |a, &b| a.max(b));
        let max_memory = memory_values.iter().fold(0.0, |a, &b| a.max(b));
        
        // Detect anomalies using statistical analysis
        let mut anomaly_count = 0;
        let cpu_std_dev = calculate_std_dev(&cpu_values, avg_cpu);
        let memory_std_dev = calculate_std_dev(&memory_values, avg_memory);
        
        for reading in readings {
            // 3-sigma rule for CPU
            if (reading.cpu_usage - avg_cpu).abs() > 3.0 * cpu_std_dev {
                anomaly_count += 1;
            }
            
            // 3-sigma rule for memory
            if (reading.memory_usage - avg_memory).abs() > 3.0 * memory_std_dev {
                anomaly_count += 1;
            }
        }

        // Calculate efficiency score
        let efficiency_score = calculate_efficiency_score(avg_cpu, avg_memory, max_cpu, max_memory);

        // Generate recommendations
        let recommendations = self.generate_performance_recommendations(
            avg_cpu,
            avg_memory,
            max_cpu,
            max_memory,
            efficiency_score,
        ).await;

        Ok(ThermalAnalysisResult {
            session_id: session_id.to_string(),
            total_readings: readings.len(),
            anomalies_detected: anomaly_count,
            avg_cpu_usage: avg_cpu,
            avg_memory_usage: avg_memory,
            peak_resource_usage: PeakResourceUsage {
                cpu: max_cpu,
                memory: max_memory,
                network_rx: readings.iter().map(|r| r.network_rx_bytes).max().unwrap_or(0),
                network_tx: readings.iter().map(|r| r.network_tx_bytes).max().unwrap_or(0),
                disk_read: readings.iter().map(|r| r.disk_read_bytes).max().unwrap_or(0),
                disk_write: readings.iter().map(|r| r.disk_write_bytes).max().unwrap_or(0),
            },
            efficiency_score,
            recommendations,
        })
    }

    /// Generate optimization suggestions based on analysis
    async fn generate_optimization_suggestions(
        &self,
        analysis: &ThermalAnalysisResult,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        if analysis.efficiency_score < 0.7 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::ResourceUtilization,
                description: "Consider optimizing resource allocation for better efficiency".to_string(),
                potential_improvement: format!("Up to {:.1}% efficiency improvement", (0.9 - analysis.efficiency_score) * 100.0),
            });
        }

        if analysis.peak_resource_usage.cpu > 0.9 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::CPUOptimization,
                description: "High CPU usage detected, consider workload distribution".to_string(),
                potential_improvement: "Reduce CPU bottlenecks".to_string(),
            });
        }

        if analysis.peak_resource_usage.memory > 0.9 {
            suggestions.push(OptimizationSuggestion {
                suggestion_type: OptimizationType::MemoryOptimization,
                description: "High memory usage detected, consider memory optimization".to_string(),
                potential_improvement: "Prevent memory-related performance issues".to_string(),
            });
        }

        suggestions
    }

    /// Generate performance recommendations
    async fn generate_performance_recommendations(
        &self,
        avg_cpu: f64,
        avg_memory: f64,
        max_cpu: f64,
        max_memory: f64,
        efficiency_score: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if avg_cpu < 0.3 && max_cpu < 0.6 {
            recommendations.push("CPU usage is low - consider increasing workload concurrency".to_string());
        }

        if avg_memory < 0.4 && max_memory < 0.7 {
            recommendations.push("Memory usage is low - consider optimizing memory allocation".to_string());
        }

        if efficiency_score < 0.6 {
            recommendations.push("Overall resource efficiency is low - review resource allocation strategy".to_string());
        }

        if max_cpu > 0.95 {
            recommendations.push("CPU usage peaks near maximum - consider load balancing".to_string());
        }

        if max_memory > 0.95 {
            recommendations.push("Memory usage peaks near maximum - monitor for potential issues".to_string());
        }

        recommendations
    }

    /// Cleanup session data
    async fn cleanup_session_data(&self, session_id: &str) -> SecurityResult<()> {
        // Remove thermal signature
        {
            let mut signatures = self.thermal_signatures.write().await;
            signatures.remove(session_id);
        }

        // Keep resource history for analysis, but mark as completed
        {
            let mut history = self.resource_history.write().await;
            if let Some(resource_history) = history.get_mut(session_id) {
                resource_history.mark_completed();
            }
        }

        tracing::debug!("Cleaned up thermal data for session {}", session_id);
        Ok(())
    }

    /// Start background thermal monitoring
    async fn start_background_monitoring(&self) -> SecurityResult<()> {
        let config = self.thermal_config.clone();
        let system_monitor = self.system_monitor.clone();
        let thermal_signatures = self.thermal_signatures.clone();
        let active_monitoring = self.active_monitoring.clone();

        {
            let mut active = active_monitoring.write().await;
            *active = true;
        }

        let handle = tokio::spawn(async move {
            let mut interval = {
                let config_guard = config.read().await;
                tokio::time::interval(config_guard.sampling_frequency)
            };

            loop {
                interval.tick().await;
                
                let should_continue = {
                    let active = active_monitoring.read().await;
                    *active
                };

                if !should_continue {
                    break;
                }

                // Refresh system information
                {
                    let mut system = system_monitor.write().await;
                    system.refresh_all();
                }

                // Update thermal signatures for all active sessions
                let session_ids: Vec<String> = {
                    let signatures = thermal_signatures.read().await;
                    signatures.keys().cloned().collect()
                };

                for session_id in session_ids {
                    // Background monitoring logic would go here
                    // This is simplified for the example
                }
            }
        });

        {
            let mut sampling_handle = self.sampling_handle.write().await;
            *sampling_handle = Some(handle);
        }

        tracing::info!("Started background thermal monitoring");
        Ok(())
    }

    /// Stop background thermal monitoring
    async fn stop_background_monitoring(&self) -> SecurityResult<()> {
        {
            let mut active = self.active_monitoring.write().await;
            *active = false;
        }

        if let Some(handle) = {
            let mut sampling_handle = self.sampling_handle.write().await;
            sampling_handle.take()
        } {
            handle.abort();
        }

        tracing::info!("Stopped background thermal monitoring");
        Ok(())
    }

    // Helper methods for system metrics (simplified implementations)
    
    async fn get_network_rx_bytes(&self) -> Option<u64> {
        // Simplified - would need platform-specific implementation
        Some(0)
    }

    async fn get_network_tx_bytes(&self) -> Option<u64> {
        // Simplified - would need platform-specific implementation
        Some(0)
    }

    async fn get_disk_read_bytes(&self) -> Option<u64> {
        // Simplified - would need platform-specific implementation
        Some(0)
    }

    async fn get_disk_write_bytes(&self) -> Option<u64> {
        // Simplified - would need platform-specific implementation
        Some(0)
    }

    async fn get_process_cpu_usage(&self, _session_id: &str) -> Option<f64> {
        // Simplified - would need to track process IDs
        Some(0.0)
    }

    async fn get_process_memory_usage(&self, _session_id: &str) -> Option<u64> {
        // Simplified - would need to track process IDs
        Some(0)
    }
}

#[async_trait]
impl SecurityLayer for ThermalDetection {
    fn layer_id(&self) -> usize {
        self.base.layer_id()
    }

    fn layer_name(&self) -> &str {
        self.base.layer_name()
    }

    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.base.initialize(config, crypto).await?;
        
        // Extract thermal detection settings
        if let LayerSettings::ThermalDetection { 
            sampling_frequency,
            history_retention,
            cpu_threshold,
            memory_threshold,
            network_threshold,
            storage_threshold,
        } = &config.settings {
            let mut thermal_config = self.thermal_config.write().await;
            thermal_config.sampling_frequency = *sampling_frequency;
            thermal_config.history_retention = *history_retention;

            let mut thresholds = self.anomaly_thresholds.write().await;
            thresholds.cpu_threshold = *cpu_threshold;
            thresholds.memory_threshold = *memory_threshold;
            thresholds.network_threshold = *network_threshold;
            thresholds.storage_threshold = *storage_threshold;
        }

        // Initialize system monitor
        {
            let mut system = self.system_monitor.write().await;
            system.refresh_all();
        }

        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.base.start().await?;
        self.start_background_monitoring().await?;
        Ok(())
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        self.stop_background_monitoring().await?;
        
        // Clear all monitoring state
        {
            let mut signatures = self.thermal_signatures.write().await;
            signatures.clear();
        }

        {
            let mut history = self.resource_history.write().await;
            history.clear();
        }

        self.base.stop().await
    }

    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.start_thermal_monitoring(data, context).await
    }

    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.complete_thermal_monitoring(data, context).await
    }

    async fn status(&self) -> LayerStatus {
        self.base.status().await
    }

    async fn metrics(&self) -> LayerMetrics {
        let mut base_metrics = self.base.metrics().await;
        
        // Add thermal detection specific metrics
        let signatures = self.thermal_signatures.read().await;
        base_metrics.operations_processed = signatures.len() as u64;
        
        base_metrics
    }

    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        // Adjust monitoring sensitivity based on security events
        match event.severity {
            SecuritySeverity::High | SecuritySeverity::Critical => {
                tracing::warn!("High severity event detected, increasing thermal monitoring sensitivity");
                // Could temporarily lower thresholds for more sensitive detection
            },
            _ => {}
        }
        
        self.base.handle_event(event).await
    }
}

/// Thermal monitoring configuration
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    pub sampling_frequency: Duration,
    pub history_retention: Duration,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            sampling_frequency: Duration::from_secs(1),
            history_retention: Duration::from_secs(30 * 24 * 60 * 60), // 30 days
        }
    }
}

/// Anomaly detection thresholds
#[derive(Debug, Clone)]
pub struct AnomalyThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub network_threshold: u64,
    pub storage_threshold: u64,
}

impl Default for AnomalyThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.9,        // 90% CPU usage
            memory_threshold: 0.9,     // 90% memory usage
            network_threshold: 100_000_000, // 100 MB/s
            storage_threshold: 50_000_000,  // 50 MB/s
        }
    }
}

/// Thermal signature for a computation session
#[derive(Debug, Clone)]
pub struct ThermalSignature {
    pub session_id: String,
    pub start_time: Instant,
    pub data_size: usize,
    pub completed: bool,
    pub analysis_result: Option<ThermalAnalysisResult>,
}

impl ThermalSignature {
    pub fn new(session_id: String, data_size: usize) -> Self {
        Self {
            session_id,
            start_time: Instant::now(),
            data_size,
            completed: false,
            analysis_result: None,
        }
    }

    pub fn complete_analysis(&mut self, result: ThermalAnalysisResult) {
        self.completed = true;
        self.analysis_result = Some(result);
    }
}

/// Thermal reading from system
#[derive(Debug, Clone)]
pub struct ThermalReading {
    pub timestamp: u64,
    pub session_id: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub network_rx_bytes: u64,
    pub network_tx_bytes: u64,
    pub disk_read_bytes: u64,
    pub disk_write_bytes: u64,
    pub temperatures: Vec<f32>,
    pub process_cpu_usage: f64,
    pub process_memory_bytes: u64,
}

/// Resource usage history for a session
#[derive(Debug)]
pub struct ResourceHistory {
    pub session_id: String,
    pub readings: VecDeque<ThermalReading>,
    pub max_readings: usize,
    pub completed: bool,
}

impl ResourceHistory {
    pub fn new(session_id: String) -> Self {
        Self {
            session_id,
            readings: VecDeque::new(),
            max_readings: 10000, // Limit memory usage
            completed: false,
        }
    }

    pub fn add_reading(&mut self, reading: ThermalReading) {
        self.readings.push_back(reading);
        
        // Keep only recent readings
        while self.readings.len() > self.max_readings {
            self.readings.pop_front();
        }
    }

    pub fn cleanup_old_readings(&mut self, cutoff_timestamp: u64) {
        while let Some(front) = self.readings.front() {
            if front.timestamp < cutoff_timestamp {
                self.readings.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn mark_completed(&mut self) {
        self.completed = true;
    }
}

/// Threshold violation information
#[derive(Debug, Clone)]
pub struct ThresholdViolation {
    pub resource_type: ResourceType,
    pub threshold: f64,
    pub actual_value: f64,
    pub severity: SecuritySeverity,
}

/// Resource type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceType {
    CPU,
    Memory,
    NetworkRx,
    NetworkTx,
    DiskRead,
    DiskWrite,
    Temperature,
}

/// Thermal analysis result
#[derive(Debug, Clone)]
pub struct ThermalAnalysisResult {
    pub session_id: String,
    pub total_readings: usize,
    pub anomalies_detected: usize,
    pub avg_cpu_usage: f64,
    pub avg_memory_usage: f64,
    pub peak_resource_usage: PeakResourceUsage,
    pub efficiency_score: f64,
    pub recommendations: Vec<String>,
}

/// Peak resource usage during session
#[derive(Debug, Clone)]
pub struct PeakResourceUsage {
    pub cpu: f64,
    pub memory: f64,
    pub network_rx: u64,
    pub network_tx: u64,
    pub disk_read: u64,
    pub disk_write: u64,
}

impl Default for PeakResourceUsage {
    fn default() -> Self {
        Self {
            cpu: 0.0,
            memory: 0.0,
            network_rx: 0,
            network_tx: 0,
            disk_read: 0,
            disk_write: 0,
        }
    }
}

/// Optimization suggestion
#[derive(Debug, Clone)]
pub struct OptimizationSuggestion {
    pub suggestion_type: OptimizationType,
    pub description: String,
    pub potential_improvement: String,
}

/// Optimization type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    ResourceUtilization,
    CPUOptimization,
    MemoryOptimization,
    NetworkOptimization,
    StorageOptimization,
}

// Helper functions

/// Calculate standard deviation
fn calculate_std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    
    let variance = values.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;
    
    variance.sqrt()
}

/// Calculate efficiency score
fn calculate_efficiency_score(avg_cpu: f64, avg_memory: f64, max_cpu: f64, max_memory: f64) -> f64 {
    // Simple efficiency calculation - could be more sophisticated
    let cpu_efficiency = if max_cpu > 0.0 { avg_cpu / max_cpu } else { 1.0 };
    let memory_efficiency = if max_memory > 0.0 { avg_memory / max_memory } else { 1.0 };
    
    (cpu_efficiency + memory_efficiency) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::CryptoConfig;

    #[tokio::test]
    async fn test_thermal_detection_creation() {
        let layer = ThermalDetection::new();
        assert_eq!(layer.layer_id(), 5);
        assert_eq!(layer.layer_name(), "Thermal Detection");
    }

    #[tokio::test]
    async fn test_thermal_config() {
        let config = ThermalConfig::default();
        assert_eq!(config.sampling_frequency, Duration::from_secs(1));
        assert_eq!(config.history_retention, Duration::from_secs(30 * 24 * 60 * 60));
    }

    #[tokio::test]
    async fn test_anomaly_thresholds() {
        let thresholds = AnomalyThresholds::default();
        assert_eq!(thresholds.cpu_threshold, 0.9);
        assert_eq!(thresholds.memory_threshold, 0.9);
        assert_eq!(thresholds.network_threshold, 100_000_000);
        assert_eq!(thresholds.storage_threshold, 50_000_000);
    }

    #[tokio::test]
    async fn test_thermal_signature() {
        let mut signature = ThermalSignature::new("test-session".to_string(), 1024);
        
        assert_eq!(signature.session_id, "test-session");
        assert_eq!(signature.data_size, 1024);
        assert!(!signature.completed);
        assert!(signature.analysis_result.is_none());
        
        let analysis = ThermalAnalysisResult {
            session_id: "test-session".to_string(),
            total_readings: 10,
            anomalies_detected: 0,
            avg_cpu_usage: 0.5,
            avg_memory_usage: 0.6,
            peak_resource_usage: PeakResourceUsage::default(),
            efficiency_score: 0.8,
            recommendations: vec!["Test recommendation".to_string()],
        };
        
        signature.complete_analysis(analysis);
        assert!(signature.completed);
        assert!(signature.analysis_result.is_some());
    }

    #[tokio::test]
    async fn test_resource_history() {
        let mut history = ResourceHistory::new("test-session".to_string());
        
        let reading = ThermalReading {
            timestamp: 1000,
            session_id: "test-session".to_string(),
            cpu_usage: 0.5,
            memory_usage: 0.6,
            network_rx_bytes: 1024,
            network_tx_bytes: 512,
            disk_read_bytes: 2048,
            disk_write_bytes: 1024,
            temperatures: vec![45.0, 50.0],
            process_cpu_usage: 0.3,
            process_memory_bytes: 1024 * 1024,
        };
        
        history.add_reading(reading.clone());
        assert_eq!(history.readings.len(), 1);
        
        // Test cleanup
        history.cleanup_old_readings(2000);
        assert_eq!(history.readings.len(), 0); // Should be removed as it's older than cutoff
    }

    #[test]
    fn test_calculate_std_dev() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = calculate_std_dev(&values, mean);
        
        assert!(std_dev > 0.0);
        // Standard deviation of 1,2,3,4,5 with mean 3 should be ~1.58
        assert!((std_dev - 1.58).abs() < 0.1);
    }

    #[test]
    fn test_calculate_efficiency_score() {
        let score = calculate_efficiency_score(0.5, 0.6, 1.0, 1.0);
        assert_eq!(score, 0.55); // (0.5 + 0.6) / 2 = 0.55
        
        let perfect_score = calculate_efficiency_score(1.0, 1.0, 1.0, 1.0);
        assert_eq!(perfect_score, 1.0);
        
        let zero_score = calculate_efficiency_score(0.0, 0.0, 1.0, 1.0);
        assert_eq!(zero_score, 0.0);
    }

    #[tokio::test]
    async fn test_layer_initialization() {
        let mut layer = ThermalDetection::new();
        let config = LayerConfig::thermal_detection();
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());

        let result = layer.initialize(&config, crypto).await;
        assert!(result.is_ok());
        assert_eq!(layer.status().await, LayerStatus::Ready);
    }

    #[tokio::test]
    async fn test_threshold_violations() {
        let layer = ThermalDetection::new();
        
        let reading = ThermalReading {
            timestamp: 1000,
            session_id: "test-session".to_string(),
            cpu_usage: 0.95, // Above default threshold of 0.9
            memory_usage: 0.5,
            network_rx_bytes: 1024,
            network_tx_bytes: 512,
            disk_read_bytes: 2048,
            disk_write_bytes: 1024,
            temperatures: vec![45.0],
            process_cpu_usage: 0.3,
            process_memory_bytes: 1024 * 1024,
        };
        
        let violations = layer.check_threshold_violations("test-session", &reading).await.unwrap();
        
        // Should detect CPU threshold violation
        assert_eq!(violations.len(), 1);
        assert!(matches!(violations[0].resource_type, ResourceType::CPU));
        assert_eq!(violations[0].threshold, 0.9);
        assert_eq!(violations[0].actual_value, 0.95);
    }
}