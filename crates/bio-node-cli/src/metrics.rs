//! Prometheus metrics collection for Bio P2P Node monitoring
//!
//! This module provides comprehensive metrics collection including network statistics,
//! resource usage, biological behavior metrics, and performance indicators.

use anyhow::{Context, Result};
use prometheus::{
    CounterVec, GaugeVec, HistogramVec, IntCounterVec, IntGaugeVec, 
    Registry, Encoder, TextEncoder, Opts, Histogram, Counter, Gauge, IntCounter, IntGauge
};
use serde::{Deserialize, Serialize};
use tokio::{sync::broadcast, time::{interval, Duration}};
use tracing::{debug, error, info, warn};
use std::sync::Arc;
use std::collections::HashMap;

use crate::{
    config::{NodeConfig, BiologicalRole},
    node::BiologicalNode,
};

/// Metrics collection system for comprehensive node monitoring
#[derive(Debug)]
pub struct MetricsCollector {
    /// Node configuration
    config: NodeConfig,
    
    /// Reference to biological node
    biological_node: Arc<BiologicalNode>,
    
    /// Prometheus metrics registry
    registry: Registry,
    
    /// Network metrics
    network_metrics: NetworkMetrics,
    
    /// Resource metrics
    resource_metrics: ResourceMetrics,
    
    /// Biological behavior metrics
    biological_metrics: BiologicalMetrics,
    
    /// Performance metrics
    performance_metrics: PerformanceMetrics,
    
    /// Security metrics
    security_metrics: SecurityMetrics,
    
    /// Economic metrics
    economic_metrics: EconomicMetrics,
    
    /// HTTP server for metrics endpoint
    metrics_server: Option<Arc<MetricsHttpServer>>,
}

/// Network-related metrics
#[derive(Debug)]
pub struct NetworkMetrics {
    /// Connected peers gauge
    pub connected_peers: IntGauge,
    
    /// Total connection attempts counter
    pub connection_attempts_total: IntCounter,
    
    /// Failed connection attempts counter
    pub connection_failures_total: IntCounter,
    
    /// Messages sent counter by type
    pub messages_sent_total: IntCounterVec,
    
    /// Messages received counter by type
    pub messages_received_total: IntCounterVec,
    
    /// Network latency histogram
    pub network_latency_seconds: HistogramVec,
    
    /// Bandwidth usage gauge (bytes/sec)
    pub bandwidth_usage_bytes: GaugeVec,
    
    /// Packet loss ratio gauge
    pub packet_loss_ratio: Gauge,
}

/// Resource usage metrics
#[derive(Debug)]
pub struct ResourceMetrics {
    /// CPU usage percentage gauge
    pub cpu_usage_percentage: Gauge,
    
    /// Memory usage bytes gauge
    pub memory_usage_bytes: Gauge,
    
    /// Disk usage bytes gauge
    pub disk_usage_bytes: Gauge,
    
    /// Network I/O counters
    pub network_io_bytes_total: CounterVec,
    
    /// Disk I/O counters
    pub disk_io_bytes_total: CounterVec,
    
    /// Resource allocation efficiency gauge
    pub allocation_efficiency: Gauge,
    
    /// Compartment resource usage by type
    pub compartment_usage: GaugeVec,
}

/// Biological behavior metrics
#[derive(Debug)]
pub struct BiologicalMetrics {
    /// Active biological roles gauge
    pub active_roles_count: IntGaugeVec,
    
    /// Trust relationships gauge
    pub trust_relationships: IntGauge,
    
    /// Cooperation success rate
    pub cooperation_success_rate: Gauge,
    
    /// Learning efficiency gauge
    pub learning_efficiency: Gauge,
    
    /// Crisis events counter
    pub crisis_events_total: IntCounter,
    
    /// HAVOC activations counter
    pub havoc_activations_total: IntCounter,
    
    /// Role transitions counter
    pub role_transitions_total: IntCounterVec,
    
    /// Biological algorithm execution time
    pub bio_algorithm_duration_seconds: HistogramVec,
}

/// Performance metrics
#[derive(Debug)]
pub struct PerformanceMetrics {
    /// Task completion counter by result
    pub tasks_completed_total: IntCounterVec,
    
    /// Task duration histogram
    pub task_duration_seconds: HistogramVec,
    
    /// Queue depth gauge
    pub queue_depth: IntGauge,
    
    /// Throughput gauge (tasks/sec)
    pub throughput_tasks_per_second: Gauge,
    
    /// Error rate gauge
    pub error_rate: Gauge,
    
    /// Response time histogram
    pub response_time_seconds: HistogramVec,
    
    /// Active tasks gauge
    pub active_tasks: IntGauge,
}

/// Security metrics
#[derive(Debug)]
pub struct SecurityMetrics {
    /// Security events counter by type
    pub security_events_total: IntCounterVec,
    
    /// Threat detection counter
    pub threats_detected_total: IntCounter,
    
    /// Quarantined nodes gauge
    pub quarantined_nodes: IntGauge,
    
    /// Security layer activations
    pub security_layer_activations: IntCounterVec,
    
    /// Anomaly detection rate
    pub anomaly_detection_rate: Gauge,
    
    /// Trust score distribution
    pub trust_score_distribution: HistogramVec,
}

/// Economic metrics
#[derive(Debug)]
pub struct EconomicMetrics {
    /// Token balance gauge
    pub token_balance: Gauge,
    
    /// Tokens earned counter by activity
    pub tokens_earned_total: CounterVec,
    
    /// Tokens spent counter by service
    pub tokens_spent_total: CounterVec,
    
    /// Reputation score gauge
    pub reputation_score: Gauge,
    
    /// Staking amount gauge
    pub staking_amount: Gauge,
    
    /// Revenue generated gauge
    pub revenue_generated: Gauge,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub async fn new(config: &NodeConfig, biological_node: Arc<BiologicalNode>) -> Result<Self> {
        let registry = Registry::new();
        
        // Initialize metrics
        let network_metrics = NetworkMetrics::new(&registry)?;
        let resource_metrics = ResourceMetrics::new(&registry)?;
        let biological_metrics = BiologicalMetrics::new(&registry)?;
        let performance_metrics = PerformanceMetrics::new(&registry)?;
        let security_metrics = SecurityMetrics::new(&registry)?;
        let economic_metrics = EconomicMetrics::new(&registry)?;
        
        // Create HTTP server if enabled
        let metrics_server = if config.monitoring.enable_metrics {
            Some(Arc::new(MetricsHttpServer::new(
                &config.monitoring.metrics_addr,
                config.monitoring.metrics_port,
                registry.clone()
            ).await.context("Failed to create metrics HTTP server")?))
        } else {
            None
        };
        
        Ok(Self {
            config: config.clone(),
            biological_node,
            registry,
            network_metrics,
            resource_metrics,
            biological_metrics,
            performance_metrics,
            security_metrics,
            economic_metrics,
            metrics_server,
        })
    }
    
    /// Start metrics collection background tasks
    pub async fn start(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        info!("Starting metrics collector");
        
        // Start HTTP server if enabled
        if let Some(ref server) = self.metrics_server {
            let server_shutdown = shutdown_rx.resubscribe();
            let server = server.clone();
            
            tokio::spawn(async move {
                if let Err(e) = server.start(server_shutdown).await {
                    error!("Metrics HTTP server failed: {}", e);
                }
            });
        }
        
        // Start metrics collection loop
        let collection_interval = self.config.monitoring.performance.sample_interval;
        let mut collect_interval = interval(collection_interval);
        
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Metrics collector shutting down");
                    break;
                }
                
                _ = collect_interval.tick() => {
                    if let Err(e) = self.collect_all_metrics().await {
                        error!("Metrics collection failed: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Collect all categories of metrics
    async fn collect_all_metrics(&self) -> Result<()> {
        debug!("Collecting metrics");
        
        // Collect network metrics
        self.collect_network_metrics().await?;
        
        // Collect resource metrics  
        self.collect_resource_metrics().await?;
        
        // Collect biological metrics
        self.collect_biological_metrics().await?;
        
        // Collect performance metrics
        self.collect_performance_metrics().await?;
        
        // Collect security metrics
        self.collect_security_metrics().await?;
        
        // Collect economic metrics
        if self.config.economics.enable_token_economics {
            self.collect_economic_metrics().await?;
        }
        
        debug!("Metrics collection completed");
        Ok(())
    }
    
    /// Collect network-related metrics
    async fn collect_network_metrics(&self) -> Result<()> {
        if let Some(network_info) = self.biological_node.get_network_info().await {
            self.network_metrics.connected_peers.set(network_info.connected_peers as i64);
        }
        
        // Update bandwidth usage
        if let Some(bandwidth) = self.biological_node.get_current_bandwidth().await {
            self.network_metrics.bandwidth_usage_bytes
                .with_label_values(&["inbound"])
                .set(bandwidth.inbound as f64);
            self.network_metrics.bandwidth_usage_bytes
                .with_label_values(&["outbound"])
                .set(bandwidth.outbound as f64);
        }
        
        // Update packet loss
        if let Some(packet_loss) = self.biological_node.get_packet_loss().await {
            self.network_metrics.packet_loss_ratio.set(packet_loss);
        }
        
        Ok(())
    }
    
    /// Collect resource usage metrics
    async fn collect_resource_metrics(&self) -> Result<()> {
        if let Some(resource_usage) = self.biological_node.get_resource_usage().await {
            self.resource_metrics.cpu_usage_percentage.set(resource_usage.cpu_usage * 100.0);
            self.resource_metrics.memory_usage_bytes.set(resource_usage.memory_usage * self.config.resources.max_memory_mb as f64 * 1024.0 * 1024.0);
            self.resource_metrics.disk_usage_bytes.set(resource_usage.disk_usage * self.config.resources.max_disk_gb as f64 * 1024.0 * 1024.0 * 1024.0);
        }
        
        // Update allocation efficiency
        if let Some(efficiency) = self.biological_node.get_allocation_efficiency().await {
            self.resource_metrics.allocation_efficiency.set(efficiency);
        }
        
        // Update compartment usage
        if let Some(compartment_usage) = self.biological_node.get_compartment_usage().await {
            for (compartment, usage) in compartment_usage {
                self.resource_metrics.compartment_usage
                    .with_label_values(&[&compartment])
                    .set(usage);
            }
        }
        
        Ok(())
    }
    
    /// Collect biological behavior metrics
    async fn collect_biological_metrics(&self) -> Result<()> {
        let active_roles = self.biological_node.get_active_roles().await;
        
        // Reset role counters
        self.biological_metrics.active_roles_count.reset();
        
        // Set active role counts
        for role in &active_roles {
            let role_name = format!("{:?}", role);
            self.biological_metrics.active_roles_count
                .with_label_values(&[&role_name])
                .set(1);
        }
        
        // Update trust relationships
        if let Some(trust_count) = self.biological_node.get_trust_relationship_count().await {
            self.biological_metrics.trust_relationships.set(trust_count as i64);
        }
        
        // Update cooperation rate
        if let Some(cooperation_rate) = self.biological_node.get_cooperation_rate().await {
            self.biological_metrics.cooperation_success_rate.set(cooperation_rate);
        }
        
        // Update learning efficiency
        if let Some(learning_efficiency) = self.biological_node.get_learning_efficiency().await {
            self.biological_metrics.learning_efficiency.set(learning_efficiency);
        }
        
        Ok(())
    }
    
    /// Collect performance metrics
    async fn collect_performance_metrics(&self) -> Result<()> {
        // Update queue depth
        if let Some(queue_depth) = self.biological_node.get_queue_depth().await {
            self.performance_metrics.queue_depth.set(queue_depth as i64);
        }
        
        // Update throughput
        if let Some(throughput) = self.biological_node.get_throughput().await {
            self.performance_metrics.throughput_tasks_per_second.set(throughput);
        }
        
        // Update error rate
        if let Some(error_rate) = self.biological_node.get_error_rate().await {
            self.performance_metrics.error_rate.set(error_rate);
        }
        
        // Update active tasks
        if let Some(active_tasks) = self.biological_node.get_active_task_count().await {
            self.performance_metrics.active_tasks.set(active_tasks as i64);
        }
        
        Ok(())
    }
    
    /// Collect security metrics
    async fn collect_security_metrics(&self) -> Result<()> {
        // Update quarantined nodes
        if let Some(quarantined) = self.biological_node.get_quarantined_node_count().await {
            self.security_metrics.quarantined_nodes.set(quarantined as i64);
        }
        
        // Update anomaly detection rate
        if let Some(anomaly_rate) = self.biological_node.get_anomaly_detection_rate().await {
            self.security_metrics.anomaly_detection_rate.set(anomaly_rate);
        }
        
        Ok(())
    }
    
    /// Collect economic metrics
    async fn collect_economic_metrics(&self) -> Result<()> {
        // Update token balance
        if let Some(balance) = self.biological_node.get_token_balance().await {
            self.economic_metrics.token_balance.set(balance);
        }
        
        // Update reputation score
        if let Some(reputation) = self.biological_node.get_reputation_score().await {
            self.economic_metrics.reputation_score.set(reputation);
        }
        
        // Update staking amount
        if let Some(staking) = self.biological_node.get_staking_amount().await {
            self.economic_metrics.staking_amount.set(staking);
        }
        
        Ok(())
    }
    
    /// Record task completion
    pub async fn record_task_completion(&self, task_type: &str, success: bool, duration_secs: f64) {
        let result = if success { "success" } else { "failure" };
        
        self.performance_metrics.tasks_completed_total
            .with_label_values(&[task_type, result])
            .inc();
        
        self.performance_metrics.task_duration_seconds
            .with_label_values(&[task_type])
            .observe(duration_secs);
    }
    
    /// Record network message
    pub async fn record_network_message(&self, message_type: &str, direction: &str, size_bytes: u64) {
        match direction {
            "sent" => {
                self.network_metrics.messages_sent_total
                    .with_label_values(&[message_type])
                    .inc();
            }
            "received" => {
                self.network_metrics.messages_received_total
                    .with_label_values(&[message_type])
                    .inc();
            }
            _ => {}
        }
    }
    
    /// Record network latency
    pub async fn record_network_latency(&self, peer_type: &str, latency_secs: f64) {
        self.network_metrics.network_latency_seconds
            .with_label_values(&[peer_type])
            .observe(latency_secs);
    }
    
    /// Record security event
    pub async fn record_security_event(&self, event_type: &str) {
        self.security_metrics.security_events_total
            .with_label_values(&[event_type])
            .inc();
    }
    
    /// Record biological role transition
    pub async fn record_role_transition(&self, from_role: &str, to_role: &str) {
        self.biological_metrics.role_transitions_total
            .with_label_values(&[from_role, to_role])
            .inc();
    }
    
    /// Get metrics in Prometheus format
    pub async fn get_metrics_text(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)
            .context("Failed to encode metrics")?;
        
        String::from_utf8(buffer)
            .context("Failed to convert metrics to UTF-8")
    }
}

impl NetworkMetrics {
    /// Create network metrics
    fn new(registry: &Registry) -> Result<Self> {
        let connected_peers = IntGauge::new(
            "bio_node_connected_peers",
            "Number of connected peers in the P2P network"
        ).context("Failed to create connected_peers metric")?;
        registry.register(Box::new(connected_peers.clone()))?;
        
        let connection_attempts_total = IntCounter::new(
            "bio_node_connection_attempts_total",
            "Total number of connection attempts"
        ).context("Failed to create connection_attempts_total metric")?;
        registry.register(Box::new(connection_attempts_total.clone()))?;
        
        let connection_failures_total = IntCounter::new(
            "bio_node_connection_failures_total", 
            "Total number of failed connection attempts"
        ).context("Failed to create connection_failures_total metric")?;
        registry.register(Box::new(connection_failures_total.clone()))?;
        
        let messages_sent_total = IntCounterVec::new(
            Opts::new("bio_node_messages_sent_total", "Total messages sent by type"),
            &["message_type"]
        ).context("Failed to create messages_sent_total metric")?;
        registry.register(Box::new(messages_sent_total.clone()))?;
        
        let messages_received_total = IntCounterVec::new(
            Opts::new("bio_node_messages_received_total", "Total messages received by type"),
            &["message_type"]
        ).context("Failed to create messages_received_total metric")?;
        registry.register(Box::new(messages_received_total.clone()))?;
        
        let network_latency_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("bio_node_network_latency_seconds", "Network latency by peer type"),
            &["peer_type"]
        ).context("Failed to create network_latency_seconds metric")?;
        registry.register(Box::new(network_latency_seconds.clone()))?;
        
        let bandwidth_usage_bytes = GaugeVec::new(
            Opts::new("bio_node_bandwidth_usage_bytes_per_second", "Bandwidth usage by direction"),
            &["direction"]
        ).context("Failed to create bandwidth_usage_bytes metric")?;
        registry.register(Box::new(bandwidth_usage_bytes.clone()))?;
        
        let packet_loss_ratio = Gauge::new(
            "bio_node_packet_loss_ratio",
            "Packet loss ratio (0.0 to 1.0)"
        ).context("Failed to create packet_loss_ratio metric")?;
        registry.register(Box::new(packet_loss_ratio.clone()))?;
        
        Ok(Self {
            connected_peers,
            connection_attempts_total,
            connection_failures_total,
            messages_sent_total,
            messages_received_total,
            network_latency_seconds,
            bandwidth_usage_bytes,
            packet_loss_ratio,
        })
    }
}

impl ResourceMetrics {
    /// Create resource metrics
    fn new(registry: &Registry) -> Result<Self> {
        let cpu_usage_percentage = Gauge::new(
            "bio_node_cpu_usage_percentage",
            "CPU usage percentage (0-100)"
        ).context("Failed to create cpu_usage_percentage metric")?;
        registry.register(Box::new(cpu_usage_percentage.clone()))?;
        
        let memory_usage_bytes = Gauge::new(
            "bio_node_memory_usage_bytes",
            "Memory usage in bytes"
        ).context("Failed to create memory_usage_bytes metric")?;
        registry.register(Box::new(memory_usage_bytes.clone()))?;
        
        let disk_usage_bytes = Gauge::new(
            "bio_node_disk_usage_bytes",
            "Disk usage in bytes"
        ).context("Failed to create disk_usage_bytes metric")?;
        registry.register(Box::new(disk_usage_bytes.clone()))?;
        
        let network_io_bytes_total = CounterVec::new(
            Opts::new("bio_node_network_io_bytes_total", "Total network I/O bytes by direction"),
            &["direction"]
        ).context("Failed to create network_io_bytes_total metric")?;
        registry.register(Box::new(network_io_bytes_total.clone()))?;
        
        let disk_io_bytes_total = CounterVec::new(
            Opts::new("bio_node_disk_io_bytes_total", "Total disk I/O bytes by operation"),
            &["operation"]
        ).context("Failed to create disk_io_bytes_total metric")?;
        registry.register(Box::new(disk_io_bytes_total.clone()))?;
        
        let allocation_efficiency = Gauge::new(
            "bio_node_allocation_efficiency_ratio",
            "Resource allocation efficiency (0.0 to 1.0)"
        ).context("Failed to create allocation_efficiency metric")?;
        registry.register(Box::new(allocation_efficiency.clone()))?;
        
        let compartment_usage = GaugeVec::new(
            Opts::new("bio_node_compartment_usage_ratio", "Compartment resource usage by type"),
            &["compartment"]
        ).context("Failed to create compartment_usage metric")?;
        registry.register(Box::new(compartment_usage.clone()))?;
        
        Ok(Self {
            cpu_usage_percentage,
            memory_usage_bytes,
            disk_usage_bytes,
            network_io_bytes_total,
            disk_io_bytes_total,
            allocation_efficiency,
            compartment_usage,
        })
    }
}

impl BiologicalMetrics {
    /// Create biological behavior metrics
    fn new(registry: &Registry) -> Result<Self> {
        let active_roles_count = IntGaugeVec::new(
            Opts::new("bio_node_active_roles_count", "Count of active biological roles"),
            &["role"]
        ).context("Failed to create active_roles_count metric")?;
        registry.register(Box::new(active_roles_count.clone()))?;
        
        let trust_relationships = IntGauge::new(
            "bio_node_trust_relationships",
            "Number of trust relationships with other nodes"
        ).context("Failed to create trust_relationships metric")?;
        registry.register(Box::new(trust_relationships.clone()))?;
        
        let cooperation_success_rate = Gauge::new(
            "bio_node_cooperation_success_rate",
            "Success rate of cooperation attempts (0.0 to 1.0)"
        ).context("Failed to create cooperation_success_rate metric")?;
        registry.register(Box::new(cooperation_success_rate.clone()))?;
        
        let learning_efficiency = Gauge::new(
            "bio_node_learning_efficiency",
            "Efficiency of learning algorithms (0.0 to 1.0)"
        ).context("Failed to create learning_efficiency metric")?;
        registry.register(Box::new(learning_efficiency.clone()))?;
        
        let crisis_events_total = IntCounter::new(
            "bio_node_crisis_events_total",
            "Total number of crisis events detected"
        ).context("Failed to create crisis_events_total metric")?;
        registry.register(Box::new(crisis_events_total.clone()))?;
        
        let havoc_activations_total = IntCounter::new(
            "bio_node_havoc_activations_total",
            "Total number of HAVOC node activations"
        ).context("Failed to create havoc_activations_total metric")?;
        registry.register(Box::new(havoc_activations_total.clone()))?;
        
        let role_transitions_total = IntCounterVec::new(
            Opts::new("bio_node_role_transitions_total", "Total biological role transitions"),
            &["from_role", "to_role"]
        ).context("Failed to create role_transitions_total metric")?;
        registry.register(Box::new(role_transitions_total.clone()))?;
        
        let bio_algorithm_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("bio_node_bio_algorithm_duration_seconds", "Biological algorithm execution time"),
            &["algorithm"]
        ).context("Failed to create bio_algorithm_duration_seconds metric")?;
        registry.register(Box::new(bio_algorithm_duration_seconds.clone()))?;
        
        Ok(Self {
            active_roles_count,
            trust_relationships,
            cooperation_success_rate,
            learning_efficiency,
            crisis_events_total,
            havoc_activations_total,
            role_transitions_total,
            bio_algorithm_duration_seconds,
        })
    }
}

impl PerformanceMetrics {
    /// Create performance metrics
    fn new(registry: &Registry) -> Result<Self> {
        let tasks_completed_total = IntCounterVec::new(
            Opts::new("bio_node_tasks_completed_total", "Total completed tasks by result"),
            &["task_type", "result"]
        ).context("Failed to create tasks_completed_total metric")?;
        registry.register(Box::new(tasks_completed_total.clone()))?;
        
        let task_duration_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("bio_node_task_duration_seconds", "Task completion time"),
            &["task_type"]
        ).context("Failed to create task_duration_seconds metric")?;
        registry.register(Box::new(task_duration_seconds.clone()))?;
        
        let queue_depth = IntGauge::new(
            "bio_node_queue_depth",
            "Current task queue depth"
        ).context("Failed to create queue_depth metric")?;
        registry.register(Box::new(queue_depth.clone()))?;
        
        let throughput_tasks_per_second = Gauge::new(
            "bio_node_throughput_tasks_per_second", 
            "Current task processing throughput"
        ).context("Failed to create throughput_tasks_per_second metric")?;
        registry.register(Box::new(throughput_tasks_per_second.clone()))?;
        
        let error_rate = Gauge::new(
            "bio_node_error_rate",
            "Current error rate (0.0 to 1.0)"
        ).context("Failed to create error_rate metric")?;
        registry.register(Box::new(error_rate.clone()))?;
        
        let response_time_seconds = HistogramVec::new(
            prometheus::HistogramOpts::new("bio_node_response_time_seconds", "Response time by operation"),
            &["operation"]
        ).context("Failed to create response_time_seconds metric")?;
        registry.register(Box::new(response_time_seconds.clone()))?;
        
        let active_tasks = IntGauge::new(
            "bio_node_active_tasks",
            "Number of currently active tasks"
        ).context("Failed to create active_tasks metric")?;
        registry.register(Box::new(active_tasks.clone()))?;
        
        Ok(Self {
            tasks_completed_total,
            task_duration_seconds,
            queue_depth,
            throughput_tasks_per_second,
            error_rate,
            response_time_seconds,
            active_tasks,
        })
    }
}

impl SecurityMetrics {
    /// Create security metrics
    fn new(registry: &Registry) -> Result<Self> {
        let security_events_total = IntCounterVec::new(
            Opts::new("bio_node_security_events_total", "Total security events by type"),
            &["event_type"]
        ).context("Failed to create security_events_total metric")?;
        registry.register(Box::new(security_events_total.clone()))?;
        
        let threats_detected_total = IntCounter::new(
            "bio_node_threats_detected_total",
            "Total number of threats detected"
        ).context("Failed to create threats_detected_total metric")?;
        registry.register(Box::new(threats_detected_total.clone()))?;
        
        let quarantined_nodes = IntGauge::new(
            "bio_node_quarantined_nodes",
            "Number of currently quarantined nodes"
        ).context("Failed to create quarantined_nodes metric")?;
        registry.register(Box::new(quarantined_nodes.clone()))?;
        
        let security_layer_activations = IntCounterVec::new(
            Opts::new("bio_node_security_layer_activations_total", "Security layer activations by layer"),
            &["layer"]
        ).context("Failed to create security_layer_activations metric")?;
        registry.register(Box::new(security_layer_activations.clone()))?;
        
        let anomaly_detection_rate = Gauge::new(
            "bio_node_anomaly_detection_rate",
            "Rate of anomaly detection (0.0 to 1.0)"
        ).context("Failed to create anomaly_detection_rate metric")?;
        registry.register(Box::new(anomaly_detection_rate.clone()))?;
        
        let trust_score_distribution = HistogramVec::new(
            prometheus::HistogramOpts::new("bio_node_trust_score_distribution", "Distribution of trust scores"),
            &["score_range"]
        ).context("Failed to create trust_score_distribution metric")?;
        registry.register(Box::new(trust_score_distribution.clone()))?;
        
        Ok(Self {
            security_events_total,
            threats_detected_total,
            quarantined_nodes,
            security_layer_activations,
            anomaly_detection_rate,
            trust_score_distribution,
        })
    }
}

impl EconomicMetrics {
    /// Create economic metrics
    fn new(registry: &Registry) -> Result<Self> {
        let token_balance = Gauge::new(
            "bio_node_token_balance",
            "Current token balance"
        ).context("Failed to create token_balance metric")?;
        registry.register(Box::new(token_balance.clone()))?;
        
        let tokens_earned_total = CounterVec::new(
            Opts::new("bio_node_tokens_earned_total", "Total tokens earned by activity"),
            &["activity"]
        ).context("Failed to create tokens_earned_total metric")?;
        registry.register(Box::new(tokens_earned_total.clone()))?;
        
        let tokens_spent_total = CounterVec::new(
            Opts::new("bio_node_tokens_spent_total", "Total tokens spent by service"),
            &["service"]
        ).context("Failed to create tokens_spent_total metric")?;
        registry.register(Box::new(tokens_spent_total.clone()))?;
        
        let reputation_score = Gauge::new(
            "bio_node_reputation_score",
            "Current reputation score"
        ).context("Failed to create reputation_score metric")?;
        registry.register(Box::new(reputation_score.clone()))?;
        
        let staking_amount = Gauge::new(
            "bio_node_staking_amount",
            "Current staking amount"
        ).context("Failed to create staking_amount metric")?;
        registry.register(Box::new(staking_amount.clone()))?;
        
        let revenue_generated = Gauge::new(
            "bio_node_revenue_generated",
            "Total revenue generated"
        ).context("Failed to create revenue_generated metric")?;
        registry.register(Box::new(revenue_generated.clone()))?;
        
        Ok(Self {
            token_balance,
            tokens_earned_total,
            tokens_spent_total,
            reputation_score,
            staking_amount,
            revenue_generated,
        })
    }
}

/// HTTP server for metrics endpoints
#[derive(Debug)]
pub struct MetricsHttpServer {
    /// Server bind address
    addr: String,
    /// Server port
    port: u16,
    /// Metrics registry
    registry: Registry,
}

impl MetricsHttpServer {
    /// Create new metrics HTTP server
    pub async fn new(addr: &str, port: u16, registry: Registry) -> Result<Self> {
        Ok(Self {
            addr: addr.to_string(),
            port,
            registry,
        })
    }
    
    /// Start HTTP server
    pub async fn start(&self, mut shutdown_rx: broadcast::Receiver<()>) -> Result<()> {
        use warp::Filter;
        
        info!("Starting metrics HTTP server on {}:{}", self.addr, self.port);
        
        let registry = self.registry.clone();
        
        // Metrics endpoint
        let metrics = warp::path("metrics")
            .and(warp::get())
            .and_then(move || {
                let registry = registry.clone();
                async move {
                    let encoder = TextEncoder::new();
                    let metric_families = registry.gather();
                    
                    let mut buffer = Vec::new();
                    if encoder.encode(&metric_families, &mut buffer).is_err() {
                        return Err(warp::reject::not_found());
                    }
                    
                    match String::from_utf8(buffer) {
                        Ok(metrics_text) => Ok(warp::reply::with_header(
                            metrics_text,
                            "content-type",
                            "text/plain; version=0.0.4; charset=utf-8"
                        )),
                        Err(_) => Err(warp::reject::not_found()),
                    }
                }
            });
        
        let addr: std::net::SocketAddr = format!("{}:{}", self.addr, self.port).parse()
            .context("Invalid server address")?;
        
        let (_, server) = warp::serve(metrics)
            .bind_with_graceful_shutdown(addr, async move {
                let _ = shutdown_rx.recv().await;
                info!("Metrics HTTP server shutting down");
            });
        
        server.await;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NodeConfig;
    use tempfile::TempDir;
    
    async fn create_test_metrics_collector() -> (MetricsCollector, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = NodeConfig::default();
        
        // Configure test paths
        config.storage.data_dir = temp_dir.path().join("data");
        config.storage.cache_dir = temp_dir.path().join("cache");
        config.storage.log_dir = temp_dir.path().join("logs");
        config.network.node_key_path = temp_dir.path().join("node_key.pem");
        config.monitoring.enable_metrics = false; // Disable HTTP server for tests
        
        // Create biological node
        let biological_node = Arc::new(BiologicalNode::new(&config).await.unwrap());
        
        let metrics_collector = MetricsCollector::new(&config, biological_node).await.unwrap();
        
        (metrics_collector, temp_dir)
    }
    
    #[tokio::test]
    async fn test_metrics_collector_creation() {
        let (metrics_collector, _temp_dir) = create_test_metrics_collector().await;
        
        // Test that metrics are registered
        let metrics_text = metrics_collector.get_metrics_text().await.unwrap();
        assert!(metrics_text.contains("bio_node_"));
    }
    
    #[tokio::test]
    async fn test_task_completion_recording() {
        let (metrics_collector, _temp_dir) = create_test_metrics_collector().await;
        
        // Record some task completions
        metrics_collector.record_task_completion("compute", true, 15.5).await;
        metrics_collector.record_task_completion("storage", false, 5.2).await;
        
        let metrics_text = metrics_collector.get_metrics_text().await.unwrap();
        assert!(metrics_text.contains("bio_node_tasks_completed_total"));
        assert!(metrics_text.contains("bio_node_task_duration_seconds"));
    }
    
    #[tokio::test]
    async fn test_network_message_recording() {
        let (metrics_collector, _temp_dir) = create_test_metrics_collector().await;
        
        // Record network messages
        metrics_collector.record_network_message("package", "sent", 1024).await;
        metrics_collector.record_network_message("heartbeat", "received", 64).await;
        
        let metrics_text = metrics_collector.get_metrics_text().await.unwrap();
        assert!(metrics_text.contains("bio_node_messages_sent_total"));
        assert!(metrics_text.contains("bio_node_messages_received_total"));
    }
    
    #[tokio::test]
    async fn test_security_event_recording() {
        let (metrics_collector, _temp_dir) = create_test_metrics_collector().await;
        
        // Record security events
        metrics_collector.record_security_event("anomaly_detected").await;
        metrics_collector.record_security_event("threat_blocked").await;
        
        let metrics_text = metrics_collector.get_metrics_text().await.unwrap();
        assert!(metrics_text.contains("bio_node_security_events_total"));
    }
}