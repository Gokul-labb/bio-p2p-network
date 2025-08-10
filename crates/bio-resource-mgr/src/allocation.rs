//! Resource Allocation and Management System
//! 
//! Implements biological resource allocation strategies inspired by natural systems
//! for efficient distributed computing resource management.

use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, RwLock as TokioRwLock};
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{ResourceMetrics, AllocationMetrics, PerformanceMetrics};
use crate::thermal::ThermalSignature;

/// Resource request information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Unique request identifier
    pub id: Uuid,
    /// Requesting node identifier
    pub requester_id: String,
    /// Resource type requested
    pub resource_type: String,
    /// Amount of resource requested
    pub amount: f64,
    /// Priority level (1-10, 10 = highest)
    pub priority: u8,
    /// Maximum acceptable latency
    pub max_latency: Duration,
    /// Minimum quality requirements
    pub min_quality: f64,
    /// Request timestamp
    pub timestamp: DateTime<Utc>,
    /// Expected duration of resource usage
    pub duration: Option<Duration>,
    /// Preferred allocation strategy
    pub preferred_strategy: Option<AllocationStrategy>,
}

impl ResourceRequest {
    /// Create a new resource request
    pub fn new(
        requester_id: String,
        resource_type: String,
        amount: f64,
        priority: u8,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            requester_id,
            resource_type,
            amount,
            priority: priority.min(10),
            max_latency: Duration::from_secs(30),
            min_quality: 0.5,
            timestamp: Utc::now(),
            duration: None,
            preferred_strategy: None,
        }
    }
    
    /// Calculate request urgency score
    pub fn urgency_score(&self) -> f64 {
        let priority_score = self.priority as f64 / 10.0;
        let age_score = {
            let age = Utc::now().signed_duration_since(self.timestamp);
            let age_seconds = age.num_seconds() as f64;
            (age_seconds / 300.0).min(1.0) // Max 5 minutes for full age score
        };
        
        (priority_score + age_score) / 2.0
    }
    
    /// Check if request has expired
    pub fn is_expired(&self) -> bool {
        let age = Utc::now().signed_duration_since(self.timestamp);
        age > chrono::Duration::from_std(self.max_latency).unwrap_or_default()
    }
}

/// Resource allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Unique allocation identifier
    pub id: Uuid,
    /// Related request identifier
    pub request_id: Uuid,
    /// Allocated resource providers
    pub providers: Vec<ResourceProvider>,
    /// Total amount allocated
    pub total_allocated: f64,
    /// Allocation strategy used
    pub strategy_used: AllocationStrategy,
    /// Quality score of allocation
    pub quality_score: f64,
    /// Expected completion time
    pub estimated_completion: Duration,
    /// Allocation timestamp
    pub allocation_timestamp: DateTime<Utc>,
    /// Current status
    pub status: AllocationStatus,
}

/// Resource provider information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceProvider {
    /// Provider node identifier
    pub node_id: String,
    /// Available resource amount
    pub available_amount: f64,
    /// Resource quality score (0.0-1.0)
    pub quality_score: f64,
    /// Current utilization level (0.0-1.0)
    pub utilization_level: f64,
    /// Response latency
    pub response_latency: Duration,
    /// Reliability score based on history
    pub reliability_score: f64,
    /// Current thermal signature
    pub thermal_signature: Option<ThermalSignature>,
}

impl ResourceProvider {
    /// Calculate overall provider score for allocation decisions
    pub fn provider_score(&self) -> f64 {
        let capacity_score = self.available_amount.min(1.0);
        let quality_score = self.quality_score;
        let utilization_score = 1.0 - self.utilization_level;
        let latency_score = 1.0 / (1.0 + self.response_latency.as_millis() as f64 / 1000.0);
        let reliability_score = self.reliability_score;
        
        (capacity_score + quality_score + utilization_score + latency_score + reliability_score) / 5.0
    }
    
    /// Check if provider can satisfy request
    pub fn can_satisfy(&self, request: &ResourceRequest) -> bool {
        self.available_amount >= request.amount &&
        self.quality_score >= request.min_quality &&
        self.response_latency <= request.max_latency
    }
}

/// Allocation status tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStatus {
    /// Allocation is pending
    Pending,
    /// Allocation is active
    Active,
    /// Allocation completed successfully
    Completed,
    /// Allocation failed
    Failed,
    /// Allocation was cancelled
    Cancelled,
}

/// Resource allocation strategies inspired by biological systems
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Greedy allocation - take best available resources immediately (predator behavior)
    Greedy,
    /// Balanced allocation - optimize across multiple criteria (pack hunting)
    Balanced,
    /// Conservative allocation - preserve resources for critical needs (hibernation)
    Conservative,
    /// Collaborative allocation - share resources cooperatively (symbiosis)
    Collaborative,
    /// Adaptive allocation - adjust based on current conditions (migration patterns)
    Adaptive,
    /// Redundant allocation - ensure fault tolerance (swarm resilience)
    Redundant,
}

impl AllocationStrategy {
    /// Get strategy description
    pub fn description(&self) -> &'static str {
        match self {
            Self::Greedy => "Immediate allocation of best available resources",
            Self::Balanced => "Optimized allocation balancing multiple criteria",
            Self::Conservative => "Resource-preserving allocation for critical needs",
            Self::Collaborative => "Cooperative resource sharing allocation",
            Self::Adaptive => "Dynamic allocation based on current conditions",
            Self::Redundant => "Fault-tolerant allocation with redundancy",
        }
    }
    
    /// Get allocation priority multiplier
    pub fn priority_multiplier(&self) -> f64 {
        match self {
            Self::Greedy => 1.5,
            Self::Balanced => 1.0,
            Self::Conservative => 0.7,
            Self::Collaborative => 1.2,
            Self::Adaptive => 1.1,
            Self::Redundant => 0.8,
        }
    }
}

/// Main resource allocator implementing biological allocation strategies
pub struct ResourceAllocator {
    /// Allocator identifier
    pub id: String,
    /// Available resource providers
    providers: Arc<DashMap<String, ResourceProvider>>,
    /// Pending resource requests
    pending_requests: Arc<RwLock<VecDeque<ResourceRequest>>>,
    /// Active allocations
    active_allocations: Arc<DashMap<Uuid, ResourceAllocation>>,
    /// Allocation history
    allocation_history: Arc<RwLock<VecDeque<ResourceAllocation>>>,
    /// Resource metrics
    metrics: Arc<RwLock<AllocationMetrics>>,
    /// Allocation configuration
    config: AllocationConfig,
    /// Request processing channel
    request_sender: mpsc::Sender<ResourceRequest>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Resource allocation configuration
#[derive(Debug, Clone)]
pub struct AllocationConfig {
    /// Maximum pending requests
    pub max_pending_requests: usize,
    /// History size for tracking allocations
    pub history_size: usize,
    /// Default allocation strategy
    pub default_strategy: AllocationStrategy,
    /// Request processing interval
    pub processing_interval: Duration,
    /// Minimum resource threshold for allocation
    pub min_resource_threshold: f64,
    /// Enable adaptive strategies
    pub adaptive_strategies: bool,
    /// Resource fragmentation threshold
    pub fragmentation_threshold: f64,
}

impl Default for AllocationConfig {
    fn default() -> Self {
        Self {
            max_pending_requests: 1000,
            history_size: crate::constants::ALLOCATION_HISTORY_WINDOW,
            default_strategy: AllocationStrategy::Balanced,
            processing_interval: Duration::from_millis(100),
            min_resource_threshold: 0.01,
            adaptive_strategies: true,
            fragmentation_threshold: 0.8,
        }
    }
}

impl ResourceAllocator {
    /// Create a new resource allocator
    pub fn new(id: String, config: AllocationConfig) -> Self {
        let (request_sender, _) = mpsc::channel(1000);
        
        Self {
            id,
            providers: Arc::new(DashMap::new()),
            pending_requests: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_pending_requests))),
            active_allocations: Arc::new(DashMap::new()),
            allocation_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.history_size))),
            metrics: Arc::new(RwLock::new(AllocationMetrics::default())),
            config,
            request_sender,
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the resource allocator
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::allocation_failed("Resource allocator already running"));
            }
            *running = true;
        }
        
        info!("Starting resource allocator: {}", self.id);
        
        // Start request processing task
        self.start_processing_task().await;
        
        Ok(())
    }
    
    /// Stop the resource allocator
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping resource allocator: {}", self.id);
        Ok(())
    }
    
    /// Register a resource provider
    pub async fn register_provider(&self, provider: ResourceProvider) -> ResourceResult<()> {
        info!("Registering resource provider: {}", provider.node_id);
        self.providers.insert(provider.node_id.clone(), provider);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_providers = self.providers.len();
        }
        
        Ok(())
    }
    
    /// Unregister a resource provider
    pub async fn unregister_provider(&self, node_id: &str) -> ResourceResult<()> {
        info!("Unregistering resource provider: {}", node_id);
        self.providers.remove(node_id);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_providers = self.providers.len();
        }
        
        Ok(())
    }
    
    /// Submit a resource request
    pub async fn request_resources(&self, request: ResourceRequest) -> ResourceResult<Uuid> {
        let request_id = request.id;
        
        // Add to pending requests
        {
            let mut pending = self.pending_requests.write();
            if pending.len() >= self.config.max_pending_requests {
                return Err(ResourceError::allocation_failed("Request queue is full"));
            }
            pending.push_back(request.clone());
        }
        
        // Send for processing
        if let Err(e) = self.request_sender.send(request).await {
            error!("Failed to send request for processing: {}", e);
            return Err(ResourceError::allocation_failed("Failed to queue request"));
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_requests += 1;
            metrics.pending_requests = self.pending_requests.read().len();
        }
        
        Ok(request_id)
    }
    
    /// Process resource allocation using biological strategies
    pub async fn allocate_resources(&self, request: &ResourceRequest) -> ResourceResult<ResourceAllocation> {
        let strategy = request.preferred_strategy.unwrap_or(self.config.default_strategy);
        
        // Get suitable providers
        let suitable_providers = self.find_suitable_providers(request).await?;
        
        if suitable_providers.is_empty() {
            return Err(ResourceError::insufficient_resources(
                request.amount, 
                0.0
            ));
        }
        
        // Apply allocation strategy
        let selected_providers = self.apply_allocation_strategy(
            strategy, 
            request, 
            suitable_providers
        ).await?;
        
        // Calculate allocation metrics
        let total_allocated: f64 = selected_providers.iter()
            .map(|p| p.available_amount.min(request.amount))
            .sum();
        
        let quality_score = selected_providers.iter()
            .map(|p| p.quality_score)
            .sum::<f64>() / selected_providers.len() as f64;
        
        let estimated_completion = selected_providers.iter()
            .map(|p| p.response_latency)
            .max()
            .unwrap_or(Duration::from_secs(1));
        
        // Create allocation
        let allocation = ResourceAllocation {
            id: Uuid::new_v4(),
            request_id: request.id,
            providers: selected_providers,
            total_allocated,
            strategy_used: strategy,
            quality_score,
            estimated_completion,
            allocation_timestamp: Utc::now(),
            status: AllocationStatus::Pending,
        };
        
        // Store active allocation
        self.active_allocations.insert(allocation.id, allocation.clone());
        
        // Update resource provider utilization
        self.update_provider_utilization(&allocation).await?;
        
        info!("Allocated resources: {} units using {} strategy", 
            total_allocated, 
            format!("{:?}", strategy)
        );
        
        Ok(allocation)
    }
    
    /// Get allocation status
    pub fn get_allocation(&self, allocation_id: Uuid) -> Option<ResourceAllocation> {
        self.active_allocations.get(&allocation_id).map(|entry| entry.value().clone())
    }
    
    /// Complete an allocation
    pub async fn complete_allocation(&self, allocation_id: Uuid) -> ResourceResult<()> {
        if let Some((_, mut allocation)) = self.active_allocations.remove(&allocation_id) {
            allocation.status = AllocationStatus::Completed;
            
            // Move to history
            {
                let mut history = self.allocation_history.write();
                history.push_back(allocation.clone());
                
                // Maintain history size
                while history.len() > self.config.history_size {
                    history.pop_front();
                }
            }
            
            // Release provider resources
            self.release_provider_resources(&allocation).await?;
            
            // Update metrics
            {
                let mut metrics = self.metrics.write();
                metrics.completed_allocations += 1;
                metrics.active_allocations = self.active_allocations.len();
            }
            
            Ok(())
        } else {
            Err(ResourceError::allocation_failed("Allocation not found"))
        }
    }
    
    /// Cancel an allocation
    pub async fn cancel_allocation(&self, allocation_id: Uuid) -> ResourceResult<()> {
        if let Some((_, mut allocation)) = self.active_allocations.remove(&allocation_id) {
            allocation.status = AllocationStatus::Cancelled;
            
            // Release provider resources
            self.release_provider_resources(&allocation).await?;
            
            // Update metrics
            {
                let mut metrics = self.metrics.write();
                metrics.cancelled_allocations += 1;
                metrics.active_allocations = self.active_allocations.len();
            }
            
            Ok(())
        } else {
            Err(ResourceError::allocation_failed("Allocation not found"))
        }
    }
    
    /// Get resource utilization statistics
    pub fn get_resource_utilization(&self) -> ResourceUtilization {
        let providers: Vec<_> = self.providers.iter().map(|entry| entry.value().clone()).collect();
        
        if providers.is_empty() {
            return ResourceUtilization::default();
        }
        
        let total_capacity: f64 = providers.iter().map(|p| p.available_amount).sum();
        let total_utilized: f64 = providers.iter()
            .map(|p| p.available_amount * p.utilization_level)
            .sum();
        
        let avg_utilization = providers.iter()
            .map(|p| p.utilization_level)
            .sum::<f64>() / providers.len() as f64;
        
        let avg_quality = providers.iter()
            .map(|p| p.quality_score)
            .sum::<f64>() / providers.len() as f64;
        
        ResourceUtilization {
            total_capacity,
            total_utilized,
            utilization_percentage: if total_capacity > 0.0 { total_utilized / total_capacity } else { 0.0 },
            avg_utilization,
            avg_quality,
            active_providers: providers.len(),
            timestamp: Utc::now(),
        }
    }
    
    /// Get allocation metrics
    pub fn get_metrics(&self) -> AllocationMetrics {
        let mut metrics = self.metrics.write();
        metrics.active_allocations = self.active_allocations.len();
        metrics.pending_requests = self.pending_requests.read().len();
        metrics.clone()
    }
    
    /// Calculate optimal resource allocation using biological formulas
    pub fn calculate_optimal_allocation(&self, demand_signal: f64, thermal_feedback: f64) -> f64 {
        let base_capacity = self.get_total_capacity();
        let adaptive_factor = if self.config.adaptive_strategies { 0.8 } else { 0.5 };
        
        crate::math::dynamic_resource_allocation(
            base_capacity,
            adaptive_factor,
            demand_signal,
            thermal_feedback
        )
    }
    
    // Private methods
    
    async fn start_processing_task(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let pending_requests = Arc::clone(&self.pending_requests);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.processing_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Process pending requests
                let requests_to_process: Vec<ResourceRequest> = {
                    let mut pending = pending_requests.write();
                    let mut requests = Vec::new();
                    
                    // Process up to 10 requests per cycle
                    for _ in 0..10 {
                        if let Some(request) = pending.pop_front() {
                            requests.push(request);
                        } else {
                            break;
                        }
                    }
                    
                    requests
                };
                
                // This would normally process each request
                // For now, just log the processing
                if !requests_to_process.is_empty() {
                    debug!("Processing {} pending requests", requests_to_process.len());
                }
            }
        });
    }
    
    async fn find_suitable_providers(&self, request: &ResourceRequest) -> ResourceResult<Vec<ResourceProvider>> {
        let mut suitable_providers = Vec::new();
        
        for provider_entry in self.providers.iter() {
            let provider = provider_entry.value();
            
            if provider.can_satisfy(request) {
                suitable_providers.push(provider.clone());
            }
        }
        
        Ok(suitable_providers)
    }
    
    async fn apply_allocation_strategy(
        &self,
        strategy: AllocationStrategy,
        request: &ResourceRequest,
        mut providers: Vec<ResourceProvider>
    ) -> ResourceResult<Vec<ResourceProvider>> {
        match strategy {
            AllocationStrategy::Greedy => {
                // Sort by provider score (highest first) and take the best
                providers.sort_by(|a, b| b.provider_score().partial_cmp(&a.provider_score()).unwrap());
                Ok(providers.into_iter().take(1).collect())
            },
            
            AllocationStrategy::Balanced => {
                // Sort by balanced criteria
                providers.sort_by(|a, b| {
                    let score_a = a.provider_score() * request.urgency_score();
                    let score_b = b.provider_score() * request.urgency_score();
                    score_b.partial_cmp(&score_a).unwrap()
                });
                
                // Select top providers that can fulfill the request
                let mut selected = Vec::new();
                let mut remaining_need = request.amount;
                
                for provider in providers {
                    if remaining_need <= 0.0 {
                        break;
                    }
                    
                    selected.push(provider.clone());
                    remaining_need -= provider.available_amount;
                }
                
                Ok(selected)
            },
            
            AllocationStrategy::Conservative => {
                // Select providers with highest reliability and lowest utilization
                providers.sort_by(|a, b| {
                    let score_a = a.reliability_score * (1.0 - a.utilization_level);
                    let score_b = b.reliability_score * (1.0 - b.utilization_level);
                    score_b.partial_cmp(&score_a).unwrap()
                });
                
                Ok(providers.into_iter().take(1).collect())
            },
            
            AllocationStrategy::Collaborative => {
                // Distribute load across multiple providers
                providers.sort_by(|a, b| a.utilization_level.partial_cmp(&b.utilization_level).unwrap());
                
                let target_providers = (providers.len() / 2 + 1).min(providers.len());
                Ok(providers.into_iter().take(target_providers).collect())
            },
            
            AllocationStrategy::Adaptive => {
                // Adapt selection based on current network conditions
                let network_stress = self.calculate_network_stress();
                
                if network_stress > 0.8 {
                    // High stress - use conservative approach
                    self.apply_allocation_strategy(AllocationStrategy::Conservative, request, providers).await
                } else if network_stress < 0.3 {
                    // Low stress - use greedy approach
                    self.apply_allocation_strategy(AllocationStrategy::Greedy, request, providers).await
                } else {
                    // Medium stress - use balanced approach
                    self.apply_allocation_strategy(AllocationStrategy::Balanced, request, providers).await
                }
            },
            
            AllocationStrategy::Redundant => {
                // Select multiple providers for redundancy
                providers.sort_by(|a, b| b.provider_score().partial_cmp(&a.provider_score()).unwrap());
                
                let redundancy_count = (providers.len() / 3 + 1).min(providers.len()).max(2);
                Ok(providers.into_iter().take(redundancy_count).collect())
            },
        }
    }
    
    async fn update_provider_utilization(&self, allocation: &ResourceAllocation) -> ResourceResult<()> {
        for provider_info in &allocation.providers {
            if let Some(mut provider_entry) = self.providers.get_mut(&provider_info.node_id) {
                let provider = provider_entry.value_mut();
                
                // Update utilization based on allocation
                let allocated_ratio = provider_info.available_amount / provider.available_amount;
                provider.utilization_level = (provider.utilization_level + allocated_ratio).min(1.0);
            }
        }
        
        Ok(())
    }
    
    async fn release_provider_resources(&self, allocation: &ResourceAllocation) -> ResourceResult<()> {
        for provider_info in &allocation.providers {
            if let Some(mut provider_entry) = self.providers.get_mut(&provider_info.node_id) {
                let provider = provider_entry.value_mut();
                
                // Release utilization
                let released_ratio = provider_info.available_amount / provider.available_amount;
                provider.utilization_level = (provider.utilization_level - released_ratio).max(0.0);
            }
        }
        
        Ok(())
    }
    
    fn get_total_capacity(&self) -> f64 {
        self.providers.iter()
            .map(|entry| entry.value().available_amount)
            .sum()
    }
    
    fn calculate_network_stress(&self) -> f64 {
        let providers: Vec<_> = self.providers.iter().map(|entry| entry.value().clone()).collect();
        
        if providers.is_empty() {
            return 0.0;
        }
        
        let avg_utilization = providers.iter()
            .map(|p| p.utilization_level)
            .sum::<f64>() / providers.len() as f64;
        
        let capacity_stress = 1.0 - (self.get_total_capacity() / 1000.0).min(1.0); // Assume 1000 is max expected capacity
        
        (avg_utilization + capacity_stress) / 2.0
    }
}

/// Resource utilization statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Total resource capacity
    pub total_capacity: f64,
    /// Total resources currently utilized
    pub total_utilized: f64,
    /// Utilization percentage (0.0-1.0)
    pub utilization_percentage: f64,
    /// Average utilization across providers
    pub avg_utilization: f64,
    /// Average quality score
    pub avg_quality: f64,
    /// Number of active providers
    pub active_providers: usize,
    /// Timestamp of statistics
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_resource_request_creation() {
        let request = ResourceRequest::new(
            "test-node".to_string(),
            "cpu".to_string(),
            0.5,
            5
        );
        
        assert_eq!(request.requester_id, "test-node");
        assert_eq!(request.resource_type, "cpu");
        assert_eq!(request.amount, 0.5);
        assert_eq!(request.priority, 5);
        assert!(!request.id.is_nil());
    }
    
    #[test]
    fn test_resource_provider_scoring() {
        let provider = ResourceProvider {
            node_id: "test-provider".to_string(),
            available_amount: 1.0,
            quality_score: 0.9,
            utilization_level: 0.3,
            response_latency: Duration::from_millis(50),
            reliability_score: 0.95,
            thermal_signature: None,
        };
        
        let score = provider.provider_score();
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }
    
    #[test]
    fn test_request_urgency_calculation() {
        let mut request = ResourceRequest::new(
            "test-node".to_string(),
            "memory".to_string(),
            0.3,
            8
        );
        
        let urgency = request.urgency_score();
        assert!(urgency >= 0.0);
        assert!(urgency <= 1.0);
        
        // High priority should increase urgency
        request.priority = 10;
        let high_urgency = request.urgency_score();
        assert!(high_urgency > urgency);
    }
    
    #[tokio::test]
    async fn test_allocator_creation() {
        let config = AllocationConfig::default();
        let allocator = ResourceAllocator::new("test-allocator".to_string(), config);
        
        assert_eq!(allocator.id, "test-allocator");
        assert_eq!(allocator.get_total_capacity(), 0.0);
    }
    
    #[tokio::test]
    async fn test_provider_registration() {
        let config = AllocationConfig::default();
        let allocator = ResourceAllocator::new("test-allocator".to_string(), config);
        
        let provider = ResourceProvider {
            node_id: "provider1".to_string(),
            available_amount: 1.0,
            quality_score: 0.8,
            utilization_level: 0.2,
            response_latency: Duration::from_millis(100),
            reliability_score: 0.9,
            thermal_signature: None,
        };
        
        allocator.register_provider(provider).await.unwrap();
        
        assert_eq!(allocator.get_total_capacity(), 1.0);
        
        let metrics = allocator.get_metrics();
        assert_eq!(metrics.total_providers, 1);
    }
    
    #[test]
    fn test_allocation_strategies() {
        assert_eq!(AllocationStrategy::Greedy.priority_multiplier(), 1.5);
        assert_eq!(AllocationStrategy::Balanced.priority_multiplier(), 1.0);
        
        assert!(!AllocationStrategy::Greedy.description().is_empty());
    }
    
    #[test]
    fn test_dynamic_allocation_formula() {
        let allocator = ResourceAllocator::new("test".to_string(), AllocationConfig::default());
        
        let allocation = allocator.calculate_optimal_allocation(0.8, 0.1);
        assert!(allocation >= 0.0);
    }
}