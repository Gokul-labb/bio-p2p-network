//! Social and Collaborative Resource Management
//! 
//! Implements biological social behaviors for resource sharing including
//! Friendship Nodes (tick-host symbiotic networks) and Buddy Nodes (primate grooming networks).

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;
use tracing::{debug, info, warn, error};
use uuid::Uuid;

use crate::errors::{ResourceError, ResourceResult};
use crate::metrics::{SocialMetrics, ResourceMetrics};
use crate::allocation::{ResourceProvider, ResourceRequest, ResourceAllocation};

/// Trust level categories for social relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// No trust established
    None = 0,
    /// Low trust - limited cooperation
    Low = 1,
    /// Medium trust - standard cooperation
    Medium = 2,
    /// High trust - preferred cooperation
    High = 3,
    /// Maximum trust - priority cooperation
    Maximum = 4,
}

impl TrustLevel {
    /// Convert trust level to numeric value (0.0-1.0)
    pub fn to_numeric(&self) -> f64 {
        match self {
            TrustLevel::None => 0.0,
            TrustLevel::Low => 0.25,
            TrustLevel::Medium => 0.5,
            TrustLevel::High => 0.75,
            TrustLevel::Maximum => 1.0,
        }
    }
    
    /// Convert numeric value to trust level
    pub fn from_numeric(value: f64) -> Self {
        if value >= 0.875 {
            TrustLevel::Maximum
        } else if value >= 0.625 {
            TrustLevel::High
        } else if value >= 0.375 {
            TrustLevel::Medium
        } else if value >= 0.125 {
            TrustLevel::Low
        } else {
            TrustLevel::None
        }
    }
    
    /// Get cooperation priority multiplier
    pub fn cooperation_multiplier(&self) -> f64 {
        match self {
            TrustLevel::None => 0.5,
            TrustLevel::Low => 0.8,
            TrustLevel::Medium => 1.0,
            TrustLevel::High => 1.3,
            TrustLevel::Maximum => 1.5,
        }
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::None => write!(f, "NONE"),
            TrustLevel::Low => write!(f, "LOW"),
            TrustLevel::Medium => write!(f, "MEDIUM"),
            TrustLevel::High => write!(f, "HIGH"),
            TrustLevel::Maximum => write!(f, "MAXIMUM"),
        }
    }
}

/// Social relationship between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SocialRelationship {
    /// Unique relationship identifier
    pub id: Uuid,
    /// Source node identifier
    pub source_node: String,
    /// Target node identifier  
    pub target_node: String,
    /// Relationship type
    pub relationship_type: RelationshipType,
    /// Current trust level
    pub trust_level: TrustLevel,
    /// Trust score (0.0-1.0)
    pub trust_score: f64,
    /// Number of successful collaborations
    pub successful_collaborations: u64,
    /// Number of failed collaborations
    pub failed_collaborations: u64,
    /// Total resources shared
    pub total_resources_shared: f64,
    /// Reciprocity score (how much partner shares back)
    pub reciprocity_score: f64,
    /// Relationship establishment timestamp
    pub established_at: DateTime<Utc>,
    /// Last interaction timestamp
    pub last_interaction: DateTime<Utc>,
    /// Relationship status
    pub status: RelationshipStatus,
}

/// Types of social relationships
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Friendship relationship - temporary, mutual benefit
    Friendship,
    /// Buddy relationship - permanent, default cooperation
    Buddy,
    /// Acquaintance - limited trust and cooperation
    Acquaintance,
    /// Rival - competitive relationship
    Rival,
}

impl RelationshipType {
    /// Get default trust level for relationship type
    pub fn default_trust_level(&self) -> TrustLevel {
        match self {
            RelationshipType::Friendship => TrustLevel::Medium,
            RelationshipType::Buddy => TrustLevel::High,
            RelationshipType::Acquaintance => TrustLevel::Low,
            RelationshipType::Rival => TrustLevel::None,
        }
    }
    
    /// Get cooperation willingness (0.0-1.0)
    pub fn cooperation_willingness(&self) -> f64 {
        match self {
            RelationshipType::Friendship => 0.8,
            RelationshipType::Buddy => 0.95,
            RelationshipType::Acquaintance => 0.4,
            RelationshipType::Rival => 0.1,
        }
    }
}

/// Relationship operational status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipStatus {
    /// Relationship is being established
    Establishing,
    /// Relationship is active
    Active,
    /// Relationship is temporarily suspended
    Suspended,
    /// Relationship is being terminated
    Terminating,
    /// Relationship has been terminated
    Terminated,
}

/// Resource sharing transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingTransaction {
    /// Transaction identifier
    pub id: Uuid,
    /// Relationship this transaction belongs to
    pub relationship_id: Uuid,
    /// Node providing resources
    pub provider_node: String,
    /// Node receiving resources
    pub receiver_node: String,
    /// Resource type shared
    pub resource_type: String,
    /// Amount shared
    pub amount: f64,
    /// Transaction timestamp
    pub timestamp: DateTime<Utc>,
    /// Transaction status
    pub status: TransactionStatus,
    /// Expected reciprocation
    pub expected_reciprocation: Option<f64>,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// Transaction is pending
    Pending,
    /// Transaction completed successfully
    Completed,
    /// Transaction failed
    Failed,
    /// Transaction was cancelled
    Cancelled,
}

/// Friendship Node - Implements tick-host symbiotic network behavior
/// 
/// Nodes with same or nearby addresses prioritize helping each other in job assignments.
/// Implements reciprocal relationship management where assistance is tracked and repaid.
pub struct FriendshipNode {
    /// Node identifier
    pub id: String,
    /// Node address for proximity-based relationships
    pub address: String,
    /// Active friendship relationships
    relationships: Arc<DashMap<String, SocialRelationship>>,
    /// Sharing transaction history
    transaction_history: Arc<RwLock<VecDeque<SharingTransaction>>>,
    /// Social metrics
    metrics: Arc<RwLock<SocialMetrics>>,
    /// Configuration
    config: SocialConfig,
    /// Relationship event broadcaster
    relationship_events: broadcast::Sender<RelationshipEvent>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Buddy Node - Implements primate grooming network behavior
/// 
/// Default (not on-demand) mutual compute resource sharing between paired nodes.
/// Provides built-in fault tolerance through automatic resource backup and redundancy.
pub struct BuddyNode {
    /// Node identifier
    pub id: String,
    /// Buddy partner node
    buddy_partner: Arc<RwLock<Option<String>>>,
    /// Buddy relationships (can have multiple buddies)
    buddy_relationships: Arc<DashMap<String, BuddyRelationship>>,
    /// Resource sharing state
    sharing_state: Arc<RwLock<BuddyState>>,
    /// Social metrics
    metrics: Arc<RwLock<SocialMetrics>>,
    /// Configuration
    config: SocialConfig,
    /// Buddy event broadcaster
    buddy_events: broadcast::Sender<BuddyEvent>,
    /// Running state
    running: Arc<parking_lot::RwLock<bool>>,
}

/// Configuration for social nodes
#[derive(Debug, Clone)]
pub struct SocialConfig {
    /// Maximum number of relationships per node
    pub max_relationships: usize,
    /// Trust decay rate per day (0.0-1.0)
    pub trust_decay_rate: f64,
    /// Minimum trust score for cooperation
    pub min_cooperation_trust: f64,
    /// Reciprocity tracking window (hours)
    pub reciprocity_window_hours: u64,
    /// Transaction history size
    pub transaction_history_size: usize,
    /// Enable automatic relationship management
    pub auto_relationship_management: bool,
    /// Relationship evaluation interval
    pub evaluation_interval: Duration,
}

impl Default for SocialConfig {
    fn default() -> Self {
        Self {
            max_relationships: 10,
            trust_decay_rate: crate::constants::TRUST_DECAY_RATE,
            min_cooperation_trust: 0.3,
            reciprocity_window_hours: 24,
            transaction_history_size: 1000,
            auto_relationship_management: true,
            evaluation_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Buddy relationship specific information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuddyRelationship {
    /// Base social relationship
    pub relationship: SocialRelationship,
    /// Default resource sharing percentage (0.0-1.0)
    pub default_sharing_percentage: f64,
    /// Automatic backup enabled
    pub auto_backup: bool,
    /// Fault tolerance level (1-5)
    pub fault_tolerance_level: u8,
    /// Last resource sync timestamp
    pub last_sync: DateTime<Utc>,
}

/// Current buddy sharing state
#[derive(Debug, Default)]
struct BuddyState {
    /// Resources currently shared with buddies
    shared_resources: HashMap<String, f64>,
    /// Resources received from buddies
    received_resources: HashMap<String, f64>,
    /// Active backup operations
    active_backups: HashSet<String>,
    /// Pending sync operations
    pending_syncs: HashSet<String>,
}

/// Relationship event information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationshipEvent {
    /// Event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: RelationshipEventType,
    /// Source node
    pub source_node: String,
    /// Target node
    pub target_node: String,
    /// Relationship affected
    pub relationship_id: Option<Uuid>,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event details
    pub details: String,
}

/// Types of relationship events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipEventType {
    /// Relationship established
    Established,
    /// Trust level changed
    TrustChanged,
    /// Resource sharing occurred
    ResourceShared,
    /// Reciprocation received
    ReciprocatedSharing,
    /// Relationship suspended
    Suspended,
    /// Relationship terminated
    Terminated,
}

/// Buddy-specific events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuddyEvent {
    /// Event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: BuddyEventType,
    /// Buddy nodes involved
    pub nodes: Vec<String>,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event details
    pub details: String,
}

/// Types of buddy events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuddyEventType {
    /// Buddy partnership formed
    PartnershipFormed,
    /// Default sharing activated
    DefaultSharingActivated,
    /// Backup operation started
    BackupStarted,
    /// Backup operation completed
    BackupCompleted,
    /// Resource sync performed
    ResourceSynced,
    /// Fault tolerance activated
    FaultToleranceActivated,
}

impl FriendshipNode {
    /// Create a new friendship node
    pub fn new(id: String, address: String, config: SocialConfig) -> Self {
        let (relationship_events, _) = broadcast::channel(1000);
        
        Self {
            id,
            address,
            relationships: Arc::new(DashMap::new()),
            transaction_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.transaction_history_size))),
            metrics: Arc::new(RwLock::new(SocialMetrics::default())),
            config,
            relationship_events,
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the friendship node
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::social_error("Friendship node already running"));
            }
            *running = true;
        }
        
        info!("Starting friendship node: {} at address {}", self.id, self.address);
        
        // Start relationship management if enabled
        if self.config.auto_relationship_management {
            self.start_relationship_management().await;
        }
        
        Ok(())
    }
    
    /// Stop the friendship node
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping friendship node: {}", self.id);
        Ok(())
    }
    
    /// Establish a friendship relationship
    pub async fn establish_friendship(&self, target_node: String) -> ResourceResult<Uuid> {
        // Check if relationship already exists
        if self.relationships.contains_key(&target_node) {
            return Err(ResourceError::social_error("Friendship already exists"));
        }
        
        // Check relationship limit
        if self.relationships.len() >= self.config.max_relationships {
            return Err(ResourceError::social_error("Maximum relationships reached"));
        }
        
        let relationship = SocialRelationship {
            id: Uuid::new_v4(),
            source_node: self.id.clone(),
            target_node: target_node.clone(),
            relationship_type: RelationshipType::Friendship,
            trust_level: RelationshipType::Friendship.default_trust_level(),
            trust_score: RelationshipType::Friendship.default_trust_level().to_numeric(),
            successful_collaborations: 0,
            failed_collaborations: 0,
            total_resources_shared: 0.0,
            reciprocity_score: 0.5, // Start neutral
            established_at: Utc::now(),
            last_interaction: Utc::now(),
            status: RelationshipStatus::Active,
        };
        
        let relationship_id = relationship.id;
        self.relationships.insert(target_node.clone(), relationship);
        
        // Broadcast event
        let event = RelationshipEvent {
            id: Uuid::new_v4(),
            event_type: RelationshipEventType::Established,
            source_node: self.id.clone(),
            target_node: target_node.clone(),
            relationship_id: Some(relationship_id),
            timestamp: Utc::now(),
            details: "Friendship relationship established".to_string(),
        };
        
        if let Err(e) = self.relationship_events.send(event) {
            warn!("Failed to broadcast relationship event: {}", e);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_relationships += 1;
            metrics.active_friendships += 1;
        }
        
        info!("Established friendship with {}", target_node);
        Ok(relationship_id)
    }
    
    /// Share resources with a friend
    pub async fn share_resources(
        &self,
        target_node: String,
        resource_type: String,
        amount: f64,
    ) -> ResourceResult<Uuid> {
        // Get relationship
        let relationship = self.relationships
            .get(&target_node)
            .ok_or_else(|| ResourceError::social_error("No friendship relationship exists"))?;
        
        // Check trust level
        if relationship.trust_score < self.config.min_cooperation_trust {
            return Err(ResourceError::social_error("Insufficient trust for resource sharing"));
        }
        
        // Create sharing transaction
        let transaction = SharingTransaction {
            id: Uuid::new_v4(),
            relationship_id: relationship.id,
            provider_node: self.id.clone(),
            receiver_node: target_node.clone(),
            resource_type: resource_type.clone(),
            amount,
            timestamp: Utc::now(),
            status: TransactionStatus::Completed,
            expected_reciprocation: Some(amount * relationship.reciprocity_score),
        };
        
        let transaction_id = transaction.id;
        
        // Store transaction
        {
            let mut history = self.transaction_history.write();
            history.push_back(transaction);
            
            // Maintain history size
            while history.len() > self.config.transaction_history_size {
                history.pop_front();
            }
        }
        
        // Update relationship
        if let Some(mut relationship_entry) = self.relationships.get_mut(&target_node) {
            let relationship = relationship_entry.value_mut();
            relationship.total_resources_shared += amount;
            relationship.last_interaction = Utc::now();
            relationship.successful_collaborations += 1;
            
            // Update trust based on successful sharing
            relationship.trust_score = (relationship.trust_score + 0.05).min(1.0);
            relationship.trust_level = TrustLevel::from_numeric(relationship.trust_score);
        }
        
        // Broadcast event
        let event = RelationshipEvent {
            id: Uuid::new_v4(),
            event_type: RelationshipEventType::ResourceShared,
            source_node: self.id.clone(),
            target_node: target_node.clone(),
            relationship_id: Some(relationship.id),
            timestamp: Utc::now(),
            details: format!("Shared {} units of {}", amount, resource_type),
        };
        
        if let Err(e) = self.relationship_events.send(event) {
            warn!("Failed to broadcast sharing event: {}", e);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_resources_shared += amount;
            metrics.successful_shares += 1;
        }
        
        info!("Shared {} units of {} with friend {}", amount, resource_type, target_node);
        Ok(transaction_id)
    }
    
    /// Process reciprocal sharing received from a friend
    pub async fn receive_reciprocation(
        &self,
        source_node: String,
        resource_type: String,
        amount: f64,
    ) -> ResourceResult<()> {
        // Update relationship reciprocity
        if let Some(mut relationship_entry) = self.relationships.get_mut(&source_node) {
            let relationship = relationship_entry.value_mut();
            
            // Calculate reciprocity improvement
            let expected_total = relationship.total_resources_shared;
            if expected_total > 0.0 {
                let reciprocity_ratio = amount / expected_total;
                relationship.reciprocity_score = (relationship.reciprocity_score * 0.8 + reciprocity_ratio * 0.2).min(1.0);
                
                // Improve trust based on reciprocation
                relationship.trust_score = (relationship.trust_score + reciprocity_ratio * 0.1).min(1.0);
                relationship.trust_level = TrustLevel::from_numeric(relationship.trust_score);
            }
            
            relationship.last_interaction = Utc::now();
        }
        
        // Broadcast event
        let event = RelationshipEvent {
            id: Uuid::new_v4(),
            event_type: RelationshipEventType::ReciprocatedSharing,
            source_node: source_node.clone(),
            target_node: self.id.clone(),
            relationship_id: self.relationships.get(&source_node).map(|r| r.id),
            timestamp: Utc::now(),
            details: format!("Received {} units of {} as reciprocation", amount, resource_type),
        };
        
        if let Err(e) = self.relationship_events.send(event) {
            warn!("Failed to broadcast reciprocation event: {}", e);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_resources_received += amount;
            metrics.successful_reciprocations += 1;
        }
        
        Ok(())
    }
    
    /// Get priority nodes for resource allocation (proximity-based)
    pub fn get_priority_nodes(&self) -> Vec<String> {
        let mut priority_nodes = Vec::new();
        
        // First priority: same address nodes
        for relationship_entry in self.relationships.iter() {
            let relationship = relationship_entry.value();
            if relationship.trust_level >= TrustLevel::Medium {
                // Check address proximity (simplified - would use real address comparison)
                if self.is_nearby_address(&relationship.target_node) {
                    priority_nodes.push(relationship.target_node.clone());
                }
            }
        }
        
        // Second priority: high trust nodes regardless of address
        for relationship_entry in self.relationships.iter() {
            let relationship = relationship_entry.value();
            if relationship.trust_level == TrustLevel::Maximum {
                if !priority_nodes.contains(&relationship.target_node) {
                    priority_nodes.push(relationship.target_node.clone());
                }
            }
        }
        
        priority_nodes
    }
    
    /// Get all relationships
    pub fn get_relationships(&self) -> Vec<SocialRelationship> {
        self.relationships.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Get social metrics
    pub fn get_metrics(&self) -> SocialMetrics {
        self.metrics.read().clone()
    }
    
    /// Subscribe to relationship events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<RelationshipEvent> {
        self.relationship_events.subscribe()
    }
    
    // Private methods
    
    async fn start_relationship_management(&self) {
        let config = self.config.clone();
        let running = Arc::clone(&self.running);
        let relationships = Arc::clone(&self.relationships);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.evaluation_interval);
            
            while *running.read() {
                interval.tick().await;
                
                // Apply trust decay
                for mut relationship_entry in relationships.iter_mut() {
                    let relationship = relationship_entry.value_mut();
                    
                    // Calculate time-based trust decay
                    let hours_since_interaction = Utc::now()
                        .signed_duration_since(relationship.last_interaction)
                        .num_hours() as f64;
                    
                    if hours_since_interaction > 24.0 {
                        let decay_factor = config.trust_decay_rate.powf(hours_since_interaction / 24.0);
                        relationship.trust_score *= decay_factor;
                        relationship.trust_level = TrustLevel::from_numeric(relationship.trust_score);
                    }
                }
            }
        });
    }
    
    fn is_nearby_address(&self, _target_node: &str) -> bool {
        // Simplified proximity check - would implement real address comparison
        true
    }
}

impl BuddyNode {
    /// Create a new buddy node
    pub fn new(id: String, config: SocialConfig) -> Self {
        let (buddy_events, _) = broadcast::channel(1000);
        
        Self {
            id,
            buddy_partner: Arc::new(RwLock::new(None)),
            buddy_relationships: Arc::new(DashMap::new()),
            sharing_state: Arc::new(RwLock::new(BuddyState::default())),
            metrics: Arc::new(RwLock::new(SocialMetrics::default())),
            config,
            buddy_events,
            running: Arc::new(parking_lot::RwLock::new(false)),
        }
    }
    
    /// Start the buddy node
    pub async fn start(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            if *running {
                return Err(ResourceError::social_error("Buddy node already running"));
            }
            *running = true;
        }
        
        info!("Starting buddy node: {}", self.id);
        
        // Start automatic resource sharing
        self.start_automatic_sharing().await;
        
        Ok(())
    }
    
    /// Stop the buddy node
    pub async fn stop(&self) -> ResourceResult<()> {
        {
            let mut running = self.running.write();
            *running = false;
        }
        
        info!("Stopping buddy node: {}", self.id);
        Ok(())
    }
    
    /// Establish a buddy relationship
    pub async fn establish_buddy_relationship(
        &self,
        partner_node: String,
        sharing_percentage: f64,
        fault_tolerance_level: u8,
    ) -> ResourceResult<Uuid> {
        // Check if relationship already exists
        if self.buddy_relationships.contains_key(&partner_node) {
            return Err(ResourceError::social_error("Buddy relationship already exists"));
        }
        
        let base_relationship = SocialRelationship {
            id: Uuid::new_v4(),
            source_node: self.id.clone(),
            target_node: partner_node.clone(),
            relationship_type: RelationshipType::Buddy,
            trust_level: RelationshipType::Buddy.default_trust_level(),
            trust_score: RelationshipType::Buddy.default_trust_level().to_numeric(),
            successful_collaborations: 0,
            failed_collaborations: 0,
            total_resources_shared: 0.0,
            reciprocity_score: 1.0, // Buddies expect full reciprocity
            established_at: Utc::now(),
            last_interaction: Utc::now(),
            status: RelationshipStatus::Active,
        };
        
        let buddy_relationship = BuddyRelationship {
            relationship: base_relationship.clone(),
            default_sharing_percentage: sharing_percentage.clamp(0.0, 1.0),
            auto_backup: true,
            fault_tolerance_level: fault_tolerance_level.min(5),
            last_sync: Utc::now(),
        };
        
        let relationship_id = base_relationship.id;
        self.buddy_relationships.insert(partner_node.clone(), buddy_relationship);
        
        // Set as primary buddy if none exists
        {
            let mut primary_buddy = self.buddy_partner.write();
            if primary_buddy.is_none() {
                *primary_buddy = Some(partner_node.clone());
            }
        }
        
        // Broadcast event
        let event = BuddyEvent {
            id: Uuid::new_v4(),
            event_type: BuddyEventType::PartnershipFormed,
            nodes: vec![self.id.clone(), partner_node.clone()],
            timestamp: Utc::now(),
            details: format!("Buddy partnership formed with {}% default sharing", sharing_percentage * 100.0),
        };
        
        if let Err(e) = self.buddy_events.send(event) {
            warn!("Failed to broadcast buddy event: {}", e);
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write();
            metrics.total_relationships += 1;
            metrics.active_buddy_relationships += 1;
        }
        
        info!("Established buddy relationship with {} ({}% sharing, FT level {})", 
            partner_node, sharing_percentage * 100.0, fault_tolerance_level);
        
        Ok(relationship_id)
    }
    
    /// Activate default resource sharing with buddies
    pub async fn activate_default_sharing(&self) -> ResourceResult<()> {
        let mut sharing_state = self.sharing_state.write();
        
        for buddy_entry in self.buddy_relationships.iter() {
            let buddy = buddy_entry.value();
            let partner_node = &buddy.relationship.target_node;
            
            // Calculate default sharing amount (simplified)
            let sharing_amount = buddy.default_sharing_percentage;
            
            sharing_state.shared_resources.insert(partner_node.clone(), sharing_amount);
            
            // Activate backup if enabled
            if buddy.auto_backup {
                sharing_state.active_backups.insert(partner_node.clone());
            }
        }
        
        // Broadcast event
        let buddy_nodes: Vec<String> = self.buddy_relationships
            .iter()
            .map(|entry| entry.value().relationship.target_node.clone())
            .collect();
        
        let event = BuddyEvent {
            id: Uuid::new_v4(),
            event_type: BuddyEventType::DefaultSharingActivated,
            nodes: buddy_nodes,
            timestamp: Utc::now(),
            details: "Default resource sharing activated for all buddy relationships".to_string(),
        };
        
        if let Err(e) = self.buddy_events.send(event) {
            warn!("Failed to broadcast sharing activation event: {}", e);
        }
        
        info!("Activated default sharing with {} buddies", self.buddy_relationships.len());
        Ok(())
    }
    
    /// Perform automatic backup with buddy
    pub async fn backup_with_buddy(&self, buddy_node: String, backup_data: Vec<u8>) -> ResourceResult<()> {
        // Check if buddy relationship exists
        let buddy_relationship = self.buddy_relationships
            .get(&buddy_node)
            .ok_or_else(|| ResourceError::social_error("No buddy relationship exists"))?;
        
        // Check fault tolerance level
        if buddy_relationship.fault_tolerance_level < 3 {
            return Err(ResourceError::social_error("Insufficient fault tolerance level for backup"));
        }
        
        // Add to active backups
        {
            let mut sharing_state = self.sharing_state.write();
            sharing_state.active_backups.insert(buddy_node.clone());
        }
        
        // Broadcast event
        let event = BuddyEvent {
            id: Uuid::new_v4(),
            event_type: BuddyEventType::BackupStarted,
            nodes: vec![self.id.clone(), buddy_node.clone()],
            timestamp: Utc::now(),
            details: format!("Backup operation started ({} bytes)", backup_data.len()),
        };
        
        if let Err(e) = self.buddy_events.send(event) {
            warn!("Failed to broadcast backup event: {}", e);
        }
        
        // Simulate backup completion (would be actual backup operation)
        tokio::time::sleep(Duration::from_millis(100)).await;
        
        // Mark backup as completed
        let completion_event = BuddyEvent {
            id: Uuid::new_v4(),
            event_type: BuddyEventType::BackupCompleted,
            nodes: vec![self.id.clone(), buddy_node.clone()],
            timestamp: Utc::now(),
            details: "Backup operation completed successfully".to_string(),
        };
        
        if let Err(e) = self.buddy_events.send(completion_event) {
            warn!("Failed to broadcast backup completion event: {}", e);
        }
        
        info!("Completed backup with buddy {} ({} bytes)", buddy_node, backup_data.len());
        Ok(())
    }
    
    /// Get current primary buddy
    pub fn get_primary_buddy(&self) -> Option<String> {
        self.buddy_partner.read().clone()
    }
    
    /// Get all buddy relationships
    pub fn get_buddy_relationships(&self) -> Vec<BuddyRelationship> {
        self.buddy_relationships.iter().map(|entry| entry.value().clone()).collect()
    }
    
    /// Get current sharing state
    pub fn get_sharing_state(&self) -> HashMap<String, f64> {
        self.sharing_state.read().shared_resources.clone()
    }
    
    /// Get social metrics
    pub fn get_metrics(&self) -> SocialMetrics {
        self.metrics.read().clone()
    }
    
    /// Subscribe to buddy events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<BuddyEvent> {
        self.buddy_events.subscribe()
    }
    
    // Private methods
    
    async fn start_automatic_sharing(&self) {
        let running = Arc::clone(&self.running);
        let buddy_relationships = Arc::clone(&self.buddy_relationships);
        let buddy_events = self.buddy_events.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Check every minute
            
            while *running.read() {
                interval.tick().await;
                
                // Perform automatic resource synchronization
                for buddy_entry in buddy_relationships.iter() {
                    let buddy = buddy_entry.value();
                    
                    // Check if sync is needed (simplified)
                    let time_since_sync = Utc::now().signed_duration_since(buddy.last_sync);
                    
                    if time_since_sync > chrono::Duration::minutes(5) {
                        // Broadcast sync event
                        let event = BuddyEvent {
                            id: Uuid::new_v4(),
                            event_type: BuddyEventType::ResourceSynced,
                            nodes: vec![buddy.relationship.source_node.clone(), buddy.relationship.target_node.clone()],
                            timestamp: Utc::now(),
                            details: "Automatic resource synchronization performed".to_string(),
                        };
                        
                        if let Err(e) = buddy_events.send(event) {
                            debug!("Failed to broadcast sync event: {}", e);
                        }
                    }
                }
            }
        });
    }
}

/// Trust calculation utilities
pub struct TrustCalculator;

impl TrustCalculator {
    /// Calculate trust score with decay
    pub fn calculate_trust_with_decay(
        base_trust: f64,
        decay_factor: f64,
        time_elapsed_days: f64,
        interaction_bonus: f64,
    ) -> f64 {
        crate::math::trust_score_with_decay(base_trust, decay_factor, time_elapsed_days, interaction_bonus)
    }
    
    /// Update trust based on successful interaction
    pub fn update_trust_success(current_trust: f64, interaction_impact: f64) -> f64 {
        (current_trust + interaction_impact).min(1.0)
    }
    
    /// Update trust based on failed interaction  
    pub fn update_trust_failure(current_trust: f64, failure_penalty: f64) -> f64 {
        (current_trust - failure_penalty).max(0.0)
    }
    
    /// Calculate reciprocity score
    pub fn calculate_reciprocity(given: f64, received: f64) -> f64 {
        if given == 0.0 {
            0.5 // Neutral when nothing has been given
        } else {
            (received / given).min(1.0) // Cap at 1.0 for full reciprocity
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trust_level_conversion() {
        assert_eq!(TrustLevel::Maximum.to_numeric(), 1.0);
        assert_eq!(TrustLevel::None.to_numeric(), 0.0);
        
        assert_eq!(TrustLevel::from_numeric(0.9), TrustLevel::Maximum);
        assert_eq!(TrustLevel::from_numeric(0.1), TrustLevel::Low);
        
        assert!(TrustLevel::High.cooperation_multiplier() > TrustLevel::Low.cooperation_multiplier());
    }
    
    #[test]
    fn test_relationship_type_properties() {
        assert_eq!(RelationshipType::Buddy.default_trust_level(), TrustLevel::High);
        assert_eq!(RelationshipType::Rival.default_trust_level(), TrustLevel::None);
        
        assert!(RelationshipType::Buddy.cooperation_willingness() > RelationshipType::Acquaintance.cooperation_willingness());
    }
    
    #[tokio::test]
    async fn test_friendship_node_creation() {
        let config = SocialConfig::default();
        let node = FriendshipNode::new(
            "friend-test".to_string(),
            "127.0.0.1".to_string(),
            config,
        );
        
        assert_eq!(node.id, "friend-test");
        assert_eq!(node.address, "127.0.0.1");
        assert_eq!(node.get_relationships().len(), 0);
    }
    
    #[tokio::test]
    async fn test_buddy_node_creation() {
        let config = SocialConfig::default();
        let node = BuddyNode::new("buddy-test".to_string(), config);
        
        assert_eq!(node.id, "buddy-test");
        assert!(node.get_primary_buddy().is_none());
        assert_eq!(node.get_buddy_relationships().len(), 0);
    }
    
    #[tokio::test]
    async fn test_friendship_establishment() {
        let config = SocialConfig::default();
        let node = FriendshipNode::new(
            "friend-test".to_string(),
            "127.0.0.1".to_string(),
            config,
        );
        
        let relationship_id = node.establish_friendship("friend-target".to_string()).await.unwrap();
        assert!(!relationship_id.is_nil());
        
        let relationships = node.get_relationships();
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].target_node, "friend-target");
        assert_eq!(relationships[0].relationship_type, RelationshipType::Friendship);
    }
    
    #[tokio::test]
    async fn test_buddy_relationship_establishment() {
        let config = SocialConfig::default();
        let node = BuddyNode::new("buddy-test".to_string(), config);
        
        let relationship_id = node.establish_buddy_relationship(
            "buddy-partner".to_string(),
            0.5,
            4,
        ).await.unwrap();
        
        assert!(!relationship_id.is_nil());
        assert_eq!(node.get_primary_buddy(), Some("buddy-partner".to_string()));
        
        let relationships = node.get_buddy_relationships();
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].default_sharing_percentage, 0.5);
        assert_eq!(relationships[0].fault_tolerance_level, 4);
    }
    
    #[tokio::test]
    async fn test_resource_sharing() {
        let config = SocialConfig::default();
        let node = FriendshipNode::new(
            "friend-test".to_string(),
            "127.0.0.1".to_string(),
            config,
        );
        
        // Establish friendship first
        node.establish_friendship("friend-target".to_string()).await.unwrap();
        
        // Share resources
        let transaction_id = node.share_resources(
            "friend-target".to_string(),
            "cpu".to_string(),
            0.3,
        ).await.unwrap();
        
        assert!(!transaction_id.is_nil());
        
        let metrics = node.get_metrics();
        assert_eq!(metrics.successful_shares, 1);
        assert_eq!(metrics.total_resources_shared, 0.3);
    }
    
    #[test]
    fn test_trust_calculator() {
        let trust = TrustCalculator::calculate_trust_with_decay(1.0, 0.95, 1.0, 1.1);
        assert!(trust > 0.9);
        
        let updated_trust = TrustCalculator::update_trust_success(0.5, 0.2);
        assert_eq!(updated_trust, 0.7);
        
        let failed_trust = TrustCalculator::update_trust_failure(0.8, 0.3);
        assert_eq!(failed_trust, 0.5);
        
        let reciprocity = TrustCalculator::calculate_reciprocity(1.0, 0.8);
        assert_eq!(reciprocity, 0.8);
    }
    
    #[tokio::test]
    async fn test_buddy_backup() {
        let config = SocialConfig::default();
        let node = BuddyNode::new("buddy-test".to_string(), config);
        
        // Establish buddy relationship with high fault tolerance
        node.establish_buddy_relationship(
            "buddy-partner".to_string(),
            0.5,
            4,
        ).await.unwrap();
        
        // Perform backup
        let backup_data = vec![1, 2, 3, 4, 5];
        node.backup_with_buddy("buddy-partner".to_string(), backup_data).await.unwrap();
        
        // Check that backup was recorded (simplified verification)
        let sharing_state = node.get_sharing_state();
        assert!(!sharing_state.is_empty());
    }
}