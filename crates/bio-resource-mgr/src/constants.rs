//! Constants for Biological Resource Management
//! 
//! Defines biological-inspired constants and parameters used throughout
//! the resource management system.

/// Default scaling step size for resource adjustments (20%)
pub const DEFAULT_SCALING_STEP: f64 = 0.2;

/// Trust decay rate per day (5% daily decay)
pub const TRUST_DECAY_RATE: f64 = 0.95;

/// Crisis detection sensitivity threshold
pub const CRISIS_DETECTION_SENSITIVITY: f64 = 0.8;

/// Default allocation history window size
pub const ALLOCATION_HISTORY_WINDOW: usize = 1000;

/// Maximum compartments per node
pub const MAX_COMPARTMENTS_PER_NODE: usize = 10;

/// Minimum resource allocation percentage
pub const MIN_RESOURCE_ALLOCATION: f64 = 0.01;

/// Maximum resource allocation percentage
pub const MAX_RESOURCE_ALLOCATION: f64 = 0.95;

/// Default thermal monitoring interval (seconds)
pub const THERMAL_MONITORING_INTERVAL_SECS: u64 = 5;

/// Thermal signature cache size
pub const THERMAL_SIGNATURE_CACHE_SIZE: usize = 1000;

/// Social relationship timeout (hours)
pub const SOCIAL_RELATIONSHIP_TIMEOUT_HOURS: u64 = 48;

/// Maximum friendship relationships per node
pub const MAX_FRIENDSHIP_RELATIONSHIPS: usize = 15;

/// Maximum buddy relationships per node
pub const MAX_BUDDY_RELATIONSHIPS: usize = 3;

/// HAVOC response timeout (seconds)
pub const HAVOC_RESPONSE_TIMEOUT_SECS: u64 = 60;

/// Network stress calculation window size
pub const NETWORK_STRESS_WINDOW_SIZE: usize = 100;

/// Biological scaling bounds
pub const MIN_BIOLOGICAL_SCALING_FACTOR: f64 = 0.1;
pub const MAX_BIOLOGICAL_SCALING_FACTOR: f64 = 5.0;

/// Energy efficiency baseline
pub const BASELINE_ENERGY_EFFICIENCY: f64 = 0.7;

/// Compartment rebalancing threshold
pub const COMPARTMENT_REBALANCING_THRESHOLD: f64 = 0.1;

/// Resource provider quality threshold
pub const MIN_PROVIDER_QUALITY: f64 = 0.3;

/// Default cooperation multipliers by trust level
pub const TRUST_NONE_MULTIPLIER: f64 = 0.5;
pub const TRUST_LOW_MULTIPLIER: f64 = 0.8;
pub const TRUST_MEDIUM_MULTIPLIER: f64 = 1.0;
pub const TRUST_HIGH_MULTIPLIER: f64 = 1.3;
pub const TRUST_MAXIMUM_MULTIPLIER: f64 = 1.5;