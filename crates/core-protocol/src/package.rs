use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use crate::{NetworkAddress, BiologicalRole, ProtocolError, Result, ThermalSignature};

/// Compute package implementing cellular biology-inspired task encapsulation
/// This mirrors biological cellular transport and processing mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputePackage {
    /// Unique package identifier
    pub id: Uuid,
    
    /// Package creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Current package state in the 9-step lifecycle
    pub state: PackageState,
    
    /// Source node address
    pub source: NetworkAddress,
    
    /// Destination node address (if specific routing required)
    pub destination: Option<NetworkAddress>,
    
    /// Package payload data
    pub payload: PackagePayload,
    
    /// Security validation results
    pub security_validation: SecurityValidation,
    
    /// Processing requirements and constraints
    pub processing_requirements: ProcessingRequirements,
    
    /// Package lifecycle tracking
    pub lifecycle: PackageLifecycle,
    
    /// Thermal signatures collected during processing
    pub thermal_signatures: Vec<ThermalSignature>,
    
    /// Package metadata and routing information
    pub metadata: HashMap<String, String>,
}

/// 9-step package processing lifecycle inspired by cellular biology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageLifecycle {
    /// Step 1: Package creation and initial security validation
    pub creation_cleaning: Option<DateTime<Utc>>,
    
    /// Step 2: Containerization with security scanning and updates
    pub containerization_analysis: Option<DateTime<Utc>>,
    
    /// Step 3: Secure data integration into package containers
    pub data_loading: Option<DateTime<Utc>>,
    
    /// Step 4: Distribution queue preparation with priority assignment
    pub distribution_queue_loading: Option<DateTime<Utc>>,
    
    /// Step 5: Package distribution to target computational nodes
    pub package_distribution: Option<DateTime<Utc>>,
    
    /// Step 6: Processing strategy assignment based on requirements
    pub processing_strategy_assignment: Option<DateTime<Utc>>,
    
    /// Step 7: Encrypted processing within action nodes
    pub action_node_processing: Option<DateTime<Utc>>,
    
    /// Step 8: Result return with comprehensive reporting
    pub result_return: Option<DateTime<Utc>>,
    
    /// Step 9: Package analysis and optimization feedback
    pub package_analysis: Option<DateTime<Utc>>,
}

/// Current state of package in the processing lifecycle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PackageState {
    /// Step 1: Initial creation and cleaning
    Created,
    /// Step 2: Containerization and security scanning
    Containerized,
    /// Step 3: Data loading and integration
    DataLoaded,
    /// Step 4: Queued for distribution
    Queued,
    /// Step 5: Distributed to processing nodes
    Distributed,
    /// Step 6: Processing strategy assigned
    StrategyAssigned,
    /// Step 7: Currently being processed
    Processing,
    /// Step 8: Results being returned
    ReturningResults,
    /// Step 9: Analysis and optimization
    Analyzing,
    /// Processing completed successfully
    Completed,
    /// Processing failed with error
    Failed,
    /// Package cancelled or aborted
    Cancelled,
}

/// Package payload containing the actual computational work
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackagePayload {
    /// Type of computational task
    pub task_type: TaskType,
    
    /// Task-specific data and parameters
    pub data: Vec<u8>,
    
    /// Expected result format
    pub result_format: ResultFormat,
    
    /// Task priority (0-10, higher is more urgent)
    pub priority: u8,
    
    /// Maximum allowed processing time
    pub max_processing_time: chrono::Duration,
    
    /// Resource requirements for processing
    pub resource_requirements: ResourceRequirements,
}

/// Types of computational tasks supported by the network
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    /// Machine learning model training
    MLTraining,
    /// Machine learning inference
    MLInference,
    /// General computation
    Computation,
    /// Data processing and transformation
    DataProcessing,
    /// Scientific simulation
    Simulation,
    /// Cryptographic operations
    Cryptographic,
    /// Image/video processing
    MediaProcessing,
    /// Database operations
    Database,
    /// Network analysis
    NetworkAnalysis,
    /// Custom task type
    Custom(String),
}

/// Expected result format specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResultFormat {
    /// Binary data result
    Binary,
    /// JSON-formatted result
    Json,
    /// Text result
    Text,
    /// Numerical result
    Numerical,
    /// Image result
    Image,
    /// Model weights/parameters
    Model,
    /// Custom format
    Custom(String),
}

/// Resource requirements for package processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU cores required
    pub min_cpu_cores: u8,
    
    /// Minimum memory in MB
    pub min_memory_mb: u32,
    
    /// GPU required (optional)
    pub gpu_required: bool,
    
    /// Minimum GPU memory in MB (if GPU required)
    pub min_gpu_memory_mb: Option<u32>,
    
    /// Storage requirements in MB
    pub storage_mb: u32,
    
    /// Network bandwidth requirements in Mbps
    pub bandwidth_mbps: u16,
    
    /// Specialized hardware requirements
    pub specialized_hardware: Vec<String>,
}

/// Processing requirements and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRequirements {
    /// Biological roles preferred for processing
    pub preferred_roles: Vec<BiologicalRole>,
    
    /// Biological roles to avoid
    pub excluded_roles: Vec<BiologicalRole>,
    
    /// Geographic constraints (regions to prefer/avoid)
    pub geographic_constraints: Option<GeographicConstraints>,
    
    /// Security level required
    pub security_level: SecurityLevel,
    
    /// Privacy requirements
    pub privacy_requirements: PrivacyRequirements,
    
    /// Redundancy requirements (number of parallel processing nodes)
    pub redundancy_level: u8,
    
    /// Consensus requirements for result validation
    pub consensus_required: bool,
    
    /// Custom processing constraints
    pub custom_constraints: HashMap<String, String>,
}

/// Geographic processing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicConstraints {
    /// Preferred regions for processing
    pub preferred_regions: Vec<u16>,
    
    /// Excluded regions
    pub excluded_regions: Vec<u16>,
    
    /// Data locality requirements
    pub data_locality_required: bool,
    
    /// Maximum distance from source (in network hops)
    pub max_distance_hops: Option<u32>,
}

/// Security level requirements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Basic security validation
    Basic,
    /// Enhanced security with behavioral monitoring
    Enhanced,
    /// High security with multi-layer validation
    High,
    /// Critical security with all protection mechanisms
    Critical,
}

/// Privacy requirements for package processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyRequirements {
    /// Encryption required during processing
    pub encryption_required: bool,
    
    /// Data must not leave specified regions
    pub data_residency_regions: Vec<u16>,
    
    /// Trusted nodes only
    pub trusted_nodes_only: bool,
    
    /// Minimum trust score required
    pub min_trust_score: Option<f64>,
    
    /// Audit trail required
    pub audit_trail_required: bool,
}

/// Security validation results for package
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityValidation {
    /// Overall validation status
    pub status: ValidationStatus,
    
    /// Individual security layer results
    pub layer_results: HashMap<String, LayerValidationResult>,
    
    /// Security warnings or notes
    pub warnings: Vec<String>,
    
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
    
    /// Validating node information
    pub validator: NetworkAddress,
}

/// Package validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed all checks
    Passed,
    /// Validation passed with warnings
    PassedWithWarnings,
    /// Validation failed
    Failed,
    /// Validation pending
    Pending,
    /// Validation not performed
    NotValidated,
}

/// Individual security layer validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerValidationResult {
    /// Layer validation status
    pub status: ValidationStatus,
    
    /// Detailed result message
    pub message: String,
    
    /// Risk score (0.0-1.0, higher is riskier)
    pub risk_score: f64,
    
    /// Validation timestamp
    pub timestamp: DateTime<Utc>,
}

impl ComputePackage {
    /// Create a new compute package
    pub fn new(
        source: NetworkAddress,
        payload: PackagePayload,
        processing_requirements: ProcessingRequirements,
    ) -> Result<Self> {
        // Validate package size
        if payload.data.len() > crate::MAX_PACKAGE_SIZE {
            return Err(ProtocolError::PackageTooLarge {
                size: payload.data.len(),
                max_size: crate::MAX_PACKAGE_SIZE,
            });
        }
        
        let now = Utc::now();
        let id = Uuid::new_v4();
        
        let mut lifecycle = PackageLifecycle::new();
        lifecycle.creation_cleaning = Some(now);
        
        Ok(Self {
            id,
            created_at: now,
            state: PackageState::Created,
            source,
            destination: None,
            payload,
            security_validation: SecurityValidation {
                status: ValidationStatus::NotValidated,
                layer_results: HashMap::new(),
                warnings: Vec::new(),
                validated_at: now,
                validator: source.clone(),
            },
            processing_requirements,
            lifecycle,
            thermal_signatures: Vec::new(),
            metadata: HashMap::new(),
        })
    }
    
    /// Transition to next state in the lifecycle
    pub fn transition_state(&mut self, new_state: PackageState) -> Result<()> {
        // Validate state transition
        if !self.is_valid_transition(&self.state, &new_state) {
            return Err(ProtocolError::PackageLifecycleViolation {
                expected: format!("valid transition from {:?}", self.state),
                actual: format!("{:?}", new_state),
            });
        }
        
        let now = Utc::now();
        self.state = new_state.clone();
        
        // Update lifecycle timestamps
        match new_state {
            PackageState::Containerized => self.lifecycle.containerization_analysis = Some(now),
            PackageState::DataLoaded => self.lifecycle.data_loading = Some(now),
            PackageState::Queued => self.lifecycle.distribution_queue_loading = Some(now),
            PackageState::Distributed => self.lifecycle.package_distribution = Some(now),
            PackageState::StrategyAssigned => self.lifecycle.processing_strategy_assignment = Some(now),
            PackageState::Processing => self.lifecycle.action_node_processing = Some(now),
            PackageState::ReturningResults => self.lifecycle.result_return = Some(now),
            PackageState::Analyzing => self.lifecycle.package_analysis = Some(now),
            _ => {}
        }
        
        Ok(())
    }
    
    /// Check if state transition is valid
    fn is_valid_transition(&self, current: &PackageState, new: &PackageState) -> bool {
        use PackageState::*;
        
        match (current, new) {
            (Created, Containerized) => true,
            (Containerized, DataLoaded) => true,
            (DataLoaded, Queued) => true,
            (Queued, Distributed) => true,
            (Distributed, StrategyAssigned) => true,
            (StrategyAssigned, Processing) => true,
            (Processing, ReturningResults) => true,
            (ReturningResults, Analyzing) => true,
            (Analyzing, Completed) => true,
            
            // Error states can be reached from any state
            (_, Failed) => true,
            (_, Cancelled) => true,
            
            // Invalid transitions
            _ => false,
        }
    }
    
    /// Add thermal signature from processing
    pub fn add_thermal_signature(&mut self, signature: ThermalSignature) {
        self.thermal_signatures.push(signature);
    }
    
    /// Get processing time so far
    pub fn processing_time(&self) -> chrono::Duration {
        Utc::now() - self.created_at
    }
    
    /// Check if package has exceeded maximum processing time
    pub fn is_expired(&self) -> bool {
        self.processing_time() > self.payload.max_processing_time
    }
    
    /// Update security validation
    pub fn update_security_validation(&mut self, 
                                     status: ValidationStatus,
                                     validator: NetworkAddress,
                                     layer_results: HashMap<String, LayerValidationResult>) {
        self.security_validation = SecurityValidation {
            status,
            layer_results,
            warnings: self.security_validation.warnings.clone(),
            validated_at: Utc::now(),
            validator,
        };
    }
    
    /// Add security warning
    pub fn add_security_warning(&mut self, warning: String) {
        self.security_validation.warnings.push(warning);
    }
    
    /// Get package metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }
    
    /// Set package metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
    
    /// Calculate overall risk score based on all thermal signatures and security results
    pub fn risk_score(&self) -> f64 {
        let mut risk_scores = Vec::new();
        
        // Add security layer risk scores
        for result in self.security_validation.layer_results.values() {
            risk_scores.push(result.risk_score);
        }
        
        // Add thermal anomaly scores (if any)
        for thermal in &self.thermal_signatures {
            if thermal.is_anomalous() {
                risk_scores.push(thermal.anomaly_score());
            }
        }
        
        if risk_scores.is_empty() {
            0.0
        } else {
            risk_scores.iter().sum::<f64>() / risk_scores.len() as f64
        }
    }
}

impl PackageLifecycle {
    /// Create new package lifecycle
    pub fn new() -> Self {
        Self {
            creation_cleaning: None,
            containerization_analysis: None,
            data_loading: None,
            distribution_queue_loading: None,
            package_distribution: None,
            processing_strategy_assignment: None,
            action_node_processing: None,
            result_return: None,
            package_analysis: None,
        }
    }
    
    /// Get current lifecycle step
    pub fn current_step(&self) -> u8 {
        let steps = [
            self.creation_cleaning,
            self.containerization_analysis,
            self.data_loading,
            self.distribution_queue_loading,
            self.package_distribution,
            self.processing_strategy_assignment,
            self.action_node_processing,
            self.result_return,
            self.package_analysis,
        ];
        
        steps.iter().filter(|s| s.is_some()).count() as u8
    }
    
    /// Get total processing time across all steps
    pub fn total_processing_time(&self) -> Option<chrono::Duration> {
        if let (Some(start), Some(end)) = (self.creation_cleaning, self.package_analysis) {
            Some(end - start)
        } else {
            None
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            min_cpu_cores: 1,
            min_memory_mb: 512,
            gpu_required: false,
            min_gpu_memory_mb: None,
            storage_mb: 100,
            bandwidth_mbps: 10,
            specialized_hardware: Vec::new(),
        }
    }
}

impl Default for ProcessingRequirements {
    fn default() -> Self {
        Self {
            preferred_roles: Vec::new(),
            excluded_roles: Vec::new(),
            geographic_constraints: None,
            security_level: SecurityLevel::Basic,
            privacy_requirements: PrivacyRequirements {
                encryption_required: false,
                data_residency_regions: Vec::new(),
                trusted_nodes_only: false,
                min_trust_score: None,
                audit_trail_required: false,
            },
            redundancy_level: 1,
            consensus_required: false,
            custom_constraints: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkAddress;
    
    #[test]
    fn test_package_creation() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let payload = PackagePayload {
            task_type: TaskType::Computation,
            data: vec![1, 2, 3, 4],
            result_format: ResultFormat::Binary,
            priority: 5,
            max_processing_time: chrono::Duration::seconds(300),
            resource_requirements: ResourceRequirements::default(),
        };
        
        let package = ComputePackage::new(
            source.clone(),
            payload,
            ProcessingRequirements::default(),
        ).unwrap();
        
        assert_eq!(package.source, source);
        assert_eq!(package.state, PackageState::Created);
        assert!(package.lifecycle.creation_cleaning.is_some());
    }
    
    #[test]
    fn test_package_too_large() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let payload = PackagePayload {
            task_type: TaskType::Computation,
            data: vec![0; crate::MAX_PACKAGE_SIZE + 1],
            result_format: ResultFormat::Binary,
            priority: 5,
            max_processing_time: chrono::Duration::seconds(300),
            resource_requirements: ResourceRequirements::default(),
        };
        
        let result = ComputePackage::new(
            source,
            payload,
            ProcessingRequirements::default(),
        );
        
        assert!(matches!(result, Err(ProtocolError::PackageTooLarge { .. })));
    }
    
    #[test]
    fn test_state_transitions() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let payload = PackagePayload {
            task_type: TaskType::Computation,
            data: vec![1, 2, 3],
            result_format: ResultFormat::Binary,
            priority: 5,
            max_processing_time: chrono::Duration::seconds(300),
            resource_requirements: ResourceRequirements::default(),
        };
        
        let mut package = ComputePackage::new(
            source,
            payload,
            ProcessingRequirements::default(),
        ).unwrap();
        
        // Valid transitions
        assert!(package.transition_state(PackageState::Containerized).is_ok());
        assert!(package.transition_state(PackageState::DataLoaded).is_ok());
        assert!(package.transition_state(PackageState::Queued).is_ok());
        
        // Invalid transition
        assert!(package.transition_state(PackageState::Created).is_err());
    }
    
    #[test]
    fn test_lifecycle_tracking() {
        let lifecycle = PackageLifecycle::new();
        assert_eq!(lifecycle.current_step(), 0);
        
        let mut lifecycle = PackageLifecycle::new();
        lifecycle.creation_cleaning = Some(Utc::now());
        lifecycle.containerization_analysis = Some(Utc::now());
        assert_eq!(lifecycle.current_step(), 2);
    }
    
    #[test]
    fn test_risk_score_calculation() {
        let source = NetworkAddress::new(1, 2, 3).unwrap();
        let payload = PackagePayload {
            task_type: TaskType::Computation,
            data: vec![1, 2, 3],
            result_format: ResultFormat::Binary,
            priority: 5,
            max_processing_time: chrono::Duration::seconds(300),
            resource_requirements: ResourceRequirements::default(),
        };
        
        let mut package = ComputePackage::new(
            source.clone(),
            payload,
            ProcessingRequirements::default(),
        ).unwrap();
        
        // Initially no risk
        assert_eq!(package.risk_score(), 0.0);
        
        // Add security validation with risk
        let mut layer_results = HashMap::new();
        layer_results.insert("layer1".to_string(), LayerValidationResult {
            status: ValidationStatus::PassedWithWarnings,
            message: "Warning detected".to_string(),
            risk_score: 0.3,
            timestamp: Utc::now(),
        });
        
        package.update_security_validation(
            ValidationStatus::PassedWithWarnings,
            source,
            layer_results,
        );
        
        assert_eq!(package.risk_score(), 0.3);
    }
}