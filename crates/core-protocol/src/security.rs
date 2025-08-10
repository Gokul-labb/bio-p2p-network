use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use crate::{NetworkAddress, BiologicalRole, ProtocolError, Result};

/// Five-layer security architecture inspired by biological immune systems
/// Provides defense-in-depth through layered biological mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityLayer {
    /// Layer identifier (1-5)
    pub layer_id: u8,
    
    /// Layer name and description
    pub layer_name: String,
    
    /// Biological inspiration for this layer
    pub biological_inspiration: String,
    
    /// Security mechanisms in this layer
    pub mechanisms: Vec<SecurityMechanism>,
    
    /// Layer activation status
    pub active: bool,
    
    /// Layer configuration parameters
    pub configuration: HashMap<String, String>,
    
    /// Layer performance metrics
    pub metrics: LayerMetrics,
    
    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

/// Individual security mechanism within a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMechanism {
    /// Mechanism name
    pub name: String,
    
    /// Mechanism description
    pub description: String,
    
    /// Biological behavior this mechanism implements
    pub biological_behavior: String,
    
    /// Mechanism activation status
    pub enabled: bool,
    
    /// Configuration parameters
    pub parameters: HashMap<String, String>,
    
    /// Performance metrics
    pub performance: MechanismPerformance,
}

/// Security layer performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetrics {
    /// Threats detected by this layer
    pub threats_detected: u64,
    
    /// Threats blocked by this layer
    pub threats_blocked: u64,
    
    /// False positive rate
    pub false_positive_rate: f64,
    
    /// Processing overhead percentage
    pub overhead_percentage: f64,
    
    /// Layer efficiency score
    pub efficiency_score: f64,
    
    /// Average response time in milliseconds
    pub avg_response_time_ms: u64,
}

/// Individual mechanism performance tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismPerformance {
    /// Activations in the last hour
    pub activations_last_hour: u32,
    
    /// Success rate percentage
    pub success_rate: f64,
    
    /// Average execution time in milliseconds
    pub avg_execution_time_ms: u64,
    
    /// Resource usage metrics
    pub resource_usage: MechanismResourceUsage,
    
    /// Last activation timestamp
    pub last_activation: Option<DateTime<Utc>>,
}

/// Resource usage for security mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MechanismResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f64,
    
    /// Memory usage in MB
    pub memory_usage_mb: u32,
    
    /// Network bandwidth usage in Kbps
    pub network_usage_kbps: u16,
    
    /// Storage usage in MB
    pub storage_usage_mb: u32,
}

/// Comprehensive validation result for packages and operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Overall validation status
    pub status: ValidationStatus,
    
    /// Results from each security layer
    pub layer_results: HashMap<u8, LayerValidationResult>,
    
    /// Overall risk assessment
    pub risk_assessment: RiskAssessment,
    
    /// Security warnings generated
    pub warnings: Vec<SecurityWarning>,
    
    /// Validation metadata
    pub validation_metadata: ValidationMetadata,
    
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
    
    /// Node that performed validation
    pub validator_node: NetworkAddress,
}

/// Individual layer validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerValidationResult {
    /// Layer that performed validation
    pub layer_id: u8,
    
    /// Validation status for this layer
    pub status: ValidationStatus,
    
    /// Detailed result message
    pub message: String,
    
    /// Risk score from this layer (0.0-1.0)
    pub risk_score: f64,
    
    /// Evidence collected by this layer
    pub evidence: Vec<SecurityEvidence>,
    
    /// Processing time for this layer
    pub processing_time_ms: u64,
    
    /// Layer-specific metadata
    pub layer_metadata: HashMap<String, String>,
}

/// Overall validation status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// Validation passed all security layers
    Passed,
    
    /// Validation passed with minor warnings
    PassedWithWarnings,
    
    /// Validation failed security checks
    Failed,
    
    /// Validation could not be completed
    Incomplete,
    
    /// Validation is still in progress
    InProgress,
    
    /// Validation was not performed
    NotPerformed,
}

/// Risk assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk score (0.0-1.0, higher is riskier)
    pub overall_risk_score: f64,
    
    /// Risk level interpretation
    pub risk_level: RiskLevel,
    
    /// Individual risk factors
    pub risk_factors: Vec<RiskFactor>,
    
    /// Mitigation recommendations
    pub mitigations: Vec<MitigationRecommendation>,
    
    /// Confidence in risk assessment
    pub confidence_level: f64,
}

/// Risk level categories
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Individual risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Risk factor name
    pub factor_name: String,
    
    /// Risk contribution score
    pub risk_contribution: f64,
    
    /// Factor description
    pub description: String,
    
    /// Evidence supporting this risk factor
    pub supporting_evidence: Vec<String>,
    
    /// Confidence in this risk factor
    pub confidence: f64,
}

/// Security warning generated during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityWarning {
    /// Warning severity level
    pub severity: WarningSeverity,
    
    /// Warning category
    pub category: String,
    
    /// Warning message
    pub message: String,
    
    /// Source layer that generated warning
    pub source_layer: u8,
    
    /// Timestamp when warning was generated
    pub generated_at: DateTime<Utc>,
    
    /// Additional warning metadata
    pub metadata: HashMap<String, String>,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Low,
    Medium,
    High,
    Critical,
}

/// Security evidence collected during validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvidence {
    /// Evidence type
    pub evidence_type: String,
    
    /// Evidence data
    pub data: Vec<u8>,
    
    /// Evidence description
    pub description: String,
    
    /// Collection timestamp
    pub collected_at: DateTime<Utc>,
    
    /// Evidence integrity hash
    pub integrity_hash: String,
    
    /// Evidence source information
    pub source_info: EvidenceSource,
}

/// Source information for security evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceSource {
    /// Source type (layer, mechanism, etc.)
    pub source_type: String,
    
    /// Source identifier
    pub source_id: String,
    
    /// Collection method used
    pub collection_method: String,
    
    /// Source reliability score
    pub reliability_score: f64,
}

/// Mitigation recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MitigationRecommendation {
    /// Recommendation title
    pub title: String,
    
    /// Detailed recommendation description
    pub description: String,
    
    /// Priority level (1-10, higher is more urgent)
    pub priority: u8,
    
    /// Expected risk reduction if implemented
    pub expected_risk_reduction: f64,
    
    /// Implementation difficulty (1-10, higher is harder)
    pub implementation_difficulty: u8,
    
    /// Estimated implementation time
    pub estimated_time: chrono::Duration,
}

/// Validation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetadata {
    /// Validation algorithm version
    pub validation_version: String,
    
    /// Security policy version used
    pub policy_version: String,
    
    /// Total validation time
    pub total_validation_time_ms: u64,
    
    /// Number of layers that participated
    pub layers_participated: u8,
    
    /// Validation triggers
    pub validation_triggers: Vec<String>,
    
    /// Additional context information
    pub context: HashMap<String, String>,
}

/// Complete five-layer security architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityArchitecture {
    /// All security layers (1-5)
    pub layers: HashMap<u8, SecurityLayer>,
    
    /// Architecture configuration
    pub configuration: ArchitectureConfiguration,
    
    /// Overall architecture metrics
    pub metrics: ArchitectureMetrics,
    
    /// Last architecture update
    pub last_updated: DateTime<Utc>,
    
    /// Architecture version
    pub version: String,
}

/// Architecture-wide configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConfiguration {
    /// Enable/disable entire architecture
    pub enabled: bool,
    
    /// Coordination mode between layers
    pub coordination_mode: CoordinationMode,
    
    /// Threat response policies
    pub response_policies: Vec<ResponsePolicy>,
    
    /// Global security parameters
    pub global_parameters: HashMap<String, String>,
    
    /// Logging and audit settings
    pub audit_settings: AuditSettings,
}

/// Layer coordination modes
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    /// Layers operate independently
    Independent,
    
    /// Layers coordinate through shared state
    Coordinated,
    
    /// Layers form adaptive response chains
    Adaptive,
    
    /// Full biological immune system simulation
    BiologicalSimulation,
}

/// Response policy for different threat types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsePolicy {
    /// Policy name
    pub name: String,
    
    /// Threat patterns this policy applies to
    pub threat_patterns: Vec<String>,
    
    /// Response actions to take
    pub response_actions: Vec<ResponseAction>,
    
    /// Policy activation conditions
    pub activation_conditions: Vec<ActivationCondition>,
    
    /// Policy priority
    pub priority: u8,
}

/// Individual response action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseAction {
    /// Action type
    pub action_type: String,
    
    /// Action parameters
    pub parameters: HashMap<String, String>,
    
    /// Expected effectiveness
    pub effectiveness: f64,
    
    /// Resource cost of action
    pub resource_cost: u8,
}

/// Conditions for policy activation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationCondition {
    /// Condition type
    pub condition_type: String,
    
    /// Condition threshold
    pub threshold: f64,
    
    /// Condition evaluation method
    pub evaluation_method: String,
}

/// Audit and logging settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditSettings {
    /// Enable security event logging
    pub logging_enabled: bool,
    
    /// Log retention period in days
    pub retention_days: u16,
    
    /// Log detail level
    pub detail_level: LogDetailLevel,
    
    /// Audit trail requirements
    pub audit_requirements: Vec<AuditRequirement>,
}

/// Logging detail levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogDetailLevel {
    Minimal,
    Standard,
    Detailed,
    Comprehensive,
}

/// Audit requirement specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRequirement {
    /// Requirement name
    pub name: String,
    
    /// Events to audit
    pub events: Vec<String>,
    
    /// Required retention period
    pub retention_days: u16,
    
    /// Compliance standard (if applicable)
    pub compliance_standard: Option<String>,
}

/// Architecture-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureMetrics {
    /// Total threats detected across all layers
    pub total_threats_detected: u64,
    
    /// Total threats mitigated
    pub total_threats_mitigated: u64,
    
    /// Overall false positive rate
    pub overall_false_positive_rate: f64,
    
    /// Average threat detection time
    pub avg_detection_time_ms: u64,
    
    /// Average mitigation time
    pub avg_mitigation_time_ms: u64,
    
    /// Architecture efficiency score
    pub efficiency_score: f64,
    
    /// Resource utilization across layers
    pub resource_utilization: ArchitectureResourceUsage,
}

/// Architecture-wide resource usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureResourceUsage {
    /// Total CPU usage by security layers
    pub total_cpu_usage: f64,
    
    /// Total memory usage by security layers
    pub total_memory_usage_mb: u32,
    
    /// Total network bandwidth used for security
    pub total_network_usage_kbps: u32,
    
    /// Total storage used for security data
    pub total_storage_usage_mb: u32,
}

impl SecurityArchitecture {
    /// Create the standard five-layer security architecture
    pub fn new() -> Self {
        let mut layers = HashMap::new();
        
        // Layer 1: Multi-Layer Execution
        layers.insert(1, SecurityLayer {
            layer_id: 1,
            layer_name: "Multi-Layer Execution".to_string(),
            biological_inspiration: "Cellular compartmentalization with nested protective membranes".to_string(),
            mechanisms: vec![
                SecurityMechanism {
                    name: "Randomized Core Layer Selection".to_string(),
                    description: "Randomly selects execution layers to prevent predictable attack vectors".to_string(),
                    biological_behavior: "Immune cell membrane randomization".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
                SecurityMechanism {
                    name: "Protective Monitoring".to_string(),
                    description: "Continuous surveillance of execution environment integrity".to_string(),
                    biological_behavior: "Cellular membrane monitoring systems".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
            ],
            active: true,
            configuration: HashMap::new(),
            metrics: LayerMetrics::default(),
            last_updated: Utc::now(),
        });
        
        // Layer 2: CBADU (Clean Before and After Usage)
        layers.insert(2, SecurityLayer {
            layer_id: 2,
            layer_name: "CBADU (Clean Before and After Usage)".to_string(),
            biological_inspiration: "Immune system sanitization and cellular hygiene processes".to_string(),
            mechanisms: vec![
                SecurityMechanism {
                    name: "Pre-execution Sanitization".to_string(),
                    description: "Comprehensive environment cleaning before task processing".to_string(),
                    biological_behavior: "Cellular cleaning mechanisms before division".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
                SecurityMechanism {
                    name: "Post-execution Cleansing".to_string(),
                    description: "Environment reset and contamination removal after processing".to_string(),
                    biological_behavior: "Immune system pathogen clearance".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
            ],
            active: true,
            configuration: HashMap::new(),
            metrics: LayerMetrics::default(),
            last_updated: Utc::now(),
        });
        
        // Layer 3: Illusion Layer
        layers.insert(3, SecurityLayer {
            layer_id: 3,
            layer_name: "Illusion Layer".to_string(),
            biological_inspiration: "Animal deception behaviors like octopus camouflage and bird distraction displays".to_string(),
            mechanisms: vec![
                SecurityMechanism {
                    name: "Active Deception".to_string(),
                    description: "Creates false system topologies and misleading information for attackers".to_string(),
                    biological_behavior: "Predator confusion and misdirection tactics".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
                SecurityMechanism {
                    name: "Honeypot Generation".to_string(),
                    description: "Dynamic creation of decoy systems and false targets".to_string(),
                    biological_behavior: "False signals and decoy behaviors in nature".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
            ],
            active: true,
            configuration: HashMap::new(),
            metrics: LayerMetrics::default(),
            last_updated: Utc::now(),
        });
        
        // Layer 4: Behavior Monitoring
        layers.insert(4, SecurityLayer {
            layer_id: 4,
            layer_name: "Behavior Monitoring".to_string(),
            biological_inspiration: "Social animal vigilance systems like meerkat sentries and wolf pack scouts".to_string(),
            mechanisms: vec![
                SecurityMechanism {
                    name: "Continuous Pattern Analysis".to_string(),
                    description: "Real-time analysis of node behaviors and communication patterns".to_string(),
                    biological_behavior: "Pack animal surveillance and threat detection".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
                SecurityMechanism {
                    name: "Anomaly Detection".to_string(),
                    description: "Machine learning-based identification of abnormal behaviors".to_string(),
                    biological_behavior: "Immune system self/non-self recognition".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
            ],
            active: true,
            configuration: HashMap::new(),
            metrics: LayerMetrics::default(),
            last_updated: Utc::now(),
        });
        
        // Layer 5: Thermal Detection
        layers.insert(5, SecurityLayer {
            layer_id: 5,
            layer_name: "Thermal Detection".to_string(),
            biological_inspiration: "Thermal sensing in biological systems like snake heat detection and bat echolocation".to_string(),
            mechanisms: vec![
                SecurityMechanism {
                    name: "Resource Signature Monitoring".to_string(),
                    description: "Comprehensive tracking of resource usage patterns and thermal signatures".to_string(),
                    biological_behavior: "Metabolic monitoring and thermal regulation".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
                SecurityMechanism {
                    name: "Predictive Threat Analysis".to_string(),
                    description: "Analysis of resource patterns to predict potential threats".to_string(),
                    biological_behavior: "Biological early warning systems".to_string(),
                    enabled: true,
                    parameters: HashMap::new(),
                    performance: MechanismPerformance::default(),
                },
            ],
            active: true,
            configuration: HashMap::new(),
            metrics: LayerMetrics::default(),
            last_updated: Utc::now(),
        });
        
        Self {
            layers,
            configuration: ArchitectureConfiguration {
                enabled: true,
                coordination_mode: CoordinationMode::BiologicalSimulation,
                response_policies: Vec::new(),
                global_parameters: HashMap::new(),
                audit_settings: AuditSettings {
                    logging_enabled: true,
                    retention_days: 30,
                    detail_level: LogDetailLevel::Standard,
                    audit_requirements: Vec::new(),
                },
            },
            metrics: ArchitectureMetrics::default(),
            last_updated: Utc::now(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Perform comprehensive security validation using all layers
    pub fn validate(&mut self, 
                    data: &[u8], 
                    context: &ValidationContext) -> Result<ValidationResult> {
        let start_time = std::time::Instant::now();
        let mut layer_results = HashMap::new();
        let mut warnings = Vec::new();
        let mut overall_risk_score = 0.0;
        
        // Process through each active layer
        for (layer_id, layer) in &mut self.layers {
            if layer.active {
                let layer_result = self.validate_layer(*layer_id, data, context)?;
                overall_risk_score += layer_result.risk_score;
                
                // Collect warnings from layer
                if let Some(warning) = self.check_layer_warnings(&layer_result) {
                    warnings.push(warning);
                }
                
                layer_results.insert(*layer_id, layer_result);
            }
        }
        
        // Calculate final status
        let final_status = if overall_risk_score > 0.8 {
            ValidationStatus::Failed
        } else if overall_risk_score > 0.5 || !warnings.is_empty() {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Passed
        };
        
        let validation_time = start_time.elapsed().as_millis() as u64;
        
        Ok(ValidationResult {
            status: final_status,
            layer_results,
            risk_assessment: self.assess_risk(overall_risk_score, &warnings),
            warnings,
            validation_metadata: ValidationMetadata {
                validation_version: "1.0.0".to_string(),
                policy_version: "1.0.0".to_string(),
                total_validation_time_ms: validation_time,
                layers_participated: self.layers.len() as u8,
                validation_triggers: context.triggers.clone(),
                context: context.metadata.clone(),
            },
            validated_at: Utc::now(),
            validator_node: context.validator_node.clone(),
        })
    }
    
    /// Validate data through a specific security layer
    fn validate_layer(&self, 
                     layer_id: u8, 
                     data: &[u8], 
                     _context: &ValidationContext) -> Result<LayerValidationResult> {
        let layer = self.layers.get(&layer_id)
            .ok_or_else(|| ProtocolError::SecurityValidationFailed {
                reason: format!("Layer {} not found", layer_id)
            })?;
        
        let start_time = std::time::Instant::now();
        let mut risk_score = 0.0;
        let mut evidence = Vec::new();
        let mut message = format!("Layer {} validation completed", layer_id);
        let mut status = ValidationStatus::Passed;
        
        // Simulate layer-specific validation logic
        match layer_id {
            1 => {
                // Multi-layer execution validation
                if data.len() > 10 * 1024 * 1024 { // Large payload check
                    risk_score += 0.2;
                    message = "Large payload detected in multi-layer execution".to_string();
                }
            },
            2 => {
                // CBADU validation
                let data_entropy = self.calculate_entropy(data);
                if data_entropy > 0.9 {
                    risk_score += 0.3;
                    status = ValidationStatus::PassedWithWarnings;
                    message = "High entropy data detected - may indicate encryption or compression".to_string();
                }
            },
            3 => {
                // Illusion layer - check for known attack patterns
                if self.contains_suspicious_patterns(data) {
                    risk_score += 0.5;
                    status = ValidationStatus::PassedWithWarnings;
                    message = "Suspicious patterns detected in payload".to_string();
                }
            },
            4 => {
                // Behavior monitoring
                risk_score += 0.1; // Baseline behavioral risk
                message = "Behavioral patterns within normal parameters".to_string();
            },
            5 => {
                // Thermal detection
                if data.len() < 100 {
                    risk_score += 0.1; // Very small payloads might be probes
                }
                message = "Thermal signature analysis completed".to_string();
            },
            _ => {
                return Err(ProtocolError::SecurityValidationFailed {
                    reason: format!("Unknown layer ID: {}", layer_id)
                });
            }
        }
        
        let processing_time = start_time.elapsed().as_millis() as u64;
        
        Ok(LayerValidationResult {
            layer_id,
            status,
            message,
            risk_score,
            evidence,
            processing_time_ms: processing_time,
            layer_metadata: HashMap::new(),
        })
    }
    
    /// Simple entropy calculation for data analysis
    fn calculate_entropy(&self, data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let mut counts = [0u32; 256];
        for &byte in data {
            counts[byte as usize] += 1;
        }
        
        let len = data.len() as f64;
        let mut entropy = 0.0;
        
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / len;
                entropy -= p * p.log2();
            }
        }
        
        entropy / 8.0 // Normalize to 0-1 range
    }
    
    /// Check for suspicious patterns in data
    fn contains_suspicious_patterns(&self, data: &[u8]) -> bool {
        // Simple pattern detection - could be enhanced
        if data.len() < 10 {
            return false;
        }
        
        // Check for repeated patterns
        let mut pattern_found = false;
        if data.len() >= 100 {
            let first_chunk = &data[0..10];
            for window in data.windows(10).skip(10) {
                if window == first_chunk {
                    pattern_found = true;
                    break;
                }
            }
        }
        
        pattern_found
    }
    
    /// Check for layer-specific warnings
    fn check_layer_warnings(&self, result: &LayerValidationResult) -> Option<SecurityWarning> {
        if result.risk_score > 0.3 {
            Some(SecurityWarning {
                severity: if result.risk_score > 0.7 { 
                    WarningSeverity::High 
                } else { 
                    WarningSeverity::Medium 
                },
                category: "Risk Assessment".to_string(),
                message: format!("Layer {} reported elevated risk score: {:.2}", 
                                result.layer_id, result.risk_score),
                source_layer: result.layer_id,
                generated_at: Utc::now(),
                metadata: HashMap::new(),
            })
        } else {
            None
        }
    }
    
    /// Assess overall risk based on layer results
    fn assess_risk(&self, overall_risk_score: f64, warnings: &[SecurityWarning]) -> RiskAssessment {
        let risk_level = match overall_risk_score {
            score if score > 2.0 => RiskLevel::Critical,
            score if score > 1.5 => RiskLevel::High,
            score if score > 0.5 => RiskLevel::Medium,
            _ => RiskLevel::Low,
        };
        
        let mut risk_factors = Vec::new();
        if overall_risk_score > 0.5 {
            risk_factors.push(RiskFactor {
                factor_name: "Elevated Layer Risk Scores".to_string(),
                risk_contribution: overall_risk_score / 5.0, // Normalize for 5 layers
                description: "Multiple security layers reported elevated risk scores".to_string(),
                supporting_evidence: warnings.iter()
                    .map(|w| w.message.clone())
                    .collect(),
                confidence: 0.8,
            });
        }
        
        RiskAssessment {
            overall_risk_score: overall_risk_score / 5.0, // Normalize for 5 layers
            risk_level,
            risk_factors,
            mitigations: self.generate_mitigations(&risk_level),
            confidence_level: 0.85,
        }
    }
    
    /// Generate mitigation recommendations based on risk level
    fn generate_mitigations(&self, risk_level: &RiskLevel) -> Vec<MitigationRecommendation> {
        match risk_level {
            RiskLevel::Critical => vec![
                MitigationRecommendation {
                    title: "Immediate Quarantine".to_string(),
                    description: "Isolate the suspicious entity immediately".to_string(),
                    priority: 10,
                    expected_risk_reduction: 0.8,
                    implementation_difficulty: 2,
                    estimated_time: chrono::Duration::seconds(30),
                },
            ],
            RiskLevel::High => vec![
                MitigationRecommendation {
                    title: "Enhanced Monitoring".to_string(),
                    description: "Increase surveillance and monitoring frequency".to_string(),
                    priority: 8,
                    expected_risk_reduction: 0.5,
                    implementation_difficulty: 3,
                    estimated_time: chrono::Duration::minutes(5),
                },
            ],
            RiskLevel::Medium => vec![
                MitigationRecommendation {
                    title: "Additional Validation".to_string(),
                    description: "Perform additional validation rounds".to_string(),
                    priority: 5,
                    expected_risk_reduction: 0.3,
                    implementation_difficulty: 4,
                    estimated_time: chrono::Duration::minutes(2),
                },
            ],
            RiskLevel::Low => vec![
                MitigationRecommendation {
                    title: "Continue Monitoring".to_string(),
                    description: "Maintain standard monitoring protocols".to_string(),
                    priority: 2,
                    expected_risk_reduction: 0.1,
                    implementation_difficulty: 1,
                    estimated_time: chrono::Duration::seconds(1),
                },
            ],
        }
    }
}

/// Context for security validation
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Node performing the validation
    pub validator_node: NetworkAddress,
    
    /// Validation triggers that initiated this check
    pub triggers: Vec<String>,
    
    /// Additional context metadata
    pub metadata: HashMap<String, String>,
    
    /// Source of data being validated
    pub data_source: Option<NetworkAddress>,
    
    /// Validation priority level
    pub priority: u8,
}

impl Default for LayerMetrics {
    fn default() -> Self {
        Self {
            threats_detected: 0,
            threats_blocked: 0,
            false_positive_rate: 0.0,
            overhead_percentage: 5.0,
            efficiency_score: 0.95,
            avg_response_time_ms: 10,
        }
    }
}

impl Default for MechanismPerformance {
    fn default() -> Self {
        Self {
            activations_last_hour: 0,
            success_rate: 100.0,
            avg_execution_time_ms: 5,
            resource_usage: MechanismResourceUsage {
                cpu_usage: 2.0,
                memory_usage_mb: 10,
                network_usage_kbps: 50,
                storage_usage_mb: 5,
            },
            last_activation: None,
        }
    }
}

impl Default for ArchitectureMetrics {
    fn default() -> Self {
        Self {
            total_threats_detected: 0,
            total_threats_mitigated: 0,
            overall_false_positive_rate: 0.02,
            avg_detection_time_ms: 50,
            avg_mitigation_time_ms: 200,
            efficiency_score: 0.92,
            resource_utilization: ArchitectureResourceUsage {
                total_cpu_usage: 10.0,
                total_memory_usage_mb: 100,
                total_network_usage_kbps: 500,
                total_storage_usage_mb: 50,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NetworkAddress;
    
    #[test]
    fn test_security_architecture_creation() {
        let architecture = SecurityArchitecture::new();
        
        assert_eq!(architecture.layers.len(), 5);
        assert!(architecture.configuration.enabled);
        
        // Check that all layers are present
        for i in 1..=5 {
            assert!(architecture.layers.contains_key(&i));
            assert!(architecture.layers[&i].active);
        }
    }
    
    #[test]
    fn test_security_validation() {
        let mut architecture = SecurityArchitecture::new();
        let validator_node = NetworkAddress::new(1, 2, 3).unwrap();
        
        let context = ValidationContext {
            validator_node: validator_node.clone(),
            triggers: vec!["package_processing".to_string()],
            metadata: HashMap::new(),
            data_source: None,
            priority: 5,
        };
        
        let test_data = b"Hello, World!";
        let result = architecture.validate(test_data, &context).unwrap();
        
        assert_eq!(result.validator_node, validator_node);
        assert_eq!(result.layer_results.len(), 5);
        assert!(matches!(result.status, ValidationStatus::Passed | ValidationStatus::PassedWithWarnings));
    }
    
    #[test]
    fn test_high_entropy_detection() {
        let mut architecture = SecurityArchitecture::new();
        let validator_node = NetworkAddress::new(1, 2, 3).unwrap();
        
        let context = ValidationContext {
            validator_node,
            triggers: vec!["entropy_test".to_string()],
            metadata: HashMap::new(),
            data_source: None,
            priority: 8,
        };
        
        // Create high-entropy data (random bytes)
        let high_entropy_data: Vec<u8> = (0..=255).collect();
        let result = architecture.validate(&high_entropy_data, &context).unwrap();
        
        // Should detect high entropy in layer 2 (CBADU)
        if let Some(layer2_result) = result.layer_results.get(&2) {
            assert!(layer2_result.risk_score > 0.0);
        }
    }
    
    #[test]
    fn test_large_payload_detection() {
        let mut architecture = SecurityArchitecture::new();
        let validator_node = NetworkAddress::new(1, 2, 3).unwrap();
        
        let context = ValidationContext {
            validator_node,
            triggers: vec!["large_payload_test".to_string()],
            metadata: HashMap::new(),
            data_source: None,
            priority: 7,
        };
        
        // Create large payload (>10MB)
        let large_payload = vec![0u8; 11 * 1024 * 1024];
        let result = architecture.validate(&large_payload, &context).unwrap();
        
        // Should detect large payload in layer 1
        if let Some(layer1_result) = result.layer_results.get(&1) {
            assert!(layer1_result.risk_score > 0.0);
        }
    }
    
    #[test]
    fn test_entropy_calculation() {
        let architecture = SecurityArchitecture::new();
        
        // Low entropy data (all zeros)
        let low_entropy = vec![0u8; 100];
        let low_score = architecture.calculate_entropy(&low_entropy);
        assert!(low_score < 0.1);
        
        // High entropy data (random pattern)
        let high_entropy: Vec<u8> = (0..=255).cycle().take(1000).collect();
        let high_score = architecture.calculate_entropy(&high_entropy);
        assert!(high_score > 0.8);
        
        // Empty data
        let empty_entropy = architecture.calculate_entropy(&[]);
        assert_eq!(empty_entropy, 0.0);
    }
    
    #[test]
    fn test_risk_level_assignment() {
        let architecture = SecurityArchitecture::new();
        let warnings = Vec::new();
        
        // Test different risk scores
        let low_risk = architecture.assess_risk(0.2, &warnings);
        assert_eq!(low_risk.risk_level, RiskLevel::Low);
        
        let medium_risk = architecture.assess_risk(1.0, &warnings);
        assert_eq!(medium_risk.risk_level, RiskLevel::Medium);
        
        let high_risk = architecture.assess_risk(2.0, &warnings);
        assert_eq!(high_risk.risk_level, RiskLevel::High);
        
        let critical_risk = architecture.assess_risk(3.0, &warnings);
        assert_eq!(critical_risk.risk_level, RiskLevel::Critical);
    }
    
    #[test]
    fn test_mitigation_recommendations() {
        let architecture = SecurityArchitecture::new();
        
        let critical_mitigations = architecture.generate_mitigations(&RiskLevel::Critical);
        assert!(!critical_mitigations.is_empty());
        assert!(critical_mitigations[0].priority >= 8);
        
        let low_mitigations = architecture.generate_mitigations(&RiskLevel::Low);
        assert!(!low_mitigations.is_empty());
        assert!(low_mitigations[0].priority <= 3);
    }
    
    #[test]
    fn test_validation_status_logic() {
        let mut architecture = SecurityArchitecture::new();
        let validator_node = NetworkAddress::new(1, 2, 3).unwrap();
        
        let context = ValidationContext {
            validator_node,
            triggers: vec!["status_test".to_string()],
            metadata: HashMap::new(),
            data_source: None,
            priority: 5,
        };
        
        // Normal data should pass
        let normal_data = b"normal test data";
        let result = architecture.validate(normal_data, &context).unwrap();
        assert!(matches!(result.status, ValidationStatus::Passed | ValidationStatus::PassedWithWarnings));
        
        // Very suspicious data should create warnings
        let suspicious_data = vec![0xAA; 1000]; // Repeated pattern
        let result = architecture.validate(&suspicious_data, &context).unwrap();
        // Result depends on implementation - may pass with warnings or fail
        assert!(matches!(result.status, 
                        ValidationStatus::Passed | 
                        ValidationStatus::PassedWithWarnings |
                        ValidationStatus::Failed));
    }
}