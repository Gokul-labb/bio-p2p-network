use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::{NetworkAddress, ProtocolError, Result, constants};

/// Reputation score calculation based on biological trust mechanisms
/// Formula: TaskCompletionScore + Avg.Lifetime + AnomalyScore - No.ofSystemShutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Task completion score (0-100 points)
    pub task_completion_score: u32,
    
    /// Average lifetime score in hours converted to points (0-50 points max)
    pub average_lifetime_hours: f64,
    
    /// Anomaly score based on behavioral consistency (0-30 points max)
    pub anomaly_score: u32,
    
    /// Number of unexpected system shutdowns (negative impact)
    pub system_shutdowns: u32,
    
    /// Calculated total reputation score
    pub total_score: i64,
    
    /// Score calculation timestamp
    pub calculated_at: DateTime<Utc>,
    
    /// Score calculation details for transparency
    pub calculation_details: ScoreCalculationDetails,
}

/// Trust score calculation based on service delivery performance
/// Formula: ServiceTime - PromiseTimeAllotted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScore {
    /// Actual time taken to complete tasks (seconds)
    pub service_time_seconds: u64,
    
    /// Promised/committed time for task completion (seconds)
    pub promise_time_seconds: u64,
    
    /// Calculated trust score (positive = faster than promised)
    pub score: i64,
    
    /// Trust level interpretation
    pub trust_level: TrustLevel,
    
    /// Score calculation timestamp
    pub calculated_at: DateTime<Utc>,
    
    /// Historical performance tracking
    pub performance_history: PerformanceHistory,
}

/// Performance score calculation based on efficiency metrics
/// Formula: ExpectedTime - ActualTimeTaken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceScore {
    /// Expected processing time based on task complexity (seconds)
    pub expected_time_seconds: u64,
    
    /// Actual time taken for processing (seconds)
    pub actual_time_seconds: u64,
    
    /// Calculated performance score (positive = better than expected)
    pub score: i64,
    
    /// Performance level interpretation
    pub performance_level: PerformanceLevel,
    
    /// Score calculation timestamp
    pub calculated_at: DateTime<Utc>,
    
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Detailed score calculation breakdown for transparency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreCalculationDetails {
    /// Task completion weighted by complexity
    pub task_completion_breakdown: TaskCompletionBreakdown,
    
    /// Lifetime score conversion details
    pub lifetime_conversion: LifetimeConversion,
    
    /// Anomaly detection results
    pub anomaly_analysis: AnomalyAnalysis,
    
    /// Shutdown penalty application
    pub shutdown_penalties: ShutdownPenalties,
    
    /// Calculation metadata
    pub calculation_metadata: CalculationMetadata,
}

/// Task completion score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskCompletionBreakdown {
    /// Total tasks attempted
    pub total_tasks: u64,
    
    /// Successfully completed tasks
    pub completed_tasks: u64,
    
    /// Failed tasks
    pub failed_tasks: u64,
    
    /// Completion rate percentage
    pub completion_rate: f64,
    
    /// Weighted score based on task complexity
    pub complexity_weighted_score: u32,
    
    /// Task type distribution
    pub task_type_distribution: std::collections::HashMap<String, u32>,
}

/// Lifetime score conversion from hours to points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifetimeConversion {
    /// Total operational hours
    pub total_hours: f64,
    
    /// Continuous operational periods
    pub operational_periods: Vec<OperationalPeriod>,
    
    /// Lifetime stability score
    pub stability_score: f64,
    
    /// Converted lifetime points (capped at 50)
    pub lifetime_points: u32,
}

/// Operational period tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationalPeriod {
    /// Period start time
    pub start: DateTime<Utc>,
    
    /// Period end time (None if ongoing)
    pub end: Option<DateTime<Utc>>,
    
    /// Duration in hours
    pub duration_hours: f64,
    
    /// Tasks completed during period
    pub tasks_completed: u32,
    
    /// Interruption reason (if any)
    pub interruption_reason: Option<String>,
}

/// Anomaly analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAnalysis {
    /// Behavioral consistency score
    pub behavioral_consistency: f64,
    
    /// Security incident count
    pub security_incidents: u32,
    
    /// Performance anomalies detected
    pub performance_anomalies: Vec<PerformanceAnomaly>,
    
    /// Communication pattern anomalies
    pub communication_anomalies: Vec<CommunicationAnomaly>,
    
    /// Overall anomaly score (0-30 points)
    pub anomaly_points: u32,
}

/// Performance anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnomaly {
    /// Anomaly type description
    pub anomaly_type: String,
    
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,
    
    /// Severity level (0.0-1.0)
    pub severity: f64,
    
    /// Anomaly details
    pub details: String,
    
    /// Recovery status
    pub recovered: bool,
}

/// Communication pattern anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationAnomaly {
    /// Communication pattern type
    pub pattern_type: String,
    
    /// Deviation from normal pattern
    pub deviation_score: f64,
    
    /// Frequency of anomalous behavior
    pub frequency: u32,
    
    /// Impact on network operations
    pub network_impact: f64,
}

/// Shutdown penalty tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownPenalties {
    /// Unexpected shutdowns count
    pub unexpected_shutdowns: u32,
    
    /// Planned maintenance shutdowns
    pub planned_shutdowns: u32,
    
    /// Emergency shutdowns due to failures
    pub emergency_shutdowns: u32,
    
    /// Total penalty points applied
    pub total_penalty_points: i32,
    
    /// Shutdown pattern analysis
    pub shutdown_patterns: Vec<ShutdownPattern>,
}

/// Shutdown pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownPattern {
    /// Pattern description
    pub pattern_description: String,
    
    /// Pattern frequency
    pub frequency: u32,
    
    /// Pattern impact assessment
    pub impact_level: f64,
    
    /// Suggested improvements
    pub improvements: Vec<String>,
}

/// Calculation metadata for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalculationMetadata {
    /// Calculation version/algorithm
    pub calculation_version: String,
    
    /// Data sources used
    pub data_sources: Vec<String>,
    
    /// Calculation duration
    pub calculation_duration_ms: u64,
    
    /// Confidence level in calculation
    pub confidence_level: f64,
    
    /// Next calculation scheduled time
    pub next_calculation: DateTime<Utc>,
}

/// Trust level interpretation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Consistently delivers faster than promised
    Excellent,
    /// Usually delivers on time or slightly faster
    Good,
    /// Delivers exactly as promised
    Reliable,
    /// Sometimes delivers slower than promised
    Questionable,
    /// Frequently delivers slower than promised
    Poor,
    /// Consistently fails to meet promises
    Unreliable,
}

/// Performance level interpretation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceLevel {
    /// Exceptional performance, significantly better than expected
    Outstanding,
    /// Good performance, better than expected
    AboveAverage,
    /// Performance meets expectations
    Average,
    /// Performance below expectations
    BelowAverage,
    /// Poor performance, significantly worse than expected
    Poor,
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Historical trust scores
    pub historical_trust_scores: Vec<TrustScoreEntry>,
    
    /// Average trust score over time
    pub average_trust_score: f64,
    
    /// Trust score trend (improving/declining)
    pub trend: ScoreTrend,
    
    /// Reliability metrics
    pub reliability_metrics: ReliabilityMetrics,
}

/// Individual trust score entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScoreEntry {
    /// Score value
    pub score: i64,
    
    /// Timestamp of score
    pub timestamp: DateTime<Utc>,
    
    /// Task that generated this score
    pub task_context: String,
    
    /// Additional notes
    pub notes: Option<String>,
}

/// Score trend analysis
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreTrend {
    Improving,
    Stable,
    Declining,
    Volatile,
    Unknown,
}

/// Reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    /// Consistency in performance
    pub consistency_score: f64,
    
    /// Predictability of performance
    pub predictability_score: f64,
    
    /// Variance in delivery times
    pub delivery_variance: f64,
    
    /// Success rate percentage
    pub success_rate: f64,
}

/// Efficiency metrics for performance scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    
    /// Time efficiency compared to baseline
    pub time_efficiency: f64,
    
    /// Energy efficiency metrics
    pub energy_efficiency: f64,
    
    /// Cost efficiency assessment
    pub cost_efficiency: f64,
    
    /// Overall efficiency score
    pub overall_efficiency: f64,
}

impl ReputationScore {
    /// Calculate reputation score using biological trust formula
    pub fn calculate(
        task_completion_score: u32,
        average_lifetime_hours: f64,
        anomaly_score: u32,
        system_shutdowns: u32,
    ) -> Result<Self> {
        // Validate input parameters
        if task_completion_score > constants::MAX_TASK_COMPLETION_SCORE {
            return Err(ProtocolError::ReputationCalculationFailed {
                reason: format!("Task completion score {} exceeds maximum {}", 
                               task_completion_score, constants::MAX_TASK_COMPLETION_SCORE),
            });
        }
        
        if anomaly_score > constants::MAX_ANOMALY_SCORE {
            return Err(ProtocolError::ReputationCalculationFailed {
                reason: format!("Anomaly score {} exceeds maximum {}", 
                               anomaly_score, constants::MAX_ANOMALY_SCORE),
            });
        }
        
        // Convert lifetime hours to points (capped at MAX_LIFETIME_SCORE)
        let lifetime_points = (average_lifetime_hours / 24.0).min(constants::MAX_LIFETIME_SCORE as f64) as u32;
        
        // Calculate total score
        let total_score = task_completion_score as i64
            + lifetime_points as i64
            + anomaly_score as i64
            - (system_shutdowns as i64 * constants::SHUTDOWN_PENALTY.abs() as i64);
        
        Ok(Self {
            task_completion_score,
            average_lifetime_hours,
            anomaly_score,
            system_shutdowns,
            total_score,
            calculated_at: Utc::now(),
            calculation_details: Self::create_calculation_details(
                task_completion_score,
                lifetime_points,
                anomaly_score,
                system_shutdowns,
            ),
        })
    }
    
    /// Create detailed calculation breakdown
    fn create_calculation_details(
        task_completion: u32,
        lifetime_points: u32,
        anomaly_score: u32,
        shutdowns: u32,
    ) -> ScoreCalculationDetails {
        ScoreCalculationDetails {
            task_completion_breakdown: TaskCompletionBreakdown {
                total_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                completion_rate: 0.0,
                complexity_weighted_score: task_completion,
                task_type_distribution: std::collections::HashMap::new(),
            },
            lifetime_conversion: LifetimeConversion {
                total_hours: 0.0,
                operational_periods: Vec::new(),
                stability_score: 0.0,
                lifetime_points,
            },
            anomaly_analysis: AnomalyAnalysis {
                behavioral_consistency: 0.0,
                security_incidents: 0,
                performance_anomalies: Vec::new(),
                communication_anomalies: Vec::new(),
                anomaly_points: anomaly_score,
            },
            shutdown_penalties: ShutdownPenalties {
                unexpected_shutdowns: shutdowns,
                planned_shutdowns: 0,
                emergency_shutdowns: 0,
                total_penalty_points: -(shutdowns as i32 * constants::SHUTDOWN_PENALTY.abs()),
                shutdown_patterns: Vec::new(),
            },
            calculation_metadata: CalculationMetadata {
                calculation_version: "1.0".to_string(),
                data_sources: vec!["task_logs".to_string(), "system_metrics".to_string()],
                calculation_duration_ms: 10,
                confidence_level: 0.95,
                next_calculation: Utc::now() + chrono::Duration::hours(24),
            },
        }
    }
    
    /// Get reputation level interpretation
    pub fn reputation_level(&self) -> ReputationLevel {
        match self.total_score {
            s if s >= 150 => ReputationLevel::Excellent,
            s if s >= 100 => ReputationLevel::Good,
            s if s >= 50 => ReputationLevel::Average,
            s if s >= 0 => ReputationLevel::BelowAverage,
            _ => ReputationLevel::Poor,
        }
    }
    
    /// Check if node is trustworthy based on reputation
    pub fn is_trustworthy(&self) -> bool {
        self.total_score >= 50 && self.system_shutdowns <= 10
    }
}

impl TrustScore {
    /// Calculate trust score based on service delivery
    pub fn calculate(
        service_time_seconds: u64,
        promise_time_seconds: u64,
    ) -> Self {
        let score = promise_time_seconds as i64 - service_time_seconds as i64;
        
        let trust_level = match score {
            s if s >= 60 => TrustLevel::Excellent,    // 1+ minute faster
            s if s >= 10 => TrustLevel::Good,         // 10+ seconds faster
            s if s >= -10 => TrustLevel::Reliable,    // Within 10 seconds
            s if s >= -60 => TrustLevel::Questionable, // Up to 1 minute slower
            s if s >= -300 => TrustLevel::Poor,       // Up to 5 minutes slower
            _ => TrustLevel::Unreliable,              // More than 5 minutes slower
        };
        
        Self {
            service_time_seconds,
            promise_time_seconds,
            score,
            trust_level,
            calculated_at: Utc::now(),
            performance_history: PerformanceHistory {
                historical_trust_scores: Vec::new(),
                average_trust_score: score as f64,
                trend: ScoreTrend::Unknown,
                reliability_metrics: ReliabilityMetrics {
                    consistency_score: 0.0,
                    predictability_score: 0.0,
                    delivery_variance: 0.0,
                    success_rate: 100.0,
                },
            },
        }
    }
    
    /// Update trust score with new measurement
    pub fn update(&mut self, service_time: u64, promise_time: u64) {
        let new_score = promise_time as i64 - service_time as i64;
        
        // Add to history
        self.performance_history.historical_trust_scores.push(TrustScoreEntry {
            score: new_score,
            timestamp: Utc::now(),
            task_context: "task_execution".to_string(),
            notes: None,
        });
        
        // Update current score
        self.service_time_seconds = service_time;
        self.promise_time_seconds = promise_time;
        self.score = new_score;
        self.calculated_at = Utc::now();
        
        // Update trust level
        self.trust_level = match new_score {
            s if s >= 60 => TrustLevel::Excellent,
            s if s >= 10 => TrustLevel::Good,
            s if s >= -10 => TrustLevel::Reliable,
            s if s >= -60 => TrustLevel::Questionable,
            s if s >= -300 => TrustLevel::Poor,
            _ => TrustLevel::Unreliable,
        };
        
        // Update average
        let scores: Vec<i64> = self.performance_history.historical_trust_scores
            .iter().map(|entry| entry.score).collect();
        if !scores.is_empty() {
            self.performance_history.average_trust_score = 
                scores.iter().sum::<i64>() as f64 / scores.len() as f64;
        }
        
        // Analyze trend (simplified)
        if scores.len() >= 3 {
            let recent_avg = scores[scores.len()-3..].iter().sum::<i64>() as f64 / 3.0;
            let overall_avg = self.performance_history.average_trust_score;
            
            self.performance_history.trend = if recent_avg > overall_avg + 10.0 {
                ScoreTrend::Improving
            } else if recent_avg < overall_avg - 10.0 {
                ScoreTrend::Declining
            } else {
                ScoreTrend::Stable
            };
        }
    }
    
    /// Get trust reliability assessment
    pub fn reliability_assessment(&self) -> f64 {
        match self.trust_level {
            TrustLevel::Excellent => 1.0,
            TrustLevel::Good => 0.8,
            TrustLevel::Reliable => 0.6,
            TrustLevel::Questionable => 0.4,
            TrustLevel::Poor => 0.2,
            TrustLevel::Unreliable => 0.0,
        }
    }
}

impl PerformanceScore {
    /// Calculate performance score based on efficiency
    pub fn calculate(
        expected_time_seconds: u64,
        actual_time_seconds: u64,
    ) -> Self {
        let score = expected_time_seconds as i64 - actual_time_seconds as i64;
        
        let performance_level = match score {
            s if s >= 300 => PerformanceLevel::Outstanding,  // 5+ minutes faster
            s if s >= 60 => PerformanceLevel::AboveAverage,  // 1+ minute faster
            s if s >= -60 => PerformanceLevel::Average,      // Within 1 minute
            s if s >= -300 => PerformanceLevel::BelowAverage, // Up to 5 minutes slower
            _ => PerformanceLevel::Poor,                     // More than 5 minutes slower
        };
        
        let efficiency = if expected_time_seconds > 0 {
            actual_time_seconds as f64 / expected_time_seconds as f64
        } else {
            1.0
        };
        
        Self {
            expected_time_seconds,
            actual_time_seconds,
            score,
            performance_level,
            calculated_at: Utc::now(),
            efficiency_metrics: EfficiencyMetrics {
                resource_efficiency: 1.0 / efficiency, // Inverse relationship
                time_efficiency: efficiency,
                energy_efficiency: 1.0 / efficiency,   // Assume faster = more efficient
                cost_efficiency: 1.0 / efficiency,
                overall_efficiency: 1.0 / efficiency,
            },
        }
    }
    
    /// Get performance rating (0.0-1.0)
    pub fn performance_rating(&self) -> f64 {
        match self.performance_level {
            PerformanceLevel::Outstanding => 1.0,
            PerformanceLevel::AboveAverage => 0.8,
            PerformanceLevel::Average => 0.6,
            PerformanceLevel::BelowAverage => 0.4,
            PerformanceLevel::Poor => 0.2,
        }
    }
}

/// Reputation level interpretation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReputationLevel {
    Excellent,
    Good,
    Average,
    BelowAverage,
    Poor,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reputation_score_calculation() {
        let score = ReputationScore::calculate(95, 720.0, 10, 2).unwrap();
        
        // Expected: 95 + 30 (720/24 capped at 50) + 10 - 10 = 125
        assert_eq!(score.total_score, 125);
        assert_eq!(score.reputation_level(), ReputationLevel::Good);
        assert!(score.is_trustworthy());
    }
    
    #[test]
    fn test_reputation_score_validation() {
        // Test task completion score validation
        let result = ReputationScore::calculate(150, 100.0, 5, 1);
        assert!(result.is_err());
        
        // Test anomaly score validation
        let result = ReputationScore::calculate(50, 100.0, 50, 1);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_trust_score_calculation() {
        // Node delivers 15 seconds faster than promised
        let score = TrustScore::calculate(45, 60);
        assert_eq!(score.score, 15); // 60 - 45 = 15
        assert_eq!(score.trust_level, TrustLevel::Good);
        assert_eq!(score.reliability_assessment(), 0.8);
        
        // Node delivers 30 seconds slower than promised
        let score = TrustScore::calculate(90, 60);
        assert_eq!(score.score, -30); // 60 - 90 = -30
        assert_eq!(score.trust_level, TrustLevel::Questionable);
    }
    
    #[test]
    fn test_trust_score_update() {
        let mut score = TrustScore::calculate(60, 60);
        assert_eq!(score.trust_level, TrustLevel::Reliable);
        
        // Update with better performance
        score.update(30, 60);
        assert_eq!(score.score, 30);
        assert_eq!(score.trust_level, TrustLevel::Good);
        assert_eq!(score.performance_history.historical_trust_scores.len(), 1);
    }
    
    #[test]
    fn test_performance_score_calculation() {
        // Task completed 2 minutes faster than expected
        let score = PerformanceScore::calculate(300, 180);
        assert_eq!(score.score, 120); // 300 - 180 = 120
        assert_eq!(score.performance_level, PerformanceLevel::AboveAverage);
        assert_eq!(score.performance_rating(), 0.8);
        
        // Task took longer than expected
        let score = PerformanceScore::calculate(60, 180);
        assert_eq!(score.score, -120); // 60 - 180 = -120
        assert_eq!(score.performance_level, PerformanceLevel::BelowAverage);
    }
    
    #[test]
    fn test_reputation_levels() {
        let excellent = ReputationScore::calculate(100, 2400.0, 30, 0).unwrap();
        assert_eq!(excellent.reputation_level(), ReputationLevel::Excellent);
        
        let poor = ReputationScore::calculate(20, 24.0, 0, 50).unwrap();
        assert_eq!(poor.reputation_level(), ReputationLevel::Poor);
        assert!(!poor.is_trustworthy());
    }
    
    #[test]
    fn test_trust_level_mapping() {
        assert_eq!(TrustScore::calculate(0, 70).trust_level, TrustLevel::Excellent);
        assert_eq!(TrustScore::calculate(50, 55).trust_level, TrustLevel::Reliable);
        assert_eq!(TrustScore::calculate(120, 60).trust_level, TrustLevel::Questionable);
        assert_eq!(TrustScore::calculate(400, 60).trust_level, TrustLevel::Unreliable);
    }
    
    #[test]
    fn test_performance_efficiency_metrics() {
        let score = PerformanceScore::calculate(120, 60);
        
        // Should be efficient (completed in half the time)
        assert!(score.efficiency_metrics.time_efficiency < 1.0);
        assert!(score.efficiency_metrics.resource_efficiency > 1.0);
        assert_eq!(score.performance_level, PerformanceLevel::AboveAverage);
    }
}