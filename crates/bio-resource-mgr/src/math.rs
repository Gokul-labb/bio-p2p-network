//! Mathematical Models and Formulas for Biological Resource Management
//! 
//! Implements biological mathematical models for resource allocation, thermal analysis,
//! and optimization algorithms based on natural systems.

use std::f64::consts::{E, PI};

/// Dynamic Resource Allocation Formula
/// 
/// ResourceAllocation(t) = BaseCapacity + AdaptiveFactor × DemandSignal(t) + ThermalFeedback
/// 
/// This formula implements biological resource allocation strategies that adapt
/// to changing conditions while maintaining system stability.
pub fn dynamic_resource_allocation(
    base_capacity: f64,
    adaptive_factor: f64,
    demand_signal: f64,
    thermal_feedback: f64,
) -> f64 {
    let demand_component = adaptive_factor * demand_signal;
    let thermal_adjustment = if thermal_feedback > 0.8 {
        -0.2 * (thermal_feedback - 0.8) // Reduce allocation when overheating
    } else {
        0.1 * (0.8 - thermal_feedback) // Increase allocation when cool
    };
    
    (base_capacity + demand_component + thermal_feedback + thermal_adjustment).max(0.0)
}

/// HAVOC Response Strength Calculation
/// 
/// HAVOCResponse = EmergencyThreshold × NetworkStress × AvailableResources / CriticalityFactor
/// 
/// Calculates the strength of HAVOC (crisis) response based on network conditions
/// and available resources, inspired by biological emergency response systems.
pub fn havoc_response_strength(
    emergency_threshold: f64,
    network_stress: f64,
    available_resources: f64,
    criticality_factor: f64,
) -> f64 {
    if criticality_factor <= 0.0 {
        return 0.0;
    }
    
    let base_response = emergency_threshold * network_stress;
    let resource_multiplier = if available_resources > 1.0 {
        1.0 + (available_resources - 1.0) * 0.5 // Diminishing returns
    } else {
        available_resources
    };
    
    (base_response * resource_multiplier) / criticality_factor
}

/// Thermal Signature Analysis
/// 
/// ThermalSignature = CPU_Usage × Memory_Pattern × Network_Bandwidth × Storage_Access
/// 
/// Calculates a composite thermal signature representing overall system stress
/// based on resource usage patterns, similar to biological thermal regulation.
pub fn thermal_signature(
    cpu_usage: f64,
    memory_pattern: f64,
    network_bandwidth: f64,
    storage_access: f64,
) -> f64 {
    // Weight different resources based on their thermal impact
    let cpu_weight = 0.4;
    let memory_weight = 0.3;
    let network_weight = 0.2;
    let storage_weight = 0.1;
    
    let weighted_sum = cpu_usage * cpu_weight + 
                      memory_pattern * memory_weight + 
                      network_bandwidth * network_weight + 
                      storage_access * storage_weight;
    
    // Apply non-linear scaling to emphasize high usage
    weighted_sum.powf(1.2)
}

/// Biological Scaling Factor
/// 
/// Implements transformer-like scaling characteristics inspired by biological
/// energy regulation systems. Returns optimal scaling factor based on demand.
pub fn scaling_factor(demand_ratio: f64, capacity_factor: f64) -> f64 {
    if demand_ratio <= 0.0 || capacity_factor <= 0.0 {
        return 1.0;
    }
    
    // Biological scaling follows a sigmoid curve for stability
    let scaling_input = (demand_ratio / capacity_factor - 1.0) * 3.0; // Scale the input
    let sigmoid_response = 1.0 / (1.0 + (-scaling_input).exp());
    
    // Map sigmoid output to scaling factor range (0.1 to 5.0)
    0.1 + sigmoid_response * 4.9
}

/// Trust Score with Decay
/// 
/// Models biological trust relationships with natural decay over time
/// and positive reinforcement through interactions.
pub fn trust_score_with_decay(
    base_trust: f64,
    decay_factor: f64,
    time_elapsed_days: f64,
    interaction_bonus: f64,
) -> f64 {
    // Apply exponential decay
    let decayed_trust = base_trust * decay_factor.powf(time_elapsed_days);
    
    // Add interaction bonus with diminishing returns
    let bonus_factor = 1.0 - (-interaction_bonus).exp();
    let final_trust = decayed_trust + bonus_factor * (1.0 - decayed_trust);
    
    final_trust.clamp(0.0, 1.0)
}

/// Resource Efficiency Optimization
/// 
/// Calculates optimal resource distribution using biological efficiency principles
/// inspired by metabolic rate optimization in living systems.
pub fn resource_efficiency_optimization(
    current_allocation: &[f64],
    demand_patterns: &[f64],
    efficiency_weights: &[f64],
) -> Vec<f64> {
    if current_allocation.len() != demand_patterns.len() || 
       current_allocation.len() != efficiency_weights.len() {
        return current_allocation.to_vec();
    }
    
    let total_resources: f64 = current_allocation.iter().sum();
    if total_resources <= 0.0 {
        return current_allocation.to_vec();
    }
    
    // Calculate weighted demand scores
    let weighted_demands: Vec<f64> = demand_patterns.iter()
        .zip(efficiency_weights.iter())
        .map(|(demand, weight)| demand * weight)
        .collect();
    
    let total_weighted_demand: f64 = weighted_demands.iter().sum();
    
    if total_weighted_demand <= 0.0 {
        return current_allocation.to_vec();
    }
    
    // Distribute resources proportionally to weighted demand
    weighted_demands.iter()
        .map(|weighted_demand| (weighted_demand / total_weighted_demand) * total_resources)
        .collect()
}

/// Compartment Load Balancing
/// 
/// Implements ant colony optimization principles for load balancing
/// across compartments with different specializations.
pub fn compartment_load_balancing(
    compartment_capacities: &[f64],
    current_loads: &[f64],
    specialization_factors: &[f64],
) -> Vec<f64> {
    if compartment_capacities.len() != current_loads.len() || 
       compartment_capacities.len() != specialization_factors.len() {
        return current_loads.to_vec();
    }
    
    let mut balanced_loads = current_loads.to_vec();
    let total_load: f64 = current_loads.iter().sum();
    
    if total_load <= 0.0 {
        return balanced_loads;
    }
    
    // Calculate utilization rates
    let utilization_rates: Vec<f64> = current_loads.iter()
        .zip(compartment_capacities.iter())
        .map(|(load, capacity)| if *capacity > 0.0 { load / capacity } else { 0.0 })
        .collect();
    
    // Find over-utilized and under-utilized compartments
    let avg_utilization: f64 = utilization_rates.iter().sum::<f64>() / utilization_rates.len() as f64;
    
    for i in 0..balanced_loads.len() {
        let current_utilization = utilization_rates[i];
        let specialization = specialization_factors[i];
        
        if current_utilization > avg_utilization * 1.2 {
            // Over-utilized - reduce load based on specialization
            let reduction_factor = 0.9 + 0.1 * specialization;
            balanced_loads[i] *= reduction_factor;
        } else if current_utilization < avg_utilization * 0.8 {
            // Under-utilized - increase load based on available capacity
            let capacity_factor = compartment_capacities[i] / compartment_capacities.iter().sum::<f64>();
            let increase_factor = 1.1 * capacity_factor * specialization;
            balanced_loads[i] = (balanced_loads[i] * increase_factor).min(compartment_capacities[i]);
        }
    }
    
    balanced_loads
}

/// Network Stress Calculation
/// 
/// Calculates overall network stress level using biological stress response models
/// that consider multiple stress factors and their interactions.
pub fn network_stress_calculation(
    cpu_stress: f64,
    memory_stress: f64,
    bandwidth_stress: f64,
    node_failure_rate: f64,
    temporal_factor: f64,
) -> f64 {
    // Weight different stress components
    let cpu_weight = 0.3;
    let memory_weight = 0.25;
    let bandwidth_weight = 0.2;
    let failure_weight = 0.15;
    let temporal_weight = 0.1;
    
    let base_stress = cpu_stress * cpu_weight + 
                     memory_stress * memory_weight + 
                     bandwidth_stress * bandwidth_weight + 
                     node_failure_rate * failure_weight + 
                     temporal_factor * temporal_weight;
    
    // Apply stress amplification for high stress conditions (biological stress response)
    if base_stress > 0.7 {
        let amplification = 1.0 + (base_stress - 0.7) * 2.0;
        (base_stress * amplification).min(1.0)
    } else {
        base_stress
    }
}

/// Swarm Intelligence Routing Optimization
/// 
/// Implements ant colony optimization for network routing based on
/// pheromone trail simulation and path optimization.
pub fn swarm_routing_optimization(
    path_lengths: &[f64],
    pheromone_trails: &[f64],
    congestion_levels: &[f64],
    alpha: f64,  // Pheromone importance
    beta: f64,   // Path length importance
    gamma: f64,  // Congestion avoidance importance
) -> Vec<f64> {
    if path_lengths.len() != pheromone_trails.len() || 
       path_lengths.len() != congestion_levels.len() {
        return vec![0.0; path_lengths.len()];
    }
    
    let mut path_probabilities = Vec::with_capacity(path_lengths.len());
    
    // Calculate attractiveness for each path
    for i in 0..path_lengths.len() {
        let pheromone_factor = pheromone_trails[i].powf(alpha);
        let distance_factor = if path_lengths[i] > 0.0 {
            (1.0 / path_lengths[i]).powf(beta)
        } else {
            0.0
        };
        let congestion_factor = (1.0 - congestion_levels[i]).powf(gamma);
        
        let attractiveness = pheromone_factor * distance_factor * congestion_factor;
        path_probabilities.push(attractiveness);
    }
    
    // Normalize probabilities
    let total_attractiveness: f64 = path_probabilities.iter().sum();
    if total_attractiveness > 0.0 {
        for prob in &mut path_probabilities {
            *prob /= total_attractiveness;
        }
    }
    
    path_probabilities
}

/// Energy Efficiency Calculation
/// 
/// Calculates energy efficiency using biological metabolic models
/// that optimize energy usage while maintaining performance.
pub fn energy_efficiency_calculation(
    computational_work: f64,
    energy_consumed: f64,
    thermal_overhead: f64,
    idle_energy_ratio: f64,
) -> f64 {
    if energy_consumed <= 0.0 {
        return 0.0;
    }
    
    // Base efficiency: work done per unit energy
    let base_efficiency = computational_work / energy_consumed;
    
    // Thermal penalty (energy lost to heat)
    let thermal_penalty = thermal_overhead * 0.1;
    
    // Idle energy penalty
    let idle_penalty = idle_energy_ratio * 0.05;
    
    // Biological efficiency follows a curve with diminishing returns
    let efficiency_factor = 1.0 - (-base_efficiency).exp();
    
    (efficiency_factor - thermal_penalty - idle_penalty).max(0.0).min(1.0)
}

/// Adaptive Threshold Calculation
/// 
/// Calculates adaptive thresholds for resource management based on
/// biological homeostasis and adaptation mechanisms.
pub fn adaptive_threshold_calculation(
    current_threshold: f64,
    system_performance: f64,
    target_performance: f64,
    adaptation_rate: f64,
    stability_factor: f64,
) -> f64 {
    let performance_error = target_performance - system_performance;
    
    // Biological adaptation follows proportional-integral-like control
    let proportional_adjustment = adaptation_rate * performance_error;
    
    // Stability constraint prevents oscillation
    let stability_constraint = stability_factor * current_threshold.abs();
    let max_adjustment = stability_constraint * 0.1; // Max 10% change per adjustment
    
    let adjustment = proportional_adjustment.clamp(-max_adjustment, max_adjustment);
    
    (current_threshold + adjustment).clamp(0.0, 1.0)
}

/// Biological Consensus Strength
/// 
/// Calculates consensus strength using biological group decision models
/// inspired by swarm intelligence and collective behavior.
pub fn biological_consensus_strength(
    agreement_ratio: f64,
    confidence_levels: &[f64],
    node_reliability: &[f64],
    quorum_threshold: f64,
) -> f64 {
    if confidence_levels.len() != node_reliability.len() {
        return 0.0;
    }
    
    // Check if quorum is met
    if agreement_ratio < quorum_threshold {
        return 0.0;
    }
    
    // Weight consensus by node reliability and confidence
    let weighted_consensus: f64 = confidence_levels.iter()
        .zip(node_reliability.iter())
        .map(|(confidence, reliability)| confidence * reliability)
        .sum();
    
    let total_weight: f64 = node_reliability.iter().sum();
    
    if total_weight > 0.0 {
        let base_strength = weighted_consensus / total_weight;
        
        // Apply biological consensus amplification (strong agreement reinforcement)
        if base_strength > 0.8 {
            base_strength + (base_strength - 0.8) * 2.0
        } else {
            base_strength
        }.min(1.0)
    } else {
        0.0
    }
}

/// Predator-Prey Resource Balance
/// 
/// Models resource allocation using predator-prey dynamics for
/// automatic load balancing and resource distribution.
pub fn predator_prey_resource_balance(
    resource_supply: f64,
    resource_demand: f64,
    supply_growth_rate: f64,
    demand_growth_rate: f64,
    interaction_strength: f64,
    time_delta: f64,
) -> (f64, f64) {
    // Lotka-Volterra inspired equations adapted for resource management
    let supply_change = supply_growth_rate * resource_supply - 
                       interaction_strength * resource_supply * resource_demand;
    
    let demand_change = interaction_strength * resource_supply * resource_demand - 
                       demand_growth_rate * resource_demand;
    
    let new_supply = (resource_supply + supply_change * time_delta).max(0.0);
    let new_demand = (resource_demand + demand_change * time_delta).max(0.0);
    
    (new_supply, new_demand)
}

/// Fractal Resource Distribution
/// 
/// Implements fractal distribution patterns found in biological systems
/// for hierarchical resource allocation across network levels.
pub fn fractal_resource_distribution(
    total_resources: f64,
    hierarchy_levels: usize,
    fractal_dimension: f64,
    level_weights: &[f64],
) -> Vec<f64> {
    if hierarchy_levels == 0 || level_weights.len() != hierarchy_levels {
        return vec![total_resources];
    }
    
    let mut distribution = Vec::with_capacity(hierarchy_levels);
    let mut remaining_resources = total_resources;
    
    // Calculate fractal scaling factors
    let total_weight: f64 = level_weights.iter()
        .enumerate()
        .map(|(i, weight)| weight * (i as f64 + 1.0).powf(-fractal_dimension))
        .sum();
    
    for (level, weight) in level_weights.iter().enumerate() {
        let fractal_factor = (level as f64 + 1.0).powf(-fractal_dimension);
        let level_allocation = if total_weight > 0.0 {
            total_resources * (weight * fractal_factor) / total_weight
        } else {
            0.0
        };
        
        let actual_allocation = level_allocation.min(remaining_resources);
        distribution.push(actual_allocation);
        remaining_resources -= actual_allocation;
    }
    
    distribution
}

/// Biological Clock Synchronization
/// 
/// Implements circadian rhythm-inspired synchronization for
/// periodic resource allocation and network coordination.
pub fn biological_clock_synchronization(
    current_time: f64,
    period: f64,
    phase_offset: f64,
    amplitude: f64,
    baseline: f64,
) -> f64 {
    if period <= 0.0 {
        return baseline;
    }
    
    let normalized_time = (current_time + phase_offset) / period;
    let phase = (normalized_time * 2.0 * PI) % (2.0 * PI);
    
    baseline + amplitude * phase.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_resource_allocation() {
        let allocation = dynamic_resource_allocation(1.0, 0.5, 0.8, 0.2);
        assert!(allocation > 1.0);
        
        // Test with high thermal feedback
        let hot_allocation = dynamic_resource_allocation(1.0, 0.5, 0.8, 0.9);
        assert!(hot_allocation < allocation);
    }
    
    #[test]
    fn test_havoc_response_strength() {
        let response = havoc_response_strength(0.8, 0.9, 2.0, 1.5);
        assert!(response > 0.0);
        
        // Zero criticality should return zero
        let zero_response = havoc_response_strength(0.8, 0.9, 2.0, 0.0);
        assert_eq!(zero_response, 0.0);
    }
    
    #[test]
    fn test_thermal_signature() {
        let signature = thermal_signature(0.8, 0.7, 0.6, 0.5);
        assert!(signature > 0.0);
        assert!(signature <= 1.0);
        
        // Higher values should produce higher signatures
        let high_signature = thermal_signature(0.9, 0.9, 0.8, 0.7);
        assert!(high_signature > signature);
    }
    
    #[test]
    fn test_scaling_factor() {
        let factor = scaling_factor(0.8, 1.0);
        assert!(factor >= 0.1 && factor <= 5.0);
        
        // High demand should increase scaling factor
        let high_demand_factor = scaling_factor(1.5, 1.0);
        assert!(high_demand_factor > factor);
    }
    
    #[test]
    fn test_trust_score_with_decay() {
        let trust = trust_score_with_decay(1.0, 0.95, 1.0, 0.1);
        assert!(trust >= 0.0 && trust <= 1.0);
        assert!(trust < 1.0); // Should decay over time
        
        // With interaction bonus, should be higher
        let bonus_trust = trust_score_with_decay(1.0, 0.95, 1.0, 0.5);
        assert!(bonus_trust > trust);
    }
    
    #[test]
    fn test_resource_efficiency_optimization() {
        let current_allocation = vec![1.0, 2.0, 1.5];
        let demand_patterns = vec![0.8, 1.2, 0.6];
        let efficiency_weights = vec![1.0, 1.0, 1.0];
        
        let optimized = resource_efficiency_optimization(&current_allocation, &demand_patterns, &efficiency_weights);
        assert_eq!(optimized.len(), 3);
        
        // Total resources should be conserved
        let total_before: f64 = current_allocation.iter().sum();
        let total_after: f64 = optimized.iter().sum();
        assert!((total_before - total_after).abs() < 1e-10);
    }
    
    #[test]
    fn test_compartment_load_balancing() {
        let capacities = vec![2.0, 3.0, 1.5];
        let loads = vec![1.8, 1.0, 1.4]; // First is over-utilized, second under-utilized
        let specialization = vec![1.0, 1.0, 1.0];
        
        let balanced = compartment_load_balancing(&capacities, &loads, &specialization);
        assert_eq!(balanced.len(), 3);
        
        // Over-utilized compartment should have reduced load
        assert!(balanced[0] < loads[0]);
    }
    
    #[test]
    fn test_network_stress_calculation() {
        let stress = network_stress_calculation(0.6, 0.7, 0.5, 0.1, 0.3);
        assert!(stress >= 0.0 && stress <= 1.0);
        
        // High stress should trigger amplification
        let high_stress = network_stress_calculation(0.9, 0.8, 0.8, 0.3, 0.7);
        assert!(high_stress > stress);
    }
    
    #[test]
    fn test_swarm_routing_optimization() {
        let path_lengths = vec![1.0, 2.0, 1.5];
        let pheromone_trails = vec![0.8, 0.4, 0.6];
        let congestion_levels = vec![0.2, 0.7, 0.3];
        
        let probabilities = swarm_routing_optimization(&path_lengths, &pheromone_trails, &congestion_levels, 1.0, 2.0, 1.0);
        assert_eq!(probabilities.len(), 3);
        
        // Probabilities should sum to approximately 1.0
        let sum: f64 = probabilities.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        
        // Shortest path with high pheromone should have highest probability
        assert!(probabilities[0] > probabilities[1]);
    }
    
    #[test]
    fn test_energy_efficiency_calculation() {
        let efficiency = energy_efficiency_calculation(10.0, 12.0, 0.1, 0.2);
        assert!(efficiency >= 0.0 && efficiency <= 1.0);
        
        // More work per energy should increase efficiency
        let better_efficiency = energy_efficiency_calculation(15.0, 12.0, 0.1, 0.2);
        assert!(better_efficiency > efficiency);
    }
    
    #[test]
    fn test_adaptive_threshold_calculation() {
        let new_threshold = adaptive_threshold_calculation(0.5, 0.6, 0.8, 0.1, 0.9);
        assert!(new_threshold >= 0.0 && new_threshold <= 1.0);
        
        // Should adjust towards target performance
        assert!(new_threshold > 0.5); // Should increase since performance < target
    }
    
    #[test]
    fn test_biological_consensus_strength() {
        let confidence = vec![0.9, 0.8, 0.85];
        let reliability = vec![0.95, 0.9, 0.88];
        
        let strength = biological_consensus_strength(0.9, &confidence, &reliability, 0.7);
        assert!(strength >= 0.0 && strength <= 1.0);
        
        // Below quorum should return 0
        let no_quorum = biological_consensus_strength(0.6, &confidence, &reliability, 0.7);
        assert_eq!(no_quorum, 0.0);
    }
    
    #[test]
    fn test_predator_prey_resource_balance() {
        let (new_supply, new_demand) = predator_prey_resource_balance(10.0, 5.0, 0.1, 0.05, 0.01, 1.0);
        
        assert!(new_supply > 0.0);
        assert!(new_demand > 0.0);
        
        // Supply should generally increase when demand is low
        assert!(new_supply != 10.0); // Should change
    }
    
    #[test]
    fn test_fractal_resource_distribution() {
        let weights = vec![1.0, 0.8, 0.6, 0.4];
        let distribution = fractal_resource_distribution(100.0, 4, 1.5, &weights);
        
        assert_eq!(distribution.len(), 4);
        
        // Total should equal input (approximately)
        let total: f64 = distribution.iter().sum();
        assert!((total - 100.0).abs() < 1e-10);
        
        // Higher levels should generally get less (fractal scaling)
        assert!(distribution[0] >= distribution[1]);
    }
    
    #[test]
    fn test_biological_clock_synchronization() {
        let sync_value = biological_clock_synchronization(0.0, 24.0, 0.0, 0.5, 1.0);
        assert!((sync_value - 1.0).abs() < 1e-10); // Should be at baseline at phase 0
        
        let quarter_cycle = biological_clock_synchronization(6.0, 24.0, 0.0, 0.5, 1.0);
        assert!(quarter_cycle > 1.0); // Should be above baseline at quarter cycle
    }
}