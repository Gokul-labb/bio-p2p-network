//! Basic Resource Manager Usage Example
//! 
//! Demonstrates basic setup and usage of the biological resource management system.

use std::time::Duration;
use tokio::time::sleep;

use bio_resource_mgr::{
    BiologicalResourceManager, ManagerConfig, CrisisType, TrustLevel,
    ResourceError, ResourceResult,
};

#[tokio::main]
async fn main() -> ResourceResult<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    println!("🧬 Biological Resource Manager - Basic Usage Example");
    println!("====================================================\n");
    
    // Create manager configuration
    let config = ManagerConfig {
        enable_havoc: true,
        enable_social: true,
        enable_thermal: true,
        enable_auto_scaling: true,
        metrics_interval: Duration::from_secs(30),
        max_nodes_per_type: 5,
    };
    
    // Create and initialize the resource manager
    let mut manager = BiologicalResourceManager::new(config);
    println!("📊 Initializing biological resource manager...");
    
    manager.initialize("bio-manager-001".to_string()).await?;
    println!("✅ Manager initialized successfully\n");
    
    // Add various node types
    println!("🔧 Adding biological nodes...");
    
    // Add step-up nodes for performance scaling
    manager.add_step_up_node("step-up-cpu".to_string()).await?;
    manager.add_step_up_node("step-up-memory".to_string()).await?;
    println!("⬆️  Added 2 step-up nodes for performance scaling");
    
    // Add step-down nodes for energy conservation
    manager.add_step_down_node("step-down-cpu".to_string()).await?;
    manager.add_step_down_node("step-down-memory".to_string()).await?;
    println!("⬇️  Added 2 step-down nodes for energy conservation");
    
    // Add friendship nodes for social collaboration
    manager.add_friendship_node("friend-alpha".to_string(), "192.168.1.100".to_string()).await?;
    manager.add_friendship_node("friend-beta".to_string(), "192.168.1.101".to_string()).await?;
    println!("🤝 Added 2 friendship nodes for social collaboration");
    
    // Add buddy nodes for permanent partnerships
    manager.add_buddy_node("buddy-primary".to_string()).await?;
    manager.add_buddy_node("buddy-secondary".to_string()).await?;
    println!("👥 Added 2 buddy nodes for permanent partnerships");
    
    // Add thermal management nodes
    manager.add_thermal_node("thermal-cpu".to_string())?;
    manager.add_thermal_node("thermal-gpu".to_string())?;
    println!("🌡️  Added 2 thermal management nodes\n");
    
    // Simulate some resource usage updates
    println!("📈 Simulating resource usage...");
    
    let thermal_readings = vec![
        ("thermal-cpu".to_string(), 0.75, 0.60, 0.40, 0.30), // CPU, Memory, Network, Storage
        ("thermal-gpu".to_string(), 0.85, 0.70, 0.20, 0.15),
    ];
    
    manager.update_thermal_readings(&thermal_readings)?;
    println!("✅ Updated thermal readings for all nodes");
    
    // Get initial system metrics
    let initial_metrics = manager.get_system_metrics();
    println!("📊 Initial system health: {:.2}%", initial_metrics.system_health_score() * 100.0);
    
    // Get system status summary
    let status = manager.get_status_summary();
    println!("📋 System Status:");
    println!("   - Overall Health: {:.2}%", status.overall_health * 100.0);
    println!("   - Active Compartments: {}", status.active_compartments);
    println!("   - Resource Efficiency: {:.2}%", status.resource_efficiency * 100.0);
    println!("   - Thermal Status: {}", status.thermal_status);
    println!("   - Uptime: {:.2} hours\n", status.uptime_hours);
    
    // Demonstrate crisis detection and emergency response
    println!("🚨 Testing crisis management...");
    
    let crisis_id = manager.trigger_emergency_reallocation(
        CrisisType::ResourceShortage,
        vec!["node-001".to_string(), "node-002".to_string()],
        0.6, // 60% resource reallocation
    ).await?;
    
    println!("⚡ Emergency reallocation triggered (ID: {})", crisis_id);
    
    // Wait a bit for the system to process
    sleep(Duration::from_secs(2)).await;
    
    // Check network stress level
    let network_stress = manager.get_network_stress();
    println!("📊 Current network stress: {:.2}%", network_stress * 100.0);
    
    // Demonstrate scaling operations
    println!("\n🔧 Testing adaptive scaling...");
    
    // Scale up for high demand
    let scale_up_results = manager.scale_up_all(1.5).await?;
    let successful_scale_ups = scale_up_results.iter().filter(|r| r.is_ok()).count();
    println!("⬆️  Scaled up {} nodes successfully", successful_scale_ups);
    
    // Wait a moment
    sleep(Duration::from_seconds(1)).await;
    
    // Scale down for energy conservation
    let scale_down_results = manager.scale_down_all(0.7).await?;
    let successful_scale_downs = scale_down_results.iter().filter(|r| r.is_ok()).count();
    println!("⬇️  Scaled down {} nodes successfully", successful_scale_downs);
    
    // Get comprehensive health report
    println!("\n📊 Generating comprehensive health report...");
    let health_report = manager.get_health_report();
    
    println!("🏥 Health Report:");
    println!("   - Overall Health: {:.2}%", health_report.overall_health * 100.0);
    println!("   - Component Health:");
    println!("     • HAVOC (Crisis): {:.2}%", health_report.component_health.havoc * 100.0);
    println!("     • Scaling: {:.2}%", health_report.component_health.scaling * 100.0);
    println!("     • Social: {:.2}%", health_report.component_health.social * 100.0);
    println!("     • Thermal: {:.2}%", health_report.component_health.thermal * 100.0);
    
    println!("   - Active Nodes:");
    println!("     • HAVOC: {}", health_report.active_nodes.havoc_nodes);
    println!("     • Step-up: {}", health_report.active_nodes.step_up_nodes);
    println!("     • Step-down: {}", health_report.active_nodes.step_down_nodes);
    println!("     • Friendship: {}", health_report.active_nodes.friendship_nodes);
    println!("     • Buddy: {}", health_report.active_nodes.buddy_nodes);
    println!("     • Thermal: {}", health_report.active_nodes.thermal_nodes);
    println!("     • Total: {}", health_report.active_nodes.total());
    
    println!("   - Recommendations:");
    for (i, recommendation) in health_report.recommendations.iter().enumerate() {
        println!("     {}. {}", i + 1, recommendation);
    }
    
    // Simulate system operation for a short period
    println!("\n⏳ Simulating system operation for 10 seconds...");
    
    for i in 1..=10 {
        sleep(Duration::from_secs(1)).await;
        
        // Update thermal readings with some variation
        let dynamic_readings = vec![
            ("thermal-cpu".to_string(), 
             0.75 + (i as f64 * 0.02), 
             0.60 + (i as f64 * 0.01), 
             0.40, 0.30),
            ("thermal-gpu".to_string(), 
             0.85 - (i as f64 * 0.01), 
             0.70, 0.20, 0.15),
        ];
        
        manager.update_thermal_readings(&dynamic_readings)?;
        
        if i % 3 == 0 {
            let current_stress = manager.get_network_stress();
            println!("   Step {}: Network stress at {:.1}%", i, current_stress * 100.0);
        }
    }
    
    // Final metrics
    let final_metrics = manager.get_system_metrics();
    println!("\n📊 Final Results:");
    println!("   - System Health: {:.2}% (was {:.2}%)", 
        final_metrics.system_health_score() * 100.0,
        initial_metrics.system_health_score() * 100.0);
    println!("   - Total Uptime: {:.2} minutes", final_metrics.uptime.as_secs() as f64 / 60.0);
    println!("   - Resource Efficiency: {:.2}%", final_metrics.resources.allocation_efficiency * 100.0);
    
    // Demonstrate graceful shutdown
    println!("\n🔌 Shutting down biological resource manager...");
    manager.shutdown().await?;
    println!("✅ All nodes shutdown gracefully");
    
    println!("\n🎉 Basic usage example completed successfully!");
    println!("   The biological resource management system demonstrated:");
    println!("   ✓ Multi-node biological architecture");
    println!("   ✓ Crisis detection and emergency response");
    println!("   ✓ Adaptive scaling (step-up/step-down)");
    println!("   ✓ Social collaboration features");
    println!("   ✓ Thermal management");
    println!("   ✓ Comprehensive health monitoring");
    println!("   ✓ Graceful shutdown procedures");
    
    Ok(())
}

/// Helper function to demonstrate error handling
fn _demonstrate_error_handling() -> ResourceResult<()> {
    // This would normally come from actual operations
    let error = ResourceError::insufficient_resources("CPU", 4.0, 2.0);
    
    println!("Error category: {}", error.category());
    println!("Error severity: {}", error.severity());
    println!("Is recoverable: {}", error.is_recoverable());
    println!("Recovery suggestions: {:?}", error.recovery_suggestions());
    
    Err(error)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_basic_usage_integration() {
        let config = ManagerConfig::default();
        let mut manager = BiologicalResourceManager::new(config);
        
        // Initialize
        manager.initialize("test-manager".to_string()).await.unwrap();
        
        // Add nodes
        manager.add_step_up_node("test-step-up".to_string()).await.unwrap();
        manager.add_friendship_node("test-friend".to_string(), "127.0.0.1".to_string()).await.unwrap();
        manager.add_thermal_node("test-thermal".to_string()).unwrap();
        
        // Get metrics
        let metrics = manager.get_system_metrics();
        assert!(metrics.system_health_score() > 0.0);
        
        // Get health report
        let report = manager.get_health_report();
        assert!(report.active_nodes.total() > 0);
        
        // Shutdown
        manager.shutdown().await.unwrap();
    }
}