//! Comprehensive test suite for the biological security framework
//! 
//! Tests all layers, nodes, and integration scenarios to ensure
//! the biological security principles work correctly.

use std::sync::Arc;
use tokio::time::{timeout, Duration};

use bio_security::{
    SecurityFramework,
    config::{SecurityConfig, LayerConfig, CryptoConfig},
    errors::{SecurityEvent, SecuritySeverity, SecurityResult},
    layers::{SecurityContext, RiskLevel},
    nodes::{SecurityNode, DOSNode, InvestigationNode},
    crypto::CryptoContext,
};

#[tokio::test]
async fn test_complete_framework_integration() {
    // Initialize logging for tests
    let _ = tracing_subscriber::fmt::try_init();
    
    // Create test configuration
    let config = SecurityConfig::for_testing();
    
    // Create and initialize framework
    let mut framework = SecurityFramework::new(config).await
        .expect("Failed to create framework");
    
    framework.initialize().await
        .expect("Failed to initialize framework");
    
    framework.start().await
        .expect("Failed to start framework");
    
    // Test secure execution
    let test_data = b"biological computation test data";
    let result = timeout(
        Duration::from_secs(30),
        framework.execute_secure("integration_test", test_data)
    ).await;
    
    assert!(result.is_ok(), "Execution timed out");
    let execution_result = result.unwrap();
    assert!(execution_result.is_ok(), "Execution failed: {:?}", execution_result.err());
    
    let output = execution_result.unwrap();
    assert!(!output.is_empty(), "Output should not be empty");
    
    // Verify framework metrics
    let metrics = framework.metrics().await;
    assert!(metrics.executions_processed > 0, "Should have processed executions");
    assert_eq!(metrics.executions_successful, metrics.executions_processed, "All executions should be successful");
    
    // Test layer status
    let layer_status = framework.layer_status().await;
    assert_eq!(layer_status.len(), 5, "Should have 5 layers");
    
    for (layer_id, name, status) in layer_status {
        println!("Layer {}: {} - {:?}", layer_id, name, status);
        // All layers should be active or ready
        assert!(matches!(status, bio_security::layers::LayerStatus::Active | bio_security::layers::LayerStatus::Ready));
    }
    
    // Test node status
    let node_status = framework.node_status().await;
    assert!(!node_status.is_empty(), "Should have security nodes");
    
    for (node_id, status) in node_status {
        println!("Node {}: {:?}", node_id, status);
        // All nodes should be active or ready
        assert!(matches!(status, bio_security::nodes::SecurityNodeStatus::Active | bio_security::nodes::SecurityNodeStatus::Ready));
    }
    
    // Clean shutdown
    framework.stop().await
        .expect("Failed to stop framework");
    
    println!("✅ Complete framework integration test passed");
}

#[tokio::test]
async fn test_layer_by_layer_functionality() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = SecurityConfig::for_testing();
    let mut framework = SecurityFramework::new(config).await.unwrap();
    
    framework.initialize().await.unwrap();
    framework.start().await.unwrap();
    
    // Test multiple executions with different risk levels
    let test_scenarios = vec![
        ("low_risk", RiskLevel::Low, b"low risk data"),
        ("medium_risk", RiskLevel::Medium, b"medium risk computational workload"),
        ("high_risk", RiskLevel::High, b"high risk sensitive computation"),
        ("critical_risk", RiskLevel::Critical, b"critical risk maximum security needed"),
    ];
    
    for (test_name, risk_level, test_data) in test_scenarios {
        println!("Testing scenario: {} with risk level: {:?}", test_name, risk_level);
        
        let context = SecurityContext::new(test_name.to_string(), "test_node".to_string())
            .with_risk_level(risk_level);
        
        let result = framework.execute_secure_with_context(test_data, &context).await;
        assert!(result.is_ok(), "Failed to execute {}: {:?}", test_name, result.err());
        
        let output = result.unwrap();
        assert!(!output.is_empty(), "Output should not be empty for {}", test_name);
    }
    
    // Verify layer metrics show activity
    let layer_metrics = framework.layer_metrics().await;
    assert_eq!(layer_metrics.len(), 5, "Should have metrics for 5 layers");
    
    for metrics in layer_metrics {
        println!("Layer {}: {} operations, {} events", 
            metrics.layer_id, metrics.operations_processed, metrics.security_events);
        
        // Each layer should have processed operations
        assert!(metrics.operations_processed > 0, 
            "Layer {} should have processed operations", metrics.layer_id);
    }
    
    framework.stop().await.unwrap();
    println!("✅ Layer-by-layer functionality test passed");
}

#[tokio::test]
async fn test_security_event_processing() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = SecurityConfig::for_testing();
    let mut framework = SecurityFramework::new(config).await.unwrap();
    
    framework.initialize().await.unwrap();
    framework.start().await.unwrap();
    
    // Create a test event handler to capture events
    let event_handler = Arc::new(TestEventHandler::new());
    framework.add_event_handler(event_handler.clone()).await.unwrap();
    
    // Execute with a context that should trigger security events
    let high_risk_context = SecurityContext::new(
        "security_test".to_string(), 
        "suspicious_node".to_string()
    ).with_risk_level(RiskLevel::Critical);
    
    let test_data = b"potentially malicious payload";
    let result = framework.execute_secure_with_context(test_data, &high_risk_context).await;
    
    // Should still succeed but with security events
    assert!(result.is_ok(), "Execution should succeed despite high risk");
    
    // Give some time for event processing
    tokio::time::sleep(Duration::from_millis(100)).await;
    
    let events_handled = event_handler.get_event_count().await;
    println!("Events handled: {}", events_handled);
    
    // Should have generated some security events
    assert!(events_handled > 0, "Should have handled security events");
    
    framework.stop().await.unwrap();
    println!("✅ Security event processing test passed");
}

#[tokio::test]
async fn test_dos_node_functionality() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Test DOS Node independently
    let crypto_config = CryptoConfig::for_testing();
    let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());
    
    let mut dos_node = DOSNode::new("test_dos_node".to_string());
    dos_node.initialize(crypto).await.unwrap();
    dos_node.start().await.unwrap();
    
    // Create a DOS attack event
    let dos_event = SecurityEvent::new(
        SecuritySeverity::High,
        "dos_attack_detected",
        "Simulated DOS attack for testing"
    ).with_node("attacker_node".to_string());
    
    // Process the event
    let response_events = dos_node.process_event(&dos_event).await.unwrap();
    
    // Should generate response events
    assert!(!response_events.is_empty(), "DOS node should generate response events");
    
    for event in &response_events {
        println!("DOS Response Event: {} - {}", event.event_type, event.message);
        assert!(matches!(event.severity, SecuritySeverity::Medium | SecuritySeverity::High));
    }
    
    // Check node metrics
    let metrics = dos_node.metrics().await;
    assert!(metrics.events_processed > 0, "DOS node should have processed events");
    assert!(metrics.threats_detected > 0, "DOS node should have detected threats");
    
    dos_node.stop().await.unwrap();
    println!("✅ DOS node functionality test passed");
}

#[tokio::test]
async fn test_investigation_node_functionality() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Test Investigation Node independently
    let crypto_config = CryptoConfig::for_testing();
    let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());
    
    let mut investigation_node = InvestigationNode::new("test_investigation_node".to_string());
    investigation_node.initialize(crypto).await.unwrap();
    investigation_node.start().await.unwrap();
    
    // Create an event requiring investigation
    let suspicious_event = SecurityEvent::new(
        SecuritySeverity::High,
        "behavioral_anomaly_detected",
        "Suspicious node behavior detected"
    ).with_node("suspicious_node".to_string());
    
    // Process the event
    let response_events = investigation_node.process_event(&suspicious_event).await.unwrap();
    
    // Should generate investigation events
    assert!(!response_events.is_empty(), "Investigation node should generate response events");
    
    for event in &response_events {
        println!("Investigation Response Event: {} - {}", event.event_type, event.message);
    }
    
    // Check node metrics
    let metrics = investigation_node.metrics().await;
    assert!(metrics.events_processed > 0, "Investigation node should have processed events");
    
    investigation_node.stop().await.unwrap();
    println!("✅ Investigation node functionality test passed");
}

#[tokio::test]
async fn test_concurrent_executions() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = SecurityConfig::for_testing();
    let mut framework = SecurityFramework::new(config).await.unwrap();
    
    framework.initialize().await.unwrap();
    framework.start().await.unwrap();
    
    // Create multiple concurrent executions
    let mut handles = Vec::new();
    
    for i in 0..10 {
        let framework_ref = &framework;
        let execution_id = format!("concurrent_test_{}", i);
        let test_data = format!("concurrent test data {}", i);
        
        let handle = tokio::spawn(async move {
            let result = framework_ref.execute_secure(&execution_id, test_data.as_bytes()).await;
            (i, result)
        });
        
        handles.push(handle);
    }
    
    // Wait for all executions to complete
    let mut successful_count = 0;
    for handle in handles {
        let (i, result) = handle.await.unwrap();
        match result {
            Ok(output) => {
                successful_count += 1;
                assert!(!output.is_empty(), "Output {} should not be empty", i);
            },
            Err(e) => {
                println!("Execution {} failed: {:?}", i, e);
            }
        }
    }
    
    assert!(successful_count > 0, "At least some concurrent executions should succeed");
    println!("Successful concurrent executions: {}/10", successful_count);
    
    // Verify metrics
    let metrics = framework.metrics().await;
    assert!(metrics.executions_processed >= successful_count, 
        "Metrics should reflect processed executions");
    
    framework.stop().await.unwrap();
    println!("✅ Concurrent executions test passed");
}

#[tokio::test]
async fn test_framework_resilience() {
    let _ = tracing_subscriber::fmt::try_init();
    
    let config = SecurityConfig::for_testing();
    let mut framework = SecurityFramework::new(config).await.unwrap();
    
    framework.initialize().await.unwrap();
    framework.start().await.unwrap();
    
    // Test with various edge cases
    let edge_cases = vec![
        ("empty_data", b""),
        ("small_data", b"x"),
        ("large_data", &vec![b'A'; 10000]),
        ("binary_data", &[0u8, 255u8, 127u8, 1u8, 254u8]),
    ];
    
    for (test_name, test_data) in edge_cases {
        println!("Testing edge case: {}", test_name);
        
        let result = framework.execute_secure(
            &format!("edge_case_{}", test_name), 
            test_data
        ).await;
        
        // Framework should handle all edge cases gracefully
        assert!(result.is_ok(), "Failed to handle edge case {}: {:?}", 
            test_name, result.err());
        
        let output = result.unwrap();
        // Output can be empty for empty input, but shouldn't crash
        println!("Edge case {} - input: {} bytes, output: {} bytes", 
            test_name, test_data.len(), output.len());
    }
    
    framework.stop().await.unwrap();
    println!("✅ Framework resilience test passed");
}

#[tokio::test]
async fn test_biological_principles_effectiveness() {
    let _ = tracing_subscriber::fmt::try_init();
    
    // Test biological principles by comparing with and without certain features
    let mut config_with_bio = SecurityConfig::for_testing();
    let mut config_minimal = SecurityConfig::for_testing();
    
    // Disable biological features in minimal config
    for layer_config in &mut config_minimal.layers {
        match &mut layer_config.settings {
            bio_security::config::LayerSettings::IllusionLayer { deception_enabled, .. } => {
                *deception_enabled = false;
            },
            bio_security::config::LayerSettings::BehaviorMonitoring { ml_enabled, .. } => {
                *ml_enabled = false;
            },
            _ => {}
        }
    }
    
    // Test with biological features
    let mut framework_bio = SecurityFramework::new(config_with_bio).await.unwrap();
    framework_bio.initialize().await.unwrap();
    framework_bio.start().await.unwrap();
    
    // Test with minimal features
    let mut framework_minimal = SecurityFramework::new(config_minimal).await.unwrap();
    framework_minimal.initialize().await.unwrap();
    framework_minimal.start().await.unwrap();
    
    let test_data = b"biological effectiveness test";
    
    // Execute on both frameworks
    let result_bio = framework_bio.execute_secure("bio_test", test_data).await;
    let result_minimal = framework_minimal.execute_secure("minimal_test", test_data).await;
    
    assert!(result_bio.is_ok(), "Biological framework should succeed");
    assert!(result_minimal.is_ok(), "Minimal framework should succeed");
    
    // Compare metrics (biological should have more activity)
    let metrics_bio = framework_bio.metrics().await;
    let metrics_minimal = framework_minimal.metrics().await;
    
    println!("Biological framework - executions: {}, success rate: {:.2}", 
        metrics_bio.executions_processed, metrics_bio.success_rate());
    println!("Minimal framework - executions: {}, success rate: {:.2}", 
        metrics_minimal.executions_processed, metrics_minimal.success_rate());
    
    // Both should succeed but biological may have different characteristics
    assert!(metrics_bio.success_rate() > 0.0, "Biological framework should have successful executions");
    assert!(metrics_minimal.success_rate() > 0.0, "Minimal framework should have successful executions");
    
    framework_bio.stop().await.unwrap();
    framework_minimal.stop().await.unwrap();
    
    println!("✅ Biological principles effectiveness test passed");
}

/// Test event handler that captures events for verification
struct TestEventHandler {
    event_count: Arc<tokio::sync::RwLock<usize>>,
}

impl TestEventHandler {
    fn new() -> Self {
        Self {
            event_count: Arc::new(tokio::sync::RwLock::new(0)),
        }
    }
    
    async fn get_event_count(&self) -> usize {
        let count = self.event_count.read().await;
        *count
    }
}

#[async_trait::async_trait]
impl bio_security::framework::SecurityEventHandler for TestEventHandler {
    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        let mut count = self.event_count.write().await;
        *count += 1;
        
        println!("Test handler received event: {} - {} ({})", 
            event.event_type, event.message, event.severity);
        
        Ok(())
    }
}

/// Integration test helper functions
#[allow(dead_code)]
mod test_helpers {
    use super::*;
    
    pub async fn create_test_framework() -> SecurityResult<SecurityFramework> {
        let config = SecurityConfig::for_testing();
        let mut framework = SecurityFramework::new(config).await?;
        framework.initialize().await?;
        framework.start().await?;
        Ok(framework)
    }
    
    pub async fn shutdown_framework(mut framework: SecurityFramework) -> SecurityResult<()> {
        framework.stop().await
    }
    
    pub fn create_test_security_context(execution_id: &str, risk_level: RiskLevel) -> SecurityContext {
        SecurityContext::new(execution_id.to_string(), "test_node".to_string())
            .with_risk_level(risk_level)
    }
}

// Run all tests with: cargo test --package bio-security --test integration -- --nocapture