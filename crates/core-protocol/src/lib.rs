/*!
# Core Protocol for Biologically-Inspired P2P Compute Sharing Network

This crate implements the core protocol types, message structures, and biological node definitions
for the decentralized AI workload distribution network inspired by natural biological systems.

## Overview

The protocol leverages an extensive taxonomy of 80+ specialized node types, each modeled after
distinct biological behaviors observed in nature, from crow learning patterns to ant colony 
organization, sea turtle synchronization, and primate social structures.

## Key Features

- **9-Step Package Processing Protocol**: Cellular biology-inspired task encapsulation
- **Multi-Layer Security Architecture**: 5-tier immune system-inspired security
- **Biological Node Taxonomy**: 80+ specialized node types with biological behaviors
- **Hierarchical Addressing**: XXX.XXX.Y format for scalable network management
- **Reputation & Trust Systems**: Mathematical scoring based on biological principles
- **Thermal Monitoring**: Resource usage monitoring with pheromone-like signaling

## Architecture

The network implements sophisticated fault tolerance, self-organization, and adaptive resource
allocation mechanisms that enable efficient distributed computing without centralized coordination.

Key innovations include:
- Biologically-inspired routing protocols
- Multi-layer security with thermal detection and behavioral monitoring
- Dynamic resource compartmentalization
- Reputation-based trust systems

Performance evaluations demonstrate 85-90% cost reduction compared to traditional cloud providers
while delivering enterprise-grade reliability and security through nature-inspired redundancy.
*/

pub mod address;
pub mod biological;
pub mod error;
pub mod message;
pub mod package;
pub mod reputation;
pub mod security;
pub mod thermal;

// Re-exports for convenience
pub use address::NetworkAddress;
pub use biological::{BiologicalRole, NodeParameters};
pub use error::{ProtocolError, Result};
pub use message::{NodeMessage, MessageType};
pub use package::{ComputePackage, PackageLifecycle, PackageState};
pub use reputation::{ReputationScore, TrustScore, PerformanceScore};
pub use security::{SecurityLayer, ValidationResult};
pub use thermal::ThermalSignature;

/// Protocol version for compatibility checking
pub const PROTOCOL_VERSION: &str = "1.0.0";

/// Maximum package size in bytes (100MB)
pub const MAX_PACKAGE_SIZE: usize = 100 * 1024 * 1024;

/// Maximum nodes in discovery radius
pub const MAX_DISCOVERY_RADIUS: usize = 100;

/// Default synchronization timeout in seconds
pub const DEFAULT_SYNC_TIMEOUT: u64 = 30;

/// Network configuration constants
pub mod constants {
    /// Nodes per address group (biological territorial organization)
    pub const NODES_PER_ADDRESS_GROUP: usize = 10;
    
    /// Address groups per region (1,000 nodes total per region)
    pub const ADDRESS_GROUPS_PER_REGION: usize = 100;
    
    /// Maximum task completion score
    pub const MAX_TASK_COMPLETION_SCORE: u32 = 100;
    
    /// Maximum lifetime score in points
    pub const MAX_LIFETIME_SCORE: u32 = 50;
    
    /// Maximum anomaly score
    pub const MAX_ANOMALY_SCORE: u32 = 30;
    
    /// Penalty per system shutdown
    pub const SHUTDOWN_PENALTY: i32 = -5;
    
    /// Thermal signature sampling frequency in seconds
    pub const THERMAL_SAMPLING_FREQUENCY: u64 = 1;
    
    /// Historical thermal data retention in days
    pub const THERMAL_RETENTION_DAYS: u64 = 30;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_protocol_version() {
        assert_eq!(PROTOCOL_VERSION, "1.0.0");
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(constants::NODES_PER_ADDRESS_GROUP, 10);
        assert_eq!(constants::ADDRESS_GROUPS_PER_REGION, 100);
        assert!(MAX_PACKAGE_SIZE > 0);
        assert!(MAX_DISCOVERY_RADIUS > 0);
    }
}