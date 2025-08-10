pub mod config;
pub mod error;
pub mod behavior;
pub mod protocols;
pub mod network;
pub mod node;

// Re-export main types
pub use config::NodeConfig;
pub use error::{P2PError, Result};
pub use node::{Node, NodeBuilder};
pub use behavior::{BiologicalBehavior, BiologicalBehaviorFactory};
pub use protocols::{BiologicalProtocolMessage, BiologicalProtocolHandler};
pub use network::{NodeBehaviour, TrustManager, PeerInfo};

// Re-export core protocol types for convenience
pub use core_protocol::{BiologicalRole, NodeMessage, NetworkAddress, TrustScore, ReputationScore, ThermalSignature};

/// Current version of the p2p-node crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default service name for mDNS discovery
pub const DEFAULT_SERVICE_NAME: &str = "bio-p2p";

/// Default protocol version
pub const PROTOCOL_VERSION: &str = "/bio-p2p/1.0.0";

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_SERVICE_NAME, "bio-p2p");
        assert_eq!(PROTOCOL_VERSION, "/bio-p2p/1.0.0");
    }
}