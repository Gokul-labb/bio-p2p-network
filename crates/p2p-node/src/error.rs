use thiserror::Error;
use std::time::Duration;

pub type Result<T> = std::result::Result<T, P2PError>;

#[derive(Error, Debug)]
pub enum P2PError {
    #[error("Network initialization failed: {reason}")]
    NetworkInitializationFailed { reason: String },
    
    #[error("Connection failed to peer {peer_id}: {reason}")]
    ConnectionFailed { peer_id: String, reason: String },
    
    #[error("Protocol negotiation failed: {protocol}")]
    ProtocolNegotiationFailed { protocol: String },
    
    #[error("Message serialization failed: {reason}")]
    SerializationFailed { reason: String },
    
    #[error("Message deserialization failed: {reason}")]
    DeserializationFailed { reason: String },
    
    #[error("Gossipsub error: {reason}")]
    GossipsubError { reason: String },
    
    #[error("Kademlia DHT error: {reason}")]
    KademliaError { reason: String },
    
    #[error("mDNS discovery error: {reason}")]
    MdnsError { reason: String },
    
    #[error("Role assignment failed: {role} - {reason}")]
    RoleAssignmentFailed { role: String, reason: String },
    
    #[error("Biological behavior error: {behavior} - {reason}")]
    BiologicalBehaviorError { behavior: String, reason: String },
    
    #[error("Trust calculation failed for peer {peer_id}: {reason}")]
    TrustCalculationFailed { peer_id: String, reason: String },
    
    #[error("Resource allocation failed: {resource_type} - {reason}")]
    ResourceAllocationFailed { resource_type: String, reason: String },
    
    #[error("Configuration error: {field} - {reason}")]
    ConfigurationError { field: String, reason: String },
    
    #[error("Timeout occurred after {duration:?}: {operation}")]
    Timeout { duration: Duration, operation: String },
    
    #[error("Peer not found: {peer_id}")]
    PeerNotFound { peer_id: String },
    
    #[error("Invalid biological role: {role}")]
    InvalidBiologicalRole { role: String },
    
    #[error("Network partition detected: {affected_peers} peers affected")]
    NetworkPartition { affected_peers: usize },
    
    #[error("Byzantine behavior detected from peer {peer_id}: {evidence}")]
    ByzantineBehavior { peer_id: String, evidence: String },
    
    #[error("Security validation failed: {details}")]
    SecurityValidationFailed { details: String },
    
    #[error("Core protocol error: {source}")]
    CoreProtocolError {
        #[from]
        source: core_protocol::ProtocolError,
    },
    
    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },
    
    #[error("Serialization error: {source}")]
    SerdeError {
        #[from]
        source: serde_json::Error,
    },
    
    #[error("Bincode error: {source}")]
    BincodeError {
        #[from]
        source: bincode::Error,
    },
    
    #[error("libp2p error: {details}")]
    Libp2pError { details: String },
}