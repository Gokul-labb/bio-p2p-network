use p2p_node::{Node, NodeConfig, BiologicalRole, NetworkAddress};
use tracing::{info, error};
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,p2p_node=debug,libp2p=info")
        .init();

    info!("Starting simple bio-p2p node example");

    // Create a simple configuration
    let mut config = NodeConfig::for_testing();
    config.identity.name = Some("simple-node".to_string());
    config.network.tcp_addresses = vec!["127.0.0.1:0".parse()?];
    config.biological.primary_role = "Young".to_string();

    // Create network address
    let network_address = NetworkAddress::new(1, 1, 1)?;

    // Build and start the node
    let node = Node::builder()
        .with_config(config)
        .with_initial_role(BiologicalRole::Young)
        .with_network_address(network_address)
        .build()
        .await?;

    info!("Node created with peer ID: {}", node.peer_id());
    info!("Network address: {}", node.network_address());
    info!("Initial role: {:?}", node.current_role());

    // Start the node (this will run indefinitely)
    tokio::select! {
        result = node.start() => {
            match result {
                Ok(_) => info!("Node shut down gracefully"),
                Err(e) => error!("Node error: {}", e),
            }
        }
        _ = tokio::signal::ctrl_c() => {
            info!("Received Ctrl+C, shutting down...");
        }
    }

    Ok(())
}