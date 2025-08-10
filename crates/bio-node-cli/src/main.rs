//! Main entry point for Bio P2P Node CLI
//!
//! This module provides the main function and top-level coordination for the
//! Bio P2P Node command-line interface and daemon.

use clap::Parser;
use anyhow::Result;
use tracing::{error, info, Level};

mod cli;
mod commands;
mod config;
mod daemon;
mod health;
mod logging;
mod metrics;
mod node;
mod signals;

use cli::{Cli, CommandContext};

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let cli = Cli::parse();
    
    // Initialize minimal logging for startup
    let _early_logging = logging::LoggingSystem::init_early(Level::INFO, cli.quiet)?;
    
    // Create command context
    let context = CommandContext::new(cli);
    
    // Execute the requested command
    if let Err(e) = commands::execute_command(context).await {
        error!("Command failed: {}", e);
        
        // Print error chain for debugging
        let mut source = e.source();
        while let Some(err) = source {
            error!("  Caused by: {}", err);
            source = err.source();
        }
        
        std::process::exit(1);
    }
    
    info!("Command completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;
    
    #[test]
    fn test_cli_parsing() {
        // Test basic command parsing
        let cli = Cli::try_parse_from(["bio-node", "start"]);
        assert!(cli.is_ok());
        
        let cli = Cli::try_parse_from(["bio-node", "stop"]);
        assert!(cli.is_ok());
        
        let cli = Cli::try_parse_from(["bio-node", "status"]);
        assert!(cli.is_ok());
        
        let cli = Cli::try_parse_from(["bio-node", "config"]);
        assert!(cli.is_ok());
    }
    
    #[test]
    fn test_cli_help() {
        // Ensure help can be generated without panicking
        let cmd = Cli::command();
        let help = cmd.render_help();
        assert!(!help.to_string().is_empty());
    }
    
    #[test]
    fn test_cli_version() {
        // Test version information
        let cli = Cli::try_parse_from(["bio-node", "--version"]);
        // This will exit the process in normal operation
        // but in tests we just want to ensure it parses correctly
    }
}