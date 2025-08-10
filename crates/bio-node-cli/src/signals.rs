//! Signal handling for graceful daemon lifecycle management
//!
//! This module provides comprehensive Unix signal handling for daemon operations
//! including graceful shutdown, configuration reload, and status reporting.

use anyhow::{Context, Result};
use tokio::sync::broadcast;
use tracing::{info, warn, error, debug};
use std::sync::Arc;

#[cfg(unix)]
use signal_hook::{consts::*, iterator::Signals};

#[cfg(windows)]
use tokio::signal::ctrl_c;

/// Signal handler for daemon process management
#[derive(Debug, Clone)]
pub struct SignalHandler {
    /// Shutdown signal sender
    shutdown_tx: broadcast::Sender<()>,
    
    /// Configuration reload signal
    reload_tx: Option<broadcast::Sender<()>>,
    
    /// Status dump signal  
    status_tx: Option<broadcast::Sender<()>>,
}

impl SignalHandler {
    /// Create new signal handler
    pub fn new(shutdown_tx: broadcast::Sender<()>) -> Result<Self> {
        Ok(Self {
            shutdown_tx,
            reload_tx: None,
            status_tx: None,
        })
    }
    
    /// Create signal handler with reload capability
    pub fn with_reload(shutdown_tx: broadcast::Sender<()>, reload_tx: broadcast::Sender<()>) -> Result<Self> {
        Ok(Self {
            shutdown_tx,
            reload_tx: Some(reload_tx),
            status_tx: None,
        })
    }
    
    /// Create signal handler with full capabilities
    pub fn with_status_dump(
        shutdown_tx: broadcast::Sender<()>,
        reload_tx: broadcast::Sender<()>,
        status_tx: broadcast::Sender<()>
    ) -> Result<Self> {
        Ok(Self {
            shutdown_tx,
            reload_tx: Some(reload_tx),
            status_tx: Some(status_tx),
        })
    }
    
    /// Run the signal handler event loop
    pub async fn run(&self) -> Result<()> {
        info!("Starting signal handler");
        
        #[cfg(unix)]
        {
            self.run_unix_signals().await
        }
        
        #[cfg(windows)]
        {
            self.run_windows_signals().await
        }
    }
    
    #[cfg(unix)]
    async fn run_unix_signals(&self) -> Result<()> {
        // Register signal handlers
        let mut signals = Signals::new(&[
            SIGTERM, // Graceful shutdown
            SIGINT,  // Interrupt (Ctrl+C)
            SIGQUIT, // Quit signal
            SIGHUP,  // Reload configuration
            SIGUSR1, // Status dump
            SIGUSR2, // Custom user signal
        ]).context("Failed to register signal handlers")?;
        
        info!("Registered Unix signal handlers: SIGTERM, SIGINT, SIGQUIT, SIGHUP, SIGUSR1, SIGUSR2");
        
        // Signal handling loop
        loop {
            if let Some(signal) = signals.pending().next() {
                match signal {
                    SIGTERM | SIGINT | SIGQUIT => {
                        info!("Received shutdown signal: {}", signal_name(signal));
                        
                        // Send graceful shutdown signal
                        if let Err(e) = self.shutdown_tx.send(()) {
                            warn!("Failed to send shutdown signal: {}", e);
                        }
                        
                        break;
                    }
                    
                    SIGHUP => {
                        info!("Received SIGHUP - configuration reload requested");
                        
                        if let Some(ref reload_tx) = self.reload_tx {
                            if let Err(e) = reload_tx.send(()) {
                                warn!("Failed to send reload signal: {}", e);
                            }
                        } else {
                            warn!("Configuration reload not supported - restart required");
                        }
                    }
                    
                    SIGUSR1 => {
                        info!("Received SIGUSR1 - status dump requested");
                        
                        if let Some(ref status_tx) = self.status_tx {
                            if let Err(e) = status_tx.send(()) {
                                warn!("Failed to send status dump signal: {}", e);
                            }
                        } else {
                            info!("Status dump via signal not supported - use 'bio-node status' command");
                        }
                    }
                    
                    SIGUSR2 => {
                        debug!("Received SIGUSR2 - custom user signal");
                        // Reserved for future custom functionality
                    }
                    
                    _ => {
                        warn!("Received unexpected signal: {}", signal);
                    }
                }
            }
            
            // Small delay to prevent busy waiting
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }
        
        info!("Signal handler exiting");
        Ok(())
    }
    
    #[cfg(windows)]
    async fn run_windows_signals(&self) -> Result<()> {
        info!("Starting Windows signal handler (Ctrl+C only)");
        
        // Windows only supports Ctrl+C handling
        match ctrl_c().await {
            Ok(()) => {
                info!("Received Ctrl+C - initiating graceful shutdown");
                
                if let Err(e) = self.shutdown_tx.send(()) {
                    warn!("Failed to send shutdown signal: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to listen for Ctrl+C: {}", e);
                return Err(e.into());
            }
        }
        
        info!("Windows signal handler exiting");
        Ok(())
    }
}

#[cfg(unix)]
fn signal_name(signal: i32) -> &'static str {
    match signal {
        SIGTERM => "SIGTERM",
        SIGINT => "SIGINT", 
        SIGQUIT => "SIGQUIT",
        SIGHUP => "SIGHUP",
        SIGUSR1 => "SIGUSR1",
        SIGUSR2 => "SIGUSR2",
        _ => "UNKNOWN",
    }
}

/// PID file management utilities
pub struct PidFile {
    path: std::path::PathBuf,
}

impl PidFile {
    /// Create new PID file manager
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            path: path.into(),
        }
    }
    
    /// Write current process PID to file
    pub async fn create(&self) -> Result<()> {
        let pid = std::process::id();
        let pid_content = pid.to_string();
        
        // Ensure parent directory exists
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await
                .with_context(|| format!("Failed to create PID file directory: {}", parent.display()))?;
        }
        
        tokio::fs::write(&self.path, pid_content).await
            .with_context(|| format!("Failed to write PID file: {}", self.path.display()))?;
        
        info!("Created PID file: {} (PID: {})", self.path.display(), pid);
        Ok(())
    }
    
    /// Read PID from file
    pub async fn read(&self) -> Result<u32> {
        let content = tokio::fs::read_to_string(&self.path).await
            .with_context(|| format!("Failed to read PID file: {}", self.path.display()))?;
        
        content.trim().parse::<u32>()
            .with_context(|| format!("Invalid PID in file: {}", content))
    }
    
    /// Check if process with PID is running
    #[cfg(unix)]
    pub fn is_process_running(&self, pid: u32) -> bool {
        use nix::sys::signal::{self, Signal};
        use nix::unistd::Pid;
        
        match signal::kill(Pid::from_raw(pid as i32), Signal::SIGTERM) {
            Ok(()) => true, // Process exists and signal was sent
            Err(nix::errno::Errno::ESRCH) => false, // No such process
            Err(_) => true, // Other error, assume process exists
        }
    }
    
    #[cfg(windows)]
    pub fn is_process_running(&self, pid: u32) -> bool {
        // Windows process checking - simplified implementation
        // In production, would use Windows API calls
        true
    }
    
    /// Remove PID file
    pub async fn remove(&self) -> Result<()> {
        if self.path.exists() {
            tokio::fs::remove_file(&self.path).await
                .with_context(|| format!("Failed to remove PID file: {}", self.path.display()))?;
            
            info!("Removed PID file: {}", self.path.display());
        }
        
        Ok(())
    }
    
    /// Check if daemon is already running
    pub async fn check_running(&self) -> Result<Option<u32>> {
        if !self.path.exists() {
            return Ok(None);
        }
        
        let pid = self.read().await?;
        
        if self.is_process_running(pid) {
            Ok(Some(pid))
        } else {
            // Stale PID file - remove it
            self.remove().await?;
            Ok(None)
        }
    }
}

/// Signal broadcasting system for daemon coordination
#[derive(Debug, Clone)]
pub struct SignalBroadcaster {
    /// Shutdown signal
    shutdown_tx: broadcast::Sender<()>,
    
    /// Configuration reload signal
    reload_tx: broadcast::Sender<()>,
    
    /// Status dump signal
    status_tx: broadcast::Sender<()>,
}

impl SignalBroadcaster {
    /// Create new signal broadcaster
    pub fn new() -> Self {
        let (shutdown_tx, _) = broadcast::channel(16);
        let (reload_tx, _) = broadcast::channel(16);
        let (status_tx, _) = broadcast::channel(16);
        
        Self {
            shutdown_tx,
            reload_tx,
            status_tx,
        }
    }
    
    /// Get shutdown signal receiver
    pub fn subscribe_shutdown(&self) -> broadcast::Receiver<()> {
        self.shutdown_tx.subscribe()
    }
    
    /// Get reload signal receiver  
    pub fn subscribe_reload(&self) -> broadcast::Receiver<()> {
        self.reload_tx.subscribe()
    }
    
    /// Get status signal receiver
    pub fn subscribe_status(&self) -> broadcast::Receiver<()> {
        self.status_tx.subscribe()
    }
    
    /// Send shutdown signal
    pub fn send_shutdown(&self) -> Result<()> {
        self.shutdown_tx.send(())
            .map_err(|_| anyhow!("No shutdown listeners"))?;
        Ok(())
    }
    
    /// Send reload signal
    pub fn send_reload(&self) -> Result<()> {
        self.reload_tx.send(())
            .map_err(|_| anyhow!("No reload listeners"))?;
        Ok(())
    }
    
    /// Send status dump signal
    pub fn send_status_dump(&self) -> Result<()> {
        self.status_tx.send(())
            .map_err(|_| anyhow!("No status listeners"))?;
        Ok(())
    }
    
    /// Create signal handler for this broadcaster
    pub fn create_handler(&self) -> Result<SignalHandler> {
        SignalHandler::with_status_dump(
            self.shutdown_tx.clone(),
            self.reload_tx.clone(),
            self.status_tx.clone()
        )
    }
}

impl Default for SignalBroadcaster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[tokio::test]
    async fn test_pid_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let pid_file_path = temp_dir.path().join("test.pid");
        let pid_file = PidFile::new(&pid_file_path);
        
        // Test creation
        pid_file.create().await.unwrap();
        assert!(pid_file_path.exists());
        
        // Test reading
        let pid = pid_file.read().await.unwrap();
        assert_eq!(pid, std::process::id());
        
        // Test removal
        pid_file.remove().await.unwrap();
        assert!(!pid_file_path.exists());
    }
    
    #[tokio::test] 
    async fn test_signal_broadcaster() {
        let broadcaster = SignalBroadcaster::new();
        
        let mut shutdown_rx = broadcaster.subscribe_shutdown();
        let mut reload_rx = broadcaster.subscribe_reload();
        let mut status_rx = broadcaster.subscribe_status();
        
        // Test shutdown signal
        broadcaster.send_shutdown().unwrap();
        assert!(shutdown_rx.try_recv().is_ok());
        
        // Test reload signal
        broadcaster.send_reload().unwrap();
        assert!(reload_rx.try_recv().is_ok());
        
        // Test status signal
        broadcaster.send_status_dump().unwrap();
        assert!(status_rx.try_recv().is_ok());
    }
    
    #[tokio::test]
    async fn test_signal_handler_creation() {
        let (shutdown_tx, _) = broadcast::channel(16);
        let handler = SignalHandler::new(shutdown_tx);
        
        assert!(handler.is_ok());
    }
}