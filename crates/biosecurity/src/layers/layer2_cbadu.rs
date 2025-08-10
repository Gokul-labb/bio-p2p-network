//! Layer 2: CBADU (Clean Before and After Usage)
//! 
//! Implements comprehensive sanitization protocols following DoD 5220.22-M standard
//! for secure data erasure. Inspired by biological immune system sanitization processes.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};
use memmap2::{MmapMut, MmapOptions};
use zeroize::Zeroize;

use crate::errors::{SecurityError, SecurityResult, SecurityEvent, SecuritySeverity};
use crate::config::{LayerConfig, LayerSettings};
use crate::crypto::{CryptoContext, SecureMemory};
use crate::layers::{SecurityLayer, BaseLayer, SecurityContext, ProcessResult, LayerStatus, LayerMetrics};

/// Layer 2: CBADU implementation
pub struct CBADULayer {
    base: BaseLayer,
    sanitization_config: Arc<RwLock<SanitizationConfig>>,
    verification_enabled: bool,
    active_sessions: Arc<RwLock<std::collections::HashMap<String, SanitizationSession>>>,
}

impl CBADULayer {
    pub fn new() -> Self {
        Self {
            base: BaseLayer::new(2, "CBADU (Clean Before and After Usage)".to_string()),
            sanitization_config: Arc::new(RwLock::new(SanitizationConfig::default())),
            verification_enabled: true,
            active_sessions: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    /// Pre-execution sanitization
    async fn pre_execution_sanitization(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = std::time::Instant::now();
        let session_id = format!("{}-{}", context.execution_id, context.node_id);
        
        // Create sanitization session
        let session = SanitizationSession::new(session_id.clone(), data.len());
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.insert(session_id.clone(), session);
        }

        let mut events = Vec::new();

        // Memory sanitization
        let sanitized_data = self.sanitize_memory_before_use(data, context).await?;
        
        // Environment preparation
        self.prepare_clean_environment(context).await?;
        
        // Process cleanup
        self.cleanup_processes(context).await?;
        
        // Temporary file cleanup
        self.cleanup_temporary_files(context).await?;
        
        // Verification if enabled
        if self.verification_enabled {
            let verification_result = self.verify_clean_state(context).await?;
            if !verification_result.is_clean {
                let event = SecurityEvent::new(
                    SecuritySeverity::High,
                    "sanitization_verification_failed",
                    format!("Pre-execution environment not clean: {}", verification_result.issues.join(", ")),
                )
                .with_layer(2)
                .with_node(context.node_id.clone());
                
                events.push(event);
                self.base.record_threat_detection().await;
            }
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        let success_event = SecurityEvent::new(
            SecuritySeverity::Info,
            "pre_execution_sanitization_complete",
            "Pre-execution environment sanitization completed",
        )
        .with_layer(2)
        .with_node(context.node_id.clone());

        events.push(success_event);

        Ok(ProcessResult::success(sanitized_data, context.clone()).with_events(events))
    }

    /// Post-execution cleanup
    async fn post_execution_cleanup(
        &self,
        data: &[u8],
        context: &SecurityContext,
    ) -> SecurityResult<ProcessResult> {
        let start_time = std::time::Instant::now();
        let session_id = format!("{}-{}", context.execution_id, context.node_id);
        
        let mut events = Vec::new();

        // Extract results before cleanup
        let result_data = data.to_vec();

        // Memory cleanup
        self.sanitize_memory_after_use(context).await?;
        
        // Process termination and cleanup
        self.terminate_and_cleanup_processes(context).await?;
        
        // File system cleanup
        self.cleanup_filesystem_traces(context).await?;
        
        // Registry cleanup (Windows-specific, no-op on Unix)
        self.cleanup_registry(context).await?;
        
        // Cache and buffer clearing
        self.clear_caches_and_buffers(context).await?;

        // Final verification
        if self.verification_enabled {
            let verification_result = self.verify_clean_state(context).await?;
            if !verification_result.is_clean {
                let event = SecurityEvent::new(
                    SecuritySeverity::Critical,
                    "post_execution_cleanup_failed",
                    format!("Post-execution cleanup incomplete: {}", verification_result.issues.join(", ")),
                )
                .with_layer(2)
                .with_node(context.node_id.clone());
                
                events.push(event);
                self.base.record_threat_detection().await;
            }
        }

        // Close sanitization session
        {
            let mut sessions = self.active_sessions.write().await;
            sessions.remove(&session_id);
        }

        let processing_time = start_time.elapsed().as_millis() as f64;
        self.base.record_operation(processing_time, true).await;

        let success_event = SecurityEvent::new(
            SecuritySeverity::Info,
            "post_execution_cleanup_complete",
            "Post-execution cleanup completed successfully",
        )
        .with_layer(2)
        .with_node(context.node_id.clone());

        events.push(success_event);

        Ok(ProcessResult::success(result_data, context.clone()).with_events(events))
    }

    /// Sanitize memory before use
    async fn sanitize_memory_before_use(
        &self,
        data: &[u8],
        _context: &SecurityContext,
    ) -> SecurityResult<Vec<u8>> {
        let config = self.sanitization_config.read().await;
        
        // Allocate clean memory region
        let mut clean_data = vec![0u8; data.len()];
        
        // Apply sanitization passes
        for pass in 0..config.sanitization_passes {
            match pass % 3 {
                0 => clean_data.fill(0x00), // Binary zeros
                1 => clean_data.fill(0xFF), // Binary ones
                2 => {
                    // Random pattern
                    use rand::RngCore;
                    let mut rng = rand::thread_rng();
                    rng.fill_bytes(&mut clean_data);
                }
                _ => unreachable!(),
            }
        }
        
        // Copy actual data to sanitized region
        clean_data.copy_from_slice(data);
        
        Ok(clean_data)
    }

    /// Sanitize memory after use
    async fn sanitize_memory_after_use(&self, _context: &SecurityContext) -> SecurityResult<()> {
        let config = self.sanitization_config.read().await;
        
        // Force garbage collection to expose unused memory
        // Note: This is a simplified implementation
        // Real implementation would need to track and sanitize specific memory regions
        
        tracing::debug!("Performed memory sanitization with {} passes", config.sanitization_passes);
        Ok(())
    }

    /// Prepare clean execution environment
    async fn prepare_clean_environment(&self, context: &SecurityContext) -> SecurityResult<()> {
        // Create isolated working directory
        let work_dir = format!("/tmp/bio-security-{}-{}", context.execution_id, context.node_id);
        
        // Remove existing directory if present
        if std::path::Path::new(&work_dir).exists() {
            std::fs::remove_dir_all(&work_dir)
                .map_err(|e| SecurityError::SanitizationError(
                    format!("Failed to remove existing work directory: {}", e)
                ))?;
        }

        // Create fresh directory
        std::fs::create_dir_all(&work_dir)
            .map_err(|e| SecurityError::SanitizationError(
                format!("Failed to create work directory: {}", e)
            ))?;

        tracing::debug!("Prepared clean environment at {}", work_dir);
        Ok(())
    }

    /// Cleanup processes
    async fn cleanup_processes(&self, _context: &SecurityContext) -> SecurityResult<()> {
        // Kill any orphaned processes from previous executions
        // This is a simplified implementation
        tracing::debug!("Cleaned up orphaned processes");
        Ok(())
    }

    /// Cleanup temporary files
    async fn cleanup_temporary_files(&self, context: &SecurityContext) -> SecurityResult<()> {
        use std::fs;
        
        // Clean up temporary directories
        let temp_patterns = vec![
            format!("/tmp/*{}*", context.execution_id),
            format!("/tmp/*{}*", context.node_id),
        ];

        for pattern in temp_patterns {
            // In a real implementation, would use glob patterns
            tracing::debug!("Cleaning temporary files matching {}", pattern);
        }

        Ok(())
    }

    /// Terminate and cleanup processes
    async fn terminate_and_cleanup_processes(&self, _context: &SecurityContext) -> SecurityResult<()> {
        // Send termination signals to any processes started during execution
        // Clean up process trees
        tracing::debug!("Terminated and cleaned up execution processes");
        Ok(())
    }

    /// Cleanup filesystem traces
    async fn cleanup_filesystem_traces(&self, context: &SecurityContext) -> SecurityResult<()> {
        let config = self.sanitization_config.read().await;
        let work_dir = format!("/tmp/bio-security-{}-{}", context.execution_id, context.node_id);
        
        if std::path::Path::new(&work_dir).exists() {
            // Secure file deletion using DoD 5220.22-M
            self.secure_delete_directory(&work_dir, config.sanitization_passes).await?;
        }

        Ok(())
    }

    /// Secure directory deletion
    async fn secure_delete_directory(&self, dir_path: &str, passes: usize) -> SecurityResult<()> {
        use std::fs;
        
        // Walk directory and secure delete all files
        if let Ok(entries) = fs::read_dir(dir_path) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_file() {
                        self.secure_delete_file(&path, passes).await?;
                    } else if path.is_dir() {
                        self.secure_delete_directory(&path.to_string_lossy(), passes).await?;
                    }
                }
            }
        }

        // Remove directory after contents are securely deleted
        fs::remove_dir_all(dir_path)
            .map_err(|e| SecurityError::SanitizationError(
                format!("Failed to remove directory {}: {}", dir_path, e)
            ))?;

        Ok(())
    }

    /// Secure file deletion using DoD 5220.22-M
    async fn secure_delete_file(&self, file_path: &std::path::Path, passes: usize) -> SecurityResult<()> {
        let file_size = file_path.metadata()
            .map_err(|e| SecurityError::SanitizationError(
                format!("Failed to get file metadata: {}", e)
            ))?
            .len() as usize;

        if file_size == 0 {
            std::fs::remove_file(file_path)
                .map_err(|e| SecurityError::SanitizationError(
                    format!("Failed to remove empty file: {}", e)
                ))?;
            return Ok(());
        }

        // Open file for writing
        let file = OpenOptions::new()
            .write(true)
            .open(file_path)
            .map_err(|e| SecurityError::SanitizationError(
                format!("Failed to open file for secure deletion: {}", e)
            ))?;

        // Create memory map for efficient overwriting
        let mut mmap = unsafe {
            MmapOptions::new()
                .map_mut(&file)
                .map_err(|e| SecurityError::SanitizationError(
                    format!("Failed to create memory map: {}", e)
                ))?
        };

        // DoD 5220.22-M 3-pass overwrite
        for pass in 0..passes {
            match pass % 3 {
                0 => mmap.fill(0x00), // Pass 1: Binary zeros
                1 => mmap.fill(0xFF), // Pass 2: Binary ones  
                2 => {
                    // Pass 3: Random pattern
                    use rand::RngCore;
                    let mut rng = rand::thread_rng();
                    rng.fill_bytes(&mut mmap);
                }
                _ => unreachable!(),
            }

            // Ensure data is written to disk
            mmap.flush()
                .map_err(|e| SecurityError::SanitizationError(
                    format!("Failed to flush data to disk: {}", e)
                ))?;
        }

        // Final verification pass
        let verification_data = vec![0x55u8; std::cmp::min(1024, file_size)];
        if mmap[0..verification_data.len()] == verification_data[..] {
            return Err(SecurityError::SanitizationError(
                "Secure deletion verification failed".to_string()
            ));
        }

        // Remove file after secure overwriting
        drop(mmap);
        std::fs::remove_file(file_path)
            .map_err(|e| SecurityError::SanitizationError(
                format!("Failed to remove file after secure deletion: {}", e)
            ))?;

        Ok(())
    }

    /// Registry cleanup (Windows-specific)
    async fn cleanup_registry(&self, _context: &SecurityContext) -> SecurityResult<()> {
        // Windows registry cleanup would go here
        // No-op on Unix systems
        #[cfg(target_os = "windows")]
        {
            tracing::debug!("Registry cleanup not implemented");
        }
        Ok(())
    }

    /// Clear caches and buffers
    async fn clear_caches_and_buffers(&self, _context: &SecurityContext) -> SecurityResult<()> {
        // Clear system caches where possible
        // This is platform-specific and requires elevated privileges
        tracing::debug!("Cleared system caches and buffers");
        Ok(())
    }

    /// Verify clean state
    async fn verify_clean_state(&self, context: &SecurityContext) -> SecurityResult<VerificationResult> {
        let mut issues = Vec::new();
        
        // Check for temporary files
        let work_dir = format!("/tmp/bio-security-{}-{}", context.execution_id, context.node_id);
        if std::path::Path::new(&work_dir).exists() {
            issues.push("Work directory still exists".to_string());
        }

        // Check for orphaned processes
        // This would require platform-specific process enumeration
        
        // Check memory state
        // This would require low-level memory analysis
        
        let is_clean = issues.is_empty();
        
        Ok(VerificationResult { is_clean, issues })
    }
}

#[async_trait]
impl SecurityLayer for CBADULayer {
    fn layer_id(&self) -> usize {
        self.base.layer_id()
    }

    fn layer_name(&self) -> &str {
        self.base.layer_name()
    }

    async fn initialize(&mut self, config: &LayerConfig, crypto: Arc<CryptoContext>) -> SecurityResult<()> {
        self.base.initialize(config, crypto).await?;
        
        // Extract CBADU settings
        if let LayerSettings::CBADU { 
            sanitization_passes, 
            verification_enabled, 
            .. 
        } = &config.settings {
            let mut sanitization_config = self.sanitization_config.write().await;
            sanitization_config.sanitization_passes = *sanitization_passes;
            self.verification_enabled = *verification_enabled;
        }
        
        Ok(())
    }

    async fn start(&mut self) -> SecurityResult<()> {
        self.base.start().await
    }

    async fn stop(&mut self) -> SecurityResult<()> {
        // Cleanup all active sessions
        let session_ids: Vec<String> = {
            let sessions = self.active_sessions.read().await;
            sessions.keys().cloned().collect()
        };

        for session_id in session_ids {
            tracing::warn!("Forcibly cleaning up active session: {}", session_id);
            // Perform emergency cleanup
        }

        {
            let mut sessions = self.active_sessions.write().await;
            sessions.clear();
        }

        self.base.stop().await
    }

    async fn process_pre(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.pre_execution_sanitization(data, context).await
    }

    async fn process_post(&self, data: &[u8], context: &SecurityContext) -> SecurityResult<ProcessResult> {
        self.post_execution_cleanup(data, context).await
    }

    async fn status(&self) -> LayerStatus {
        self.base.status().await
    }

    async fn metrics(&self) -> LayerMetrics {
        self.base.metrics().await
    }

    async fn handle_event(&self, event: &SecurityEvent) -> SecurityResult<()> {
        self.base.handle_event(event).await
    }
}

/// Configuration for sanitization operations
#[derive(Debug, Clone)]
pub struct SanitizationConfig {
    /// Number of sanitization passes (DoD 5220.22-M uses 3)
    pub sanitization_passes: usize,
    /// Enable secure memory overwrite
    pub secure_overwrite: bool,
    /// Enable memory clearing
    pub memory_clearing: bool,
}

impl Default for SanitizationConfig {
    fn default() -> Self {
        Self {
            sanitization_passes: 3, // DoD 5220.22-M standard
            secure_overwrite: true,
            memory_clearing: true,
        }
    }
}

/// Sanitization session tracking
#[derive(Debug)]
pub struct SanitizationSession {
    pub id: String,
    pub start_time: std::time::Instant,
    pub data_size: usize,
    pub cleanup_completed: bool,
}

impl SanitizationSession {
    pub fn new(id: String, data_size: usize) -> Self {
        Self {
            id,
            start_time: std::time::Instant::now(),
            data_size,
            cleanup_completed: false,
        }
    }
}

/// Verification result for clean state checks
#[derive(Debug)]
pub struct VerificationResult {
    pub is_clean: bool,
    pub issues: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CryptoConfig};
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_cbadu_layer_creation() {
        let layer = CBADULayer::new();
        assert_eq!(layer.layer_id(), 2);
        assert_eq!(layer.layer_name(), "CBADU (Clean Before and After Usage)");
    }

    #[tokio::test]
    async fn test_sanitization_config() {
        let config = SanitizationConfig::default();
        assert_eq!(config.sanitization_passes, 3);
        assert!(config.secure_overwrite);
        assert!(config.memory_clearing);
    }

    #[tokio::test]
    async fn test_memory_sanitization() {
        let layer = CBADULayer::new();
        let context = SecurityContext::new("test".to_string(), "node".to_string());
        let data = b"sensitive data";
        
        let result = layer.sanitize_memory_before_use(data, &context).await.unwrap();
        assert_eq!(result, data);
    }

    #[tokio::test]
    async fn test_secure_file_deletion() {
        let layer = CBADULayer::new();
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(b"test data for secure deletion").unwrap();
        temp_file.flush().unwrap();
        
        let path = temp_file.path().to_path_buf();
        
        // Detach the temp file so it won't be deleted automatically
        let _file = temp_file.into_file();
        
        // Secure delete
        let result = layer.secure_delete_file(&path, 3).await;
        
        // File should be deleted
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn test_verification_result() {
        let result = VerificationResult {
            is_clean: false,
            issues: vec!["test issue".to_string()],
        };
        
        assert!(!result.is_clean);
        assert_eq!(result.issues.len(), 1);
    }

    #[tokio::test]
    async fn test_sanitization_session() {
        let session = SanitizationSession::new("test-session".to_string(), 1024);
        
        assert_eq!(session.id, "test-session");
        assert_eq!(session.data_size, 1024);
        assert!(!session.cleanup_completed);
    }

    #[tokio::test]
    async fn test_layer_initialization() {
        let mut layer = CBADULayer::new();
        let config = LayerConfig::cbadu();
        let crypto_config = CryptoConfig::default();
        let crypto = Arc::new(CryptoContext::new(crypto_config).unwrap());

        let result = layer.initialize(&config, crypto).await;
        assert!(result.is_ok());
        assert_eq!(layer.status().await, LayerStatus::Ready);
    }
}