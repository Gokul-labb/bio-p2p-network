//! Main entry point for Bio P2P Desktop Application
//! 
//! This module initializes the Tauri application, sets up the embedded node,
//! configures the UI system, and launches the desktop interface.

mod app;
mod commands;
mod config;
mod events;
mod node;
mod state;
mod ui;

use anyhow::{Context, Result};
use std::sync::Arc;
use tauri::{
    AppHandle, CustomMenuItem, Manager, SystemTray, SystemTrayEvent, 
    SystemTrayMenu, SystemTrayMenuItem, WindowEvent
};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use app::BioP2PApp;
use commands::*;
use config::AppConfig;
use events::EventManager;
use node::EmbeddedNode;
use state::AppState;

/// Application metadata
const APP_NAME: &str = "Bio P2P Network";
const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging system
    init_logging().await?;

    info!("Starting {} v{}", APP_NAME, APP_VERSION);

    // Load application configuration
    let config = AppConfig::load_or_default().await
        .context("Failed to load application configuration")?;

    // Initialize application state
    let app_state = Arc::new(RwLock::new(
        AppState::new().await
            .context("Failed to initialize application state")?
    ));

    // Create embedded Bio P2P node
    let embedded_node = Arc::new(RwLock::new(
        EmbeddedNode::new().await
            .context("Failed to create embedded node")?
    ));

    // Auto-start node if configured
    if config.ui.start_minimized || std::env::args().any(|arg| arg == "--auto-start") {
        info!("Auto-starting embedded node");
        let mut node = embedded_node.write().await;
        if let Err(e) = node.start().await {
            error!("Failed to auto-start node: {}", e);
        }
    }

    // Create system tray menu
    let tray_menu = create_system_tray_menu();
    let system_tray = SystemTray::new()
        .with_menu(tray_menu)
        .with_tooltip(APP_NAME);

    // Build Tauri application
    let app = tauri::Builder::default()
        .system_tray(system_tray)
        .on_system_tray_event(handle_system_tray_event)
        .manage(app_state.clone())
        .manage(embedded_node.clone())
        .setup(move |app| {
            setup_app(app, app_state.clone(), embedded_node.clone())
        })
        .invoke_handler(tauri::generate_handler![
            // Network commands
            get_network_status,
            add_peer,
            remove_peer,
            export_network_topology,
            
            // Biological role commands
            get_biological_roles,
            set_biological_role,
            get_available_biological_roles,
            get_biological_role_info,
            trigger_havoc_response,
            
            // Peer management commands
            get_peer_list,
            validate_peer_address,
            
            // Resource management commands
            get_resource_usage,
            get_thermal_signatures,
            update_resource_limits,
            
            // Security commands
            get_security_status,
            
            // Package processing commands
            get_package_queue,
            
            // Node control commands
            start_node,
            stop_node,
            restart_node,
            
            // Configuration commands
            import_configuration,
            export_configuration,
            
            // Utility commands
            get_node_logs,
            get_performance_metrics,
        ])
        .on_window_event(handle_window_event)
        .build(tauri::generate_context!())
        .context("Failed to build Tauri application")?;

    // Run the application
    app.run(|_app_handle, event| {
        match event {
            tauri::RunEvent::ExitRequested { api, .. } => {
                // Prevent default exit behavior to handle graceful shutdown
                api.prevent_exit();
            }
            _ => {}
        }
    });

    Ok(())
}

async fn init_logging() -> Result<()> {
    let config = AppConfig::load_or_default().await?;
    
    // Create log directory if it doesn't exist
    let log_dir = AppConfig::get_data_dir()?.join("logs");
    tokio::fs::create_dir_all(&log_dir).await?;
    
    let log_file = log_dir.join("bio-p2p-desktop.log");
    let file_appender = tracing_appender::rolling::daily(&log_dir, "bio-p2p-desktop.log");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let subscriber = tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    format!("bio_p2p_ui_desktop={},bio_p2p_core={},bio_p2p_p2p={},bio_p2p_security={},bio_p2p_resource={}",
                        config.logging.global_log_level,
                        config.logging.module_log_levels.get("bio_p2p_core").unwrap_or(&config.logging.global_log_level),
                        config.logging.module_log_levels.get("bio_p2p_p2p").unwrap_or(&config.logging.global_log_level),
                        config.logging.module_log_levels.get("bio_p2p_security").unwrap_or(&config.logging.global_log_level),
                        config.logging.module_log_levels.get("bio_p2p_resource").unwrap_or(&config.logging.global_log_level)
                    ).into()
                })
        );

    // Add console logging if enabled
    let subscriber = if config.logging.enable_console_logging {
        subscriber.with(tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(config.logging.colored_output))
    } else {
        subscriber
    };

    // Add file logging if enabled
    let subscriber = if config.logging.enable_file_logging {
        subscriber.with(tracing_subscriber::fmt::layer()
            .with_writer(non_blocking)
            .with_ansi(false)) // No ANSI colors in log files
    } else {
        subscriber
    };

    subscriber.init();

    info!("Logging initialized - log level: {}", config.logging.global_log_level);
    info!("Log file: {:?}", log_file);

    Ok(())
}

fn setup_app(
    app: &mut tauri::App,
    app_state: Arc<RwLock<AppState>>,
    embedded_node: Arc<RwLock<EmbeddedNode>>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Setting up Tauri application");

    let app_handle = app.handle();

    // Initialize event manager
    let event_manager = EventManager::new(app_handle.clone());
    let event_sender = event_manager.get_event_sender();

    // Start event processing loop
    let event_handle = app_handle.clone();
    tauri::async_runtime::spawn(async move {
        if let Err(e) = event_manager.start_event_loop().await {
            error!("Event manager loop failed: {}", e);
        }
    });

    // Setup egui application if using hybrid mode
    #[cfg(feature = "egui-integration")]
    {
        let egui_app = BioP2PApp::new(
            &eframe::CreationContext::default(),
            app_state.clone(),
            embedded_node.clone(),
        );

        // This would require additional Tauri-egui integration
        // For now, we'll use pure Tauri with web frontend
    }

    // Configure main window
    let main_window = app.get_window("main").unwrap();
    main_window.set_title(APP_NAME)?;

    // Load window state from configuration
    tauri::async_runtime::spawn(async move {
        let state = app_state.read().await;
        let window_config = &state.ui_preferences.window_layout;
        
        if let Some(size) = Some(window_config.main_window_size) {
            if let Err(e) = main_window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                width: size.0 as u32,
                height: size.1 as u32,
            })) {
                warn!("Failed to set window size: {}", e);
            }
        }

        if let Some(position) = window_config.main_window_position {
            if let Err(e) = main_window.set_position(tauri::Position::Physical(tauri::PhysicalPosition {
                x: position.0 as i32,
                y: position.1 as i32,
            })) {
                warn!("Failed to set window position: {}", e);
            }
        }

        if window_config.maximized {
            if let Err(e) = main_window.maximize() {
                warn!("Failed to maximize window: {}", e);
            }
        }
    });

    // Start periodic status updates
    let app_handle_clone = app_handle.clone();
    let embedded_node_clone = embedded_node.clone();
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            
            // Update network status
            if let Ok(node) = embedded_node_clone.try_read() {
                if let Ok(status) = node.get_network_status().await {
                    let _ = app_handle_clone.emit_all("network_status_update", &status);
                }
            }
        }
    });

    info!("Application setup completed");
    Ok(())
}

fn create_system_tray_menu() -> SystemTrayMenu {
    let show_hide = CustomMenuItem::new("show_hide".to_string(), "Show/Hide");
    let network_status = CustomMenuItem::new("network_status".to_string(), "Network Status");
    let biological_roles = CustomMenuItem::new("biological_roles".to_string(), "Biological Roles");
    let security_status = CustomMenuItem::new("security_status".to_string(), "Security Status");
    let separator = SystemTrayMenuItem::Separator;
    let preferences = CustomMenuItem::new("preferences".to_string(), "Preferences");
    let about = CustomMenuItem::new("about".to_string(), "About");
    let quit = CustomMenuItem::new("quit".to_string(), "Quit");

    SystemTrayMenu::new()
        .add_item(show_hide)
        .add_native_item(separator.clone())
        .add_item(network_status)
        .add_item(biological_roles)
        .add_item(security_status)
        .add_native_item(separator.clone())
        .add_item(preferences)
        .add_item(about)
        .add_native_item(separator)
        .add_item(quit)
}

fn handle_system_tray_event(app: &AppHandle, event: SystemTrayEvent) {
    match event {
        SystemTrayEvent::LeftClick {
            position: _,
            size: _,
            ..
        } => {
            let window = app.get_window("main").unwrap();
            if let Ok(is_visible) = window.is_visible() {
                if is_visible {
                    let _ = window.hide();
                } else {
                    let _ = window.show();
                    let _ = window.set_focus();
                }
            }
        }
        SystemTrayEvent::MenuItemClick { id, .. } => {
            match id.as_str() {
                "show_hide" => {
                    let window = app.get_window("main").unwrap();
                    if let Ok(is_visible) = window.is_visible() {
                        if is_visible {
                            let _ = window.hide();
                        } else {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                }
                "network_status" => {
                    // Show network status notification or panel
                    let _ = app.emit_all("show_panel", "network");
                }
                "biological_roles" => {
                    // Show biological roles panel
                    let _ = app.emit_all("show_panel", "biological");
                }
                "security_status" => {
                    // Show security status panel
                    let _ = app.emit_all("show_panel", "security");
                }
                "preferences" => {
                    // Open preferences window/panel
                    let _ = app.emit_all("show_panel", "preferences");
                }
                "about" => {
                    // Show about dialog
                    let _ = app.emit_all("show_dialog", "about");
                }
                "quit" => {
                    info!("User requested application quit");
                    // Perform graceful shutdown
                    app.exit(0);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

fn handle_window_event(event: tauri::GlobalWindowEvent) {
    match event.event() {
        WindowEvent::CloseRequested { api, .. } => {
            info!("Window close requested");
            
            // Check if we should minimize to tray instead of closing
            // This would need to read from configuration
            let should_minimize_to_tray = true; // TODO: Read from config

            if should_minimize_to_tray {
                event.window().hide().unwrap();
                api.prevent_close();
            } else {
                // Allow window to close and exit application
                info!("Closing application");
                std::process::exit(0);
            }
        }
        WindowEvent::Focused(focused) => {
            if *focused {
                info!("Window gained focus");
            } else {
                info!("Window lost focus");
            }
        }
        WindowEvent::Resized(size) => {
            info!("Window resized to: {}x{}", size.width, size.height);
            // TODO: Save window size to configuration
        }
        WindowEvent::Moved(position) => {
            info!("Window moved to: {}x{}", position.x, position.y);
            // TODO: Save window position to configuration
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_logging_initialization() {
        // Test that logging can be initialized without errors
        let result = init_logging().await;
        assert!(result.is_ok(), "Logging initialization should succeed");
    }

    #[test]
    fn test_system_tray_menu_creation() {
        let menu = create_system_tray_menu();
        // Basic test that menu creation doesn't panic
        // In a real test, we'd verify menu items are present
        assert!(true);
    }

    #[tokio::test]
    async fn test_app_state_creation() {
        let result = AppState::new().await;
        assert!(result.is_ok(), "App state creation should succeed");
    }

    #[tokio::test]
    async fn test_embedded_node_creation() {
        let result = EmbeddedNode::new().await;
        assert!(result.is_ok(), "Embedded node creation should succeed");
    }
}