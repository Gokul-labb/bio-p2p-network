[package]
name = "ui-desktop"
version.workspace = true
authors.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
repository.workspace = true
description = "Desktop GUI for bio-p2p network using Tauri and egui"
keywords = ["gui", "desktop", "tauri", "egui"]

[build-dependencies]
tauri-build = { version = "1.5", features = [] }

[dependencies]
core-protocol = { path = "../core-protocol" }
p2p-node = { path = "../p2p-node" }
security = { path = "../security" }
resource-mgr = { path = "../resource-mgr" }
tauri.workspace = true
egui.workspace = true
eframe.workspace = true
serde.workspace = true
serde_json.workspace = true
tokio.workspace = true
tracing.workspace = true
anyhow.workspace = true

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]

[lints]
workspace = true
