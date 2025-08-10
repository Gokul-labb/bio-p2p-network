//! Build instructions and platform configuration for Bio P2P Desktop
//! 
//! This document provides comprehensive build instructions, cross-platform
//! configuration, and deployment guidance for the desktop application.

# Bio P2P Desktop Application Build Guide

## Overview

The Bio P2P Desktop Application is built using:
- **Tauri** - Cross-platform desktop app framework
- **egui** - Immediate mode GUI framework
- **Rust** - Core application logic and embedded node
- **libp2p** - Peer-to-peer networking stack

## Prerequisites

### System Requirements

**Minimum Requirements:**
- RAM: 4 GB
- Storage: 1 GB free space  
- CPU: 64-bit processor
- Network: Broadband internet connection

**Recommended Requirements:**
- RAM: 8 GB or more
- Storage: 5 GB free space
- CPU: Multi-core 64-bit processor
- Network: High-speed broadband

### Development Environment

1. **Rust Toolchain** (1.70+)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup update stable
```

2. **Node.js** (16+) - For Tauri CLI
```bash
# Via package manager or from nodejs.org
npm install -g @tauri-apps/cli@latest
```

3. **System Dependencies**

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    wget \
    libssl-dev \
    libgtk-3-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libwebkit2gtk-4.0-dev \
    protobuf-compiler
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install -y \
    gcc \
    gcc-c++ \
    openssl-devel \
    gtk3-devel \
    libayatana-appindicator-gtk3-devel \
    librsvg2-devel \
    webkit2gtk3-devel \
    protobuf-compiler
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Via Homebrew
brew install protobuf
```

**Windows:**
- Install Visual Studio Build Tools or Visual Studio with C++ support
- Install protobuf: `choco install protoc` (via Chocolatey)
- Or download protobuf from GitHub releases

## Building from Source

### Clone Repository

```bash
git clone https://github.com/bio-p2p/bio-p2p.git
cd bio-p2p/bio-p2p-ui-desktop
```

### Development Build

```bash
# Install dependencies and build core components
cargo build

# Run in development mode with hot reload
cargo tauri dev
```

### Production Build

```bash
# Build optimized release version
cargo tauri build
```

### Build Artifacts

After building, find artifacts in:
- **Linux:** `src-tauri/target/release/bundle/`
  - `deb/bio-p2p-desktop_0.1.0_amd64.deb`
  - `rpm/bio-p2p-desktop-0.1.0-1.x86_64.rpm`
  - `appimage/bio-p2p-desktop_0.1.0_amd64.AppImage`

- **macOS:** `src-tauri/target/release/bundle/`
  - `macos/Bio P2P Network.app`
  - `dmg/Bio P2P Network_0.1.0_x64.dmg`

- **Windows:** `src-tauri/target/release/bundle/`
  - `msi/Bio P2P Network_0.1.0_x64_en-US.msi`
  - `nsis/Bio P2P Network_0.1.0_x64-setup.exe`

## Platform-Specific Configuration

### Linux

**Desktop Integration:**
```ini
# /usr/share/applications/bio-p2p-desktop.desktop
[Desktop Entry]
Name=Bio P2P Network
Comment=Biological Peer-to-Peer Network Node
Exec=/usr/bin/bio-p2p-desktop
Icon=bio-p2p-desktop
Terminal=false
Type=Application
Categories=Network;P2P;
StartupWMClass=bio-p2p-desktop
```

**System Service (Optional):**
```ini
# /etc/systemd/user/bio-p2p-desktop.service
[Unit]
Description=Bio P2P Network Desktop Service
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/bio-p2p-desktop --service
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

### macOS

**App Bundle Structure:**
```
Bio P2P Network.app/
├── Contents/
│   ├── Info.plist
│   ├── MacOS/
│   │   └── bio-p2p-desktop
│   ├── Resources/
│   │   ├── icon.icns
│   │   └── assets/
│   └── Frameworks/
```

**Code Signing (for distribution):**
```bash
# Sign the application
codesign --force --options runtime \
  --sign "Developer ID Application: Your Name" \
  --entitlements entitlements.plist \
  "Bio P2P Network.app"

# Create notarized DMG
create-dmg \
  --volname "Bio P2P Network" \
  --window-pos 200 120 \
  --window-size 800 450 \
  --icon-size 100 \
  --icon "Bio P2P Network.app" 200 190 \
  --hide-extension "Bio P2P Network.app" \
  --app-drop-link 600 185 \
  "Bio P2P Network.dmg" \
  "Bio P2P Network.app"
```

### Windows

**Installer Configuration:**
```nsi
# installer.nsi (NSIS script)
!define APPNAME "Bio P2P Network"
!define COMPANYNAME "Bio P2P Organization"
!define DESCRIPTION "Biological Peer-to-Peer Network Node"
!define VERSIONMAJOR 0
!define VERSIONMINOR 1
!define VERSIONBUILD 0

# Registry integration
WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
  "DisplayName" "${APPNAME}"
WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
  "UninstallString" "$INSTDIR\uninstall.exe"
WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APPNAME}" \
  "QuietUninstallString" "$INSTDIR\uninstall.exe /S"
```

**Windows Defender Exclusion (if needed):**
```powershell
# Add to Windows Defender exclusions
Add-MpPreference -ExclusionPath "C:\Program Files\Bio P2P Network"
Add-MpPreference -ExclusionProcess "bio-p2p-desktop.exe"
```

## Icon Assets

### Required Icon Formats

Create icons in multiple formats and sizes:

**ICO (Windows):**
- 16x16, 32x32, 48x48, 64x64, 128x128, 256x256 pixels

**ICNS (macOS):**
- 16x16@1x, 16x16@2x, 32x32@1x, 32x32@2x
- 128x128@1x, 128x128@2x, 256x256@1x, 256x256@2x
- 512x512@1x, 512x512@2x

**PNG (Linux):**
- 16x16, 32x32, 64x64, 128x128, 256x256, 512x512 pixels

### Icon Generation Script

```bash
#!/bin/bash
# generate_icons.sh

SOURCE_PNG="icon-source.png"  # 1024x1024 source image

# Create PNG sizes
for size in 16 32 64 128 256 512; do
    convert "$SOURCE_PNG" -resize ${size}x${size} "icons/${size}x${size}.png"
done

# Create ICO file (Windows)
convert "$SOURCE_PNG" -define icon:auto-resize=256,128,64,48,32,16 "icons/icon.ico"

# Create ICNS file (macOS) - requires iconutil on macOS
mkdir -p Bio.iconset
for size in 16 32 128 256 512; do
    convert "$SOURCE_PNG" -resize ${size}x${size} "Bio.iconset/icon_${size}x${size}.png"
    convert "$SOURCE_PNG" -resize $((size*2))x$((size*2)) "Bio.iconset/icon_${size}x${size}@2x.png"
done
iconutil -c icns Bio.iconset
mv Bio.icns icons/icon.icns
rm -rf Bio.iconset
```

## Configuration Files

### Tauri Configuration

Key configuration sections in `tauri.conf.json`:

```json
{
  "build": {
    "devPath": "../dist",
    "distDir": "../dist",
    "beforeBuildCommand": "",
    "beforeDevCommand": ""
  },
  "package": {
    "productName": "Bio P2P Network",
    "version": "0.1.0"
  },
  "tauri": {
    "bundle": {
      "identifier": "org.bio-p2p.desktop",
      "category": "Network",
      "shortDescription": "Biological P2P Network Node",
      "longDescription": "A revolutionary distributed computing platform...",
      "targets": ["deb", "rpm", "appimage", "dmg", "msi", "nsis"]
    },
    "allowlist": {
      "all": false,
      "fs": {
        "all": true,
        "scope": ["$APPDATA", "$DOCUMENT"]
      },
      "notification": {
        "all": true
      },
      "window": {
        "all": false,
        "close": true,
        "hide": true,
        "show": true
      }
    },
    "windows": [{
      "title": "Bio P2P Network",
      "width": 1200,
      "height": 800,
      "minWidth": 800,
      "minHeight": 600
    }],
    "systemTray": {
      "iconPath": "icons/icon.png",
      "tooltip": "Bio P2P Network"
    }
  }
}
```

### Cargo Configuration

```toml
# .cargo/config.toml
[build]
target = "x86_64-unknown-linux-gnu"

[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=lld"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "link-arg=-Wl,-rpath,@loader_path"]

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-cpu=x86-64"]
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/build.yml
name: Build Bio P2P Desktop

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    strategy:
      matrix:
        platform: [ubuntu-latest, macos-latest, windows-latest]
    runs-on: ${{ matrix.platform }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          profile: minimal
          override: true
      
      - name: Cache Cargo
        uses: actions/cache@v3
        with:
          path: |
            ~/.cargo/registry
            ~/.cargo/git
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
      
      - name: Install Linux dependencies
        if: matrix.platform == 'ubuntu-latest'
        run: |
          sudo apt update
          sudo apt install -y libgtk-3-dev libwebkit2gtk-4.0-dev \
            libayatana-appindicator3-dev librsvg2-dev
      
      - name: Install Tauri CLI
        run: cargo install tauri-cli@^1.0
      
      - name: Build application
        run: cargo tauri build
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: bio-p2p-desktop-${{ matrix.platform }}
          path: src-tauri/target/release/bundle/
```

## Troubleshooting

### Common Build Issues

**1. WebKit2GTK Missing (Linux):**
```bash
error: failed to run custom build command for `webkit2gtk-sys`

# Solution:
sudo apt install libwebkit2gtk-4.0-dev
```

**2. Protobuf Compiler Missing:**
```bash
error: failed to run custom build command for `libp2p`

# Solution:
# Linux: sudo apt install protobuf-compiler
# macOS: brew install protobuf  
# Windows: choco install protoc
```

**3. Icon Generation Issues:**
```bash
# Ensure ImageMagick is installed
# Linux: sudo apt install imagemagick
# macOS: brew install imagemagick
# Windows: Download from imagemagick.org
```

**4. Code Signing Errors (macOS):**
```bash
# Check developer certificate
security find-identity -v -p codesigning

# Verify entitlements
codesign -d --entitlements :- "Bio P2P Network.app"
```

### Performance Optimization

**1. Binary Size Reduction:**
```toml
# Cargo.toml
[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true
```

**2. Compile Time Optimization:**
```toml
# Use mold linker on Linux
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold"]
```

**3. Runtime Performance:**
- Enable hardware acceleration
- Optimize icon loading
- Use async operations for network calls

## Distribution

### Package Managers

**Linux - Debian/Ubuntu:**
```bash
# Add to APT repository
echo "deb [signed-by=/usr/share/keyrings/bio-p2p.gpg] \
  https://packages.bio-p2p.org/debian stable main" | \
  sudo tee /etc/apt/sources.list.d/bio-p2p.list

sudo apt update
sudo apt install bio-p2p-desktop
```

**macOS - Homebrew:**
```bash
# Add tap
brew tap bio-p2p/bio-p2p

# Install
brew install --cask bio-p2p-desktop
```

**Windows - Chocolatey:**
```bash
# Install via Chocolatey
choco install bio-p2p-desktop
```

**Universal - Flatpak:**
```bash
# Install via Flatpak
flatpak install flathub org.bio_p2p.desktop
```

### Direct Downloads

Provide direct download links for all platforms:
- **Linux:** `.deb`, `.rpm`, `.AppImage`
- **macOS:** `.dmg`  
- **Windows:** `.msi`, `.exe`

Include checksums (SHA256) for verification:
```bash
# Generate checksums
sha256sum bio-p2p-desktop_0.1.0_amd64.deb > checksums.txt
sha256sum Bio\ P2P\ Network_0.1.0_x64.dmg >> checksums.txt
sha256sum Bio\ P2P\ Network_0.1.0_x64-setup.exe >> checksums.txt
```

## User Interface Mockups

### Main Dashboard
```
┌─────────────────────────────────────────────────────────────────┐
│ 🧬 Bio P2P Network                              ● 23 peers      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  🌐 Network        🐜 Biological      📊 Resources             │
│ ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│ │ ● Connected │   │ 3 Active    │   │ CPU: 45%    │             │
│ │ 23 peers    │   │ Roles       │   │ RAM: 2.1GB  │             │
│ │ Quality:    │   │ Learning:   │   │ Net: 5MB/s  │             │
│ │ Excellent   │   │ 72%         │   │ Thermal:Good│             │
│ └─────────────┘   └─────────────┘   └─────────────┘             │
│                                                                 │
│  🛡️ Security      📦 Packages      🗺️ Topology                │
│ ┌─────────────┐   ┌─────────────┐   ┌─────────────┐             │
│ │ All Layers  │   │ 5 in Queue  │   │ Interactive │             │
│ │ Active      │   │ 156 Done    │   │ Network     │             │
│ │ 0 Threats   │   │ 99.2% Rate  │   │ Map         │             │
│ │ Secure      │   │ Processing  │   │             │             │
│ └─────────────┘   └─────────────┘   └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Biological Roles Panel
```
┌─────────────────────────────────────────────────────────────────┐
│ 🐜 Biological Roles                      🔄 Auto-Assignment: ✓ │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Active Roles:                                                   │
│                                                                 │
│ ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐│
│ │🐦 YoungNode      │  │🤝 TrustNode      │  │🌡️ ThermalNode    ││
│ │Crow Learning     │  │Social Bonding    │  │Temperature Mon.  ││
│ │                  │  │                  │  │                  ││
│ │Performance: 78%  │  │Performance: 85%  │  │Performance: 92%  ││
│ │Energy: 82%       │  │Energy: 75%       │  │Energy: 88%       ││
│ │Adaptation: 91%   │  │Adaptation: 63%   │  │Adaptation: 45%   ││
│ │                  │  │                  │  │                  ││
│ │📊 ⏸ ⚙           │  │📊 ⏸ ⚙           │  │📊 ⏸ ⚙           ││
│ └──────────────────┘  └──────────────────┘  └──────────────────┘│
│                                                                 │
│ Available Roles:                                                │
│ 🐜 CasteNode        - Ant Colony Division    [Activate]         │
│ 🦟 HAVOCNode        - Emergency Response     [Activate]         │
│ 🦜 ImitateNode      - Pattern Learning       [Activate]         │
│ 🐢 HatchNode        - Group Coordination     [Activate]         │
└─────────────────────────────────────────────────────────────────┘
```

This build guide provides comprehensive instructions for building, configuring, and distributing the Bio P2P Desktop Application across all major platforms, ensuring users can easily install and run the biological P2P network node.