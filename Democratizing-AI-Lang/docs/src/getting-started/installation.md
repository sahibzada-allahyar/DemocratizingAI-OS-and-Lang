# Installing Democratising

This guide will help you install Democratising and set up your development environment.

## System Requirements

### Minimum Requirements
- CPU: x86_64 processor
- RAM: 4GB
- Storage: 2GB free space
- Operating System:
  - Linux (Ubuntu 20.04+, Fedora 34+)
  - macOS (11.0+)
  - Windows 10/11 with WSL2

### Recommended Requirements
- CPU: Modern multi-core processor
- RAM: 8GB or more
- Storage: 5GB free space
- GPU: NVIDIA GPU with CUDA support (for GPU acceleration)

### Required Software
- Rust 1.75.0 or later
- LLVM 16.0 or later
- CMake 3.10 or later
- Git
- Python 3.7+ (for documentation and testing)

## Installation Methods

### Using Cargo (Recommended)

1. Install Rust if you haven't already:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install Democratising:
```bash
cargo install democratising
```

3. Verify the installation:
```bash
democratising --version
```

### Building from Source

1. Clone the repository:
```bash
git clone https://github.com/democratising/democratising.git
cd democratising
```

2. Build the project:
```bash
cargo build --release
```

3. Install the binary:
```bash
cargo install --path .
```

## Platform-Specific Instructions

### Linux

1. Install system dependencies:

#### Ubuntu/Debian:
```bash
# Install LLVM and other dependencies
sudo apt update
sudo apt install llvm-16-dev libclang-16-dev cmake python3 python3-pip

# Install CUDA (optional, for GPU support)
sudo apt install nvidia-cuda-toolkit
```

#### Fedora:
```bash
sudo dnf install llvm-devel clang-devel cmake python3 python3-pip

# Install CUDA (optional, for GPU support)
sudo dnf install cuda
```

### macOS

1. Install dependencies using Homebrew:
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install llvm cmake python3

# Install CUDA (optional, for GPU support on Intel Macs)
# Note: M1/M2 Macs currently don't support CUDA
```

### Windows

1. Install WSL2 (recommended):
```powershell
wsl --install
```

2. Follow Linux instructions within WSL2

OR

1. Install dependencies natively:
- Install [Visual Studio](https://visualstudio.microsoft.com/) with C++ support
- Install [LLVM](https://releases.llvm.org/)
- Install [CMake](https://cmake.org/download/)
- Install [Python](https://www.python.org/downloads/)
- Install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (optional)

## GPU Support

### NVIDIA GPUs

1. Install CUDA Toolkit:
- Download from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
- Follow platform-specific installation instructions

2. Install Democratising with CUDA support:
```bash
cargo install democratising --features="cuda"
```

3. Verify GPU support:
```bash
democratising --check-gpu
```

### Other GPUs

- AMD ROCm support is planned for future releases
- Apple Metal support is under development for M1/M2 Macs
- Vulkan support is planned for cross-platform GPU acceleration

## Development Tools

### IDE Support

1. Visual Studio Code (recommended):
- Install [VS Code](https://code.visualstudio.com/)
- Install recommended extensions:
  - Democratising Language Support
  - rust-analyzer
  - CodeLLDB
  - Even Better TOML

2. Other supported IDEs:
- CLion with Rust plugin
- IntelliJ IDEA with Rust plugin
- Vim/Neovim with rust.vim

### Additional Tools

1. Install development tools:
```bash
# Install useful Cargo extensions
cargo install cargo-watch cargo-edit cargo-expand

# Install documentation tools
cargo install mdbook
```

## Environment Variables

Add these to your shell configuration:

```bash
# Add Democratising to PATH
export PATH="$HOME/.cargo/bin:$PATH"

# CUDA configuration (if using GPU support)
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Optional: Enable debug logging
export RUST_LOG=debug
```

## Troubleshooting

### Common Issues

1. LLVM not found:
```bash
# Ubuntu/Debian
sudo apt install llvm-16-dev libclang-16-dev

# macOS
brew install llvm
export LLVM_CONFIG=$(brew --prefix llvm)/bin/llvm-config
```

2. CUDA not found:
```bash
# Check CUDA installation
nvcc --version

# Ensure CUDA is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

3. Compilation errors:
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build
```

### Getting Help

If you encounter any issues:

1. Check the [FAQ](../faq.md)
2. Search [existing issues](https://github.com/democratising/democratising/issues)
3. Ask on [Discord](https://discord.gg/democratising)
4. Open a new [GitHub issue](https://github.com/democratising/democratising/issues/new)

## Next Steps

- [Set up your development environment](development-environment.md)
- [Try the quick start guide](quick-start.md)
- [Read the basic syntax guide](../language-guide/basic-syntax.md)
