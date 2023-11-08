#!/bin/bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up Democratizing AI OS development environment...${NC}\n"

# Check if running on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}Error: This script is designed for macOS systems${NC}"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
brew update
brew install \
    rustup-init \
    qemu \
    aarch64-none-elf-gcc \
    aarch64-none-elf-binutils \
    llvm \
    cmake \
    ninja \
    pkg-config \
    gdb

# Install Rust
echo -e "${YELLOW}Installing Rust...${NC}"
rustup-init -y --default-toolchain nightly
source $HOME/.cargo/env

# Add required targets
echo -e "${YELLOW}Adding Rust targets...${NC}"
rustup target add aarch64-unknown-none
rustup component add rust-src
rustup component add llvm-tools-preview
rustup component add rustfmt
rustup component add clippy

# Install cargo tools
echo -e "${YELLOW}Installing cargo tools...${NC}"
cargo install cargo-binutils
cargo install cargo-watch
cargo install cargo-expand
cargo install cargo-edit
cargo install cargo-udeps
cargo install cargo-audit

# Create required directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p boot
mkdir -p initrd
mkdir -p sysroot
mkdir -p toolchain
mkdir -p cross
mkdir -p images
mkdir -p qemu
mkdir -p vm

# Set up Git hooks
echo -e "${YELLOW}Setting up Git hooks...${NC}"
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e

echo "Running pre-commit checks..."

# Format code
cargo fmt -- --check

# Run clippy
cargo clippy -- -D warnings

# Run tests
cargo test --all

echo "Pre-commit checks passed!"
EOF
chmod +x .git/hooks/pre-commit

# Configure VSCode
echo -e "${YELLOW}Configuring VSCode...${NC}"
code --install-extension rust-lang.rust-analyzer
code --install-extension tamasfe.even-better-toml
code --install-extension serayuzgur.crates
code --install-extension vadimcn.vscode-lldb
code --install-extension ms-vscode.cpptools
code --install-extension webfreak.debug
code --install-extension bungcip.better-toml
code --install-extension yzhang.markdown-all-in-one
code --install-extension streetsidesoftware.code-spell-checker

# Build initial project
echo -e "${YELLOW}Building project...${NC}"
cargo build --target aarch64-unknown-none

# Print success message
echo -e "\n${GREEN}Setup complete! You can now start developing Democratizing AI OS.${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Review the README.md file for development guidelines"
echo "2. Run 'cargo build' to build the project"
echo "3. Run 'cargo run' to test in QEMU"
echo "4. Start developing!"

# Print environment information
echo -e "\n${YELLOW}Environment Information:${NC}"
echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"
echo "QEMU version: $(qemu-system-aarch64 --version | head -n1)"
echo "GCC version: $(aarch64-none-elf-gcc --version | head -n1)"
echo "LLVM version: $(llvm-config --version)"

# Check if everything is working
echo -e "\n${YELLOW}Checking if everything is working...${NC}"
if cargo check --target aarch64-unknown-none; then
    echo -e "${GREEN}Everything is set up correctly!${NC}"
else
    echo -e "${RED}There might be some issues with the setup. Please check the error messages above.${NC}"
    exit 1
fi
