#!/bin/bash

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Default values
QEMU_MEMORY="4G"
QEMU_CPUS="4"
QEMU_MACHINE="virt"
QEMU_CPU="cortex-a72"
BUILD_TYPE="debug"
VERBOSE=""

# Help message
usage() {
    echo "Usage: $0 [command] [options]"
    echo
    echo "Commands:"
    echo "  build        Build the project"
    echo "  run          Run in QEMU"
    echo "  debug        Debug with GDB"
    echo "  test         Run tests"
    echo "  clean        Clean build artifacts"
    echo "  check        Run cargo check"
    echo "  fmt         Format code"
    echo "  clippy      Run clippy"
    echo "  doc         Generate documentation"
    echo "  update      Update dependencies"
    echo "  release     Build release version"
    echo
    echo "Options:"
    echo "  -v, --verbose       Enable verbose output"
    echo "  -m, --memory SIZE   Set QEMU memory size (default: 4G)"
    echo "  -c, --cpus NUM      Set number of CPUs (default: 4)"
    echo "  -r, --release       Use release build"
    echo "  -h, --help          Show this help message"
    exit 1
}

# Parse command line arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        build|run|debug|test|clean|check|fmt|clippy|doc|update|release)
            COMMAND="$1"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -m|--memory)
            QEMU_MEMORY="$2"
            shift 2
            ;;
        -c|--cpus)
            QEMU_CPUS="$2"
            shift 2
            ;;
        -r|--release)
            BUILD_TYPE="release"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Check if command is provided
if [ -z "$COMMAND" ]; then
    echo -e "${RED}No command specified${NC}"
    usage
fi

# Build function
build() {
    echo -e "${BLUE}Building project (${BUILD_TYPE})...${NC}"
    if [ "$BUILD_TYPE" = "release" ]; then
        cargo build --release --target aarch64-unknown-none $VERBOSE
    else
        cargo build --target aarch64-unknown-none $VERBOSE
    fi
}

# Run function
run() {
    build
    echo -e "${BLUE}Running in QEMU...${NC}"
    qemu-system-aarch64 \
        -machine "$QEMU_MACHINE" \
        -cpu "$QEMU_CPU" \
        -smp "$QEMU_CPUS" \
        -m "$QEMU_MEMORY" \
        -nographic \
        -kernel "target/aarch64-unknown-none/$BUILD_TYPE/kernel"
}

# Debug function
debug() {
    build
    echo -e "${BLUE}Starting debug session...${NC}"
    # Start QEMU in the background
    qemu-system-aarch64 \
        -machine "$QEMU_MACHINE" \
        -cpu "$QEMU_CPU" \
        -smp "$QEMU_CPUS" \
        -m "$QEMU_MEMORY" \
        -nographic \
        -kernel "target/aarch64-unknown-none/$BUILD_TYPE/kernel" \
        -S -s &
    QEMU_PID=$!

    # Start GDB
    aarch64-none-elf-gdb \
        -ex "target remote localhost:1234" \
        -ex "symbol-file target/aarch64-unknown-none/$BUILD_TYPE/kernel" \
        -ex "break kernel_main" \
        -ex "continue"

    # Clean up QEMU when GDB exits
    kill $QEMU_PID
}

# Execute command
case $COMMAND in
    build)
        build
        ;;
    run)
        run
        ;;
    debug)
        debug
        ;;
    test)
        echo -e "${BLUE}Running tests...${NC}"
        cargo test --target aarch64-unknown-none $VERBOSE
        ;;
    clean)
        echo -e "${BLUE}Cleaning build artifacts...${NC}"
        cargo clean
        ;;
    check)
        echo -e "${BLUE}Running cargo check...${NC}"
        cargo check --target aarch64-unknown-none $VERBOSE
        ;;
    fmt)
        echo -e "${BLUE}Formatting code...${NC}"
        cargo fmt --all
        ;;
    clippy)
        echo -e "${BLUE}Running clippy...${NC}"
        cargo clippy --target aarch64-unknown-none -- -D warnings
        ;;
    doc)
        echo -e "${BLUE}Generating documentation...${NC}"
        cargo doc --no-deps --target aarch64-unknown-none $VERBOSE
        ;;
    update)
        echo -e "${BLUE}Updating dependencies...${NC}"
        cargo update
        ;;
    release)
        BUILD_TYPE="release"
        build
        ;;
esac

echo -e "${GREEN}Done!${NC}"
