# Makefile for Democratizing AI OS

# Configuration
ARCH := aarch64
TARGET := $(ARCH)-unknown-none
BUILD_TYPE := debug
QEMU_MEMORY := 4G
QEMU_CPUS := 4
QEMU_MACHINE := virt
QEMU_CPU := cortex-a72

# Tools
CARGO := cargo
RUSTC := rustc
QEMU := qemu-system-$(ARCH)
GDB := $(ARCH)-none-elf-gdb
OBJCOPY := $(ARCH)-none-elf-objcopy
OBJDUMP := $(ARCH)-none-elf-objdump
SIZE := $(ARCH)-none-elf-size

# Directories
BUILD_DIR := target/$(TARGET)/$(BUILD_TYPE)
BOOT_DIR := boot
INITRD_DIR := initrd
SYSROOT_DIR := sysroot
IMAGES_DIR := images

# Files
KERNEL := $(BUILD_DIR)/kernel
KERNEL_BIN := $(IMAGES_DIR)/kernel.bin
KERNEL_IMG := $(IMAGES_DIR)/kernel.img

# Default target
.PHONY: all
all: kernel

# Build targets
.PHONY: kernel
kernel:
	$(CARGO) build --target $(TARGET) $(if $(filter release,$(BUILD_TYPE)),--release)
	@mkdir -p $(IMAGES_DIR)
	$(OBJCOPY) -O binary $(KERNEL) $(KERNEL_BIN)
	$(SIZE) $(KERNEL)

.PHONY: release
release:
	$(MAKE) BUILD_TYPE=release kernel

# Run targets
.PHONY: run
run: kernel
	$(QEMU) \
		-machine $(QEMU_MACHINE) \
		-cpu $(QEMU_CPU) \
		-smp $(QEMU_CPUS) \
		-m $(QEMU_MEMORY) \
		-nographic \
		-kernel $(KERNEL)

.PHONY: debug
debug: kernel
	$(QEMU) \
		-machine $(QEMU_MACHINE) \
		-cpu $(QEMU_CPU) \
		-smp $(QEMU_CPUS) \
		-m $(QEMU_MEMORY) \
		-nographic \
		-kernel $(KERNEL) \
		-S -s &
	$(GDB) \
		-ex "target remote localhost:1234" \
		-ex "symbol-file $(KERNEL)" \
		-ex "break kernel_main" \
		-ex "continue"

# Development targets
.PHONY: check
check:
	$(CARGO) check --target $(TARGET)

.PHONY: test
test:
	$(CARGO) test --target $(TARGET)

.PHONY: fmt
fmt:
	$(CARGO) fmt --all

.PHONY: clippy
clippy:
	$(CARGO) clippy --target $(TARGET) -- -D warnings

.PHONY: doc
doc:
	$(CARGO) doc --no-deps --target $(TARGET)

.PHONY: clean
clean:
	$(CARGO) clean
	rm -rf $(IMAGES_DIR)

# Setup targets
.PHONY: setup
setup:
	./scripts/setup.sh

.PHONY: update
update:
	$(CARGO) update

# Analysis targets
.PHONY: size
size: kernel
	$(SIZE) $(KERNEL)

.PHONY: dump
dump: kernel
	$(OBJDUMP) -d $(KERNEL) > $(KERNEL).dump

.PHONY: symbols
symbols: kernel
	$(OBJDUMP) -t $(KERNEL) > $(KERNEL).symbols

# Help target
.PHONY: help
help:
	@echo "Democratizing AI OS Makefile"
	@echo
	@echo "Build targets:"
	@echo "  make              - Build debug kernel"
	@echo "  make release      - Build release kernel"
	@echo "  make kernel       - Build kernel (debug or release)"
	@echo
	@echo "Run targets:"
	@echo "  make run          - Run kernel in QEMU"
	@echo "  make debug        - Debug kernel with GDB"
	@echo
	@echo "Development targets:"
	@echo "  make check        - Check code"
	@echo "  make test         - Run tests"
	@echo "  make fmt          - Format code"
	@echo "  make clippy       - Run clippy"
	@echo "  make doc          - Generate documentation"
	@echo "  make clean        - Clean build artifacts"
	@echo
	@echo "Setup targets:"
	@echo "  make setup        - Set up development environment"
	@echo "  make update       - Update dependencies"
	@echo
	@echo "Analysis targets:"
	@echo "  make size         - Show kernel size"
	@echo "  make dump         - Dump kernel assembly"
	@echo "  make symbols      - Dump kernel symbols"
	@echo
	@echo "Configuration:"
	@echo "  BUILD_TYPE       = $(BUILD_TYPE)"
	@echo "  QEMU_MEMORY     = $(QEMU_MEMORY)"
	@echo "  QEMU_CPUS       = $(QEMU_CPUS)"
	@echo "  QEMU_MACHINE    = $(QEMU_MACHINE)"
	@echo "  QEMU_CPU        = $(QEMU_CPU)"
