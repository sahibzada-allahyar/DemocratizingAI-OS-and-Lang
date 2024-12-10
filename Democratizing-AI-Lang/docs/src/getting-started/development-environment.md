# Setting Up Your Development Environment

This guide will help you set up a productive development environment for working with Democratising.

## Recommended IDE: Visual Studio Code

We recommend using Visual Studio Code (VS Code) as your primary IDE for Democratising development.

### Installing VS Code

1. Download VS Code from [code.visualstudio.com](https://code.visualstudio.com)
2. Install for your operating system
3. Launch VS Code

### Required Extensions

Install these essential extensions for Democratising development:

1. **Democratising Language Support**
   - Syntax highlighting
   - Code completion
   - Error diagnostics
   - Go to definition
   - Find references

2. **rust-analyzer**
   - Rust language support
   - Code navigation
   - Auto-completion
   - Live error detection

3. **CodeLLDB**
   - Debugging support
   - Breakpoint management
   - Variable inspection

4. **Even Better TOML**
   - TOML file support
   - Cargo.toml editing

### Recommended Extensions

These additional extensions can enhance your development experience:

1. **GitHub Copilot** (optional)
   - AI-powered code suggestions
   - Context-aware completions

2. **GitLens**
   - Enhanced Git integration
   - Code authorship
   - History viewing

3. **Error Lens**
   - Inline error display
   - Quick problem identification

4. **Test Explorer UI**
   - Test discovery and running
   - Test result visualization

## VS Code Configuration

### Settings

Add these settings to your `settings.json`:

```json
{
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "editor.renderWhitespace": "all",
    "editor.bracketPairColorization.enabled": true,
    "editor.guides.bracketPairs": true,
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.cargo.features": ["all"],
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer",
        "editor.formatOnSave": true
    }
}
```

### Keyboard Shortcuts

Recommended keyboard shortcuts for productivity:

1. Code Navigation:
   - Go to Definition: `F12`
   - Find References: `Shift+F12`
   - Quick Fix: `Ctrl+.`
   - Format Document: `Shift+Alt+F`

2. Build and Test:
   - Build: `Ctrl+Shift+B`
   - Run Tests: `Ctrl+Shift+T`
   - Debug: `F5`
   - Stop Debug: `Shift+F5`

## Alternative IDEs

### CLion

1. Install CLion from [jetbrains.com](https://www.jetbrains.com/clion/)
2. Install the Rust plugin
3. Configure Toolchains:
   - Settings → Languages & Frameworks → Rust
   - Add Cargo and rustc paths

### IntelliJ IDEA

1. Install IntelliJ IDEA
2. Install the Rust plugin
3. Configure similar to CLion

### Vim/Neovim

1. Install rust.vim
2. Add LSP support:
   - Install rust-analyzer
   - Configure LSP client
   - Add completion support

## Command Line Tools

### Essential Tools

1. **Rust Toolchain**:
```bash
rustup update
rustup component add rustfmt clippy
```

2. **Build Tools**:
```bash
cargo install cargo-watch   # Auto-rebuild on changes
cargo install cargo-edit   # Dependency management
cargo install cargo-expand # Macro expansion
```

3. **Documentation**:
```bash
cargo install mdbook      # Documentation generator
cargo install cargo-doc   # API documentation
```

### Development Tools

1. **Testing Tools**:
```bash
cargo install cargo-nextest  # Enhanced test runner
cargo install cargo-tarpaulin # Code coverage
```

2. **Debugging**:
```bash
cargo install cargo-lldb    # LLDB integration
```

3. **Performance**:
```bash
cargo install cargo-flamegraph # Performance profiling
```

## Git Configuration

### Global Git Settings

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global core.editor "code --wait"
```

### Git Hooks

Add these to `.git/hooks/pre-commit`:

```bash
#!/bin/sh
cargo fmt -- --check
cargo clippy -- -D warnings
cargo test
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Project Structure

Recommended project structure:

```
democratising/
├── .vscode/                # VS Code settings
├── src/                    # Source code
├── tests/                  # Tests
├── examples/              # Example code
├── benches/              # Benchmarks
├── docs/                 # Documentation
└── .git/                 # Git repository
```

## Environment Variables

Add these to your shell configuration:

```bash
# Rust
export RUST_BACKTRACE=1
export RUSTFLAGS="-C target-cpu=native"
export RUST_LOG=info

# CUDA (if using GPU)
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Development
export DEMOCRATISING_DEV=1
```

## Debugging Setup

### VS Code Launch Configurations

Add to `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug",
            "cargo": {
                "args": ["build"]
            },
            "args": []
        },
        {
            "type": "lldb",
            "request": "launch",
            "name": "Debug Tests",
            "cargo": {
                "args": ["test", "--no-run"]
            },
            "args": []
        }
    ]
}
```

### Debug Profiles

1. Local Development:
   - Debug symbols enabled
   - Optimizations disabled
   - CUDA debugging enabled

2. Release Testing:
   - Optimizations enabled
   - Debug info stripped
   - Performance profiling

## Code Style and Linting

### Rustfmt Configuration

Create `rustfmt.toml`:
```toml
max_width = 100
tab_spaces = 4
edition = "2021"
```

### Clippy Configuration

Create `clippy.toml`:
```toml
cognitive-complexity-threshold = 25
```

## Recommended Workflow

1. Development Cycle:
   - Write code
   - Run tests (`cargo test`)
   - Format code (`cargo fmt`)
   - Check lints (`cargo clippy`)
   - Commit changes

2. Using `cargo watch`:
```bash
cargo watch -x check -x test
```

3. Documentation:
```bash
# Generate and view docs
cargo doc --open
```

## Troubleshooting

### Common Issues

1. rust-analyzer not working:
   - Delete `target` directory
   - Reload VS Code
   - Run `cargo check`

2. CUDA not found:
   - Check PATH and LD_LIBRARY_PATH
   - Verify CUDA installation
   - Run `nvcc --version`

3. Build errors:
   - Clean build: `cargo clean`
   - Update dependencies: `cargo update`
   - Check RUST_BACKTRACE=1 output

## Getting Help

- [Discord Server](https://discord.gg/democratising)
- [GitHub Issues](https://github.com/democratising/democratising/issues)
- [Documentation](https://docs.democratising.ai)
