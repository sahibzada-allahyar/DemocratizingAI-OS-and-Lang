# Contributing to Democratising

First off, thank you for considering contributing to Democratising! It's people like you that make Democratising such a great tool for democratizing AI development globally.

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [INSERT CONTACT METHOD].

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots and animated GIFs if possible
* Include your environment details (OS, Rust version, CUDA version if applicable)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

* Use a clear and descriptive title
* Provide a step-by-step description of the suggested enhancement
* Provide specific examples to demonstrate the steps
* Describe the current behavior and explain which behavior you expected to see instead
* Explain why this enhancement would be useful
* List some other languages or projects where this enhancement exists

### Pull Requests

* Fill in the required template
* Do not include issue numbers in the PR title
* Follow the [Rust styleguide](#rust-styleguide)
* Include thoughtfully-worded, well-structured tests
* Document new code based on the [Documentation Styleguide](#documentation-styleguide)
* End all files with a newline

### Development Process

1. Fork the repo
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`cargo test --all-features`)
5. Run the linter (`cargo clippy --all-features -- -D warnings`)
6. Format your code (`cargo fmt`)
7. Commit your changes (`git commit -m 'Add some amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Rust Styleguide

* Follow the official [Rust Style Guide](https://doc.rust-lang.org/1.0.0/style/README.html)
* Use `rustfmt` with our project's configuration
* Use `clippy` to catch common mistakes and improve your code
* Write documentation for all public items
* Keep functions focused and small
* Use meaningful variable names
* Add tests for new functionality

## Documentation Styleguide

* Use [Markdown](https://guides.github.com/features/mastering-markdown/)
* Reference functions, classes, and modules in backticks
* Use clear and consistent terminology
* Include code examples where appropriate
* Keep line length to 100 characters
* Use proper spelling and grammar
* Link to relevant documentation and resources

## Project Structure

```
democratising/
├── benches/          # Benchmarks
├── compiler/         # Compiler implementation
├── docs/            # Documentation
├── examples/        # Example programs
├── stdlib/          # Standard library
└── tests/           # Integration tests
```

### Key Components

* **Compiler**: The core compiler implementation
* **Standard Library**: Built-in AI and tensor operations
* **Examples**: Demonstration programs
* **Documentation**: User and developer guides
* **Tests**: Comprehensive test suite

## Getting Started

1. Install dependencies:
   ```bash
   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   # Install LLVM
   # On Ubuntu/Debian:
   sudo apt-get install llvm-14-dev libclang-14-dev

   # Install CUDA (optional, for GPU support)
   # Follow NVIDIA's installation guide for your platform
   ```

2. Build the project:
   ```bash
   cargo build --all-features
   ```

3. Run the tests:
   ```bash
   cargo test --all-features
   ```

## Community

* Join our [Discord server](INSERT_DISCORD_LINK)
* Follow us on [Twitter](INSERT_TWITTER_LINK)
* Read our [blog](INSERT_BLOG_LINK)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
