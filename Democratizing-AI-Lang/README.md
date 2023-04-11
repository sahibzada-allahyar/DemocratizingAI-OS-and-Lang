# Democratising

[![CI](https://github.com/sahibzada/democratising/workflows/CI/badge.svg)](https://github.com/sahibzada/democratising/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.75%2B-blue.svg)](https://www.rust-lang.org)
[![CUDA Support](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

Democratising is a new programming language and compiler implemented in Rust, designed to make AI development accessible to developers worldwide. It combines the performance and safety of systems languages with the ease-of-use of scripting languages, providing built-in support for AI/ML operations, automatic differentiation, and GPU acceleration.

## Features

- **High Performance**: Native code compilation with LLVM, delivering C/C++-level performance
- **Memory Safety**: Rust-inspired ownership system without garbage collection overhead
- **GPU Acceleration**: First-class CUDA support for tensor operations and neural networks
- **AI-First Design**: Built-in tensor operations, automatic differentiation, and neural network primitives
- **Developer Friendly**: Clean Python-like syntax with powerful type inference
- **Cross-Platform**: Runs on Linux, macOS, and Windows, with optional GPU support
- **Rich Standard Library**: Comprehensive ML/AI functionality included out of the box
- **Excellent Tooling**: LSP support, formatter, linter, and debugging tools

## Quick Start

### Installation

```bash
# Install Rust (required)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone the repository
git clone https://github.com/sahibzada/democratising.git
cd democratising

# Build the project
cargo build --release
```

### Hello World

```rust
fn main() {
    println!("Hello, AI World!");
}
```

### Neural Network Example

```rust
use democratising::nn::{Model, Dense, ReLU, Sigmoid};
use democratising::tensor::Tensor;

fn main() -> Result<()> {
    // Create a simple neural network
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));

    // Generate some random training data
    let x = Tensor::randn(&[100, 784])?;
    let y = Tensor::randn(&[100, 10])?;

    // Train the model
    model.fit(&x, &y, epochs=10, batch_size=32)?;

    // Make predictions
    let predictions = model.predict(&x)?;
    println!("Predictions: {:?}", predictions);

    Ok(())
}
```

## Documentation

- [Installation Guide](docs/src/getting-started/installation.md)
- [Language Guide](docs/src/language-guide/basic-syntax.md)
- [AI Features](docs/src/ai-features/neural-networks.md)
- [API Reference](docs/api/stdlib.md)
- [Examples](examples/)

## Examples

The [examples](examples/) directory contains various examples demonstrating the language features:

- [Basic Tensor Operations](examples/tensor_ops.rs)
- [Neural Network Training](examples/neural_network.rs)
- [Distributed Training](examples/distributed_training.rs)
- [CUDA Acceleration](examples/cuda_acceleration.rs)
- [MNIST Classification](examples/mnist.rs)

## Project Structure

```
democratising/
├── benches/          # Performance benchmarks
├── compiler/         # Compiler implementation
├── docs/            # Documentation
├── examples/        # Example programs
├── stdlib/          # Standard library
└── tests/           # Integration tests
```

## Performance

Democratising aims to deliver superior performance compared to Python for AI workloads:

- Native code compilation with LLVM optimizations
- No Global Interpreter Lock (GIL)
- Efficient memory management without GC
- GPU acceleration with CUDA
- Parallel and distributed computing support

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Getting Started with Development

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and lints
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.

## Community

- [Discord](INSERT_DISCORD_LINK)
- [Twitter](INSERT_TWITTER_LINK)
- [Blog](INSERT_BLOG_LINK)

## Roadmap

- [x] Basic language implementation
- [x] Tensor operations
- [x] Automatic differentiation
- [x] Neural network primitives
- [x] CUDA support
- [ ] Distributed training
- [ ] Advanced optimizations
- [ ] More ML/AI algorithms
- [ ] Enhanced tooling
- [ ] Package manager

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Rust community for their excellent tools and documentation
- The LLVM team for their compiler infrastructure
- The CUDA team for GPU computing support
- All [contributors](CONTRIBUTORS.md) who have helped with the project

## Citation

If you use Democratising in your research, please cite:

```bibtex
@software{democratising2025,
  author = {Allahyar, Sahibzada},
  title = {Democratising: A Programming Language for AI Development},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/sahibzada/democratising}
}
