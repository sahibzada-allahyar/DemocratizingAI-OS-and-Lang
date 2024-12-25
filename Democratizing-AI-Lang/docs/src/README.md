# Democratising Programming Language

Welcome to the official documentation for Democratising, a new programming language designed to make AI development accessible to everyone. This documentation will help you understand and use Democratising effectively.

## What is Democratising?

Democratising is a modern programming language and compiler implemented in Rust that combines:

- The performance and safety of systems languages
- The ease-of-use of scripting languages
- Built-in AI and machine learning capabilities
- First-class GPU acceleration support
- Distributed computing features

Our goal is to lower the barriers to AI development by providing a language that is:

- **Easy to Learn**: Clean, Python-like syntax with strong type inference
- **Safe**: Memory and thread-safe by design
- **Fast**: Native compilation with LLVM optimization
- **AI-First**: Built-in tensor operations, automatic differentiation, and neural network primitives
- **Globally Accessible**: Runs efficiently on modest hardware

## Quick Example

Here's a simple neural network implementation in Democratising:

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a simple neural network
    let model = Sequential::new()
        .add(Dense::new(784, 128).with_activation(activation::relu))
        .add(Dense::new(128, 10).with_activation(activation::softmax))
        .build()?;

    // Load MNIST dataset
    let dataset = Dataset::mnist()?;
    let (train_data, test_data) = dataset.split(0.8)?;

    // Configure training
    let trainer = Trainer::new(
        model,
        TrainerConfig::new()
            .optimizer(Adam::new(0.001)?)
            .loss(CrossEntropyLoss::new()?)
            .batch_size(32)
            .epochs(10)
            .build()?,
    )?;

    // Train the model
    trainer.train(train_data)?;

    Ok(())
}
```

## Key Features

### AI and Machine Learning
- Built-in tensor operations
- Automatic differentiation engine
- Neural network primitives
- Common architectures (CNN, RNN, Transformer)
- Popular optimizers and loss functions
- Data loading and preprocessing utilities

### Performance
- Native code compilation with LLVM
- Zero-cost abstractions
- SIMD vectorization
- Multi-threading support
- GPU acceleration
- Distributed training

### Safety and Reliability
- Strong static typing
- Memory safety without garbage collection
- Thread safety guarantees
- Comprehensive error handling
- Built-in testing and benchmarking

### Developer Experience
- Helpful error messages
- IDE integration
- Built-in formatting and linting
- Documentation generator
- Package manager
- REPL for interactive development

## Getting Started

1. [Install Democratising](getting-started/installation.md)
2. [Set up your development environment](getting-started/development-environment.md)
3. [Follow the quick start guide](getting-started/quick-start.md)
4. [Try the Hello World example](getting-started/hello-world.md)

## Learning Path

### For Beginners
1. Start with the [Basic Syntax](language-guide/basic-syntax.md)
2. Learn about [Types](language-guide/types.md)
3. Understand [Memory Management](language-guide/memory-management.md)
4. Explore [Neural Networks](ai-features/neural-networks.md)

### For Experienced Developers
1. Review the [Language Reference](language-reference.md)
2. Dive into [Advanced Features](ai-features/autodiff.md)
3. Learn about [Performance Optimization](ai-features/optimization.md)
4. Explore [System Integration](language-guide/system-integration.md)

## Community and Support

- [Join our Discord server](https://discord.gg/democratising)
- [Follow us on Twitter](https://twitter.com/democratising)
- [Report issues on GitHub](https://github.com/democratising/democratising/issues)
- [Read our blog](https://blog.democratising.ai)

## Contributing

We welcome contributions! See our [Contributing Guide](../CONTRIBUTING.md) to get started.

## License

Democratising is open source software licensed under the [MIT License](../LICENSE).

## Acknowledgments

Special thanks to:
- The Rust community for inspiration and tools
- LLVM project for the compiler backend
- Our contributors and early adopters
- The open source AI/ML community

## Citation

If you use Democratising in your research, please cite:

```bibtex
@software{democratising2025,
  title = {Democratising: An AI-First Programming Language},
  author = {Allahyar, Sahibzada},
  year = {2025},
  url = {https://github.com/democratising/democratising}
}
```

## Status

Democratising is under active development. While it's already useful for many tasks, some advanced features are still in progress. See our [roadmap](roadmap.md) for planned features and improvements.
