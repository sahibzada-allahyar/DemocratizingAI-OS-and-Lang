# Democratising Examples

This directory contains example programs demonstrating various features and capabilities of the Democratising programming language.

## Overview

Each example is designed to showcase different aspects of the language, from basic tensor operations to complex neural network training. The examples are structured to be educational and serve as reference implementations for common AI/ML tasks.

## Examples

### 1. Tensor Operations ([tensor_ops.rs](tensor_ops.rs))
Demonstrates basic tensor operations and GPU acceleration:
- Matrix multiplication
- Element-wise operations
- Reduction operations
- Performance benchmarking
- GPU vs CPU comparison

```bash
# Run with CPU
cargo run --example tensor_ops

# Run with GPU
cargo run --example tensor_ops --features cuda
```

### 2. Neural Network Training ([neural_network.rs](neural_network.rs))
Shows how to build and train neural networks:
- Model architecture definition
- Training loop implementation
- Loss functions and optimizers
- Training visualization
- Model saving/loading

```bash
# Train a simple MLP
cargo run --example neural_network -- --model mlp

# Train a CNN with GPU acceleration
cargo run --example neural_network -- --model cnn --gpu
```

### 3. Distributed Training ([distributed_training.rs](distributed_training.rs))
Illustrates distributed model training:
- Multi-worker training
- Parameter synchronization
- Progress monitoring
- Performance scaling

```bash
# Run with 4 workers
cargo run --example distributed_training -- --num-workers 4
```

### 4. CUDA Acceleration ([cuda_acceleration.rs](cuda_acceleration.rs))
Showcases GPU acceleration features:
- CUDA kernel integration
- Memory management
- Stream handling
- Multi-GPU support

```bash
# Run CUDA examples
cargo run --example cuda_acceleration --features cuda
```

### 5. MNIST Classification ([mnist.rs](mnist.rs))
Complete example of training on the MNIST dataset:
- Data loading and preprocessing
- Model architecture
- Training and evaluation
- Visualization of results

```bash
# Train on MNIST
cargo run --example mnist --features mnist-data
```

## Running the Examples

### Prerequisites

1. Install Rust:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Install CUDA (optional, for GPU support):
```bash
# Follow NVIDIA's installation guide for your platform
```

3. Build the project:
```bash
cargo build --all-features
```

### Common Options

Most examples support the following command-line options:

- `--gpu`: Enable GPU acceleration (requires CUDA)
- `--output-dir`: Directory for saving outputs
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate for optimization

### Features

The examples support various feature flags:

- `cuda`: Enable CUDA support
- `cpu`: Force CPU-only execution
- `distributed`: Enable distributed training support
- `mnist-data`: Download and use MNIST dataset
- `visualization`: Enable result visualization
- `benchmarking`: Enable performance benchmarking

## Performance Tips

1. **GPU Acceleration**:
   - Use `--gpu` flag when available
   - Adjust batch size for optimal GPU utilization
   - Monitor GPU memory usage

2. **Distributed Training**:
   - Scale workers based on available CPU cores
   - Tune communication frequency
   - Monitor network bandwidth

3. **Memory Usage**:
   - Use appropriate data types
   - Enable memory profiling when needed
   - Clean up resources properly

## Contributing

Feel free to contribute additional examples! Please follow these guidelines:

1. Create a new `.rs` file in this directory
2. Add appropriate documentation and comments
3. Update this README.md with example details
4. Add necessary dependencies to `Cargo.toml`
5. Include tests and benchmarks
6. Follow the project's code style

## License

All examples are licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
