# Quick Start Guide

This guide will help you get started with Democratising by walking through some basic examples and common use cases.

## Creating Your First Project

1. Create a new project:
```bash
cargo new my_first_project
cd my_first_project
```

2. Add Democratising as a dependency in `Cargo.toml`:
```toml
[dependencies]
democratising = "0.1.0"
```

3. Create a simple program in `src/main.rs`:
```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a tensor
    let a = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
    println!("Tensor a:\n{}", a);

    // Perform some operations
    let b = a.transpose(0, 1)?;
    println!("\nTransposed:\n{}", b);

    Ok(())
}
```

4. Run your program:
```bash
cargo run
```

## Basic Tensor Operations

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create tensors
    let a = Tensor::randn(&[3, 3], &Device::cpu())?;
    let b = Tensor::ones(&[3, 3], &Device::cpu())?;

    // Arithmetic operations
    let c = &a + &b;
    let d = &a * &b;
    let e = a.matmul(&b)?;

    // Reductions
    let sum = e.sum(None)?;
    let mean = e.mean(1)?;
    let max = e.max(0)?;

    println!("Sum: {}", sum);
    println!("Mean along dim 1:\n{}", mean);
    println!("Max along dim 0:\n{}", max);

    Ok(())
}
```

## Simple Neural Network

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a simple feed-forward network
    let model = Sequential::new()
        .add(Dense::new(784, 128).with_activation(activation::relu))
        .add(Dense::new(128, 10).with_activation(activation::softmax))
        .build()?;

    // Generate some dummy data
    let x = Tensor::randn(&[32, 784], &Device::cpu())?;
    let y = Tensor::randn(&[32, 10], &Device::cpu())?;

    // Create optimizer and loss function
    let mut optimizer = Adam::new(model.parameters(), 0.001)?;
    let loss_fn = CrossEntropyLoss::new();

    // Single training step
    let output = model.forward(&x)?;
    let loss = loss_fn.forward(&output, &y)?;

    model.zero_grad();
    loss.backward()?;
    optimizer.step()?;

    println!("Loss: {}", loss);

    Ok(())
}
```

## GPU Acceleration

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Check if CUDA is available
    let device = if cuda::is_available() {
        Device::cuda(0)?
    } else {
        Device::cpu()
    };
    println!("Using device: {}", device);

    // Create tensors on device
    let a = Tensor::randn(&[1000, 1000], &device)?;
    let b = Tensor::randn(&[1000, 1000], &device)?;

    // Operations automatically run on GPU if available
    let c = a.matmul(&b)?;
    println!("Matrix multiplication shape: {:?}", c.shape());

    Ok(())
}
```

## Data Loading and Processing

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Load MNIST dataset
    let dataset = Dataset::mnist()?;
    let (train_data, test_data) = dataset.split(0.8)?;

    // Create data loader with batching and shuffling
    let train_loader = DataLoader::new(train_data)
        .with_batch_size(32)
        .with_shuffle(true)
        .build();

    // Iterate over batches
    for (images, labels) in train_loader {
        // Preprocess images
        let normalized = images.div(255.0)?;

        // Your training code here
        println!("Batch shape: {:?}", normalized.shape());
    }

    Ok(())
}
```

## Distributed Training

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Initialize distributed environment
    let world = init_distributed()?;

    if world.rank() == 0 {
        // Parameter server code
        let mut server = ParameterServer::new(
            "0.0.0.0:5000",
            GradientAggregation::Mean,
        )?;
        server.run().await?;
    } else {
        // Worker code
        let mut worker = Worker::new(
            world.rank(),
            "localhost:5000",
            BatchSize::new(32),
        )?;
        worker.train().await?;
    }

    Ok(())
}
```

## Model Saving and Loading

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create and train a model
    let mut model = Sequential::new()
        .add(Dense::new(100, 50).with_activation(activation::relu))
        .add(Dense::new(50, 10))
        .build()?;

    // Save the model
    model.save("my_model.pt")?;

    // Load the model
    let loaded_model = Sequential::load("my_model.pt")?;

    // Use the loaded model
    let input = Tensor::randn(&[1, 100], &Device::cpu())?;
    let output = loaded_model.forward(&input)?;
    println!("Prediction: {}", output);

    Ok(())
}
```

## Next Steps

- Read the [Basic Syntax Guide](../language-guide/basic-syntax.md) for more details
- Learn about [Memory Management](../language-guide/memory-management.md)
- Explore [Neural Network Features](../ai-features/neural-networks.md)
- Check out the [Examples](https://github.com/democratising/democratising/tree/main/examples)

## Common Patterns and Best Practices

1. Always use the `Result` type for error handling:
```rust
fn my_function() -> Result<()> {
    // Your code here
    Ok(())
}
```

2. Use type inference where possible:
```rust
let x = Tensor::zeros(&[3, 3])?; // Type inferred as f32
```

3. Move tensors to the appropriate device:
```rust
let x = tensor.to_device(&device)?;
```

4. Use the builder pattern for configuration:
```rust
let model = ModelConfig::new()
    .learning_rate(0.001)
    .optimizer(Adam::new)
    .build()?;
```

5. Clean up resources explicitly:
```rust
drop(large_tensor); // Free memory immediately
```

## Getting Help

- Join our [Discord server](https://discord.gg/democratising)
- Check the [API Documentation](https://docs.democratising.ai/api)
- Ask questions on [Stack Overflow](https://stackoverflow.com/questions/tagged/democratising)
- Report issues on [GitHub](https://github.com/democratising/democratising/issues)
