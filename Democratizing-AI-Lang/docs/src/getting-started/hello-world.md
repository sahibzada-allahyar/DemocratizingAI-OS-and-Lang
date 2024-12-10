# Hello World in Democratising

This guide will walk you through creating your first Democratising program, explaining each component along the way.

## Basic Hello World

Let's start with the simplest possible program:

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    println!("Hello, Democratising!");
    Ok(())
}
```

Save this in `src/main.rs` and run:
```bash
cargo run
```

### Understanding the Components

1. `use democratising::prelude::*;`
   - Imports commonly used types and traits
   - Similar to Python's `import numpy as np`

2. `fn main() -> Result<()>`
   - Main function returns a `Result` type
   - Enables proper error handling
   - `()` is the unit type (similar to `void`)

3. `Ok(())`
   - Returns success with no value
   - Required for functions returning `Result`

## Hello World with Tensors

Let's make it more interesting by using tensors:

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a tensor
    let tensor = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    println!("Hello, Tensors!\n{}", tensor);

    // Perform some operations
    let sum = tensor.sum(None)?;
    let mean = tensor.mean(1)?;

    println!("\nSum: {}", sum);
    println!("Mean along dimension 1:\n{}", mean);

    Ok(())
}
```

### Understanding Tensor Operations

1. Creating a tensor:
```rust
let tensor = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
```
- `&[2, 3]` specifies a 2x3 shape
- `vec![...]` provides the data
- `?` operator propagates any errors

2. Tensor display:
```rust
println!("Hello, Tensors!\n{}", tensor);
```
Outputs:
```
Hello, Tensors!
[[1.0, 2.0, 3.0],
 [4.0, 5.0, 6.0]]
```

3. Basic operations:
```rust
let sum = tensor.sum(None)?;    // Global sum
let mean = tensor.mean(1)?;     // Mean along dimension 1
```

## Hello World with Neural Networks

Let's create a simple neural network:

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a simple neural network
    let model = Sequential::new()
        .add(Dense::new(2, 4).with_activation(activation::relu))
        .add(Dense::new(4, 1))
        .build()?;

    // Create input data
    let input = Tensor::new(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

    // Forward pass
    let output = model.forward(&input)?;

    println!("Hello, Neural Networks!");
    println!("Input:\n{}", input);
    println!("\nOutput:\n{}", output);

    Ok(())
}
```

### Understanding Neural Network Components

1. Model creation:
```rust
let model = Sequential::new()
    .add(Dense::new(2, 4).with_activation(activation::relu))
    .add(Dense::new(4, 1))
    .build()?;
```
- Creates a sequential model
- Adds two dense layers
- First layer has ReLU activation
- Uses builder pattern for configuration

2. Input data:
```rust
let input = Tensor::new(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
```
- Creates a 3x2 input tensor
- Each row is a sample
- Each column is a feature

3. Forward pass:
```rust
let output = model.forward(&input)?;
```
- Runs input through the network
- Returns predictions

## Hello World with GPU

Let's run our tensor operations on GPU:

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Select device
    let device = if cuda::is_available() {
        println!("Using CUDA GPU");
        Device::cuda(0)?
    } else {
        println!("Using CPU");
        Device::cpu()
    };

    // Create tensor on device
    let tensor = Tensor::randn(&[1000, 1000], &device)?;

    // Operations run on GPU automatically
    let result = tensor.matmul(&tensor.transpose(0, 1)?)?;

    println!("\nMatrix multiplication shape: {:?}", result.shape());
    println!("Device: {}", result.device());

    Ok(())
}
```

### Understanding GPU Operations

1. Device selection:
```rust
let device = if cuda::is_available() {
    Device::cuda(0)?
} else {
    Device::cpu()
};
```
- Checks for CUDA availability
- Falls back to CPU if needed

2. Tensor creation on device:
```rust
let tensor = Tensor::randn(&[1000, 1000], &device)?;
```
- Creates tensor directly on GPU
- Avoids unnecessary transfers

3. GPU operations:
```rust
let result = tensor.matmul(&tensor.transpose(0, 1)?)?;
```
- Operations execute on GPU automatically
- No explicit device management needed

## Common Patterns

1. Error handling:
```rust
fn my_function() -> Result<()> {
    let x = some_operation()?;
    Ok(())
}
```

2. Device management:
```rust
let x = tensor.to_device(&device)?;
```

3. Builder pattern:
```rust
let model = ModelConfig::new()
    .learning_rate(0.001)
    .optimizer(Adam::new)
    .build()?;
```

## Next Steps

- Try modifying these examples
- Experiment with different tensor operations
- Create more complex neural networks
- Read the [Quick Start Guide](quick-start.md)
- Explore the [Basic Syntax Guide](../language-guide/basic-syntax.md)

## Troubleshooting

Common issues and solutions:

1. Shape mismatch:
```rust
// Error
let a = Tensor::new(&[2, 3], ...)?;
let b = Tensor::new(&[4, 5], ...)?;
let c = a.matmul(&b)?; // Error: incompatible shapes

// Fix
let b = Tensor::new(&[3, 4], ...)?;
let c = a.matmul(&b)?; // Works
```

2. Device mismatch:
```rust
// Error
let a = Tensor::new(&[2, 3], ...)?;
let b = a.to_device(&Device::cuda(0)?)?;
let c = &a + &b; // Error: different devices

// Fix
let a = a.to_device(&Device::cuda(0)?)?;
let c = &a + &b; // Works
```

3. Type inference:
```rust
// Error
let x = Tensor::zeros(&[3, 3]); // Error: type needed

// Fix
let x: Tensor<f32> = Tensor::zeros(&[3, 3])?;
// or
let x = Tensor::zeros(&[3, 3])?.to_dtype(DType::F32)?;
```

## Getting Help

If you encounter any issues:
- Check the [FAQ](../faq.md)
- Join our [Discord](https://discord.gg/democratising)
- Ask on [Stack Overflow](https://stackoverflow.com/questions/tagged/democratising)
