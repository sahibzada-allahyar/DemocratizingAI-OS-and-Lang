# Modules

Democratising uses a module system to organize code into logical units. This guide explains how to create, use, and organize modules effectively.

## Basic Module Structure

### File Organization

```
src/
├── lib.rs         # Library root
├── main.rs        # Binary root
├── model.rs       # Module file
└── data/          # Module directory
    ├── mod.rs     # Module root
    ├── loader.rs  # Submodule
    └── dataset.rs # Submodule
```

### Module Declaration

```rust
// lib.rs
mod model;    // Declares the model module
mod data;     // Declares the data module

// data/mod.rs
mod loader;   // Declares the loader submodule
mod dataset;  // Declares the dataset submodule

pub use loader::DataLoader;  // Re-export
pub use dataset::Dataset;    // Re-export
```

## Visibility and Access Control

### Public vs Private

```rust
// Private by default
struct Model {
    weights: Tensor,
    bias: Tensor,
}

// Public items
pub struct PublicModel {
    pub weights: Tensor,     // Public field
    private: Tensor,         // Private field
}

// Public functions
pub fn train(model: &mut Model) -> Result<()> {
    // Implementation
}
```

### Module Privacy

```rust
mod network {
    // Public interface
    pub struct Layer {
        weights: Tensor,
    }

    impl Layer {
        pub fn new() -> Self {
            // Public constructor
        }

        fn initialize(&mut self) {
            // Private method
        }
    }

    // Private helper function
    fn compute_gradients() {
        // Implementation
    }
}
```

## Organizing AI Code

### Neural Network Modules

```rust
// nn/mod.rs
pub mod layers;
pub mod optimizers;
pub mod losses;
pub mod activations;

pub use layers::Layer;
pub use optimizers::Optimizer;
pub use losses::Loss;

// nn/layers/mod.rs
pub mod dense;
pub mod conv;
pub mod dropout;

pub use dense::Dense;
pub use conv::Conv2D;
pub use dropout::Dropout;
```

### Data Processing Modules

```rust
// data/mod.rs
pub mod loader;
pub mod transform;
pub mod dataset;

pub use loader::DataLoader;
pub use transform::Transform;
pub use dataset::Dataset;

// Example dataset implementation
pub struct MNISTDataset {
    data: Tensor,
    labels: Tensor,
}

impl Dataset for MNISTDataset {
    type Item = (Tensor, Tensor);

    fn get(&self, index: usize) -> Option<Self::Item> {
        // Implementation
    }
}
```

## Module Hierarchies

### Deep Module Trees

```rust
democratising::
    nn::                    // Neural network functionality
        layers::            // Network layers
            conv::          // Convolution layers
                Conv1d
                Conv2d
                Conv3d
            rnn::           // Recurrent layers
                LSTM
                GRU
            attention::     // Attention mechanisms
                MultiHead
                SelfAttention
        optimizers::        // Optimization algorithms
            sgd::
                SGD
                Momentum
            adaptive::
                Adam
                RMSprop
```

### Cross-Module Communication

```rust
// Using types from different modules
use crate::nn::layers::Dense;
use crate::nn::optimizers::Adam;
use crate::data::DataLoader;

pub fn train_model(
    model: &mut impl Layer,
    optimizer: &mut impl Optimizer,
    data: &mut impl DataLoader,
) -> Result<()> {
    // Training implementation
}
```

## Prelude Pattern

### Creating a Prelude

```rust
// prelude/mod.rs
pub use crate::nn::{Layer, Sequential, Dense};
pub use crate::tensor::Tensor;
pub use crate::optimizers::{Optimizer, Adam};
pub use crate::losses::{Loss, MSELoss};
pub use crate::Result;

// Using the prelude
use democratising::prelude::*;

fn main() -> Result<()> {
    let model = Sequential::new()
        .add(Dense::new(784, 128))
        .add(Dense::new(128, 10))
        .build()?;

    Ok(())
}
```

## Testing Modules

### Unit Tests

```rust
// In the same file as the code
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer() -> Result<()> {
        let layer = Dense::new(10, 5);
        let input = Tensor::randn(&[2, 10])?;
        let output = layer.forward(&input)?;
        assert_eq!(output.shape(), &[2, 5]);
        Ok(())
    }
}
```

### Integration Tests

```rust
// tests/integration_test.rs
use democratising::prelude::*;

#[test]
fn test_model_training() -> Result<()> {
    let mut model = Sequential::new()
        .add(Dense::new(784, 128))
        .add(Dense::new(128, 10))
        .build()?;

    let optimizer = Adam::new(model.parameters(), 0.001)?;
    // Test implementation
    Ok(())
}
```

## Best Practices

### Module Organization

1. Keep related functionality together:
```rust
mod neural_network {
    mod layers;      // Layer implementations
    mod optimizers;  // Optimization algorithms
    mod losses;      // Loss functions
}
```

2. Use clear hierarchies:
```rust
mod data {
    mod preprocessing;  // Data preprocessing
    mod augmentation;   // Data augmentation
    mod loading;        // Data loading
}
```

### Visibility Guidelines

1. Minimize public interface:
```rust
pub struct Model {
    weights: Tensor,  // Private by default
    pub config: ModelConfig,  // Public when needed
}
```

2. Use re-exports for convenience:
```rust
// mod.rs
pub use self::dense::Dense;
pub use self::conv::Conv2D;
pub use self::dropout::Dropout;
```

### Documentation

1. Document public interfaces:
```rust
/// A dense neural network layer
pub struct Dense {
    /// Number of input features
    pub in_features: usize,
    /// Number of output features
    pub out_features: usize,
}
```

2. Include examples:
```rust
/// Creates a new dense layer
///
/// # Examples
///
/// ```
/// let layer = Dense::new(784, 128);
/// ```
pub fn new(in_features: usize, out_features: usize) -> Self {
    // Implementation
}
```

## Common Patterns

### Feature Flags

```rust
// Conditional compilation
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "distributed")]
pub mod distributed;

// Using features
#[cfg(feature = "cuda")]
pub fn gpu_accelerated() -> bool {
    cuda::is_available()
}
```

### Module Configuration

```rust
// Configuration module
pub mod config {
    #[derive(Debug, Clone)]
    pub struct ModelConfig {
        pub learning_rate: f32,
        pub batch_size: usize,
        pub device: Device,
    }
}

// Using configuration
use crate::config::ModelConfig;

pub fn train(config: &ModelConfig) -> Result<()> {
    // Implementation
}
```

## Next Steps

- Learn about [Types](types.md)
- Explore [Error Handling](error-handling.md)
- Study [Memory Management](memory-management.md)
