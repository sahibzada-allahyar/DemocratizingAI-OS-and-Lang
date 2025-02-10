# Activation Functions

Democratising provides a variety of activation functions for neural networks, with support for custom activations and advanced features.

## Basic Activations

### Common Activation Functions

```rust
use democratising::prelude::*;
use democratising::nn::activations::*;

fn basic_activations() -> Result<()> {
    // ReLU
    let relu = ReLU::new();
    let output = relu.forward(&input)?;

    // Sigmoid
    let sigmoid = Sigmoid::new();
    let output = sigmoid.forward(&input)?;

    // Tanh
    let tanh = Tanh::new();
    let output = tanh.forward(&input)?;

    // LeakyReLU
    let leaky_relu = LeakyReLU::new(negative_slope: 0.01);
    let output = leaky_relu.forward(&input)?;

    Ok(())
}
```

### Advanced Activation Functions

```rust
fn advanced_activations() -> Result<()> {
    // GELU
    let gelu = GELU::new();
    let output = gelu.forward(&input)?;

    // Swish
    let swish = Swish::new(beta: 1.0);
    let output = swish.forward(&input)?;

    // Mish
    let mish = Mish::new();
    let output = mish.forward(&input)?;

    // ELU
    let elu = ELU::new(alpha: 1.0);
    let output = elu.forward(&input)?;

    Ok(())
}
```

## Using with Layers

### In Neural Networks

```rust
fn network_with_activations() -> Result<()> {
    let model = Sequential::new(vec![
        Box::new(Dense::new(784, 128, Box::new(ReLU::new()))),
        Box::new(Dense::new(128, 64, Box::new(LeakyReLU::new(0.01)))),
        Box::new(Dense::new(64, 10, Box::new(Softmax::new()))),
    ]);

    // Forward pass
    let output = model.forward(&input)?;

    Ok(())
}
```

### Functional Interface

```rust
fn functional_activations() -> Result<()> {
    // Using activation functions directly
    let output = relu(&input)?;
    let output = sigmoid(&output)?;
    
    // Chaining activations
    let output = input
        .apply(relu)?
        .apply(dropout(0.5))?
        .apply(softmax)?;

    Ok(())
}
```

## Custom Activations

### Creating Custom Activation

```rust
#[derive(Debug, Clone)]
struct CustomActivation {
    alpha: f64,
}

impl Activation for CustomActivation {
    fn forward(&self, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Custom forward implementation
        let positive = input.relu()?;
        let negative = input.min(0.0)? * self.alpha;
        positive + negative
    }

    fn backward(&self, grad_output: &Tensor<f64>, input: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Custom backward implementation
        let grad = input.map(|x| if x > 0.0 { 1.0 } else { self.alpha })?;
        grad_output * grad
    }
}

impl CustomActivation {
    fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}
```

### Using Custom Activation

```rust
fn use_custom_activation() -> Result<()> {
    let activation = CustomActivation::new(0.01);
    
    // Use in layer
    let layer = Dense::new(
        784,
        128,
        Box::new(activation.clone()),
    );

    // Use directly
    let output = activation.forward(&input)?;

    Ok(())
}
```

## Advanced Features

### Parameterized Activations

```rust
fn parameterized_activations() -> Result<()> {
    // PReLU with learnable parameters
    let prelu = PReLU::new(
        num_parameters: 1,
        init: 0.25,
    );

    // Parametric Softplus
    let softplus = ParametricSoftplus::new(
        beta: Variable::new(Tensor::scalar(1.0)),
        threshold: 20.0,
    );

    Ok(())
}
```

### Activation Layers

```rust
fn activation_layers() -> Result<()> {
    // Activation as a layer
    let layer = ActivationLayer::new(Box::new(ReLU::new()));

    // With configuration
    let layer = ActivationLayer::builder()
        .activation(Box::new(LeakyReLU::new(0.01)))
        .inplace(true)
        .build()?;

    Ok(())
}
```

## Performance Optimization

### In-Place Operations

```rust
fn inplace_activations() -> Result<()> {
    // In-place ReLU
    let mut x = Tensor::random_normal(vec![1000, 1000])?;
    relu_inplace(&mut x)?;

    // In-place layer
    let layer = ReLU::new().inplace(true);
    layer.forward_inplace(&mut x)?;

    Ok(())
}
```

### Fused Operations

```rust
fn fused_operations() -> Result<()> {
    // Fused Linear + ReLU
    let layer = LinearReLU::new(
        in_features: 784,
        out_features: 128,
    );

    // Fused Conv + BatchNorm + ReLU
    let layer = ConvBNReLU::new(
        in_channels: 3,
        out_channels: 64,
        kernel_size: 3,
    );

    Ok(())
}
```

## Best Practices

### Activation Selection

1. Choose based on task:
   ```rust
   let activation = match task_type {
       TaskType::Classification => softmax,
       TaskType::Regression => linear,
       TaskType::ImageGen => tanh,
   };
   ```

2. Consider gradients:
   ```rust
   // Use GELU or Swish for deep networks
   let activation = if depth > 50 {
       Box::new(GELU::new())
   } else {
       Box::new(ReLU::new())
   };
   ```

### Performance Tips

1. Use in-place operations when possible:
   ```rust
   let layer = ReLU::new()
       .inplace(true)
       .build()?;
   ```

2. Batch operations:
   ```rust
   // Better than loop
   let output = activation.forward_batch(&inputs)?;
   ```

### Numerical Stability

1. Use stable variants:
   ```rust
   // Instead of exp
   let output = log_softmax(&input)?;
   ```

2. Handle edge cases:
   ```rust
   let output = if input.abs()? > 20.0 {
       input.sign()?
   } else {
       tanh(&input)?
   };
   ```

## Next Steps

- Learn about [Layers](layers.md)
- Explore [Loss Functions](loss-functions.md)
- Study [Neural Networks](../neural-networks.md)
- Understand [Automatic Differentiation](../autodiff.md)
