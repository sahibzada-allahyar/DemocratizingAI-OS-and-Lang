# Automatic Differentiation

Democratising provides built-in automatic differentiation (autodiff) capabilities. This guide explains how to use autodiff effectively for gradient computation and optimization.

## Basic Usage

### Computing Gradients

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create tensors that require gradients
    let x = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0])?
        .requires_grad(true);
    let y = Tensor::new(&[2, 2], vec![2.0, 3.0, 4.0, 5.0])?
        .requires_grad(true);

    // Forward computation
    let z = (&x * &y)?.sum(None)?;

    // Backward pass
    z.backward()?;

    // Access gradients
    let dx = x.grad()?;
    let dy = y.grad()?;

    println!("dx: {}", dx);
    println!("dy: {}", dy);

    Ok(())
}
```

### Gradient Functions

```rust
fn compute_function_gradients(x: &Tensor) -> Result<(Tensor, Tensor)> {
    // Function: f(x) = x^2 * sin(x)
    let y = x.pow(2.0)?.mul(&x.sin()?)?;

    // Compute gradient
    y.backward()?;

    // Return both output and gradient
    Ok((y, x.grad()?))
}
```

## Advanced Features

### Custom Gradients

```rust
#[derive(AutoDiff)]
struct CustomFunction;

impl Function for CustomFunction {
    fn forward(&self, inputs: &[&Tensor]) -> Result<Tensor> {
        // Custom forward computation
        let x = inputs[0];
        x.pow(3.0)
    }

    fn backward(&self, grad_output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
        // Custom gradient computation
        let x = inputs[0];
        let dx = grad_output * &(3.0 * &x.pow(2.0)?)?;
        Ok(vec![dx])
    }
}
```

### Higher-Order Derivatives

```rust
fn compute_hessian(f: impl Fn(&Tensor) -> Result<Tensor>, x: &Tensor) -> Result<Tensor> {
    // First derivative
    let y = f(x)?;
    y.backward()?;
    let first_grad = x.grad()?.detach()?;

    // Reset gradients
    x.zero_grad();

    // Second derivative
    first_grad.backward()?;
    let hessian = x.grad()?;

    Ok(hessian)
}
```

## Neural Network Integration

### Layer Gradients

```rust
impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Forward pass with gradient tracking
        let output = input.matmul(&self.weights)?;
        (&output + &self.bias)?.relu()
    }

    fn backward(&self, grad_output: &Tensor) -> Result<Tensor> {
        // Gradient computation is automatic
        grad_output.backward()?;

        // Access gradients
        let weight_grad = self.weights.grad()?;
        let bias_grad = self.bias.grad()?;

        // Return gradient with respect to input
        self.weights.t()?.matmul(grad_output)
    }
}
```

### Optimizer Integration

```rust
impl Adam {
    fn step(&mut self) -> Result<()> {
        for param in &mut self.parameters {
            // Get parameter gradients
            let grad = param.grad()?;

            // Update momentum terms
            self.update_momentum(param, &grad)?;

            // Apply gradient update
            self.apply_update(param)?;

            // Reset gradients
            param.zero_grad();
        }
        Ok(())
    }
}
```

## Memory Management

### Gradient Checkpointing

```rust
fn train_with_checkpointing(model: &mut Sequential, data: &Tensor) -> Result<()> {
    // Split model into segments
    let segments = model.checkpoint_segments(3)?;

    // Forward pass with checkpoints
    let mut activations = Vec::new();
    let mut current = data.clone();

    for segment in &segments {
        // Save input for backward pass
        let checkpoint = current.detach()?.requires_grad(true);

        // Forward through segment
        current = segment.forward(&checkpoint)?;
        activations.push(checkpoint);
    }

    // Backward pass using checkpoints
    let mut grad = current;
    for (segment, activation) in segments.iter().zip(activations.iter()).rev() {
        grad = segment.backward(&grad, activation)?;
    }

    Ok(())
}
```

### Memory Efficiency

```rust
fn efficient_backward() -> Result<()> {
    // Use in-place operations where possible
    let mut grad = Tensor::ones(&[100, 100])?;
    grad.mul_inplace(&0.5)?;

    // Free intermediate results
    {
        let temp = some_computation()?;
        let result = use_temp(&temp)?;
    } // temp freed here

    Ok(())
}
```

## Advanced Techniques

### Vector-Jacobian Products

```rust
fn compute_vjp(
    f: impl Fn(&Tensor) -> Result<Tensor>,
    x: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    // Forward pass
    let y = f(x)?;

    // Vector-Jacobian product
    y.backward_with_gradient(v)?;

    // Return VJP result
    Ok(x.grad()?)
}
```

### Jacobian-Vector Products

```rust
fn compute_jvp(
    f: impl Fn(&Tensor) -> Result<Tensor>,
    x: &Tensor,
    v: &Tensor,
) -> Result<Tensor> {
    // Forward-mode differentiation
    let x_dual = x.make_dual(v)?;
    let y_dual = f(&x_dual)?;

    // Extract tangent
    Ok(y_dual.tangent()?)
}
```

## Best Practices

### Performance Optimization

1. Use gradient accumulation for large models:
```rust
fn accumulate_gradients(
    model: &mut Sequential,
    data: &DataLoader,
    accumulation_steps: usize,
) -> Result<()> {
    for (i, batch) in data.enumerate() {
        // Forward and backward passes
        let loss = model.forward(&batch)?.mean()?;
        (loss / accumulation_steps as f32)?.backward()?;

        // Update only after accumulation
        if (i + 1) % accumulation_steps == 0 {
            optimizer.step()?;
            model.zero_grad();
        }
    }
    Ok(())
}
```

2. Avoid unnecessary gradient computation:
```rust
fn evaluate_model(model: &Sequential, data: &Tensor) -> Result<Tensor> {
    // Disable gradient computation during evaluation
    no_grad(|| {
        model.forward(data)
    })
}
```

### Memory Management

1. Clear gradients when not needed:
```rust
fn train_epoch(model: &mut Sequential, data: &DataLoader) -> Result<()> {
    for batch in data {
        // Clear old gradients
        model.zero_grad();

        // Compute new gradients
        let loss = model.forward(&batch)?;
        loss.backward()?;

        // Update parameters
        optimizer.step()?;
    }
    Ok(())
}
```

2. Use gradient checkpointing for large models:
```rust
fn train_large_model(model: &mut Sequential, data: &Tensor) -> Result<()> {
    // Enable checkpointing
    model.enable_checkpointing()?;

    // Training with reduced memory usage
    let loss = model.forward(data)?;
    loss.backward()?;

    Ok(())
}
```

## Next Steps

- Learn about [Neural Networks](neural-networks.md)
- Explore [GPU Acceleration](gpu.md)
- Study [Performance Optimization](optimization.md)
