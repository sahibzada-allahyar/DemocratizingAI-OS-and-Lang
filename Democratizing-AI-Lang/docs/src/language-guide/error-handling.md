# Error Handling

Democratising uses Rust's error handling system with some enhancements for AI-specific error cases. This guide explains how to handle errors effectively in your code.

## Basic Error Handling

### The Result Type

All operations that can fail return a `Result`:

```rust
fn main() -> Result<()> {
    let tensor = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let sum = tensor.sum(None)?;
    println!("Sum: {}", sum);
    Ok(())
}
```

### Error Propagation

Use the `?` operator to propagate errors:

```rust
fn process_tensor(tensor: &Tensor) -> Result<f32> {
    let squared = tensor.pow(2.0)?;
    let sum = squared.sum(None)?;
    let mean = sum / tensor.numel()? as f32;
    Ok(mean)
}
```

## Error Types

### Standard Error Types

```rust
#[derive(Debug, Error)]
pub enum DemoError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Device mismatch: {0} vs {1}")]
    DeviceMismatch(Device, Device),

    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaError),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}
```

### Custom Error Types

Creating your own error type:

```rust
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Invalid layer configuration: {0}")]
    InvalidConfig(String),

    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    #[error("Shape mismatch in {layer}: expected {expected:?}, got {actual:?}")]
    LayerShapeMismatch {
        layer: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
```

## Error Context

### Adding Context

Use `context` to add information to errors:

```rust
use anyhow::Context;

fn load_model(path: &str) -> Result<Model> {
    let file = File::open(path)
        .context("Failed to open model file")?;

    let model = Model::from_file(&file)
        .context("Failed to deserialize model")?;

    Ok(model)
}
```

### Rich Error Messages

```rust
fn validate_tensor(tensor: &Tensor) -> Result<()> {
    if tensor.numel()? == 0 {
        bail!("Tensor cannot be empty");
    }

    ensure!(
        tensor.dim()? <= 4,
        "Tensor dimension {} exceeds maximum of 4",
        tensor.dim()?
    );

    Ok(())
}
```

## AI-Specific Error Handling

### GPU Errors

```rust
fn gpu_operation() -> Result<Tensor> {
    if !cuda::is_available() {
        bail!("CUDA is not available");
    }

    let device = Device::cuda(0)
        .context("Failed to initialize CUDA device")?;

    let tensor = Tensor::randn(&[1000, 1000])
        .context("Failed to create tensor")?
        .to_device(&device)
        .context("Failed to move tensor to GPU")?;

    Ok(tensor)
}
```

### Neural Network Errors

```rust
impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let input_dim = input.size(-1)?;
        if input_dim != self.input_size {
            return Err(ModelError::LayerShapeMismatch {
                layer: "Dense".into(),
                expected: vec![self.input_size],
                actual: vec![input_dim],
            }.into());
        }

        // Layer computation...
        Ok(output)
    }
}
```

## Error Recovery

### Fallback Strategies

```rust
fn compute_with_fallback(tensor: &Tensor) -> Result<Tensor> {
    // Try GPU first
    if let Ok(device) = Device::cuda(0) {
        if let Ok(result) = compute_on_gpu(tensor, &device) {
            return Ok(result);
        }
        // GPU failed, fall back to CPU
        warn!("GPU computation failed, falling back to CPU");
    }

    compute_on_cpu(tensor)
}
```

### Graceful Degradation

```rust
fn train_model(model: &mut Model, data: &DataLoader) -> Result<()> {
    for (batch_idx, (x, y)) in data.enumerate() {
        match train_batch(model, &x, &y) {
            Ok(_) => continue,
            Err(e) => {
                warn!("Failed to train batch {}: {}", batch_idx, e);
                // Skip this batch and continue
                continue;
            }
        }
    }
    Ok(())
}
```

## Best Practices

### Error Logging

```rust
use log::{error, warn, info};

fn process() -> Result<()> {
    let tensor = match create_tensor() {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create tensor: {}", e);
            return Err(e);
        }
    };

    if let Err(e) = validate_tensor(&tensor) {
        warn!("Tensor validation failed: {}", e);
    }

    info!("Tensor processing completed");
    Ok(())
}
```

### Error Conversion

```rust
impl From<CudaError> for DemoError {
    fn from(error: CudaError) -> Self {
        DemoError::CudaError(error)
    }
}

impl From<std::io::Error> for DemoError {
    fn from(error: std::io::Error) -> Self {
        DemoError::IoError(error)
    }
}
```

### Testing Error Conditions

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch() -> Result<()> {
        let a = Tensor::zeros(&[2, 3])?;
        let b = Tensor::zeros(&[4, 5])?;

        let result = a.matmul(&b);
        assert!(matches!(
            result.unwrap_err(),
            DemoError::ShapeMismatch { .. }
        ));

        Ok(())
    }
}
```

## Common Error Patterns

### Resource Cleanup

```rust
fn process_file() -> Result<()> {
    let file = File::open("data.txt")?;
    // File automatically closed when dropped

    let tensor = Tensor::from_file(&file)?;
    // Tensor memory automatically freed

    Ok(())
}
```

### Error Chaining

```rust
fn complex_operation() -> Result<()> {
    step1()
        .context("Step 1 failed")?;

    step2()
        .context("Step 2 failed")?;

    step3()
        .context("Step 3 failed")?;

    Ok(())
}
```

### Conditional Error Handling

```rust
fn process_optional(value: Option<&Tensor>) -> Result<()> {
    let tensor = value.ok_or_else(|| {
        DemoError::InvalidConfig("Tensor is required".into())
    })?;

    process_tensor(tensor)
}
```

## Next Steps

- Learn about [Memory Management](memory-management.md)
- Study [Resource Management](resource-management.md)
- Explore [System Integration](system-integration.md)
