# Loss Functions

Democratising provides a comprehensive set of loss functions for training neural networks, with support for custom loss functions and advanced features.

## Basic Loss Functions

### Mean Squared Error (MSE)

```rust
use democratising::prelude::*;
use democratising::nn::losses::*;

fn mse_example() -> Result<()> {
    let predictions = Tensor::random_normal(vec![32, 10])?;
    let targets = Tensor::random_normal(vec![32, 10])?;
    
    // Compute MSE loss
    let loss = mse(&predictions, &targets)?;
    println!("MSE Loss: {}", loss);
    
    // With reduction options
    let loss = mse_with_config(
        &predictions,
        &targets,
        MSEConfig::new()
            .reduction(Reduction::Mean)
            .build()?,
    )?;

    Ok(())
}
```

### Cross Entropy

```rust
fn cross_entropy_example() -> Result<()> {
    // For multi-class classification
    let logits = model.forward(&inputs)?;
    let loss = cross_entropy(
        &logits,
        &targets,
        CrossEntropyConfig::new()
            .label_smoothing(0.1)
            .class_weights(Some(weights))
            .build()?,
    )?;

    // Binary cross entropy
    let binary_loss = binary_cross_entropy(
        &predictions,
        &targets,
        BCEConfig::new()
            .with_logits(true)
            .pos_weight(Some(pos_weight))
            .build()?,
    )?;

    Ok(())
}
```

## Advanced Loss Functions

### Focal Loss

```rust
fn focal_loss_example() -> Result<()> {
    let loss = focal_loss(
        &predictions,
        &targets,
        FocalLossConfig::new()
            .alpha(0.25)
            .gamma(2.0)
            .reduction(Reduction::Mean)
            .build()?,
    )?;

    // For object detection
    let loss = focal_loss_with_boxes(
        &predictions,
        &targets,
        &anchors,
        FocalLossConfig::new()
            .alpha(0.25)
            .gamma(2.0)
            .build()?,
    )?;

    Ok(())
}
```

### Contrastive Loss

```rust
fn contrastive_loss_example() -> Result<()> {
    let loss = contrastive_loss(
        &embeddings1,
        &embeddings2,
        &labels,
        ContrastiveLossConfig::new()
            .margin(1.0)
            .reduction(Reduction::Mean)
            .build()?,
    )?;

    // For self-supervised learning
    let loss = simclr_loss(
        &features,
        SimCLRConfig::new()
            .temperature(0.07)
            .build()?,
    )?;

    Ok(())
}
```

## Custom Loss Functions

### Creating Custom Loss

```rust
#[derive(Debug)]
struct CustomLoss {
    weight: Option<Tensor<f64>>,
    reduction: Reduction,
}

impl Loss for CustomLoss {
    fn forward(&self, input: &Tensor<f64>, target: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Custom loss computation
        let diff = (input - target)?;
        let squared = (diff.clone() * diff)?;
        
        match self.reduction {
            Reduction::None => Ok(squared),
            Reduction::Mean => squared.mean(),
            Reduction::Sum => squared.sum(),
        }
    }

    fn backward(&self, grad_output: &Tensor<f64>) -> Result<Tensor<f64>> {
        // Custom gradient computation
        // ...
        Ok(grad)
    }
}
```

### Using Custom Loss

```rust
fn use_custom_loss() -> Result<()> {
    let criterion = CustomLoss {
        weight: None,
        reduction: Reduction::Mean,
    };

    // Training loop
    for (inputs, targets) in data_loader.iter() {
        let outputs = model.forward(&inputs)?;
        let loss = criterion.forward(&outputs, &targets)?;
        
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

## Multi-Task Learning

### Combined Losses

```rust
fn multi_task_loss() -> Result<()> {
    // Create multiple loss functions
    let classification_loss = cross_entropy;
    let regression_loss = mse;
    
    // Combine losses with weights
    let outputs = model.forward(&inputs)?;
    let (class_out, reg_out) = outputs.split(1)?;
    
    let class_loss = classification_loss(&class_out, &class_targets)?;
    let reg_loss = regression_loss(&reg_out, &reg_targets)?;
    
    // Weighted combination
    let total_loss = (class_loss * 0.7 + reg_loss * 0.3)?;

    Ok(())
}
```

### Dynamic Weighting

```rust
fn dynamic_loss_weighting() -> Result<()> {
    // Implement uncertainty weighting
    let loss_weights = Variable::new(Tensor::ones(2));
    
    for epoch in 0..num_epochs {
        let class_loss = classification_loss(&class_out, &class_targets)?;
        let reg_loss = regression_loss(&reg_out, &reg_targets)?;
        
        // Update weights based on task uncertainty
        let weighted_loss = (
            class_loss / (2.0 * loss_weights[0].exp()) +
            reg_loss / (2.0 * loss_weights[1].exp()) +
            loss_weights.sum()
        )?;
        
        weighted_loss.backward()?;
    }

    Ok(())
}
```

## Advanced Features

### Loss Scheduling

```rust
fn loss_scheduling() -> Result<()> {
    let scheduler = LossScheduler::new(
        criterion,
        LossSchedulerConfig::new()
            .warmup_epochs(5)
            .schedule(Schedule::Linear)
            .build()?,
    )?;

    for epoch in 0..num_epochs {
        let loss = scheduler.compute_loss(&outputs, &targets)?;
        scheduler.step()?;
    }

    Ok(())
}
```

### Regularization

```rust
fn regularization() -> Result<()> {
    // L1 regularization
    let l1_loss = l1_regularization(
        model.parameters(),
        RegConfig::new()
            .weight(0.01)
            .build()?,
    )?;

    // L2 regularization
    let l2_loss = l2_regularization(
        model.parameters(),
        RegConfig::new()
            .weight(0.01)
            .build()?,
    )?;

    // Combined loss
    let total_loss = (task_loss + l1_loss + l2_loss)?;

    Ok(())
}
```

## Best Practices

### Loss Selection

1. Choose based on task:
   ```rust
   let criterion = match task {
       Task::Classification => cross_entropy,
       Task::Regression => mse,
       Task::Detection => focal_loss,
   };
   ```

2. Configure properly:
   ```rust
   let config = LossConfig::new()
       .reduction(Reduction::Mean)
       .weight(Some(class_weights))
       .build()?;
   ```

### Numerical Stability

1. Use stable variants:
   ```rust
   // Instead of regular cross entropy
   let loss = cross_entropy_with_logits(&logits, &targets)?;
   ```

2. Handle edge cases:
   ```rust
   let loss = if loss.is_nan()? {
       Error::numerical_error("Loss is NaN")
   } else if loss.is_infinite()? {
       Error::numerical_error("Loss is infinite")
   } else {
       Ok(loss)
   }?;
   ```

## Next Steps

- Learn about [Optimizers](optimizers.md)
- Explore [Neural Networks](../neural-networks.md)
- Study [Automatic Differentiation](../autodiff.md)
- Understand [GPU Acceleration](../gpu.md)
