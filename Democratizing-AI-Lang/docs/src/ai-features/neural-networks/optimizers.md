# Optimizers

Democratising provides a variety of optimizers for training neural networks, with support for advanced optimization techniques and customization.

## Basic Optimizers

### Stochastic Gradient Descent (SGD)

```rust
use democratising::prelude::*;
use democratising::nn::optimizers::*;

fn sgd_example() -> Result<()> {
    // Basic SGD
    let optimizer = SGD::new(
        model.parameters(),
        SGDConfig::new()
            .learning_rate(0.01)
            .momentum(0.9)
            .build()?,
    )?;

    // Training loop
    for (inputs, targets) in data_loader.iter() {
        let outputs = model.forward(&inputs)?;
        let loss = criterion(&outputs, &targets)?;
        
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

### Adam

```rust
fn adam_example() -> Result<()> {
    // Configure Adam optimizer
    let optimizer = Adam::new(
        model.parameters(),
        AdamConfig::new()
            .learning_rate(0.001)
            .betas((0.9, 0.999))
            .epsilon(1e-8)
            .weight_decay(0.0)
            .amsgrad(false)
            .build()?,
    )?;

    // Training with Adam
    for epoch in 0..num_epochs {
        for (inputs, targets) in data_loader.iter() {
            let outputs = model.forward(&inputs)?;
            let loss = criterion(&outputs, &targets)?;
            
            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;
        }
    }

    Ok(())
}
```

## Advanced Optimizers

### AdamW

```rust
fn adamw_example() -> Result<()> {
    let optimizer = AdamW::new(
        model.parameters(),
        AdamWConfig::new()
            .learning_rate(0.001)
            .betas((0.9, 0.999))
            .epsilon(1e-8)
            .weight_decay(0.01)
            .build()?,
    )?;

    // Training with decoupled weight decay
    for batch in data_loader.iter() {
        let loss = compute_loss(model, batch)?;
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

### RMSprop

```rust
fn rmsprop_example() -> Result<()> {
    let optimizer = RMSprop::new(
        model.parameters(),
        RMSpropConfig::new()
            .learning_rate(0.001)
            .alpha(0.99)
            .epsilon(1e-8)
            .momentum(0.0)
            .centered(false)
            .build()?,
    )?;

    // Training loop
    for batch in data_loader.iter() {
        let loss = compute_loss(model, batch)?;
        optimizer.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

## Learning Rate Scheduling

### Step Decay

```rust
fn step_decay() -> Result<()> {
    let scheduler = StepLR::new(
        optimizer,
        StepLRConfig::new()
            .step_size(30)
            .gamma(0.1)
            .build()?,
    )?;

    for epoch in 0..num_epochs {
        train_epoch(model, data_loader, optimizer)?;
        scheduler.step()?;
        
        println!("Epoch {}: lr = {}", epoch, scheduler.get_lr());
    }

    Ok(())
}
```

### Cosine Annealing

```rust
fn cosine_annealing() -> Result<()> {
    let scheduler = CosineAnnealingLR::new(
        optimizer,
        CosineConfig::new()
            .t_max(100)
            .eta_min(1e-6)
            .build()?,
    )?;

    for epoch in 0..num_epochs {
        train_epoch(model, data_loader, optimizer)?;
        scheduler.step()?;
    }

    Ok(())
}
```

## Custom Optimizers

### Creating a Custom Optimizer

```rust
#[derive(Debug)]
struct CustomOptimizer {
    params: Vec<Variable>,
    lr: f64,
    momentum: f64,
    velocity: HashMap<usize, Tensor>,
}

impl Optimizer for CustomOptimizer {
    fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter_mut().enumerate() {
            if let Some(grad) = param.gradient() {
                // Update velocity
                let v = self.velocity.entry(i).or_insert_with(|| {
                    Tensor::zeros_like(&grad)
                });
                
                *v = (v * self.momentum + grad * (1.0 - self.momentum))?;
                
                // Update parameter
                *param -= (v * self.lr)?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.clear_grad();
        }
    }
}
```

### Using Custom Optimizer

```rust
fn use_custom_optimizer() -> Result<()> {
    let optimizer = CustomOptimizer {
        params: model.parameters(),
        lr: 0.01,
        momentum: 0.9,
        velocity: HashMap::new(),
    };

    // Training loop
    for epoch in 0..num_epochs {
        for batch in data_loader.iter() {
            let loss = compute_loss(model, batch)?;
            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;
        }
    }

    Ok(())
}
```

## Advanced Features

### Parameter Groups

```rust
fn parameter_groups() -> Result<()> {
    // Create parameter groups with different settings
    let groups = vec![
        ParamGroup::new()
            .params(model.layer(0).parameters())
            .learning_rate(0.01)
            .weight_decay(0.001)
            .build()?,
        ParamGroup::new()
            .params(model.layer(1).parameters())
            .learning_rate(0.001)
            .weight_decay(0.0)
            .build()?,
    ];

    let optimizer = Adam::new_with_groups(groups)?;

    Ok(())
}
```

### Gradient Clipping

```rust
fn gradient_clipping() -> Result<()> {
    // Clip gradients by norm
    optimizer.clip_grad_norm(1.0)?;
    
    // Clip gradients by value
    optimizer.clip_grad_value(0.5)?;
    
    // Custom clipping
    for param in model.parameters() {
        if let Some(grad) = param.gradient() {
            let clipped = grad.clamp(-1.0, 1.0)?;
            param.set_grad(clipped)?;
        }
    }

    Ok(())
}
```

## Best Practices

### Optimizer Selection

1. Choose based on task:
   ```rust
   let optimizer = match task_type {
       TaskType::Vision => Adam::new(params, AdamConfig::default())?,
       TaskType::NLP => AdamW::new(params, AdamWConfig::default())?,
       TaskType::RL => RMSprop::new(params, RMSpropConfig::default())?,
   };
   ```

2. Configure hyperparameters:
   ```rust
   let config = OptimizerConfig::new()
       .learning_rate(lr_finder.suggest()?)
       .weight_decay(1e-4)
       .build()?;
   ```

### Performance Tips

1. Use AMP (Automatic Mixed Precision):
   ```rust
   let scaler = GradScaler::new();
   
   // In training loop
   let loss = scaler.scale(compute_loss()?)?;
   loss.backward()?;
   scaler.step(&mut optimizer)?;
   scaler.update()?;
   ```

2. Optimize memory usage:
   ```rust
   optimizer.set_grad_cache(true);  // Cache gradients
   optimizer.set_foreach(true);     // Use vectorized operations
   ```

## Next Steps

- Learn about [Loss Functions](loss-functions.md)
- Explore [Neural Networks](../neural-networks.md)
- Study [GPU Acceleration](../gpu.md)
- Understand [Automatic Differentiation](../autodiff.md)
