# Training

Democratising provides comprehensive training capabilities for machine learning models. This guide explains how to effectively train models using various techniques and optimizations.

## Basic Training

### Training Loop

```rust
use democratising::prelude::*;

fn train_model() -> Result<()> {
    // Create model
    let mut model = Sequential::new()
        .add(Dense::new(784, 128))
        .add(ReLU::new())
        .add(Dense::new(128, 10))
        .build()?;

    // Create optimizer and loss
    let mut optimizer = Adam::new(model.parameters(), 0.001)?;
    let loss_fn = CrossEntropyLoss::new();

    // Training loop
    for epoch in 0..num_epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (x, y) in train_loader {
            // Forward pass
            let predictions = model.forward(&x)?;
            let loss = loss_fn.forward(&predictions, &y)?;

            // Backward pass
            model.zero_grad();
            loss.backward()?;

            // Update weights
            optimizer.step()?;

            epoch_loss += loss.item();
            num_batches += 1;
        }

        println!(
            "Epoch {}/{}: Loss = {:.4}",
            epoch + 1,
            num_epochs,
            epoch_loss / num_batches as f32
        );
    }

    Ok(())
}
```

### Training Configuration

```rust
#[derive(Debug, Clone)]
struct TrainingConfig {
    learning_rate: f32,
    batch_size: usize,
    num_epochs: usize,
    device: Device,
    optimizer: OptimizerConfig,
    scheduler: Option<SchedulerConfig>,
}

fn configure_training() -> Result<()> {
    let config = TrainingConfig {
        learning_rate: 0.001,
        batch_size: 32,
        num_epochs: 100,
        device: Device::cuda(0)?,
        optimizer: OptimizerConfig::Adam {
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 1e-4,
        },
        scheduler: Some(SchedulerConfig::CosineAnnealing {
            T_max: 100,
            eta_min: 1e-6,
        }),
    };

    train_with_config(model, data, config)?;

    Ok(())
}
```

## Advanced Training

### Distributed Training

```rust
fn train_distributed() -> Result<()> {
    // Initialize distributed environment
    let world = World::init()?;
    let rank = world.rank();
    let world_size = world.size();

    // Create distributed model
    let model = DistributedModel::new(
        create_model()?,
        &world,
        DistributedStrategy::DataParallel,
    )?;

    // Create distributed data loader
    let loader = DistributedDataLoader::new(dataset)?
        .batch_size(32)
        .num_workers(4)
        .build()?;

    // Training loop with synchronization
    for epoch in 0..num_epochs {
        for batch in loader {
            // Forward and backward passes
            let loss = model.forward(&batch)?;
            model.backward(&loss)?;

            // Synchronize gradients across processes
            model.all_reduce_gradients(ReduceOp::Mean)?;

            // Update on all processes
            if rank == 0 {
                model.step()?;
            }
            model.broadcast_parameters(0)?;
        }
    }

    Ok(())
}
```

### Mixed Precision Training

```rust
fn train_with_amp() -> Result<()> {
    // Create AMP gradient scaler
    let scaler = GradScaler::new()
        .init_scale(2f32.powi(16))
        .growth_factor(2.0)
        .backoff_factor(0.5)
        .growth_interval(2000);

    for epoch in 0..num_epochs {
        for batch in train_loader {
            // Forward pass in FP16
            let output = model.forward_amp(&batch.x)?;
            let loss = loss_fn.forward_amp(&output, &batch.y)?;

            // Backward pass with scaling
            model.zero_grad();
            scaler.scale(loss)?.backward()?;

            // Optimizer step with unscaling
            scaler.step(&mut optimizer)?;
            scaler.update()?;
        }
    }

    Ok(())
}
```

### Gradient Accumulation

```rust
fn train_with_accumulation() -> Result<()> {
    let accumulation_steps = 4;  // Effective batch size = batch_size * accumulation_steps

    for epoch in 0..num_epochs {
        for (i, batch) in train_loader.enumerate() {
            // Forward and backward passes
            let loss = model.forward(&batch)?.mean()?;
            (loss / accumulation_steps as f32)?.backward()?;

            // Update weights only after accumulation
            if (i + 1) % accumulation_steps == 0 {
                optimizer.step()?;
                model.zero_grad();
            }
        }
    }

    Ok(())
}
```

## Training Features

### Learning Rate Scheduling

```rust
fn train_with_scheduler() -> Result<()> {
    // Create scheduler
    let scheduler = CosineAnnealingLR::new(
        &optimizer,
        num_epochs,
        1e-6,  // minimum learning rate
    )?;

    for epoch in 0..num_epochs {
        // Training loop
        train_epoch(&mut model, &mut optimizer)?;

        // Update learning rate
        scheduler.step()?;

        // Log current learning rate
        println!("LR: {}", scheduler.get_last_lr());
    }

    Ok(())
}
```

### Early Stopping

```rust
struct EarlyStopping {
    patience: usize,
    min_delta: f32,
    counter: usize,
    best_loss: f32,
    best_model_state: Option<ModelState>,
}

impl EarlyStopping {
    fn step(&mut self, model: &Model, loss: f32) -> Result<bool> {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.counter = 0;
            self.best_model_state = Some(model.state_dict()?);
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    fn restore_best_model(&self, model: &mut Model) -> Result<()> {
        if let Some(state) = &self.best_model_state {
            model.load_state_dict(state)?;
        }
        Ok(())
    }
}
```

### Model Checkpointing

```rust
fn save_checkpoint(
    model: &Model,
    optimizer: &Optimizer,
    epoch: usize,
    loss: f32,
    path: &Path,
) -> Result<()> {
    let checkpoint = Checkpoint {
        model_state: model.state_dict()?,
        optimizer_state: optimizer.state_dict()?,
        epoch,
        loss,
    };

    // Save atomically
    let temp_path = path.with_extension("tmp");
    checkpoint.save(&temp_path)?;
    fs::rename(&temp_path, path)?;

    Ok(())
}
```

## Best Practices

### Memory Management

1. Gradient cleanup:
```rust
fn clean_memory() -> Result<()> {
    // Clear gradients after each step
    model.zero_grad();

    // Free unused memory
    cuda::empty_cache()?;

    // Move model to CPU if needed
    if cuda::get_memory_info(0)?.free < threshold {
        model.to_device(&Device::cpu())?;
    }

    Ok(())
}
```

2. Memory-efficient training:
```rust
fn efficient_training() -> Result<()> {
    // Use gradient checkpointing
    model.enable_checkpointing()?;

    // Use mixed precision training
    let scaler = GradScaler::new();

    // Train with memory optimization
    for batch in train_loader {
        let loss = model.forward_amp(&batch)?;
        scaler.scale(loss)?.backward()?;
        scaler.step(&mut optimizer)?;
        scaler.update()?;
    }

    Ok(())
}
```

### Performance Optimization

1. Batch size optimization:
```rust
fn optimize_batch_size() -> Result<()> {
    // Start with small batch size
    let mut batch_size = 32;
    let mut best_throughput = 0.0;

    // Try increasing batch size
    while let Ok(throughput) = measure_throughput(batch_size) {
        if throughput > best_throughput {
            best_throughput = throughput;
            batch_size *= 2;
        } else {
            batch_size /= 2;
            break;
        }
    }

    Ok(())
}
```

2. Multi-GPU training:
```rust
fn optimize_multi_gpu() -> Result<()> {
    // Create data parallel model
    let model = DataParallel::new(model, Device::cuda_all()?)?;

    // Use synchronized batch norm
    model.convert_sync_batchnorm()?;

    // Train with multiple GPUs
    for batch in train_loader {
        // Automatically handles multi-GPU training
        let loss = model.forward(&batch)?;
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

## Next Steps

- Learn about [Evaluation](evaluation.md)
- Explore [Hyperparameter Tuning](hyperparameter-tuning.md)
- Study [Distributed Training](distributed.md)
