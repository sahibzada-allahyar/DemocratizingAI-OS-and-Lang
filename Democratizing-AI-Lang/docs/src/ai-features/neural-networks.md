# Neural Networks

Democratising provides comprehensive support for neural network development. This guide explains how to create, train, and deploy neural networks effectively.

## Basic Neural Network

### Creating a Simple Network

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Create a simple feedforward network
    let model = Sequential::new()
        .add(Dense::new(784, 128).with_activation(activation::relu))
        .add(Dense::new(128, 64).with_activation(activation::relu))
        .add(Dense::new(64, 10).with_activation(activation::softmax))
        .build()?;

    // Initialize optimizer and loss function
    let optimizer = Adam::new(model.parameters(), 0.001)?;
    let loss_fn = CrossEntropyLoss::new();

    Ok(())
}
```

### Training Loop

```rust
fn train_model(
    model: &mut Sequential,
    data: &DataLoader,
    optimizer: &mut Adam,
    loss_fn: &CrossEntropyLoss,
    epochs: usize,
) -> Result<()> {
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (x, y) in data {
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
            epochs,
            epoch_loss / num_batches as f32
        );
    }

    Ok(())
}
```

## Network Architectures

### Convolutional Neural Network

```rust
fn create_cnn() -> Result<Sequential> {
    Sequential::new()
        // First convolutional block
        .add(Conv2d::new(3, 32, 3)
            .with_padding(1)
            .with_activation(activation::relu))
        .add(MaxPool2d::new(2))
        .add(BatchNorm2d::new(32))

        // Second convolutional block
        .add(Conv2d::new(32, 64, 3)
            .with_padding(1)
            .with_activation(activation::relu))
        .add(MaxPool2d::new(2))
        .add(BatchNorm2d::new(64))

        // Dense layers
        .add(Flatten::new())
        .add(Dense::new(64 * 7 * 7, 512).with_activation(activation::relu))
        .add(Dropout::new(0.5))
        .add(Dense::new(512, 10).with_activation(activation::softmax))
        .build()
}
```

### Recurrent Neural Network

```rust
fn create_rnn() -> Result<Sequential> {
    Sequential::new()
        // Embedding layer
        .add(Embedding::new(vocab_size, 256))

        // LSTM layers
        .add(LSTM::new(256, 128)
            .with_bidirectional(true)
            .with_num_layers(2)
            .with_dropout(0.2))

        // Output layer
        .add(Dense::new(256, num_classes).with_activation(activation::softmax))
        .build()
}
```

### Transformer

```rust
fn create_transformer() -> Result<Sequential> {
    Sequential::new()
        // Embedding layers
        .add(Embedding::new(vocab_size, d_model))
        .add(PositionalEncoding::new(d_model, max_len))

        // Transformer blocks
        .add(TransformerEncoder::new(
            num_layers = 6,
            d_model = 512,
            nhead = 8,
            dim_feedforward = 2048,
            dropout = 0.1,
        ))

        // Output layer
        .add(Dense::new(d_model, num_classes))
        .build()
}
```

## Custom Layers

### Creating a Custom Layer

```rust
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    conv2: Conv2d,
    bn1: BatchNorm2d,
    bn2: BatchNorm2d,
}

impl Layer for ResidualBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();

        let out = self.conv1.forward(x)?
            .apply(&self.bn1)?
            .relu()?
            .apply(&self.conv2)?
            .apply(&self.bn2)?;

        (&out + &identity)?.relu()
    }

    fn backward(&self, grad: &Tensor) -> Result<Tensor> {
        // Implement backward pass
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.bn2.parameters());
        params
    }
}
```

### Using Custom Layers

```rust
fn create_resnet() -> Result<Sequential> {
    Sequential::new()
        .add(Conv2d::new(3, 64, 7).with_stride(2))
        .add(MaxPool2d::new(3).with_stride(2))
        .add(ResidualBlock::new(64, 64))
        .add(ResidualBlock::new(64, 128).with_stride(2))
        .add(ResidualBlock::new(128, 256).with_stride(2))
        .add(AdaptiveAvgPool2d::new(1))
        .add(Flatten::new())
        .add(Dense::new(256, num_classes))
        .build()
}
```

## Training Features

### Learning Rate Scheduling

```rust
fn train_with_scheduler(
    model: &mut Sequential,
    optimizer: &mut Adam,
    scheduler: &mut LRScheduler,
) -> Result<()> {
    for epoch in 0..num_epochs {
        train_epoch(model, optimizer)?;

        // Update learning rate
        let new_lr = scheduler.step(epoch, validation_loss)?;
        optimizer.set_learning_rate(new_lr)?;
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
}

impl EarlyStopping {
    fn step(&mut self, loss: f32) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }
}
```

### Model Checkpointing

```rust
fn save_checkpoint(
    model: &Sequential,
    optimizer: &Adam,
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

    checkpoint.save(path)
}
```

## GPU Acceleration

### Moving to GPU

```rust
fn train_on_gpu() -> Result<()> {
    let device = Device::cuda(0)?;

    // Create model on GPU
    let model = create_model()?.to_device(&device)?;

    // Move data to GPU
    let (x, y) = load_data()?;
    let x_gpu = x.to_device(&device)?;
    let y_gpu = y.to_device(&device)?;

    // Train on GPU
    train_model(&mut model, &x_gpu, &y_gpu)?;

    Ok(())
}
```

### Multi-GPU Training

```rust
fn train_distributed() -> Result<()> {
    let devices = Device::cuda_all()?;

    // Create model parallel wrapper
    let model = DistributedModel::new(
        create_model()?,
        &devices,
        DataParallel::new(),
    )?;

    // Train with multiple GPUs
    train_model(&mut model, &data)?;

    Ok(())
}
```

## Model Deployment

### Model Export

```rust
fn export_model(model: &Sequential, path: &Path) -> Result<()> {
    // Save model architecture and weights
    model.save(path)?;

    // Export to ONNX format
    model.to_onnx("model.onnx")?;

    // Export to TorchScript
    model.to_torchscript("model.pt")?;

    Ok(())
}
```

### Inference Server

```rust
#[tokio::main]
async fn main() -> Result<()> {
    // Load model
    let model = Sequential::load("model.pt")?;

    // Start inference server
    let server = InferenceServer::new()
        .with_model(model)
        .with_batch_size(32)
        .with_max_queue_size(100)
        .build()?;

    server.run("0.0.0.0:8080").await
}
```

## Best Practices

### Memory Management

1. Use gradient accumulation for large models:
```rust
fn train_with_accumulation(
    model: &mut Sequential,
    data: &DataLoader,
    accumulation_steps: usize,
) -> Result<()> {
    for (i, (x, y)) in data.enumerate() {
        let loss = model.forward(&x)?.cross_entropy_loss(&y)?;
        (loss / accumulation_steps as f32)?.backward()?;

        if (i + 1) % accumulation_steps == 0 {
            optimizer.step()?;
            model.zero_grad();
        }
    }

    Ok(())
}
```

2. Free memory when possible:
```rust
fn train_large_model() -> Result<()> {
    for epoch in 0..num_epochs {
        train_epoch(model, data)?;

        // Free unused memory
        cuda::empty_cache()?;

        validate_epoch(model, val_data)?;
    }

    Ok(())
}
```

### Performance Optimization

1. Use appropriate batch sizes:
```rust
fn optimize_batch_size() -> Result<()> {
    // Start with small batch size
    let mut batch_size = 32;

    // Gradually increase while monitoring memory
    while let Ok(_) = train_batch(batch_size) {
        batch_size *= 2;
    }

    // Use largest successful batch size
    batch_size /= 2;

    Ok(())
}
```

2. Enable AMP (Automatic Mixed Precision):
```rust
fn train_with_amp() -> Result<()> {
    let scaler = GradScaler::new();

    for (x, y) in data {
        // Forward pass in FP16
        let output = model.forward_amp(&x)?;
        let loss = loss_fn.forward_amp(&output, &y)?;

        // Backward pass with scaling
        scaler.scale(loss)?.backward()?;

        // Update with unscaling
        scaler.step(&mut optimizer)?;
        scaler.update()?;
    }

    Ok(())
}
```

## Next Steps

- Learn about [Automatic Differentiation](autodiff.md)
- Explore [GPU Acceleration](gpu.md)
- Study [Distributed Training](distributed.md)
