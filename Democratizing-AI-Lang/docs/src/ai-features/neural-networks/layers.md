# Neural Network Layers

Democratising provides a comprehensive set of neural network layers for building deep learning models, with support for custom layers and advanced features.

## Basic Layers

### Dense (Fully Connected) Layer

```rust
use democratising::prelude::*;
use democratising::nn::layers::*;

fn dense_layer() -> Result<()> {
    // Basic dense layer
    let layer = Dense::new(
        in_features: 784,
        out_features: 128,
        activation: Box::new(activations::relu),
    );

    // With configuration
    let layer = Dense::builder()
        .in_features(784)
        .out_features(128)
        .activation(activations::relu)
        .bias(true)
        .weight_init(init::xavier_uniform)
        .bias_init(init::zeros)
        .build()?;

    Ok(())
}
```

### Convolutional Layers

```rust
fn conv_layers() -> Result<()> {
    // 2D Convolution
    let conv2d = Conv2d::new(
        in_channels: 3,
        out_channels: 64,
        kernel_size: 3,
        stride: 1,
        padding: 1,
    );

    // Advanced configuration
    let conv2d = Conv2d::builder()
        .in_channels(3)
        .out_channels(64)
        .kernel_size(3)
        .stride(2)
        .padding(1)
        .dilation(1)
        .groups(1)
        .bias(true)
        .padding_mode(PaddingMode::Zeros)
        .build()?;

    // 3D Convolution
    let conv3d = Conv3d::new(
        in_channels: 3,
        out_channels: 64,
        kernel_size: (3, 3, 3),
        stride: (1, 1, 1),
        padding: (1, 1, 1),
    );

    Ok(())
}
```

### Recurrent Layers

```rust
fn recurrent_layers() -> Result<()> {
    // LSTM layer
    let lstm = LSTM::new(
        input_size: 256,
        hidden_size: 512,
        num_layers: 2,
        dropout: 0.1,
        bidirectional: true,
    );

    // GRU layer
    let gru = GRU::new(
        input_size: 256,
        hidden_size: 512,
        num_layers: 2,
    );

    // RNN with custom cell
    let rnn = RNN::new(
        cell: Box::new(CustomCell::new(256, 512)),
        num_layers: 2,
    );

    Ok(())
}
```

## Advanced Layers

### Attention Mechanisms

```rust
fn attention_layers() -> Result<()> {
    // Self-attention
    let self_attention = MultiHeadAttention::new(
        embed_dim: 512,
        num_heads: 8,
        dropout: 0.1,
    );

    // Transformer encoder layer
    let encoder_layer = TransformerEncoderLayer::new(
        d_model: 512,
        nhead: 8,
        dim_feedforward: 2048,
        dropout: 0.1,
        activation: Box::new(activations::gelu),
    );

    // Complete transformer
    let transformer = Transformer::new(
        d_model: 512,
        nhead: 8,
        num_encoder_layers: 6,
        num_decoder_layers: 6,
        dim_feedforward: 2048,
        dropout: 0.1,
    );

    Ok(())
}
```

### Normalization Layers

```rust
fn normalization_layers() -> Result<()> {
    // Batch normalization
    let batch_norm = BatchNorm2d::new(
        num_features: 64,
        eps: 1e-5,
        momentum: 0.1,
        affine: true,
        track_running_stats: true,
    );

    // Layer normalization
    let layer_norm = LayerNorm::new(
        normalized_shape: vec![512],
        eps: 1e-5,
        elementwise_affine: true,
    );

    // Instance normalization
    let instance_norm = InstanceNorm2d::new(
        num_features: 64,
        eps: 1e-5,
        momentum: 0.1,
        affine: true,
    );

    Ok(())
}
```

## Custom Layers

### Creating Custom Layer

```rust
#[derive(Debug)]
struct CustomLayer {
    weight: Variable<f64>,
    bias: Variable<f64>,
    activation: Box<dyn Activation>,
}

impl Layer for CustomLayer {
    type Input = Tensor<f64>;
    type Output = Tensor<f64>;

    fn forward(&self, input: &Self::Input) -> Result<Self::Output> {
        let output = input.matmul(&self.weight)?;
        let output = (output + &self.bias)?;
        self.activation.forward(&output)
    }

    fn parameters(&self) -> Vec<Variable<f64>> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

impl CustomLayer {
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        Ok(Self {
            weight: Variable::new(init::xavier_uniform(vec![in_features, out_features])?),
            bias: Variable::new(init::zeros(vec![out_features])?),
            activation: Box::new(activations::ReLU::new()),
        })
    }
}
```

### Using Custom Layer

```rust
fn use_custom_layer() -> Result<()> {
    let layer = CustomLayer::new(784, 128)?;
    
    // Use in sequential model
    let model = Sequential::new(vec![
        Box::new(layer),
        Box::new(Dense::new(128, 10, Box::new(activations::softmax))),
    ]);

    Ok(())
}
```

## Layer Composition

### Sequential Containers

```rust
fn sequential_composition() -> Result<()> {
    // Basic sequential
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(3, 64, 3, 1, 1)),
        Box::new(BatchNorm2d::new(64)),
        Box::new(activations::ReLU::new()),
        Box::new(MaxPool2d::new(2, 2)),
    ]);

    // With branching
    let model = Sequential::new(vec![
        Box::new(Conv2d::new(3, 64, 3, 1, 1)),
        Box::new(Branch::new(vec![
            Box::new(path1()),
            Box::new(path2()),
        ])),
        Box::new(Concat::new(1)), // Concatenate along channel dimension
    ]);

    Ok(())
}
```

### Residual Connections

```rust
fn residual_blocks() -> Result<()> {
    // Basic residual block
    let block = ResidualBlock::new(
        main_path: Box::new(Sequential::new(vec![
            Box::new(Conv2d::new(64, 64, 3, 1, 1)),
            Box::new(BatchNorm2d::new(64)),
            Box::new(activations::ReLU::new()),
            Box::new(Conv2d::new(64, 64, 3, 1, 1)),
            Box::new(BatchNorm2d::new(64)),
        ])),
        shortcut: None, // Identity shortcut
    );

    // With projection shortcut
    let block = ResidualBlock::new(
        main_path: Box::new(Sequential::new(vec![
            Box::new(Conv2d::new(64, 128, 3, 2, 1)),
            Box::new(BatchNorm2d::new(128)),
            Box::new(activations::ReLU::new()),
            Box::new(Conv2d::new(128, 128, 3, 1, 1)),
            Box::new(BatchNorm2d::new(128)),
        ])),
        shortcut: Some(Box::new(Sequential::new(vec![
            Box::new(Conv2d::new(64, 128, 1, 2, 0)),
            Box::new(BatchNorm2d::new(128)),
        ]))),
    );

    Ok(())
}
```

## Best Practices

### Layer Selection

1. Choose based on task:
   ```rust
   let layer = match task_type {
       TaskType::Vision => Conv2d::new(3, 64, 3, 1, 1),
       TaskType::NLP => TransformerEncoderLayer::new(512, 8, 2048, 0.1),
       TaskType::Audio => Conv1d::new(1, 64, 3, 1, 1),
   };
   ```

2. Configure properly:
   ```rust
   let config = LayerConfig::new()
       .initialization(init::xavier_uniform)
       .dropout(0.1)
       .build()?;
   ```

### Performance Tips

1. Use appropriate batch size:
   ```rust
   let batch_size = if cfg!(feature = "gpu") {
       32  // Larger for GPU
   } else {
       8   // Smaller for CPU
   };
   ```

2. Enable layer fusion:
   ```rust
   let layer = ConvBNReLU::new(  // Fused Conv+BN+ReLU
       in_channels: 3,
       out_channels: 64,
       kernel_size: 3,
   );
   ```

## Next Steps

- Learn about [Optimizers](optimizers.md)
- Explore [Loss Functions](loss-functions.md)
- Study [GPU Acceleration](../gpu.md)
- Understand [Automatic Differentiation](../autodiff.md)
