# Weight Initializers

Democratising provides various weight initialization methods for neural networks, crucial for proper training and convergence.

## Basic Initializers

### Common Initialization Methods

```rust
use democratising::prelude::*;
use democratising::nn::init::*;

fn basic_initialization() -> Result<()> {
    // Zeros initialization
    let zeros = zeros(vec![100, 100])?;

    // Ones initialization
    let ones = ones(vec![100, 100])?;

    // Random uniform initialization
    let uniform = uniform(
        vec![100, 100],
        UniformConfig::new()
            .min(-0.1)
            .max(0.1)
            .build()?,
    )?;

    // Random normal initialization
    let normal = normal(
        vec![100, 100],
        NormalConfig::new()
            .mean(0.0)
            .std(0.01)
            .build()?,
    )?;

    Ok(())
}
```

### Xavier/Glorot Initialization

```rust
fn xavier_initialization() -> Result<()> {
    // Xavier uniform
    let weights = xavier_uniform(
        vec![784, 128],
        XavierConfig::new()
            .gain(1.0)
            .build()?,
    )?;

    // Xavier normal
    let weights = xavier_normal(
        vec![784, 128],
        XavierConfig::new()
            .gain(1.0)
            .build()?,
    )?;

    Ok(())
}
```

## Advanced Initializers

### Kaiming/He Initialization

```rust
fn kaiming_initialization() -> Result<()> {
    // Kaiming uniform
    let weights = kaiming_uniform(
        vec![784, 128],
        KaimingConfig::new()
            .mode(FanMode::FanIn)
            .nonlinearity(Nonlinearity::ReLU)
            .build()?,
    )?;

    // Kaiming normal
    let weights = kaiming_normal(
        vec![784, 128],
        KaimingConfig::new()
            .mode(FanMode::FanOut)
            .nonlinearity(Nonlinearity::LeakyReLU)
            .build()?,
    )?;

    Ok(())
}
```

### Orthogonal Initialization

```rust
fn orthogonal_initialization() -> Result<()> {
    // Basic orthogonal
    let weights = orthogonal(vec![100, 100])?;

    // With gain
    let weights = orthogonal_with_config(
        vec![100, 100],
        OrthogonalConfig::new()
            .gain(1.414)  // sqrt(2) for ReLU
            .build()?,
    )?;

    Ok(())
}
```

## Layer-Specific Initialization

### Dense Layer Initialization

```rust
fn dense_layer_init() -> Result<()> {
    // Initialize dense layer
    let layer = Dense::builder()
        .in_features(784)
        .out_features(128)
        .weight_init(xavier_uniform)
        .bias_init(zeros)
        .build()?;

    // Custom initialization
    let layer = Dense::builder()
        .in_features(784)
        .out_features(128)
        .weight_init(|shape| {
            let weights = kaiming_normal(shape, Default::default())?;
            weights * 0.1
        })
        .build()?;

    Ok(())
}
```

### Convolutional Layer Initialization

```rust
fn conv_layer_init() -> Result<()> {
    // Initialize conv layer
    let layer = Conv2d::builder()
        .in_channels(3)
        .out_channels(64)
        .kernel_size(3)
        .weight_init(kaiming_uniform)
        .bias_init(zeros)
        .build()?;

    // For different activation functions
    let layer = Conv2d::builder()
        .in_channels(3)
        .out_channels(64)
        .kernel_size(3)
        .weight_init(|shape| {
            kaiming_normal(
                shape,
                KaimingConfig::new()
                    .nonlinearity(Nonlinearity::LeakyReLU)
                    .build()?,
            )
        })
        .build()?;

    Ok(())
}
```

## Custom Initializers

### Creating Custom Initializer

```rust
#[derive(Debug)]
struct CustomInitializer {
    scale: f64,
    distribution: Distribution,
}

impl WeightInit for CustomInitializer {
    fn initialize(&self, shape: &[usize]) -> Result<Tensor<f64>> {
        let tensor = match self.distribution {
            Distribution::Uniform => {
                uniform(shape, UniformConfig::new().min(-1.0).max(1.0).build()?)?
            }
            Distribution::Normal => {
                normal(shape, NormalConfig::new().mean(0.0).std(1.0).build()?)?
            }
        };
        
        tensor * self.scale
    }
}

impl CustomInitializer {
    fn new(scale: f64, distribution: Distribution) -> Self {
        Self { scale, distribution }
    }
}
```

### Using Custom Initializer

```rust
fn use_custom_init() -> Result<()> {
    let initializer = CustomInitializer::new(0.01, Distribution::Normal);
    
    // Use with layer
    let layer = Dense::builder()
        .in_features(784)
        .out_features(128)
        .weight_init(|shape| initializer.initialize(shape))
        .build()?;

    // Use directly
    let weights = initializer.initialize(&[784, 128])?;

    Ok(())
}
```

## Best Practices

### Initialization Selection

1. Choose based on activation:
   ```rust
   let init = match activation {
       Activation::ReLU => kaiming_normal,
       Activation::Tanh => xavier_uniform,
       Activation::Sigmoid => xavier_normal,
       _ => uniform,
   };
   ```

2. Consider layer type:
   ```rust
   let init = match layer_type {
       LayerType::Conv => kaiming_uniform,
       LayerType::Linear => xavier_normal,
       LayerType::LSTM => orthogonal,
   };
   ```

### Scale Adjustment

1. Adjust for depth:
   ```rust
   let scale = 1.0 / (depth as f64).sqrt();
   let weights = init(shape)? * scale;
   ```

2. Layer-specific scaling:
   ```rust
   let weights = match layer {
       Layer::Output => init(shape)? * 0.01,  // Small for output
       Layer::Hidden => init(shape)?,         // Normal for hidden
       Layer::Embedding => init(shape)? * 0.1, // Scaled for embeddings
   };
   ```

### Numerical Stability

1. Check initialization:
   ```rust
   let weights = init(shape)?;
   assert!(!weights.has_nan()?);
   assert!(!weights.has_inf()?);
   ```

2. Monitor statistics:
   ```rust
   let mean = weights.mean()?;
   let std = weights.std(0)?;
   assert!(mean.abs()? < 1e-6);
   assert!((std - target_std).abs()? < 1e-6);
   ```

## Next Steps

- Learn about [Layers](layers.md)
- Explore [Optimizers](optimizers.md)
- Study [Neural Networks](../neural-networks.md)
- Understand [Training](../training.md)
