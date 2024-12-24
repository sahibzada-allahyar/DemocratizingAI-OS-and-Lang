# Serialization

Democratising provides robust serialization capabilities for saving and loading models, tensors, and other data structures. This guide explains how to effectively use these features.

## Basic Serialization

### Saving and Loading Tensors

```rust
use democratising::prelude::*;

fn save_tensor() -> Result<()> {
    // Create a tensor
    let tensor = Tensor::randn(&[100, 100])?;

    // Save to file
    tensor.save("tensor.pt")?;

    // Load from file
    let loaded = Tensor::load("tensor.pt")?;

    assert_eq!(tensor, loaded);
    Ok(())
}
```

### Model Serialization

```rust
fn save_model() -> Result<()> {
    // Create and train a model
    let model = Sequential::new()
        .add(Dense::new(784, 128))
        .add(Dense::new(128, 10))
        .build()?;

    // Save model
    model.save("model.pt")?;

    // Load model
    let loaded_model = Sequential::load("model.pt")?;

    Ok(())
}
```

## Advanced Features

### Checkpointing

```rust
struct Checkpoint {
    model_state: ModelState,
    optimizer_state: OptimizerState,
    epoch: usize,
    metrics: HashMap<String, f32>,
}

fn save_checkpoint(
    model: &Sequential,
    optimizer: &Adam,
    epoch: usize,
    metrics: HashMap<String, f32>,
    path: &Path,
) -> Result<()> {
    let checkpoint = Checkpoint {
        model_state: model.state_dict()?,
        optimizer_state: optimizer.state_dict()?,
        epoch,
        metrics,
    };

    // Save checkpoint
    checkpoint.save(path)?;

    Ok(())
}

fn load_checkpoint(path: &Path) -> Result<(Sequential, Adam, usize)> {
    // Load checkpoint
    let checkpoint = Checkpoint::load(path)?;

    // Create model and optimizer
    let mut model = create_model()?;
    let mut optimizer = create_optimizer(&model)?;

    // Restore state
    model.load_state_dict(&checkpoint.model_state)?;
    optimizer.load_state_dict(&checkpoint.optimizer_state)?;

    Ok((model, optimizer, checkpoint.epoch))
}
```

### Format Conversion

```rust
fn export_model(model: &Sequential) -> Result<()> {
    // Export to ONNX format
    model.to_onnx("model.onnx")?;

    // Export to TorchScript
    model.to_torchscript("model.pt")?;

    // Export to TensorFlow SavedModel
    model.to_tensorflow("model")?;

    Ok(())
}
```

## Memory Mapping

### Memory-Mapped Tensors

```rust
fn use_mmap() -> Result<()> {
    // Create memory-mapped tensor
    let tensor = Tensor::mmap("large_tensor.bin")?;

    // Access data without loading into memory
    let slice = tensor.slice(0..1000)?;
    process_slice(&slice)?;

    Ok(())
}
```

### Streaming Data

```rust
fn stream_dataset() -> Result<()> {
    // Create streaming dataset
    let dataset = StreamingDataset::new("data.bin")?
        .batch_size(32)
        .shuffle(1000)
        .prefetch(2);

    // Process in chunks without loading entire dataset
    for batch in dataset {
        process_batch(&batch)?;
    }

    Ok(())
}
```

## Custom Serialization

### Custom Serializable Types

```rust
#[derive(Serialize, Deserialize)]
struct CustomModel {
    layers: Vec<Box<dyn Layer>>,
    config: ModelConfig,
}

impl CustomModel {
    fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Serialize model structure
        bincode::serialize_into(&mut writer, &self.config)?;

        // Serialize layers
        for layer in &self.layers {
            layer.serialize(&mut writer)?;
        }

        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Deserialize model structure
        let config: ModelConfig = bincode::deserialize_from(&mut reader)?;

        // Deserialize layers
        let mut layers = Vec::new();
        while let Ok(layer) = Layer::deserialize(&mut reader) {
            layers.push(layer);
        }

        Ok(CustomModel { layers, config })
    }
}
```

### Versioned Serialization

```rust
#[derive(Serialize, Deserialize)]
struct VersionedModel {
    version: u32,
    #[serde(with = "model_serializer")]
    model: Sequential,
}

impl VersionedModel {
    fn save(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        // Save with version information
        serde_json::to_writer(writer, self)?;

        Ok(())
    }

    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        // Load and handle version differences
        let versioned: VersionedModel = serde_json::from_reader(reader)?;
        versioned.migrate()?;

        Ok(versioned)
    }

    fn migrate(&mut self) -> Result<()> {
        match self.version {
            1 => self.migrate_v1_to_v2()?,
            2 => (), // Current version
            _ => return Err(Error::UnsupportedVersion(self.version)),
        }
        Ok(())
    }
}
```

## Best Practices

### Error Handling

1. Handle corrupted files:
```rust
fn safe_load(path: &Path) -> Result<Model> {
    // Verify file integrity
    if !verify_checksum(path)? {
        return Err(Error::CorruptedFile(path.to_path_buf()));
    }

    // Load with proper error handling
    match Model::load(path) {
        Ok(model) => Ok(model),
        Err(e) => {
            // Try loading backup
            let backup_path = path.with_extension("backup");
            if backup_path.exists() {
                Model::load(&backup_path)
            } else {
                Err(e)
            }
        }
    }
}
```

2. Atomic saves:
```rust
fn atomic_save(model: &Model, path: &Path) -> Result<()> {
    // Create temporary file
    let temp_path = path.with_extension("tmp");
    model.save(&temp_path)?;

    // Verify save was successful
    if !verify_save(&temp_path)? {
        fs::remove_file(&temp_path)?;
        return Err(Error::SaveVerificationFailed);
    }

    // Atomically rename
    fs::rename(&temp_path, path)?;

    Ok(())
}
```

### Performance Optimization

1. Use appropriate formats:
```rust
fn optimize_storage() -> Result<()> {
    // For small models: JSON for readability
    if model.size() < 1_000_000 {
        model.save_json("small_model.json")?;
    } else {
        // For large models: Binary for efficiency
        model.save_binary("large_model.bin")?;
    }

    Ok(())
}
```

2. Compress large models:
```rust
fn save_compressed(model: &Model) -> Result<()> {
    // Create compressed writer
    let file = File::create("model.gz")?;
    let encoder = GzEncoder::new(file, Compression::default());
    let mut writer = BufWriter::new(encoder);

    // Serialize model
    model.serialize(&mut writer)?;

    Ok(())
}
```

## Next Steps

- Learn about [Model Deployment](deployment.md)
- Explore [Data Loading](data/loading.md)
- Study [Model Evaluation](evaluation.md)
