# Data Loading

Democratising provides efficient data loading capabilities for handling various data formats and sources. This guide explains how to effectively load and preprocess data for machine learning tasks.

## Basic Data Loading

### Loading Common Formats

```rust
use democratising::prelude::*;

fn load_data() -> Result<()> {
    // Load CSV data
    let csv_data = DataLoader::from_csv("data.csv")?
        .batch_size(32)
        .shuffle(true)
        .build()?;

    // Load image data
    let image_data = DataLoader::from_images("images/")?
        .batch_size(16)
        .transform(transforms::Compose::new()
            .add(transforms::Resize::new(224, 224))
            .add(transforms::Normalize::new(mean, std))
        )
        .build()?;

    // Load numpy arrays
    let numpy_data = DataLoader::from_numpy("data.npy")?
        .batch_size(64)
        .build()?;

    Ok(())
}
```

### Custom Dataset Implementation

```rust
struct CustomDataset {
    data: Vec<Tensor>,
    labels: Vec<i64>,
}

impl Dataset for CustomDataset {
    type Item = (Tensor, Tensor);

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        Some((
            self.data[index].clone(),
            self.labels[index].into_tensor()?,
        ))
    }
}

fn use_custom_dataset() -> Result<()> {
    let dataset = CustomDataset::new()?;
    let loader = DataLoader::new(dataset)?
        .batch_size(32)
        .shuffle(true)
        .num_workers(4)
        .build()?;

    for (x, y) in loader {
        train_batch(&x, &y)?;
    }

    Ok(())
}
```

## Advanced Features

### Parallel Data Loading

```rust
fn parallel_loading() -> Result<()> {
    // Configure parallel loader
    let loader = DataLoader::new(dataset)?
        .batch_size(32)
        .num_workers(4)
        .prefetch_factor(2)
        .pin_memory(true)
        .build()?;

    // Process data in parallel
    for batch in loader {
        let (x, y) = batch;

        // Data is automatically loaded in parallel
        process_batch(&x, &y)?;
    }

    Ok(())
}
```

### Streaming Data

```rust
struct StreamingDataset {
    reader: BufReader<File>,
    buffer: Vec<u8>,
}

impl StreamingDataset {
    fn new(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(1024 * 1024, file);

        Ok(StreamingDataset {
            reader,
            buffer: Vec::new(),
        })
    }
}

impl Iterator for StreamingDataset {
    type Item = Result<Tensor>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_next_chunk() {
            Ok(Some(data)) => Some(Ok(data)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
```

### Memory Mapped Data

```rust
fn mmap_dataset() -> Result<()> {
    // Create memory mapped dataset
    let dataset = MmapDataset::new("large_dataset.bin")?
        .chunk_size(1024)
        .cache_size(100);

    // Iterate over data without loading everything
    for chunk in dataset {
        process_chunk(&chunk)?;
    }

    Ok(())
}
```

## Data Preprocessing

### Transform Pipeline

```rust
fn create_transform_pipeline() -> Result<()> {
    // Create transform pipeline
    let transforms = transforms::Compose::new()
        // Image transforms
        .add(transforms::Resize::new(224, 224))
        .add(transforms::RandomCrop::new(200, 200))
        .add(transforms::RandomHorizontalFlip::new(0.5))

        // Tensor transforms
        .add(transforms::ToTensor::new())
        .add(transforms::Normalize::new(mean, std));

    // Apply transforms
    let loader = DataLoader::new(dataset)?
        .transform(transforms)
        .batch_size(32)
        .build()?;

    Ok(())
}
```

### Custom Transforms

```rust
struct CustomTransform {
    probability: f32,
}

impl Transform for CustomTransform {
    fn transform(&self, x: &Tensor) -> Result<Tensor> {
        if rand::random::<f32>() < self.probability {
            // Apply custom transformation
            x.custom_operation()?
        } else {
            Ok(x.clone())
        }
    }
}
```

## Performance Optimization

### Caching

```rust
fn use_caching() -> Result<()> {
    // Create caching dataset
    let dataset = CachingDataset::new(base_dataset)?
        .cache_size(1000)
        .cache_dir("cache/")
        .build()?;

    // First epoch: load and cache
    for batch in &dataset {
        process_batch(&batch)?;
    }

    // Subsequent epochs: use cache
    for batch in &dataset {
        // Data loaded from cache
        process_batch(&batch)?;
    }

    Ok(())
}
```

### Prefetching

```rust
fn optimize_loading() -> Result<()> {
    let loader = DataLoader::new(dataset)?
        .batch_size(32)
        .prefetch_factor(2)  // Prefetch 2 batches
        .num_workers(4)      // 4 worker threads
        .pin_memory(true)    // Pin memory for faster GPU transfer
        .build()?;

    for batch in loader {
        // Process current batch while next is being loaded
        process_batch(&batch)?;
    }

    Ok(())
}
```

## Best Practices

### Memory Management

1. Control batch sizes:
```rust
fn manage_memory() -> Result<()> {
    // Adjust batch size based on available memory
    let batch_size = if cuda::get_memory_info(0)?.free > 8 * 1024 * 1024 * 1024 {
        128 // Large batch size for high memory
    } else {
        32  // Small batch size for low memory
    };

    let loader = DataLoader::new(dataset)?
        .batch_size(batch_size)
        .build()?;

    Ok(())
}
```

2. Use streaming for large datasets:
```rust
fn handle_large_data() -> Result<()> {
    let loader = DataLoader::new(dataset)?
        .streaming(true)
        .chunk_size(1024 * 1024)  // 1MB chunks
        .build()?;

    for chunk in loader {
        process_chunk(&chunk)?;
    }

    Ok(())
}
```

### Error Handling

1. Handle corrupted data:
```rust
fn robust_loading() -> Result<()> {
    let loader = DataLoader::new(dataset)?
        .on_error(|e| {
            log::warn!("Error loading data: {}", e);
            None  // Skip corrupted items
        })
        .build()?;

    for batch in loader {
        match process_batch(&batch) {
            Ok(_) => (),
            Err(e) => log::error!("Error processing batch: {}", e),
        }
    }

    Ok(())
}
```

2. Implement retry logic:
```rust
fn retry_loading() -> Result<()> {
    let loader = DataLoader::new(dataset)?
        .retry_count(3)
        .retry_delay(Duration::from_secs(1))
        .build()?;

    for batch in loader {
        // Automatically retries on failure
        process_batch(&batch)?;
    }

    Ok(())
}
```

## Next Steps

- Learn about [Data Pipelines](pipelines.md)
- Explore [Training](../training.md)
- Study [Performance Optimization](../optimization.md)
