# GPU Acceleration

Democratising provides comprehensive GPU acceleration support through CUDA and other GPU backends. This guide explains how to effectively use GPU acceleration for your AI workloads.

## Basic Usage

### Device Management

```rust
use democratising::prelude::*;

fn main() -> Result<()> {
    // Get available devices
    let devices = Device::available()?;
    println!("Available devices: {:?}", devices);

    // Select specific GPU
    let gpu = Device::cuda(0)?;
    println!("Using GPU: {}", gpu);

    // Check device capabilities
    let caps = gpu.capabilities()?;
    println!("CUDA compute capability: {}.{}", caps.major, caps.minor);

    Ok(())
}
```

### Moving Data to GPU

```rust
fn use_gpu() -> Result<()> {
    // Create tensor on CPU
    let cpu_tensor = Tensor::randn(&[1000, 1000])?;

    // Move to GPU
    let gpu_tensor = cpu_tensor.to_device(&Device::cuda(0)?)?;

    // Perform computation on GPU
    let result = gpu_tensor.matmul(&gpu_tensor)?;

    // Move result back to CPU if needed
    let cpu_result = result.to_device(&Device::cpu())?;

    Ok(())
}
```

## Memory Management

### CUDA Memory Allocation

```rust
fn manage_gpu_memory() -> Result<()> {
    // Configure memory pool
    cuda::MemoryPool::configure()
        .initial_size(1024 * 1024 * 1024) // 1GB
        .growth_factor(2.0)
        .max_size(8 * 1024 * 1024 * 1024) // 8GB
        .enable()?;

    // Allocate tensor using pool
    let tensor = Tensor::zeros(&[1000, 1000], &Device::cuda(0)?)?;

    // Get memory info
    let info = cuda::get_memory_info(0)?;
    println!(
        "Free memory: {} MB, Total memory: {} MB",
        info.free / 1024 / 1024,
        info.total / 1024 / 1024
    );

    Ok(())
}
```

### Memory Efficiency

```rust
fn optimize_memory() -> Result<()> {
    // Use in-place operations where possible
    let mut x = Tensor::randn(&[1000, 1000], &Device::cuda(0)?)?;
    x.mul_inplace(&2.0)?; // Avoid creating new tensor

    // Free unused memory
    cuda::empty_cache()?;

    // Use streams for overlapping operations
    let stream = cuda::Stream::new()?;
    stream.run(|| {
        // Asynchronous operations
        x.matmul(&x)?;
        Ok(())
    })?;

    Ok(())
}
```

## Neural Networks on GPU

### Training on GPU

```rust
fn train_on_gpu() -> Result<()> {
    let device = Device::cuda(0)?;

    // Create model on GPU
    let model = Sequential::new()
        .add(Dense::new(784, 128))
        .add(Dense::new(128, 10))
        .build()?
        .to_device(&device)?;

    // Create optimizer
    let optimizer = Adam::new(model.parameters(), 0.001)?;

    // Training loop
    for (x, y) in data {
        // Move batch to GPU
        let x_gpu = x.to_device(&device)?;
        let y_gpu = y.to_device(&device)?;

        // Forward pass
        let output = model.forward(&x_gpu)?;
        let loss = cross_entropy_loss(&output, &y_gpu)?;

        // Backward pass
        model.zero_grad();
        loss.backward()?;
        optimizer.step()?;
    }

    Ok(())
}
```

### Multi-GPU Training

```rust
fn train_distributed() -> Result<()> {
    // Get all available GPUs
    let devices = Device::cuda_all()?;

    // Create data parallel model
    let model = DistributedModel::new(
        create_model()?,
        &devices,
        DataParallel::new(),
    )?;

    // Training with automatic batch splitting
    for (x, y) in data {
        let loss = model.forward(&x)?;
        loss.backward()?;
        model.step()?;
    }

    Ok(())
}
```

## Advanced Features

### Custom CUDA Kernels

```rust
#[cuda_kernel]
fn custom_activation(input: CudaSlice<f32>, output: CudaMutSlice<f32>) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < input.len() {
        // Custom activation function
        output[idx] = if input[idx] > 0.0 {
            input[idx]
        } else {
            input[idx] * 0.01
        };
    }
}

fn apply_custom_kernel(x: &Tensor) -> Result<Tensor> {
    let output = Tensor::zeros_like(x)?;

    unsafe {
        custom_activation.launch(
            cuda::LaunchConfig::for_num_elems(x.numel() as u32),
            &x.cuda_slice()?,
            &output.cuda_slice_mut()?,
        )?;
    }

    Ok(output)
}
```

### CUDA Streams

```rust
fn use_streams() -> Result<()> {
    let stream1 = cuda::Stream::new()?;
    let stream2 = cuda::Stream::new()?;

    // Parallel execution on different streams
    stream1.run(|| {
        let x = Tensor::randn(&[1000, 1000], &Device::cuda(0)?)?;
        x.matmul(&x)
    })?;

    stream2.run(|| {
        let y = Tensor::randn(&[1000, 1000], &Device::cuda(0)?)?;
        y.matmul(&y)
    })?;

    // Synchronize when needed
    stream1.synchronize()?;
    stream2.synchronize()?;

    Ok(())
}
```

## Performance Optimization

### Memory Transfer Optimization

1. Minimize CPU-GPU transfers:
```rust
fn optimize_transfers() -> Result<()> {
    let device = Device::cuda(0)?;

    // Bad: Frequent transfers
    for x in data {
        let x_gpu = x.to_device(&device)?;
        process_on_gpu(&x_gpu)?;
    }

    // Good: Batch transfers
    let data_gpu = data.to_device(&device)?;
    for x in data_gpu.chunks(batch_size)? {
        process_on_gpu(&x)?;
    }

    Ok(())
}
```

2. Use pinned memory:
```rust
fn use_pinned_memory() -> Result<()> {
    // Allocate pinned memory for faster transfers
    let pinned_buffer = cuda::PinnedBuffer::new(size)?;

    // Copy data to pinned buffer
    pinned_buffer.copy_from_slice(&host_data)?;

    // Fast transfer to GPU
    let gpu_tensor = Tensor::from_pinned(&pinned_buffer, &Device::cuda(0)?)?;

    Ok(())
}
```

### Computation Optimization

1. Use appropriate batch sizes:
```rust
fn optimize_batch_size() -> Result<()> {
    // Profile different batch sizes
    let batch_sizes = [32, 64, 128, 256];
    let mut best_throughput = 0.0;
    let mut optimal_batch_size = 32;

    for &size in &batch_sizes {
        let throughput = benchmark_batch_size(size)?;
        if throughput > best_throughput {
            best_throughput = throughput;
            optimal_batch_size = size;
        }
    }

    println!("Optimal batch size: {}", optimal_batch_size);
    Ok(())
}
```

2. Enable tensor cores:
```rust
fn use_tensor_cores() -> Result<()> {
    // Configure for tensor core usage
    cuda::set_math_mode(cuda::MathMode::TensorCore)?;

    // Create tensors with appropriate type and alignment
    let a = Tensor::randn(&[8192, 8192], &Device::cuda(0)?)?
        .to_dtype(DType::F16)?;
    let b = Tensor::randn(&[8192, 8192], &Device::cuda(0)?)?
        .to_dtype(DType::F16)?;

    // Matrix multiplication using tensor cores
    let c = a.matmul(&b)?;

    Ok(())
}
```

## Best Practices

### Memory Management

1. Monitor memory usage:
```rust
fn track_memory() -> Result<()> {
    // Get initial memory state
    let start_mem = cuda::get_memory_info(0)?;

    // Run computation
    let result = compute_intensive_operation()?;

    // Check memory after
    let end_mem = cuda::get_memory_info(0)?;
    println!(
        "Memory used: {} MB",
        (end_mem.free - start_mem.free) / 1024 / 1024
    );

    Ok(())
}
```

2. Handle out-of-memory:
```rust
fn handle_oom() -> Result<()> {
    match large_computation() {
        Ok(result) => Ok(result),
        Err(DemoError::CudaError(CudaError::OutOfMemory)) => {
            // Free cache and retry
            cuda::empty_cache()?;
            large_computation()
        }
        Err(e) => Err(e),
    }
}
```

## Next Steps

- Learn about [Distributed Training](distributed.md)
- Explore [Performance Optimization](optimization.md)
- Study [Hardware Integration](hardware.md)
