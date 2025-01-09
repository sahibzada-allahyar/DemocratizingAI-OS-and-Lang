# Resource Management

Democratising provides robust resource management capabilities to handle system resources efficiently. This guide explains how to manage memory, file handles, GPU resources, and other system resources.

## RAII Pattern

### Basic Resource Management

```rust
fn process_data() -> Result<()> {
    // File automatically closed when scope ends
    let file = File::open("data.txt")?;

    // Tensor memory automatically freed
    let tensor = Tensor::from_file(&file)?;

    // GPU memory automatically freed
    let gpu_tensor = tensor.to_device(&Device::cuda(0)?)?;

    Ok(())
} // All resources cleaned up here
```

### Custom Resource Types

```rust
struct ModelCheckpoint {
    file: File,
    path: PathBuf,
    temp_path: PathBuf,
}

impl Drop for ModelCheckpoint {
    fn drop(&mut self) {
        // Clean up temporary files
        if self.temp_path.exists() {
            let _ = fs::remove_file(&self.temp_path);
        }
    }
}
```

## Memory Management

### Tensor Memory

```rust
fn optimize_memory() -> Result<()> {
    // Stack allocation for small tensors
    let small_tensor = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0])?;

    // Heap allocation for large tensors
    let large_tensor = Tensor::zeros(&[1000, 1000])?;

    // Explicit cleanup
    drop(large_tensor);

    // Continue processing with small_tensor
    Ok(())
}
```

### Memory Pools

```rust
fn use_memory_pool() -> Result<()> {
    // Configure memory pool
    MemoryPool::configure()
        .initial_size(1024 * 1024)
        .growth_factor(2.0)
        .max_size(1024 * 1024 * 1024)
        .enable()?;

    // Tensors will use pooled memory
    for _ in 0..1000 {
        let tensor = Tensor::zeros(&[100, 100])?;
        process(&tensor)?;
    } // Memory returned to pool

    Ok(())
}
```

## GPU Resources

### CUDA Memory Management

```rust
fn manage_gpu_memory() -> Result<()> {
    // Configure CUDA memory pool
    cuda::MemoryPool::configure()
        .initial_size(1024 * 1024 * 1024)
        .enable()?;

    // Allocate on GPU
    let gpu_tensor = Tensor::zeros(&[1000, 1000], &Device::cuda(0)?)?;

    // Automatically freed when dropped
    drop(gpu_tensor);

    // Check memory usage
    let info = cuda::get_memory_info(0)?;
    println!("Free memory: {} bytes", info.free);

    Ok(())
}
```

### Multi-GPU Management

```rust
fn manage_multiple_gpus() -> Result<()> {
    let devices = Device::cuda_all()?;

    // Allocate across GPUs
    let mut tensors = Vec::new();
    for (i, device) in devices.iter().enumerate() {
        let tensor = Tensor::zeros(&[1000, 1000], device)?;
        tensors.push(tensor);
    }

    // Resources freed when tensors dropped
    Ok(())
}
```

## File System Resources

### File Handling

```rust
fn handle_files() -> Result<()> {
    // Open file with automatic cleanup
    let file = File::open("model.pt")?;

    // Create temporary file
    let temp_file = tempfile::NamedTempFile::new()?;

    // Write data safely
    {
        let mut writer = BufWriter::new(&temp_file);
        serialize_model(&mut writer)?;
    } // writer flushed and closed here

    // Rename atomically
    temp_file.persist("model.pt")?;

    Ok(())
}
```

### Directory Management

```rust
fn manage_directories() -> Result<()> {
    // Create directory structure
    let cache_dir = Path::new("cache");
    fs::create_dir_all(cache_dir)?;

    // Temporary directory
    let temp_dir = tempfile::Builder::new()
        .prefix("training-")
        .tempdir()?;

    // Use temporary directory
    let model_path = temp_dir.path().join("model.pt");
    save_checkpoint(&model_path)?;

    Ok(())
} // temp_dir automatically cleaned up
```

## Network Resources

### Connection Management

```rust
async fn handle_connections() -> Result<()> {
    // TCP listener with automatic cleanup
    let listener = TcpListener::bind("127.0.0.1:8080").await?;

    while let Ok((socket, _)) = listener.accept().await {
        tokio::spawn(async move {
            handle_client(socket).await
        });
    }

    Ok(())
}

async fn handle_client(socket: TcpStream) -> Result<()> {
    let (reader, writer) = socket.split();

    // Process client request
    process_request(reader, writer).await?;

    Ok(())
} // socket closed automatically
```

### Resource Pools

```rust
struct ConnectionPool {
    connections: Vec<TcpStream>,
    max_size: usize,
}

impl ConnectionPool {
    fn get_connection(&mut self) -> Result<PooledConnection> {
        if let Some(conn) = self.connections.pop() {
            Ok(PooledConnection {
                conn: Some(conn),
                pool: self,
            })
        } else {
            let conn = TcpStream::connect("server:port")?;
            Ok(PooledConnection {
                conn: Some(conn),
                pool: self,
            })
        }
    }
}

impl Drop for PooledConnection {
    fn drop(&mut self) {
        if let Some(conn) = self.conn.take() {
            self.pool.connections.push(conn);
        }
    }
}
```

## Best Practices

### Resource Cleanup

1. Use RAII pattern:
```rust
struct ResourceHandle {
    data: Vec<u8>,
    file: File,
}

impl Drop for ResourceHandle {
    fn drop(&mut self) {
        // Cleanup code here
    }
}
```

2. Explicit cleanup when needed:
```rust
fn process_large_data() -> Result<()> {
    let tensor = Tensor::zeros(&[10000, 10000])?;
    process(&tensor)?;
    drop(tensor); // Free memory early

    // Continue with other work
    Ok(())
}
```

### Resource Limits

1. Set memory limits:
```rust
fn configure_limits() -> Result<()> {
    // Set memory pool limits
    MemoryPool::configure()
        .max_size(8 * 1024 * 1024 * 1024)
        .enable()?;

    // Set CUDA memory limits
    cuda::set_memory_limit(0, 4 * 1024 * 1024 * 1024)?;

    Ok(())
}
```

2. Handle resource exhaustion:
```rust
fn handle_out_of_memory() -> Result<()> {
    match allocate_large_tensor() {
        Ok(tensor) => process(tensor),
        Err(DemoError::OutOfMemory) => {
            // Try to free some memory
            gc::collect()?;
            // Retry with smaller size
            allocate_smaller_tensor()
        }
        Err(e) => Err(e),
    }
}
```

### Performance Optimization

1. Reuse resources:
```rust
fn optimize_allocations() -> Result<()> {
    // Preallocate buffer
    let mut buffer = Vec::with_capacity(1000);

    for _ in 0..1000 {
        buffer.clear(); // Reuse allocation
        fill_buffer(&mut buffer)?;
        process(&buffer)?;
    }

    Ok(())
}
```

2. Use resource pools:
```rust
fn use_connection_pool() -> Result<()> {
    let pool = ConnectionPool::new(10);

    for _ in 0..100 {
        let conn = pool.get()?;
        process(&conn)?;
    } // Connections returned to pool

    Ok(())
}
```

## Next Steps

- Learn about [Memory Management](memory-management.md)
- Explore [Error Handling](error-handling.md)
- Study [System Integration](system-integration.md)
