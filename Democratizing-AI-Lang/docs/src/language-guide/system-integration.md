# System Integration

Democratising provides robust integration capabilities with other programming languages, systems, and tools. This guide explains how to effectively integrate Democratising with various systems.

## Python Integration

### Calling Python from Democratising

```rust
use democratising::prelude::*;
use pyo3::prelude::*;

fn use_python_library() -> Result<()> {
    Python::with_gil(|py| {
        // Import numpy
        let numpy = py.import("numpy")?;

        // Create numpy array
        let array = numpy.call_method1("array", ([1.0, 2.0, 3.0],))?;

        // Convert to Democratising tensor
        let tensor = Tensor::from_numpy(array)?;

        Ok(())
    })
}
```

### Using Democratising from Python

```python
import democratising as demo

# Create tensor
tensor = demo.Tensor.randn([100, 100])

# Create model
model = demo.Sequential([
    demo.Dense(784, 128, activation='relu'),
    demo.Dense(128, 10)
])

# Train model
model.train(x_train, y_train, epochs=10)
```

## C/C++ Integration

### Foreign Function Interface (FFI)

```rust
// Expose Democratising functions to C
#[no_mangle]
pub extern "C" fn demo_create_tensor(
    data: *const f32,
    shape: *const i64,
    ndim: i32,
) -> *mut Tensor {
    // Implementation
}

// Call C functions from Democratising
extern "C" {
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}
```

### C++ Bindings

```cpp
// C++ header
class Tensor {
public:
    Tensor(const std::vector<float>& data, const std::vector<int64_t>& shape);
    void to_device(const Device& device);
    std::vector<float> to_vec();
private:
    // Implementation details
};

// Rust implementation
#[cxx::bridge]
mod ffi {
    extern "Rust" {
        type Tensor;

        fn new_tensor(data: &[f32], shape: &[i64]) -> Box<Tensor>;
        fn to_device(&mut self, device: Device);
        fn to_vec(&self) -> Vec<f32>;
    }
}
```

## CUDA Integration

### Custom CUDA Kernels

```rust
use cuda_runtime_sys as cuda;

#[cuda_kernel]
fn custom_kernel(input: CudaSlice<f32>, output: CudaMutSlice<f32>) {
    let idx = thread_idx_x() + block_idx_x() * block_dim_x();
    if idx < input.len() {
        output[idx] = input[idx].max(0.0);
    }
}

fn launch_kernel(input: &Tensor) -> Result<Tensor> {
    let output = Tensor::zeros_like(input)?;

    // Launch kernel
    unsafe {
        custom_kernel.launch(
            cuda::LaunchConfig::for_num_elems(input.numel() as u32),
            &input.cuda_slice()?,
            &output.cuda_slice_mut()?,
        )?;
    }

    Ok(output)
}
```

### cuBLAS Integration

```rust
use cublas_sys as cublas;

fn gpu_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let handle = cublas::create()?;

    // Set up matrices
    let (m, n, k) = get_matrix_dims(a, b)?;
    let c = Tensor::zeros(&[m, n])?;

    // Perform matrix multiplication
    unsafe {
        cublas::sgemm(
            handle,
            cublas::Operation::N,
            cublas::Operation::N,
            m as i32,
            n as i32,
            k as i32,
            &1.0,
            a.as_ptr(),
            m as i32,
            b.as_ptr(),
            k as i32,
            &0.0,
            c.as_mut_ptr(),
            m as i32,
        )?;
    }

    Ok(c)
}
```

## System APIs

### File System Operations

```rust
use std::fs;
use std::path::Path;

fn save_model(model: &Model, path: &Path) -> Result<()> {
    // Create directory if it doesn't exist
    fs::create_dir_all(path.parent().unwrap())?;

    // Serialize model
    let bytes = bincode::serialize(model)?;

    // Write to file
    fs::write(path, bytes)?;

    Ok(())
}

fn load_model(path: &Path) -> Result<Model> {
    // Read from file
    let bytes = fs::read(path)?;

    // Deserialize model
    let model = bincode::deserialize(&bytes)?;

    Ok(model)
}
```

### Network Operations

```rust
use tokio::net::{TcpListener, TcpStream};

async fn serve_model(model: Model, addr: &str) -> Result<()> {
    let listener = TcpListener::bind(addr).await?;

    while let Ok((socket, _)) = listener.accept().await {
        let model = model.clone();
        tokio::spawn(async move {
            handle_connection(socket, model).await
        });
    }

    Ok(())
}

async fn handle_connection(socket: TcpStream, model: Model) -> Result<()> {
    // Handle inference requests
    let (reader, writer) = socket.split();

    // Process requests
    while let Some(request) = read_request(reader).await? {
        let output = model.forward(&request)?;
        send_response(writer, &output).await?;
    }

    Ok(())
}
```

## Web Integration

### WebAssembly Support

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmModel {
    model: Model,
}

#[wasm_bindgen]
impl WasmModel {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmModel> {
        let model = Model::new()?;
        Ok(WasmModel { model })
    }

    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>> {
        let tensor = Tensor::from_slice(input)?;
        let output = self.model.forward(&tensor)?;
        Ok(output.to_vec()?)
    }
}
```

### REST API

```rust
use actix_web::{web, App, HttpServer, Result};

async fn predict(
    model: web::Data<Model>,
    input: web::Json<Vec<f32>>,
) -> Result<web::Json<Vec<f32>>> {
    let tensor = Tensor::from_slice(&input)?;
    let output = model.forward(&tensor)?;
    Ok(web::Json(output.to_vec()?))
}

#[actix_web::main]
async fn main() -> Result<()> {
    let model = Model::load("model.pt")?;

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(model.clone()))
            .route("/predict", web::post().to(predict))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

## Best Practices

### Memory Management

1. Handle cross-language memory:
```rust
// Properly manage Python GIL
Python::with_gil(|py| {
    // Python operations here
});

// Release CUDA memory
unsafe {
    cuda::free(ptr);
}
```

2. Use appropriate ownership:
```rust
// Transfer ownership to C++
Box::into_raw(tensor);

// Take ownership from C++
unsafe {
    Box::from_raw(ptr);
}
```

### Error Handling

1. Convert between error types:
```rust
impl From<PyErr> for DemoError {
    fn from(err: PyErr) -> Self {
        DemoError::PythonError(err.to_string())
    }
}
```

2. Proper error propagation:
```rust
fn integrated_operation() -> Result<()> {
    // Python operation
    let py_result = Python::with_gil(|py| {
        // Python code
    }).map_err(DemoError::from)?;

    // CUDA operation
    let cuda_result = unsafe {
        // CUDA code
    }.map_err(DemoError::from)?;

    Ok(())
}
```

## Next Steps

- Learn about [Memory Management](memory-management.md)
- Explore [Error Handling](error-handling.md)
- Study [Concurrency](concurrency.md)
