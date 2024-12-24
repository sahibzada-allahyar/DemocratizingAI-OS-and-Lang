# Model Deployment

Democratising provides comprehensive model deployment capabilities. This guide explains how to effectively deploy models to various environments and platforms.

## Basic Deployment

### Model Export

```rust
use democratising::prelude::*;

fn export_model(model: &Model, path: &Path) -> Result<()> {
    // Save model architecture and weights
    model.save(path)?;

    // Export to different formats
    model.export()
        .to_onnx("model.onnx")?
        .to_torchscript("model.pt")?
        .to_tensorflow("saved_model")?;

    Ok(())
}
```

### Model Loading

```rust
fn load_model() -> Result<()> {
    // Load from native format
    let model = Model::load("model.pt")?;

    // Load from ONNX
    let onnx_model = Model::from_onnx("model.onnx")?;

    // Load with specific device
    let gpu_model = Model::load("model.pt")?
        .to_device(&Device::cuda(0)?)?;

    Ok(())
}
```

## Serving Models

### REST API Server

```rust
use actix_web::{web, App, HttpServer, Result};

#[derive(Serialize, Deserialize)]
struct PredictRequest {
    inputs: Vec<f32>,
    shape: Vec<usize>,
}

#[derive(Serialize)]
struct PredictResponse {
    predictions: Vec<f32>,
    probabilities: Vec<f32>,
}

async fn predict(
    model: web::Data<Model>,
    request: web::Json<PredictRequest>,
) -> Result<web::Json<PredictResponse>> {
    // Create input tensor
    let input = Tensor::from_slice(&request.inputs)?
        .reshape(&request.shape)?;

    // Run inference
    let output = model.forward(&input)?;
    let probs = output.softmax(-1)?;

    Ok(web::Json(PredictResponse {
        predictions: output.to_vec()?,
        probabilities: probs.to_vec()?,
    }))
}

#[actix_web::main]
async fn main() -> Result<()> {
    // Load model
    let model = Model::load("model.pt")?;
    let model_data = web::Data::new(model);

    // Start server
    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .route("/predict", web::post().to(predict))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
```

### gRPC Service

```rust
#[derive(Clone)]
struct InferenceService {
    model: Arc<Model>,
}

#[tonic::async_trait]
impl Inference for InferenceService {
    async fn predict(
        &self,
        request: Request<PredictRequest>,
    ) -> Result<Response<PredictResponse>> {
        let input = request.into_inner();

        // Run inference
        let output = self.model.forward(&input.tensor)?;

        Ok(Response::new(PredictResponse {
            tensor: output.into_proto()?,
        }))
    }
}
```

## Optimization

### Model Optimization

```rust
fn optimize_model(model: &Model) -> Result<Model> {
    // Quantize model
    let quantized = model.quantize(QuantizationConfig {
        dtype: DType::Int8,
        calibration_data: get_calibration_data()?,
        algorithm: QuantizationAlgorithm::MinMaxObserver,
    })?;

    // Optimize for inference
    let optimized = quantized.optimize(OptimizationConfig {
        inference_only: true,
        fuse_operations: true,
        enable_tensorrt: true,
    })?;

    // Validate accuracy
    let accuracy = evaluate_model(&optimized, &test_data)?;
    println!("Optimized model accuracy: {:.2}%", accuracy * 100.0);

    Ok(optimized)
}
```

### Performance Tuning

```rust
fn tune_performance(model: &mut Model) -> Result<()> {
    // Profile model
    let profile = ModelProfiler::new()
        .profile_execution(model, &sample_input)?;

    // Auto-tune parameters
    model.tune(TuningConfig {
        batch_size: profile.optimal_batch_size()?,
        num_threads: num_cpus::get(),
        enable_cuda_graphs: true,
        workspace_size: 1024 * 1024 * 1024,  // 1GB
    })?;

    Ok(())
}
```

## Containerization

### Docker Deployment

```rust
fn create_docker_deployment() -> Result<()> {
    // Create deployment configuration
    let config = DockerConfig::new()
        .base_image("nvidia/cuda:11.8.0-runtime-ubuntu22.04")
        .expose_port(8080)
        .add_model("model.pt")
        .add_environment("NUM_THREADS", "4")
        .add_environment("CUDA_VISIBLE_DEVICES", "0")
        .build()?;

    // Generate Dockerfile
    config.generate_dockerfile("Dockerfile")?;

    // Build image
    config.build_image("my-model:latest")?;

    Ok(())
}
```

### Kubernetes Deployment

```rust
fn create_kubernetes_deployment() -> Result<()> {
    // Create Kubernetes configuration
    let config = KubernetesConfig::new()
        .name("model-service")
        .replicas(3)
        .image("my-model:latest")
        .resources(ResourceRequirements {
            requests: Resources {
                cpu: "2",
                memory: "4Gi",
                gpu: "1",
            },
            limits: Resources {
                cpu: "4",
                memory: "8Gi",
                gpu: "1",
            },
        })
        .build()?;

    // Generate Kubernetes manifests
    config.generate_manifests("k8s/")?;

    Ok(())
}
```

## Cloud Deployment

### Cloud Functions

```rust
#[function_framework::http]
fn predict_http(request: HttpRequest) -> Result<HttpResponse> {
    // Load model (cached between invocations)
    static MODEL: OnceCell<Model> = OnceCell::new();
    let model = MODEL.get_or_init(|| Model::load("model.pt").unwrap());

    // Parse request
    let input: Tensor = request.json()?;

    // Run inference
    let output = model.forward(&input)?;

    // Return response
    HttpResponse::Ok().json(output)
}
```

### Cloud Run

```rust
fn create_cloud_run_service() -> Result<()> {
    // Configure Cloud Run service
    let config = CloudRunConfig::new()
        .name("model-service")
        .memory("2Gi")
        .cpu(2)
        .min_instances(1)
        .max_instances(10)
        .concurrency(80)
        .build()?;

    // Deploy service
    config.deploy()?;

    Ok(())
}
```

## Best Practices

### Production Readiness

1. Health checks:
```rust
async fn healthcheck(model: web::Data<Model>) -> Result<HttpResponse> {
    // Check model health
    let status = model.check_health()?;

    // Check GPU status if using
    if let Some(gpu) = status.gpu_status {
        if gpu.memory_used > 0.95 * gpu.memory_total {
            return Ok(HttpResponse::ServiceUnavailable().finish());
        }
    }

    Ok(HttpResponse::Ok().finish())
}
```

2. Monitoring:
```rust
fn setup_monitoring() -> Result<()> {
    // Configure metrics
    let metrics = MetricsConfig::new()
        .add_counter("inference_requests_total")
        .add_histogram("inference_latency_ms")
        .add_gauge("gpu_memory_usage")
        .build()?;

    // Configure logging
    let logger = Logger::new()
        .json_format(true)
        .add_context("service", "model-inference")
        .build()?;

    Ok(())
}
```

### Error Handling

1. Graceful degradation:
```rust
fn handle_inference_errors(error: Error) -> Result<Tensor> {
    match error {
        // Handle GPU out of memory
        Error::CudaError(CudaError::OutOfMemory) => {
            // Try falling back to CPU
            model.to_device(&Device::cpu())?;
            model.forward(&input)
        }

        // Handle timeout
        Error::Timeout => {
            // Return cached result if available
            get_cached_result().ok_or(error)
        }

        // Propagate other errors
        _ => Err(error),
    }
}
```

2. Input validation:
```rust
fn validate_input(input: &Tensor) -> Result<()> {
    // Check shape
    if input.ndim() != 4 {
        return Err(Error::InvalidInput("Expected 4D input tensor"));
    }

    // Check values
    if input.min()? < 0.0 || input.max()? > 1.0 {
        return Err(Error::InvalidInput("Input values must be in [0, 1]"));
    }

    Ok(())
}
```

## Next Steps

- Learn about [Performance Optimization](optimization.md)
- Explore [Model Serving](serving.md)
- Study [Production Monitoring](monitoring.md)
