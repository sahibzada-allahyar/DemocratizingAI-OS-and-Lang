# Benchmarking

Democratising provides comprehensive benchmarking capabilities for measuring and comparing performance. This guide explains how to effectively benchmark your code and analyze results.

## Basic Benchmarking

### Performance Benchmarks

```rust
use democratising::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_tensor_ops(c: &mut Criterion) {
    let x = Tensor::randn(&[1000, 1000])?;
    let y = Tensor::randn(&[1000, 1000])?;

    c.bench_function("matrix multiply", |b| {
        b.iter(|| {
            black_box(&x).matmul(black_box(&y))
        })
    });

    c.bench_function("element-wise add", |b| {
        b.iter(|| {
            black_box(&x + &y)
        })
    });
}

criterion_group!(benches, benchmark_tensor_ops);
criterion_main!(benches);
```

### Memory Benchmarks

```rust
fn benchmark_memory() -> Result<()> {
    let mut bench = MemoryBenchmark::new()
        .warmup_iterations(5)
        .measure_iterations(20)
        .build()?;

    // Benchmark memory allocation
    let stats = bench.measure(|| {
        let tensor = Tensor::zeros(&[1000, 1000])?;
        process_tensor(&tensor)
    })?;

    println!("Memory Statistics:");
    println!("Peak usage: {}MB", stats.peak_mb);
    println!("Average usage: {}MB", stats.average_mb);
    println!("Allocation rate: {}/sec", stats.allocations_per_sec);

    Ok(())
}
```

## Model Benchmarking

### Training Benchmarks

```rust
fn benchmark_training() -> Result<()> {
    // Configure benchmark
    let bench = TrainingBenchmark::new()
        .batch_sizes(vec![32, 64, 128])
        .num_epochs(3)
        .build()?;

    // Run benchmark
    let results = bench.run(|config| {
        let mut model = create_model()?;
        let mut optimizer = Adam::new(model.parameters(), 0.001)?;

        // Training loop
        for epoch in 0..config.num_epochs {
            for batch in data.batch(config.batch_size) {
                let loss = model.forward(&batch.x)?;
                loss.backward()?;
                optimizer.step()?;
            }
        }

        Ok(())
    })?;

    // Print results
    for result in results {
        println!(
            "Batch size {}: {:.2} samples/sec",
            result.batch_size,
            result.throughput
        );
    }

    Ok(())
}
```

### Inference Benchmarks

```rust
fn benchmark_inference() -> Result<()> {
    // Create benchmark suite
    let mut suite = InferenceBenchmark::new()
        .model(model)
        .input_shapes(vec![[1, 3, 224, 224], [8, 3, 224, 224]])
        .precision(vec![DType::Float32, DType::Float16])
        .devices(vec![Device::cpu(), Device::cuda(0)?])
        .build()?;

    // Run benchmarks
    let results = suite.run()?;

    // Analyze results
    for result in results {
        println!("Configuration:");
        println!("  Input shape: {:?}", result.input_shape);
        println!("  Precision: {:?}", result.precision);
        println!("  Device: {:?}", result.device);
        println!("Performance:");
        println!("  Latency: {:.2}ms", result.latency_ms);
        println!("  Throughput: {:.2} inferences/sec", result.throughput);
        println!("  Memory: {:.2}MB", result.memory_mb);
    }

    Ok(())
}
```

## Hardware Benchmarks

### GPU Benchmarks

```rust
fn benchmark_gpu() -> Result<()> {
    // Configure GPU benchmark
    let bench = GpuBenchmark::new()
        .operations(vec![
            GpuOp::MatMul,
            GpuOp::Conv2d,
            GpuOp::BatchNorm,
        ])
        .tensor_sizes(vec![[1000, 1000], [2000, 2000]])
        .build()?;

    // Run benchmarks
    let results = bench.run()?;

    // Print results
    println!("GPU Performance:");
    for result in results {
        println!(
            "{} ({:?}): {:.2} TFLOPS, {:.2}GB/s bandwidth",
            result.operation,
            result.size,
            result.tflops,
            result.bandwidth_gbs
        );
    }

    Ok(())
}
```

### Multi-Device Benchmarks

```rust
fn benchmark_multi_device() -> Result<()> {
    // Create multi-device benchmark
    let bench = MultiDeviceBenchmark::new()
        .devices(Device::cuda_all()?)
        .communication_patterns(vec![
            CommPattern::AllReduce,
            CommPattern::Broadcast,
            CommPattern::P2P,
        ])
        .build()?;

    // Run communication benchmarks
    let results = bench.run()?;

    // Analyze results
    for result in results {
        println!("Communication Pattern: {}", result.pattern);
        println!("  Latency: {:.2}us", result.latency_us);
        println!("  Bandwidth: {:.2}GB/s", result.bandwidth_gbs);
        println!("  Scaling efficiency: {:.2}%", result.scaling_efficiency);
    }

    Ok(())
}
```

## Comparative Benchmarking

### Framework Comparison

```rust
fn compare_frameworks() -> Result<()> {
    // Setup comparison benchmark
    let bench = FrameworkBenchmark::new()
        .add_framework(Framework::Democratising)
        .add_framework(Framework::PyTorch)
        .add_framework(Framework::TensorFlow)
        .model(ResNet50::new())
        .dataset(ImageNet::new())
        .metrics(vec!["throughput", "memory", "accuracy"])
        .build()?;

    // Run comparison
    let results = bench.run()?;

    // Generate comparison report
    results.generate_report("framework_comparison.html")?;

    Ok(())
}
```

### Version Comparison

```rust
fn compare_versions() -> Result<()> {
    // Configure version benchmark
    let bench = VersionBenchmark::new()
        .versions(vec!["0.1.0", "0.2.0", "main"])
        .benchmarks(vec![
            "tensor_ops",
            "model_inference",
            "training_loop",
        ])
        .build()?;

    // Run benchmarks
    let results = bench.run()?;

    // Analyze performance changes
    for change in results.significant_changes() {
        println!(
            "{}: {:.2}x {} in version {}",
            change.benchmark,
            change.factor,
            if change.is_improvement { "faster" } else { "slower" },
            change.version
        );
    }

    Ok(())
}
```

## Best Practices

### Benchmark Design

1. Reliable measurements:
```rust
fn reliable_benchmark() -> Result<()> {
    let bench = Benchmark::new()
        // Warm up system
        .warmup_iterations(10)
        // Multiple iterations for stability
        .measure_iterations(100)
        // Statistical analysis
        .confidence_interval(0.95)
        .build()?;

    // Run benchmark
    let stats = bench.run(|| {
        // Benchmark implementation
    })?;

    // Report results with confidence intervals
    println!(
        "Time: {:.2} ± {:.2}ms (95% CI)",
        stats.mean_ms,
        stats.confidence_interval_ms
    );

    Ok(())
}
```

2. Representative workloads:
```rust
fn representative_benchmark() -> Result<()> {
    // Use realistic data sizes
    let bench = ModelBenchmark::new()
        .batch_sizes(vec![1, 8, 32])  // Including inference and training
        .input_sizes(vec![[224, 224], [320, 320]])  // Common image sizes
        .data_types(vec![DType::Float32, DType::Float16])
        .build()?;

    // Include real-world scenarios
    bench.run_scenario("batch_inference")?;
    bench.run_scenario("online_inference")?;
    bench.run_scenario("training")?;

    Ok(())
}
```

### Results Analysis

1. Comprehensive metrics:
```rust
fn analyze_results() -> Result<()> {
    let analysis = BenchmarkAnalysis::new()
        // Performance metrics
        .add_metric(Metric::Latency)
        .add_metric(Metric::Throughput)
        // Resource usage
        .add_metric(Metric::MemoryUsage)
        .add_metric(Metric::GpuUtilization)
        // Quality metrics
        .add_metric(Metric::Accuracy)
        .build()?;

    // Generate detailed report
    analysis.generate_report("benchmark_results.pdf")?;

    Ok(())
}
```

2. Visualization:
```rust
fn visualize_results() -> Result<()> {
    let viz = BenchmarkVisualization::new()
        // Different plot types
        .add_plot(Plot::Histogram("latency"))
        .add_plot(Plot::Timeline("memory"))
        .add_plot(Plot::Heatmap("gpu_utilization"))
        // Interactive dashboard
        .interactive(true)
        .build()?;

    // Save visualizations
    viz.save("benchmark_viz.html")?;

    Ok(())
}
```

## Next Steps

- Learn about [Profiling](profiling.md)
- Explore [Performance Optimization](optimization.md)
- Study [Hardware Integration](hardware.md)
