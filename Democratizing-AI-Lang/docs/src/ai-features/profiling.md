# Profiling

Democratising provides comprehensive profiling capabilities for analyzing performance and resource usage. This guide explains how to effectively profile your code and interpret the results.

## Basic Profiling

### CPU Profiling

```rust
use democratising::prelude::*;

fn profile_cpu() -> Result<()> {
    // Create CPU profiler
    let mut profiler = CpuProfiler::new()
        .sample_rate(1000)  // 1000 Hz
        .enable_call_graph(true)
        .build()?;

    // Profile section
    profiler.start()?;
    let result = compute_intensive_operation()?;
    let report = profiler.stop()?;

    // Print results
    println!("CPU Profile:");
    println!("Total time: {}ms", report.total_time_ms);
    println!("Hot functions:");
    for (func, stats) in report.function_stats {
        println!(
            "{}: {}ms ({}% of total time)",
            func.name,
            stats.total_time_ms,
            stats.percentage
        );
    }

    Ok(())
}
```

### Memory Profiling

```rust
fn profile_memory() -> Result<()> {
    // Create memory profiler
    let mut profiler = MemoryProfiler::new()
        .track_allocations(true)
        .track_peak_usage(true)
        .build()?;

    // Profile memory usage
    profiler.start()?;
    let result = memory_intensive_operation()?;
    let report = profiler.stop()?;

    // Analyze results
    println!("Memory Profile:");
    println!("Peak usage: {}MB", report.peak_mb);
    println!("Total allocations: {}", report.num_allocations);
    println!("Largest allocation: {}MB", report.largest_allocation_mb);

    Ok(())
}
```

## GPU Profiling

### CUDA Profiling

```rust
fn profile_gpu() -> Result<()> {
    // Create CUDA profiler
    let mut profiler = CudaProfiler::new()
        .track_memory(true)
        .track_kernels(true)
        .track_transfers(true)
        .build()?;

    // Profile GPU operations
    profiler.start()?;
    let gpu_tensor = tensor.to_device(&Device::cuda(0)?)?;
    let result = model.forward(&gpu_tensor)?;
    let report = profiler.stop()?;

    // Print kernel statistics
    println!("CUDA Kernel Profile:");
    for kernel in report.kernels {
        println!(
            "{}: {}us, {}% occupancy, {}MB memory",
            kernel.name,
            kernel.duration_us,
            kernel.occupancy * 100.0,
            kernel.memory_mb
        );
    }

    Ok(())
}
```

### Multi-GPU Analysis

```rust
fn profile_multi_gpu() -> Result<()> {
    // Create multi-GPU profiler
    let mut profiler = MultiGpuProfiler::new()
        .devices(Device::cuda_all()?)
        .synchronize(true)
        .build()?;

    // Profile distributed computation
    profiler.start()?;
    let result = distributed_training_step()?;
    let report = profiler.stop()?;

    // Analyze per-device metrics
    for (device, stats) in report.device_stats {
        println!("Device {}", device);
        println!("  Compute utilization: {}%", stats.compute_util);
        println!("  Memory utilization: {}%", stats.memory_util);
        println!("  P2P bandwidth: {}GB/s", stats.p2p_bandwidth_gbs);
    }

    Ok(())
}
```

## Advanced Profiling

### Trace Collection

```rust
fn collect_traces() -> Result<()> {
    // Configure tracer
    let mut tracer = Tracer::new()
        .sampling_rate(10000)  // 10kHz
        .buffer_size(1024 * 1024)  // 1MB buffer
        .build()?;

    // Record traces
    tracer.start()?;
    let result = model.forward(&input)?;
    let traces = tracer.stop()?;

    // Save traces
    traces.save("profile.json")?;

    // Analyze critical path
    let critical_path = traces.analyze_critical_path()?;
    println!("Critical path operations:");
    for op in critical_path {
        println!(
            "{}: {}us ({}% of total time)",
            op.name,
            op.duration_us,
            op.percentage
        );
    }

    Ok(())
}
```

### Custom Metrics

```rust
fn track_custom_metrics() -> Result<()> {
    // Create custom metric tracker
    let mut tracker = MetricTracker::new()
        .add_counter("matrix_multiplies")
        .add_histogram("activation_sparsity")
        .add_gauge("memory_usage")
        .build()?;

    // Track metrics during execution
    tracker.start()?;
    for batch in data_loader {
        // Update metrics
        tracker.increment("matrix_multiplies")?;
        tracker.record("activation_sparsity", compute_sparsity(&batch)?)?;
        tracker.set("memory_usage", cuda::get_memory_used(0)?)?;

        let output = model.forward(&batch)?;
    }
    let report = tracker.stop()?;

    // Generate statistics
    report.save_metrics("metrics.json")?;
    report.plot_metrics("metrics.png")?;

    Ok(())
}
```

## Performance Analysis

### Bottleneck Detection

```rust
fn analyze_bottlenecks() -> Result<()> {
    // Create analyzer
    let analyzer = PerformanceAnalyzer::new()
        .roofline_analysis(true)
        .memory_bandwidth_analysis(true)
        .build()?;

    // Collect performance data
    let data = analyzer.profile(|| {
        model.forward(&input)
    })?;

    // Identify bottlenecks
    let bottlenecks = data.find_bottlenecks()?;
    println!("Performance bottlenecks:");
    for bottleneck in bottlenecks {
        println!(
            "{}: {} bound ({}% of theoretical peak)",
            bottleneck.operation,
            bottleneck.limitation,
            bottleneck.utilization * 100.0
        );
    }

    Ok(())
}
```

### Optimization Recommendations

```rust
fn get_recommendations() -> Result<()> {
    // Create optimization advisor
    let advisor = OptimizationAdvisor::new()
        .target_device(Device::cuda(0)?)
        .target_batch_size(32)
        .build()?;

    // Analyze model
    let recommendations = advisor.analyze(&model)?;

    // Print recommendations
    println!("Optimization recommendations:");
    for rec in recommendations {
        println!("Priority {}: {}", rec.priority, rec.description);
        println!("Expected improvement: {}x", rec.expected_speedup);
        println!("Implementation:");
        for step in rec.steps {
            println!("- {}", step);
        }
    }

    Ok(())
}
```

## Best Practices

### Profiling Guidelines

1. Systematic profiling:
```rust
fn systematic_profiling() -> Result<()> {
    // Profile different aspects
    let compute_profile = profile_computation()?;
    let memory_profile = profile_memory_usage()?;
    let io_profile = profile_data_loading()?;

    // Combine insights
    let analysis = PerformanceAnalysis::new()
        .add_profile(compute_profile)
        .add_profile(memory_profile)
        .add_profile(io_profile)
        .analyze()?;

    // Generate report
    analysis.generate_report("profiling_report.html")?;

    Ok(())
}
```

2. Continuous profiling:
```rust
fn continuous_profiling() -> Result<()> {
    // Setup continuous profiler
    let profiler = ContinuousProfiler::new()
        .sampling_interval(Duration::from_secs(60))
        .metrics(vec!["cpu", "memory", "gpu"])
        .build()?;

    // Start profiling
    profiler.start()?;

    // Run long training job
    train_model()?;

    // Stop and analyze
    let profile_data = profiler.stop()?;
    analyze_performance_trends(&profile_data)?;

    Ok(())
}
```

### Profile Visualization

1. Timeline visualization:
```rust
fn visualize_timeline() -> Result<()> {
    // Create timeline
    let mut timeline = Timeline::new()
        .add_track("CPU")
        .add_track("GPU")
        .add_track("Memory")
        .build()?;

    // Record events
    timeline.record(|| {
        model.forward(&input)
    })?;

    // Generate visualization
    timeline.save_chrome_trace("timeline.json")?;
    timeline.save_flamegraph("flamegraph.svg")?;

    Ok(())
}
```

2. Interactive analysis:
```rust
fn interactive_analysis() -> Result<()> {
    // Create interactive profiler
    let profiler = InteractiveProfiler::new()
        .web_interface(8080)
        .real_time_updates(true)
        .build()?;

    // Start profiling server
    profiler.serve()?;

    // Profile with live updates
    profiler.record(|| {
        train_model()
    })?;

    Ok(())
}
```

## Next Steps

- Learn about [Benchmarking](benchmarking.md)
- Explore [Performance Optimization](optimization.md)
- Study [Hardware Integration](hardware.md)
