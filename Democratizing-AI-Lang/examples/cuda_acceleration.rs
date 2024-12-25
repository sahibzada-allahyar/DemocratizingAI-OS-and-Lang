use anyhow::Result;
use clap::Parser;
use democratising::{cuda::CudaDevice, tensor::Tensor, Device};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use plotters::{
    prelude::*,
    style::full_palette::{BLUE, GREEN, RED},
};
use std::{path::PathBuf, time::Instant};

/// CUDA acceleration example demonstrating GPU-accelerated tensor operations
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Matrix size (N x N)
    #[clap(long, default_value = "4096")]
    size: usize,

    /// Number of iterations for benchmarking
    #[clap(long, default_value = "100")]
    iterations: usize,

    /// Output directory for plots
    #[clap(long, default_value = "output/cuda")]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Check CUDA availability
    let num_gpus = CudaDevice::count()?;
    if num_gpus == 0 {
        println!("No CUDA devices found. Please ensure CUDA is installed and enabled.");
        std::process::exit(1);
    }
    println!("Found {} CUDA device(s)", num_gpus);

    // Print device information
    for gpu_id in 0..num_gpus {
        let device = CudaDevice::new(gpu_id)?;
        println!("\nGPU {}: {}", gpu_id, device.name()?);
        println!(
            "  Compute capability: {}.{}",
            device.major()?,
            device.minor()?
        );
        println!(
            "  Total memory: {:.1} GB",
            device.total_memory()? as f64 / 1e9
        );
        println!(
            "  Free memory: {:.1} GB",
            device.free_memory()? as f64 / 1e9
        );
    }

    // Run benchmarks
    let (cpu_times, gpu_times) = run_benchmarks(&args)?;

    // Plot results
    plot_results(&cpu_times, &gpu_times, &args.output_dir)?;

    // Print summary
    print_summary(&cpu_times, &gpu_times);

    Ok(())
}

fn run_benchmarks(args: &Args) -> Result<(Vec<f32>, Vec<f32>)> {
    let multi_progress = MultiProgress::new();
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();

    // Create tensors
    println!(
        "\nGenerating random matrices of size {}x{}",
        args.size, args.size
    );
    let a = Tensor::randn(&[args.size, args.size])?;
    let b = Tensor::randn(&[args.size, args.size])?;

    // CPU benchmarks
    println!("\nRunning CPU benchmarks...");
    let cpu_progress = multi_progress.add(ProgressBar::new(args.iterations as u64));
    cpu_progress.set_style(progress_style.clone());
    let mut cpu_times = Vec::with_capacity(args.iterations);

    for _ in 0..args.iterations {
        let start = Instant::now();
        let _c = &a * &b?;
        cpu_times.push(start.elapsed().as_secs_f32() * 1000.0);
        cpu_progress.inc(1);
        cpu_progress.set_message(format!("Time: {:.2} ms", cpu_times.last().unwrap()));
    }

    cpu_progress.finish_with_message("CPU benchmarks complete");

    // Move tensors to GPU
    println!("\nMoving tensors to GPU...");
    let a_gpu = a.to_device(Device::GPU(0))?;
    let b_gpu = b.to_device(Device::GPU(0))?;

    // GPU benchmarks
    println!("\nRunning GPU benchmarks...");
    let gpu_progress = multi_progress.add(ProgressBar::new(args.iterations as u64));
    gpu_progress.set_style(progress_style);
    let mut gpu_times = Vec::with_capacity(args.iterations);

    for _ in 0..args.iterations {
        let start = Instant::now();
        let _c = &a_gpu * &b_gpu?;
        gpu_times.push(start.elapsed().as_secs_f32() * 1000.0);
        gpu_progress.inc(1);
        gpu_progress.set_message(format!("Time: {:.2} ms", gpu_times.last().unwrap()));
    }

    gpu_progress.finish_with_message("GPU benchmarks complete");

    Ok((cpu_times, gpu_times))
}

fn plot_results(cpu_times: &[f32], gpu_times: &[f32], output_dir: &PathBuf) -> Result<()> {
    let path = output_dir.join("benchmark.png");
    let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_time = cpu_times
        .iter()
        .chain(gpu_times.iter())
        .fold(0f32, |a, &b| a.max(b));

    let mut chart = ChartBuilder::on(&root)
        .caption("CPU vs GPU Performance", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(0f32..cpu_times.len() as f32, 0f32..max_time)?;

    chart
        .configure_mesh()
        .x_desc("Iteration")
        .y_desc("Time (ms)")
        .draw()?;

    // Plot CPU times
    chart.draw_series(LineSeries::new(
        cpu_times.iter().enumerate().map(|(i, &v)| (i as f32, v)),
        RED.mix(0.5),
    ))?;

    // Plot GPU times
    chart.draw_series(LineSeries::new(
        gpu_times.iter().enumerate().map(|(i, &v)| (i as f32, v)),
        BLUE.mix(0.5),
    ))?;

    // Add legend
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    Ok(())
}

fn print_summary(cpu_times: &[f32], gpu_times: &[f32]) {
    // Calculate statistics
    let cpu_mean = cpu_times.iter().sum::<f32>() / cpu_times.len() as f32;
    let gpu_mean = gpu_times.iter().sum::<f32>() / gpu_times.len() as f32;

    let cpu_std = (cpu_times
        .iter()
        .map(|&x| (x - cpu_mean).powi(2))
        .sum::<f32>()
        / cpu_times.len() as f32)
        .sqrt();
    let gpu_std = (gpu_times
        .iter()
        .map(|&x| (x - gpu_mean).powi(2))
        .sum::<f32>()
        / gpu_times.len() as f32)
        .sqrt();

    let speedup = cpu_mean / gpu_mean;

    println!("\nPerformance Summary:");
    println!("  CPU: {:.2} ± {:.2} ms", cpu_mean, cpu_std);
    println!("  GPU: {:.2} ± {:.2} ms", gpu_mean, gpu_std);
    println!("  Speedup: {:.1}x", speedup);

    // Memory bandwidth calculation
    let bytes_per_element = std::mem::size_of::<f32>();
    let matrix_size = (cpu_times[0].sqrt() as usize).pow(2);
    let bytes_processed = matrix_size * matrix_size * bytes_per_element * 3; // Read A, B, write C
    let bandwidth_cpu = bytes_processed as f64 / (cpu_mean as f64 * 1e-3) / 1e9;
    let bandwidth_gpu = bytes_processed as f64 / (gpu_mean as f64 * 1e-3) / 1e9;

    println!("\nMemory Bandwidth:");
    println!("  CPU: {:.1} GB/s", bandwidth_cpu);
    println!("  GPU: {:.1} GB/s", bandwidth_gpu);

    // FLOPS calculation
    let flops_per_element = 2; // One multiply and one add per element
    let total_flops = matrix_size * matrix_size * flops_per_element;
    let tflops_cpu = total_flops as f64 / (cpu_mean as f64 * 1e-3) / 1e12;
    let tflops_gpu = total_flops as f64 / (gpu_mean as f64 * 1e-3) / 1e12;

    println!("\nCompute Performance:");
    println!("  CPU: {:.2} TFLOPS", tflops_cpu);
    println!("  GPU: {:.2} TFLOPS", tflops_gpu);
}
