use anyhow::Result;
use clap::Parser;
use democratising::{tensor::Tensor, Device};
use indicatif::{ProgressBar, ProgressStyle};
use plotters::{
    prelude::*,
    style::full_palette::{BLUE, GREEN, RED},
};
use std::{path::PathBuf, time::Instant};

/// Tensor operations example demonstrating basic tensor manipulations
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Matrix size (N x N)
    #[clap(long, default_value = "1024")]
    size: usize,

    /// Number of iterations for benchmarking
    #[clap(long, default_value = "100")]
    iterations: usize,

    /// Output directory for plots
    #[clap(long, default_value = "output/tensor_ops")]
    output_dir: PathBuf,

    /// Use GPU if available
    #[clap(long)]
    gpu: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Select device
    let device = if args.gpu && cfg!(feature = "gpu") { Device::GPU(0) } else { Device::CPU };
    println!("Using device: {:?}", device);

    // Run benchmarks
    let results = run_benchmarks(&args, device)?;

    // Plot results
    plot_results(&results, &args.output_dir)?;

    // Print summary
    print_summary(&results);

    Ok(())
}

#[derive(Debug)]
struct BenchmarkResults {
    operation: String,
    times: Vec<f32>,
    flops: f64,
    memory_bandwidth: f64,
}

fn run_benchmarks(args: &Args, device: Device) -> Result<Vec<BenchmarkResults>> {
    let mut results = Vec::new();

    // Create progress bar
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();

    // Create tensors
    println!(
        "\nGenerating random matrices of size {}x{}",
        args.size, args.size
    );
    let a = Tensor::randn(&[args.size, args.size])?.to_device(device)?;
    let b = Tensor::randn(&[args.size, args.size])?.to_device(device)?;
    let scalar = 2.0f32;

    // Benchmark matrix multiplication
    {
        println!("\nBenchmarking matrix multiplication...");
        let progress = ProgressBar::new(args.iterations as u64);
        progress.set_style(progress_style.clone());

        let mut times = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            let start = Instant::now();
            let _c = a.matmul(&b)?;
            times.push(start.elapsed().as_secs_f32() * 1000.0);
            progress.inc(1);
            progress.set_message(format!("Time: {:.2} ms", times.last().unwrap()));
        }
        progress.finish_with_message("Complete");

        // Calculate performance metrics
        let flops = 2.0 * args.size.pow(3) as f64; // n^3 multiplications and n^3 additions
        let bytes = 3 * args.size.pow(2) * std::mem::size_of::<f32>(); // Read A, B, write C
        let avg_time = times.iter().sum::<f32>() / args.iterations as f32;
        let flops_per_sec = flops / (avg_time as f64 * 1e-3);
        let bandwidth = bytes as f64 / (avg_time as f64 * 1e-3);

        results.push(BenchmarkResults {
            operation: "Matrix Multiplication".to_string(),
            times,
            flops: flops_per_sec,
            memory_bandwidth: bandwidth,
        });
    }

    // Benchmark element-wise addition
    {
        println!("\nBenchmarking element-wise addition...");
        let progress = ProgressBar::new(args.iterations as u64);
        progress.set_style(progress_style.clone());

        let mut times = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            let start = Instant::now();
            let _c = &a + &b?;
            times.push(start.elapsed().as_secs_f32() * 1000.0);
            progress.inc(1);
            progress.set_message(format!("Time: {:.2} ms", times.last().unwrap()));
        }
        progress.finish_with_message("Complete");

        // Calculate performance metrics
        let flops = args.size.pow(2) as f64; // n^2 additions
        let bytes = 3 * args.size.pow(2) * std::mem::size_of::<f32>(); // Read A, B, write C
        let avg_time = times.iter().sum::<f32>() / args.iterations as f32;
        let flops_per_sec = flops / (avg_time as f64 * 1e-3);
        let bandwidth = bytes as f64 / (avg_time as f64 * 1e-3);

        results.push(BenchmarkResults {
            operation: "Element-wise Addition".to_string(),
            times,
            flops: flops_per_sec,
            memory_bandwidth: bandwidth,
        });
    }

    // Benchmark scalar multiplication
    {
        println!("\nBenchmarking scalar multiplication...");
        let progress = ProgressBar::new(args.iterations as u64);
        progress.set_style(progress_style.clone());

        let mut times = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            let start = Instant::now();
            let _c = &a * scalar;
            times.push(start.elapsed().as_secs_f32() * 1000.0);
            progress.inc(1);
            progress.set_message(format!("Time: {:.2} ms", times.last().unwrap()));
        }
        progress.finish_with_message("Complete");

        // Calculate performance metrics
        let flops = args.size.pow(2) as f64; // n^2 multiplications
        let bytes = 2 * args.size.pow(2) * std::mem::size_of::<f32>(); // Read A, write C
        let avg_time = times.iter().sum::<f32>() / args.iterations as f32;
        let flops_per_sec = flops / (avg_time as f64 * 1e-3);
        let bandwidth = bytes as f64 / (avg_time as f64 * 1e-3);

        results.push(BenchmarkResults {
            operation: "Scalar Multiplication".to_string(),
            times,
            flops: flops_per_sec,
            memory_bandwidth: bandwidth,
        });
    }

    // Benchmark reduction (sum)
    {
        println!("\nBenchmarking reduction (sum)...");
        let progress = ProgressBar::new(args.iterations as u64);
        progress.set_style(progress_style.clone());

        let mut times = Vec::with_capacity(args.iterations);
        for _ in 0..args.iterations {
            let start = Instant::now();
            let _c = a.sum(&[0, 1])?;
            times.push(start.elapsed().as_secs_f32() * 1000.0);
            progress.inc(1);
            progress.set_message(format!("Time: {:.2} ms", times.last().unwrap()));
        }
        progress.finish_with_message("Complete");

        // Calculate performance metrics
        let flops = args.size.pow(2) as f64; // n^2 additions
        let bytes = args.size.pow(2) * std::mem::size_of::<f32>(); // Read A
        let avg_time = times.iter().sum::<f32>() / args.iterations as f32;
        let flops_per_sec = flops / (avg_time as f64 * 1e-3);
        let bandwidth = bytes as f64 / (avg_time as f64 * 1e-3);

        results.push(BenchmarkResults {
            operation: "Reduction (Sum)".to_string(),
            times,
            flops: flops_per_sec,
            memory_bandwidth: bandwidth,
        });
    }

    Ok(results)
}

fn plot_results(results: &[BenchmarkResults], output_dir: &PathBuf) -> Result<()> {
    // Plot execution times
    {
        let path = output_dir.join("execution_times.png");
        let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_time = results
            .iter()
            .flat_map(|r| r.times.iter())
            .fold(0f32, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption("Execution Times by Operation", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..results[0].times.len() as f32, 0f32..max_time * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Iteration")
            .y_desc("Time (ms)")
            .draw()?;

        let colors = [RED, BLUE, GREEN];
        for (result, &color) in results.iter().zip(colors.iter()) {
            chart.draw_series(LineSeries::new(
                result.times.iter().enumerate().map(|(i, &v)| (i as f32, v)),
                color.mix(0.5),
            ))?;
        }

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    Ok(())
}

fn print_summary(results: &[BenchmarkResults]) {
    println!("\nPerformance Summary:");
    println!("-------------------");

    for result in results {
        let avg_time = result.times.iter().sum::<f32>() / result.times.len() as f32;
        let std_dev = (result
            .times
            .iter()
            .map(|&x| (x - avg_time).powi(2))
            .sum::<f32>()
            / result.times.len() as f32)
            .sqrt();

        println!("\n{}", result.operation);
        println!("  Time: {:.2} ± {:.2} ms", avg_time, std_dev);
        println!("  Performance: {:.2} GFLOPS", result.flops / 1e9);
        println!(
            "  Memory Bandwidth: {:.2} GB/s",
            result.memory_bandwidth / 1e9
        );
    }
}
