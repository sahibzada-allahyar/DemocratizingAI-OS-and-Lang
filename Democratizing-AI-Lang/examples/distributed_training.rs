use anyhow::Result;
use clap::Parser;
use democratising::{
    nn::{
        activation::{ReLU, Sigmoid},
        layer::{Dense, Dropout},
        loss::CrossEntropyLoss,
        optimizer::Adam,
        Model,
    },
    tensor::Tensor,
    Device,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use plotters::{
    prelude::*,
    style::full_palette::{BLUE, GREEN, RED},
};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
    time::Instant,
};

/// Distributed training example demonstrating parallel model training
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Number of worker threads
    #[clap(long, default_value = "4")]
    num_workers: usize,

    /// Batch size per worker
    #[clap(long, default_value = "32")]
    batch_size: usize,

    /// Number of epochs
    #[clap(long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[clap(long, default_value = "0.001")]
    learning_rate: f32,

    /// Output directory for plots and model
    #[clap(long, default_value = "output/distributed")]
    output_dir: PathBuf,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Create shared model
    let model = create_model(args.learning_rate)?;
    let model = Arc::new(Mutex::new(model));

    // Create progress bars
    let multi_progress = MultiProgress::new();
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();

    // Training history
    let history = Arc::new(Mutex::new(TrainingHistory {
        train_losses: Vec::new(),
        train_accuracies: Vec::new(),
        test_losses: Vec::new(),
        test_accuracies: Vec::new(),
    }));

    // Create test data
    let (test_data, test_labels) = create_test_data(1000)?;

    println!(
        "\nStarting distributed training with {} workers",
        args.num_workers
    );
    println!(
        "Each worker will process batches of size {}",
        args.batch_size
    );

    // Training loop
    for epoch in 0..args.epochs {
        let epoch_start = Instant::now();
        println!("\nEpoch {}/{}", epoch + 1, args.epochs);

        // Spawn worker threads
        let mut handles = Vec::new();
        let mut progress_bars = Vec::new();

        for worker_id in 0..args.num_workers {
            let model = Arc::clone(&model);
            let history = Arc::clone(&history);
            let progress = multi_progress.add(ProgressBar::new(100));
            progress.set_style(progress_style.clone());
            progress_bars.push(progress.clone());

            let handle = thread::spawn(move || -> Result<()> {
                // Each worker processes its own batches
                for step in 0..100 {
                    // Create training batch
                    let (batch_data, batch_labels) = create_training_batch(args.batch_size)?;

                    // Train step
                    let mut model = model.lock().unwrap();
                    let loss = model.train_step(&batch_data, &batch_labels)?;
                    let accuracy = compute_accuracy(&model.forward(&batch_data)?, &batch_labels)?;

                    // Update progress
                    progress.inc(1);
                    progress.set_message(format!(
                        "Worker {}: Loss = {:.4}, Acc = {:.2}%",
                        worker_id,
                        loss,
                        accuracy * 100.0
                    ));

                    // Record metrics
                    if step % 10 == 0 {
                        let mut history = history.lock().unwrap();
                        history.train_losses.push(loss);
                        history.train_accuracies.push(accuracy);
                    }
                }
                Ok(())
            });
            handles.push(handle);
        }

        // Wait for all workers
        for handle in handles {
            handle.join().unwrap()?;
        }

        // Evaluate on test set
        let model = model.lock().unwrap();
        let test_predictions = model.forward(&test_data)?;
        let test_loss = model.compute_loss(&test_predictions, &test_labels)?;
        let test_accuracy = compute_accuracy(&test_predictions, &test_labels)?;

        let mut history = history.lock().unwrap();
        history.test_losses.push(test_loss);
        history.test_accuracies.push(test_accuracy);

        let epoch_time = epoch_start.elapsed();
        println!(
            "Epoch complete - Test Loss = {:.4}, Test Acc = {:.2}%, Time = {:.2}s",
            test_loss,
            test_accuracy * 100.0,
            epoch_time.as_secs_f32()
        );

        // Clear progress bars
        for bar in &progress_bars {
            bar.finish_and_clear();
        }
    }

    // Plot training history
    let history = history.lock().unwrap();
    plot_training_history(&history, &args.output_dir)?;

    // Save final model
    let model = model.lock().unwrap();
    let model_path = args.output_dir.join("model.bin");
    model.save(&model_path)?;
    println!("\nModel saved to: {}", model_path.display());

    Ok(())
}

struct TrainingHistory {
    train_losses: Vec<f32>,
    train_accuracies: Vec<f32>,
    test_losses: Vec<f32>,
    test_accuracies: Vec<f32>,
}

fn create_model(learning_rate: f32) -> Result<Model<f32>> {
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dropout::new(0.5));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(learning_rate));
    model.set_loss(CrossEntropyLoss::default());
    Ok(model)
}

fn create_training_batch(batch_size: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
    // Generate random training data for demonstration
    let data = Tensor::randn(&[batch_size, 784])?;
    let labels = Tensor::randn(&[batch_size, 10])?;
    Ok((data, labels))
}

fn create_test_data(num_samples: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
    // Generate random test data for demonstration
    let data = Tensor::randn(&[num_samples, 784])?;
    let labels = Tensor::randn(&[num_samples, 10])?;
    Ok((data, labels))
}

fn compute_accuracy(predictions: &Tensor<f32>, labels: &Tensor<f32>) -> Result<f32> {
    let pred_indices = predictions.argmax(1)?;
    let label_indices = labels.argmax(1)?;
    let correct = pred_indices
        .iter()
        .zip(label_indices.iter())
        .filter(|(&p, &l)| p == l)
        .count();
    Ok(correct as f32 / predictions.shape()[0] as f32)
}

fn plot_training_history(history: &TrainingHistory, output_dir: &PathBuf) -> Result<()> {
    // Plot loss
    {
        let path = output_dir.join("loss.png");
        let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let max_loss = history
            .train_losses
            .iter()
            .chain(history.test_losses.iter())
            .fold(0f32, |a, &b| a.max(b));

        let mut chart = ChartBuilder::on(&root)
            .caption("Training History: Loss", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(
                0f32..history.train_losses.len() as f32,
                0f32..max_loss * 1.1,
            )?;

        chart
            .configure_mesh()
            .x_desc("Step")
            .y_desc("Loss")
            .draw()?;

        chart.draw_series(LineSeries::new(
            history
                .train_losses
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as f32, v)),
            RED.mix(0.5),
        ))?;

        chart.draw_series(LineSeries::new(
            history
                .test_losses
                .iter()
                .enumerate()
                .map(|(i, &v)| ((i * 100) as f32, v)),
            BLUE.mix(0.5),
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    // Plot accuracy
    {
        let path = output_dir.join("accuracy.png");
        let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training History: Accuracy", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(50)
            .y_label_area_size(60)
            .build_cartesian_2d(0f32..history.train_accuracies.len() as f32, 0f32..1f32)?;

        chart
            .configure_mesh()
            .x_desc("Step")
            .y_desc("Accuracy")
            .draw()?;

        chart.draw_series(LineSeries::new(
            history
                .train_accuracies
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as f32, v)),
            RED.mix(0.5),
        ))?;

        chart.draw_series(LineSeries::new(
            history
                .test_accuracies
                .iter()
                .enumerate()
                .map(|(i, &v)| ((i * 100) as f32, v)),
            BLUE.mix(0.5),
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .position(SeriesLabelPosition::UpperRight)
            .draw()?;
    }

    Ok(())
}
