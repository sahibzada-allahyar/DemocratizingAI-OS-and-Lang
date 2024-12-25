use anyhow::Result;
use clap::Parser;
use democratising::{
    nn::{
        activation::{ReLU, Sigmoid, Tanh},
        layer::{Conv2D, Dense, Dropout, Flatten, MaxPool2D},
        loss::{CrossEntropyLoss, MSELoss},
        optimizer::{Adam, SGD},
        Model,
    },
    tensor::Tensor,
    Device,
};
use indicatif::{ProgressBar, ProgressStyle};
use plotters::{
    prelude::*,
    style::full_palette::{BLUE, GREEN, RED},
};
use std::{path::PathBuf, time::Instant};

/// Neural network example demonstrating various architectures and training options
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Model architecture (mlp, cnn, lstm)
    #[clap(long, default_value = "mlp")]
    model: String,

    /// Batch size
    #[clap(long, default_value = "32")]
    batch_size: usize,

    /// Number of epochs
    #[clap(long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[clap(long, default_value = "0.001")]
    learning_rate: f32,

    /// Optimizer (adam, sgd)
    #[clap(long, default_value = "adam")]
    optimizer: String,

    /// Loss function (mse, cross_entropy)
    #[clap(long, default_value = "cross_entropy")]
    loss: String,

    /// Use GPU if available
    #[clap(long)]
    gpu: bool,

    /// Output directory for plots and model
    #[clap(long, default_value = "output/neural_network")]
    output_dir: PathBuf,
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

    // Create model
    let mut model = create_model(&args)?;
    println!("\nModel architecture:");
    model.summary();

    // Create data
    println!("\nGenerating training data...");
    let (train_data, train_labels) = create_training_data(1000, &args)?;
    let (test_data, test_labels) = create_test_data(100, &args)?;

    // Train model
    let history = train_model(
        &mut model,
        &train_data,
        &train_labels,
        &test_data,
        &test_labels,
        &args,
    )?;

    // Plot training history
    plot_training_history(&history, &args.output_dir)?;

    // Save model
    let model_path = args.output_dir.join("model.bin");
    model.save(&model_path)?;
    println!("\nModel saved to: {}", model_path.display());

    // Final evaluation
    let test_predictions = model.forward(&test_data)?;
    let test_loss = model.compute_loss(&test_predictions, &test_labels)?;
    let test_accuracy = compute_accuracy(&test_predictions, &test_labels)?;
    println!(
        "\nFinal test metrics: Loss = {:.4}, Accuracy = {:.2}%",
        test_loss,
        test_accuracy * 100.0
    );

    Ok(())
}

fn create_model(args: &Args) -> Result<Model<f32>> {
    let mut model = Model::new();

    // Add layers based on architecture
    match args.model.as_str() {
        "mlp" => {
            model.add_layer(Dense::new(784, 256, ReLU::default()));
            model.add_layer(Dropout::new(0.5));
            model.add_layer(Dense::new(256, 128, ReLU::default()));
            model.add_layer(Dropout::new(0.3));
            model.add_layer(Dense::new(128, 10, Sigmoid::default()));
        }
        "cnn" => {
            model.add_layer(Conv2D::new(1, 32, 3, ReLU::default()));
            model.add_layer(MaxPool2D::new(2));
            model.add_layer(Conv2D::new(32, 64, 3, ReLU::default()));
            model.add_layer(MaxPool2D::new(2));
            model.add_layer(Flatten::new());
            model.add_layer(Dense::new(64 * 5 * 5, 128, ReLU::default()));
            model.add_layer(Dropout::new(0.5));
            model.add_layer(Dense::new(128, 10, Sigmoid::default()));
        }
        "lstm" => {
            // TODO: Add LSTM implementation
            unimplemented!("LSTM architecture not yet implemented");
        }
        _ => {
            return Err(anyhow::anyhow!(
                "Unknown model architecture: {}",
                args.model
            ));
        }
    }

    // Set optimizer
    match args.optimizer.as_str() {
        "adam" => model.set_optimizer(Adam::new(args.learning_rate)),
        "sgd" => model.set_optimizer(SGD::new(args.learning_rate)),
        _ => {
            return Err(anyhow::anyhow!("Unknown optimizer: {}", args.optimizer));
        }
    }

    // Set loss function
    match args.loss.as_str() {
        "mse" => model.set_loss(MSELoss::default()),
        "cross_entropy" => model.set_loss(CrossEntropyLoss::default()),
        _ => {
            return Err(anyhow::anyhow!("Unknown loss function: {}", args.loss));
        }
    }

    Ok(model)
}

fn create_training_data(num_samples: usize, args: &Args) -> Result<(Tensor<f32>, Tensor<f32>)> {
    match args.model.as_str() {
        "mlp" => {
            let data = Tensor::randn(&[num_samples, 784])?;
            let labels = Tensor::randn(&[num_samples, 10])?;
            Ok((data, labels))
        }
        "cnn" => {
            let data = Tensor::randn(&[num_samples, 1, 28, 28])?;
            let labels = Tensor::randn(&[num_samples, 10])?;
            Ok((data, labels))
        }
        _ => Err(anyhow::anyhow!(
            "Data generation not implemented for architecture: {}",
            args.model
        )),
    }
}

fn create_test_data(num_samples: usize, args: &Args) -> Result<(Tensor<f32>, Tensor<f32>)> {
    create_training_data(num_samples, args)
}

struct TrainingHistory {
    train_losses: Vec<f32>,
    train_accuracies: Vec<f32>,
    test_losses: Vec<f32>,
    test_accuracies: Vec<f32>,
}

fn train_model(
    model: &mut Model<f32>,
    train_data: &Tensor<f32>,
    train_labels: &Tensor<f32>,
    test_data: &Tensor<f32>,
    test_labels: &Tensor<f32>,
    args: &Args,
) -> Result<TrainingHistory> {
    let num_samples = train_data.shape()[0];
    let num_batches = (num_samples + args.batch_size - 1) / args.batch_size;

    // Create progress bar
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();

    // Training history
    let mut history = TrainingHistory {
        train_losses: Vec::new(),
        train_accuracies: Vec::new(),
        test_losses: Vec::new(),
        test_accuracies: Vec::new(),
    };

    println!("\nTraining model...");
    for epoch in 0..args.epochs {
        let start_time = Instant::now();
        let mut total_loss = 0.0;

        // Create progress bar for this epoch
        let progress = ProgressBar::new(num_batches as u64);
        progress.set_style(progress_style.clone());

        // Training loop
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * args.batch_size;
            let end_idx = (start_idx + args.batch_size).min(num_samples);

            // Extract batch
            let batch_data = train_data.slice(start_idx..end_idx)?;
            let batch_labels = train_labels.slice(start_idx..end_idx)?;

            // Train step
            let loss = model.train_step(&batch_data, &batch_labels)?;
            total_loss += loss;

            progress.inc(1);
            progress.set_message(format!("Loss: {:.4}", loss));
        }

        progress.finish_and_clear();

        // Compute epoch metrics
        let train_loss = total_loss / num_batches as f32;
        let train_accuracy = compute_accuracy(&model.forward(train_data)?, train_labels)?;
        let test_loss = model.compute_loss(&model.forward(test_data)?, test_labels)?;
        let test_accuracy = compute_accuracy(&model.forward(test_data)?, test_labels)?;

        // Update history
        history.train_losses.push(train_loss);
        history.train_accuracies.push(train_accuracy);
        history.test_losses.push(test_loss);
        history.test_accuracies.push(test_accuracy);

        let elapsed = start_time.elapsed();
        println!(
            "Epoch {}/{}: Train Loss = {:.4}, Train Acc = {:.2}%, Test Loss = {:.4}, Test Acc = {:.2}%, Time = {:.2}s",
            epoch + 1,
            args.epochs,
            train_loss,
            train_accuracy * 100.0,
            test_loss,
            test_accuracy * 100.0,
            elapsed.as_secs_f32()
        );
    }

    Ok(history)
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
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(
                0f32..history.train_losses.len() as f32,
                0f32..max_loss * 1.1,
            )?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
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
                .map(|(i, &v)| (i as f32, v)),
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
            .x_label_area_size(30)
            .y_label_area_size(50)
            .build_cartesian_2d(0f32..history.train_accuracies.len() as f32, 0f32..1f32)?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
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
                .map(|(i, &v)| (i as f32, v)),
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
