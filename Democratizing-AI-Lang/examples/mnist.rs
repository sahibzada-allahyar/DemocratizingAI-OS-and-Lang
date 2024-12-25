use anyhow::Result;
use clap::Parser;
use democratising::{
    nn::{
        activation::{ReLU, Softmax},
        layer::{Conv2D, Dense, Dropout, Flatten, MaxPool2D},
        loss::CrossEntropyLoss,
        optimizer::Adam,
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
use std::{fs::File, path::PathBuf, time::Instant};

/// MNIST digit classification example
#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Path to MNIST data directory
    #[clap(long, default_value = "data/mnist")]
    data_dir: PathBuf,

    /// Batch size
    #[clap(long, default_value = "32")]
    batch_size: usize,

    /// Number of epochs
    #[clap(long, default_value = "10")]
    epochs: usize,

    /// Learning rate
    #[clap(long, default_value = "0.001")]
    learning_rate: f32,

    /// Use GPU if available
    #[clap(long)]
    gpu: bool,

    /// Output directory for plots and model
    #[clap(long, default_value = "output/mnist")]
    output_dir: PathBuf,
}

const IMAGE_SIZE: usize = 28;
const NUM_CLASSES: usize = 10;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Parse command line arguments
    let args = Args::parse();

    // Select device
    let device = if args.gpu && cfg!(feature = "gpu") { Device::GPU(0) } else { Device::CPU };
    println!("Using device: {:?}", device);

    // Create output directory
    std::fs::create_dir_all(&args.output_dir)?;

    // Load MNIST dataset
    println!("\nLoading MNIST dataset...");
    let (train_images, train_labels) = load_mnist(&args.data_dir, "train")?;
    let (test_images, test_labels) = load_mnist(&args.data_dir, "t10k")?;

    // Create model
    let mut model = create_model(device)?;
    println!("\nModel architecture:");
    model.summary();

    // Train model
    let (train_losses, train_accuracies, test_losses, test_accuracies) = train_model(
        &mut model,
        &train_images,
        &train_labels,
        &test_images,
        &test_labels,
        &args,
    )?;

    // Plot training history
    plot_training_history(
        &train_losses,
        &train_accuracies,
        &test_losses,
        &test_accuracies,
        &args.output_dir,
    )?;

    // Save model
    let model_path = args.output_dir.join("model.bin");
    model.save(&model_path)?;
    println!("\nModel saved to: {}", model_path.display());

    // Final evaluation
    let test_loss = evaluate_model(&model, &test_images, &test_labels)?;
    let test_accuracy = compute_accuracy(&model, &test_images, &test_labels)?;
    println!(
        "\nFinal test metrics: Loss = {:.4}, Accuracy = {:.2}%",
        test_loss,
        test_accuracy * 100.0
    );

    Ok(())
}

fn create_model(device: Device) -> Result<Model<f32>> {
    let mut model = Model::new();

    // Convolutional layers
    model.add_layer(Conv2D::new(1, 32, 3, ReLU::default()));
    model.add_layer(Conv2D::new(32, 32, 3, ReLU::default()));
    model.add_layer(MaxPool2D::new(2));
    model.add_layer(Dropout::new(0.25));

    model.add_layer(Conv2D::new(32, 64, 3, ReLU::default()));
    model.add_layer(Conv2D::new(64, 64, 3, ReLU::default()));
    model.add_layer(MaxPool2D::new(2));
    model.add_layer(Dropout::new(0.25));

    // Fully connected layers
    model.add_layer(Flatten::new());
    model.add_layer(Dense::new(64 * 5 * 5, 512, ReLU::default()));
    model.add_layer(Dropout::new(0.5));
    model.add_layer(Dense::new(512, NUM_CLASSES, Softmax::default()));

    // Set optimizer and loss function
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());

    // Move model to device
    model.to_device(device)?;

    Ok(model)
}

fn train_model(
    model: &mut Model<f32>,
    train_images: &Tensor<f32>,
    train_labels: &Tensor<f32>,
    test_images: &Tensor<f32>,
    test_labels: &Tensor<f32>,
    args: &Args,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let num_samples = train_images.shape()[0];
    let num_batches = (num_samples + args.batch_size - 1) / args.batch_size;

    // Create progress bar
    let progress_style = ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
        .unwrap();

    // Training history
    let mut train_losses = Vec::with_capacity(args.epochs);
    let mut train_accuracies = Vec::with_capacity(args.epochs);
    let mut test_losses = Vec::with_capacity(args.epochs);
    let mut test_accuracies = Vec::with_capacity(args.epochs);

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
            let batch_images = train_images.slice(start_idx..end_idx)?;
            let batch_labels = train_labels.slice(start_idx..end_idx)?;

            // Train step
            let loss = model.train_step(&batch_images, &batch_labels)?;
            total_loss += loss;

            progress.inc(1);
            progress.set_message(format!("Loss: {:.4}", loss));
        }

        progress.finish_and_clear();

        // Compute epoch metrics
        let train_loss = total_loss / num_batches as f32;
        let train_accuracy = compute_accuracy(model, train_images, train_labels)?;
        let test_loss = evaluate_model(model, test_images, test_labels)?;
        let test_accuracy = compute_accuracy(model, test_images, test_labels)?;

        // Update history
        train_losses.push(train_loss);
        train_accuracies.push(train_accuracy);
        test_losses.push(test_loss);
        test_accuracies.push(test_accuracy);

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

    Ok((train_losses, train_accuracies, test_losses, test_accuracies))
}

fn evaluate_model(model: &Model<f32>, images: &Tensor<f32>, labels: &Tensor<f32>) -> Result<f32> {
    let predictions = model.forward(images)?;
    let loss = model.compute_loss(&predictions, labels)?;
    Ok(loss)
}

fn compute_accuracy(model: &Model<f32>, images: &Tensor<f32>, labels: &Tensor<f32>) -> Result<f32> {
    let predictions = model.forward(images)?;
    let pred_indices = predictions.argmax(1)?;
    let label_indices = labels.argmax(1)?;
    let correct = pred_indices
        .iter()
        .zip(label_indices.iter())
        .filter(|(&p, &l)| p == l)
        .count();
    Ok(correct as f32 / images.shape()[0] as f32)
}

fn load_mnist(data_dir: &PathBuf, dataset: &str) -> Result<(Tensor<f32>, Tensor<f32>)> {
    // Load images
    let images_path = data_dir.join(format!("{}-images-idx3-ubyte", dataset));
    let mut images_file = File::open(images_path)?;
    let images = read_idx(&mut images_file)?;
    let images = images.reshape(&[-1, 1, IMAGE_SIZE, IMAGE_SIZE])?;
    let images = images.map(|x| x as f32 / 255.0)?;

    // Load labels
    let labels_path = data_dir.join(format!("{}-labels-idx1-ubyte", dataset));
    let mut labels_file = File::open(labels_path)?;
    let labels = read_idx(&mut labels_file)?;
    let labels = one_hot_encode(&labels, NUM_CLASSES)?;

    Ok((images, labels))
}

fn read_idx(file: &mut File) -> Result<Tensor<u8>> {
    use byteorder::{BigEndian, ReadBytesExt};
    use std::io::Read;

    // Read magic number
    let magic = file.read_u32::<BigEndian>()?;
    let is_images = (magic >> 8) == 0x0803;
    let num_dims = (magic & 0xff) as usize;

    // Read dimensions
    let mut dims = Vec::with_capacity(num_dims);
    for _ in 0..num_dims {
        dims.push(file.read_u32::<BigEndian>()? as usize);
    }

    // Read data
    let num_elements = dims.iter().product();
    let mut data = vec![0u8; num_elements];
    file.read_exact(&mut data)?;

    // Create tensor
    Tensor::from_vec(data, &dims)
}

fn one_hot_encode(labels: &Tensor<u8>, num_classes: usize) -> Result<Tensor<f32>> {
    let num_samples = labels.shape()[0];
    let mut encoded = Tensor::zeros(&[num_samples, num_classes])?;
    for (i, &label) in labels.iter().enumerate() {
        encoded[[i, label as usize]] = 1.0;
    }
    Ok(encoded)
}

fn plot_training_history(
    train_losses: &[f32],
    train_accuracies: &[f32],
    test_losses: &[f32],
    test_accuracies: &[f32],
    output_dir: &PathBuf,
) -> Result<()> {
    // Plot loss
    {
        let path = output_dir.join("loss.png");
        let root = BitMapBackend::new(&path, (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption("Training History: Loss", ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(
                0f32..train_losses.len() as f32,
                0f32..*train_losses.iter().chain(test_losses).fold(&0f32, |a, b| {
                    if b > a {
                        b
                    } else {
                        a
                    }
                }),
            )?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Loss")
            .draw()?;

        chart.draw_series(LineSeries::new(
            train_losses.iter().enumerate().map(|(i, &v)| (i as f32, v)),
            RED,
        ))?;

        chart.draw_series(LineSeries::new(
            test_losses.iter().enumerate().map(|(i, &v)| (i as f32, v)),
            BLUE,
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
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
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..train_accuracies.len() as f32, 0f32..1f32)?;

        chart
            .configure_mesh()
            .x_desc("Epoch")
            .y_desc("Accuracy")
            .draw()?;

        chart.draw_series(LineSeries::new(
            train_accuracies
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as f32, v)),
            RED,
        ))?;

        chart.draw_series(LineSeries::new(
            test_accuracies
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as f32, v)),
            BLUE,
        ))?;

        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;
    }

    Ok(())
}
