use anyhow::Result;
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
use std::{fs, path::PathBuf};

/// Create a simple MLP model for testing
pub fn create_test_model() -> Model<f32> {
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dropout::new(0.5));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());
    model
}

/// Create random test data
pub fn create_test_data(batch_size: usize) -> Result<(Tensor<f32>, Tensor<f32>)> {
    let x = Tensor::randn(&[batch_size, 784])?;
    let y = Tensor::randn(&[batch_size, 10])?;
    Ok((x, y))
}

/// Create a temporary directory for test files
pub fn create_test_dir() -> Result<tempfile::TempDir> {
    Ok(tempfile::tempdir()?)
}

/// Save and load a tensor for testing
pub fn test_tensor_serialization(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
    let temp_dir = create_test_dir()?;
    let path = temp_dir.path().join("tensor.bin");
    tensor.save(&path)?;
    Tensor::load(&path)
}

/// Save and load a model for testing
pub fn test_model_serialization(model: &Model<f32>) -> Result<Model<f32>> {
    let temp_dir = create_test_dir()?;
    let path = temp_dir.path().join("model.bin");
    model.save(&path)?;
    Model::load(&path)
}

/// Compute accuracy between predictions and labels
pub fn compute_accuracy(predictions: &Tensor<f32>, labels: &Tensor<f32>) -> Result<f32> {
    let pred_indices = predictions.argmax(1)?;
    let label_indices = labels.argmax(1)?;
    let correct = pred_indices
        .iter()
        .zip(label_indices.iter())
        .filter(|(&p, &l)| p == l)
        .count();
    Ok(correct as f32 / predictions.shape()[0] as f32)
}

/// Assert that two tensors have the same shape
pub fn assert_tensor_shape(a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape());
}

/// Assert that a tensor has a specific shape
pub fn assert_shape(tensor: &Tensor<f32>, shape: &[usize]) {
    assert_eq!(tensor.shape(), shape);
}

/// Assert that a value is within a range
pub fn assert_in_range(value: f32, min: f32, max: f32) {
    assert!(value >= min && value <= max);
}

/// Assert that all tensor elements are finite
pub fn assert_all_finite(tensor: &Tensor<f32>) -> Result<()> {
    assert!(tensor.iter().all(|&x| x.is_finite()));
    Ok(())
}

/// Assert that gradients exist and have correct shapes
pub fn assert_gradients(tensor: &Tensor<f32>, expected_shape: &[usize]) -> Result<()> {
    let grad = tensor.grad().expect("Gradient should exist");
    assert_eq!(grad.shape(), expected_shape);
    assert_all_finite(&grad)?;
    Ok(())
}

/// Helper to run a model through a single training step
pub fn train_step(
    model: &mut Model<f32>,
    batch_size: usize,
    learning_rate: f32,
) -> Result<(f32, f32)> {
    // Create data
    let (x, y) = create_test_data(batch_size)?;

    // Set learning rate
    model.set_optimizer(Adam::new(learning_rate));

    // Forward and backward pass
    let loss = model.train_step(&x, &y)?;
    let accuracy = compute_accuracy(&model.forward(&x)?, &y)?;

    Ok((loss, accuracy))
}

/// Helper to evaluate a model on test data
pub fn evaluate_model(
    model: &Model<f32>,
    test_data: &Tensor<f32>,
    test_labels: &Tensor<f32>,
) -> Result<(f32, f32)> {
    let predictions = model.forward(test_data)?;
    let loss = model.compute_loss(&predictions, test_labels)?;
    let accuracy = compute_accuracy(&predictions, test_labels)?;
    Ok((loss, accuracy))
}

#[cfg(feature = "gpu")]
pub mod gpu {
    use super::*;

    /// Move tensors to GPU for testing
    pub fn to_gpu(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
        tensor.to_device(Device::GPU(0))
    }

    /// Move tensors back to CPU for testing
    pub fn to_cpu(tensor: &Tensor<f32>) -> Result<Tensor<f32>> {
        tensor.to_device(Device::CPU)
    }

    /// Run GPU tests if available
    pub fn run_if_gpu_available<F>(test_fn: F) -> Result<()>
    where
        F: FnOnce() -> Result<()>,
    {
        if cfg!(feature = "gpu") {
            test_fn()
        } else {
            println!("Skipping GPU test - GPU support not enabled");
            Ok(())
        }
    }

    /// Compare CPU and GPU results
    pub fn compare_cpu_gpu_results<F>(compute_fn: F) -> Result<()>
    where
        F: Fn(&Tensor<f32>) -> Result<Tensor<f32>>,
    {
        let input = Tensor::randn(&[32, 32])?;

        // CPU computation
        let cpu_result = compute_fn(&input)?;

        // GPU computation
        let gpu_input = to_gpu(&input)?;
        let gpu_result = compute_fn(&gpu_input)?;
        let gpu_result_cpu = to_cpu(&gpu_result)?;

        // Compare results
        assert_eq!(cpu_result.shape(), gpu_result_cpu.shape());

        Ok(())
    }
}

/// Test utilities for distributed training
pub mod distributed {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    /// Simulate distributed training with multiple threads
    pub fn simulate_distributed_training(
        num_workers: usize,
        batch_size: usize,
        iterations: usize,
    ) -> Result<()> {
        // Create shared model
        let model = create_test_model();
        let model = Arc::new(Mutex::new(model));

        // Spawn worker threads
        let mut handles = vec![];
        for worker_id in 0..num_workers {
            let model = Arc::clone(&model);
            let handle = thread::spawn(move || -> Result<()> {
                for _ in 0..iterations {
                    let (x, y) = create_test_data(batch_size)?;
                    let mut model = model.lock().unwrap();
                    let _ = model.train_step(&x, &y)?;
                }
                Ok(())
            });
            handles.push(handle);
        }

        // Wait for all workers
        for handle in handles {
            handle.join().unwrap()?;
        }

        Ok(())
    }
}
