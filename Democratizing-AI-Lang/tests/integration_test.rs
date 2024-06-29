use anyhow::Result;
use democratising::{
    nn::{
        activation::{ReLU, Sigmoid},
        layer::{Conv2D, Dense, Dropout, Flatten, MaxPool2D},
        loss::CrossEntropyLoss,
        optimizer::Adam,
        Model,
    },
    tensor::Tensor,
    Device,
};
use std::{fs, path::PathBuf};

mod common;

#[test]
fn test_tensor_operations() -> Result<()> {
    // Basic tensor creation
    let a = Tensor::randn(&[32, 32])?;
    let b = Tensor::randn(&[32, 32])?;
    let scalar = 2.0f32;

    // Element-wise operations
    let c = &a + &b?;
    let d = &a - &b?;
    let e = &a * &b?;
    let f = &a / &b?;
    let g = &a * scalar;

    assert_eq!(c.shape(), &[32, 32]);
    assert_eq!(d.shape(), &[32, 32]);
    assert_eq!(e.shape(), &[32, 32]);
    assert_eq!(f.shape(), &[32, 32]);
    assert_eq!(g.shape(), &[32, 32]);

    // Matrix multiplication
    let h = a.matmul(&b)?;
    assert_eq!(h.shape(), &[32, 32]);

    // Reductions
    let i = a.sum(&[0])?;
    let j = a.mean(&[1])?;
    assert_eq!(i.shape(), &[32]);
    assert_eq!(j.shape(), &[32]);

    Ok(())
}

#[test]
fn test_neural_network() -> Result<()> {
    // Create model
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dropout::new(0.5));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());

    // Create dummy data
    let x = Tensor::randn(&[32, 784])?;
    let y = Tensor::randn(&[32, 10])?;

    // Forward pass
    let output = model.forward(&x)?;
    assert_eq!(output.shape(), &[32, 10]);

    // Backward pass
    let loss = model.train_step(&x, &y)?;
    assert!(loss >= 0.0);

    Ok(())
}

#[test]
fn test_cnn() -> Result<()> {
    // Create CNN model
    let mut model = Model::new();
    model.add_layer(Conv2D::new(1, 32, 3, ReLU::default()));
    model.add_layer(MaxPool2D::new(2));
    model.add_layer(Conv2D::new(32, 64, 3, ReLU::default()));
    model.add_layer(MaxPool2D::new(2));
    model.add_layer(Flatten::new());
    model.add_layer(Dense::new(64 * 5 * 5, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());

    // Create dummy data
    let x = Tensor::randn(&[32, 1, 28, 28])?;
    let y = Tensor::randn(&[32, 10])?;

    // Forward pass
    let output = model.forward(&x)?;
    assert_eq!(output.shape(), &[32, 10]);

    // Backward pass
    let loss = model.train_step(&x, &y)?;
    assert!(loss >= 0.0);

    Ok(())
}

#[test]
fn test_model_save_load() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let model_path = temp_dir.path().join("model.bin");

    // Create and train model
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());

    let x = Tensor::randn(&[32, 784])?;
    let y = Tensor::randn(&[32, 10])?;
    let _ = model.train_step(&x, &y)?;

    // Save model
    model.save(&model_path)?;
    assert!(model_path.exists());

    // Load model
    let loaded_model = Model::load(&model_path)?;
    let output = loaded_model.forward(&x)?;
    assert_eq!(output.shape(), &[32, 10]);

    Ok(())
}

#[test]
#[cfg(feature = "gpu")]
fn test_gpu_operations() -> Result<()> {
    // Create CPU tensors
    let a = Tensor::randn(&[32, 32])?;
    let b = Tensor::randn(&[32, 32])?;

    // Move to GPU
    let a_gpu = a.to_device(Device::GPU(0))?;
    let b_gpu = b.to_device(Device::GPU(0))?;

    // GPU operations
    let c_gpu = &a_gpu + &b_gpu?;
    let d_gpu = a_gpu.matmul(&b_gpu)?;

    // Move back to CPU
    let c = c_gpu.to_device(Device::CPU)?;
    let d = d_gpu.to_device(Device::CPU)?;

    assert_eq!(c.shape(), &[32, 32]);
    assert_eq!(d.shape(), &[32, 32]);

    Ok(())
}

#[test]
fn test_autodiff() -> Result<()> {
    // Create tensors with gradients
    let x = Tensor::randn(&[32, 32])?.requires_grad(true);
    let y = Tensor::randn(&[32, 32])?.requires_grad(true);

    // Forward operations
    let z = (&x * &y)? + &x.sum(&[1])?.reshape(&[-1, 1])?;
    let loss = z.mean(&[0, 1])?;

    // Backward pass
    loss.backward()?;

    // Check gradients
    assert!(x.grad().is_some());
    assert!(y.grad().is_some());
    assert_eq!(x.grad().unwrap().shape(), &[32, 32]);
    assert_eq!(y.grad().unwrap().shape(), &[32, 32]);

    Ok(())
}

#[test]
fn test_data_loading() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let data_path = temp_dir.path().join("data.bin");

    // Create and save tensor
    let data = Tensor::randn(&[100, 784])?;
    data.save(&data_path)?;
    assert!(data_path.exists());

    // Load tensor
    let loaded_data = Tensor::load(&data_path)?;
    assert_eq!(loaded_data.shape(), &[100, 784]);

    Ok(())
}

#[test]
fn test_model_evaluation() -> Result<()> {
    // Create model
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.add_layer(Dense::new(256, 10, Sigmoid::default()));
    model.set_optimizer(Adam::new(0.001));
    model.set_loss(CrossEntropyLoss::default());

    // Create evaluation data
    let x = Tensor::randn(&[100, 784])?;
    let y = Tensor::randn(&[100, 10])?;

    // Evaluate model
    let predictions = model.forward(&x)?;
    let loss = model.compute_loss(&predictions, &y)?;
    assert!(loss >= 0.0);

    // Compute accuracy
    let pred_indices = predictions.argmax(1)?;
    let label_indices = y.argmax(1)?;
    let correct = pred_indices
        .iter()
        .zip(label_indices.iter())
        .filter(|(&p, &l)| p == l)
        .count();
    let accuracy = correct as f32 / x.shape()[0] as f32;
    assert!(accuracy >= 0.0 && accuracy <= 1.0);

    Ok(())
}

#[test]
fn test_error_handling() {
    // Invalid shapes for matrix multiplication
    let a = Tensor::randn(&[32, 32]).unwrap();
    let b = Tensor::randn(&[16, 16]).unwrap();
    assert!(a.matmul(&b).is_err());

    // Invalid axis for reduction
    let x = Tensor::randn(&[32, 32]).unwrap();
    assert!(x.sum(&[2]).is_err());

    // Invalid model configuration
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    assert!(model.forward(&Tensor::randn(&[32, 32]).unwrap()).is_err());
}

#[test]
fn test_serialization() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Test tensor serialization
    let tensor_path = temp_dir.path().join("tensor.bin");
    let x = Tensor::randn(&[32, 32])?;
    x.save(&tensor_path)?;
    let loaded_x = Tensor::load(&tensor_path)?;
    assert_eq!(x.shape(), loaded_x.shape());

    // Test model serialization
    let model_path = temp_dir.path().join("model.bin");
    let mut model = Model::new();
    model.add_layer(Dense::new(784, 256, ReLU::default()));
    model.save(&model_path)?;
    let _loaded_model = Model::load(&model_path)?;

    Ok(())
}
