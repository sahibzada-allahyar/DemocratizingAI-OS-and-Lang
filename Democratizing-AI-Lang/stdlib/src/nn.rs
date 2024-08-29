use crate::{
    core::error::Result,
    tensor::{Tensor, TensorOps},
};
use num::traits::Float;

/// Neural network layer trait
pub trait Layer<T: Float> {
    /// Forward pass through the layer
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;

    /// Get layer parameters
    fn parameters(&self) -> Vec<Tensor<T>>;

    /// Update layer parameters with gradients
    fn update_parameters(&mut self, gradients: &[Tensor<T>], learning_rate: T) -> Result<()>;
}

/// Dense (fully connected) layer
pub struct Dense<T: Float> {
    weights: Tensor<T>,
    bias: Tensor<T>,
    activation: Box<dyn Activation<T>>,
}

impl<T: Float> Dense<T> {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: impl Activation<T> + 'static,
    ) -> Self {
        let weights = Tensor::randn(&[input_size, output_size]).unwrap() * T::from(0.01).unwrap();
        let bias = Tensor::zeros(&[1, output_size]).unwrap();
        Self {
            weights,
            bias,
            activation: Box::new(activation),
        }
    }
}

impl<T: Float> Layer<T> for Dense<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let linear = input.matmul(&self.weights)? + &self.bias;
        self.activation.forward(&linear)
    }

    fn parameters(&self) -> Vec<Tensor<T>> {
        vec![self.weights.clone(), self.bias.clone()]
    }

    fn update_parameters(&mut self, gradients: &[Tensor<T>], learning_rate: T) -> Result<()> {
        assert_eq!(gradients.len(), 2);
        self.weights = &self.weights - &(&gradients[0] * learning_rate)?;
        self.bias = &self.bias - &(&gradients[1] * learning_rate)?;
        Ok(())
    }
}

/// Activation function trait
pub trait Activation<T: Float>: std::fmt::Debug {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>>;
    fn backward(&self, grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>>;
}

/// ReLU activation function
#[derive(Debug, Default)]
pub struct ReLU;

impl<T: Float> Activation<T> for ReLU {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.map(|x| x.max(T::zero()))
    }

    fn backward(&self, grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mask = input.map(|&x| if x > T::zero() { T::one() } else { T::zero() })?;
        grad_output.mul(&mask)
    }
}

/// Sigmoid activation function
#[derive(Debug, Default)]
pub struct Sigmoid;

impl<T: Float> Activation<T> for Sigmoid {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.map(|x| T::one() / (T::one() + (-x).exp()))
    }

    fn backward(&self, grad_output: &Tensor<T>, input: &Tensor<T>) -> Result<Tensor<T>> {
        let sigmoid = self.forward(input)?;
        let grad = &sigmoid * &(T::one() - &sigmoid)?;
        grad_output.mul(&grad)
    }
}

/// Loss function trait
pub trait Loss<T: Float>: std::fmt::Debug {
    fn forward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>;
    fn backward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>>;
}

/// Cross entropy loss
#[derive(Debug, Default)]
pub struct CrossEntropyLoss;

impl<T: Float> Loss<T> for CrossEntropyLoss {
    fn forward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>> {
        let epsilon = T::from(1e-7).unwrap();
        let clipped = predictions.map(|&x| x.max(epsilon).min(T::one() - epsilon))?;
        let loss = -(&targets * &clipped.map(|x| x.ln())?)?;
        Ok(loss.mean(&[0, 1])?)
    }

    fn backward(&self, predictions: &Tensor<T>, targets: &Tensor<T>) -> Result<Tensor<T>> {
        let epsilon = T::from(1e-7).unwrap();
        let clipped = predictions.map(|&x| x.max(epsilon).min(T::one() - epsilon))?;
        Ok(-(&targets / &clipped)?)
    }
}

/// Optimizer trait
pub trait Optimizer<T: Float>: std::fmt::Debug {
    fn step(&mut self, model: &mut Model<T>, learning_rate: T) -> Result<()>;
}

/// Adam optimizer
#[derive(Debug)]
pub struct Adam<T: Float> {
    learning_rate: T,
    beta1: T,
    beta2: T,
    epsilon: T,
    m: Vec<Tensor<T>>,
    v: Vec<Tensor<T>>,
    t: usize,
}

impl<T: Float> Adam<T> {
    pub fn new(learning_rate: T, beta1: T, beta2: T, epsilon: T) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl<T: Float> Default for Adam<T> {
    fn default() -> Self {
        Self::new(
            T::from(0.001).unwrap(),
            T::from(0.9).unwrap(),
            T::from(0.999).unwrap(),
            T::from(1e-8).unwrap(),
        )
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    fn step(&mut self, model: &mut Model<T>, learning_rate: T) -> Result<()> {
        self.t += 1;
        let t = T::from(self.t).unwrap();

        // Initialize momentum and velocity if needed
        if self.m.is_empty() {
            for layer in &model.layers {
                for param in layer.parameters() {
                    self.m.push(Tensor::zeros(param.shape())?);
                    self.v.push(Tensor::zeros(param.shape())?);
                }
            }
        }

        let mut i = 0;
        for layer in &mut model.layers {
            for param in layer.parameters() {
                let grad = param.grad().unwrap();

                // Update momentum
                self.m[i] = &(&self.m[i] * self.beta1)? + &(&grad * (T::one() - self.beta1))?;

                // Update velocity
                self.v[i] =
                    &(&self.v[i] * self.beta2)? + &(&(&grad * &grad)? * (T::one() - self.beta2))?;

                // Bias correction
                let m_hat = &self.m[i] / (T::one() - self.beta1.powi(self.t as i32))?;
                let v_hat = &self.v[i] / (T::one() - self.beta2.powi(self.t as i32))?;

                // Update parameters
                let update = &m_hat / &(v_hat.map(|x| x.sqrt() + self.epsilon))?;
                layer.update_parameters(&[&update * learning_rate], learning_rate)?;

                i += 1;
            }
        }
        Ok(())
    }
}

/// Neural network model
#[derive(Debug)]
pub struct Model<T: Float> {
    pub layers: Vec<Box<dyn Layer<T>>>,
    pub optimizer: Option<Box<dyn Optimizer<T>>>,
    pub loss_fn: Box<dyn Loss<T>>,
}

impl<T: Float> Model<T> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            optimizer: None,
            loss_fn: Box::new(CrossEntropyLoss::default()),
        }
    }

    pub fn add_layer(&mut self, layer: impl Layer<T> + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn set_optimizer(&mut self, optimizer: impl Optimizer<T> + 'static) {
        self.optimizer = Some(Box::new(optimizer));
    }

    pub fn set_loss(&mut self, loss: impl Loss<T> + 'static) {
        self.loss_fn = Box::new(loss);
    }

    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    pub fn train_step(&mut self, input: &Tensor<T>, target: &Tensor<T>) -> Result<T> {
        // Forward pass
        let predictions = self.forward(input)?;

        // Compute loss
        let loss = self.loss_fn.forward(&predictions, target)?;

        // Backward pass
        let grad_output = self.loss_fn.backward(&predictions, target)?;
        predictions.backward_with_grad(&grad_output)?;

        // Optimizer step
        if let Some(optimizer) = &mut self.optimizer {
            optimizer.step(self, T::from(0.001).unwrap())?;
        }

        Ok(loss.data()[0])
    }
}

impl<T: Float> Default for Model<T> {
    fn default() -> Self {
        Self::new()
    }
}
