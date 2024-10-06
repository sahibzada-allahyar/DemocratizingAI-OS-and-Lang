use std::fmt::Debug;

use super::{
    error::Result,
    types::{DataType, Device, Layout, Numeric},
};

/// Trait for types that can be converted to a device
pub trait ToDevice {
    /// Convert to the specified device
    fn to_device(&self, device: Device) -> Result<Self>
    where
        Self: Sized;

    /// Get the current device
    fn device(&self) -> Device;
}

/// Trait for types that support serialization
pub trait Serialize: Sized {
    /// Serialize to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserialize from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self>;

    /// Save to a file
    fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Load from a file
    fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }
}

/// Trait for types that can be cloned on a specific device
pub trait CloneToDevice {
    /// Clone to the specified device
    fn clone_to_device(&self, device: Device) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support shape operations
pub trait Shape {
    /// Get the shape
    fn shape(&self) -> &[usize];

    /// Get the number of dimensions
    fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements
    fn size(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if empty
    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Reshape into a new shape
    fn reshape(&self, shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Broadcast to a new shape
    fn broadcast_to(&self, shape: &[usize]) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support data type operations
pub trait DataTyped {
    /// Get the data type
    fn dtype(&self) -> DataType;

    /// Convert to a different data type
    fn to_dtype(&self, dtype: DataType) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support memory layout operations
pub trait MemoryLayout {
    /// Get the memory layout
    fn layout(&self) -> Layout;

    /// Convert to a different memory layout
    fn to_layout(&self, layout: Layout) -> Result<Self>
    where
        Self: Sized;

    /// Get the strides
    fn strides(&self) -> &[usize];
}

/// Trait for types that support gradient operations
pub trait Gradient {
    /// Get the gradient
    fn grad(&self) -> Option<&Self>;

    /// Get a mutable reference to the gradient
    fn grad_mut(&mut self) -> Option<&mut Self>;

    /// Set the gradient
    fn set_grad(&mut self, grad: Option<Self>);

    /// Zero the gradient
    fn zero_grad(&mut self);

    /// Backward pass
    fn backward(&mut self) -> Result<()>;
}

/// Trait for types that support automatic differentiation
pub trait AutoDiff: Gradient {
    /// Enable gradient computation
    fn requires_grad(&mut self, requires_grad: bool);

    /// Check if gradient computation is enabled
    fn requires_grad_enabled(&self) -> bool;

    /// Get the gradient function
    fn grad_fn(&self) -> Option<&dyn Fn(&Self) -> Result<Self>>;

    /// Set the gradient function
    fn set_grad_fn(&mut self, grad_fn: Option<Box<dyn Fn(&Self) -> Result<Self>>>);
}

/// Trait for types that support random initialization
pub trait Random {
    /// Initialize with random values
    fn rand(shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Initialize with random normal values
    fn randn(shape: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Initialize with random uniform values
    fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support reduction operations
pub trait Reduce {
    /// Sum along axes
    fn sum(&self, axes: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Mean along axes
    fn mean(&self, axes: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Maximum along axes
    fn max(&self, axes: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Minimum along axes
    fn min(&self, axes: &[usize]) -> Result<Self>
    where
        Self: Sized;

    /// Product along axes
    fn prod(&self, axes: &[usize]) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support element-wise operations
pub trait ElementWise<Rhs = Self> {
    /// Output type
    type Output;

    /// Add element-wise
    fn add(&self, rhs: &Rhs) -> Result<Self::Output>;

    /// Subtract element-wise
    fn sub(&self, rhs: &Rhs) -> Result<Self::Output>;

    /// Multiply element-wise
    fn mul(&self, rhs: &Rhs) -> Result<Self::Output>;

    /// Divide element-wise
    fn div(&self, rhs: &Rhs) -> Result<Self::Output>;

    /// Power element-wise
    fn pow(&self, rhs: &Rhs) -> Result<Self::Output>;
}

/// Trait for types that support matrix operations
pub trait MatrixOps {
    /// Matrix multiplication
    fn matmul(&self, rhs: &Self) -> Result<Self>
    where
        Self: Sized;

    /// Matrix transpose
    fn transpose(&self) -> Result<Self>
    where
        Self: Sized;

    /// Matrix inverse
    fn inverse(&self) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that can be converted to a tensor
pub trait IntoTensor<T: Numeric> {
    /// Convert to a tensor
    fn into_tensor(self) -> Result<crate::tensor::Tensor<T>>;
}

/// Trait for types that support neural network operations
pub trait NeuralOps {
    /// Convolution operation
    fn conv2d(&self, kernel: &Self, stride: usize, padding: usize) -> Result<Self>
    where
        Self: Sized;

    /// Max pooling operation
    fn max_pool2d(&self, kernel_size: usize, stride: usize) -> Result<Self>
    where
        Self: Sized;

    /// Average pooling operation
    fn avg_pool2d(&self, kernel_size: usize, stride: usize) -> Result<Self>
    where
        Self: Sized;

    /// Batch normalization
    fn batch_norm(&self, eps: f32, momentum: f32) -> Result<Self>
    where
        Self: Sized;

    /// Dropout
    fn dropout(&self, p: f32, training: bool) -> Result<Self>
    where
        Self: Sized;
}

/// Trait for types that support optimization
pub trait Optimize {
    /// Update parameters with gradients
    fn step(&mut self) -> Result<()>;

    /// Zero all gradients
    fn zero_grad(&mut self);

    /// Get the current learning rate
    fn learning_rate(&self) -> f32;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f32);
}

/// Trait for types that can be used as model parameters
pub trait Parameter: AutoDiff + Clone + Debug + Send + Sync {
    /// Get the parameter name
    fn name(&self) -> &str;

    /// Set the parameter name
    fn set_name(&mut self, name: &str);

    /// Get whether the parameter is trainable
    fn trainable(&self) -> bool;

    /// Set whether the parameter is trainable
    fn set_trainable(&mut self, trainable: bool);
}
