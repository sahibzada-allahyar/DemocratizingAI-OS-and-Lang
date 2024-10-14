use std::fmt;
use std::ops::{Add, Div, Mul, Sub};

/// Device type for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// CPU device
    CPU,
    /// GPU device with device ID
    GPU(usize),
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::CPU => write!(f, "cpu"),
            Device::GPU(id) => write!(f, "cuda:{}", id),
        }
    }
}

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// 32-bit integer
    Int32,
    /// 64-bit integer
    Int64,
    /// 16-bit floating point
    Float16,
    /// Boolean
    Bool,
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float32 => write!(f, "float32"),
            DataType::Float64 => write!(f, "float64"),
            DataType::Int32 => write!(f, "int32"),
            DataType::Int64 => write!(f, "int64"),
            DataType::Float16 => write!(f, "float16"),
            DataType::Bool => write!(f, "bool"),
        }
    }
}

/// Memory layout for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

/// Numeric trait for types that support arithmetic operations
pub trait Numeric:
    Sized
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    /// Zero value
    fn zero() -> Self;
    /// One value
    fn one() -> Self;
    /// Convert from f32
    fn from_f32(value: f32) -> Self;
    /// Convert to f32
    fn to_f32(self) -> f32;
    /// Convert from f64
    fn from_f64(value: f64) -> Self;
    /// Convert to f64
    fn to_f64(self) -> f64;
}

macro_rules! impl_numeric {
    ($type:ty, $zero:expr, $one:expr) => {
        impl Numeric for $type {
            #[inline]
            fn zero() -> Self {
                $zero
            }

            #[inline]
            fn one() -> Self {
                $one
            }

            #[inline]
            fn from_f32(value: f32) -> Self {
                value as Self
            }

            #[inline]
            fn to_f32(self) -> f32 {
                self as f32
            }

            #[inline]
            fn from_f64(value: f64) -> Self {
                value as Self
            }

            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}

impl_numeric!(f32, 0.0f32, 1.0f32);
impl_numeric!(f64, 0.0f64, 1.0f64);
impl_numeric!(i32, 0i32, 1i32);
impl_numeric!(i64, 0i64, 1i64);

/// Reduction operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Sum reduction
    Sum,
    /// Mean reduction
    Mean,
    /// Maximum reduction
    Max,
    /// Minimum reduction
    Min,
    /// Product reduction
    Prod,
}

/// Padding type for convolution operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// No padding
    Valid,
    /// Pad to maintain input size
    Same,
    /// Custom padding
    Custom(usize),
}

/// Initialization type for weights
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Init {
    /// Zero initialization
    Zeros,
    /// One initialization
    Ones,
    /// Constant initialization
    Constant(f32),
    /// Uniform random initialization
    Uniform { low: f32, high: f32 },
    /// Normal random initialization
    Normal { mean: f32, std: f32 },
    /// Xavier/Glorot initialization
    Xavier,
    /// Kaiming/He initialization
    Kaiming,
}

/// Activation function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    /// Linear activation (identity)
    Linear,
    /// ReLU activation
    ReLU,
    /// Leaky ReLU activation
    LeakyReLU(f32),
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax activation
    Softmax,
}

/// Loss function type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Loss {
    /// Mean squared error
    MSE,
    /// Cross entropy
    CrossEntropy,
    /// Binary cross entropy
    BCE,
    /// Hinge loss
    Hinge,
    /// Huber loss
    Huber(f32),
}

/// Optimizer type
#[derive(Debug, Clone, PartialEq)]
pub enum Optimizer {
    /// Stochastic gradient descent
    SGD {
        learning_rate: f32,
        momentum: Option<f32>,
        nesterov: bool,
    },
    /// Adam optimizer
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
    },
}

/// Metric type for evaluation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    /// Accuracy metric
    Accuracy,
    /// Precision metric
    Precision,
    /// Recall metric
    Recall,
    /// F1 score
    F1Score,
    /// Area under ROC curve
    AUC,
    /// Mean absolute error
    MAE,
    /// Root mean squared error
    RMSE,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_display() {
        assert_eq!(Device::CPU.to_string(), "cpu");
        assert_eq!(Device::GPU(0).to_string(), "cuda:0");
    }

    #[test]
    fn test_data_type_display() {
        assert_eq!(DataType::Float32.to_string(), "float32");
        assert_eq!(DataType::Int64.to_string(), "int64");
    }

    #[test]
    fn test_numeric_trait() {
        assert_eq!(f32::zero(), 0.0);
        assert_eq!(f32::one(), 1.0);
        assert_eq!(f32::from_f64(2.5), 2.5);
        assert_eq!(i32::from_f32(3.7).to_f32(), 3.0);
    }
}
