use std::fmt;
use thiserror::Error;

/// Result type for stdlib operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for stdlib operations
#[derive(Error, Debug)]
pub enum Error {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },

    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    #[error("Invalid axis: {0}")]
    InvalidAxis(String),

    #[error("Invalid index: {0}")]
    InvalidIndex(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Type error: {0}")]
    TypeError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Neural network error: {0}")]
    NeuralNetworkError(String),

    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    #[error("Loss function error: {0}")]
    LossError(String),

    #[error("Gradient error: {0}")]
    GradientError(String),

    #[error("Data loading error: {0}")]
    DataLoadingError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Internal error: {0}")]
    InternalError(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Create a new shape mismatch error
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Error::ShapeMismatch { expected: expected.to_vec(), actual: actual.to_vec() }
    }

    /// Create a new invalid shape error
    pub fn invalid_shape(msg: impl fmt::Display) -> Self {
        Error::InvalidShape(msg.to_string())
    }

    /// Create a new invalid axis error
    pub fn invalid_axis(msg: impl fmt::Display) -> Self {
        Error::InvalidAxis(msg.to_string())
    }

    /// Create a new invalid index error
    pub fn invalid_index(msg: impl fmt::Display) -> Self {
        Error::InvalidIndex(msg.to_string())
    }

    /// Create a new invalid operation error
    pub fn invalid_operation(msg: impl fmt::Display) -> Self {
        Error::InvalidOperation(msg.to_string())
    }

    /// Create a new CUDA error
    pub fn cuda_error(msg: impl fmt::Display) -> Self {
        Error::CudaError(msg.to_string())
    }

    /// Create a new memory error
    pub fn memory_error(msg: impl fmt::Display) -> Self {
        Error::MemoryError(msg.to_string())
    }

    /// Create a new device error
    pub fn device_error(msg: impl fmt::Display) -> Self {
        Error::DeviceError(msg.to_string())
    }

    /// Create a new type error
    pub fn type_error(msg: impl fmt::Display) -> Self {
        Error::TypeError(msg.to_string())
    }

    /// Create a new serialization error
    pub fn serialization_error(msg: impl fmt::Display) -> Self {
        Error::SerializationError(msg.to_string())
    }

    /// Create a new neural network error
    pub fn neural_network_error(msg: impl fmt::Display) -> Self {
        Error::NeuralNetworkError(msg.to_string())
    }

    /// Create a new optimizer error
    pub fn optimizer_error(msg: impl fmt::Display) -> Self {
        Error::OptimizerError(msg.to_string())
    }

    /// Create a new loss function error
    pub fn loss_error(msg: impl fmt::Display) -> Self {
        Error::LossError(msg.to_string())
    }

    /// Create a new gradient error
    pub fn gradient_error(msg: impl fmt::Display) -> Self {
        Error::GradientError(msg.to_string())
    }

    /// Create a new data loading error
    pub fn data_loading_error(msg: impl fmt::Display) -> Self {
        Error::DataLoadingError(msg.to_string())
    }

    /// Create a new validation error
    pub fn validation_error(msg: impl fmt::Display) -> Self {
        Error::ValidationError(msg.to_string())
    }

    /// Create a new not implemented error
    pub fn not_implemented(msg: impl fmt::Display) -> Self {
        Error::NotImplemented(msg.to_string())
    }

    /// Create a new internal error
    pub fn internal_error(msg: impl fmt::Display) -> Self {
        Error::InternalError(msg.to_string())
    }

    /// Returns true if this is a shape mismatch error
    pub fn is_shape_mismatch(&self) -> bool {
        matches!(self, Error::ShapeMismatch { .. })
    }

    /// Returns true if this is a CUDA error
    pub fn is_cuda_error(&self) -> bool {
        matches!(self, Error::CudaError(_))
    }

    /// Returns true if this is a memory error
    pub fn is_memory_error(&self) -> bool {
        matches!(self, Error::MemoryError(_))
    }

    /// Returns true if this is a device error
    pub fn is_device_error(&self) -> bool {
        matches!(self, Error::DeviceError(_))
    }

    /// Returns true if this is a type error
    pub fn is_type_error(&self) -> bool {
        matches!(self, Error::TypeError(_))
    }

    /// Returns true if this is a validation error
    pub fn is_validation_error(&self) -> bool {
        matches!(self, Error::ValidationError(_))
    }
}

/// Macro for creating a new error
#[macro_export]
macro_rules! error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::core::error::Error::$kind(format!($($arg)*))
    };
}

/// Macro for ensuring a condition is true
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::core::error::Error::$kind(format!($($arg)*)));
        }
    };
}

/// Macro for bailing with an error
#[macro_export]
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::core::error::Error::$kind(format!($($arg)*)));
    };
}
