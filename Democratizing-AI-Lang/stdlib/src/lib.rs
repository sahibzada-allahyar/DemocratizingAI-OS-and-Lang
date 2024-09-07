#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(unused_imports)]
#![deny(unused_variables)]
#![deny(unused_must_use)]

//! Democratising Standard Library
//!
//! This library provides the core functionality for the Democratising programming language,
//! including tensor operations, neural networks, automatic differentiation, and GPU acceleration.

pub mod autodiff;
pub mod core;
pub mod cuda;
pub mod nn;
pub mod tensor;

#[cfg(feature = "gpu")]
pub mod kernels {
    //! CUDA kernels for GPU acceleration
    pub mod elementwise;
}

// Re-export commonly used types and traits
pub use crate::{
    autodiff::{Context, Node},
    core::{
        error::{Error, Result, ResultExt},
        traits::{
            Broadcast, DeviceTransfer, Differentiable, ElementWise, GradientClone, Initialize,
            IntoTensor, MatrixOps, Reduce, Serialize, ShapeOps, TensorElement,
        },
        types::{
            DataType, Device, GradientMode, InitMethod, KaimingMode, Layout, MemoryFormat,
            ReductionType, Shape,
        },
    },
    nn::{
        activation::{Activation, ReLU, Sigmoid, Tanh},
        loss::{CrossEntropyLoss, MSELoss},
        optimizer::{Adam, SGD},
        Dense, Layer, Loss, Model, Optimizer,
    },
    tensor::Tensor,
};

/// Initialize the library
pub fn init() -> Result<()> {
    // Initialize core module
    core::init()?;

    // Initialize CUDA if available
    #[cfg(feature = "gpu")]
    cuda::init()?;

    Ok(())
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library authors
pub const AUTHORS: &str = env!("CARGO_PKG_AUTHORS");

/// Library description
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Library repository
pub const REPOSITORY: &str = env!("CARGO_PKG_REPOSITORY");

/// Library license
pub const LICENSE: &str = env!("CARGO_PKG_LICENSE");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init() {
        assert!(init().is_ok());
    }

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_metadata() {
        assert!(!AUTHORS.is_empty());
        assert!(!DESCRIPTION.is_empty());
        assert!(!REPOSITORY.is_empty());
        assert!(!LICENSE.is_empty());
    }
}
