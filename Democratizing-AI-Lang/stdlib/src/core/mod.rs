pub mod array;
pub mod error;
pub mod traits;
pub mod types;

pub use array::Array;
pub use error::{Error, Result};
pub use traits::{
    AutoDiff, CloneToDevice, DataTyped, ElementWise, Gradient, IntoTensor, MatrixOps, MemoryLayout,
    NeuralOps, Optimize, Parameter, Random, Reduce, Serialize, Shape, ToDevice,
};
pub use types::{
    Activation, DataType, Device, Init, Layout, Loss, Metric, Numeric, Optimizer, Padding,
    Reduction,
};

/// Re-export common macros
pub use crate::{bail, ensure, error};

/// Re-export common traits from std
pub use std::{
    fmt::{Debug, Display},
    ops::{Add, Deref, DerefMut, Div, Index, IndexMut, Mul, Sub},
};

/// Re-export common types from std
pub use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};

/// Re-export common external crates
pub use {
    anyhow,
    half::f16,
    ndarray,
    num::{self, Complex},
    rand::{self, Rng},
    rayon,
    serde::{self, Deserialize, Serialize as SerdeSerialize},
    thiserror,
};

/// Common type aliases
pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

/// Common constants
pub const PI: f64 = std::f64::consts::PI;
pub const E: f64 = std::f64::consts::E;
pub const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Common functions
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::is_available()
    }
    #[cfg(not(feature = "cuda"))]
    false
}

/// Get the number of available CUDA devices
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::device_count().unwrap_or(0)
    }
    #[cfg(not(feature = "cuda"))]
    0
}

/// Get the current CUDA device
pub fn current_cuda_device() -> Option<usize> {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::current_device().ok()
    }
    #[cfg(not(feature = "cuda"))]
    None
}

/// Set the current CUDA device
pub fn set_cuda_device(device: usize) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::set_device(device)
    }
    #[cfg(not(feature = "cuda"))]
    Err(Error::cuda_error("CUDA support not enabled"))
}

/// Get the number of CPU cores
pub fn cpu_count() -> usize {
    num_cpus::get()
}

/// Get the number of physical CPU cores
pub fn physical_cpu_count() -> usize {
    num_cpus::get_physical()
}

/// Initialize the library
pub fn init() -> Result<()> {
    // Initialize logging
    env_logger::init();

    // Initialize CUDA if available
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::init()?;
    }

    // Set number of threads for parallel operations
    rayon::ThreadPoolBuilder::new()
        .num_threads(physical_cpu_count())
        .build_global()
        .map_err(|e| Error::internal_error(e))?;

    Ok(())
}

/// Shutdown the library
pub fn shutdown() -> Result<()> {
    // Shutdown CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        use crate::cuda;
        cuda::shutdown()?;
    }

    Ok(())
}

/// Set the number of threads for parallel operations
pub fn set_num_threads(threads: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| Error::internal_error(e))
}

/// Get the library version
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get the library authors
pub fn authors() -> &'static str {
    env!("CARGO_PKG_AUTHORS")
}

/// Get the library description
pub fn description() -> &'static str {
    env!("CARGO_PKG_DESCRIPTION")
}

/// Get the library homepage
pub fn homepage() -> &'static str {
    env!("CARGO_PKG_HOMEPAGE")
}

/// Get the library repository
pub fn repository() -> &'static str {
    env!("CARGO_PKG_REPOSITORY")
}

/// Get the library license
pub fn license() -> &'static str {
    env!("CARGO_PKG_LICENSE")
}
