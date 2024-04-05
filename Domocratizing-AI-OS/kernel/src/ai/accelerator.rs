//! AI accelerator support

use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use super::tensor::Tensor;

/// Accelerator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorType {
    /// CPU
    CPU,
    /// GPU
    GPU,
    /// NPU
    NPU,
}

/// Accelerator capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct AcceleratorCapabilities: u32 {
        /// Supports 32-bit float
        const FLOAT32 = 1 << 0;
        /// Supports 64-bit float
        const FLOAT64 = 1 << 1;
        /// Supports 8-bit integer
        const INT8 = 1 << 2;
        /// Supports 16-bit integer
        const INT16 = 1 << 3;
        /// Supports 32-bit integer
        const INT32 = 1 << 4;
        /// Supports 64-bit integer
        const INT64 = 1 << 5;
        /// Supports tensor operations
        const TENSOR_OPS = 1 << 6;
        /// Supports matrix operations
        const MATRIX_OPS = 1 << 7;
        /// Supports convolution
        const CONVOLUTION = 1 << 8;
        /// Supports pooling
        const POOLING = 1 << 9;
        /// Supports activation functions
        const ACTIVATION = 1 << 10;
        /// Supports normalization
        const NORMALIZATION = 1 << 11;
    }
}

/// Accelerator trait
pub trait Accelerator: Send + Sync {
    /// Get accelerator type
    fn typ(&self) -> AcceleratorType;

    /// Get accelerator capabilities
    fn capabilities(&self) -> AcceleratorCapabilities;

    /// Get memory size
    fn memory_size(&self) -> usize;

    /// Get memory used
    fn memory_used(&self) -> usize;

    /// Run tensor operation
    fn run(&self, input: Arc<Tensor>, output: Arc<Tensor>) -> Result<(), &'static str>;
}

/// CPU accelerator
pub struct CpuAccelerator;

impl CpuAccelerator {
    /// Create new CPU accelerator
    pub const fn new() -> Self {
        CpuAccelerator
    }
}

impl Accelerator for CpuAccelerator {
    fn typ(&self) -> AcceleratorType {
        AcceleratorType::CPU
    }

    fn capabilities(&self) -> AcceleratorCapabilities {
        AcceleratorCapabilities::FLOAT32
            | AcceleratorCapabilities::FLOAT64
            | AcceleratorCapabilities::INT8
            | AcceleratorCapabilities::INT16
            | AcceleratorCapabilities::INT32
            | AcceleratorCapabilities::INT64
            | AcceleratorCapabilities::TENSOR_OPS
            | AcceleratorCapabilities::MATRIX_OPS
    }

    fn memory_size(&self) -> usize {
        // Use system memory
        0x1000_0000 // 256MB
    }

    fn memory_used(&self) -> usize {
        0 // TODO: Track memory usage
    }

    fn run(&self, input: Arc<Tensor>, output: Arc<Tensor>) -> Result<(), &'static str> {
        // Copy input to output
        unsafe {
            output.copy_from_slice(input.as_slice::<u8>());
        }
        Ok(())
    }
}

/// GPU accelerator
pub struct GpuAccelerator {
    /// Memory size
    memory_size: usize,
    /// Memory used
    memory_used: Mutex<usize>,
}

impl GpuAccelerator {
    /// Create new GPU accelerator
    pub const fn new(memory_size: usize) -> Self {
        GpuAccelerator {
            memory_size,
            memory_used: Mutex::new(0),
        }
    }
}

impl Accelerator for GpuAccelerator {
    fn typ(&self) -> AcceleratorType {
        AcceleratorType::GPU
    }

    fn capabilities(&self) -> AcceleratorCapabilities {
        AcceleratorCapabilities::FLOAT32
            | AcceleratorCapabilities::INT8
            | AcceleratorCapabilities::INT16
            | AcceleratorCapabilities::INT32
            | AcceleratorCapabilities::TENSOR_OPS
            | AcceleratorCapabilities::MATRIX_OPS
            | AcceleratorCapabilities::CONVOLUTION
            | AcceleratorCapabilities::POOLING
            | AcceleratorCapabilities::ACTIVATION
            | AcceleratorCapabilities::NORMALIZATION
    }

    fn memory_size(&self) -> usize {
        self.memory_size
    }

    fn memory_used(&self) -> usize {
        *self.memory_used.lock()
    }

    fn run(&self, input: Arc<Tensor>, output: Arc<Tensor>) -> Result<(), &'static str> {
        // TODO: Implement GPU operations
        Err("GPU operations not implemented")
    }
}

/// NPU accelerator
pub struct NpuAccelerator {
    /// Memory size
    memory_size: usize,
    /// Memory used
    memory_used: Mutex<usize>,
}

impl NpuAccelerator {
    /// Create new NPU accelerator
    pub const fn new(memory_size: usize) -> Self {
        NpuAccelerator {
            memory_size,
            memory_used: Mutex::new(0),
        }
    }
}

impl Accelerator for NpuAccelerator {
    fn typ(&self) -> AcceleratorType {
        AcceleratorType::NPU
    }

    fn capabilities(&self) -> AcceleratorCapabilities {
        AcceleratorCapabilities::FLOAT32
            | AcceleratorCapabilities::INT8
            | AcceleratorCapabilities::TENSOR_OPS
            | AcceleratorCapabilities::MATRIX_OPS
            | AcceleratorCapabilities::CONVOLUTION
            | AcceleratorCapabilities::POOLING
            | AcceleratorCapabilities::ACTIVATION
            | AcceleratorCapabilities::NORMALIZATION
    }

    fn memory_size(&self) -> usize {
        self.memory_size
    }

    fn memory_used(&self) -> usize {
        *self.memory_used.lock()
    }

    fn run(&self, input: Arc<Tensor>, output: Arc<Tensor>) -> Result<(), &'static str> {
        // TODO: Implement NPU operations
        Err("NPU operations not implemented")
    }
}

/// Available accelerators
static ACCELERATORS: Mutex<Vec<Arc<dyn Accelerator>>> = Mutex::new(Vec::new());

/// Initialize accelerators
pub fn init() {
    let mut accelerators = ACCELERATORS.lock();

    // Add CPU accelerator
    accelerators.push(Arc::new(CpuAccelerator::new()));

    // Add GPU accelerator if available
    // TODO: Detect GPU
    accelerators.push(Arc::new(GpuAccelerator::new(0x4000_0000))); // 1GB

    // Add NPU accelerator if available
    // TODO: Detect NPU
    accelerators.push(Arc::new(NpuAccelerator::new(0x2000_0000))); // 512MB
}

/// Get available accelerators
pub fn get_accelerators() -> Vec<Arc<dyn Accelerator>> {
    ACCELERATORS.lock().clone()
}

/// Get accelerator by type
pub fn get_accelerator(typ: AcceleratorType) -> Option<Arc<dyn Accelerator>> {
    ACCELERATORS
        .lock()
        .iter()
        .find(|a| a.typ() == typ)
        .map(Arc::clone)
}

/// Get best accelerator for tensor
pub fn get_best_accelerator(tensor: &Tensor) -> Option<Arc<dyn Accelerator>> {
    let accelerators = ACCELERATORS.lock();

    // Find accelerator with most capabilities that supports tensor
    accelerators
        .iter()
        .filter(|a| {
            let caps = a.capabilities();
            match tensor.dtype() {
                super::tensor::DataType::Float32 => caps.contains(AcceleratorCapabilities::FLOAT32),
                super::tensor::DataType::Float64 => caps.contains(AcceleratorCapabilities::FLOAT64),
                super::tensor::DataType::Int8 => caps.contains(AcceleratorCapabilities::INT8),
                super::tensor::DataType::Int16 => caps.contains(AcceleratorCapabilities::INT16),
                super::tensor::DataType::Int32 => caps.contains(AcceleratorCapabilities::INT32),
                super::tensor::DataType::Int64 => caps.contains(AcceleratorCapabilities::INT64),
            }
        })
        .max_by_key(|a| a.capabilities().bits())
        .map(Arc::clone)
}
