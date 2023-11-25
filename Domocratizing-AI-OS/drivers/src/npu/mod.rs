//! NPU (Neural Processing Unit) driver

use alloc::string::String;
use alloc::sync::Arc;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// NPU capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct NpuCapabilities: u32 {
        /// Supports 8-bit integer operations
        const INT8 = 1 << 0;
        /// Supports 16-bit integer operations
        const INT16 = 1 << 1;
        /// Supports 32-bit integer operations
        const INT32 = 1 << 2;
        /// Supports 16-bit floating point operations
        const FLOAT16 = 1 << 3;
        /// Supports 32-bit floating point operations
        const FLOAT32 = 1 << 4;
        /// Supports matrix multiplication
        const MATMUL = 1 << 5;
        /// Supports convolution
        const CONV = 1 << 6;
        /// Supports pooling
        const POOL = 1 << 7;
        /// Supports activation functions
        const ACTIVATION = 1 << 8;
        /// Supports normalization
        const NORM = 1 << 9;
        /// Supports tensor operations
        const TENSOR = 1 << 10;
        /// Supports sparse operations
        const SPARSE = 1 << 11;
        /// Supports quantization
        const QUANT = 1 << 12;
        /// Supports pruning
        const PRUNE = 1 << 13;
        /// Supports compression
        const COMPRESS = 1 << 14;
        /// Supports encryption
        const ENCRYPT = 1 << 15;
    }
}

/// NPU driver
pub struct NpuDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// NPU capabilities
    npu_capabilities: NpuCapabilities,
}

impl NpuDriver {
    /// Create new NPU driver
    pub fn new() -> Self {
        NpuDriver {
            name: String::from("npu"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            npu_capabilities: NpuCapabilities::INT8
                | NpuCapabilities::INT16
                | NpuCapabilities::FLOAT16
                | NpuCapabilities::FLOAT32
                | NpuCapabilities::MATMUL
                | NpuCapabilities::CONV
                | NpuCapabilities::POOL
                | NpuCapabilities::ACTIVATION
                | NpuCapabilities::NORM
                | NpuCapabilities::TENSOR,
        }
    }

    /// Get NPU capabilities
    pub fn npu_capabilities(&self) -> NpuCapabilities {
        self.npu_capabilities
    }
}

impl Driver for NpuDriver {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn capabilities(&self) -> DriverCapabilities {
        self.capabilities
    }

    fn init(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn probe(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn remove(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn suspend(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn resume(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn shutdown(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn reset(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn status(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn statistics(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn debug(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn error(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn interrupt(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn dma(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn power(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn hotplug(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn msi(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn msi_x(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn sr_iov(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn ats(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn pri(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn pasid(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn tph(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn ltr(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn obff(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn flr(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn vf(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn af(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Global NPU driver
static NPU_DRIVER: Mutex<Option<Arc<NpuDriver>>> = Mutex::new(None);

/// Initialize NPU driver
pub fn init() {
    let driver = Arc::new(NpuDriver::new());
    *NPU_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get NPU driver
pub fn get_driver() -> Option<Arc<NpuDriver>> {
    NPU_DRIVER.lock().as_ref().map(Arc::clone)
}
