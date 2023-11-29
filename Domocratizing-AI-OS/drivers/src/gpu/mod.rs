//! GPU driver

use alloc::string::String;
use alloc::sync::Arc;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// GPU capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct GpuCapabilities: u32 {
        /// Supports 2D acceleration
        const ACCEL_2D = 1 << 0;
        /// Supports 3D acceleration
        const ACCEL_3D = 1 << 1;
        /// Supports video acceleration
        const ACCEL_VIDEO = 1 << 2;
        /// Supports compute
        const COMPUTE = 1 << 3;
        /// Supports display
        const DISPLAY = 1 << 4;
        /// Supports multiple displays
        const MULTI_DISPLAY = 1 << 5;
        /// Supports hardware cursor
        const HW_CURSOR = 1 << 6;
        /// Supports page flipping
        const PAGE_FLIP = 1 << 7;
        /// Supports vertical sync
        const VSYNC = 1 << 8;
        /// Supports double buffering
        const DOUBLE_BUFFER = 1 << 9;
        /// Supports triple buffering
        const TRIPLE_BUFFER = 1 << 10;
        /// Supports overlay
        const OVERLAY = 1 << 11;
        /// Supports scaling
        const SCALING = 1 << 12;
        /// Supports rotation
        const ROTATION = 1 << 13;
        /// Supports color correction
        const COLOR_CORRECTION = 1 << 14;
        /// Supports gamma correction
        const GAMMA_CORRECTION = 1 << 15;
    }
}

/// GPU driver
pub struct GpuDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// GPU capabilities
    gpu_capabilities: GpuCapabilities,
}

impl GpuDriver {
    /// Create new GPU driver
    pub fn new() -> Self {
        GpuDriver {
            name: String::from("gpu"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            gpu_capabilities: GpuCapabilities::ACCEL_2D
                | GpuCapabilities::ACCEL_3D
                | GpuCapabilities::COMPUTE,
        }
    }

    /// Get GPU capabilities
    pub fn gpu_capabilities(&self) -> GpuCapabilities {
        self.gpu_capabilities
    }
}

impl Driver for GpuDriver {
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

/// Global GPU driver
static GPU_DRIVER: Mutex<Option<Arc<GpuDriver>>> = Mutex::new(None);

/// Initialize GPU driver
pub fn init() {
    let driver = Arc::new(GpuDriver::new());
    *GPU_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get GPU driver
pub fn get_driver() -> Option<Arc<GpuDriver>> {
    GPU_DRIVER.lock().as_ref().map(Arc::clone)
}
