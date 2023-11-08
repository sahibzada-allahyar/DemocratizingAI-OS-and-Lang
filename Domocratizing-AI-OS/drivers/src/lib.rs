#![no_std]
#![feature(alloc_error_handler)]
#![feature(const_mut_refs)]
#![feature(asm_const)]
#![feature(naked_functions)]
#![feature(core_intrinsics)]
#![feature(panic_info_message)]
#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![feature(slice_ptr_len)]
#![feature(strict_provenance)]
#![feature(ptr_metadata)]
#![feature(pointer_is_aligned)]

extern crate alloc;

pub mod gpu;
pub mod npu;
pub mod network;
pub mod storage;
pub mod usb;
pub mod pci;

/// Initialize drivers
pub fn init() {
    // Initialize PCI
    pci::init();

    // Initialize GPU
    gpu::init();

    // Initialize NPU
    npu::init();

    // Initialize network
    network::init();

    // Initialize storage
    storage::init();

    // Initialize USB
    usb::init();
}

/// Driver capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct DriverCapabilities: u32 {
        /// Driver supports DMA
        const DMA = 1 << 0;
        /// Driver supports interrupts
        const INTERRUPTS = 1 << 1;
        /// Driver supports power management
        const POWER_MANAGEMENT = 1 << 2;
        /// Driver supports hot-plugging
        const HOT_PLUG = 1 << 3;
        /// Driver supports MSI
        const MSI = 1 << 4;
        /// Driver supports MSI-X
        const MSI_X = 1 << 5;
        /// Driver supports SR-IOV
        const SR_IOV = 1 << 6;
        /// Driver supports ATS
        const ATS = 1 << 7;
        /// Driver supports PRI
        const PRI = 1 << 8;
        /// Driver supports PASID
        const PASID = 1 << 9;
        /// Driver supports TPH
        const TPH = 1 << 10;
        /// Driver supports LTR
        const LTR = 1 << 11;
        /// Driver supports OBFF
        const OBFF = 1 << 12;
        /// Driver supports FLR
        const FLR = 1 << 13;
        /// Driver supports VF
        const VF = 1 << 14;
        /// Driver supports AF
        const AF = 1 << 15;
    }
}

/// Driver trait
pub trait Driver: Send + Sync {
    /// Get driver name
    fn name(&self) -> &str;

    /// Get driver version
    fn version(&self) -> &str;

    /// Get driver capabilities
    fn capabilities(&self) -> DriverCapabilities;

    /// Initialize driver
    fn init(&self) -> Result<(), &'static str>;

    /// Probe driver
    fn probe(&self) -> Result<(), &'static str>;

    /// Remove driver
    fn remove(&self) -> Result<(), &'static str>;

    /// Suspend driver
    fn suspend(&self) -> Result<(), &'static str>;

    /// Resume driver
    fn resume(&self) -> Result<(), &'static str>;

    /// Shutdown driver
    fn shutdown(&self) -> Result<(), &'static str>;

    /// Reset driver
    fn reset(&self) -> Result<(), &'static str>;

    /// Get driver status
    fn status(&self) -> Result<(), &'static str>;

    /// Get driver statistics
    fn statistics(&self) -> Result<(), &'static str>;

    /// Get driver debug information
    fn debug(&self) -> Result<(), &'static str>;

    /// Get driver error information
    fn error(&self) -> Result<(), &'static str>;

    /// Get driver interrupt information
    fn interrupt(&self) -> Result<(), &'static str>;

    /// Get driver DMA information
    fn dma(&self) -> Result<(), &'static str>;

    /// Get driver power management information
    fn power(&self) -> Result<(), &'static str>;

    /// Get driver hot-plug information
    fn hotplug(&self) -> Result<(), &'static str>;

    /// Get driver MSI information
    fn msi(&self) -> Result<(), &'static str>;

    /// Get driver MSI-X information
    fn msi_x(&self) -> Result<(), &'static str>;

    /// Get driver SR-IOV information
    fn sr_iov(&self) -> Result<(), &'static str>;

    /// Get driver ATS information
    fn ats(&self) -> Result<(), &'static str>;

    /// Get driver PRI information
    fn pri(&self) -> Result<(), &'static str>;

    /// Get driver PASID information
    fn pasid(&self) -> Result<(), &'static str>;

    /// Get driver TPH information
    fn tph(&self) -> Result<(), &'static str>;

    /// Get driver LTR information
    fn ltr(&self) -> Result<(), &'static str>;

    /// Get driver OBFF information
    fn obff(&self) -> Result<(), &'static str>;

    /// Get driver FLR information
    fn flr(&self) -> Result<(), &'static str>;

    /// Get driver VF information
    fn vf(&self) -> Result<(), &'static str>;

    /// Get driver AF information
    fn af(&self) -> Result<(), &'static str>;
}

/// Driver manager
pub struct DriverManager {
    /// Drivers
    drivers: alloc::vec::Vec<&'static dyn Driver>,
}

impl DriverManager {
    /// Create new driver manager
    pub const fn new() -> Self {
        DriverManager {
            drivers: alloc::vec::Vec::new(),
        }
    }

    /// Register driver
    pub fn register(&mut self, driver: &'static dyn Driver) {
        self.drivers.push(driver);
    }

    /// Unregister driver
    pub fn unregister(&mut self, driver: &'static dyn Driver) {
        if let Some(index) = self.drivers.iter().position(|d| *d as *const _ == driver as *const _) {
            self.drivers.remove(index);
        }
    }

    /// Get driver by name
    pub fn get_driver(&self, name: &str) -> Option<&'static dyn Driver> {
        self.drivers.iter().find(|d| d.name() == name).copied()
    }

    /// Get all drivers
    pub fn get_drivers(&self) -> &[&'static dyn Driver] {
        &self.drivers
    }
}

/// Global driver manager
static DRIVER_MANAGER: spin::Mutex<DriverManager> = spin::Mutex::new(DriverManager::new());

/// Register driver
pub fn register_driver(driver: &'static dyn Driver) {
    DRIVER_MANAGER.lock().register(driver);
}

/// Unregister driver
pub fn unregister_driver(driver: &'static dyn Driver) {
    DRIVER_MANAGER.lock().unregister(driver);
}

/// Get driver by name
pub fn get_driver(name: &str) -> Option<&'static dyn Driver> {
    DRIVER_MANAGER.lock().get_driver(name)
}

/// Get all drivers
pub fn get_drivers() -> alloc::vec::Vec<&'static dyn Driver> {
    DRIVER_MANAGER.lock().get_drivers().to_vec()
}
