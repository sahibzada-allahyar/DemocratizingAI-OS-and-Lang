//! Storage driver

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// Storage capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct StorageCapabilities: u32 {
        /// Supports read
        const READ = 1 << 0;
        /// Supports write
        const WRITE = 1 << 1;
        /// Supports trim
        const TRIM = 1 << 2;
        /// Supports flush
        const FLUSH = 1 << 3;
        /// Supports secure erase
        const SECURE_ERASE = 1 << 4;
        /// Supports NCQ
        const NCQ = 1 << 5;
        /// Supports SMART
        const SMART = 1 << 6;
        /// Supports power management
        const POWER_MANAGEMENT = 1 << 7;
        /// Supports write cache
        const WRITE_CACHE = 1 << 8;
        /// Supports read cache
        const READ_CACHE = 1 << 9;
        /// Supports DMA
        const DMA = 1 << 10;
        /// Supports 48-bit LBA
        const LBA48 = 1 << 11;
        /// Supports command queuing
        const COMMAND_QUEUING = 1 << 12;
        /// Supports SATA
        const SATA = 1 << 13;
        /// Supports NVMe
        const NVME = 1 << 14;
        /// Supports SCSI
        const SCSI = 1 << 15;
    }
}

/// Storage device
pub struct StorageDevice {
    /// Device name
    name: String,
    /// Device model
    model: String,
    /// Device serial number
    serial: String,
    /// Device firmware version
    firmware: String,
    /// Device capacity
    capacity: u64,
    /// Device sector size
    sector_size: usize,
    /// Device capabilities
    capabilities: StorageCapabilities,
    /// Device statistics
    statistics: StorageStatistics,
}

/// Storage statistics
#[derive(Debug, Default)]
pub struct StorageStatistics {
    /// Bytes read
    bytes_read: u64,
    /// Bytes written
    bytes_written: u64,
    /// Read operations
    read_ops: u64,
    /// Write operations
    write_ops: u64,
    /// Read errors
    read_errors: u64,
    /// Write errors
    write_errors: u64,
    /// Power on time
    power_on_time: u64,
    /// Temperature
    temperature: i32,
}

/// Storage driver
pub struct StorageDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// Storage devices
    devices: Vec<StorageDevice>,
}

impl StorageDriver {
    /// Create new storage driver
    pub fn new() -> Self {
        StorageDriver {
            name: String::from("storage"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            devices: Vec::new(),
        }
    }

    /// Get storage devices
    pub fn devices(&self) -> &[StorageDevice] {
        &self.devices
    }

    /// Add storage device
    pub fn add_device(&mut self, device: StorageDevice) {
        self.devices.push(device);
    }

    /// Remove storage device
    pub fn remove_device(&mut self, name: &str) {
        if let Some(index) = self.devices.iter().position(|d| d.name == name) {
            self.devices.remove(index);
        }
    }

    /// Get storage device by name
    pub fn get_device(&self, name: &str) -> Option<&StorageDevice> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Get storage device by serial number
    pub fn get_device_by_serial(&self, serial: &str) -> Option<&StorageDevice> {
        self.devices.iter().find(|d| d.serial == serial)
    }
}

impl Driver for StorageDriver {
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

/// Global storage driver
static STORAGE_DRIVER: Mutex<Option<Arc<StorageDriver>>> = Mutex::new(None);

/// Initialize storage driver
pub fn init() {
    let driver = Arc::new(StorageDriver::new());
    *STORAGE_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get storage driver
pub fn get_driver() -> Option<Arc<StorageDriver>> {
    STORAGE_DRIVER.lock().as_ref().map(Arc::clone)
}
