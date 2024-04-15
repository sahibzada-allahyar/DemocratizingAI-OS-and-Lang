//! Storage application

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Application, UserlandCapabilities};

/// Storage capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct StorageCapabilities: u32 {
        /// Supports read operations
        const READ = 1 << 0;
        /// Supports write operations
        const WRITE = 1 << 1;
        /// Supports trim operations
        const TRIM = 1 << 2;
        /// Supports flush operations
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

/// Storage application
pub struct StorageApplication {
    /// Application name
    name: String,
    /// Application version
    version: String,
    /// Application capabilities
    capabilities: UserlandCapabilities,
    /// Storage capabilities
    storage_capabilities: StorageCapabilities,
    /// Storage devices
    devices: Vec<StorageDevice>,
}

impl StorageApplication {
    /// Create new storage application
    pub fn new() -> Self {
        StorageApplication {
            name: String::from("storage"),
            version: String::from("0.1.0"),
            capabilities: UserlandCapabilities::all(),
            storage_capabilities: StorageCapabilities::all(),
            devices: Vec::new(),
        }
    }

    /// Get storage capabilities
    pub fn storage_capabilities(&self) -> StorageCapabilities {
        self.storage_capabilities
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

impl Application for StorageApplication {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn capabilities(&self) -> UserlandCapabilities {
        self.capabilities
    }

    fn start(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn stop(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn restart(&self) -> Result<(), &'static str> {
        self.stop()?;
        self.start()
    }

    fn pause(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn resume(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn update(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn configure(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn debug(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Global storage application
static STORAGE_APPLICATION: Mutex<Option<Arc<StorageApplication>>> = Mutex::new(None);

/// Initialize storage application
pub fn init() {
    let application = Arc::new(StorageApplication::new());
    *STORAGE_APPLICATION.lock() = Some(Arc::clone(&application));
    crate::register_application(&*application);
}

/// Get storage application
pub fn get_application() -> Option<Arc<StorageApplication>> {
    STORAGE_APPLICATION.lock().as_ref().map(Arc::clone)
}
