//! Device service

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Service, ServiceCapabilities};

/// Device capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct DeviceCapabilities: u32 {
        /// Supports character devices
        const CHAR = 1 << 0;
        /// Supports block devices
        const BLOCK = 1 << 1;
        /// Supports network devices
        const NETWORK = 1 << 2;
        /// Supports USB devices
        const USB = 1 << 3;
        /// Supports PCI devices
        const PCI = 1 << 4;
        /// Supports SCSI devices
        const SCSI = 1 << 5;
        /// Supports IDE devices
        const IDE = 1 << 6;
        /// Supports SATA devices
        const SATA = 1 << 7;
        /// Supports NVMe devices
        const NVME = 1 << 8;
        /// Supports GPU devices
        const GPU = 1 << 9;
        /// Supports NPU devices
        const NPU = 1 << 10;
        /// Supports input devices
        const INPUT = 1 << 11;
        /// Supports output devices
        const OUTPUT = 1 << 12;
        /// Supports storage devices
        const STORAGE = 1 << 13;
        /// Supports audio devices
        const AUDIO = 1 << 14;
        /// Supports video devices
        const VIDEO = 1 << 15;
    }
}

/// Device class
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// Character device
    Char,
    /// Block device
    Block,
    /// Network device
    Network,
    /// USB device
    Usb,
    /// PCI device
    Pci,
    /// SCSI device
    Scsi,
    /// IDE device
    Ide,
    /// SATA device
    Sata,
    /// NVMe device
    Nvme,
    /// GPU device
    Gpu,
    /// NPU device
    Npu,
    /// Input device
    Input,
    /// Output device
    Output,
    /// Storage device
    Storage,
    /// Audio device
    Audio,
    /// Video device
    Video,
}

/// Device
pub struct Device {
    /// Device name
    name: String,
    /// Device class
    class: DeviceClass,
    /// Device vendor ID
    vendor_id: u16,
    /// Device product ID
    product_id: u16,
    /// Device capabilities
    capabilities: DeviceCapabilities,
    /// Device driver
    driver: Option<String>,
}

/// Device service
pub struct DeviceService {
    /// Service name
    name: String,
    /// Service version
    version: String,
    /// Service capabilities
    capabilities: ServiceCapabilities,
    /// Device capabilities
    dev_capabilities: DeviceCapabilities,
    /// Devices
    devices: Vec<Device>,
}

impl DeviceService {
    /// Create new device service
    pub fn new() -> Self {
        DeviceService {
            name: String::from("device"),
            version: String::from("0.1.0"),
            capabilities: ServiceCapabilities::all(),
            dev_capabilities: DeviceCapabilities::all(),
            devices: Vec::new(),
        }
    }

    /// Get device capabilities
    pub fn dev_capabilities(&self) -> DeviceCapabilities {
        self.dev_capabilities
    }

    /// Get devices
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    /// Add device
    pub fn add_device(&mut self, device: Device) {
        self.devices.push(device);
    }

    /// Remove device
    pub fn remove_device(&mut self, name: &str) {
        if let Some(index) = self.devices.iter().position(|d| d.name == name) {
            self.devices.remove(index);
        }
    }

    /// Get device by name
    pub fn get_device(&self, name: &str) -> Option<&Device> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Get device by class
    pub fn get_devices_by_class(&self, class: DeviceClass) -> Vec<&Device> {
        self.devices.iter().filter(|d| d.class == class).collect()
    }

    /// Get device by vendor and product ID
    pub fn get_device_by_id(&self, vendor_id: u16, product_id: u16) -> Option<&Device> {
        self.devices
            .iter()
            .find(|d| d.vendor_id == vendor_id && d.product_id == product_id)
    }
}

impl Service for DeviceService {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn capabilities(&self) -> ServiceCapabilities {
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

    fn reload(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn enable(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn disable(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn mask(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn unmask(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn isolate(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn monitor(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn log(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn configure(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn secure(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn debug(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Global device service
static DEVICE_SERVICE: Mutex<Option<Arc<DeviceService>>> = Mutex::new(None);

/// Initialize device service
pub fn init() {
    let service = Arc::new(DeviceService::new());
    *DEVICE_SERVICE.lock() = Some(Arc::clone(&service));
    crate::register_service(&*service);
}

/// Get device service
pub fn get_service() -> Option<Arc<DeviceService>> {
    DEVICE_SERVICE.lock().as_ref().map(Arc::clone)
}
