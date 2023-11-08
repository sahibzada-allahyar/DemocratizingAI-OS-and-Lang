//! USB driver

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// USB capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct UsbCapabilities: u32 {
        /// Supports USB 1.1
        const USB_1_1 = 1 << 0;
        /// Supports USB 2.0
        const USB_2_0 = 1 << 1;
        /// Supports USB 3.0
        const USB_3_0 = 1 << 2;
        /// Supports USB 3.1
        const USB_3_1 = 1 << 3;
        /// Supports USB 3.2
        const USB_3_2 = 1 << 4;
        /// Supports USB 4.0
        const USB_4_0 = 1 << 5;
        /// Supports high speed
        const HIGH_SPEED = 1 << 6;
        /// Supports super speed
        const SUPER_SPEED = 1 << 7;
        /// Supports super speed plus
        const SUPER_SPEED_PLUS = 1 << 8;
        /// Supports isochronous transfers
        const ISOCHRONOUS = 1 << 9;
        /// Supports bulk transfers
        const BULK = 1 << 10;
        /// Supports interrupt transfers
        const INTERRUPT = 1 << 11;
        /// Supports control transfers
        const CONTROL = 1 << 12;
        /// Supports power delivery
        const POWER_DELIVERY = 1 << 13;
        /// Supports OTG
        const OTG = 1 << 14;
        /// Supports hub
        const HUB = 1 << 15;
    }
}

/// USB device
pub struct UsbDevice {
    /// Device name
    name: String,
    /// Device vendor ID
    vendor_id: u16,
    /// Device product ID
    product_id: u16,
    /// Device class
    class: u8,
    /// Device subclass
    subclass: u8,
    /// Device protocol
    protocol: u8,
    /// Device capabilities
    capabilities: UsbCapabilities,
    /// Device statistics
    statistics: UsbStatistics,
}

/// USB statistics
#[derive(Debug, Default)]
pub struct UsbStatistics {
    /// Bytes received
    bytes_received: u64,
    /// Bytes transmitted
    bytes_transmitted: u64,
    /// Packets received
    packets_received: u64,
    /// Packets transmitted
    packets_transmitted: u64,
    /// Errors received
    errors_received: u64,
    /// Errors transmitted
    errors_transmitted: u64,
    /// Reset count
    reset_count: u64,
    /// Suspend count
    suspend_count: u64,
    /// Resume count
    resume_count: u64,
}

/// USB driver
pub struct UsbDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// USB devices
    devices: Vec<UsbDevice>,
}

impl UsbDriver {
    /// Create new USB driver
    pub fn new() -> Self {
        UsbDriver {
            name: String::from("usb"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            devices: Vec::new(),
        }
    }

    /// Get USB devices
    pub fn devices(&self) -> &[UsbDevice] {
        &self.devices
    }

    /// Add USB device
    pub fn add_device(&mut self, device: UsbDevice) {
        self.devices.push(device);
    }

    /// Remove USB device
    pub fn remove_device(&mut self, name: &str) {
        if let Some(index) = self.devices.iter().position(|d| d.name == name) {
            self.devices.remove(index);
        }
    }

    /// Get USB device by name
    pub fn get_device(&self, name: &str) -> Option<&UsbDevice> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Get USB device by vendor and product ID
    pub fn get_device_by_id(&self, vendor_id: u16, product_id: u16) -> Option<&UsbDevice> {
        self.devices
            .iter()
            .find(|d| d.vendor_id == vendor_id && d.product_id == product_id)
    }
}

impl Driver for UsbDriver {
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

/// Global USB driver
static USB_DRIVER: Mutex<Option<Arc<UsbDriver>>> = Mutex::new(None);

/// Initialize USB driver
pub fn init() {
    let driver = Arc::new(UsbDriver::new());
    *USB_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get USB driver
pub fn get_driver() -> Option<Arc<UsbDriver>> {
    USB_DRIVER.lock().as_ref().map(Arc::clone)
}
