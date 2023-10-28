//! PCI driver

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// PCI capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct PciCapabilities: u32 {
        /// Supports power management
        const POWER_MANAGEMENT = 1 << 0;
        /// Supports AGP
        const AGP = 1 << 1;
        /// Supports VPD
        const VPD = 1 << 2;
        /// Supports slot identification
        const SLOT_ID = 1 << 3;
        /// Supports MSI
        const MSI = 1 << 4;
        /// Supports PCI-X
        const PCIX = 1 << 5;
        /// Supports HyperTransport
        const HYPERTRANSPORT = 1 << 6;
        /// Supports vendor specific
        const VENDOR = 1 << 7;
        /// Supports debug port
        const DEBUG = 1 << 8;
        /// Supports CompactPCI central resource control
        const CPCI_CENTRAL = 1 << 9;
        /// Supports PCI hot-plug
        const HOTPLUG = 1 << 10;
        /// Supports subsystem vendor ID
        const SUBVENDOR = 1 << 11;
        /// Supports AGP 8x
        const AGP8X = 1 << 12;
        /// Supports secure device
        const SECURE = 1 << 13;
        /// Supports PCI express
        const PCIE = 1 << 14,
        /// Supports MSI-X
        const MSIX = 1 << 15,
        /// Supports SATA
        const SATA = 1 << 16,
        /// Supports advanced features
        const AF = 1 << 17,
        /// Supports enhanced allocation
        const EA = 1 << 18,
        /// Supports flattening portal bridge
        const FPB = 1 << 19,
    }
}

/// PCI device
pub struct PciDevice {
    /// Device name
    name: String,
    /// Vendor ID
    vendor_id: u16,
    /// Device ID
    device_id: u16,
    /// Class code
    class: u8,
    /// Subclass code
    subclass: u8,
    /// Programming interface
    prog_if: u8,
    /// Revision ID
    revision: u8,
    /// Subsystem vendor ID
    subsystem_vendor: u16,
    /// Subsystem ID
    subsystem_id: u16,
    /// Bus number
    bus: u8,
    /// Device number
    device: u8,
    /// Function number
    function: u8,
    /// Device capabilities
    capabilities: PciCapabilities,
    /// Base address registers
    bars: [u32; 6],
    /// Interrupt line
    interrupt_line: u8,
    /// Interrupt pin
    interrupt_pin: u8,
}

impl PciDevice {
    /// Create new PCI device
    pub fn new(
        name: String,
        vendor_id: u16,
        device_id: u16,
        class: u8,
        subclass: u8,
        prog_if: u8,
        revision: u8,
        subsystem_vendor: u16,
        subsystem_id: u16,
        bus: u8,
        device: u8,
        function: u8,
        capabilities: PciCapabilities,
        bars: [u32; 6],
        interrupt_line: u8,
        interrupt_pin: u8,
    ) -> Self {
        PciDevice {
            name,
            vendor_id,
            device_id,
            class,
            subclass,
            prog_if,
            revision,
            subsystem_vendor,
            subsystem_id,
            bus,
            device,
            function,
            capabilities,
            bars,
            interrupt_line,
            interrupt_pin,
        }
    }

    /// Get device name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get vendor ID
    pub fn vendor_id(&self) -> u16 {
        self.vendor_id
    }

    /// Get device ID
    pub fn device_id(&self) -> u16 {
        self.device_id
    }

    /// Get class code
    pub fn class(&self) -> u8 {
        self.class
    }

    /// Get subclass code
    pub fn subclass(&self) -> u8 {
        self.subclass
    }

    /// Get programming interface
    pub fn prog_if(&self) -> u8 {
        self.prog_if
    }

    /// Get revision ID
    pub fn revision(&self) -> u8 {
        self.revision
    }

    /// Get subsystem vendor ID
    pub fn subsystem_vendor(&self) -> u16 {
        self.subsystem_vendor
    }

    /// Get subsystem ID
    pub fn subsystem_id(&self) -> u16 {
        self.subsystem_id
    }

    /// Get bus number
    pub fn bus(&self) -> u8 {
        self.bus
    }

    /// Get device number
    pub fn device(&self) -> u8 {
        self.device
    }

    /// Get function number
    pub fn function(&self) -> u8 {
        self.function
    }

    /// Get device capabilities
    pub fn capabilities(&self) -> PciCapabilities {
        self.capabilities
    }

    /// Get base address registers
    pub fn bars(&self) -> &[u32; 6] {
        &self.bars
    }

    /// Get interrupt line
    pub fn interrupt_line(&self) -> u8 {
        self.interrupt_line
    }

    /// Get interrupt pin
    pub fn interrupt_pin(&self) -> u8 {
        self.interrupt_pin
    }
}

/// PCI driver
pub struct PciDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// PCI devices
    devices: Vec<PciDevice>,
}

impl PciDriver {
    /// Create new PCI driver
    pub fn new() -> Self {
        PciDriver {
            name: String::from("pci"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            devices: Vec::new(),
        }
    }

    /// Get PCI devices
    pub fn devices(&self) -> &[PciDevice] {
        &self.devices
    }

    /// Add PCI device
    pub fn add_device(&mut self, device: PciDevice) {
        self.devices.push(device);
    }

    /// Remove PCI device
    pub fn remove_device(&mut self, name: &str) {
        if let Some(index) = self.devices.iter().position(|d| d.name == name) {
            self.devices.remove(index);
        }
    }

    /// Get PCI device by name
    pub fn get_device(&self, name: &str) -> Option<&PciDevice> {
        self.devices.iter().find(|d| d.name == name)
    }

    /// Get PCI device by vendor and device ID
    pub fn get_device_by_id(&self, vendor_id: u16, device_id: u16) -> Option<&PciDevice> {
        self.devices
            .iter()
            .find(|d| d.vendor_id == vendor_id && d.device_id == device_id)
    }

    /// Get PCI device by bus, device, and function numbers
    pub fn get_device_by_location(&self, bus: u8, device: u8, function: u8) -> Option<&PciDevice> {
        self.devices
            .iter()
            .find(|d| d.bus == bus && d.device == device && d.function == function)
    }
}

impl Driver for PciDriver {
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

/// Global PCI driver
static PCI_DRIVER: Mutex<Option<Arc<PciDriver>>> = Mutex::new(None);

/// Initialize PCI driver
pub fn init() {
    let driver = Arc::new(PciDriver::new());
    *PCI_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get PCI driver
pub fn get_driver() -> Option<Arc<PciDriver>> {
    PCI_DRIVER.lock().as_ref().map(Arc::clone)
}
