//! Network driver

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Driver, DriverCapabilities};

/// Network capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct NetworkCapabilities: u32 {
        /// Supports 10 Mbps
        const SPEED_10 = 1 << 0;
        /// Supports 100 Mbps
        const SPEED_100 = 1 << 1;
        /// Supports 1000 Mbps
        const SPEED_1000 = 1 << 2;
        /// Supports 10000 Mbps
        const SPEED_10000 = 1 << 3;
        /// Supports full duplex
        const FULL_DUPLEX = 1 << 4;
        /// Supports half duplex
        const HALF_DUPLEX = 1 << 5;
        /// Supports auto negotiation
        const AUTO_NEGOTIATE = 1 << 6;
        /// Supports flow control
        const FLOW_CONTROL = 1 << 7;
        /// Supports jumbo frames
        const JUMBO_FRAMES = 1 << 8;
        /// Supports VLAN
        const VLAN = 1 << 9;
        /// Supports TSO
        const TSO = 1 << 10;
        /// Supports RSS
        const RSS = 1 << 11;
        /// Supports checksum offload
        const CHECKSUM = 1 << 12;
        /// Supports scatter-gather
        const SCATTER_GATHER = 1 << 13;
        /// Supports TCP/IP offload
        const TCP_IP = 1 << 14;
        /// Supports UDP/IP offload
        const UDP_IP = 1 << 15;
    }
}

/// Network interface
pub struct NetworkInterface {
    /// Interface name
    name: String,
    /// MAC address
    mac: [u8; 6],
    /// MTU
    mtu: usize,
    /// Link speed
    speed: usize,
    /// Link duplex
    duplex: bool,
    /// Link state
    link: bool,
    /// Interface capabilities
    capabilities: NetworkCapabilities,
    /// Interface statistics
    statistics: NetworkStatistics,
}

/// Network statistics
#[derive(Debug, Default)]
pub struct NetworkStatistics {
    /// Bytes received
    rx_bytes: usize,
    /// Packets received
    rx_packets: usize,
    /// Errors received
    rx_errors: usize,
    /// Drops received
    rx_drops: usize,
    /// Bytes transmitted
    tx_bytes: usize,
    /// Packets transmitted
    tx_packets: usize,
    /// Errors transmitted
    tx_errors: usize,
    /// Drops transmitted
    tx_drops: usize,
}

/// Network driver
pub struct NetworkDriver {
    /// Driver name
    name: String,
    /// Driver version
    version: String,
    /// Driver capabilities
    capabilities: DriverCapabilities,
    /// Network interfaces
    interfaces: Vec<NetworkInterface>,
}

impl NetworkDriver {
    /// Create new network driver
    pub fn new() -> Self {
        NetworkDriver {
            name: String::from("network"),
            version: String::from("0.1.0"),
            capabilities: DriverCapabilities::DMA | DriverCapabilities::INTERRUPTS,
            interfaces: Vec::new(),
        }
    }

    /// Get network interfaces
    pub fn interfaces(&self) -> &[NetworkInterface] {
        &self.interfaces
    }

    /// Add network interface
    pub fn add_interface(&mut self, interface: NetworkInterface) {
        self.interfaces.push(interface);
    }

    /// Remove network interface
    pub fn remove_interface(&mut self, name: &str) {
        if let Some(index) = self.interfaces.iter().position(|i| i.name == name) {
            self.interfaces.remove(index);
        }
    }

    /// Get network interface by name
    pub fn get_interface(&self, name: &str) -> Option<&NetworkInterface> {
        self.interfaces.iter().find(|i| i.name == name)
    }

    /// Get network interface by MAC address
    pub fn get_interface_by_mac(&self, mac: &[u8; 6]) -> Option<&NetworkInterface> {
        self.interfaces.iter().find(|i| i.mac == *mac)
    }
}

impl Driver for NetworkDriver {
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

/// Global network driver
static NETWORK_DRIVER: Mutex<Option<Arc<NetworkDriver>>> = Mutex::new(None);

/// Initialize network driver
pub fn init() {
    let driver = Arc::new(NetworkDriver::new());
    *NETWORK_DRIVER.lock() = Some(Arc::clone(&driver));
    crate::register_driver(&*driver);
}

/// Get network driver
pub fn get_driver() -> Option<Arc<NetworkDriver>> {
    NETWORK_DRIVER.lock().as_ref().map(Arc::clone)
}
