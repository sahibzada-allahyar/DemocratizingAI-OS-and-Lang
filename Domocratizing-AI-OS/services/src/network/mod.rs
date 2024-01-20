//! Network service

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Service, ServiceCapabilities};

/// Network capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct NetworkCapabilities: u32 {
        /// Supports IPv4
        const IPV4 = 1 << 0;
        /// Supports IPv6
        const IPV6 = 1 << 1;
        /// Supports TCP
        const TCP = 1 << 2;
        /// Supports UDP
        const UDP = 1 << 3;
        /// Supports ICMP
        const ICMP = 1 << 4;
        /// Supports raw sockets
        const RAW = 1 << 5;
        /// Supports packet sockets
        const PACKET = 1 << 6;
        /// Supports multicast
        const MULTICAST = 1 << 7;
        /// Supports broadcast
        const BROADCAST = 1 << 8;
        /// Supports promiscuous mode
        const PROMISCUOUS = 1 << 9;
        /// Supports VLAN
        const VLAN = 1 << 10;
        /// Supports bridging
        const BRIDGE = 1 << 11;
        /// Supports routing
        const ROUTE = 1 << 12;
        /// Supports firewall
        const FIREWALL = 1 << 13;
        /// Supports NAT
        const NAT = 1 << 14;
        /// Supports QoS
        const QOS = 1 << 15;
    }
}

/// Network interface
pub struct NetworkInterface {
    /// Interface name
    name: String,
    /// Interface index
    index: u32,
    /// Interface type
    if_type: String,
    /// Interface flags
    flags: u32,
    /// Interface MTU
    mtu: u32,
    /// Interface MAC address
    mac: [u8; 6],
    /// Interface IPv4 addresses
    ipv4: Vec<Ipv4Address>,
    /// Interface IPv6 addresses
    ipv6: Vec<Ipv6Address>,
    /// Interface capabilities
    capabilities: NetworkCapabilities,
}

/// IPv4 address
pub struct Ipv4Address {
    /// Address
    address: [u8; 4],
    /// Netmask
    netmask: [u8; 4],
    /// Broadcast
    broadcast: [u8; 4],
}

/// IPv6 address
pub struct Ipv6Address {
    /// Address
    address: [u8; 16],
    /// Prefix length
    prefix_len: u8,
}

/// Network service
pub struct NetworkService {
    /// Service name
    name: String,
    /// Service version
    version: String,
    /// Service capabilities
    capabilities: ServiceCapabilities,
    /// Network capabilities
    net_capabilities: NetworkCapabilities,
    /// Network interfaces
    interfaces: Vec<NetworkInterface>,
}

impl NetworkService {
    /// Create new network service
    pub fn new() -> Self {
        NetworkService {
            name: String::from("network"),
            version: String::from("0.1.0"),
            capabilities: ServiceCapabilities::all(),
            net_capabilities: NetworkCapabilities::all(),
            interfaces: Vec::new(),
        }
    }

    /// Get network capabilities
    pub fn net_capabilities(&self) -> NetworkCapabilities {
        self.net_capabilities
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

    /// Get network interface by index
    pub fn get_interface_by_index(&self, index: u32) -> Option<&NetworkInterface> {
        self.interfaces.iter().find(|i| i.index == index)
    }
}

impl Service for NetworkService {
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

/// Global network service
static NETWORK_SERVICE: Mutex<Option<Arc<NetworkService>>> = Mutex::new(None);

/// Initialize network service
pub fn init() {
    let service = Arc::new(NetworkService::new());
    *NETWORK_SERVICE.lock() = Some(Arc::clone(&service));
    crate::register_service(&*service);
}

/// Get network service
pub fn get_service() -> Option<Arc<NetworkService>> {
    NETWORK_SERVICE.lock().as_ref().map(Arc::clone)
}
