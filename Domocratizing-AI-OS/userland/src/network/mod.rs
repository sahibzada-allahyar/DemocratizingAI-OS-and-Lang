//! Network application

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Application, UserlandCapabilities};

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

/// Network application
pub struct NetworkApplication {
    /// Application name
    name: String,
    /// Application version
    version: String,
    /// Application capabilities
    capabilities: UserlandCapabilities,
    /// Network capabilities
    net_capabilities: NetworkCapabilities,
    /// Network interfaces
    interfaces: Vec<NetworkInterface>,
}

impl NetworkApplication {
    /// Create new network application
    pub fn new() -> Self {
        NetworkApplication {
            name: String::from("network"),
            version: String::from("0.1.0"),
            capabilities: UserlandCapabilities::all(),
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

impl Application for NetworkApplication {
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

/// Global network application
static NETWORK_APPLICATION: Mutex<Option<Arc<NetworkApplication>>> = Mutex::new(None);

/// Initialize network application
pub fn init() {
    let application = Arc::new(NetworkApplication::new());
    *NETWORK_APPLICATION.lock() = Some(Arc::clone(&application));
    crate::register_application(&*application);
}

/// Get network application
pub fn get_application() -> Option<Arc<NetworkApplication>> {
    NETWORK_APPLICATION.lock().as_ref().map(Arc::clone)
}
