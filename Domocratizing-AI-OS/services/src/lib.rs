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

pub mod fs;
pub mod network;
pub mod device;
pub mod process;
pub mod security;
pub mod ai;

/// Initialize services
pub fn init() {
    // Initialize file system service
    fs::init();

    // Initialize network service
    network::init();

    // Initialize device service
    device::init();

    // Initialize process service
    process::init();

    // Initialize security service
    security::init();

    // Initialize AI service
    ai::init();
}

/// Service capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct ServiceCapabilities: u32 {
        /// Service can be started
        const START = 1 << 0;
        /// Service can be stopped
        const STOP = 1 << 1;
        /// Service can be restarted
        const RESTART = 1 << 2;
        /// Service can be paused
        const PAUSE = 1 << 3;
        /// Service can be resumed
        const RESUME = 1 << 4;
        /// Service can be reloaded
        const RELOAD = 1 << 5;
        /// Service can be enabled
        const ENABLE = 1 << 6;
        /// Service can be disabled
        const DISABLE = 1 << 7;
        /// Service can be masked
        const MASK = 1 << 8;
        /// Service can be unmasked
        const UNMASK = 1 << 9;
        /// Service can be isolated
        const ISOLATE = 1 << 10;
        /// Service can be monitored
        const MONITOR = 1 << 11;
        /// Service can be logged
        const LOG = 1 << 12;
        /// Service can be configured
        const CONFIG = 1 << 13;
        /// Service can be secured
        const SECURE = 1 << 14;
        /// Service can be debugged
        const DEBUG = 1 << 15;
    }
}

impl ServiceCapabilities {
    /// Get all capabilities
    pub fn all() -> Self {
        Self::from_bits_truncate(0xFFFF)
    }
}

/// Service trait
pub trait Service: Send + Sync {
    /// Get service name
    fn name(&self) -> &str;

    /// Get service version
    fn version(&self) -> &str;

    /// Get service capabilities
    fn capabilities(&self) -> ServiceCapabilities;

    /// Start service
    fn start(&self) -> Result<(), &'static str>;

    /// Stop service
    fn stop(&self) -> Result<(), &'static str>;

    /// Restart service
    fn restart(&self) -> Result<(), &'static str>;

    /// Pause service
    fn pause(&self) -> Result<(), &'static str>;

    /// Resume service
    fn resume(&self) -> Result<(), &'static str>;

    /// Reload service
    fn reload(&self) -> Result<(), &'static str>;

    /// Enable service
    fn enable(&self) -> Result<(), &'static str>;

    /// Disable service
    fn disable(&self) -> Result<(), &'static str>;

    /// Mask service
    fn mask(&self) -> Result<(), &'static str>;

    /// Unmask service
    fn unmask(&self) -> Result<(), &'static str>;

    /// Isolate service
    fn isolate(&self) -> Result<(), &'static str>;

    /// Monitor service
    fn monitor(&self) -> Result<(), &'static str>;

    /// Log service
    fn log(&self) -> Result<(), &'static str>;

    /// Configure service
    fn configure(&self) -> Result<(), &'static str>;

    /// Secure service
    fn secure(&self) -> Result<(), &'static str>;

    /// Debug service
    fn debug(&self) -> Result<(), &'static str>;
}

/// Service manager
pub struct ServiceManager {
    /// Services
    services: alloc::vec::Vec<&'static dyn Service>,
}

impl ServiceManager {
    /// Create new service manager
    pub const fn new() -> Self {
        ServiceManager {
            services: alloc::vec::Vec::new(),
        }
    }

    /// Register service
    pub fn register(&mut self, service: &'static dyn Service) {
        self.services.push(service);
    }

    /// Unregister service
    pub fn unregister(&mut self, service: &'static dyn Service) {
        if let Some(index) = self.services.iter().position(|s| *s as *const _ == service as *const _) {
            self.services.remove(index);
        }
    }

    /// Get service by name
    pub fn get_service(&self, name: &str) -> Option<&'static dyn Service> {
        self.services.iter().find(|s| s.name() == name).copied()
    }

    /// Get all services
    pub fn get_services(&self) -> &[&'static dyn Service] {
        &self.services
    }
}

/// Global service manager
static SERVICE_MANAGER: spin::Mutex<ServiceManager> = spin::Mutex::new(ServiceManager::new());

/// Register service
pub fn register_service(service: &'static dyn Service) {
    SERVICE_MANAGER.lock().register(service);
}

/// Unregister service
pub fn unregister_service(service: &'static dyn Service) {
    SERVICE_MANAGER.lock().unregister(service);
}

/// Get service by name
pub fn get_service(name: &str) -> Option<&'static dyn Service> {
    SERVICE_MANAGER.lock().get_service(name)
}

/// Get all services
pub fn get_services() -> alloc::vec::Vec<&'static dyn Service> {
    SERVICE_MANAGER.lock().get_services().to_vec()
}
