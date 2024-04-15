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

pub mod shell;
pub mod ai;
pub mod monitor;
pub mod network;
pub mod storage;
pub mod debug;

/// Initialize userland
pub fn init() {
    // Initialize shell
    shell::init();

    // Initialize AI
    ai::init();

    // Initialize monitor
    monitor::init();

    // Initialize network
    network::init();

    // Initialize storage
    storage::init();

    // Initialize debug
    debug::init();
}

/// Userland capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct UserlandCapabilities: u32 {
        /// Supports shell
        const SHELL = 1 << 0;
        /// Supports AI
        const AI = 1 << 1;
        /// Supports monitor
        const MONITOR = 1 << 2;
        /// Supports network
        const NETWORK = 1 << 3;
        /// Supports storage
        const STORAGE = 1 << 4;
        /// Supports debug
        const DEBUG = 1 << 5;
        /// Supports graphics
        const GRAPHICS = 1 << 6;
        /// Supports audio
        const AUDIO = 1 << 7;
        /// Supports input
        const INPUT = 1 << 8;
        /// Supports output
        const OUTPUT = 1 << 9;
        /// Supports filesystem
        const FILESYSTEM = 1 << 10;
        /// Supports process
        const PROCESS = 1 << 11;
        /// Supports memory
        const MEMORY = 1 << 12;
        /// Supports device
        const DEVICE = 1 << 13;
        /// Supports security
        const SECURITY = 1 << 14;
        /// Supports system
        const SYSTEM = 1 << 15;
    }
}

impl UserlandCapabilities {
    /// Get all capabilities
    pub fn all() -> Self {
        Self::from_bits_truncate(0xFFFF)
    }
}

/// Userland application
pub trait Application: Send + Sync {
    /// Get application name
    fn name(&self) -> &str;

    /// Get application version
    fn version(&self) -> &str;

    /// Get application capabilities
    fn capabilities(&self) -> UserlandCapabilities;

    /// Start application
    fn start(&self) -> Result<(), &'static str>;

    /// Stop application
    fn stop(&self) -> Result<(), &'static str>;

    /// Restart application
    fn restart(&self) -> Result<(), &'static str>;

    /// Pause application
    fn pause(&self) -> Result<(), &'static str>;

    /// Resume application
    fn resume(&self) -> Result<(), &'static str>;

    /// Update application
    fn update(&self) -> Result<(), &'static str>;

    /// Configure application
    fn configure(&self) -> Result<(), &'static str>;

    /// Debug application
    fn debug(&self) -> Result<(), &'static str>;
}

/// Application manager
pub struct ApplicationManager {
    /// Applications
    applications: alloc::vec::Vec<&'static dyn Application>,
}

impl ApplicationManager {
    /// Create new application manager
    pub const fn new() -> Self {
        ApplicationManager {
            applications: alloc::vec::Vec::new(),
        }
    }

    /// Register application
    pub fn register(&mut self, application: &'static dyn Application) {
        self.applications.push(application);
    }

    /// Unregister application
    pub fn unregister(&mut self, application: &'static dyn Application) {
        if let Some(index) = self.applications.iter().position(|a| *a as *const _ == application as *const _) {
            self.applications.remove(index);
        }
    }

    /// Get application by name
    pub fn get_application(&self, name: &str) -> Option<&'static dyn Application> {
        self.applications.iter().find(|a| a.name() == name).copied()
    }

    /// Get all applications
    pub fn get_applications(&self) -> &[&'static dyn Application] {
        &self.applications
    }
}

/// Global application manager
static APPLICATION_MANAGER: spin::Mutex<ApplicationManager> = spin::Mutex::new(ApplicationManager::new());

/// Register application
pub fn register_application(application: &'static dyn Application) {
    APPLICATION_MANAGER.lock().register(application);
}

/// Unregister application
pub fn unregister_application(application: &'static dyn Application) {
    APPLICATION_MANAGER.lock().unregister(application);
}

/// Get application by name
pub fn get_application(name: &str) -> Option<&'static dyn Application> {
    APPLICATION_MANAGER.lock().get_application(name)
}

/// Get all applications
pub fn get_applications() -> alloc::vec::Vec<&'static dyn Application> {
    APPLICATION_MANAGER.lock().get_applications().to_vec()
}
