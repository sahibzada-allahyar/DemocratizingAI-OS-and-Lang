//! Debug application

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Application, UserlandCapabilities};

/// Debug capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct DebugCapabilities: u32 {
        /// Supports breakpoints
        const BREAKPOINT = 1 << 0;
        /// Supports watchpoints
        const WATCHPOINT = 1 << 1;
        /// Supports single stepping
        const SINGLE_STEP = 1 << 2;
        /// Supports call stack
        const CALL_STACK = 1 << 3;
        /// Supports variable inspection
        const VARIABLES = 1 << 4;
        /// Supports register inspection
        const REGISTERS = 1 << 5;
        /// Supports memory inspection
        const MEMORY = 1 << 6;
        /// Supports thread control
        const THREAD = 1 << 7;
        /// Supports process control
        const PROCESS = 1 << 8;
        /// Supports core dumps
        const CORE_DUMP = 1 << 9;
        /// Supports performance profiling
        const PROFILE = 1 << 10;
        /// Supports tracing
        const TRACE = 1 << 11;
        /// Supports logging
        const LOG = 1 << 12;
        /// Supports assertions
        const ASSERT = 1 << 13;
        /// Supports diagnostics
        const DIAGNOSTIC = 1 << 14;
        /// Supports remote debugging
        const REMOTE = 1 << 15;
    }
}

/// Debug breakpoint
pub struct DebugBreakpoint {
    /// Breakpoint ID
    id: u32,
    /// Breakpoint address
    address: usize,
    /// Breakpoint type
    breakpoint_type: String,
    /// Breakpoint condition
    condition: Option<String>,
    /// Breakpoint hit count
    hit_count: u32,
    /// Breakpoint enabled
    enabled: bool,
}

/// Debug watchpoint
pub struct DebugWatchpoint {
    /// Watchpoint ID
    id: u32,
    /// Watchpoint address
    address: usize,
    /// Watchpoint size
    size: usize,
    /// Watchpoint type
    watchpoint_type: String,
    /// Watchpoint condition
    condition: Option<String>,
    /// Watchpoint hit count
    hit_count: u32,
    /// Watchpoint enabled
    enabled: bool,
}

/// Debug target
pub struct DebugTarget {
    /// Target ID
    id: u32,
    /// Target name
    name: String,
    /// Target type
    target_type: String,
    /// Target state
    state: String,
    /// Target breakpoints
    breakpoints: Vec<DebugBreakpoint>,
    /// Target watchpoints
    watchpoints: Vec<DebugWatchpoint>,
    /// Target capabilities
    capabilities: DebugCapabilities,
}

/// Debug application
pub struct DebugApplication {
    /// Application name
    name: String,
    /// Application version
    version: String,
    /// Application capabilities
    capabilities: UserlandCapabilities,
    /// Debug capabilities
    debug_capabilities: DebugCapabilities,
    /// Debug targets
    targets: Vec<DebugTarget>,
}

impl DebugApplication {
    /// Create new debug application
    pub fn new() -> Self {
        DebugApplication {
            name: String::from("debug"),
            version: String::from("0.1.0"),
            capabilities: UserlandCapabilities::all(),
            debug_capabilities: DebugCapabilities::all(),
            targets: Vec::new(),
        }
    }

    /// Get debug capabilities
    pub fn debug_capabilities(&self) -> DebugCapabilities {
        self.debug_capabilities
    }

    /// Get debug targets
    pub fn targets(&self) -> &[DebugTarget] {
        &self.targets
    }

    /// Add debug target
    pub fn add_target(&mut self, target: DebugTarget) {
        self.targets.push(target);
    }

    /// Remove debug target
    pub fn remove_target(&mut self, id: u32) {
        if let Some(index) = self.targets.iter().position(|t| t.id == id) {
            self.targets.remove(index);
        }
    }

    /// Get debug target by ID
    pub fn get_target(&self, id: u32) -> Option<&DebugTarget> {
        self.targets.iter().find(|t| t.id == id)
    }

    /// Get debug targets by type
    pub fn get_targets_by_type(&self, target_type: &str) -> Vec<&DebugTarget> {
        self.targets
            .iter()
            .filter(|t| t.target_type == target_type)
            .collect()
    }

    /// Get debug targets by state
    pub fn get_targets_by_state(&self, state: &str) -> Vec<&DebugTarget> {
        self.targets
            .iter()
            .filter(|t| t.state == state)
            .collect()
    }
}

impl Application for DebugApplication {
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

/// Global debug application
static DEBUG_APPLICATION: Mutex<Option<Arc<DebugApplication>>> = Mutex::new(None);

/// Initialize debug application
pub fn init() {
    let application = Arc::new(DebugApplication::new());
    *DEBUG_APPLICATION.lock() = Some(Arc::clone(&application));
    crate::register_application(&*application);
}

/// Get debug application
pub fn get_application() -> Option<Arc<DebugApplication>> {
    DEBUG_APPLICATION.lock().as_ref().map(Arc::clone)
}
