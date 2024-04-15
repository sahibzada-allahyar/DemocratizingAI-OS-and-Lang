//! Monitor application

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Application, UserlandCapabilities};

/// Monitor capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct MonitorCapabilities: u32 {
        /// Supports process monitoring
        const PROCESS = 1 << 0;
        /// Supports thread monitoring
        const THREAD = 1 << 1;
        /// Supports memory monitoring
        const MEMORY = 1 << 2;
        /// Supports CPU monitoring
        const CPU = 1 << 3;
        /// Supports GPU monitoring
        const GPU = 1 << 4;
        /// Supports NPU monitoring
        const NPU = 1 << 5;
        /// Supports network monitoring
        const NETWORK = 1 << 6;
        /// Supports storage monitoring
        const STORAGE = 1 << 7;
        /// Supports device monitoring
        const DEVICE = 1 << 8;
        /// Supports power monitoring
        const POWER = 1 << 9;
        /// Supports temperature monitoring
        const TEMPERATURE = 1 << 10;
        /// Supports fan monitoring
        const FAN = 1 << 11;
        /// Supports voltage monitoring
        const VOLTAGE = 1 << 12;
        /// Supports current monitoring
        const CURRENT = 1 << 13;
        /// Supports frequency monitoring
        const FREQUENCY = 1 << 14;
        /// Supports performance monitoring
        const PERFORMANCE = 1 << 15;
    }
}

/// Monitor metric
pub struct MonitorMetric {
    /// Metric name
    name: String,
    /// Metric type
    metric_type: String,
    /// Metric value
    value: f64,
    /// Metric unit
    unit: String,
    /// Metric description
    description: String,
    /// Metric timestamp
    timestamp: u64,
}

/// Monitor target
pub struct MonitorTarget {
    /// Target name
    name: String,
    /// Target type
    target_type: String,
    /// Target description
    description: String,
    /// Target metrics
    metrics: Vec<MonitorMetric>,
    /// Target capabilities
    capabilities: MonitorCapabilities,
}

/// Monitor application
pub struct MonitorApplication {
    /// Application name
    name: String,
    /// Application version
    version: String,
    /// Application capabilities
    capabilities: UserlandCapabilities,
    /// Monitor capabilities
    monitor_capabilities: MonitorCapabilities,
    /// Monitor targets
    targets: Vec<MonitorTarget>,
}

impl MonitorApplication {
    /// Create new monitor application
    pub fn new() -> Self {
        MonitorApplication {
            name: String::from("monitor"),
            version: String::from("0.1.0"),
            capabilities: UserlandCapabilities::all(),
            monitor_capabilities: MonitorCapabilities::all(),
            targets: Vec::new(),
        }
    }

    /// Get monitor capabilities
    pub fn monitor_capabilities(&self) -> MonitorCapabilities {
        self.monitor_capabilities
    }

    /// Get monitor targets
    pub fn targets(&self) -> &[MonitorTarget] {
        &self.targets
    }

    /// Add monitor target
    pub fn add_target(&mut self, target: MonitorTarget) {
        self.targets.push(target);
    }

    /// Remove monitor target
    pub fn remove_target(&mut self, name: &str) {
        if let Some(index) = self.targets.iter().position(|t| t.name == name) {
            self.targets.remove(index);
        }
    }

    /// Get monitor target by name
    pub fn get_target(&self, name: &str) -> Option<&MonitorTarget> {
        self.targets.iter().find(|t| t.name == name)
    }

    /// Get monitor targets by type
    pub fn get_targets_by_type(&self, target_type: &str) -> Vec<&MonitorTarget> {
        self.targets
            .iter()
            .filter(|t| t.target_type == target_type)
            .collect()
    }

    /// Get monitor targets by capability
    pub fn get_targets_by_capability(&self, capability: MonitorCapabilities) -> Vec<&MonitorTarget> {
        self.targets
            .iter()
            .filter(|t| t.capabilities.contains(capability))
            .collect()
    }
}

impl Application for MonitorApplication {
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

/// Global monitor application
static MONITOR_APPLICATION: Mutex<Option<Arc<MonitorApplication>>> = Mutex::new(None);

/// Initialize monitor application
pub fn init() {
    let application = Arc::new(MonitorApplication::new());
    *MONITOR_APPLICATION.lock() = Some(Arc::clone(&application));
    crate::register_application(&*application);
}

/// Get monitor application
pub fn get_application() -> Option<Arc<MonitorApplication>> {
    MONITOR_APPLICATION.lock().as_ref().map(Arc::clone)
}
