//! Architecture-independent CPU abstractions

use core::sync::atomic::{AtomicBool, Ordering};

/// CPU state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// CPU is offline
    Offline,
    /// CPU is online and running
    Running,
    /// CPU is sleeping
    Sleeping,
    /// CPU is in deep sleep
    DeepSleep,
}

/// CPU features
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct Features: u64 {
        /// Floating point support
        const FP = 1 << 0;
        /// SIMD/NEON support
        const SIMD = 1 << 1;
        /// AES hardware acceleration
        const AES = 1 << 2;
        /// SHA hardware acceleration
        const SHA = 1 << 3;
        /// CRC32 hardware acceleration
        const CRC32 = 1 << 4;
        /// Atomic operations support
        const ATOMICS = 1 << 5;
        /// Memory ordering support
        const ORDERING = 1 << 6;
        /// Vector operations support
        const VECTOR = 1 << 7;
        /// SVE support
        const SVE = 1 << 8;
        /// SVE2 support
        const SVE2 = 1 << 9;
        /// Hardware debug support
        const DEBUG = 1 << 10;
        /// Performance monitoring support
        const PMU = 1 << 11;
        /// Security extensions
        const SECURITY = 1 << 12;
        /// Virtualization support
        const VIRTUALIZATION = 1 << 13;
        /// RAS (Reliability, Availability, Serviceability) support
        const RAS = 1 << 14;
    }
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct Topology {
    /// Core ID
    pub core_id: u32,
    /// Socket ID
    pub socket_id: u32,
    /// Core type (e.g. big/little)
    pub core_type: u32,
    /// Core group
    pub core_group: u32,
}

/// CPU power state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PowerState {
    /// Full power
    Full,
    /// Low power
    Low,
    /// Idle
    Idle,
    /// Off
    Off,
}

/// CPU statistics
#[derive(Debug, Clone)]
pub struct Statistics {
    /// Time spent in user mode
    pub user_time: u64,
    /// Time spent in system mode
    pub system_time: u64,
    /// Time spent idle
    pub idle_time: u64,
    /// Number of context switches
    pub context_switches: u64,
    /// Number of interrupts
    pub interrupts: u64,
    /// Number of soft interrupts
    pub soft_interrupts: u64,
}

/// CPU core abstraction
pub struct Core {
    /// Core ID
    id: u32,
    /// Core state
    state: AtomicBool,
    /// Core features
    features: Features,
    /// Core topology
    topology: Topology,
    /// Core frequency in Hz
    frequency: u64,
    /// Core power state
    power_state: PowerState,
    /// Core statistics
    statistics: Statistics,
}

impl Core {
    /// Create new core
    pub fn new(id: u32) -> Self {
        Core {
            id,
            state: AtomicBool::new(false),
            features: Features::empty(),
            topology: Topology {
                core_id: id,
                socket_id: 0,
                core_type: 0,
                core_group: 0,
            },
            frequency: 0,
            power_state: PowerState::Off,
            statistics: Statistics {
                user_time: 0,
                system_time: 0,
                idle_time: 0,
                context_switches: 0,
                interrupts: 0,
                soft_interrupts: 0,
            },
        }
    }

    /// Get core ID
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Is core active?
    pub fn is_active(&self) -> bool {
        self.state.load(Ordering::Acquire)
    }

    /// Set core state
    pub fn set_active(&self, active: bool) {
        self.state.store(active, Ordering::Release);
    }

    /// Get core features
    pub fn features(&self) -> Features {
        self.features
    }

    /// Set core features
    pub fn set_features(&mut self, features: Features) {
        self.features = features;
    }

    /// Get core topology
    pub fn topology(&self) -> &Topology {
        &self.topology
    }

    /// Set core topology
    pub fn set_topology(&mut self, topology: Topology) {
        self.topology = topology;
    }

    /// Get core frequency
    pub fn frequency(&self) -> u64 {
        self.frequency
    }

    /// Set core frequency
    pub fn set_frequency(&mut self, frequency: u64) {
        self.frequency = frequency;
    }

    /// Get core power state
    pub fn power_state(&self) -> PowerState {
        self.power_state
    }

    /// Set core power state
    pub fn set_power_state(&mut self, state: PowerState) {
        self.power_state = state;
    }

    /// Get core statistics
    pub fn statistics(&self) -> &Statistics {
        &self.statistics
    }

    /// Update core statistics
    pub fn update_statistics(&mut self, statistics: Statistics) {
        self.statistics = statistics;
    }
}

/// CPU abstraction
pub struct Cpu {
    /// Number of cores
    num_cores: usize,
    /// Cores
    cores: [Core; 8],
    /// Boot core ID
    boot_core_id: u32,
}

impl Cpu {
    /// Create new CPU instance
    pub const fn new() -> Self {
        Cpu {
            num_cores: 0,
            cores: [
                Core::new(0),
                Core::new(1),
                Core::new(2),
                Core::new(3),
                Core::new(4),
                Core::new(5),
                Core::new(6),
                Core::new(7),
            ],
            boot_core_id: 0,
        }
    }

    /// Get number of cores
    pub fn num_cores(&self) -> usize {
        self.num_cores
    }

    /// Get boot core ID
    pub fn boot_core_id(&self) -> u32 {
        self.boot_core_id
    }

    /// Get core by ID
    pub fn core(&self, id: u32) -> Option<&Core> {
        if id < 8 {
            Some(&self.cores[id as usize])
        } else {
            None
        }
    }

    /// Get mutable core by ID
    pub fn core_mut(&mut self, id: u32) -> Option<&mut Core> {
        if id < 8 {
            Some(&mut self.cores[id as usize])
        } else {
            None
        }
    }

    /// Get current core
    pub fn current_core(&self) -> &Core {
        let id = crate::arch::processor_id();
        &self.cores[id as usize]
    }

    /// Get mutable current core
    pub fn current_core_mut(&mut self) -> &mut Core {
        let id = crate::arch::processor_id();
        &mut self.cores[id as usize]
    }

    /// Start core
    pub fn start_core(&mut self, core_id: u32, entry: extern "C" fn()) -> Result<(), &'static str> {
        if core_id >= 8 {
            return Err("Invalid core ID");
        }

        if self.cores[core_id as usize].is_active() {
            return Err("Core already active");
        }

        // Delegate to architecture-specific implementation
        crate::arch::aarch64::cpu::cpu_mut().start_core(core_id, entry)
    }

    /// Stop core
    pub fn stop_core(&mut self, core_id: u32) -> Result<(), &'static str> {
        if core_id >= 8 {
            return Err("Invalid core ID");
        }

        if !self.cores[core_id as usize].is_active() {
            return Err("Core not active");
        }

        if core_id == self.boot_core_id {
            return Err("Cannot stop boot core");
        }

        // Delegate to architecture-specific implementation
        crate::arch::aarch64::cpu::cpu_mut().stop_core(core_id)
    }

    /// Put current core to sleep
    pub fn sleep(&self) {
        // Delegate to architecture-specific implementation
        crate::arch::aarch64::cpu::cpu().sleep();
    }

    /// Wake up core
    pub fn wake_core(&self, core_id: u32) -> Result<(), &'static str> {
        if core_id >= 8 {
            return Err("Invalid core ID");
        }

        if !self.cores[core_id as usize].is_active() {
            return Err("Core not active");
        }

        // Delegate to architecture-specific implementation
        crate::arch::aarch64::cpu::cpu().wake_core(core_id)
    }
}

/// Global CPU instance
static mut CPU: Option<Cpu> = None;

/// Initialize CPU
pub fn init() {
    unsafe {
        CPU = Some(Cpu::new());
    }
}

/// Get CPU instance
pub fn cpu() -> &'static Cpu {
    unsafe { CPU.as_ref().unwrap() }
}

/// Get mutable CPU instance
pub fn cpu_mut() -> &'static mut Cpu {
    unsafe { CPU.as_mut().unwrap() }
}
