//! AArch64 system registers

use core::arch::asm;

/// System register read/write macros
macro_rules! read_reg {
    ($reg:expr) => {{
        let val: u64;
        unsafe {
            asm!(concat!("mrs {}, ", $reg), out(reg) val);
        }
        val
    }};
}

macro_rules! write_reg {
    ($reg:expr, $val:expr) => {{
        unsafe {
            asm!(concat!("msr ", $reg, ", {}"), in(reg) $val);
        }
    }};
}

/// Current Exception Level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExceptionLevel {
    EL0 = 0,
    EL1 = 1,
    EL2 = 2,
    EL3 = 3,
}

impl ExceptionLevel {
    /// Get current exception level
    pub fn current() -> Self {
        let el = (read_reg!("CurrentEL") >> 2) & 0b11;
        match el {
            0 => ExceptionLevel::EL0,
            1 => ExceptionLevel::EL1,
            2 => ExceptionLevel::EL2,
            3 => ExceptionLevel::EL3,
            _ => unreachable!(),
        }
    }
}

/// DAIF flags
pub struct DAIF;

impl DAIF {
    /// Enable interrupts
    pub fn enable_interrupts() {
        write_reg!("daifclr", 0xf);
    }

    /// Disable interrupts
    pub fn disable_interrupts() {
        write_reg!("daifset", 0xf);
    }

    /// Check if interrupts are enabled
    pub fn interrupts_enabled() -> bool {
        (read_reg!("daif") & 0xf) == 0
    }
}

/// MPIDR register
pub struct MPIDR;

impl MPIDR {
    /// Get current core ID
    pub fn core_id() -> u64 {
        read_reg!("mpidr_el1") & 0xFF
    }

    /// Get current cluster ID
    pub fn cluster_id() -> u64 {
        (read_reg!("mpidr_el1") >> 8) & 0xFF
    }
}

/// CPACR register
pub struct CPACR;

impl CPACR {
    /// Enable floating point and NEON
    pub fn enable_fp_neon() {
        write_reg!("cpacr_el1", 3 << 20);
    }

    /// Disable floating point and NEON
    pub fn disable_fp_neon() {
        write_reg!("cpacr_el1", 0);
    }

    /// Check if floating point and NEON are enabled
    pub fn fp_neon_enabled() -> bool {
        (read_reg!("cpacr_el1") & (3 << 20)) == (3 << 20)
    }
}

/// SCTLR register
pub struct SCTLR;

impl SCTLR {
    /// Enable MMU
    pub fn enable_mmu() {
        let mut val = read_reg!("sctlr_el1");
        val |= 1; // M bit
        write_reg!("sctlr_el1", val);
    }

    /// Disable MMU
    pub fn disable_mmu() {
        let mut val = read_reg!("sctlr_el1");
        val &= !1; // Clear M bit
        write_reg!("sctlr_el1", val);
    }

    /// Check if MMU is enabled
    pub fn mmu_enabled() -> bool {
        (read_reg!("sctlr_el1") & 1) == 1
    }

    /// Enable instruction cache
    pub fn enable_icache() {
        let mut val = read_reg!("sctlr_el1");
        val |= 1 << 12; // I bit
        write_reg!("sctlr_el1", val);
    }

    /// Disable instruction cache
    pub fn disable_icache() {
        let mut val = read_reg!("sctlr_el1");
        val &= !(1 << 12); // Clear I bit
        write_reg!("sctlr_el1", val);
    }

    /// Enable data cache
    pub fn enable_dcache() {
        let mut val = read_reg!("sctlr_el1");
        val |= 1 << 2; // C bit
        write_reg!("sctlr_el1", val);
    }

    /// Disable data cache
    pub fn disable_dcache() {
        let mut val = read_reg!("sctlr_el1");
        val &= !(1 << 2); // Clear C bit
        write_reg!("sctlr_el1", val);
    }
}

/// TCR register
pub struct TCR;

impl TCR {
    /// Configure translation control register
    pub fn configure() {
        let val = (16 << 0)   // T0SZ = 16 (48-bit VA)
            | (0 << 7)        // EPD0 = 0 (enable TTBR0)
            | (0 << 8)        // IRGN0 = 0b00 (Normal NC)
            | (0 << 10)       // ORGN0 = 0b00 (Normal NC)
            | (0 << 12)       // SH0 = 0b00 (Non-shareable)
            | (0 << 14)       // TG0 = 0b00 (4KB pages)
            | (16 << 16)      // T1SZ = 16 (48-bit VA)
            | (0 << 23)       // EPD1 = 0 (enable TTBR1)
            | (0 << 24)       // IRGN1 = 0b00 (Normal NC)
            | (0 << 26)       // ORGN1 = 0b00 (Normal NC)
            | (0 << 28)       // SH1 = 0b00 (Non-shareable)
            | (0 << 30)       // TG1 = 0b00 (4KB pages)
            | (1 << 32)       // IPS = 0b001 (40-bit PA)
            | (0 << 35)       // AS = 0 (8-bit ASID)
            | (0 << 36)       // TBI0 = 0 (no tagging)
            | (0 << 37); // TBI1 = 0 (no tagging)
        write_reg!("tcr_el1", val);
    }
}

/// MAIR register
pub struct MAIR;

impl MAIR {
    /// Configure memory attribute indirection register
    pub fn configure() {
        let val = (0xFF << 0)     // Attr0 = 0xFF (Normal Memory)
            | (0x04 << 8)         // Attr1 = 0x04 (Device-nGnRE)
            | (0x00 << 16)        // Attr2 = 0x00 (Device-nGnRnE)
            | (0x44 << 24)        // Attr3 = 0x44 (Normal NC)
            | (0x00 << 32)        // Attr4 = 0x00 (unused)
            | (0x00 << 40)        // Attr5 = 0x00 (unused)
            | (0x00 << 48)        // Attr6 = 0x00 (unused)
            | (0x00 << 56); // Attr7 = 0x00 (unused)
        write_reg!("mair_el1", val);
    }
}

/// Initialize system registers
pub fn init() {
    // Enable floating point and NEON
    CPACR::enable_fp_neon();

    // Configure translation tables
    TCR::configure();
    MAIR::configure();

    // Enable caches
    SCTLR::enable_icache();
    SCTLR::enable_dcache();
}
