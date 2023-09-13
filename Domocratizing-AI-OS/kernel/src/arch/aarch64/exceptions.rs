//! AArch64 exception handling

use core::arch::asm;
use core::fmt;

/// Exception context saved by the exception handlers
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ExceptionContext {
    // System registers
    pub spsr: u64,
    pub elr: u64,

    // General purpose registers
    pub x30: u64,
    pub x29: u64,
    pub x28: u64,
    pub x27: u64,
    pub x26: u64,
    pub x25: u64,
    pub x24: u64,
    pub x23: u64,
    pub x22: u64,
    pub x21: u64,
    pub x20: u64,
    pub x19: u64,
    pub x18: u64,
    pub x17: u64,
    pub x16: u64,
    pub x15: u64,
    pub x14: u64,
    pub x13: u64,
    pub x12: u64,
    pub x11: u64,
    pub x10: u64,
    pub x9: u64,
    pub x8: u64,
    pub x7: u64,
    pub x6: u64,
    pub x5: u64,
    pub x4: u64,
    pub x3: u64,
    pub x2: u64,
    pub x1: u64,
    pub x0: u64,
}

impl ExceptionContext {
    /// Get the exception syndrome register value
    pub fn get_esr(&self) -> u64 {
        let esr: u64;
        unsafe {
            asm!("mrs {}, esr_el1", out(reg) esr);
        }
        esr
    }

    /// Get the faulting address register value
    pub fn get_far(&self) -> u64 {
        let far: u64;
        unsafe {
            asm!("mrs {}, far_el1", out(reg) far);
        }
        far
    }

    /// Get the exception class (EC) from ESR
    pub fn get_exception_class(&self) -> u64 {
        (self.get_esr() >> 26) & 0x3f
    }

    /// Get the instruction specific syndrome (ISS) from ESR
    pub fn get_instruction_specific_syndrome(&self) -> u64 {
        self.get_esr() & 0xFFFFFF
    }
}

impl fmt::Display for ExceptionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Exception Context:")?;
        writeln!(f, "  SPSR: {:#018x}", self.spsr)?;
        writeln!(f, "  ELR:  {:#018x}", self.elr)?;
        writeln!(f, "  ESR:  {:#018x}", self.get_esr())?;
        writeln!(f, "  FAR:  {:#018x}", self.get_far())?;
        writeln!(f, "  EC:   {:#x}", self.get_exception_class())?;
        writeln!(f, "  ISS:  {:#x}", self.get_instruction_specific_syndrome())?;
        writeln!(f, "General Purpose Registers:")?;
        writeln!(f, "  x0:  {:#018x}  x1:  {:#018x}", self.x0, self.x1)?;
        writeln!(f, "  x2:  {:#018x}  x3:  {:#018x}", self.x2, self.x3)?;
        writeln!(f, "  x4:  {:#018x}  x5:  {:#018x}", self.x4, self.x5)?;
        writeln!(f, "  x6:  {:#018x}  x7:  {:#018x}", self.x6, self.x7)?;
        writeln!(f, "  x8:  {:#018x}  x9:  {:#018x}", self.x8, self.x9)?;
        writeln!(f, "  x10: {:#018x}  x11: {:#018x}", self.x10, self.x11)?;
        writeln!(f, "  x12: {:#018x}  x13: {:#018x}", self.x12, self.x13)?;
        writeln!(f, "  x14: {:#018x}  x15: {:#018x}", self.x14, self.x15)?;
        writeln!(f, "  x16: {:#018x}  x17: {:#018x}", self.x16, self.x17)?;
        writeln!(f, "  x18: {:#018x}  x19: {:#018x}", self.x18, self.x19)?;
        writeln!(f, "  x20: {:#018x}  x21: {:#018x}", self.x20, self.x21)?;
        writeln!(f, "  x22: {:#018x}  x23: {:#018x}", self.x22, self.x23)?;
        writeln!(f, "  x24: {:#018x}  x25: {:#018x}", self.x24, self.x25)?;
        writeln!(f, "  x26: {:#018x}  x27: {:#018x}", self.x26, self.x27)?;
        writeln!(f, "  x28: {:#018x}  x29: {:#018x}", self.x28, self.x29)?;
        writeln!(f, "  x30: {:#018x}", self.x30)
    }
}

/// Initialize exception handling
pub fn init() {
    unsafe {
        // Set up vector table
        asm!(
            "adr {tmp}, exception_vector_table",
            "msr vbar_el1, {tmp}",
            tmp = out(reg) _,
        );
    }
}

/// Exception handler for synchronous exceptions from current EL with SP0
#[no_mangle]
pub extern "C" fn handle_sync_sp0(ctx: &ExceptionContext) {
    log::error!("Synchronous exception from current EL with SP0:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for IRQ from current EL with SP0
#[no_mangle]
pub extern "C" fn handle_irq_sp0(ctx: &ExceptionContext) {
    super::gic::handle_irq();
}

/// Exception handler for FIQ from current EL with SP0
#[no_mangle]
pub extern "C" fn handle_fiq_sp0(_ctx: &ExceptionContext) {
    log::warn!("FIQ from current EL with SP0");
}

/// Exception handler for SError from current EL with SP0
#[no_mangle]
pub extern "C" fn handle_error_sp0(ctx: &ExceptionContext) {
    log::error!("SError from current EL with SP0:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for synchronous exceptions from current EL with SPx
#[no_mangle]
pub extern "C" fn handle_sync_spx(ctx: &ExceptionContext) {
    log::error!("Synchronous exception from current EL with SPx:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for IRQ from current EL with SPx
#[no_mangle]
pub extern "C" fn handle_irq_spx(ctx: &ExceptionContext) {
    super::gic::handle_irq();
}

/// Exception handler for FIQ from current EL with SPx
#[no_mangle]
pub extern "C" fn handle_fiq_spx(_ctx: &ExceptionContext) {
    log::warn!("FIQ from current EL with SPx");
}

/// Exception handler for SError from current EL with SPx
#[no_mangle]
pub extern "C" fn handle_error_spx(ctx: &ExceptionContext) {
    log::error!("SError from current EL with SPx:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for synchronous exceptions from lower EL in AArch64
#[no_mangle]
pub extern "C" fn handle_sync_aarch64(ctx: &ExceptionContext) {
    log::error!("Synchronous exception from lower EL in AArch64:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for IRQ from lower EL in AArch64
#[no_mangle]
pub extern "C" fn handle_irq_aarch64(ctx: &ExceptionContext) {
    super::gic::handle_irq();
}

/// Exception handler for FIQ from lower EL in AArch64
#[no_mangle]
pub extern "C" fn handle_fiq_aarch64(_ctx: &ExceptionContext) {
    log::warn!("FIQ from lower EL in AArch64");
}

/// Exception handler for SError from lower EL in AArch64
#[no_mangle]
pub extern "C" fn handle_error_aarch64(ctx: &ExceptionContext) {
    log::error!("SError from lower EL in AArch64:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for synchronous exceptions from lower EL in AArch32
#[no_mangle]
pub extern "C" fn handle_sync_aarch32(ctx: &ExceptionContext) {
    log::error!("Synchronous exception from lower EL in AArch32:");
    log::error!("{}", ctx);
    loop {}
}

/// Exception handler for IRQ from lower EL in AArch32
#[no_mangle]
pub extern "C" fn handle_irq_aarch32(ctx: &ExceptionContext) {
    super::gic::handle_irq();
}

/// Exception handler for FIQ from lower EL in AArch32
#[no_mangle]
pub extern "C" fn handle_fiq_aarch32(_ctx: &ExceptionContext) {
    log::warn!("FIQ from lower EL in AArch32");
}

/// Exception handler for SError from lower EL in AArch32
#[no_mangle]
pub extern "C" fn handle_error_aarch32(ctx: &ExceptionContext) {
    log::error!("SError from lower EL in AArch32:");
    log::error!("{}", ctx);
    loop {}
}
