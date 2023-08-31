//! Kernel library interface

#![no_std]
#![feature(asm_const)]
#![feature(naked_functions)]
#![feature(const_mut_refs)]
#![feature(allocator_api)]
#![feature(alloc_error_handler)]
#![feature(panic_info_message)]

extern crate alloc;

pub mod arch;
pub mod memory;
pub mod sync;
pub mod scheduler;
pub mod ai;

use core::panic::PanicInfo;

/// Kernel panic handler
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Disable interrupts
    arch::disable_interrupts();

    // Print panic message
    if let Some(location) = info.location() {
        log::error!(
            "Kernel panic at {}:{}: {}",
            location.file(),
            location.line(),
            info.message().unwrap_or(&format_args!(""))
        );
    } else {
        log::error!("Kernel panic: {}", info.message().unwrap_or(&format_args!("")));
    }

    // Print stack trace if available
    #[cfg(debug_assertions)]
    {
        log::error!("Stack trace:");
        let mut fp: usize;
        unsafe {
            asm!("mov {}, x29", out(reg) fp);
        }
        let mut depth = 0;
        while fp != 0 && depth < 16 {
            let lr = unsafe { *(fp as *const usize).offset(1) };
            log::error!("  #{}: {:#018x}", depth, lr);
            fp = unsafe { *(fp as *const usize) };
            depth += 1;
        }
    }

    // Halt all cores
    loop {
        arch::wait_for_interrupt();
    }
}

/// Out of memory handler
#[alloc_error_handler]
fn alloc_error_handler(layout: alloc::alloc::Layout) -> ! {
    panic!("Out of memory: {:?}", layout);
}

/// Initialize kernel
pub fn init() {
    // Initialize architecture
    arch::init();

    // Initialize memory management
    memory::init();

    // Initialize synchronization primitives
    sync::init();

    // Initialize scheduler
    scheduler::init();

    // Initialize AI subsystem
    ai::init();

    // Enable interrupts
    arch::enable_interrupts();
}
