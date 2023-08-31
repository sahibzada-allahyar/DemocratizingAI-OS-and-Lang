//! Kernel main entry point

#![no_std]
#![no_main]

use core::arch::global_asm;

// Include assembly files
global_asm!(include_str!("arch/aarch64/boot.s"));
global_asm!(include_str!("arch/aarch64/exceptions.s"));

/// Kernel main function
///
/// This is the entry point for the kernel after the boot assembly code
/// has set up the initial environment.
#[no_mangle]
pub extern "C" fn kernel_main() -> ! {
    // Initialize kernel
    kernel::init();

    // Print boot message
    log::info!("Democratizing AI OS");
    log::info!("Version: {}", env!("CARGO_PKG_VERSION"));
    log::info!("Author: Sahibzada Allahyar");

    // Print CPU information
    let cpu = kernel::arch::cpu::cpu();
    log::info!("CPU cores: {}", cpu.num_cores());
    log::info!("Boot core: {}", cpu.boot_core_id());

    // Print memory information
    let memory = kernel::memory::memory_info();
    log::info!(
        "Memory: {} MB total, {} MB free",
        memory.total_bytes() / 1024 / 1024,
        memory.free_bytes() / 1024 / 1024
    );

    // Print cache information
    let cache = kernel::arch::aarch64::cache::cache();
    for level in 0..cache.levels() {
        if let Some(info) = cache.level_info(level) {
            log::info!(
                "L{} cache: {} KB, {} ways, {} sets, {} byte lines",
                level + 1,
                info.size / 1024,
                info.ways,
                info.sets,
                info.line_size
            );
        }
    }

    // Print MMU information
    log::info!(
        "Page size: {} KB",
        kernel::arch::page_size() / 1024
    );
    log::info!(
        "Virtual address bits: {}",
        kernel::arch::virt_addr_bits()
    );
    log::info!(
        "Physical address bits: {}",
        kernel::arch::phys_addr_bits()
    );

    // Print AI capabilities
    let ai = kernel::ai::ai();
    log::info!("AI acceleration: {}", if ai.has_acceleration() { "Yes" } else { "No" });
    if ai.has_acceleration() {
        log::info!("AI accelerator: {}", ai.accelerator_name());
        log::info!(
            "AI performance: {} TOPS",
            ai.peak_performance() / 1_000_000_000_000
        );
    }

    // Start scheduler
    kernel::scheduler::start();

    // We should never reach this point
    loop {
        kernel::arch::wait_for_interrupt();
    }
}

/// Early exception handler
///
/// This is called by the assembly code when an exception occurs before
/// the proper exception handling is set up.
#[no_mangle]
pub extern "C" fn early_exception_handler() -> ! {
    loop {
        // Wait for interrupt (which will never come)
        unsafe {
            core::arch::asm!("wfi");
        }
    }
}

/// Early panic handler
///
/// This is called when a panic occurs before the proper panic handler
/// is set up.
#[panic_handler]
fn early_panic(_info: &core::panic::PanicInfo) -> ! {
    loop {
        // Wait for interrupt (which will never come)
        unsafe {
            core::arch::asm!("wfi");
        }
    }
}
