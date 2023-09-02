//! AArch64 architecture support

pub mod cache;
pub mod cpu;
pub mod exceptions;
pub mod mmu;
pub mod registers;

use core::arch::asm;

/// Initialize architecture
pub fn init() {
    // Initialize CPU
    cpu::init();

    // Initialize exception handling
    exceptions::init();

    // Initialize cache
    cache::init();

    // Initialize MMU
    mmu::init();
}

/// Enable interrupts
pub fn enable_interrupts() {
    unsafe {
        asm!("msr daifclr, #0xf");
    }
}

/// Disable interrupts
pub fn disable_interrupts() {
    unsafe {
        asm!("msr daifset, #0xf");
    }
}

/// Wait for interrupt
pub fn wait_for_interrupt() {
    unsafe {
        asm!("wfi");
    }
}

/// Get current exception level
pub fn current_el() -> u8 {
    let mut el: u64;
    unsafe {
        asm!("mrs {}, CurrentEL", out(reg) el);
    }
    ((el >> 2) & 0x3) as u8
}

/// Get processor ID
pub fn processor_id() -> u32 {
    let mut mpidr: u64;
    unsafe {
        asm!("mrs {}, mpidr_el1", out(reg) mpidr);
    }
    (mpidr & 0xFF) as u32
}

/// Get system counter frequency
pub fn system_counter_freq() -> u64 {
    let mut freq: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }
    freq
}

/// Get system counter value
pub fn system_counter() -> u64 {
    let mut val: u64;
    unsafe {
        asm!("mrs {}, cntpct_el0", out(reg) val);
    }
    val
}

/// Get system timer value
pub fn system_timer() -> u64 {
    let mut val: u64;
    unsafe {
        asm!("mrs {}, cntvct_el0", out(reg) val);
    }
    val
}

/// Get system timer control
pub fn system_timer_control() -> u32 {
    let mut val: u32;
    unsafe {
        asm!("mrs {}, cntv_ctl_el0", out(reg) val);
    }
    val
}

/// Set system timer control
pub fn set_system_timer_control(val: u32) {
    unsafe {
        asm!("msr cntv_ctl_el0, {}", in(reg) val);
    }
}

/// Get system timer compare value
pub fn system_timer_cmp() -> u64 {
    let mut val: u64;
    unsafe {
        asm!("mrs {}, cntv_cval_el0", out(reg) val);
    }
    val
}

/// Set system timer compare value
pub fn set_system_timer_cmp(val: u64) {
    unsafe {
        asm!("msr cntv_cval_el0, {}", in(reg) val);
    }
}

/// Get system timer frequency
pub fn system_timer_freq() -> u64 {
    let mut freq: u64;
    unsafe {
        asm!("mrs {}, cntfrq_el0", out(reg) freq);
    }
    freq
}

/// Get cache line size
pub fn cache_line_size() -> usize {
    let mut ctr: u64;
    unsafe {
        asm!("mrs {}, ctr_el0", out(reg) ctr);
    }
    4 << ((ctr >> 16) & 0xF) as usize
}

/// Get data cache zero line size
pub fn dcache_zero_line_size() -> usize {
    let mut dczid: u64;
    unsafe {
        asm!("mrs {}, dczid_el0", out(reg) dczid);
    }
    if (dczid & 0x10) != 0 {
        0
    } else {
        4 << (dczid & 0xF) as usize
    }
}

/// Get instruction cache line size
pub fn icache_line_size() -> usize {
    let mut ctr: u64;
    unsafe {
        asm!("mrs {}, ctr_el0", out(reg) ctr);
    }
    4 << (ctr & 0xF) as usize
}

/// Get page size
pub fn page_size() -> usize {
    4096
}

/// Get page shift
pub fn page_shift() -> usize {
    12
}

/// Get page mask
pub fn page_mask() -> usize {
    0xFFF
}

/// Get page table entry size
pub fn pte_size() -> usize {
    8
}

/// Get page table entry shift
pub fn pte_shift() -> usize {
    3
}

/// Get page table entries per page
pub fn ptes_per_page() -> usize {
    512
}

/// Get page table entry mask
pub fn pte_mask() -> usize {
    0x1FF
}

/// Get physical address bits
pub fn phys_addr_bits() -> usize {
    48
}

/// Get virtual address bits
pub fn virt_addr_bits() -> usize {
    48
}

/// Get physical address mask
pub fn phys_addr_mask() -> usize {
    (1 << phys_addr_bits()) - 1
}

/// Get virtual address mask
pub fn virt_addr_mask() -> usize {
    (1 << virt_addr_bits()) - 1
}

/// Get physical address shift
pub fn phys_addr_shift() -> usize {
    12
}

/// Get virtual address shift
pub fn virt_addr_shift() -> usize {
    12
}

/// Get physical address offset
pub fn phys_addr_offset() -> usize {
    0
}

/// Get virtual address offset
pub fn virt_addr_offset() -> usize {
    0xFFFF_0000_0000_0000
}
