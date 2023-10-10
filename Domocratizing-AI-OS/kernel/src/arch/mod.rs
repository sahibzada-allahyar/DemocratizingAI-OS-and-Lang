//! Architecture support

pub mod aarch64;
pub mod cpu;

/// Architecture-specific initialization
pub fn init() {
    aarch64::init();
}

/// Enable interrupts
#[inline]
pub fn enable_interrupts() {
    aarch64::enable_interrupts();
}

/// Disable interrupts
#[inline]
pub fn disable_interrupts() {
    aarch64::disable_interrupts();
}

/// Wait for interrupt
#[inline]
pub fn wait_for_interrupt() {
    aarch64::wait_for_interrupt();
}

/// Get current exception level
#[inline]
pub fn current_el() -> u8 {
    aarch64::current_el()
}

/// Get processor ID
#[inline]
pub fn processor_id() -> u32 {
    aarch64::processor_id()
}

/// Get system counter frequency
#[inline]
pub fn system_counter_freq() -> u64 {
    aarch64::system_counter_freq()
}

/// Get system counter value
#[inline]
pub fn system_counter() -> u64 {
    aarch64::system_counter()
}

/// Get system timer value
#[inline]
pub fn system_timer() -> u64 {
    aarch64::system_timer()
}

/// Get system timer control
#[inline]
pub fn system_timer_control() -> u32 {
    aarch64::system_timer_control()
}

/// Set system timer control
#[inline]
pub fn set_system_timer_control(val: u32) {
    aarch64::set_system_timer_control(val);
}

/// Get system timer compare value
#[inline]
pub fn system_timer_cmp() -> u64 {
    aarch64::system_timer_cmp()
}

/// Set system timer compare value
#[inline]
pub fn set_system_timer_cmp(val: u64) {
    aarch64::set_system_timer_cmp(val);
}

/// Get system timer frequency
#[inline]
pub fn system_timer_freq() -> u64 {
    aarch64::system_timer_freq()
}

/// Get cache line size
#[inline]
pub fn cache_line_size() -> usize {
    aarch64::cache_line_size()
}

/// Get data cache zero line size
#[inline]
pub fn dcache_zero_line_size() -> usize {
    aarch64::dcache_zero_line_size()
}

/// Get instruction cache line size
#[inline]
pub fn icache_line_size() -> usize {
    aarch64::icache_line_size()
}

/// Get page size
#[inline]
pub fn page_size() -> usize {
    aarch64::page_size()
}

/// Get page shift
#[inline]
pub fn page_shift() -> usize {
    aarch64::page_shift()
}

/// Get page mask
#[inline]
pub fn page_mask() -> usize {
    aarch64::page_mask()
}

/// Get page table entry size
#[inline]
pub fn pte_size() -> usize {
    aarch64::pte_size()
}

/// Get page table entry shift
#[inline]
pub fn pte_shift() -> usize {
    aarch64::pte_shift()
}

/// Get page table entries per page
#[inline]
pub fn ptes_per_page() -> usize {
    aarch64::ptes_per_page()
}

/// Get page table entry mask
#[inline]
pub fn pte_mask() -> usize {
    aarch64::pte_mask()
}

/// Get physical address bits
#[inline]
pub fn phys_addr_bits() -> usize {
    aarch64::phys_addr_bits()
}

/// Get virtual address bits
#[inline]
pub fn virt_addr_bits() -> usize {
    aarch64::virt_addr_bits()
}

/// Get physical address mask
#[inline]
pub fn phys_addr_mask() -> usize {
    aarch64::phys_addr_mask()
}

/// Get virtual address mask
#[inline]
pub fn virt_addr_mask() -> usize {
    aarch64::virt_addr_mask()
}

/// Get physical address shift
#[inline]
pub fn phys_addr_shift() -> usize {
    aarch64::phys_addr_shift()
}

/// Get virtual address shift
#[inline]
pub fn virt_addr_shift() -> usize {
    aarch64::virt_addr_shift()
}

/// Get physical address offset
#[inline]
pub fn phys_addr_offset() -> usize {
    aarch64::phys_addr_offset()
}

/// Get virtual address offset
#[inline]
pub fn virt_addr_offset() -> usize {
    aarch64::virt_addr_offset()
}
