//! AArch64 MMU implementation

use core::ptr;
use core::sync::atomic::{fence, Ordering};

use crate::arch::aarch64::registers::*;
use crate::memory::{Frame, PhysAddr, VirtAddr};

/// Page size (4KB)
pub const PAGE_SIZE: usize = 4096;
/// Page size shift
pub const PAGE_SHIFT: usize = 12;
/// Page table entry size
pub const PTE_SIZE: usize = 8;

/// Memory attributes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryAttributes {
    /// Normal memory
    Normal = 0,
    /// Device memory
    Device = 1,
    /// Non-cacheable normal memory
    NormalNonCacheable = 2,
}

/// Access permissions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPermissions {
    /// Kernel read-only
    KernelReadOnly = 0,
    /// Kernel read/write
    KernelReadWrite = 1,
    /// User read-only
    UserReadOnly = 2,
    /// User read/write
    UserReadWrite = 3,
}

/// Page table entry flags
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct PageTableFlags: u64 {
        /// Valid entry
        const VALID = 1 << 0;
        /// Page table entry
        const TABLE = 1 << 1;
        /// Page entry
        const PAGE = 1 << 1;
        /// Access flag
        const AF = 1 << 10;
        /// Not global
        const NG = 1 << 11;
        /// Read-only
        const RO = 1 << 7;
        /// Privileged execute-never
        const PXN = 1 << 53;
        /// Unprivileged execute-never
        const UXN = 1 << 54;
        /// Shareability field
        const SHARE = 3 << 8;
        /// Inner shareable
        const INNER_SHARE = 3 << 8;
        /// Outer shareable
        const OUTER_SHARE = 2 << 8;
    }
}

/// Page table entry
#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
pub struct PageTableEntry(u64);

impl PageTableEntry {
    /// Create a new page table entry
    pub const fn new() -> Self {
        PageTableEntry(0)
    }

    /// Create a new page table entry from raw value
    pub const fn from_raw(value: u64) -> Self {
        PageTableEntry(value)
    }

    /// Get raw value
    pub const fn raw(&self) -> u64 {
        self.0
    }

    /// Is entry valid?
    pub fn is_valid(&self) -> bool {
        self.0 & PageTableFlags::VALID.bits() != 0
    }

    /// Is entry a table?
    pub fn is_table(&self) -> bool {
        self.0 & PageTableFlags::TABLE.bits() != 0
    }

    /// Get physical address
    pub fn phys_addr(&self) -> PhysAddr {
        PhysAddr::new((self.0 & !0xFFF) as usize)
    }

    /// Set physical address
    pub fn set_phys_addr(&mut self, addr: PhysAddr) {
        self.0 = (self.0 & 0xFFF) | (addr.as_u64() & !0xFFF);
    }

    /// Get flags
    pub fn flags(&self) -> PageTableFlags {
        PageTableFlags::from_bits_truncate(self.0)
    }

    /// Set flags
    pub fn set_flags(&mut self, flags: PageTableFlags) {
        self.0 = (self.0 & !0xFFF) | flags.bits();
    }

    /// Set entry as table
    pub fn set_table(&mut self, frame: Frame, flags: PageTableFlags) {
        self.0 = frame.start_address().as_u64() | flags.bits() | PageTableFlags::TABLE.bits() | PageTableFlags::VALID.bits();
    }

    /// Set entry as page
    pub fn set_page(&mut self, frame: Frame, flags: PageTableFlags) {
        self.0 = frame.start_address().as_u64() | flags.bits() | PageTableFlags::PAGE.bits() | PageTableFlags::VALID.bits();
    }
}

/// Page table
#[repr(align(4096))]
pub struct PageTable {
    entries: [PageTableEntry; 512],
}

impl PageTable {
    /// Create a new page table
    pub const fn new() -> Self {
        PageTable {
            entries: [PageTableEntry::new(); 512],
        }
    }

    /// Get entry at index
    pub fn entry(&self, index: usize) -> &PageTableEntry {
        &self.entries[index]
    }

    /// Get mutable entry at index
    pub fn entry_mut(&mut self, index: usize) -> &mut PageTableEntry {
        &mut self.entries[index]
    }

    /// Zero the page table
    pub fn zero(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = PageTableEntry::new();
        }
    }
}

/// Initialize MMU
pub fn init() {
    unsafe {
        // Set up translation control register
        let mut tcr = 0u64;
        // T0SZ = 16 (48-bit addresses)
        tcr |= 16;
        // 4KB granule
        tcr |= 1 << 14;
        // Inner shareable
        tcr |= 3 << 8;
        // Outer shareable
        tcr |= 3 << 10;
        // Write-back, Read-allocate, Write-allocate cacheable
        tcr |= 1 << 16;
        tcr |= 1 << 17;
        set_tcr_el1(tcr);

        // Set up memory attribute indirection register
        let mut mair = 0u64;
        // Normal memory
        mair |= 0xFF << (8 * MemoryAttributes::Normal as u64);
        // Device memory
        mair |= 0x04 << (8 * MemoryAttributes::Device as u64);
        // Non-cacheable normal memory
        mair |= 0x44 << (8 * MemoryAttributes::NormalNonCacheable as u64);
        set_mair_el1(mair);

        // Ensure changes are visible
        isb();
    }
}

/// Map a virtual address to a physical address
pub fn map(
    table: &mut PageTable,
    virt: VirtAddr,
    phys: PhysAddr,
    flags: PageTableFlags,
    attributes: MemoryAttributes,
) -> Result<(), &'static str> {
    let vpn = [
        (virt.as_usize() >> 12) & 0x1FF,
        (virt.as_usize() >> 21) & 0x1FF,
        (virt.as_usize() >> 30) & 0x1FF,
        (virt.as_usize() >> 39) & 0x1FF,
    ];

    let mut entry = table.entry_mut(vpn[3]);
    if !entry.is_valid() {
        let frame = Frame::allocate().ok_or("Failed to allocate frame")?;
        entry.set_table(frame, flags);
        unsafe {
            ptr::write_bytes(frame.start_address().as_mut_ptr::<u8>(), 0, PAGE_SIZE);
        }
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let mut entry = table.entry_mut(vpn[2]);
    if !entry.is_valid() {
        let frame = Frame::allocate().ok_or("Failed to allocate frame")?;
        entry.set_table(frame, flags);
        unsafe {
            ptr::write_bytes(frame.start_address().as_mut_ptr::<u8>(), 0, PAGE_SIZE);
        }
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let mut entry = table.entry_mut(vpn[1]);
    if !entry.is_valid() {
        let frame = Frame::allocate().ok_or("Failed to allocate frame")?;
        entry.set_table(frame, flags);
        unsafe {
            ptr::write_bytes(frame.start_address().as_mut_ptr::<u8>(), 0, PAGE_SIZE);
        }
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let entry = table.entry_mut(vpn[0]);
    if entry.is_valid() {
        return Err("Page already mapped");
    }

    entry.set_page(Frame::containing_address(phys), flags);
    fence(Ordering::SeqCst);

    Ok(())
}

/// Unmap a virtual address
pub fn unmap(table: &mut PageTable, virt: VirtAddr) -> Result<(), &'static str> {
    let vpn = [
        (virt.as_usize() >> 12) & 0x1FF,
        (virt.as_usize() >> 21) & 0x1FF,
        (virt.as_usize() >> 30) & 0x1FF,
        (virt.as_usize() >> 39) & 0x1FF,
    ];

    let mut entry = table.entry_mut(vpn[3]);
    if !entry.is_valid() {
        return Err("Page not mapped");
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let mut entry = table.entry_mut(vpn[2]);
    if !entry.is_valid() {
        return Err("Page not mapped");
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let mut entry = table.entry_mut(vpn[1]);
    if !entry.is_valid() {
        return Err("Page not mapped");
    }

    let table = unsafe { &mut *(entry.phys_addr().as_mut_ptr::<PageTable>()) };
    let entry = table.entry_mut(vpn[0]);
    if !entry.is_valid() {
        return Err("Page not mapped");
    }

    *entry = PageTableEntry::new();
    fence(Ordering::SeqCst);

    Ok(())
}

/// Translate a virtual address to a physical address
pub fn translate(table: &PageTable, virt: VirtAddr) -> Option<PhysAddr> {
    let vpn = [
        (virt.as_usize() >> 12) & 0x1FF,
        (virt.as_usize() >> 21) & 0x1FF,
        (virt.as_usize() >> 30) & 0x1FF,
        (virt.as_usize() >> 39) & 0x1FF,
    ];

    let entry = table.entry(vpn[3]);
    if !entry.is_valid() {
        return None;
    }

    let table = unsafe { &*(entry.phys_addr().as_ptr::<PageTable>()) };
    let entry = table.entry(vpn[2]);
    if !entry.is_valid() {
        return None;
    }

    let table = unsafe { &*(entry.phys_addr().as_ptr::<PageTable>()) };
    let entry = table.entry(vpn[1]);
    if !entry.is_valid() {
        return None;
    }

    let table = unsafe { &*(entry.phys_addr().as_ptr::<PageTable>()) };
    let entry = table.entry(vpn[0]);
    if !entry.is_valid() {
        return None;
    }

    Some(entry.phys_addr())
}
