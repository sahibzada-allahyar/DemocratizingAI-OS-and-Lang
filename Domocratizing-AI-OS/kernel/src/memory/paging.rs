//! Memory paging

use core::ops::{Index, IndexMut};
use spin::Mutex;

use super::{Frame, PhysAddr, VirtAddr};
use crate::arch::aarch64::mmu;

/// Page size (4KB)
pub const PAGE_SIZE: usize = 4096;

/// Page
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Page {
    /// Page number
    number: usize,
}

impl Page {
    /// Create new page
    pub const fn new(number: usize) -> Self {
        Page { number }
    }

    /// Get page containing address
    pub fn containing_address(address: VirtAddr) -> Self {
        Page {
            number: address.as_usize() / PAGE_SIZE,
        }
    }

    /// Get page number
    pub fn number(&self) -> usize {
        self.number
    }

    /// Get start address
    pub fn start_address(&self) -> VirtAddr {
        VirtAddr::new(self.number * PAGE_SIZE)
    }

    /// Get end address
    pub fn end_address(&self) -> VirtAddr {
        VirtAddr::new((self.number + 1) * PAGE_SIZE)
    }
}

/// Page table entry flags
pub use mmu::EntryFlags;

/// Page table entry
#[derive(Debug, Clone, Copy)]
pub struct Entry(u64);

impl Entry {
    /// Create empty entry
    pub const fn empty() -> Self {
        Entry(0)
    }

    /// Is entry unused?
    pub fn is_unused(&self) -> bool {
        self.0 == 0
    }

    /// Get flags
    pub fn flags(&self) -> EntryFlags {
        EntryFlags::from_bits_truncate(self.0)
    }

    /// Get target address
    pub fn addr(&self) -> PhysAddr {
        PhysAddr::new((self.0 & 0x000f_ffff_ffff_f000) as usize)
    }

    /// Set entry
    pub fn set(&mut self, addr: PhysAddr, flags: EntryFlags) {
        self.0 = (addr.as_u64() & 0x000f_ffff_ffff_f000) | flags.bits();
    }
}

/// Page table
#[repr(align(4096))]
pub struct PageTable {
    entries: [Entry; 512],
}

impl PageTable {
    /// Create empty page table
    pub const fn empty() -> Self {
        PageTable {
            entries: [Entry::empty(); 512],
        }
    }

    /// Zero page table
    pub fn zero(&mut self) {
        for entry in self.entries.iter_mut() {
            *entry = Entry::empty();
        }
    }
}

impl Index<usize> for PageTable {
    type Output = Entry;

    fn index(&self, index: usize) -> &Self::Output {
        &self.entries[index]
    }
}

impl IndexMut<usize> for PageTable {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.entries[index]
    }
}

/// Active page table
pub struct ActivePageTable {
    /// Page table
    table: Mutex<&'static mut PageTable>,
}

impl ActivePageTable {
    /// Create new active page table
    pub unsafe fn new() -> Self {
        ActivePageTable {
            table: Mutex::new(&mut *(0xffff_ffff_ffff_f000 as *mut PageTable)),
        }
    }

    /// Map page
    pub fn map(&self, page: Page, frame: Frame, flags: EntryFlags) -> Result<(), &'static str> {
        unsafe {
            mmu::map_page(
                &mut *self.table.lock(),
                page.start_address(),
                frame.start_address(),
                flags,
            )
        }
    }

    /// Unmap page
    pub fn unmap(&self, page: Page) -> Result<(), &'static str> {
        unsafe { mmu::unmap_page(&mut *self.table.lock(), page.start_address()) }
    }

    /// Translate virtual address to physical address
    pub fn translate(&self, addr: VirtAddr) -> Option<PhysAddr> {
        unsafe { mmu::translate(&*self.table.lock(), addr) }
    }
}

/// Initialize paging
pub fn init() {
    // Create new page table
    let mut table = PageTable::empty();

    // Map kernel
    let kernel_start = 0x4000_0000;
    let kernel_end = 0x4020_0000;
    for frame in Frame::range_inclusive(
        Frame::containing_address(PhysAddr::new(kernel_start)),
        Frame::containing_address(PhysAddr::new(kernel_end - 1)),
    ) {
        let page = Page::containing_address(VirtAddr::new(frame.start_address().as_usize()));
        unsafe {
            mmu::map_page(
                &mut table,
                page.start_address(),
                frame.start_address(),
                EntryFlags::VALID | EntryFlags::WRITABLE,
            )
            .expect("failed to map kernel");
        }
    }

    // Switch to new page table
    unsafe {
        mmu::init();
    }
}

/// Map range
pub fn map_range(
    start: VirtAddr,
    end: VirtAddr,
    flags: EntryFlags,
) -> Result<(), &'static str> {
    let table = unsafe { ActivePageTable::new() };
    let start_page = Page::containing_address(start);
    let end_page = Page::containing_address(end - 1u64);
    for page in Page::range_inclusive(start_page, end_page) {
        let frame = Frame::allocate().ok_or("out of memory")?;
        table.map(page, frame, flags)?;
    }
    Ok(())
}

/// Unmap range
pub fn unmap_range(start: VirtAddr, end: VirtAddr) -> Result<(), &'static str> {
    let table = unsafe { ActivePageTable::new() };
    let start_page = Page::containing_address(start);
    let end_page = Page::containing_address(end - 1u64);
    for page in Page::range_inclusive(start_page, end_page) {
        table.unmap(page)?;
    }
    Ok(())
}

/// Range iterator
pub struct PageRangeInclusive {
    start: Page,
    end: Page,
}

impl Iterator for PageRangeInclusive {
    type Item = Page;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start.number() <= self.end.number() {
            let page = self.start;
            self.start.number += 1;
            Some(page)
        } else {
            None
        }
    }
}

impl Page {
    /// Create page range
    pub fn range_inclusive(start: Page, end: Page) -> PageRangeInclusive {
        PageRangeInclusive { start, end }
    }
}

impl Frame {
    /// Create frame range
    pub fn range_inclusive(start: Frame, end: Frame) -> FrameRangeInclusive {
        FrameRangeInclusive { start, end }
    }
}

/// Frame range iterator
pub struct FrameRangeInclusive {
    start: Frame,
    end: Frame,
}

impl Iterator for FrameRangeInclusive {
    type Item = Frame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start.number() <= self.end.number() {
            let frame = self.start;
            self.start.number += 1;
            Some(frame)
        } else {
            None
        }
    }
}
