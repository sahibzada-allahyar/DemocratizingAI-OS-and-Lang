//! Memory management

pub mod allocator;
pub mod frame;
pub mod heap;
pub mod paging;

use core::fmt;
use core::ops::{Add, AddAssign, Sub, SubAssign};

/// Physical memory address
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct PhysAddr(usize);

impl PhysAddr {
    /// Create new physical address
    pub const fn new(addr: usize) -> Self {
        PhysAddr(addr)
    }

    /// Get address value
    pub fn as_usize(self) -> usize {
        self.0
    }

    /// Get address value as u64
    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }

    /// Get mutable pointer
    pub fn as_mut_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }

    /// Get pointer
    pub fn as_ptr<T>(self) -> *const T {
        self.0 as *const T
    }

    /// Align address up
    pub fn align_up(self, align: usize) -> Self {
        PhysAddr((self.0 + align - 1) & !(align - 1))
    }

    /// Align address down
    pub fn align_down(self, align: usize) -> Self {
        PhysAddr(self.0 & !(align - 1))
    }

    /// Is address aligned?
    pub fn is_aligned(self, align: usize) -> bool {
        self.0 & (align - 1) == 0
    }
}

impl Add<usize> for PhysAddr {
    type Output = Self;

    fn add(self, rhs: usize) -> Self {
        PhysAddr(self.0 + rhs)
    }
}

impl AddAssign<usize> for PhysAddr {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

impl Sub<usize> for PhysAddr {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self {
        PhysAddr(self.0 - rhs)
    }
}

impl SubAssign<usize> for PhysAddr {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

impl fmt::Display for PhysAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PhysAddr({:#x})", self.0)
    }
}

/// Virtual memory address
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct VirtAddr(usize);

impl VirtAddr {
    /// Create new virtual address
    pub const fn new(addr: usize) -> Self {
        VirtAddr(addr)
    }

    /// Get address value
    pub fn as_usize(self) -> usize {
        self.0
    }

    /// Get address value as u64
    pub fn as_u64(self) -> u64 {
        self.0 as u64
    }

    /// Get mutable pointer
    pub fn as_mut_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }

    /// Get pointer
    pub fn as_ptr<T>(self) -> *const T {
        self.0 as *const T
    }

    /// Align address up
    pub fn align_up(self, align: usize) -> Self {
        VirtAddr((self.0 + align - 1) & !(align - 1))
    }

    /// Align address down
    pub fn align_down(self, align: usize) -> Self {
        VirtAddr(self.0 & !(align - 1))
    }

    /// Is address aligned?
    pub fn is_aligned(self, align: usize) -> bool {
        self.0 & (align - 1) == 0
    }
}

impl Add<usize> for VirtAddr {
    type Output = Self;

    fn add(self, rhs: usize) -> Self {
        VirtAddr(self.0 + rhs)
    }
}

impl AddAssign<usize> for VirtAddr {
    fn add_assign(&mut self, rhs: usize) {
        self.0 += rhs;
    }
}

impl Sub<usize> for VirtAddr {
    type Output = Self;

    fn sub(self, rhs: usize) -> Self {
        VirtAddr(self.0 - rhs)
    }
}

impl SubAssign<usize> for VirtAddr {
    fn sub_assign(&mut self, rhs: usize) {
        self.0 -= rhs;
    }
}

impl fmt::Display for VirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VirtAddr({:#x})", self.0)
    }
}

/// Initialize memory
pub fn init() {
    // Initialize frame allocator
    frame::init();

    // Initialize heap allocator
    heap::init();

    // Initialize paging
    paging::init();
}
