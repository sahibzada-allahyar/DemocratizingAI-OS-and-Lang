//! Memory allocator traits and implementations

use core::alloc::Layout;
use core::ptr::NonNull;

/// Memory allocator trait
pub trait Allocator {
    /// Allocate memory
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError>;

    /// Deallocate memory
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);

    /// Allocate zeroed memory
    fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        // Safety: We just allocated this memory
        unsafe {
            core::ptr::write_bytes(ptr.as_ptr() as *mut u8, 0, layout.size());
        }
        Ok(ptr)
    }

    /// Grow allocation in place
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        // Allocate new memory
        let new_ptr = self.allocate(new_layout)?;

        // Copy old memory to new memory
        core::ptr::copy_nonoverlapping(
            ptr.as_ptr(),
            new_ptr.as_ptr() as *mut u8,
            old_layout.size(),
        );

        // Deallocate old memory
        self.deallocate(ptr, old_layout);

        Ok(new_ptr)
    }

    /// Grow allocation in place with zeroed memory
    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() >= old_layout.size(),
            "`new_layout.size()` must be greater than or equal to `old_layout.size()`"
        );

        // Allocate new memory
        let new_ptr = self.allocate_zeroed(new_layout)?;

        // Copy old memory to new memory
        core::ptr::copy_nonoverlapping(
            ptr.as_ptr(),
            new_ptr.as_ptr() as *mut u8,
            old_layout.size(),
        );

        // Deallocate old memory
        self.deallocate(ptr, old_layout);

        Ok(new_ptr)
    }

    /// Shrink allocation in place
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(
            new_layout.size() <= old_layout.size(),
            "`new_layout.size()` must be less than or equal to `old_layout.size()`"
        );

        // Allocate new memory
        let new_ptr = self.allocate(new_layout)?;

        // Copy old memory to new memory
        core::ptr::copy_nonoverlapping(
            ptr.as_ptr(),
            new_ptr.as_ptr() as *mut u8,
            new_layout.size(),
        );

        // Deallocate old memory
        self.deallocate(ptr, old_layout);

        Ok(new_ptr)
    }

    /// Check if allocator supports growing in place
    fn grows_in_place(&self) -> bool {
        false
    }

    /// Check if allocator supports shrinking in place
    fn shrinks_in_place(&self) -> bool {
        false
    }
}

/// Allocation error
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocError {
    /// Out of memory
    OutOfMemory,
    /// Invalid layout
    InvalidLayout,
}

/// Null allocator that always returns out of memory
pub struct NullAllocator;

impl Allocator for NullAllocator {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Err(AllocError::OutOfMemory)
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Do nothing
    }
}

/// Fixed size allocator that allocates fixed size blocks
pub struct FixedSizeAllocator {
    /// Block size
    block_size: usize,
    /// Inner allocator
    inner: &'static dyn Allocator,
}

impl FixedSizeAllocator {
    /// Create new fixed size allocator
    pub const fn new(block_size: usize, inner: &'static dyn Allocator) -> Self {
        FixedSizeAllocator {
            block_size,
            inner,
        }
    }
}

impl Allocator for FixedSizeAllocator {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // Calculate block count
        let size = (layout.size() + self.block_size - 1) / self.block_size * self.block_size;
        let align = layout.align().max(self.block_size);

        // Create new layout
        let layout = Layout::from_size_align(size, align).map_err(|_| AllocError::InvalidLayout)?;

        // Allocate memory
        self.inner.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // Calculate block count
        let size = (layout.size() + self.block_size - 1) / self.block_size * self.block_size;
        let align = layout.align().max(self.block_size);

        // Create new layout
        let layout = Layout::from_size_align(size, align).unwrap();

        // Deallocate memory
        self.inner.deallocate(ptr, layout);
    }
}
