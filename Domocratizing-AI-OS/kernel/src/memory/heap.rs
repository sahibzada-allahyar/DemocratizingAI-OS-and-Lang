//! Kernel heap allocator

use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;

use super::{PhysAddr, VirtAddr};
use super::frame::Frame;

/// Heap allocator
pub struct HeapAllocator {
    /// Heap start
    start: VirtAddr,
    /// Heap size
    size: usize,
    /// Allocated bytes
    allocated: AtomicUsize,
    /// Next allocation position
    next: AtomicUsize,
}

impl HeapAllocator {
    /// Create new heap allocator
    pub const fn new() -> Self {
        HeapAllocator {
            start: VirtAddr::new(0),
            size: 0,
            allocated: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
        }
    }

    /// Initialize heap allocator
    pub fn init(&mut self, start: VirtAddr, size: usize) {
        self.start = start;
        self.size = size;
        self.allocated.store(0, Ordering::SeqCst);
        self.next.store(0, Ordering::SeqCst);
    }

    /// Allocate memory
    pub fn allocate(&self, layout: Layout) -> *mut u8 {
        // Calculate aligned size
        let size = (layout.size() + layout.align() - 1) & !(layout.align() - 1);

        // Get next allocation position
        let pos = self.next.fetch_add(size, Ordering::SeqCst);
        if pos + size > self.size {
            // Out of memory
            self.next.fetch_sub(size, Ordering::SeqCst);
            null_mut()
        } else {
            // Update allocated bytes
            self.allocated.fetch_add(size, Ordering::SeqCst);

            // Return pointer
            (self.start.as_usize() + pos) as *mut u8
        }
    }

    /// Deallocate memory
    pub fn deallocate(&self, _ptr: *mut u8, layout: Layout) {
        // Calculate aligned size
        let size = (layout.size() + layout.align() - 1) & !(layout.align() - 1);

        // Update allocated bytes
        self.allocated.fetch_sub(size, Ordering::SeqCst);
    }

    /// Get allocated bytes
    pub fn allocated_bytes(&self) -> usize {
        self.allocated.load(Ordering::SeqCst)
    }

    /// Get total bytes
    pub fn total_bytes(&self) -> usize {
        self.size
    }

    /// Get free bytes
    pub fn free_bytes(&self) -> usize {
        self.size - self.allocated_bytes()
    }
}

unsafe impl GlobalAlloc for HeapAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.allocate(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.deallocate(ptr, layout)
    }
}

/// Global heap allocator
#[global_allocator]
static HEAP_ALLOCATOR: Mutex<HeapAllocator> = Mutex::new(HeapAllocator::new());

/// Initialize heap
pub fn init() {
    // Calculate heap range
    let heap_start = 0x4020_0000; // After kernel
    let heap_size = 0x0100_0000; // 16MB

    // Map heap frames
    let start_frame = Frame::containing_address(PhysAddr::new(heap_start));
    let end_frame = Frame::containing_address(PhysAddr::new(heap_start + heap_size - 1));
    for frame in (start_frame.number()..=end_frame.number()).map(Frame::new) {
        // TODO: Map frame
        let _ = frame;
    }

    // Initialize heap allocator
    HEAP_ALLOCATOR
        .lock()
        .init(VirtAddr::new(heap_start), heap_size);
}

/// Get allocated bytes
pub fn allocated_bytes() -> usize {
    HEAP_ALLOCATOR.lock().allocated_bytes()
}

/// Get total bytes
pub fn total_bytes() -> usize {
    HEAP_ALLOCATOR.lock().total_bytes()
}

/// Get free bytes
pub fn free_bytes() -> usize {
    HEAP_ALLOCATOR.lock().free_bytes()
}
