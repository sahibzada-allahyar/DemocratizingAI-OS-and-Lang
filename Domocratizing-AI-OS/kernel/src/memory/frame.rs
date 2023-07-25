//! Physical frame allocator

use core::sync::atomic::{AtomicUsize, Ordering};
use spin::Mutex;

use super::PhysAddr;

/// Frame size (4KB)
pub const FRAME_SIZE: usize = 4096;

/// Frame
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Frame {
    /// Frame number
    number: usize,
}

impl Frame {
    /// Create new frame
    pub const fn new(number: usize) -> Self {
        Frame { number }
    }

    /// Get frame containing address
    pub fn containing_address(address: PhysAddr) -> Self {
        Frame {
            number: address.as_usize() / FRAME_SIZE,
        }
    }

    /// Get frame number
    pub fn number(&self) -> usize {
        self.number
    }

    /// Get start address
    pub fn start_address(&self) -> PhysAddr {
        PhysAddr::new(self.number * FRAME_SIZE)
    }

    /// Get end address
    pub fn end_address(&self) -> PhysAddr {
        PhysAddr::new((self.number + 1) * FRAME_SIZE)
    }

    /// Allocate frame
    pub fn allocate() -> Option<Self> {
        FRAME_ALLOCATOR.lock().allocate()
    }

    /// Deallocate frame
    pub fn deallocate(&self) {
        FRAME_ALLOCATOR.lock().deallocate(self);
    }
}

/// Frame allocator
pub struct FrameAllocator {
    /// Next free frame
    next_free_frame: AtomicUsize,
    /// Number of frames
    frame_count: usize,
}

impl FrameAllocator {
    /// Create new frame allocator
    pub const fn new() -> Self {
        FrameAllocator {
            next_free_frame: AtomicUsize::new(0),
            frame_count: 0,
        }
    }

    /// Initialize frame allocator
    pub fn init(&mut self, start_frame: usize, frame_count: usize) {
        self.next_free_frame.store(start_frame, Ordering::SeqCst);
        self.frame_count = frame_count;
    }

    /// Allocate frame
    pub fn allocate(&self) -> Option<Frame> {
        let frame_number = self.next_free_frame.fetch_add(1, Ordering::SeqCst);
        if frame_number >= self.frame_count {
            // Out of frames
            None
        } else {
            Some(Frame::new(frame_number))
        }
    }

    /// Deallocate frame
    pub fn deallocate(&self, frame: &Frame) {
        // TODO: Implement frame deallocation
        // For now, we just leak frames
        let _ = frame;
    }

    /// Get number of allocated frames
    pub fn allocated_frames(&self) -> usize {
        self.next_free_frame.load(Ordering::SeqCst)
    }

    /// Get total number of frames
    pub fn total_frames(&self) -> usize {
        self.frame_count
    }

    /// Get number of free frames
    pub fn free_frames(&self) -> usize {
        self.frame_count - self.allocated_frames()
    }
}

/// Global frame allocator
static FRAME_ALLOCATOR: Mutex<FrameAllocator> = Mutex::new(FrameAllocator::new());

/// Initialize frame allocator
pub fn init() {
    // Calculate available memory
    let memory_end = 0x4000_0000; // 1GB
    let kernel_end = 0x4010_0000; // Kernel end address

    // Calculate frame range
    let start_frame = (kernel_end + FRAME_SIZE - 1) / FRAME_SIZE;
    let end_frame = memory_end / FRAME_SIZE;
    let frame_count = end_frame - start_frame;

    // Initialize frame allocator
    FRAME_ALLOCATOR.lock().init(start_frame, frame_count);
}

/// Get number of allocated frames
pub fn allocated_frames() -> usize {
    FRAME_ALLOCATOR.lock().allocated_frames()
}

/// Get total number of frames
pub fn total_frames() -> usize {
    FRAME_ALLOCATOR.lock().total_frames()
}

/// Get number of free frames
pub fn free_frames() -> usize {
    FRAME_ALLOCATOR.lock().free_frames()
}
