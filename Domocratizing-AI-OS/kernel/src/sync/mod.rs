//! Synchronization primitives

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicBool, Ordering};

/// Mutex guard
pub struct MutexGuard<'a, T: ?Sized> {
    /// Mutex reference
    mutex: &'a Mutex<T>,
    /// Guard is active
    active: bool,
}

impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        if self.active {
            self.mutex.unlock();
        }
    }
}

impl<'a, T: ?Sized> Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.mutex.data.get() }
    }
}

impl<'a, T: ?Sized> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.mutex.data.get() }
    }
}

/// Mutex
pub struct Mutex<T: ?Sized> {
    /// Mutex is locked
    locked: AtomicBool,
    /// Protected data
    data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Send> Send for Mutex<T> {}
unsafe impl<T: ?Sized + Send> Sync for Mutex<T> {}

impl<T> Mutex<T> {
    /// Create new mutex
    pub const fn new(data: T) -> Self {
        Mutex {
            locked: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }
}

impl<T: ?Sized> Mutex<T> {
    /// Lock mutex
    pub fn lock(&self) -> MutexGuard<T> {
        // Spin until we can set the lock
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            // Wait for event
            unsafe {
                crate::arch::wait_for_event();
            }
        }

        MutexGuard {
            mutex: self,
            active: true,
        }
    }

    /// Try to lock mutex
    pub fn try_lock(&self) -> Option<MutexGuard<T>> {
        if self
            .locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            Some(MutexGuard {
                mutex: self,
                active: true,
            })
        } else {
            None
        }
    }

    /// Unlock mutex
    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
        // Send event
        unsafe {
            crate::arch::send_event();
        }
    }

    /// Get raw pointer to data
    pub fn as_ptr(&self) -> *mut T {
        self.data.get()
    }

    /// Get mutable reference to data
    ///
    /// # Safety
    ///
    /// This is unsafe because it allows mutable access to the protected data
    /// without locking the mutex.
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.data.get()
    }
}

/// RwLock guard
pub struct RwLockReadGuard<'a, T: ?Sized> {
    /// RwLock reference
    lock: &'a RwLock<T>,
    /// Guard is active
    active: bool,
}

impl<'a, T: ?Sized> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        if self.active {
            self.lock.read_unlock();
        }
    }
}

impl<'a, T: ?Sized> Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

/// RwLock write guard
pub struct RwLockWriteGuard<'a, T: ?Sized> {
    /// RwLock reference
    lock: &'a RwLock<T>,
    /// Guard is active
    active: bool,
}

impl<'a, T: ?Sized> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        if self.active {
            self.lock.write_unlock();
        }
    }
}

impl<'a, T: ?Sized> Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.lock.data.get() }
    }
}

impl<'a, T: ?Sized> DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.lock.data.get() }
    }
}

/// Read-write lock
pub struct RwLock<T: ?Sized> {
    /// Number of readers
    readers: AtomicBool,
    /// Writer is active
    writer: AtomicBool,
    /// Protected data
    data: UnsafeCell<T>,
}

unsafe impl<T: ?Sized + Send> Send for RwLock<T> {}
unsafe impl<T: ?Sized + Send + Sync> Sync for RwLock<T> {}

impl<T> RwLock<T> {
    /// Create new read-write lock
    pub const fn new(data: T) -> Self {
        RwLock {
            readers: AtomicBool::new(false),
            writer: AtomicBool::new(false),
            data: UnsafeCell::new(data),
        }
    }
}

impl<T: ?Sized> RwLock<T> {
    /// Read lock
    pub fn read(&self) -> RwLockReadGuard<T> {
        // Wait for writer to finish
        while self.writer.load(Ordering::Relaxed) {
            unsafe {
                crate::arch::wait_for_event();
            }
        }

        // Set reader flag
        self.readers.store(true, Ordering::Acquire);

        RwLockReadGuard {
            lock: self,
            active: true,
        }
    }

    /// Write lock
    pub fn write(&self) -> RwLockWriteGuard<T> {
        // Wait for writer to finish
        while self.writer.load(Ordering::Relaxed) {
            unsafe {
                crate::arch::wait_for_event();
            }
        }

        // Wait for readers to finish
        while self.readers.load(Ordering::Relaxed) {
            unsafe {
                crate::arch::wait_for_event();
            }
        }

        // Set writer flag
        self.writer.store(true, Ordering::Acquire);

        RwLockWriteGuard {
            lock: self,
            active: true,
        }
    }

    /// Read unlock
    fn read_unlock(&self) {
        self.readers.store(false, Ordering::Release);
        // Send event
        unsafe {
            crate::arch::send_event();
        }
    }

    /// Write unlock
    fn write_unlock(&self) {
        self.writer.store(false, Ordering::Release);
        // Send event
        unsafe {
            crate::arch::send_event();
        }
    }

    /// Get raw pointer to data
    pub fn as_ptr(&self) -> *mut T {
        self.data.get()
    }

    /// Get mutable reference to data
    ///
    /// # Safety
    ///
    /// This is unsafe because it allows mutable access to the protected data
    /// without locking the RwLock.
    pub unsafe fn get_mut(&self) -> &mut T {
        &mut *self.data.get()
    }
}
