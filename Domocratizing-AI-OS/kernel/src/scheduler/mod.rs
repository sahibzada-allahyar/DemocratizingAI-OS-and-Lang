//! Task scheduler

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use spin::Mutex;

use crate::arch;
use crate::memory::VirtAddr;

/// Task state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    /// Task is ready to run
    Ready,
    /// Task is running
    Running,
    /// Task is blocked
    Blocked,
    /// Task is sleeping
    Sleeping,
    /// Task has exited
    Exited,
}

/// Task priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Real-time priority
    RealTime,
}

/// Task flags
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct TaskFlags: u32 {
        /// Task is kernel task
        const KERNEL = 1 << 0;
        /// Task is user task
        const USER = 1 << 1;
        /// Task is AI task
        const AI = 1 << 2;
        /// Task is idle task
        const IDLE = 1 << 3;
    }
}

/// Task
pub struct Task {
    /// Task ID
    id: usize,
    /// Task name
    name: &'static str,
    /// Task state
    state: Mutex<TaskState>,
    /// Task priority
    priority: TaskPriority,
    /// Task flags
    flags: TaskFlags,
    /// Task stack pointer
    stack_ptr: Mutex<VirtAddr>,
    /// Task is initialized
    initialized: AtomicBool,
    /// Task quantum
    quantum: AtomicU64,
    /// Task runtime
    runtime: AtomicU64,
    /// Task switches
    switches: AtomicUsize,
}

impl Task {
    /// Create new task
    pub fn new(
        id: usize,
        name: &'static str,
        priority: TaskPriority,
        flags: TaskFlags,
        stack_ptr: VirtAddr,
    ) -> Self {
        Task {
            id,
            name,
            state: Mutex::new(TaskState::Ready),
            priority,
            flags,
            stack_ptr: Mutex::new(stack_ptr),
            initialized: AtomicBool::new(false),
            quantum: AtomicU64::new(0),
            runtime: AtomicU64::new(0),
            switches: AtomicUsize::new(0),
        }
    }

    /// Get task ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get task name
    pub fn name(&self) -> &str {
        self.name
    }

    /// Get task state
    pub fn state(&self) -> TaskState {
        *self.state.lock()
    }

    /// Set task state
    pub fn set_state(&self, state: TaskState) {
        *self.state.lock() = state;
    }

    /// Get task priority
    pub fn priority(&self) -> TaskPriority {
        self.priority
    }

    /// Get task flags
    pub fn flags(&self) -> TaskFlags {
        self.flags
    }

    /// Get task stack pointer
    pub fn stack_ptr(&self) -> VirtAddr {
        *self.stack_ptr.lock()
    }

    /// Set task stack pointer
    pub fn set_stack_ptr(&self, ptr: VirtAddr) {
        *self.stack_ptr.lock() = ptr;
    }

    /// Is task initialized?
    pub fn is_initialized(&self) -> bool {
        self.initialized.load(Ordering::Relaxed)
    }

    /// Get task quantum
    pub fn quantum(&self) -> u64 {
        self.quantum.load(Ordering::Relaxed)
    }

    /// Set task quantum
    pub fn set_quantum(&self, quantum: u64) {
        self.quantum.store(quantum, Ordering::Relaxed);
    }

    /// Get task runtime
    pub fn runtime(&self) -> u64 {
        self.runtime.load(Ordering::Relaxed)
    }

    /// Add task runtime
    pub fn add_runtime(&self, runtime: u64) {
        self.runtime.fetch_add(runtime, Ordering::Relaxed);
    }

    /// Get task switches
    pub fn switches(&self) -> usize {
        self.switches.load(Ordering::Relaxed)
    }

    /// Increment task switches
    pub fn increment_switches(&self) {
        self.switches.fetch_add(1, Ordering::Relaxed);
    }

    /// Initialize task
    pub fn init(&self) -> Result<(), &'static str> {
        if !self.initialized.load(Ordering::Relaxed) {
            // Initialize task
            self.initialized.store(true, Ordering::Release);
        }
        Ok(())
    }
}

/// Task scheduler
pub struct Scheduler {
    /// Next task ID
    next_id: AtomicUsize,
    /// Ready queue
    ready_queue: Mutex<VecDeque<Arc<Task>>>,
    /// Current task
    current_task: Mutex<Option<Arc<Task>>>,
    /// Idle task
    idle_task: Mutex<Option<Arc<Task>>>,
}

impl Scheduler {
    /// Create new scheduler
    pub const fn new() -> Self {
        Scheduler {
            next_id: AtomicUsize::new(1),
            ready_queue: Mutex::new(VecDeque::new()),
            current_task: Mutex::new(None),
            idle_task: Mutex::new(None),
        }
    }

    /// Create task
    pub fn create_task(
        &self,
        name: &'static str,
        priority: TaskPriority,
        flags: TaskFlags,
        stack_ptr: VirtAddr,
    ) -> Arc<Task> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let task = Arc::new(Task::new(id, name, priority, flags, stack_ptr));

        // Add task to ready queue
        if flags.contains(TaskFlags::IDLE) {
            *self.idle_task.lock() = Some(Arc::clone(&task));
        } else {
            self.ready_queue.lock().push_back(Arc::clone(&task));
        }

        task
    }

    /// Schedule next task
    pub fn schedule(&self) {
        let mut current_task = self.current_task.lock();
        let mut ready_queue = self.ready_queue.lock();

        // Get next task
        let next_task = if let Some(task) = ready_queue.pop_front() {
            task
        } else if let Some(task) = self.idle_task.lock().as_ref() {
            Arc::clone(task)
        } else {
            return;
        };

        // Switch tasks
        if let Some(task) = current_task.take() {
            // Save current task
            task.set_stack_ptr(unsafe { VirtAddr::new(arch::get_sp() as usize) });
            task.set_state(TaskState::Ready);
            ready_queue.push_back(task);
        }

        // Load next task
        next_task.set_state(TaskState::Running);
        next_task.increment_switches();
        unsafe {
            arch::set_sp(next_task.stack_ptr().as_u64());
        }
        *current_task = Some(next_task);
    }

    /// Get current task
    pub fn current_task(&self) -> Option<Arc<Task>> {
        self.current_task.lock().as_ref().map(Arc::clone)
    }

    /// Get idle task
    pub fn idle_task(&self) -> Option<Arc<Task>> {
        self.idle_task.lock().as_ref().map(Arc::clone)
    }

    /// Get task count
    pub fn task_count(&self) -> usize {
        self.ready_queue.lock().len()
            + self.current_task.lock().is_some() as usize
            + self.idle_task.lock().is_some() as usize
    }
}

/// Global scheduler
static SCHEDULER: Scheduler = Scheduler::new();

/// Initialize scheduler
pub fn init() {
    // Create idle task
    SCHEDULER.create_task(
        "idle",
        TaskPriority::Low,
        TaskFlags::KERNEL | TaskFlags::IDLE,
        VirtAddr::new(0),
    );
}

/// Create task
pub fn create_task(
    name: &'static str,
    priority: TaskPriority,
    flags: TaskFlags,
    stack_ptr: VirtAddr,
) -> Arc<Task> {
    SCHEDULER.create_task(name, priority, flags, stack_ptr)
}

/// Schedule next task
pub fn schedule() {
    SCHEDULER.schedule();
}

/// Get current task
pub fn current_task() -> Option<Arc<Task>> {
    SCHEDULER.current_task()
}

/// Get idle task
pub fn idle_task() -> Option<Arc<Task>> {
    SCHEDULER.idle_task()
}

/// Get task count
pub fn task_count() -> usize {
    SCHEDULER.task_count()
}
