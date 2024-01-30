//! AI subsystem

pub mod accelerator;
pub mod tensor;

use alloc::sync::Arc;
use spin::Mutex;

use crate::memory::VirtAddr;
use crate::scheduler::{TaskFlags, TaskPriority};

/// AI task
pub struct AiTask {
    /// Task ID
    id: usize,
    /// Task name
    name: String,
    /// Task priority
    priority: TaskPriority,
    /// Task flags
    flags: TaskFlags,
    /// Task input tensor
    input: Arc<Tensor>,
    /// Task output tensor
    output: Arc<Tensor>,
    /// Task accelerator
    accelerator: Arc<dyn Accelerator>,
}

impl AiTask {
    /// Create new AI task
    pub fn new(
        name: String,
        input: Arc<Tensor>,
        output: Arc<Tensor>,
        accelerator: Arc<dyn Accelerator>,
    ) -> Self {
        static NEXT_ID: Mutex<usize> = Mutex::new(1);
        let id = {
            let mut id = NEXT_ID.lock();
            let task_id = *id;
            *id += 1;
            task_id
        };

        AiTask {
            id,
            name,
            priority: TaskPriority::High,
            flags: TaskFlags::AI | TaskFlags::KERNEL,
            input,
            output,
            accelerator,
        }
    }

    /// Get task ID
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get task name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get task priority
    pub fn priority(&self) -> TaskPriority {
        self.priority
    }

    /// Get task flags
    pub fn flags(&self) -> TaskFlags {
        self.flags
    }

    /// Get input tensor
    pub fn input(&self) -> Arc<Tensor> {
        Arc::clone(&self.input)
    }

    /// Get output tensor
    pub fn output(&self) -> Arc<Tensor> {
        Arc::clone(&self.output)
    }

    /// Get accelerator
    pub fn accelerator(&self) -> Arc<dyn Accelerator> {
        Arc::clone(&self.accelerator)
    }

    /// Run task
    pub fn run(&self) -> Result<(), &'static str> {
        self.accelerator.run(self.input(), self.output())
    }
}

/// AI task queue
pub struct AiTaskQueue {
    /// Tasks
    tasks: Mutex<Vec<Arc<AiTask>>>,
}

impl AiTaskQueue {
    /// Create new AI task queue
    pub const fn new() -> Self {
        AiTaskQueue {
            tasks: Mutex::new(Vec::new()),
        }
    }

    /// Add task
    pub fn add_task(&self, task: Arc<AiTask>) {
        self.tasks.lock().push(task);
    }

    /// Get next task
    pub fn next_task(&self) -> Option<Arc<AiTask>> {
        self.tasks.lock().pop()
    }

    /// Get task count
    pub fn task_count(&self) -> usize {
        self.tasks.lock().len()
    }
}

/// Global AI task queue
static AI_TASK_QUEUE: AiTaskQueue = AiTaskQueue::new();

/// Initialize AI subsystem
pub fn init() {
    // Initialize accelerators
    accelerator::init();
}

/// Create AI task
pub fn create_task(
    name: String,
    input: Arc<Tensor>,
    output: Arc<Tensor>,
    accelerator: Arc<dyn Accelerator>,
) -> Arc<AiTask> {
    let task = Arc::new(AiTask::new(name, input, output, accelerator));
    AI_TASK_QUEUE.add_task(Arc::clone(&task));
    task
}

/// Get next AI task
pub fn next_task() -> Option<Arc<AiTask>> {
    AI_TASK_QUEUE.next_task()
}

/// Get AI task count
pub fn task_count() -> usize {
    AI_TASK_QUEUE.task_count()
}

use self::accelerator::Accelerator;
use self::tensor::Tensor;
