use crate::core::{
    error::{Error, Result},
    traits::Differentiable,
    types::Shape,
};
use std::sync::{Arc, RwLock};

/// A node in the computation graph
#[derive(Debug)]
pub struct Node<T> {
    /// Data stored in the node
    data: T,
    /// Gradient with respect to output
    grad: Arc<RwLock<Option<T>>>,
    /// Backward function
    backward_fn: Option<Box<dyn FnOnce() -> Result<()> + Send + Sync>>,
    /// Requires gradient computation
    requires_grad: bool,
}

impl<T> Node<T>
where
    T: Differentiable + Send + Sync + 'static,
{
    /// Create a new node
    pub fn new(data: T) -> Self {
        Self {
            data,
            grad: Arc::new(RwLock::new(None)),
            backward_fn: None,
            requires_grad: false,
        }
    }

    /// Create a new node with gradient tracking
    pub fn with_grad(data: T) -> Self {
        Self {
            data,
            grad: Arc::new(RwLock::new(None)),
            backward_fn: None,
            requires_grad: true,
        }
    }

    /// Get the data
    pub fn data(&self) -> &T {
        &self.data
    }

    /// Get the gradient
    pub fn grad(&self) -> Option<T> {
        self.grad.read().unwrap().clone()
    }

    /// Set requires gradient
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Check if requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set the backward function
    pub fn set_backward_fn<F>(&mut self, f: F)
    where
        F: FnOnce() -> Result<()> + Send + Sync + 'static,
    {
        self.backward_fn = Some(Box::new(f));
    }

    /// Backward pass
    pub fn backward(&mut self) -> Result<()> {
        if !self.requires_grad {
            return Ok(());
        }

        // Initialize gradient if None
        if self.grad.read().unwrap().is_none() {
            *self.grad.write().unwrap() = Some(T::one());
        }

        // Call backward function if exists
        if let Some(backward_fn) = self.backward_fn.take() {
            backward_fn()?;
        }

        Ok(())
    }

    /// Zero gradient
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            *self.grad.write().unwrap() = None;
        }
    }
}

/// Tape for recording operations for automatic differentiation
#[derive(Debug, Default)]
pub struct Tape {
    /// Nodes in the computation graph
    nodes: Vec<Box<dyn FnOnce() -> Result<()> + Send + Sync>>,
}

impl Tape {
    /// Create a new tape
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Record an operation
    pub fn record<F>(&mut self, f: F)
    where
        F: FnOnce() -> Result<()> + Send + Sync + 'static,
    {
        self.nodes.push(Box::new(f));
    }

    /// Backward pass
    pub fn backward(&mut self) -> Result<()> {
        while let Some(node) = self.nodes.pop() {
            node()?;
        }
        Ok(())
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.nodes.clear();
    }
}

/// Context for automatic differentiation
#[derive(Debug, Default)]
pub struct Context {
    /// Current tape
    tape: Arc<RwLock<Tape>>,
    /// Whether gradient computation is enabled
    enabled: bool,
}

impl Context {
    /// Create a new context
    pub fn new() -> Self {
        Self {
            tape: Arc::new(RwLock::new(Tape::new())),
            enabled: true,
        }
    }

    /// Enable gradient computation
    pub fn enable_grad(&mut self) {
        self.enabled = true;
    }

    /// Disable gradient computation
    pub fn disable_grad(&mut self) {
        self.enabled = false;
    }

    /// Check if gradient computation is enabled
    pub fn is_grad_enabled(&self) -> bool {
        self.enabled
    }

    /// Record an operation
    pub fn record<F>(&self, f: F)
    where
        F: FnOnce() -> Result<()> + Send + Sync + 'static,
    {
        if self.enabled {
            self.tape.write().unwrap().record(f);
        }
    }

    /// Backward pass
    pub fn backward(&self) -> Result<()> {
        if self.enabled {
            self.tape.write().unwrap().backward()?;
        }
        Ok(())
    }

    /// Clear the tape
    pub fn clear(&self) {
        if self.enabled {
            self.tape.write().unwrap().clear();
        }
    }
}

/// Global context for automatic differentiation
static CONTEXT: once_cell::sync::Lazy<RwLock<Context>> =
    once_cell::sync::Lazy::new(|| RwLock::new(Context::new()));

/// Get the global context
pub fn get_context() -> std::sync::RwLockReadGuard<'static, Context> {
    CONTEXT.read().unwrap()
}

/// Update the global context
pub fn update_context<F>(f: F)
where
    F: FnOnce(&mut Context),
{
    let mut context = CONTEXT.write().unwrap();
    f(&mut context);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node() {
        let mut node = Node::with_grad(1.0f32);
        assert!(node.requires_grad());
        assert_eq!(node.data(), &1.0);
        assert!(node.grad().is_none());

        node.set_backward_fn(|| {
            *node.grad.write().unwrap() = Some(2.0);
            Ok(())
        });

        node.backward().unwrap();
        assert_eq!(node.grad(), Some(1.0));

        node.zero_grad();
        assert!(node.grad().is_none());
    }

    #[test]
    fn test_tape() {
        let mut tape = Tape::new();
        let mut value = 0;

        tape.record(|| {
            value += 1;
            Ok(())
        });

        tape.record(|| {
            value *= 2;
            Ok(())
        });

        tape.backward().unwrap();
        assert_eq!(value, 2);

        tape.clear();
        assert!(tape.nodes.is_empty());
    }

    #[test]
    fn test_context() {
        let context = Context::new();
        assert!(context.is_grad_enabled());

        let mut value = 0;
        context.record(|| {
            value += 1;
            Ok(())
        });

        context.backward().unwrap();
        assert_eq!(value, 1);

        context.clear();
        assert!(context.tape.read().unwrap().nodes.is_empty());
    }

    #[test]
    fn test_global_context() {
        let context = get_context();
        assert!(context.is_grad_enabled());

        update_context(|context| {
            context.disable_grad();
        });

        let context = get_context();
        assert!(!context.is_grad_enabled());

        update_context(|context| {
            context.enable_grad();
        });
    }
}
