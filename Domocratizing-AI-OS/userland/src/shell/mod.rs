//! Shell application

use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use spin::Mutex;

use crate::{Application, UserlandCapabilities};

/// Shell capabilities
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct ShellCapabilities: u32 {
        /// Supports command execution
        const EXECUTE = 1 << 0;
        /// Supports command history
        const HISTORY = 1 << 1;
        /// Supports command completion
        const COMPLETION = 1 << 2;
        /// Supports command aliases
        const ALIASES = 1 << 3;
        /// Supports command scripting
        const SCRIPTING = 1 << 4;
        /// Supports command pipes
        const PIPES = 1 << 5;
        /// Supports command redirection
        const REDIRECTION = 1 << 6;
        /// Supports command substitution
        const SUBSTITUTION = 1 << 7;
        /// Supports command variables
        const VARIABLES = 1 << 8;
        /// Supports command functions
        const FUNCTIONS = 1 << 9;
        /// Supports command loops
        const LOOPS = 1 << 10;
        /// Supports command conditions
        const CONDITIONS = 1 << 11;
        /// Supports command arithmetic
        const ARITHMETIC = 1 << 12;
        /// Supports command arrays
        const ARRAYS = 1 << 13;
        /// Supports command associative arrays
        const ASSOCIATIVE = 1 << 14;
        /// Supports command debugging
        const DEBUG = 1 << 15;
    }
}

/// Shell command
pub struct ShellCommand {
    /// Command name
    name: String,
    /// Command description
    description: String,
    /// Command usage
    usage: String,
    /// Command examples
    examples: Vec<String>,
    /// Command capabilities
    capabilities: ShellCapabilities,
}

/// Shell application
pub struct ShellApplication {
    /// Application name
    name: String,
    /// Application version
    version: String,
    /// Application capabilities
    capabilities: UserlandCapabilities,
    /// Shell capabilities
    shell_capabilities: ShellCapabilities,
    /// Shell commands
    commands: Vec<ShellCommand>,
}

impl ShellApplication {
    /// Create new shell application
    pub fn new() -> Self {
        ShellApplication {
            name: String::from("shell"),
            version: String::from("0.1.0"),
            capabilities: UserlandCapabilities::all(),
            shell_capabilities: ShellCapabilities::all(),
            commands: Vec::new(),
        }
    }

    /// Get shell capabilities
    pub fn shell_capabilities(&self) -> ShellCapabilities {
        self.shell_capabilities
    }

    /// Get shell commands
    pub fn commands(&self) -> &[ShellCommand] {
        &self.commands
    }

    /// Add shell command
    pub fn add_command(&mut self, command: ShellCommand) {
        self.commands.push(command);
    }

    /// Remove shell command
    pub fn remove_command(&mut self, name: &str) {
        if let Some(index) = self.commands.iter().position(|c| c.name == name) {
            self.commands.remove(index);
        }
    }

    /// Get shell command by name
    pub fn get_command(&self, name: &str) -> Option<&ShellCommand> {
        self.commands.iter().find(|c| c.name == name)
    }

    /// Get shell commands by capability
    pub fn get_commands_by_capability(&self, capability: ShellCapabilities) -> Vec<&ShellCommand> {
        self.commands
            .iter()
            .filter(|c| c.capabilities.contains(capability))
            .collect()
    }
}

impl Application for ShellApplication {
    fn name(&self) -> &str {
        &self.name
    }

    fn version(&self) -> &str {
        &self.version
    }

    fn capabilities(&self) -> UserlandCapabilities {
        self.capabilities
    }

    fn start(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn stop(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn restart(&self) -> Result<(), &'static str> {
        self.stop()?;
        self.start()
    }

    fn pause(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn resume(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn update(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn configure(&self) -> Result<(), &'static str> {
        Ok(())
    }

    fn debug(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Global shell application
static SHELL_APPLICATION: Mutex<Option<Arc<ShellApplication>>> = Mutex::new(None);

/// Initialize shell application
pub fn init() {
    let application = Arc::new(ShellApplication::new());
    *SHELL_APPLICATION.lock() = Some(Arc::clone(&application));
    crate::register_application(&*application);
}

/// Get shell application
pub fn get_application() -> Option<Arc<ShellApplication>> {
    SHELL_APPLICATION.lock().as_ref().map(Arc::clone)
}
