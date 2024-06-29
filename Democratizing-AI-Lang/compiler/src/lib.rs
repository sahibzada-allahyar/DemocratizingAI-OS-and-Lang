//! Democratising Programming Language Compiler Library
//! 
//! This library implements the core compiler for the Democratising programming language,
//! a language designed to democratize AI development globally.

pub mod lexer;
pub mod parser;
pub mod ast;
pub mod semantic;
pub mod ir;
pub mod codegen;
pub mod error;
pub mod utils;

use error::CompilerError;

pub type Result<T> = std::result::Result<T, CompilerError>;

/// Version of the Democratising compiler
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Compile a source string into executable code
pub fn compile(source: &str) -> Result<Vec<u8>> {
    todo!("Implement full compilation pipeline")
}

/// Compile a source file into executable code
pub fn compile_file(path: &str) -> Result<Vec<u8>> {
    todo!("Implement file compilation")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
