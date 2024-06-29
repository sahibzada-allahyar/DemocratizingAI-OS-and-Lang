//! Main entry point for the Democratising compiler
//!
//! This module provides the compiler driver that coordinates
//! the different compilation phases.

use crate::codegen::CodeGenerator;
use crate::error::Result;
use crate::ir::IrBuilder;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::semantic::SemanticAnalyzer;
use inkwell::context::Context;
use inkwell::targets::{InitializationConfig, Target};
use std::path::Path;
use std::fs;

/// Compile a source file to an executable
pub fn compile_file(path: &Path) -> Result<()> {
    // Read source file
    let source = fs::read_to_string(path)?;
    let file_name = path.file_name().unwrap().to_string_lossy();

    // Initialize LLVM
    Target::initialize_all(&InitializationConfig::default());

    // Create LLVM context
    let context = Context::create();

    // Create compiler instance
    let mut compiler = Compiler::new(&context, &file_name);

    // Compile the source
    compiler.compile(&source)?;

    // Write output
    let output_path = path.with_extension("o");
    compiler.write_object_file(&output_path)?;

    Ok(())
}

/// The compiler driver
pub struct Compiler<'ctx> {
    /// LLVM context
    context: &'ctx Context,
    /// File name being compiled
    file_name: String,
    /// Semantic analyzer
    analyzer: SemanticAnalyzer,
    /// IR builder
    ir_builder: IrBuilder,
    /// Code generator
    codegen: CodeGenerator<'ctx>,
}

impl<'ctx> Compiler<'ctx> {
    /// Create a new compiler instance
    pub fn new(context: &'ctx Context, file_name: &str) -> Self {
        Self {
            context,
            file_name: file_name.to_string(),
            analyzer: SemanticAnalyzer::new(),
            ir_builder: IrBuilder::new(),
            codegen: CodeGenerator::new(context, file_name),
        }
    }

    /// Compile source code to LLVM IR
    pub fn compile(&mut self, source: &str) -> Result<()> {
        // Lexical analysis
        let mut lexer = Lexer::new(source, &self.file_name);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token()?;
            let is_eof = token.kind == crate::lexer::TokenKind::EOF;
            tokens.push(token);
            if is_eof {
                break;
            }
        }

        // Parsing
        let mut parser = Parser::new(tokens, &self.file_name);
        let ast = parser.parse_program()?;

        // Semantic analysis
        self.analyzer.check_program(&ast)?;

        // IR generation
        // TODO: Convert AST to IR
        // For now, we'll just create a simple "main" function that returns 0
        self.ir_builder.start_function(
            "main".to_string(),
            vec![],
            crate::ir::Type::Int32,
        );
        let result = self.ir_builder.new_value();
        self.ir_builder.add_instruction(crate::ir::Instruction::Binary {
            op: crate::ir::BinaryOp::Add,
            result,
            left: self.ir_builder.new_value(),
            right: self.ir_builder.new_value(),
        })?;
        self.ir_builder.set_terminator(crate::ir::Terminator::Return(Some(result)))?;
        let function = self.ir_builder.finish_function()?;

        // Code generation
        self.codegen.gen_function(&function)?;
        self.codegen.finalize()?;

        Ok(())
    }

    /// Write LLVM IR to an object file
    pub fn write_object_file(&self, path: &Path) -> Result<()> {
        let target_triple = inkwell::targets::TargetMachine::get_default_triple();
        let target = Target::from_triple(&target_triple)
            .map_err(|e| crate::error::CompilerError::internal_error(e.to_string()))?;

        let target_machine = target
            .create_target_machine(
                &target_triple,
                "generic",
                "",
                inkwell::OptimizationLevel::Default,
                inkwell::targets::RelocMode::Default,
                inkwell::targets::CodeModel::Default,
            )
            .ok_or_else(|| {
                crate::error::CompilerError::internal_error("Failed to create target machine")
            })?;

        target_machine
            .write_to_file(
                self.codegen.get_module(),
                inkwell::targets::FileType::Object,
                path,
            )
            .map_err(|e| crate::error::CompilerError::internal_error(e.to_string()))?;

        Ok(())
    }
}

fn main() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <source_file>", args[0]);
        std::process::exit(1);
    }

    // Compile the file
    let path = Path::new(&args[1]);
    if let Err(e) = compile_file(path) {
        eprintln!("Compilation error: {}", e);
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    #[test]
    fn test_compile_simple_program() {
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let source_path = dir.path().join("test.dem");
        let object_path = dir.path().join("test.o");

        // Write a simple program
        let mut file = File::create(&source_path).unwrap();
        writeln!(
            file,
            r#"
            fn main() -> i32 {{
                return 0;
            }}
            "#
        )
        .unwrap();

        // Compile it
        compile_file(&source_path).unwrap();

        // Check that the object file was created
        assert!(object_path.exists());
    }
}
