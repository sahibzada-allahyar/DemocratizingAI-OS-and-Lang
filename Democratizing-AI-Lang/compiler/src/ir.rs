//! Intermediate Representation for the Democratising compiler
//!
//! This module defines the IR used for optimization and code generation.
//! The IR is designed to be both high-level enough to perform AI-specific
//! optimizations and low-level enough to generate efficient code.

use crate::error::Result;
use std::collections::HashMap;

/// A unique identifier for a value in the IR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(pub usize);

/// A unique identifier for a basic block in the IR
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub usize);

/// A function in the IR
#[derive(Debug)]
pub struct Function {
    /// Name of the function
    pub name: String,
    /// Parameters to the function
    pub params: Vec<Parameter>,
    /// Return type
    pub return_type: Type,
    /// Basic blocks making up the function body
    pub blocks: HashMap<BlockId, BasicBlock>,
    /// Entry block
    pub entry: BlockId,
}

/// A basic block in the IR
#[derive(Debug)]
pub struct BasicBlock {
    /// Instructions in the block
    pub instructions: Vec<Instruction>,
    /// Terminator instruction
    pub terminator: Terminator,
}

/// A parameter to a function
#[derive(Debug, Clone)]
pub struct Parameter {
    /// Name of the parameter
    pub name: String,
    /// Type of the parameter
    pub ty: Type,
}

/// Types in the IR
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Void type
    Void,
    /// Boolean type
    Bool,
    /// Integer types
    Int8,
    Int16,
    Int32,
    Int64,
    /// Floating point types
    Float32,
    Float64,
    /// Array type with element type and size
    Array(Box<Type>, usize),
    /// Tensor type with element type and shape
    Tensor(Box<Type>, Vec<usize>),
    /// Function type
    Function {
        params: Vec<Type>,
        return_type: Box<Type>,
    },
    /// Structure type
    Struct(Vec<Type>),
    /// Pointer type
    Pointer(Box<Type>),
}

/// An instruction in the IR
#[derive(Debug)]
pub enum Instruction {
    /// Binary operation
    Binary {
        op: BinaryOp,
        result: ValueId,
        left: ValueId,
        right: ValueId,
    },
    /// Unary operation
    Unary {
        op: UnaryOp,
        result: ValueId,
        operand: ValueId,
    },
    /// Function call
    Call {
        result: ValueId,
        function: ValueId,
        arguments: Vec<ValueId>,
    },
    /// Load from memory
    Load {
        result: ValueId,
        pointer: ValueId,
    },
    /// Store to memory
    Store {
        value: ValueId,
        pointer: ValueId,
    },
    /// Get element pointer
    GetElementPtr {
        result: ValueId,
        base: ValueId,
        indices: Vec<ValueId>,
    },
    /// Allocate on stack
    Alloca {
        result: ValueId,
        ty: Type,
    },
    /// Cast between types
    Cast {
        result: ValueId,
        value: ValueId,
        target_type: Type,
    },
    /// AI-specific instructions
    TensorOp {
        op: TensorOp,
        result: ValueId,
        operands: Vec<ValueId>,
    },
}

/// A terminator instruction that ends a basic block
#[derive(Debug)]
pub enum Terminator {
    /// Return from function
    Return(Option<ValueId>),
    /// Unconditional branch
    Branch(BlockId),
    /// Conditional branch
    CondBranch {
        condition: ValueId,
        true_block: BlockId,
        false_block: BlockId,
    },
    /// Switch on a value
    Switch {
        value: ValueId,
        default: BlockId,
        cases: Vec<(i64, BlockId)>,
    },
}

/// Binary operations
#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    // Bitwise
    And,
    Or,
    Xor,
    Shl,
    Shr,
    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Unary operations
#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

/// Tensor operations
#[derive(Debug, Clone)]
pub enum TensorOp {
    // Basic operations
    MatMul,
    Transpose,
    Reshape { shape: Vec<usize> },
    // Neural network operations
    Conv2D {
        stride: (usize, usize),
        padding: (usize, usize),
    },
    MaxPool2D {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    // Activation functions
    ReLU,
    Sigmoid,
    Tanh,
    // Gradient operations
    Gradient {
        with_respect_to: Vec<ValueId>,
    },
}

/// The IR builder helps construct IR
pub struct IrBuilder {
    /// Next available value ID
    next_value: usize,
    /// Next available block ID
    next_block: usize,
    /// Current function being built
    current_function: Option<Function>,
    /// Current block being built
    current_block: Option<BlockId>,
}

impl IrBuilder {
    /// Create a new IR builder
    pub fn new() -> Self {
        Self {
            next_value: 0,
            next_block: 0,
            current_function: None,
            current_block: None,
        }
    }

    /// Create a new value ID
    pub fn new_value(&mut self) -> ValueId {
        let id = self.next_value;
        self.next_value += 1;
        ValueId(id)
    }

    /// Create a new block ID
    pub fn new_block(&mut self) -> BlockId {
        let id = self.next_block;
        self.next_block += 1;
        BlockId(id)
    }

    /// Start building a new function
    pub fn start_function(&mut self, name: String, params: Vec<Parameter>, return_type: Type) {
        let entry = self.new_block();
        self.current_function = Some(Function {
            name,
            params,
            return_type,
            blocks: HashMap::new(),
            entry,
        });
        self.current_block = Some(entry);
    }

    /// Start a new basic block
    pub fn start_block(&mut self, id: BlockId) {
        if let Some(ref mut function) = self.current_function {
            function.blocks.insert(id, BasicBlock {
                instructions: Vec::new(),
                terminator: Terminator::Return(None), // Temporary
            });
            self.current_block = Some(id);
        }
    }

    /// Add an instruction to the current block
    pub fn add_instruction(&mut self, instruction: Instruction) -> Result<()> {
        if let (Some(ref mut function), Some(block_id)) = (&mut self.current_function, self.current_block) {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                block.instructions.push(instruction);
                Ok(())
            } else {
                Err(crate::error::CompilerError::internal_error(
                    "current block not found in function",
                ))
            }
        } else {
            Err(crate::error::CompilerError::internal_error(
                "no current function or block",
            ))
        }
    }

    /// Set the terminator for the current block
    pub fn set_terminator(&mut self, terminator: Terminator) -> Result<()> {
        if let (Some(ref mut function), Some(block_id)) = (&mut self.current_function, self.current_block) {
            if let Some(block) = function.blocks.get_mut(&block_id) {
                block.terminator = terminator;
                Ok(())
            } else {
                Err(crate::error::CompilerError::internal_error(
                    "current block not found in function",
                ))
            }
        } else {
            Err(crate::error::CompilerError::internal_error(
                "no current function or block",
            ))
        }
    }

    /// Finish building the current function
    pub fn finish_function(&mut self) -> Result<Function> {
        if let Some(function) = self.current_function.take() {
            self.current_block = None;
            Ok(function)
        } else {
            Err(crate::error::CompilerError::internal_error(
                "no current function",
            ))
        }
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_builder() {
        let mut builder = IrBuilder::new();

        // Start building a function
        builder.start_function(
            "test".to_string(),
            vec![Parameter {
                name: "x".to_string(),
                ty: Type::Float32,
            }],
            Type::Float32,
        );

        // Create some values
        let val1 = builder.new_value();
        let val2 = builder.new_value();
        let result = builder.new_value();

        // Add instructions
        builder
            .add_instruction(Instruction::Binary {
                op: BinaryOp::Add,
                result,
                left: val1,
                right: val2,
            })
            .unwrap();

        // Set terminator
        builder
            .set_terminator(Terminator::Return(Some(result)))
            .unwrap();

        // Finish function
        let function = builder.finish_function().unwrap();

        // Verify function
        assert_eq!(function.name, "test");
        assert_eq!(function.params.len(), 1);
        assert_eq!(function.blocks.len(), 1);
    }

    #[test]
    fn test_tensor_ops() {
        let mut builder = IrBuilder::new();

        builder.start_function(
            "conv".to_string(),
            vec![
                Parameter {
                    name: "input".to_string(),
                    ty: Type::Tensor(Box::new(Type::Float32), vec![1, 28, 28, 1]),
                },
                Parameter {
                    name: "filter".to_string(),
                    ty: Type::Tensor(Box::new(Type::Float32), vec![3, 3, 1, 32]),
                },
            ],
            Type::Tensor(Box::new(Type::Float32), vec![1, 26, 26, 32]),
        );

        let input = builder.new_value();
        let filter = builder.new_value();
        let result = builder.new_value();

        builder
            .add_instruction(Instruction::TensorOp {
                op: TensorOp::Conv2D {
                    stride: (1, 1),
                    padding: (0, 0),
                },
                result,
                operands: vec![input, filter],
            })
            .unwrap();

        builder
            .set_terminator(Terminator::Return(Some(result)))
            .unwrap();

        let function = builder.finish_function().unwrap();
        assert_eq!(function.blocks.len(), 1);
    }
}
