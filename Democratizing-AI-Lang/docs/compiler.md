# Democratising Compiler Documentation

This document describes the architecture and implementation of the Democratising compiler.

## Overview

The Democratising compiler is implemented in Rust and uses LLVM as its backend. It follows a traditional compiler pipeline with several stages of processing, each designed to be modular and maintainable.

## Architecture

```
Source Code
    ↓
Lexical Analysis (Lexer)
    ↓
Syntax Analysis (Parser)
    ↓
Semantic Analysis
    ↓
IR Generation
    ↓
Optimization
    ↓
Code Generation
```

### Components

1. **Lexer** (`src/lexer.rs`)
   - Converts source text into tokens
   - Handles Unicode correctly
   - Reports detailed error locations
   - Preserves comments for documentation

2. **Parser** (`src/parser.rs`)
   - Builds Abstract Syntax Tree (AST)
   - Implements error recovery
   - Handles operator precedence
   - Supports syntax extensions

3. **Semantic Analyzer** (`src/semantic.rs`)
   - Type checking and inference
   - Name resolution
   - Lifetime analysis
   - Borrow checking

4. **IR Generator** (`src/ir.rs`)
   - Converts AST to LLVM IR
   - Handles control flow
   - Implements memory model
   - Supports debugging information

5. **Optimizer** (`src/optimizer.rs`)
   - LLVM optimization passes
   - AI-specific optimizations
   - Tensor operation fusion
   - Memory layout optimization

6. **Code Generator** (`src/codegen.rs`)
   - Generates native code
   - Handles different targets
   - Implements GPU support
   - Links with runtime library

## Compilation Pipeline

### 1. Lexical Analysis

The lexer breaks source code into tokens:

```rust
// Source code
let x: i32 = 42;

// Tokens
Token::Let
Token::Identifier("x")
Token::Colon
Token::Type("i32")
Token::Equals
Token::Number("42")
Token::Semicolon
```

### 2. Syntax Analysis

The parser builds an AST:

```rust
// AST structure
Let {
    name: "x",
    type_annotation: Some(Type::I32),
    initializer: Expression::Literal(42),
}
```

### 3. Semantic Analysis

Type checking and validation:

```rust
// Type checking
fn check_assignment(lhs: Type, rhs: Type) -> Result<()> {
    if !lhs.can_assign(rhs) {
        Err(TypeError::IncompatibleTypes { expected: lhs, got: rhs })
    } else {
        Ok(())
    }
}
```

### 4. IR Generation

Converting to LLVM IR:

```llvm
; LLVM IR
%x = alloca i32
store i32 42, i32* %x
```

### 5. Optimization

The compiler performs several optimization passes:

1. **Standard Optimizations**
   - Constant folding
   - Dead code elimination
   - Loop optimization
   - Inlining

2. **AI-Specific Optimizations**
   - Tensor operation fusion
   - GPU kernel generation
   - Memory access patterns
   - Automatic differentiation optimization

### 6. Code Generation

Final machine code generation:

```bash
# Generate native code
democ source.dem -o program

# Generate CUDA code
democ source.dem --target gpu
```

## Error Handling

The compiler provides detailed error messages:

```rust
error[E0308]: mismatched types
  --> source.dem:10:14
   |
10 |     let x: i32 = "hello";
   |             --- ^^^^^^^^ expected i32, found string
   |             |
   |             expected due to this type annotation
```

## Debugging Support

The compiler generates debugging information:

```rust
// Source with debug info
#[debug]
fn calculate(x: i32) -> i32 {
    x * 2
}
```

## GPU Support

GPU code generation is handled through multiple backends:

1. **CUDA**
   - Direct CUDA code generation
   - PTX assembly
   - Runtime library integration

2. **OpenCL**
   - Kernel generation
   - Device management
   - Memory transfers

## Compiler Extensions

The compiler supports various extensions:

### Custom Optimizations

```rust
#[optimize(tensor_fusion)]
fn matrix_multiply(a: Tensor, b: Tensor) -> Tensor {
    // ...
}
```

### Target-Specific Code

```rust
#[target(gpu)]
fn parallel_compute() {
    // GPU-specific implementation
}

#[target(cpu)]
fn parallel_compute() {
    // CPU-specific implementation
}
```

## Building the Compiler

### Prerequisites

- Rust 1.75.0 or later
- LLVM 17.0
- (Optional) CUDA Toolkit for GPU support

### Build Steps

```bash
# Clone the repository
git clone https://github.com/sahibzada-allahyar/democratising.git

# Build the compiler
cd democratising
cargo build --release

# Run tests
cargo test

# Install
cargo install --path .
```

## Usage

### Basic Compilation

```bash
# Compile a file
democ source.dem

# Compile with optimizations
democ -O3 source.dem

# Generate debug info
democ -g source.dem
```

### Advanced Options

```bash
# Emit LLVM IR
democ --emit=llvm source.dem

# Target GPU
democ --target=gpu source.dem

# Enable specific optimizations
democ --opt=tensor-fusion source.dem
```

## Internals

### AST Structure

The compiler uses a rich AST representation:

```rust
pub enum Expression {
    Literal(Literal),
    Variable(Identifier),
    BinaryOp {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    // ...
}
```

### Type System

The type system is implemented as:

```rust
pub enum Type {
    Primitive(PrimitiveType),
    Array(Box<Type>, usize),
    Tensor(Box<Type>, Vec<usize>),
    Function(Vec<Type>, Box<Type>),
    // ...
}
```

### Memory Model

The compiler implements a memory model similar to Rust's:

- Ownership tracking
- Borrowing rules
- Lifetime analysis
- RAII principles

## Performance

The compiler is designed for performance:

1. **Parallel Compilation**
   - Multiple files compiled in parallel
   - Parallel optimization passes
   - Concurrent code generation

2. **Incremental Compilation**
   - Caching of intermediate results
   - Minimal recompilation
   - Dependency tracking

3. **Memory Efficiency**
   - Efficient AST representation
   - Memory pools for temporary allocations
   - Streaming processing where possible

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details on:
- Setting up development environment
- Running tests
- Code style guidelines
- Pull request process

## Future Work

1. **Compiler Features**
   - More aggressive optimizations
   - Better error messages
   - Incremental compilation
   - Cross-compilation support

2. **Language Features**
   - Advanced type system features
   - More AI primitives
   - Better metaprogramming
   - Enhanced GPU support

3. **Tooling**
   - Language server protocol
   - Debug adapter protocol
   - Profile-guided optimization
   - Auto-vectorization

## References

- [Language Reference](language-reference.md)
- [Standard Library API](api/stdlib.md)
- [LLVM Documentation](https://llvm.org/docs/)
- [Rust Documentation](https://doc.rust-lang.org/book/)
