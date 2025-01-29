# Democratising Language Reference

This document provides a comprehensive reference for the Democratising programming language.

## Table of Contents

1. [Introduction](#introduction)
2. [Basic Syntax](#basic-syntax)
3. [Types](#types)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [AI Features](#ai-features)
7. [Memory Management](#memory-management)
8. [Concurrency](#concurrency)
9. [Interoperability](#interoperability)

## Introduction

Democratising is a statically-typed programming language designed specifically for AI development. It combines the safety and performance of systems programming languages with the ease of use typically associated with scripting languages.

### Design Goals

- **Accessibility**: Easy to learn and use, especially for AI/ML tasks
- **Performance**: Native code performance with GPU acceleration
- **Safety**: Strong type system and memory safety without garbage collection
- **Interoperability**: Seamless integration with Python and C/C++ ecosystems

## Basic Syntax

### Variables and Constants

```rust
let x = 42;              // Type inferred
let y: f64 = 3.14;      // Explicit type
let mut z = 0;          // Mutable variable
const PI: f64 = 3.14159; // Constant
```

### Basic Types

```rust
// Numeric types
let i: i32 = 42;    // 32-bit integer
let f: f64 = 3.14;  // 64-bit float
let b: bool = true; // Boolean

// Compound types
let arr = [1, 2, 3];          // Array
let tup = (1, "hello", true); // Tuple
```

## Types

### Built-in Types

- **Integers**: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`
- **Floating Point**: `f32`, `f64`
- **Boolean**: `bool`
- **Characters and Strings**: `char`, `String`
- **Arrays and Slices**: `[T; N]`, `&[T]`
- **Tuples**: `(T1, T2, ...)`

### AI-Specific Types

```rust
// Tensors
let x = tensor![1.0, 2.0, 3.0];
let matrix = tensor![[1.0, 2.0], [3.0, 4.0]];

// Neural Network Types
let layer = Dense::new(784, 128);
let model = Sequential::new(vec![layer]);
```

## Control Flow

### Conditionals

```rust
if condition {
    // code
} else if other_condition {
    // code
} else {
    // code
}
```

### Loops

```rust
// For loop
for item in collection {
    // code
}

// While loop
while condition {
    // code
}

// Loop with break/continue
loop {
    if condition {
        break;
    }
    // code
}
```

## Functions

### Function Declaration

```rust
fn add(x: i32, y: i32) -> i32 {
    x + y
}

// Function with generic type
fn identity<T>(x: T) -> T {
    x
}

// Async function
async fn fetch_data() -> Result<Data> {
    // code
}
```

### Closures

```rust
let add = |x, y| x + y;
let multiply = |x: i32, y: i32| -> i32 { x * y };
```

## AI Features

### Tensor Operations

```rust
// Create tensors
let x = Tensor::new(vec![1.0, 2.0, 3.0]);
let y = Tensor::zeros(vec![3, 3]);

// Basic operations
let z = x + y;
let w = x.matmul(&y);

// Neural network operations
let output = model.forward(&input);
let loss = mse_loss(&output, &target);
loss.backward();
```

### Automatic Differentiation

```rust
// Variables that require gradients
let w = Variable::new(Tensor::ones(vec![784, 128]));
w.set_requires_grad(true);

// Forward and backward passes
let output = model.forward(&input)?;
let loss = cross_entropy(&output, &target)?;
loss.backward()?;

// Access gradients
let gradients = w.gradient();
```

## Memory Management

### Ownership and Borrowing

```rust
fn take_ownership(x: String) {
    // x is owned here
}

fn borrow_value(x: &String) {
    // x is borrowed here
}

fn borrow_mut(x: &mut String) {
    // x is mutably borrowed here
}
```

## Concurrency

### Parallel Execution

```rust
// Parallel iteration
let result = data.par_iter()
    .map(|x| process(x))
    .collect();

// Async/await
async fn process_data() {
    let result = fetch_data().await;
    // process result
}
```

## Interoperability

### Python Integration

```rust
// Import Python module
let numpy = python::import("numpy")?;

// Convert between Tensor and NumPy array
let array = tensor.to_numpy();
let tensor = Tensor::from_numpy(array);
```

### C/C++ Integration

```rust
// FFI declaration
extern "C" {
    fn some_c_function(x: i32) -> i32;
}

// Call C function
unsafe {
    let result = some_c_function(42);
}
```

## Best Practices

1. Use type inference where it improves readability
2. Prefer immutable variables unless mutation is necessary
3. Use error handling with Result type
4. Enable GPU acceleration for tensor operations when available
5. Profile and optimize performance-critical code
6. Follow the memory safety guidelines

## Common Patterns

### Error Handling

```rust
fn fallible_operation() -> Result<Success, Error> {
    // Implementation
}

// Using the ? operator
fn process() -> Result<()> {
    let result = fallible_operation()?;
    Ok(())
}
```

### Resource Management

```rust
// Resources are automatically cleaned up when they go out of scope
fn process_data() {
    let data = load_large_dataset();
    // Process data
    // Data is automatically freed here
}
```

## Standard Library

See the [Standard Library API Documentation](api/stdlib.md) for detailed information about the available modules and functions.

## Compiler Options

The Democratising compiler (`democ`) supports various options for optimization and code generation:

```bash
democ source.dem           # Compile with default options
democ -O3 source.dem      # Compile with maximum optimization
democ --emit-llvm source.dem  # Emit LLVM IR
democ --target gpu source.dem # Enable GPU code generation
```

## Further Reading

- [Compiler Documentation](compiler.md)
- [Standard Library API](api/stdlib.md)
- [Examples](../examples/README.md)
- [Contributing Guide](../CONTRIBUTING.md)
