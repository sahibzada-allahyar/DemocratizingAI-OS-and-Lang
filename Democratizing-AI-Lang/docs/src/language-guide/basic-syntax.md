# Basic Syntax

This guide introduces the fundamental syntax and concepts of the Democratising programming language.

## Variables and Types

### Variable Declaration

Variables are declared using `let`:

```rust
let x = 42;                    // Type inference
let y: i32 = 42;              // Explicit type
let mut z = 42;               // Mutable variable
let tensor = Tensor::new(...); // Complex type
```

### Basic Types

```rust
// Numeric types
let integer: i32 = 42;
let float: f32 = 3.14;
let boolean: bool = true;

// Compound types
let tuple: (i32, f32) = (42, 3.14);
let array: [i32; 3] = [1, 2, 3];
let vector: Vec<i32> = vec![1, 2, 3];

// String types
let string: String = String::from("Hello");
let str_slice: &str = "World";
```

### Type Inference

The compiler can often infer types:

```rust
let x = 42;        // i32 inferred
let y = 3.14;      // f32 inferred
let z = true;      // bool inferred
let v = vec![1,2]; // Vec<i32> inferred
```

## Functions

### Function Declaration

```rust
fn add(a: i32, b: i32) -> i32 {
    a + b  // Implicit return
}

fn compute() -> Result<f32> {
    let x = some_operation()?;
    Ok(x)
}
```

### Generic Functions

```rust
fn process<T: Number>(value: T) -> T {
    value * value
}

fn transform<T, U>(input: T) -> U
where
    T: Into<U>,
{
    input.into()
}
```

## Control Flow

### If Expressions

```rust
let x = if condition {
    value1
} else {
    value2
};

if let Some(value) = optional {
    // Handle value
}
```

### Match Expressions

```rust
match value {
    0 => println!("Zero"),
    1 => println!("One"),
    n if n < 0 => println!("Negative"),
    _ => println!("Something else"),
}
```

### Loops

```rust
// For loop
for item in collection {
    println!("{}", item);
}

// While loop
while condition {
    // Do something
}

// Loop with break
let result = loop {
    if condition {
        break value;
    }
};
```

## Error Handling

### Using Result

```rust
fn process() -> Result<()> {
    let tensor = Tensor::new(&[2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    let result = tensor.sum(None)?;
    println!("Sum: {}", result);
    Ok(())
}
```

### Custom Errors

```rust
#[derive(Debug, Error)]
enum MyError {
    #[error("Invalid dimension: {0}")]
    InvalidDimension(usize),

    #[error("Computation failed")]
    ComputationError,
}
```

## Modules and Imports

### Module Structure

```rust
mod math {
    pub fn add(a: i32, b: i32) -> i32 {
        a + b
    }

    pub mod advanced {
        pub fn multiply(a: i32, b: i32) -> i32 {
            a * b
        }
    }
}
```

### Imports

```rust
use democratising::prelude::*;
use crate::math::{add, advanced::multiply};
use std::{fs, io};
```

## Structs and Implementations

### Struct Definition

```rust
#[derive(Debug)]
struct Model {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Box<dyn Optimizer>,
    device: Device,
}
```

### Implementation

```rust
impl Model {
    pub fn new(device: Device) -> Self {
        Self {
            layers: Vec::new(),
            optimizer: Box::new(Adam::default()),
            device,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implementation
    }
}
```

## Traits

### Trait Definition

```rust
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&self, grad: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}
```

### Trait Implementation

```rust
impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Implementation
    }

    fn backward(&self, grad: &Tensor) -> Result<Tensor> {
        // Implementation
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}
```

## Memory Management

### Ownership

```rust
let tensor = Tensor::new(...)?;
process(tensor);        // Ownership moved
// tensor no longer available here

let tensor = Tensor::new(...)?;
process(&tensor);       // Borrowed reference
// tensor still available here
```

### References

```rust
fn process(tensor: &Tensor) {
    // Immutable reference
}

fn modify(tensor: &mut Tensor) {
    // Mutable reference
}
```

## Async/Await

### Async Functions

```rust
async fn fetch_data() -> Result<Tensor> {
    let response = client.get("url").await?;
    let data = response.bytes().await?;
    Tensor::from_bytes(data)
}
```

### Using Async

```rust
#[tokio::main]
async fn main() -> Result<()> {
    let data = fetch_data().await?;
    process_data(data).await?;
    Ok(())
}
```

## Common Patterns

### Builder Pattern

```rust
let model = Sequential::new()
    .add(Dense::new(784, 128))
    .add(Activation::ReLU)
    .add(Dense::new(128, 10))
    .build()?;
```

### Error Propagation

```rust
fn process() -> Result<()> {
    let a = operation1()?;
    let b = operation2(&a)?;
    let c = operation3(&b)?;
    Ok(())
}
```

### Resource Management

```rust
let file = File::open("data.txt")?;
// File automatically closed when it goes out of scope

let tensor = Tensor::new(...)?;
drop(tensor);  // Explicitly free resources
```

## Best Practices

1. Use type inference where possible
2. Prefer `Result` over `unwrap()`
3. Use meaningful variable names
4. Document public interfaces
5. Follow the Rust naming conventions
6. Use the builder pattern for complex objects
7. Implement common traits (`Debug`, `Clone`, etc.)
8. Use proper error handling
9. Write unit tests
10. Format code with `rustfmt`

## Next Steps

- Learn about [Memory Management](memory-management.md)
- Explore [Error Handling](error-handling.md)
- Study [Neural Networks](../ai-features/neural-networks.md)
- Try the [Examples](https://github.com/democratising/democratising/tree/main/examples)
