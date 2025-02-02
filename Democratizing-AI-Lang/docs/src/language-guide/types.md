# Types in Democratising

Democratising features a strong, static type system that combines safety with flexibility. This guide covers the type system and how to use it effectively.

## Basic Types

### Primitive Types

```rust
// Integer types
let i: i32 = 42;        // 32-bit signed integer
let u: u64 = 42;        // 64-bit unsigned integer
let b: i8 = 42;         // 8-bit signed integer

// Floating point types
let f: f32 = 3.14;      // 32-bit float
let d: f64 = 3.14;      // 64-bit float

// Boolean
let bool_val: bool = true;

// Character
let char_val: char = 'a';
```

### Compound Types

```rust
// Arrays
let arr: [i32; 3] = [1, 2, 3];
let matrix: [[f32; 3]; 2] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

// Tuples
let tuple: (i32, f32, bool) = (1, 2.0, true);
let (x, y, z) = tuple;  // Destructuring

// Vectors
let vec: Vec<i32> = vec![1, 2, 3];
let dynamic: Vec<f32> = Vec::with_capacity(100);
```

## AI-Specific Types

### Tensor Type

```rust
// Basic tensor creation
let tensor: Tensor<f32> = Tensor::new(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// Type-specific tensors
let float_tensor: Tensor<f32> = Tensor::zeros(&[3, 3])?;
let int_tensor: Tensor<i32> = Tensor::ones(&[2, 2])?;
let bool_tensor: Tensor<bool> = Tensor::from_slice(&[true, false, true])?;
```

### Device Types

```rust
// Device specification
let cpu_device: Device = Device::cpu();
let gpu_device: Device = Device::cuda(0)?;

// Device-specific tensors
let cpu_tensor = Tensor::zeros(&[3, 3], &cpu_device)?;
let gpu_tensor = Tensor::zeros(&[3, 3], &gpu_device)?;
```

## Generic Types

### Generic Functions

```rust
// Generic over tensor element type
fn process<T: Number>(tensor: &Tensor<T>) -> Result<Tensor<T>> {
    tensor.pow(2.0)
}

// Generic with multiple type parameters
fn transform<T, U>(input: &Tensor<T>) -> Result<Tensor<U>>
where
    T: Number,
    U: Number + From<T>,
{
    input.cast::<U>()
}
```

### Generic Structs

```rust
// Generic model
struct Model<T: Number> {
    weights: Tensor<T>,
    bias: Tensor<T>,
    device: Device,
}

impl<T: Number> Model<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        (&input.matmul(&self.weights)? + &self.bias)?
            .relu()
    }
}
```

## Traits

### Core Traits

```rust
// Number trait for numeric operations
pub trait Number:
    Add<Output = Self> +
    Sub<Output = Self> +
    Mul<Output = Self> +
    Div<Output = Self> +
    Copy +
    Default +
    Send +
    Sync +
    'static
{
    fn zero() -> Self;
    fn one() -> Self;
}

// Implementation for float
impl Number for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
}
```

### Neural Network Traits

```rust
// Layer trait
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&self, grad: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<Tensor>;
}

// Loss function trait
pub trait Loss {
    fn forward(&self, pred: &Tensor, target: &Tensor) -> Result<Tensor>;
    fn backward(&self) -> Result<Tensor>;
}
```

## Type Inference

### Basic Inference

```rust
// Type inferred from context
let x = 42;             // i32 inferred
let y = 3.14;          // f32 inferred
let z = true;          // bool inferred

// Complex inference
let tensor = Tensor::zeros(&[3, 3])?;  // Tensor<f32> inferred
let result = tensor.matmul(&tensor)?;  // Result<Tensor<f32>> inferred
```

### Type Annotations

```rust
// Explicit type annotations
let x: i64 = 42;
let tensor: Tensor<f64> = Tensor::zeros(&[3, 3])?;

// Function return type annotations
fn compute() -> Result<Tensor<f32>> {
    let x = Tensor::ones(&[2, 2])?;
    Ok(x)
}
```

## Type Conversion

### Numeric Conversions

```rust
// Safe conversions
let i: i32 = 42;
let f: f32 = i as f32;    // Integer to float
let u: u32 = i as u32;    // Signed to unsigned

// Tensor type casting
let float_tensor: Tensor<f32> = tensor.cast()?;
let int_tensor: Tensor<i32> = float_tensor.cast()?;
```

### Device Conversions

```rust
// Moving between devices
let cpu_tensor = Tensor::zeros(&[3, 3])?;
let gpu_tensor = cpu_tensor.to_device(&Device::cuda(0)?)?;
let back_to_cpu = gpu_tensor.to_device(&Device::cpu())?;
```

## Advanced Types

### Type Bounds

```rust
// Multiple trait bounds
fn process<T>(tensor: &Tensor<T>) -> Result<Tensor<T>>
where
    T: Number + Default + Display,
{
    // Implementation
}

// Lifetime bounds
fn view<'a, T: Number>(tensor: &'a Tensor<T>) -> Result<&'a Tensor<T>> {
    Ok(tensor)
}
```

### Associated Types

```rust
pub trait DataLoader {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;
    fn batch_size(&self) -> usize;
}

impl DataLoader for MNISTLoader {
    type Item = (Tensor<f32>, Tensor<i64>);

    fn next(&mut self) -> Option<Self::Item> {
        // Implementation
    }

    fn batch_size(&self) -> usize {
        32
    }
}
```

## Best Practices

### Type Safety

1. Use appropriate numeric types:
```rust
// Good: Explicit about precision
let weights: Tensor<f32> = Tensor::randn(&[100, 100])?;

// Bad: Mixed precision without explicit intent
let result = weights.cast::<f64>()?;
```

2. Handle type conversions explicitly:
```rust
// Good: Explicit conversion
let float_val: f32 = int_val as f32;

// Bad: Implicit conversion that might fail
let int_val: i32 = float_val;  // Might truncate
```

### Performance Considerations

1. Choose appropriate numeric types:
```rust
// For most ML: f32 is sufficient and faster
let tensor: Tensor<f32> = Tensor::zeros(&[1000, 1000])?;

// For high precision: use f64
let precise: Tensor<f64> = Tensor::zeros(&[1000, 1000])?;
```

2. Use views instead of copies:
```rust
// Good: Zero-copy view
let view = tensor.view()?;

// Bad: Unnecessary copy
let copy = tensor.copy()?;
```

## Next Steps

- Learn about [Memory Management](memory-management.md)
- Explore [Error Handling](error-handling.md)
- Study [Neural Networks](../ai-features/neural-networks.md)
